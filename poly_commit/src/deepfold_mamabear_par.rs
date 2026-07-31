///! Parallel DeepFold: Multi-threaded commit and open for MamaBear field.
///!
///! Creates parallel versions of the hot paths in deepfold_mamabear.rs:
///! - `new_par`: Parallel sub-FFT + parallel leaf hashing
///! - `open_par`: Parallel combine_subs/combine_polys
///!
///! All functions produce bit-identical results to their serial counterparts.

use std::sync::Arc;

use arithmetic::{
    fft_mamabear::MamaBearFFT,
    field::{
        mamabear::{
            LazyReduction, MamaBearScalar, MamaBearScalarExt3,
            PackedExtensionField, PackedMamaBearAVX512,
            PackedMamaBearAVX512Ext3, P,
        },
        Field,
    },
};
use rayon::prelude::*;

/// Address wrapper to send raw mutable pointers across threads as usize.
/// Safety: the caller must ensure no two threads access overlapping regions.
#[derive(Clone, Copy)]
struct SendAddr(usize);
unsafe impl Send for SendAddr {}
unsafe impl Sync for SendAddr {}

impl SendAddr {
    fn from_ptr<T>(p: *mut T) -> Self {
        SendAddr(p as usize)
    }
    unsafe fn as_ptr<T>(self) -> *mut T {
        self.0 as *mut T
    }
}

use util::{
    blake3_batch,
    merkle_tree_mamabear::{MerkleTreeProverMB, HASH_SIZE},
};

use crate::deepfold_mamabear::{
    evaluate_next_domain_first_round_packed_ext3,
    evaluate_next_domain_packed_ext3,
    fold_multilinear_packed_ext3_out_of_place,
    DeepFoldExtField, DeepFoldMamaBearParam, DeepFoldMamaBearProver, DeepFoldMamaBearVerifier,
    FriFoldResult, InterpolateValueMB, NewTimings, QueryResultMB, VerifyTimings,
};
use arithmetic::field::as_bytes_vec;
use std::collections::HashMap;
use std::time::Instant;
use util::fiat_shamir::{Proof, Transcript};
use util::merkle_tree_mamabear::{MerkleTreeVerifierMB, HASH_SIZE as MT_HASH_SIZE};

type SBF = MamaBearScalar;
type PBF = PackedMamaBearAVX512;
type PEF3 = PackedMamaBearAVX512Ext3;
type SEF3 = MamaBearScalarExt3;

// =============================================================================
// Parallelization thresholds for pcs_open substages
// =============================================================================

/// Minimum total element count before split_fold parallelizes.
/// Below this, rayon overhead dominates the actual work. The split_fold
/// kernel is memory-bandwidth bound and a single core already saturates
/// most of the DRAM bandwidth on the i7-11700K, so parallel gives at most
/// ~1.1-1.2x and only at large sizes. The fold_sub_polys_packed serial-vs-par
/// sweep (i7-11700K, 8c) shows par is a wash or LOSS (0.74x-1.09x, shape
/// dependent on num_subs) below 524288 total elems, and only reliably wins
/// (>= 1.1x across shapes) at/above 524288. We set 1<<19 so par is never
/// slower than serial. (threshold retuning from 1<<15: at 32768-262144 the prior
/// threshold ran par at 0.86x-1.07x, violating never-slower for low num_subs.)
const PAR_SPLIT_FOLD_MIN_ELEMS: usize = 1 << 19;

/// When new_count (split_fold outer loop) is at least this, parallelize
/// the outer loop; otherwise parallelize the inner `j` loop of one pair
/// at a time.
const PAR_SPLIT_FOLD_OUTER_MIN: usize = 4;

/// Minimum `new_len` (output block count) before fold_multilinear_packed
/// parallelizes. This kernel (also the engine of `mle_eval`) is memory
/// bandwidth bound. The fold_multilinear_packed serial-vs-par sweep (i7-11700K,
/// 8c) shows par is SLOWER than serial across the whole realistic range —
/// 0.91x at 128, 0.18x-0.42x at 2048-16384, 0.84x at 65536, 0.98x at 131072 —
/// and only crosses over at new_len >= 262144 (1.06x), reaching 1.16x at
/// 524288. The largest fold in a nv=20 open is new_len=65536, so for all
/// realistic nv this kernel must run serial. We set 1<<18 so par only engages
/// for the first 1-2 folds at very high nv (>= 22) where it is marginally
/// faster. (threshold retuning from 1<<8: that value parallelized every fold and made
/// `mle_eval` and `multilin_fold` ~2x SLOWER than serial at nv=20. The old
/// comment's "1.4x-14x even at small new_len" was a Zen5/stale measurement.)
const PAR_FOLD_MLIN_MIN_NEW_LEN: usize = 1 << 18;

/// Minimum leaf count before FriFoldResult merkle build parallelizes.
/// The from_packed_values_par serial-vs-par sweep (i7-11700K, 8c) shows par
/// is CATASTROPHICALLY slower below 65536 leaves (0.12x at 4096, 0.45x at
/// 16384), ties at 65536 (1.04x), and wins at 262144 (1.79x) / 1M (2.51x).
/// We set 1<<17 = 131072 (interpolated ~1.3-1.4x) so par is never slower than
/// serial. (threshold retuning from 1<<12: at 4096 leaves par was 0.12x = 8x SLOWER,
/// and the per-round FRI merkle build hits 4096-32768-leaf rounds inside every
/// open, dragging the `fri_merkle` substage to 0.87x overall at nv=20.)
const PAR_FRI_MKL_MIN_LEAVES: usize = 1 << 17;

/// Minimum packed-pair count (output block count) before the within-round
/// FRI fold (`evaluate_next_domain*`) parallelizes. Each output block is a
/// pure SIMD chunk depending only on two input blocks + a per-chunk twiddle,
/// so the loop is embarrassingly data-parallel. `packed_pairs = fft.size()/16`
/// for the round's domain, so the first round (largest domain) is by far the
/// hottest and shrinks 2x per round.
///
/// The kernel returns a freshly-allocated output `Vec` per call (uninit +
/// set_len), so each parallel invocation pays an mmap + distributed
/// first-touch on top of the SIMD work. The evaluate_next_domain_packed
/// serial-vs-par sweep (i7-11700K, 8c) shows the parallel path is a within-noise TIE at
/// packed_pairs = 16384 (1.02x regular round / 0.99x first round) and only a
/// clean win at 32768 (1.53x / 2.35x), growing to 2.5x-4x at 65536-131072.
/// We set the threshold at 32768 so par is strictly never slower than serial.
/// At nv=20 (Ext3, code_rate 3, split 3) this parallelizes the two largest FRI
/// rounds (packed_pairs 65536 / 32768); the 16384-pair round (a wash) runs
/// serial. (threshold retuning from 1<<14: slice #44 set 16384 from a Zen5 measurement
/// where it was 1.63x; on the i7-11700K threshold machine 16384 is a tie.)
const PAR_FRI_FOLD_MIN_PAIRS: usize = 1 << 15;

/// Minimum `chunks_packed` (= output length / 8) before the combine kernel
/// (`combine_opt_ext3_inner_par`) parallelizes. The combine is a Horner
/// random-linear combination of `n_polys` base polys; the open path uses
/// `n_polys = 7` (HyperPlonk K). The `microbench_combine_ext3` sweep
/// (i7-11700K, 8c) at n_polys=7 shows par is slower/tied below 32768 chunks
/// (only crosses at 32768=1.55x; at 16384 it is 0.96x, a wash) and a clean
/// win at/above 32768 (1.55x).
/// We set the threshold at 32768 so par is never slower than serial at
/// the binding n_polys=7 case. With more polys the crossover is
/// far lower (n_polys=21 wins from ~256 chunks), but those only occur in the
/// open at large `chunks_packed` anyway, so a single chunk-count guard is safe.
const PAR_COMBINE_MIN_CHUNKS: usize = 1 << 15;

/// Minimum per-poly length (= 2^nv) before the parallel commit (`new_par`)
/// engages; below it the serial `DeepFoldMamaBearProver::new` runs (the
/// resulting prover/Merkle root is bit-identical). `new_par` composes the
/// parallel sub-FFT, parallel leaf hashing, and parallel Merkle build, each
/// with rayon dispatch overhead. The `microbench_new_par_ext3` sweep
/// (i7-11700K, 8c, 3 polys) shows the composite is SLOWER than serial for
/// small nv (0.06x at nv=10, 0.23x at nv=12, 0.79x at nv=14) and only wins
/// from nv=16 (1.86x), reaching 3.4x at nv>=18. We gate at 1<<16 so the
/// commit entry point is never slower than serial. In production the
/// orchestrators only call `new_par` at nv >= `CUSTOM_FULL_PAR_MIN_NV` (>= 16),
/// so this guard only protects direct small-nv callers.
const PAR_COMMIT_MIN_POLY_LEN: usize = 1 << 16;

// =============================================================================
// Parallel leaf hashing
// =============================================================================

/// Batch size for leaf hashing — same as serial version's target.
const ROUND0_PAIR_BYTES: usize = 2 * MamaBearScalar::SIZE;
const ROUND0_LEAF_HASH_TARGET_BYTES: usize = 32 * 1024;

#[inline(always)]
fn round0_leaf_hash_batch_size(leaf_bytes: usize, pairs_per_block: usize) -> usize {
    let target = (ROUND0_LEAF_HASH_TARGET_BYTES / leaf_bytes).max(1);
    let aligned = (target / pairs_per_block) * pairs_per_block;
    aligned.max(pairs_per_block)
}

#[inline(always)]
fn write_round0_pair_bytes(dst: &mut [u8], x: MamaBearScalar, nx: MamaBearScalar) {
    let x_bytes = x.0.to_le_bytes();
    let nx_bytes = nx.0.to_le_bytes();
    dst[..MamaBearScalar::SIZE].copy_from_slice(&x_bytes[..MamaBearScalar::SIZE]);
    dst[MamaBearScalar::SIZE..ROUND0_PAIR_BYTES].copy_from_slice(&nx_bytes[..MamaBearScalar::SIZE]);
}

/// Parallel leaf hashing — same output as `round0_leaf_hashes_from_pair_major_values`.
fn round0_leaf_hashes_par(
    values_mont: &[SBF],
    leaf_size: usize,
) -> Vec<[u8; HASH_SIZE]> {
    assert_eq!(leaf_size % 2, 0);
    let segment_count = leaf_size / 2;
    assert!(segment_count > 0);
    assert_eq!(values_mont.len() % segment_count, 0);

    let eval_len = values_mont.len() / segment_count;
    assert_eq!(eval_len % 2, 0);

    let leaf_count = eval_len / 2;
    let pairs_per_block = MamaBearFFT::pair_slots_per_block_for_pair_count(leaf_count);
    let leaf_bytes = leaf_size * SBF::SIZE;
    let batch_leaf_count = round0_leaf_hash_batch_size(leaf_bytes, pairs_per_block);

    // Collect batch ranges
    let batch_ranges: Vec<(usize, usize)> = (0..leaf_count)
        .step_by(batch_leaf_count)
        .map(|batch_start| {
            let active_leaves = (leaf_count - batch_start).min(batch_leaf_count);
            (batch_start, active_leaves)
        })
        .collect();

    let mut leaf_hashes = vec![[0u8; HASH_SIZE]; leaf_count];

    // If only one batch, run serially to avoid thread overhead
    if batch_ranges.len() <= 1 {
        for &(batch_start, active_leaves) in &batch_ranges {
            let mut batch_leaf_bytes = vec![0u8; active_leaves * leaf_bytes];
            hash_one_batch(
                values_mont,
                segment_count,
                eval_len,
                pairs_per_block,
                leaf_bytes,
                batch_start,
                active_leaves,
                &mut batch_leaf_bytes,
                &mut leaf_hashes[batch_start..batch_start + active_leaves],
            );
        }
        return leaf_hashes;
    }

    // Safety: each batch writes to a disjoint range [batch_start..batch_start+active_leaves]
    // and no two batches overlap.
    let leaf_hashes_addr = SendAddr::from_ptr(leaf_hashes.as_mut_ptr());
    let leaf_hashes_len = leaf_hashes.len();

    batch_ranges.par_iter().for_each(|&(batch_start, active_leaves)| {
        let mut batch_leaf_bytes = vec![0u8; active_leaves * leaf_bytes];
        let dest = unsafe {
            std::slice::from_raw_parts_mut(
                leaf_hashes_addr.as_ptr::<[u8; HASH_SIZE]>().add(batch_start),
                active_leaves.min(leaf_hashes_len - batch_start),
            )
        };
        hash_one_batch(
            values_mont,
            segment_count,
            eval_len,
            pairs_per_block,
            leaf_bytes,
            batch_start,
            active_leaves,
            &mut batch_leaf_bytes,
            dest,
        );
    });

    leaf_hashes
}

#[inline]
fn hash_one_batch(
    values_mont: &[SBF],
    segment_count: usize,
    eval_len: usize,
    pairs_per_block: usize,
    leaf_bytes: usize,
    batch_start: usize,
    active_leaves: usize,
    batch_leaf_bytes: &mut [u8],
    leaf_hashes_out: &mut [[u8; HASH_SIZE]],
) {
    for segment in 0..segment_count {
        let segment_base = segment * eval_len;
        let segment_offset = segment * ROUND0_PAIR_BYTES;

        for block_start in (0..active_leaves).step_by(pairs_per_block) {
            let src_leaf = batch_start + block_start;
            let src_base = segment_base + src_leaf * 2;
            let src = &values_mont[src_base..src_base + pairs_per_block * 2];

            for lane in 0..pairs_per_block {
                let dst_base = (block_start + lane) * leaf_bytes + segment_offset;
                write_round0_pair_bytes(
                    &mut batch_leaf_bytes[dst_base..dst_base + ROUND0_PAIR_BYTES],
                    src[lane],
                    src[lane + pairs_per_block],
                );
            }
        }
    }

    let active_bytes = active_leaves * leaf_bytes;
    blake3_batch::hash_leaves_batch_flat(
        &batch_leaf_bytes[..active_bytes],
        active_leaves,
        leaf_bytes,
        leaf_hashes_out,
    );
}

/// Parallel deep-copy of a `&[SBF]` slice into a fresh owning `Vec<SBF>`.
///
/// The destination is allocated uninitialized via `Vec::with_capacity + set_len`
/// and every slot is filled by parallel `copy_from_slice` chunks. This is the
/// transmute-free variant of the calloc fast path. `MaybeUninit` lets the
/// section: no `calloc` is needed because we immediately overwrite every byte.
///
/// The point of going parallel is to distribute the destination's first-touch
/// page faults across rayon workers. A serial `slice::to_vec()` at
/// nv = 23 faults ~192 MB / 4 KB = 49k pages single-threaded (~48 ms); this
/// version spreads those faults over 8 workers and hits DRAM bandwidth.
fn par_clone_slice(src: &[SBF]) -> Vec<SBF> {
    let len = src.len();
    let mut dst: Vec<SBF> = Vec::with_capacity(len);
    unsafe { dst.set_len(len); }

    // Grain ~16 K scalars (128 KB) -- fits in L2, amortizes scheduling.
    const GRAIN: usize = 16 * 1024;
    dst.par_chunks_mut(GRAIN)
        .zip(src.par_chunks(GRAIN))
        .for_each(|(d, s)| d.copy_from_slice(s));

    dst
}

/// Parallel deep-copy of `&[&[SBF]]` into `Vec<Vec<SBF>>`.
///
/// Three polys at nv = 23 is 192 MB total. We run the per-poly copies
/// concurrently inside a `rayon::scope`; each call to `par_clone_slice`
/// internally uses `par_chunks_mut` over its own destination so the work
/// still spreads across every worker even though there are only 3 polys.
fn par_clone_polys(poly: &[&[SBF]]) -> Vec<Vec<SBF>> {
    // Pre-allocate the outer Vec so each worker can write into a distinct slot.
    let mut out: Vec<Vec<SBF>> = Vec::with_capacity(poly.len());
    for _ in 0..poly.len() {
        out.push(Vec::new());
    }
    let out_addr = SendAddr::from_ptr(out.as_mut_ptr());

    rayon::scope(|s| {
        for (i, p) in poly.iter().enumerate() {
            let p = *p;
            s.spawn(move |_| {
                let cloned = par_clone_slice(p);
                // Safety: each task writes to a unique `out[i]` slot, and the
                // outer Vec is not read by anyone else until the scope ends.
                unsafe {
                    let slot = out_addr.as_ptr::<Vec<SBF>>().add(i);
                    std::ptr::write(slot, cloned);
                }
            });
        }
    });

    out
}

// =============================================================================
// Parallel new (commit)
// =============================================================================

/// Parallel commit: 96 independent sub-FFTs + parallel leaf hashing.
/// Returns a `DeepFoldMamaBearProver` with bit-identical output to serial `new`.
pub fn new_par<F: DeepFoldExtField>(
    pp: &DeepFoldMamaBearParam,
    poly: &[&[SBF]],
) -> DeepFoldMamaBearProver<F> {
    // Small-input guard: below the empirical commit crossover the composite
    // rayon overhead (sub-FFT + leaf-hash + Merkle) makes par slower than the
    // serial `new` (which is bit-identical). See `PAR_COMMIT_MIN_POLY_LEN`.
    if !poly.is_empty() && poly[0].len() < PAR_COMMIT_MIN_POLY_LEN {
        return DeepFoldMamaBearProver::<F>::new(pp, poly);
    }

    let fft = &pp.fft_groups[0];
    let sub_count = 1usize << pp.split_level;
    let eval_len = fft.size();
    let leaf_size = 2 * sub_count * poly.len();
    let total_elems = sub_count * eval_len * poly.len();

    // Pre-allocate output buffer as uninitialized: the FFT writes cover every
    // slot, so we skip zero-init to push page-fault first-touch into the
    // parallel FFT loop instead of paying it serially here.
    let mut sub_evals_mont: Vec<MamaBearScalar> = Vec::with_capacity(total_elems);
    unsafe { sub_evals_mont.set_len(total_elems); }

    // Build task list: (poly_idx, sub_k, write_offset, coeffs_len)
    struct FftTask<'a> {
        coeffs: &'a [SBF],
        sub_k: usize,
        sub_count: usize,
        write_offset: usize,
        eval_len: usize,
    }

    let mut tasks: Vec<FftTask> = Vec::with_capacity(poly.len() * sub_count);
    let mut write_offset = 0usize;
    for coeffs in poly.iter() {
        for k in 0..sub_count {
            tasks.push(FftTask {
                coeffs,
                sub_k: k,
                sub_count,
                write_offset,
                eval_len,
            });
            write_offset += eval_len;
        }
    }

    // Safety: each task writes to sub_evals_mont[write_offset..write_offset+eval_len]
    // and no two tasks share the same write_offset range (offsets differ by eval_len).
    let sub_evals_addr = SendAddr::from_ptr(sub_evals_mont.as_mut_ptr());

    // Sub-FFT input is at most `coeffs.len() / sub_count` elements (the strided
    // gather only writes that many; the FFT zero-pads the rest internally).
    // Reuse one scratch per worker thread via `for_each_init` to eliminate
    // 96 calloc calls per commit (1.5 GB of wasted allocation at nv=23).
    let scratch_len = poly[0].len() / sub_count;
    tasks.par_iter().for_each_init(
        || vec![MamaBearScalar(0); scratch_len],
        |sub_coeffs, task| {
            let mut len = 0usize;
            let mut idx = task.sub_k;
            while idx < task.coeffs.len() {
                sub_coeffs[len] = task.coeffs[idx];
                len += 1;
                idx += task.sub_count;
            }

            let dest = unsafe {
                std::slice::from_raw_parts_mut(
                    sub_evals_addr.as_ptr::<MamaBearScalar>().add(task.write_offset),
                    task.eval_len,
                )
            };
            fft.fft_into(&sub_coeffs[..len], dest);
        },
    );

    let sub_evals_mont: Arc<Vec<SBF>> = Arc::new(sub_evals_mont);

    // Parallel leaf hashing
    let leaf_hashes = round0_leaf_hashes_par(&sub_evals_mont, leaf_size);
    let merkle_tree = MerkleTreeProverMB::from_leaf_hashes_par(&leaf_hashes);

    let interpolation = InterpolateValueMB::from_fft_pair_major_parts_with_tree(
        sub_evals_mont.clone(),
        leaf_size,
        merkle_tree,
    );

    DeepFoldMamaBearProver {
        interpolation,
        sub_evals_mont,
        poly: par_clone_polys(poly),
        _phantom: std::marker::PhantomData,
    }
}

/// Profiled parallel commit with per-substage timing.
pub fn new_par_profiled<F: DeepFoldExtField>(
    pp: &DeepFoldMamaBearParam,
    poly: &[&[SBF]],
    timings: &mut NewTimings,
) -> DeepFoldMamaBearProver<F> {
    use std::time::Instant;

    let total_start = Instant::now();
    let fft = &pp.fft_groups[0];
    let sub_count = 1usize << pp.split_level;
    let eval_len = fft.size();
    let leaf_size = 2 * sub_count * poly.len();
    let total_elems = sub_count * eval_len * poly.len();

    let t0 = Instant::now();
    let mut sub_evals_mont: Vec<MamaBearScalar> = Vec::with_capacity(total_elems);
    unsafe { sub_evals_mont.set_len(total_elems); }
    timings.alloc_us += t0.elapsed().as_micros();

    // Build task list
    struct FftTask<'a> {
        coeffs: &'a [SBF],
        sub_k: usize,
        sub_count: usize,
        write_offset: usize,
        eval_len: usize,
    }

    let mut tasks: Vec<FftTask> = Vec::with_capacity(poly.len() * sub_count);
    let mut write_offset = 0usize;
    for coeffs in poly.iter() {
        for k in 0..sub_count {
            tasks.push(FftTask {
                coeffs,
                sub_k: k,
                sub_count,
                write_offset,
                eval_len,
            });
            write_offset += eval_len;
        }
    }

    // Parallel FFT — reuse scratch via for_each_init (one buffer per worker thread)
    let t0 = Instant::now();
    let sub_evals_addr = SendAddr::from_ptr(sub_evals_mont.as_mut_ptr());
    let scratch_len = poly[0].len() / sub_count;
    tasks.par_iter().for_each_init(
        || vec![MamaBearScalar(0); scratch_len],
        |sub_coeffs, task| {
            let mut len = 0usize;
            let mut idx = task.sub_k;
            while idx < task.coeffs.len() {
                sub_coeffs[len] = task.coeffs[idx];
                len += 1;
                idx += task.sub_count;
            }
            let dest = unsafe {
                std::slice::from_raw_parts_mut(
                    sub_evals_addr.as_ptr::<MamaBearScalar>().add(task.write_offset),
                    task.eval_len,
                )
            };
            fft.fft_into(&sub_coeffs[..len], dest);
        },
    );
    // split + fft combined into fft timing
    timings.fft_us += t0.elapsed().as_micros();

    let t0 = Instant::now();
    let sub_evals_mont: Arc<Vec<SBF>> = Arc::new(sub_evals_mont);
    timings.arc_convert_us += t0.elapsed().as_micros();

    let t0 = Instant::now();
    let leaf_hashes = round0_leaf_hashes_par(&sub_evals_mont, leaf_size);
    timings.leaf_hash_us += t0.elapsed().as_micros();

    let t0 = Instant::now();
    let merkle_tree = MerkleTreeProverMB::from_leaf_hashes_par(&leaf_hashes);
    timings.merkle_tree_us += t0.elapsed().as_micros();

    let t0 = Instant::now();
    let interpolation = InterpolateValueMB::from_fft_pair_major_parts_with_tree(
        sub_evals_mont.clone(),
        leaf_size,
        merkle_tree,
    );
    let prover = DeepFoldMamaBearProver {
        interpolation,
        sub_evals_mont,
        poly: par_clone_polys(poly),
        _phantom: std::marker::PhantomData,
    };
    timings.wrap_us += t0.elapsed().as_micros();

    timings.total_us += total_start.elapsed().as_micros();
    prover
}

// =============================================================================
// Parallel combine (Horner combination)
// =============================================================================

/// Parallel version of `combine_opt_ext3_inner`.
pub fn combine_opt_ext3_inner_par<const BASE_IS_MONT: bool>(
    base_polys: &[&[SBF]],
    r_mont: SEF3,
) -> Vec<PEF3> {
    let n_polys = base_polys.len();
    assert!(n_polys >= 1);
    let len = base_polys[0].len();
    debug_assert!(base_polys.iter().all(|p| p.len() == len));
    debug_assert_eq!(len % 8, 0);

    let chunks_packed = len / 8;

    // Small-input guard: below the empirical crossover the rayon dispatch +
    // uninit-alloc overhead makes par slower than serial. Run serial instead
    // (byte-identical). See `PAR_COMBINE_MIN_CHUNKS`.
    if chunks_packed < PAR_COMBINE_MIN_CHUNKS {
        return crate::deepfold_mamabear::combine_opt_ext3_inner::<BASE_IS_MONT>(base_polys, r_mont);
    }

    // Stage 1: precompute powers (serial — tiny)
    let mut ascending: Vec<SEF3> = Vec::with_capacity(n_polys);
    ascending.push(SEF3::from(MamaBearScalar(1)).to_montgomery());
    for i in 1..n_polys {
        ascending.push(ascending[i - 1] * r_mont);
    }
    let mut powers_scalar: Vec<SEF3> = ascending.into_iter().rev().collect();
    if !BASE_IS_MONT {
        for p in powers_scalar.iter_mut() {
            *p = p.to_montgomery();
        }
    }
    let powers_packed: Vec<PEF3> = powers_scalar
        .iter()
        .map(|p| PEF3::new(PBF::from(p.c0.0), PBF::from(p.c1.0), PBF::from(p.c2.0)))
        .collect();

    let zero_mont_pbf = PBF::broadcast(P);

    // Stage 2: parallel chunk processing. Output buffer is fully overwritten
    // by the par_chunks_mut loop below, so allocate uninit to push first-touch
    // page faults into the parallel writers (avoiding the IsZero slow path
    // that would single-thread the init for this 24 B/elem newtype).
    let mut out_packed: Vec<PEF3> = Vec::with_capacity(chunks_packed);
    unsafe { out_packed.set_len(chunks_packed); }

    let grain = 1024usize;
    out_packed.par_chunks_mut(grain).enumerate().for_each(|(chunk_base_idx, out_slice)| {
        let chunk_base = chunk_base_idx * grain;
        for (local_idx, out) in out_slice.iter_mut().enumerate() {
            let chunk = chunk_base + local_idx;
            let off = chunk * 8;

            let mut acc = PEF3::new(zero_mont_pbf, zero_mont_pbf, zero_mont_pbf);

            // 4-way ILP unroll with tree-reduced adds (mirrors the serial
            // combine_opt_ext3_inner).
            let mut j = 0;
            while j + 4 <= n_polys {
                let p1 = PBF::load_scalar_slice(&base_polys[j][off..]);
                let p2 = PBF::load_scalar_slice(&base_polys[j + 1][off..]);
                let p3 = PBF::load_scalar_slice(&base_polys[j + 2][off..]);
                let p4 = PBF::load_scalar_slice(&base_polys[j + 3][off..]);

                let prod1 = powers_packed[j].mul_base_elem(p1);
                let prod2 = powers_packed[j + 1].mul_base_elem(p2);
                let prod3 = powers_packed[j + 2].mul_base_elem(p3);
                let prod4 = powers_packed[j + 3].mul_base_elem(p4);

                let s12 = prod1.lazy_add(prod2);
                let s34 = prod3.lazy_add(prod4);
                acc = acc.lazy_add(s12).lazy_add(s34);
                j += 4;
            }
            if j + 3 <= n_polys {
                let p1 = PBF::load_scalar_slice(&base_polys[j][off..]);
                let p2 = PBF::load_scalar_slice(&base_polys[j + 1][off..]);
                let p3 = PBF::load_scalar_slice(&base_polys[j + 2][off..]);

                let prod1 = powers_packed[j].mul_base_elem(p1);
                let prod2 = powers_packed[j + 1].mul_base_elem(p2);
                let prod3 = powers_packed[j + 2].mul_base_elem(p3);

                let s12 = prod1.lazy_add(prod2);
                acc = acc.lazy_add(s12).lazy_add(prod3);
                j += 3;
            } else if j + 2 <= n_polys {
                let p1 = PBF::load_scalar_slice(&base_polys[j][off..]);
                let p2 = PBF::load_scalar_slice(&base_polys[j + 1][off..]);

                let prod1 = powers_packed[j].mul_base_elem(p1);
                let prod2 = powers_packed[j + 1].mul_base_elem(p2);

                let s12 = prod1.lazy_add(prod2);
                acc = acc.lazy_add(s12);
                j += 2;
            }
            if j < n_polys {
                let pj = PBF::load_scalar_slice(&base_polys[j][off..]);
                let prod = powers_packed[j].mul_base_elem(pj);
                acc = acc.lazy_add(prod);
            }

            *out = acc.reduce_fast();
        }
    });

    out_packed
}

// =============================================================================
// Parallel split_fold (pcs_open substage)
// =============================================================================

/// Parallel version of `fold_sub_polys_packed` for Ext3.
///
/// Ping-pong design: reads from `src`, writes into `dst`, then swaps
/// the two `Vec` handles so the caller sees the folded output in `src`
/// and `dst` becomes reusable scratch. `dst` must have capacity >= the
/// full `src` length (in practice the caller pre-allocates
/// `sub_evals_scratch` to `sub_count * eval_len_packed` once, outside
/// the fold loop, so no per-round allocation happens in the parallel
/// path).
///
/// The in-place collect-and-swap variant regressed the serial baseline
/// by ~70% at nv=23 because the per-round `Vec::with_capacity` + drop
/// triggered mmap/munmap syscalls and first-touch page faults on the
/// fresh buffer every round. Pre-allocating scratch once reclaims that
/// cost.
pub fn fold_sub_polys_packed_ext3_par(
    src: &mut Vec<PEF3>,
    dst: &mut Vec<PEF3>,
    eval_len_packed: usize,
    num_subs: usize,
    challenge_packed: PEF3,
) {
    if num_subs < 2 {
        return;
    }
    let new_count = num_subs / 2;
    let total = new_count * eval_len_packed;

    if total < PAR_SPLIT_FOLD_MIN_ELEMS {
        <SEF3 as DeepFoldExtField>::fold_sub_polys_packed(
            src.as_mut_slice(),
            eval_len_packed,
            num_subs,
            challenge_packed,
        );
        src.truncate(total);
        return;
    }

    if dst.len() < total {
        dst.reserve(total - dst.len());
        unsafe {
            dst.set_len(total);
        }
    }
    let dst_prefix: &mut [PEF3] = &mut dst[..total];

    if new_count >= PAR_SPLIT_FOLD_OUTER_MIN {
        let src_slice: &[PEF3] = src.as_slice();
        dst_prefix
            .par_chunks_mut(eval_len_packed)
            .enumerate()
            .for_each(|(m, dst_slice)| {
                let src0_base = 2 * m * eval_len_packed;
                let src1_base = (2 * m + 1) * eval_len_packed;
                for j in 0..eval_len_packed {
                    let v0 = src_slice[src0_base + j];
                    let v1 = src_slice[src1_base + j];
                    let diff = v1 - v0;
                    dst_slice[j] = v0 + challenge_packed * diff;
                }
            });
    } else {
        let grain = eval_len_packed
            .div_ceil(rayon::current_num_threads() * 4)
            .max(4096);
        let segments: Vec<&mut [PEF3]> = dst_prefix.chunks_mut(eval_len_packed).collect();
        let src_slice: &[PEF3] = src.as_slice();
        for (m, dst_segment) in segments.into_iter().enumerate() {
            let src0_base = 2 * m * eval_len_packed;
            let src1_base = (2 * m + 1) * eval_len_packed;
            dst_segment
                .par_chunks_mut(grain)
                .enumerate()
                .for_each(|(chunk_idx, dst_chunk)| {
                    let base = chunk_idx * grain;
                    for (local, dst_elem) in dst_chunk.iter_mut().enumerate() {
                        let j = base + local;
                        let v0 = src_slice[src0_base + j];
                        let v1 = src_slice[src1_base + j];
                        let diff = v1 - v0;
                        *dst_elem = v0 + challenge_packed * diff;
                    }
                });
        }
    }

    std::mem::swap(src, dst);
    src.truncate(total);
}

// =============================================================================
// Parallel fold_multilinear_packed + eval_multilinear_packed (mle_eval / mlin_fold)
// =============================================================================

/// Parallel analogue of `fold_multilinear_packed_ext3_out_of_place`.
pub fn fold_multilinear_packed_ext3_out_of_place_par(
    src: &[PEF3],
    dst: &mut [PEF3],
    challenge_packed: PEF3,
) {
    debug_assert!(src.len() >= 2);
    debug_assert_eq!(dst.len(), src.len() / 2);

    if dst.len() < PAR_FOLD_MLIN_MIN_NEW_LEN {
        fold_multilinear_packed_ext3_out_of_place(src, dst, challenge_packed);
        return;
    }

    const EVEN_IDX: [u64; 8] = [0, 2, 4, 6, 8, 10, 12, 14];
    const ODD_IDX: [u64; 8] = [1, 3, 5, 7, 9, 11, 13, 15];

    let grain = dst
        .len()
        .div_ceil(rayon::current_num_threads() * 4)
        .max(1024);
    dst.par_chunks_mut(grain).enumerate().for_each(|(chunk_idx, chunk)| {
        let base = chunk_idx * grain;
        for (local, out) in chunk.iter_mut().enumerate() {
            let i = base + local;
            let lo = src[2 * i];
            let hi = src[2 * i + 1];
            let even_c0 = lo.c0.permute2(hi.c0, EVEN_IDX);
            let even_c1 = lo.c1.permute2(hi.c1, EVEN_IDX);
            let even_c2 = lo.c2.permute2(hi.c2, EVEN_IDX);
            let odd_c0 = lo.c0.permute2(hi.c0, ODD_IDX);
            let odd_c1 = lo.c1.permute2(hi.c1, ODD_IDX);
            let odd_c2 = lo.c2.permute2(hi.c2, ODD_IDX);
            let evens = PEF3::new(even_c0, even_c1, even_c2);
            let odds = PEF3::new(odd_c0, odd_c1, odd_c2);
            let diff = odds - evens;
            *out = evens + challenge_packed * diff;
        }
    });
}

/// Parallel in-place wrapper with ping-pong scratch. Takes `src` and a
/// caller-owned `dst` scratch; writes the half-sized result into `dst`,
/// then swaps `src`/`dst` so `src` holds the folded output on return.
///
/// Ping-pong design is critical: an earlier version allocated a fresh
/// Vec of half size per call, paying mmap + first-touch on every round
/// and wiping out the kernel's real 4-14x parallelism speedup (the
/// wrapper alone dominated wall clock).
pub fn fold_multilinear_packed_ext3_par(
    src: &mut Vec<PEF3>,
    dst: &mut Vec<PEF3>,
    challenge_packed: PEF3,
) {
    debug_assert!(src.len() >= 2);
    let new_len = src.len() / 2;

    // Below the par-fold crossover, run the exact serial in-place fold (no
    // ping-pong dst traffic / first-touch) so the result is byte-identical AND
    // timing-identical to the serial open. The out-of-place par fallback below
    // still touches the dst scratch, which made `multilin_fold` ~1.3x slower
    // than serial at nv=20 even when not actually parallel.
    if new_len < PAR_FOLD_MLIN_MIN_NEW_LEN {
        crate::deepfold_mamabear::fold_multilinear_packed_ext3(src, challenge_packed);
        return;
    }

    if dst.len() < new_len {
        dst.reserve(new_len - dst.len());
        unsafe {
            dst.set_len(new_len);
        }
    }
    let dst_prefix: &mut [PEF3] = &mut dst[..new_len];
    fold_multilinear_packed_ext3_out_of_place_par(src, dst_prefix, challenge_packed);

    std::mem::swap(src, dst);
    src.truncate(new_len);
}

/// Parallel analogue of `eval_multilinear_packed_ext3`. Same structure:
/// first round writes from the borrowed input into a fresh half-sized
/// scratch buffer (out-of-place, parallel), subsequent packed rounds fold
/// in place in parallel, then the scalar tail finishes the last 3 levels.
pub fn eval_multilinear_packed_ext3_par(
    poly_evals_packed: &[PEF3],
    point: &[SEF3],
) -> SEF3 {
    // Below the par-fold crossover, run the serial eval (no per-call ping-pong
    // scratch alloc + first-touch). The largest fold here is
    // poly_evals_packed.len()/2; if that is below the threshold the whole eval
    // would fold serially anyway, so call the serial path directly. This keeps
    // `mle_eval` timing-identical to serial below the crossover.
    if poly_evals_packed.len() / 2 < PAR_FOLD_MLIN_MIN_NEW_LEN {
        return crate::deepfold_mamabear::eval_multilinear_packed_ext3(poly_evals_packed, point);
    }

    let mut idx = 0;

    if poly_evals_packed.len() < 2 {
        let mut tail = [SEF3::default(); 8];
        poly_evals_packed[0].unpack_into_slice(&mut tail);
        let mut len = 8;
        while len > 1 && idx < point.len() {
            let r = point[idx];
            let half = len / 2;
            for j in 0..half {
                let v0 = tail[2 * j];
                let v1 = tail[2 * j + 1];
                tail[j] = v0 + r * (v1 - v0);
            }
            len = half;
            idx += 1;
        }
        return tail[0];
    }

    let first_new_len = poly_evals_packed.len() / 2;
    let mut scratch: Vec<PEF3> = Vec::with_capacity(first_new_len);
    let mut dst_scratch: Vec<PEF3> = Vec::with_capacity(first_new_len);
    unsafe {
        scratch.set_len(first_new_len);
        dst_scratch.set_len(first_new_len);
    }
    let r0 = point[idx];
    let r0_packed = PEF3::new(PBF::from(r0.c0.0), PBF::from(r0.c1.0), PBF::from(r0.c2.0));
    fold_multilinear_packed_ext3_out_of_place_par(poly_evals_packed, &mut scratch, r0_packed);
    idx += 1;

    while scratch.len() >= 2 {
        let r = point[idx];
        let r_packed = PEF3::new(PBF::from(r.c0.0), PBF::from(r.c1.0), PBF::from(r.c2.0));
        fold_multilinear_packed_ext3_par(&mut scratch, &mut dst_scratch, r_packed);
        idx += 1;
    }

    let mut tail = [SEF3::default(); 8];
    scratch[0].unpack_into_slice(&mut tail);
    let mut len = 8;
    while len > 1 && idx < point.len() {
        let r = point[idx];
        let half = len / 2;
        for j in 0..half {
            let v0 = tail[2 * j];
            let v1 = tail[2 * j + 1];
            tail[j] = v0 + r * (v1 - v0);
        }
        len = half;
        idx += 1;
    }
    tail[0]
}

// =============================================================================
// Parallel FRI merkle build (fri_mkl)
// =============================================================================

/// Parallel version of `FriFoldResult::from_packed_values`.
///
/// 3 phases, each parallelized independently:
/// 1. unpack + canonicalize every packed block into a flat scalar buffer,
/// 2. reorder evens/odds into `value_query_mont`,
/// 3. build the per-leaf byte buffer in parallel, hash leaves in parallel
///    batches, then build the Merkle tree via `from_leaf_hashes_par`.
///
/// Falls back to the serial constructor for small FRI rounds (where
/// thread overhead would dominate).
pub(crate) fn fri_fold_result_from_packed_values_par<F: DeepFoldExtField>(
    value_packed: Vec<F::PackedExt>,
) -> FriFoldResult<F> {
    let len = value_packed.len() * 8;
    let leave_num = len / 2;

    if leave_num < PAR_FRI_MKL_MIN_LEAVES {
        return FriFoldResult::from_packed_values(value_packed);
    }

    // Phase 1: parallel unpack + canonicalize.
    let mut canonical_by_position: Vec<F> = Vec::with_capacity(len);
    unsafe {
        canonical_by_position.set_len(len);
    }
    {
        let dst_addr = SendAddr::from_ptr(canonical_by_position.as_mut_ptr());
        value_packed.par_iter().enumerate().for_each(|(block_idx, block)| {
            let mut tmp = [F::default(); 8];
            block.unpack_into_slice(&mut tmp);
            let off = block_idx * 8;
            unsafe {
                let base = dst_addr.as_ptr::<F>();
                for k in 0..8 {
                    base.add(off + k).write(tmp[k].reduce_canonical());
                }
            }
        });
    }

    // Phase 2: parallel evens-then-odds reorder.
    let mut value_query_mont: Vec<F> = Vec::with_capacity(len);
    unsafe {
        value_query_mont.set_len(len);
    }
    {
        let (evens_half, odds_half) = value_query_mont.split_at_mut(leave_num);
        let canonical: &[F] = &canonical_by_position;
        let grain = leave_num
            .div_ceil(rayon::current_num_threads() * 4)
            .max(1024);
        rayon::join(
            || {
                evens_half
                    .par_chunks_mut(grain)
                    .enumerate()
                    .for_each(|(chunk_idx, chunk)| {
                        let base = chunk_idx * grain;
                        for (local, out) in chunk.iter_mut().enumerate() {
                            *out = canonical[2 * (base + local)];
                        }
                    });
            },
            || {
                odds_half
                    .par_chunks_mut(grain)
                    .enumerate()
                    .for_each(|(chunk_idx, chunk)| {
                        let base = chunk_idx * grain;
                        for (local, out) in chunk.iter_mut().enumerate() {
                            *out = canonical[2 * (base + local) + 1];
                        }
                    });
            },
        );
    }

    // Phase 3a: parallel flatten leaves to a contiguous byte buffer.
    // Each leaf = [value_query_mont[i], value_query_mont[i + leave_num]].
    let leaf_bytes = 2 * F::SIZE;
    let mut flat: Vec<u8> = Vec::with_capacity(leave_num * leaf_bytes);
    unsafe {
        flat.set_len(leave_num * leaf_bytes);
    }
    {
        let vq: &[F] = &value_query_mont;
        flat.par_chunks_mut(leaf_bytes)
            .enumerate()
            .for_each(|(i, leaf)| {
                vq[i].serialize_into(&mut leaf[..F::SIZE]);
                vq[i + leave_num].serialize_into(&mut leaf[F::SIZE..leaf_bytes]);
            });
    }

    // Phase 3b: parallel leaf hashing via blake3_batch.
    let leaf_hashes = hash_leaves_flat_par(&flat, leave_num, leaf_bytes);

    // Phase 3c: parallel Merkle tree build (parents).
    let merkle_tree = MerkleTreeProverMB::from_leaf_hashes_par(&leaf_hashes);

    FriFoldResult {
        value_packed,
        value_query_mont,
        merkle_tree,
    }
}

/// Hash a flat-layout leaf buffer (`leaf_count` contiguous leaves, each
/// `leaf_bytes` wide) in parallel batches. Mirrors the batching pattern
/// of `round0_leaf_hashes_par` but for the generic `[x, nx]` FRI leaf
/// layout.
fn hash_leaves_flat_par(
    flat: &[u8],
    leaf_count: usize,
    leaf_bytes: usize,
) -> Vec<[u8; HASH_SIZE]> {
    debug_assert_eq!(flat.len(), leaf_count * leaf_bytes);

    // Batch size aimed at ~32 KiB per batch (cache-friendly for blake3
    // SIMD, matches the round0 target).
    let target_batch_bytes: usize = 32 * 1024;
    let batch_leaves_target = (target_batch_bytes / leaf_bytes.max(1)).max(1);
    let n_threads = rayon::current_num_threads();
    // Also ensure we have enough batches to saturate all worker threads.
    let batch_leaves_saturation = leaf_count.div_ceil(n_threads * 4).max(1);
    let batch_leaves = batch_leaves_target.min(batch_leaves_saturation.max(batch_leaves_target));

    // (Simplification: pick the larger of the two so we don't underflow
    // batches at small leaf_count.)
    let batch_leaves = batch_leaves.max(1);

    let batch_ranges: Vec<(usize, usize)> = (0..leaf_count)
        .step_by(batch_leaves)
        .map(|start| (start, (leaf_count - start).min(batch_leaves)))
        .collect();

    let mut leaf_hashes = vec![[0u8; HASH_SIZE]; leaf_count];

    if batch_ranges.len() <= 1 {
        blake3_batch::hash_leaves_batch_flat(flat, leaf_count, leaf_bytes, &mut leaf_hashes);
        return leaf_hashes;
    }

    let hashes_addr = SendAddr::from_ptr(leaf_hashes.as_mut_ptr());
    batch_ranges.par_iter().for_each(|&(start, active)| {
        let byte_off = start * leaf_bytes;
        let byte_len = active * leaf_bytes;
        let dst = unsafe {
            std::slice::from_raw_parts_mut(
                hashes_addr.as_ptr::<[u8; HASH_SIZE]>().add(start),
                active,
            )
        };
        blake3_batch::hash_leaves_batch_flat(
            &flat[byte_off..byte_off + byte_len],
            active,
            leaf_bytes,
            dst,
        );
    });

    leaf_hashes
}

// =============================================================================
// Parallel within-round FRI fold (evaluate_next_domain*)
// =============================================================================
//
// The serial kernels (`evaluate_next_domain_first_round_packed_ext3` and
// `evaluate_next_domain_packed_ext3` in deepfold_mamabear.rs) are pure SIMD
// loops `for chunk in 0..packed_pairs` where `out[chunk]` depends ONLY on
// `last_packed[2*chunk]`, `last_packed[2*chunk+1]`, and a per-chunk twiddle
// (`fft.bit_reversed_pair_element_inv_at`, a read-only table lookup). No
// accumulation, no hashing, no transcript -> embarrassingly data-parallel and
// byte-identical to serial regardless of thread count or grain.
//
// Round 0 (first-round) reads two adjacent packed blocks as `[x.., nx..]`
// directly; rounds >= 1 extract (x, nx) via `permute2` over the two blocks.
// Both write `result_packed[chunk]` in natural order (same as serial `push`),
// so the output buffer is bit-identical.

/// Parallel `evaluate_next_domain_first_round_packed_ext3`.
pub fn evaluate_next_domain_first_round_packed_ext3_par(
    last_packed: &[PEF3],
    fft: &MamaBearFFT,
    challenge_packed: PEF3,
) -> Vec<PEF3> {
    let len = fft.size();
    let pair_count = len / 2;
    debug_assert_eq!(last_packed.len(), len / 8);
    let packed_pairs = pair_count / 8;

    if packed_pairs < PAR_FRI_FOLD_MIN_PAIRS {
        return evaluate_next_domain_first_round_packed_ext3(last_packed, fft, challenge_packed);
    }

    let inv_2_mont = MamaBearScalar::inv_2().to_montgomery();
    let inv_2_packed = PBF::from(inv_2_mont.0);

    let mut result_packed: Vec<PEF3> = Vec::with_capacity(packed_pairs);
    unsafe {
        result_packed.set_len(packed_pairs);
    }

    let grain = packed_pairs
        .div_ceil(rayon::current_num_threads() * 4)
        .max(256);
    result_packed
        .par_chunks_mut(grain)
        .enumerate()
        .for_each(|(chunk_base_idx, out_slice)| {
            let chunk_base = chunk_base_idx * grain;
            for (local, out) in out_slice.iter_mut().enumerate() {
                let chunk = chunk_base + local;
                let x_packed = last_packed[2 * chunk];
                let nx_packed = last_packed[2 * chunk + 1];

                let base = chunk * 8;
                let mut inv_w_arr = [0u64; 8];
                for i in 0..8 {
                    inv_w_arr[i] = fft.bit_reversed_pair_element_inv_at(base + i).0;
                }
                let inv_w_packed = PBF::from_array(inv_w_arr);

                let sum = x_packed + nx_packed;
                let diff = x_packed - nx_packed;
                let diff_scaled = diff.mul_base_elem(inv_w_packed);
                let t = PEF3::new(
                    diff_scaled.c0.lazy_add_xp(2).lazy_sub(sum.c0).con_sub_xp(2),
                    diff_scaled.c1.lazy_add_xp(2).lazy_sub(sum.c1).con_sub_xp(2),
                    diff_scaled.c2.lazy_add_xp(2).lazy_sub(sum.c2).con_sub_xp(2),
                );
                let ct = challenge_packed * t;
                let new_v = sum + ct;
                let scaled = new_v.mul_base_elem(inv_2_packed);
                *out = PEF3::new(
                    scaled.c0.con_sub_xp(1),
                    scaled.c1.con_sub_xp(1),
                    scaled.c2.con_sub_xp(1),
                );
            }
        });

    result_packed
}

/// Parallel `evaluate_next_domain_packed_ext3` (rounds >= 1).
pub fn evaluate_next_domain_packed_ext3_par(
    last_packed: &[PEF3],
    fft: &MamaBearFFT,
    challenge_packed: PEF3,
) -> Vec<PEF3> {
    let len = fft.size();
    let pair_count = len / 2;
    debug_assert_eq!(last_packed.len(), len / 8);
    let packed_pairs = pair_count / 8;

    if packed_pairs < PAR_FRI_FOLD_MIN_PAIRS {
        return evaluate_next_domain_packed_ext3(last_packed, fft, challenge_packed);
    }

    let inv_2_mont = MamaBearScalar::inv_2().to_montgomery();
    let inv_2_packed = PBF::from(inv_2_mont.0);

    const EVEN_IDX: [u64; 8] = [0, 2, 4, 6, 8, 10, 12, 14];
    const ODD_IDX: [u64; 8] = [1, 3, 5, 7, 9, 11, 13, 15];

    let mut result_packed: Vec<PEF3> = Vec::with_capacity(packed_pairs);
    unsafe {
        result_packed.set_len(packed_pairs);
    }

    let grain = packed_pairs
        .div_ceil(rayon::current_num_threads() * 4)
        .max(256);
    result_packed
        .par_chunks_mut(grain)
        .enumerate()
        .for_each(|(chunk_base_idx, out_slice)| {
            let chunk_base = chunk_base_idx * grain;
            for (local, out) in out_slice.iter_mut().enumerate() {
                let chunk = chunk_base + local;
                let lo = last_packed[2 * chunk];
                let hi = last_packed[2 * chunk + 1];

                let x_c0 = lo.c0.permute2(hi.c0, EVEN_IDX);
                let x_c1 = lo.c1.permute2(hi.c1, EVEN_IDX);
                let x_c2 = lo.c2.permute2(hi.c2, EVEN_IDX);
                let nx_c0 = lo.c0.permute2(hi.c0, ODD_IDX);
                let nx_c1 = lo.c1.permute2(hi.c1, ODD_IDX);
                let nx_c2 = lo.c2.permute2(hi.c2, ODD_IDX);

                let x_packed = PEF3::new(x_c0, x_c1, x_c2);
                let nx_packed = PEF3::new(nx_c0, nx_c1, nx_c2);

                let base = chunk * 8;
                let mut inv_w_arr = [0u64; 8];
                for i in 0..8 {
                    inv_w_arr[i] = fft.bit_reversed_pair_element_inv_at(base + i).0;
                }
                let inv_w_packed = PBF::from_array(inv_w_arr);

                let sum = x_packed + nx_packed;
                let diff = x_packed - nx_packed;
                let diff_scaled = diff.mul_base_elem(inv_w_packed);
                let t = PEF3::new(
                    diff_scaled.c0.lazy_add_xp(2).lazy_sub(sum.c0).con_sub_xp(2),
                    diff_scaled.c1.lazy_add_xp(2).lazy_sub(sum.c1).con_sub_xp(2),
                    diff_scaled.c2.lazy_add_xp(2).lazy_sub(sum.c2).con_sub_xp(2),
                );
                let ct = challenge_packed * t;
                let new_v = sum + ct;
                let scaled = new_v.mul_base_elem(inv_2_packed);
                *out = PEF3::new(
                    scaled.c0.con_sub_xp(1),
                    scaled.c1.con_sub_xp(1),
                    scaled.c2.con_sub_xp(1),
                );
            }
        });

    result_packed
}

// =============================================================================
// Parallel DeepFold verify
// =============================================================================
//
// Mirrors `DeepFoldMamaBearVerifier::verify_inner` in `deepfold_mamabear.rs`
// but parallelizes the two Merkle-verification hot spots via rayon:
//
//   * `std_fri_mkl` — the `commits.len()` (typically ~20 at nv=24 / split=5)
//     standard-FRI round Merkle verifies are independent once phase A
//     (proof cursor advance + transcript append + `QueryResultMB` build) is
//     done serially. Phase B `par_iter`s over the per-round query results.
//
//   * `fat_mkl` — the `verifiers.len()` fat-leaf Merkle verifies (typically
//     2: base commitment + witness) are independent once the transcript
//     absorption is done serially. The per-round `poly_values` regroup is
//     also independent and gets collected from the per-verifier task
//     outputs before the combine-and-split-fold step.

impl<F: DeepFoldExtField> DeepFoldMamaBearVerifier<F> {
    /// Parallel variant of `verify` — see module comment above.
    pub fn verify_par(
        pp: &DeepFoldMamaBearParam,
        verifiers: Vec<&Self>,
        point: Vec<F>,
        evals: Vec<Vec<F>>,
        transcript: &mut Transcript,
        proof: &mut Proof,
    ) -> bool {
        let mut t = VerifyTimings::default();
        Self::verify_inner_par(pp, verifiers, point, evals, transcript, proof, &mut t, false)
    }

    /// Profiling variant of `verify_par`.
    pub fn verify_par_profiled(
        pp: &DeepFoldMamaBearParam,
        verifiers: Vec<&Self>,
        point: Vec<F>,
        evals: Vec<Vec<F>>,
        transcript: &mut Transcript,
        proof: &mut Proof,
        timings: &mut VerifyTimings,
    ) -> bool {
        Self::verify_inner_par(pp, verifiers, point, evals, transcript, proof, timings, true)
    }

    fn verify_inner_par(
        pp: &DeepFoldMamaBearParam,
        verifiers: Vec<&Self>,
        point: Vec<F>,
        evals: Vec<Vec<F>>,
        transcript: &mut Transcript,
        proof: &mut Proof,
        timings: &mut VerifyTimings,
        record: bool,
    ) -> bool {
        macro_rules! now {
            () => {
                if record {
                    Some(Instant::now())
                } else {
                    None
                }
            };
        }
        macro_rules! tick {
            ($t:ident, $field:ident) => {
                if let Some(t) = $t {
                    timings.$field += t.elapsed().as_micros();
                }
            };
        }

        let total_t0 = now!();

        let split_level = pp.split_level;
        let sub_count = 1usize << split_level;

        // --- Fold consistency check (serial: Fiat-Shamir dependency) ---
        let fold_t0 = now!();
        let r_raw: F = transcript.challenge_f();
        let r_mont = r_raw.to_mont();
        let mut eval_mont = F::zero().to_mont();
        for i in evals {
            for j in i {
                eval_mont = eval_mont * r_mont;
                eval_mont = eval_mont + j;
            }
        }

        let point_mont: Vec<F> = point.iter().map(|x| x.to_mont()).collect();
        let mut challenges_mont = vec![];
        let mut commits: Vec<MerkleTreeVerifierMB> = vec![];

        // DeepFold OOD binding — this par verifier inlines its own fold loop (it does NOT
        // delegate to verify_pre_round0_fold_check), so the DEEP logic is mirrored here via
        // the shared deep_verify_* helpers, kept byte-for-byte in lock-step with serial.
        // The OOD challenge alpha binds the committed vector to a unique
        // list-decoded codeword; without it the bare Merkle root does not.
        let mut deep =
            crate::deepfold_mamabear::deep_verify_init::<F>(pp.variable_num, transcript, proof);

        for i in 0..point.len() {
            let (next_eval_mont, deep_offs) = crate::deepfold_mamabear::deep_verify_reads::<F>(
                i,
                pp.variable_num,
                &mut deep,
                transcript,
                proof,
            );
            let challenge_raw: F = transcript.challenge_f();
            let challenge_mont = challenge_raw.to_mont();

            eval_mont = eval_mont + (challenge_mont - point_mont[i]) * (next_eval_mont - eval_mont);
            challenges_mont.push(challenge_mont);
            crate::deepfold_mamabear::deep_verify_update::<F>(&mut deep, &deep_offs, challenge_mont);

            if i < split_level {
                // Split fold round: no Merkle commitment
            } else if i < pp.variable_num - 1 {
                let merkle_root = proof.get_next_hash();
                transcript.append_u8_slice(&merkle_root, MT_HASH_SIZE);
                commits.push(MerkleTreeVerifierMB::new(
                    pp.fft_groups[i - split_level + 1].size() / 2,
                    merkle_root,
                ));
            } else {
                let final_mont = proof.get_next_and_step::<F>();
                transcript.append_f(final_mont);
                if final_mont.reduce_canonical() != eval_mont.reduce_canonical() {
                    tick!(fold_t0, fold_check_us);
                    tick!(total_t0, total_us);
                    return false;
                }
                if !crate::deepfold_mamabear::deep_verify_terminal_ok::<F>(&deep, eval_mont) {
                    tick!(fold_t0, fold_check_us);
                    tick!(total_t0, total_us);
                    return false;
                }
            }
        }
        tick!(fold_t0, fold_check_us);

        // --- Grinding (PoW) verification ---
        let grind_t0 = now!();
        let grind_ok = transcript.verify_grind(proof, pp.grinding_bits);
        tick!(grind_t0, grinding_us);
        if !grind_ok {
            tick!(total_t0, total_us);
            return false;
        }

        // --- Query index derivation ---
        let qprep_t0 = now!();
        let mut leaf_indices = transcript.challenge_usizes(pp.query_num);
        let mut query_results: Vec<QueryResultMB<F>> = vec![];

        let fat_domain = pp.fft_groups[0].size();
        leaf_indices = leaf_indices
            .iter_mut()
            .map(|v| *v % (fat_domain >> 1))
            .collect();
        leaf_indices.sort();
        leaf_indices.dedup();
        let mut indices = leaf_indices.clone();
        tick!(qprep_t0, query_prep_us);

        {
            // --- Fat-leaf Merkle verify + per-prover regroup ---
            //
            // Phase A (serial): read proof bytes/values + transcript absorb, and
            //   stash the per-verifier (proof_bytes, proof_values, fat_leaf_size)
            //   for phase B to consume.
            // Phase B (parallel): regroup poly_values + Merkle-verify each
            //   verifier's fat-leaf tree independently.
            let fat_t0 = now!();
            let mut stash: Vec<(Vec<u8>, Vec<MamaBearScalar>, usize)> =
                Vec::with_capacity(verifiers.len());
            for verifier in verifiers.iter() {
                let fat_leaf_size = 2 * sub_count * verifier.poly_num;
                let proof_bytes = proof.get_next_slice(verifier.commit.proof_length(&leaf_indices));
                let proof_values: Vec<MamaBearScalar> = (0..leaf_indices.len() * fat_leaf_size)
                    .map(|_| proof.get_next_and_step::<MamaBearScalar>())
                    .collect();
                transcript.append_u8_slice(&proof_bytes, proof_bytes.len());
                for k in &proof_values {
                    transcript.append_f(*k);
                }
                stash.push((proof_bytes, proof_values, fat_leaf_size));
            }

            // (2026-04-26): par version. Closure returns
            // `Option<Vec<Vec<MamaBearScalar>>>`: `Some(chunk)` on success,
            // `None` if any merkle verify or hashmap lookup fails. Outer
            // collects then checks all-Some; otherwise return false from
            // verify_inner. Replaces the `.unwrap()` + `assert!` panic
            // path that used to crash the verifier on tampered proofs.
            use rayon::prelude::*;
            let poly_values_per_verifier_opt: Vec<Option<Vec<Vec<MamaBearScalar>>>> =
                verifiers
                    .par_iter()
                    .zip(stash.par_iter())
                    .map(|(verifier, (proof_bytes, proof_values, fat_leaf_size))| {
                        let fat_leaf_size = *fat_leaf_size;
                        // Regroup values by polynomial (sub_count groups per poly).
                        let mut local: Vec<Vec<MamaBearScalar>> =
                            Vec::with_capacity(verifier.poly_num * sub_count);
                        for p in 0..verifier.poly_num {
                            for k in 0..sub_count {
                                let slot_j = (p * sub_count + k) * 2;
                                let slot_jh = (p * sub_count + k) * 2 + 1;
                                let mut vals =
                                    Vec::with_capacity(leaf_indices.len() * 2);
                                for q in 0..leaf_indices.len() {
                                    vals.push(
                                        proof_values[slot_j * leaf_indices.len() + q],
                                    );
                                }
                                for q in 0..leaf_indices.len() {
                                    vals.push(
                                        proof_values[slot_jh * leaf_indices.len() + q],
                                    );
                                }
                                local.push(vals);
                            }
                        }

                        // Merkle verify
                        let base_query_values: HashMap<usize, MamaBearScalar> =
                            proof_values
                                .iter()
                                .copied()
                                .enumerate()
                                .map(|(idx, x)| {
                                    (
                                        leaf_indices[idx % leaf_indices.len()]
                                            + (fat_domain / 2)
                                                * (idx / leaf_indices.len()),
                                        x,
                                    )
                                })
                                .collect();
                        let mut base_leaves: Vec<Vec<u8>> =
                            Vec::with_capacity(leaf_indices.len());
                        for i in leaf_indices.iter() {
                            let mut row: Vec<MamaBearScalar> =
                                Vec::with_capacity(fat_leaf_size);
                            for j in 0..fat_leaf_size {
                                match base_query_values
                                    .get(&(i + j * (fat_domain / 2)))
                                {
                                    Some(v) => row.push(*v),
                                    None => return None,
                                }
                            }
                            base_leaves.push(as_bytes_vec(&row));
                        }
                        if !verifier
                            .commit
                            .verify(proof_bytes, &leaf_indices, &base_leaves)
                        {
                            return None;
                        }
                        Some(local)
                    })
                    .collect();
            // Convert Vec<Option<chunk>> → Option<Vec<chunk>>; if any None,
            // verify rejected.
            let poly_values_per_verifier: Vec<Vec<Vec<MamaBearScalar>>> =
                match poly_values_per_verifier_opt
                    .into_iter()
                    .collect::<Option<Vec<_>>>()
                {
                    Some(v) => v,
                    None => {
                        tick!(fat_t0, fat_merkle_us);
                        tick!(total_t0, total_us);
                        return false;
                    }
                };
            let mut poly_values: Vec<Vec<MamaBearScalar>> =
                Vec::with_capacity(poly_values_per_verifier.iter().map(|v| v.len()).sum());
            for chunk in poly_values_per_verifier {
                poly_values.extend(chunk);
            }
            tick!(fat_t0, fat_merkle_us);

            // --- Split-fold in extension field (serial; small inner work) ---
            let split_t0 = now!();
            let num_queries = leaf_indices.len();
            let total_sub_groups = poly_values.len();

            let mut combined_j: Vec<Vec<F>> = vec![Vec::new(); sub_count];
            let mut combined_jh: Vec<Vec<F>> = vec![Vec::new(); sub_count];
            for k in 0..sub_count {
                combined_j[k] = vec![F::zero().to_mont(); num_queries];
                combined_jh[k] = vec![F::zero().to_mont(); num_queries];
            }

            for group_idx in 0..(total_sub_groups / sub_count) {
                for k in 0..sub_count {
                    let pv = &poly_values[group_idx * sub_count + k];
                    for q in 0..num_queries {
                        combined_j[k][q] = combined_j[k][q] * r_mont;
                        combined_j[k][q] = combined_j[k][q] + F::from_base_mont(pv[q]);
                        combined_jh[k][q] = combined_jh[k][q] * r_mont;
                        combined_jh[k][q] =
                            combined_jh[k][q] + F::from_base_mont(pv[num_queries + q]);
                    }
                }
            }

            let mut cur_sub_count = sub_count;
            for round in 0..split_level {
                let new_count = cur_sub_count / 2;
                for m in 0..new_count {
                    for q in 0..num_queries {
                        let v0_j = combined_j[2 * m][q];
                        let v1_j = combined_j[2 * m + 1][q];
                        combined_j[m][q] = v0_j + challenges_mont[round] * (v1_j - v0_j);
                        let v0_jh = combined_jh[2 * m][q];
                        let v1_jh = combined_jh[2 * m + 1][q];
                        combined_jh[m][q] = v0_jh + challenges_mont[round] * (v1_jh - v0_jh);
                    }
                }
                cur_sub_count = new_count;
            }

            query_results.push(QueryResultMB {
                proof_bytes: vec![],
                proof_values: leaf_indices
                    .iter()
                    .enumerate()
                    .map(|(idx, &x)| (2 * x, combined_j[0][idx]))
                    .chain(
                        leaf_indices
                            .iter()
                            .enumerate()
                            .map(|(idx, &x)| (2 * x + 1, combined_jh[0][idx])),
                    )
                    .collect(),
            });
            tick!(split_t0, split_fold_us);
        }

        // --- Standard FRI query Merkle verification (parallel across rounds) ---
        //
        // Phase A (serial): advance proof cursor, absorb into transcript, build
        //   per-round `QueryResultMB` and `leaf_indices` snapshots.
        // Phase B (parallel): rayon `par_iter` each round's Merkle path check.
        let stdm_t0 = now!();
        let phase_a_start = query_results.len();
        let mut per_round_indices: Vec<Vec<usize>> = Vec::with_capacity(commits.len());
        for k in 0..commits.len() {
            leaf_indices = leaf_indices.iter().map(|v| *v >> 1).collect();
            leaf_indices.sort();
            leaf_indices.dedup();

            let proof_bytes = proof.get_next_slice(commits[k].proof_length(&leaf_indices));
            let proof_values = (0..leaf_indices.len() * 2)
                .map(|_| proof.get_next_and_step::<F>())
                .collect::<Vec<_>>();
            transcript.append_u8_slice(&proof_bytes, proof_bytes.len());
            for j in &proof_values {
                transcript.append_f(*j);
            }
            let query = QueryResultMB {
                proof_bytes,
                proof_values: leaf_indices
                    .iter()
                    .enumerate()
                    .map(|(idx, &x)| (2 * x, proof_values[idx]))
                    .chain(
                        leaf_indices
                            .iter()
                            .enumerate()
                            .map(|(idx, &x)| (2 * x + 1, proof_values[leaf_indices.len() + idx])),
                    )
                    .collect(),
            };
            per_round_indices.push(leaf_indices.clone());
            query_results.push(query);
        }
        drop(leaf_indices);

        // (2026-04-26): par version. Same fix as serial:
        // propagate `false` from `verify_merkle_tree` instead of ignoring
        // it. We collect per-thread bools then reduce — if ANY round
        // fails its merkle check, the whole verify returns false.
        {
            use rayon::prelude::*;
            let slice = &query_results[phase_a_start..];
            let all_ok: bool = slice
                .par_iter()
                .zip(per_round_indices.par_iter())
                .zip(commits.par_iter())
                .map(|((query, indices), commit)| {
                    query.verify_merkle_tree(indices, 2, commit)
                })
                .reduce(|| true, |a, b| a && b);
            if !all_ok {
                tick!(stdm_t0, std_fri_merkle_us);
                return false;
            }
        }
        tick!(stdm_t0, std_fri_merkle_us);

        // --- FRI fold + consistency check (serial; tiny, chain-dependent) ---
        let folds_t0 = now!();
        let inv_2_mont = MamaBearScalar::inv_2().to_montgomery();
        for i in split_level..pp.variable_num {
            let fft_idx = i - split_level;
            let qr_idx = i - split_level;

            for &j in indices.iter() {
                let x = *query_results[qr_idx].proof_values.get(&(2 * j)).unwrap();
                let nx = *query_results[qr_idx]
                    .proof_values
                    .get(&(2 * j + 1))
                    .unwrap();
                let sum = x + nx;
                let inv_w = pp.fft_groups[fft_idx].bit_reversed_pair_element_inv_at(j);
                let new_v = sum + challenges_mont[i] * ((x - nx).mul_base_elem(inv_w) - sum);
                if i < pp.variable_num - 1 {
                    let next_val = query_results[qr_idx + 1].proof_values[&j];
                    if new_v.reduce_canonical() != next_val.double().reduce_canonical() {
                        tick!(folds_t0, fri_folds_us);
                        tick!(total_t0, total_us);
                        return false;
                    }
                } else {
                    let check = new_v.mul_base_elem(inv_2_mont);
                    if check.reduce_canonical() != eval_mont.reduce_canonical() {
                        tick!(folds_t0, fri_folds_us);
                        tick!(total_t0, total_us);
                        return false;
                    }
                }
            }

            if i < pp.variable_num - 1 {
                indices = indices.iter().map(|v| *v >> 1).collect();
                indices.sort();
                indices.dedup();
            }
        }
        tick!(folds_t0, fri_folds_us);
        tick!(total_t0, total_us);
        true
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::deepfold_mamabear::{DeepFoldMamaBearParam, DeepFoldMamaBearProver};
    use arithmetic::field::mamabear::{MamaBearScalar, MamaBearScalarExt3};
    use arithmetic::field::Field;
    use rand::rngs::SmallRng;
    use rand::SeedableRng;

    fn random_poly(n: usize, seed: u64) -> Vec<MamaBearScalar> {
        let mut rng = SmallRng::seed_from_u64(seed);
        (0..n).map(|_| MamaBearScalar::random(&mut rng)).collect()
    }

    /// Verify that parallel commit produces the same Merkle root as serial commit.
    fn test_new_par_matches_serial<F: DeepFoldExtField>(nv: usize) {
        let n = 1usize << nv;
        let polys: Vec<Vec<MamaBearScalar>> = (0..3).map(|i| random_poly(n, 0xDEAD_BEEF + i as u64)).collect();
        let poly_refs: Vec<&[MamaBearScalar]> = polys.iter().map(|p| p.as_slice()).collect();

        let pp = DeepFoldMamaBearParam::new_default(nv, 3, 34);

        let serial = DeepFoldMamaBearProver::<F>::new(&pp, &poly_refs);
        let parallel = new_par::<F>(&pp, &poly_refs);

        // Compare Merkle roots
        let serial_root = serial.commit();
        let parallel_root = parallel.commit();
        assert_eq!(
            serial_root.0, parallel_root.0,
            "Merkle root mismatch at nv={}", nv
        );

        // Compare sub_evals_mont
        assert_eq!(
            serial.sub_evals_mont.len(),
            parallel.sub_evals_mont.len(),
        );
        assert_eq!(
            &*serial.sub_evals_mont,
            &*parallel.sub_evals_mont,
            "sub_evals_mont mismatch at nv={}", nv
        );
    }

    #[test]
    fn new_par_ext3_nv14() {
        test_new_par_matches_serial::<SEF3>(14);
    }

    /// Verify that parallel combine produces the same output as serial combine.
    #[test]
    fn combine_ext3_par_matches_serial() {
        use crate::deepfold_mamabear::combine_opt_ext3_inner_test;
        let mut rng = SmallRng::seed_from_u64(42);
        let n = 8 * 1024;
        let n_polys = 7;
        let polys: Vec<Vec<MamaBearScalar>> = (0..n_polys).map(|i| random_poly(n, 42 + i as u64)).collect();
        let poly_refs: Vec<&[MamaBearScalar]> = polys.iter().map(|p| p.as_slice()).collect();
        let r = MamaBearScalarExt3::random(&mut rng).to_montgomery();

        let serial = combine_opt_ext3_inner_test::<true>(&poly_refs, r);
        let parallel = combine_opt_ext3_inner_par::<true>(&poly_refs, r);
        assert_eq!(serial.len(), parallel.len());
        for i in 0..serial.len() {
            assert_eq!(serial[i], parallel[i], "Mismatch at chunk {}", i);
        }
    }

    // ---- combine crossover microbench (PAR_COMBINE_MIN_CHUNKS tuning) ----
    //
    // The combine kernel (`combine_opt_ext3_inner_par`) chunk-parallelizes a
    // Horner combination of `n_polys` base polys. The open path uses n_polys=7
    // (HyperPlonk K). This sweep finds the `chunks_packed` (= len/8) crossover
    // where parallel overtakes serial so we can gate the par kernel below it.
    fn bench_combine_case_ext3(chunks_packed: usize, n_polys: usize) {
        let n = chunks_packed * 8;
        let polys: Vec<Vec<MamaBearScalar>> =
            (0..n_polys).map(|i| random_poly(n, 0x1234 + i as u64)).collect();
        let poly_refs: Vec<&[MamaBearScalar]> = polys.iter().map(|p| p.as_slice()).collect();
        let mut rng = SmallRng::seed_from_u64(0xC0DE);
        let r = MamaBearScalarExt3::random(&mut rng).to_montgomery();
        let reps = 20;
        let ser_ns = measure_ns(
            || {
                let _ = crate::deepfold_mamabear::combine_opt_ext3_inner_test::<true>(&poly_refs, r);
            },
            reps,
        );
        let par_ns = measure_ns(
            || {
                let _ = combine_opt_ext3_inner_par::<true>(&poly_refs, r);
            },
            reps,
        );
        println!(
            "ext3 n_polys={:>2} chunks_packed={:>8}  ser={:>9.3}us  par={:>9.3}us  speedup={:.2}x",
            n_polys,
            chunks_packed,
            ser_ns as f64 / 1e3,
            par_ns as f64 / 1e3,
            ser_ns as f64 / par_ns as f64,
        );
    }

    /// Commit (new_par) crossover vs serial new — confirms the witness-commit
    /// entry point is never slower than serial, covering the merkle
    /// `PARALLEL_THRESHOLD`, leaf-hash, and parallel-FFT thresholds together.
    #[test]
    #[ignore]
    fn microbench_new_par_ext3() {
        println!("\n=== DeepFoldMamaBearProver::new (serial) vs new_par (Ext3, 3 polys) ===");
        for &nv in &[10usize, 12, 14, 16, 18, 20] {
            let n = 1usize << nv;
            let polys: Vec<Vec<MamaBearScalar>> =
                (0..3).map(|i| random_poly(n, 0xC0DE + i as u64)).collect();
            let poly_refs: Vec<&[MamaBearScalar]> = polys.iter().map(|p| p.as_slice()).collect();
            let pp = DeepFoldMamaBearParam::new_default(nv, 3, 34);
            let reps = 5;
            let ser_ns = measure_ns(
                || {
                    let _ = DeepFoldMamaBearProver::<SEF3>::new(&pp, &poly_refs).commit();
                },
                reps,
            );
            let par_ns = measure_ns(
                || {
                    let _ = new_par::<SEF3>(&pp, &poly_refs).commit();
                },
                reps,
            );
            println!(
                "nv={:>2} leaves={:>10}  ser={:>9.3}ms  par={:>9.3}ms  speedup={:.2}x",
                nv,
                n / 2,
                ser_ns as f64 / 1e6,
                par_ns as f64 / 1e6,
                ser_ns as f64 / par_ns as f64,
            );
        }
    }

    #[test]
    #[ignore]
    fn microbench_combine_ext3() {
        println!("\n=== combine_opt_ext3_inner serial vs parallel (PAR_COMBINE_MIN_CHUNKS) ===");
        for &n_polys in &[7usize, 21] {
            for &log_chunks in &[8u32, 9, 10, 11, 12, 13, 14, 15, 16, 17] {
                bench_combine_case_ext3(1usize << log_chunks, n_polys);
            }
            println!();
        }
    }

    // Helper to build random packed inputs for the kernel tests.
    fn random_packed_ext3(rng: &mut SmallRng, count: usize) -> Vec<PEF3> {
        (0..count)
            .map(|_| {
                let c0_arr = std::array::from_fn(|_| MamaBearScalar::random(&mut *rng).to_montgomery().0);
                let c1_arr = std::array::from_fn(|_| MamaBearScalar::random(&mut *rng).to_montgomery().0);
                let c2_arr = std::array::from_fn(|_| MamaBearScalar::random(&mut *rng).to_montgomery().0);
                PEF3::new(
                    PBF::from_array(c0_arr),
                    PBF::from_array(c1_arr),
                    PBF::from_array(c2_arr),
                )
            })
            .collect()
    }

    // split_fold (fold_sub_polys_packed) parity tests.
    // Cover: outer-parallel (num_subs=32), outer-parallel (num_subs=8),
    //        inner-parallel (num_subs=4), and serial fallback (num_subs=2).
    fn check_fold_sub_polys_ext3(num_subs: usize, eval_len_packed: usize, seed: u64) {
        let mut rng = SmallRng::seed_from_u64(seed);
        let total = num_subs * eval_len_packed;
        let base = random_packed_ext3(&mut rng, total);
        let challenge_c0 = MamaBearScalar::random(&mut rng).to_montgomery().0;
        let challenge_c1 = MamaBearScalar::random(&mut rng).to_montgomery().0;
        let challenge_c2 = MamaBearScalar::random(&mut rng).to_montgomery().0;
        let challenge = PEF3::new(
            PBF::from(challenge_c0),
            PBF::from(challenge_c1),
            PBF::from(challenge_c2),
        );

        let mut serial_buf = base.clone();
        <SEF3 as DeepFoldExtField>::fold_sub_polys_packed(
            &mut serial_buf,
            eval_len_packed,
            num_subs,
            challenge,
        );

        let mut src_vec = base.clone();
        let mut dst_vec: Vec<PEF3> = vec![
            PEF3::new(
                PBF::from(0u64),
                PBF::from(0u64),
                PBF::from(0u64)
            );
            total
        ];
        fold_sub_polys_packed_ext3_par(
            &mut src_vec,
            &mut dst_vec,
            eval_len_packed,
            num_subs,
            challenge,
        );

        let new_count = num_subs / 2;
        let prefix = new_count * eval_len_packed;
        assert_eq!(src_vec.len(), prefix);
        assert_eq!(
            &serial_buf[..prefix],
            &src_vec[..],
            "fold_sub_polys_ext3 parity failed for num_subs={} eval_len_packed={}",
            num_subs,
            eval_len_packed,
        );
    }

    #[test]
    fn fold_sub_polys_ext3_par_matches_serial() {
        check_fold_sub_polys_ext3(32, 1 << 12, 0x7E7E);
        check_fold_sub_polys_ext3(8, 1 << 12, 0x7E7E);
        check_fold_sub_polys_ext3(4, 1 << 15, 0x7E7E);
        check_fold_sub_polys_ext3(2, 1 << 15, 0x7E7E);
        check_fold_sub_polys_ext3(2, 1 << 10, 0x7E7E);
    }

    // fold_multilinear_packed parity tests.
    fn check_fold_multilinear_ext3(logical_blocks: usize, seed: u64) {
        let mut rng = SmallRng::seed_from_u64(seed);
        let base = random_packed_ext3(&mut rng, logical_blocks);
        let challenge_c0 = MamaBearScalar::random(&mut rng).to_montgomery().0;
        let challenge_c1 = MamaBearScalar::random(&mut rng).to_montgomery().0;
        let challenge_c2 = MamaBearScalar::random(&mut rng).to_montgomery().0;
        let challenge = PEF3::new(
            PBF::from(challenge_c0),
            PBF::from(challenge_c1),
            PBF::from(challenge_c2),
        );

        let mut serial_buf = base.clone();
        <SEF3 as DeepFoldExtField>::fold_multilinear_packed(&mut serial_buf, challenge);

        let new_len = logical_blocks / 2;
        let mut src_vec = base.clone();
        let mut dst_vec: Vec<PEF3> = vec![
            PEF3::new(PBF::from(0u64), PBF::from(0u64), PBF::from(0u64));
            new_len
        ];
        fold_multilinear_packed_ext3_par(&mut src_vec, &mut dst_vec, challenge);

        assert_eq!(src_vec.len(), new_len);
        assert_eq!(
            &serial_buf[..], &src_vec[..],
            "fold_multilinear_packed_ext3 parity failed for logical_blocks={}",
            logical_blocks,
        );
    }

    #[test]
    fn fold_multilinear_packed_ext3_par_matches_serial() {
        check_fold_multilinear_ext3(1 << 14, 0xB1);
        check_fold_multilinear_ext3(1 << 12, 0xB2);
        check_fold_multilinear_ext3(32, 0xB3);
    }

    // eval_multilinear_packed parity tests.
    #[test]
    fn eval_multilinear_packed_ext3_par_matches_serial() {
        let mut rng = SmallRng::seed_from_u64(0xD1D1);
        let logical_blocks = 1usize << 14;
        let packed = random_packed_ext3(&mut rng, logical_blocks);
        let nv = (logical_blocks * 8).ilog2() as usize;
        let point: Vec<SEF3> = (0..nv)
            .map(|_| SEF3::random(&mut rng).to_montgomery())
            .collect();

        let serial_out = <SEF3 as DeepFoldExtField>::eval_multilinear_packed(&packed, &point);
        let par_out = eval_multilinear_packed_ext3_par(&packed, &point);
        assert_eq!(serial_out, par_out, "eval_multilinear_packed_ext3 parity failed");
    }

    // FriFoldResult::from_packed_values_par parity test.
    #[test]
    fn from_packed_values_par_matches_serial_ext3() {
        let mut rng = SmallRng::seed_from_u64(0xF1F1);
        let logical_blocks = 1usize << 14;
        let packed = random_packed_ext3(&mut rng, logical_blocks);

        let serial = FriFoldResult::<SEF3>::from_packed_values(packed.clone());
        let parallel = FriFoldResult::<SEF3>::from_packed_values_par(packed);

        assert_eq!(serial.value_packed, parallel.value_packed);
        assert_eq!(serial.value_query_mont, parallel.value_query_mont);
        assert_eq!(
            serial.merkle_tree.commit(),
            parallel.merkle_tree.commit(),
            "merkle root mismatch",
        );
    }

    // evaluate_next_domain* (within-round FRI fold) parity tests.
    // Exercises both the parallel path (packed_pairs >= PAR_FRI_FOLD_MIN_PAIRS)
    // and the serial fallback (below the threshold), for first-round and
    // rounds>=1.
    fn check_evaluate_next_domain_ext3(log_order: u32, first_round: bool, seed: u64) {
        let fft = MamaBearFFT::new(log_order);
        let last_len = fft.size() / 8;
        let mut rng = SmallRng::seed_from_u64(seed);
        let last_packed = random_packed_ext3(&mut rng, last_len);
        let challenge = PEF3::new(
            PBF::from(MamaBearScalar::random(&mut rng).to_montgomery().0),
            PBF::from(MamaBearScalar::random(&mut rng).to_montgomery().0),
            PBF::from(MamaBearScalar::random(&mut rng).to_montgomery().0),
        );

        let (serial, parallel) = if first_round {
            (
                <SEF3 as DeepFoldExtField>::evaluate_next_domain_first_round_packed(
                    &last_packed,
                    &fft,
                    challenge,
                ),
                evaluate_next_domain_first_round_packed_ext3_par(&last_packed, &fft, challenge),
            )
        } else {
            (
                <SEF3 as DeepFoldExtField>::evaluate_next_domain_packed(
                    &last_packed,
                    &fft,
                    challenge,
                ),
                evaluate_next_domain_packed_ext3_par(&last_packed, &fft, challenge),
            )
        };
        assert_eq!(serial.len(), parallel.len());
        assert_eq!(
            serial, parallel,
            "evaluate_next_domain ext3 parity failed log_order={} first_round={}",
            log_order, first_round
        );
    }

    #[test]
    fn evaluate_next_domain_ext3_par_matches_serial() {
        // log_order=20 -> packed_pairs = 2^16 = 65536 (parallel path).
        check_evaluate_next_domain_ext3(20, true, 0x7777);
        check_evaluate_next_domain_ext3(20, false, 0x8888);
        check_evaluate_next_domain_ext3(18, true, 0x9999);
        check_evaluate_next_domain_ext3(18, false, 0xAAAA);
        check_evaluate_next_domain_ext3(12, true, 0xBBBB);
        check_evaluate_next_domain_ext3(12, false, 0xCCCC);
    }

    // Base-prover full-proof parity: open_par must produce a byte-identical
    // transcript to open across the entire FRI fold + query . The FRI
    // fold output feeds the Merkle leaves, so any value difference would
    // change a root and be caught here. nv chosen large enough that several
    // FRI fold rounds run (the within-round par fold is exercised at round 0).
    fn check_open_par_matches_serial<F: DeepFoldExtField>(nv: usize, seed: u64) {
        use util::fiat_shamir::Transcript;
        let n = 1usize << nv;
        let polys: Vec<Vec<MamaBearScalar>> =
            (0..3).map(|i| random_poly(n, seed + i as u64)).collect();
        let poly_refs: Vec<&[MamaBearScalar]> = polys.iter().map(|p| p.as_slice()).collect();
        let pp = DeepFoldMamaBearParam::new_default(nv, 3, 34);

        let prover = DeepFoldMamaBearProver::<F>::new(&pp, &poly_refs);
        let root = prover.commit();

        let mut rng = SmallRng::seed_from_u64(seed ^ 0xABCD_EF01);
        let point: Vec<F> = (0..pp.variable_num)
            .map(|_| F::from(MamaBearScalar::random(&mut rng)).to_mont())
            .collect();

        let mut t_serial = Transcript::new();
        t_serial.append_u8_slice(&root.0, HASH_SIZE);
        DeepFoldMamaBearProver::<F>::open(&pp, &[&prover], point.clone(), &mut t_serial);

        let mut t_par = Transcript::new();
        t_par.append_u8_slice(&root.0, HASH_SIZE);
        DeepFoldMamaBearProver::<F>::open_par(&pp, &[&prover], point.clone(), &mut t_par);

        assert_eq!(
            t_serial.proof.bytes, t_par.proof.bytes,
            "open_par transcript must match serial open at nv={}",
            nv
        );
    }

    #[test]
    fn open_par_matches_serial_ext3_nv14() {
        check_open_par_matches_serial::<SEF3>(14, 0xBEEF_0003);
    }

    #[test]
    fn open_par_matches_serial_ext3_nv18() {
        check_open_par_matches_serial::<SEF3>(18, 0xBEEF_0004);
    }

    // =========================================================================
    // Microbenchmarks — gated on `#[ignore]`, run with
    //   RUSTFLAGS="-C target-cpu=native" cargo test -p poly_commit --release \
    //       microbench -- --ignored --nocapture --test-threads=1
    // These are intended for threshold tuning and parallelism scaling studies,
    // not as part of the normal test suite.
    // =========================================================================

    use std::time::Instant;

    fn measure_ns<F: FnMut()>(mut f: F, reps: usize) -> u128 {
        // 2 warmup reps + N measured reps, return the minimum (tail-insensitive).
        f();
        f();
        let mut best = u128::MAX;
        for _ in 0..reps {
            let t0 = Instant::now();
            f();
            let ns = t0.elapsed().as_nanos();
            if ns < best {
                best = ns;
            }
        }
        best
    }

    /// Profiling: open substage breakdown, serial open (`open_profiled`,
    /// whose `fri_fold_us` is the SERIAL FRI-fold cost = the "before" state)
    /// vs parallel open (`open_par_profiled`, whose `fri_fold_us` is the
    /// parallelized cost = the "after" state). The FRI-fold work is identical
    /// in both, so the `fri_fold_us` delta isolates the Part-A speedup.
    #[test]
    #[ignore]
    fn measure_open_substages_ext3_nv20() {
        use crate::deepfold_mamabear::OpenTimings;
        use util::fiat_shamir::Transcript;

        let nv = 20;
        let n = 1usize << nv;
        let n_polys = 7; // HyperPlonk K = 7 (4 preprocessed + 3 witness).
        let polys: Vec<Vec<MamaBearScalar>> =
            (0..n_polys).map(|i| random_poly(n, 0xF00D + i as u64)).collect();
        let poly_refs: Vec<&[MamaBearScalar]> = polys.iter().map(|p| p.as_slice()).collect();
        let pp = DeepFoldMamaBearParam::new_default(nv, 3, 34);
        let prover = new_par::<SEF3>(&pp, &poly_refs);
        let root = prover.commit();

        let mut rng = SmallRng::seed_from_u64(0xC0FFEE);
        let point: Vec<SEF3> = (0..pp.variable_num)
            .map(|_| SEF3::from(MamaBearScalar::random(&mut rng)).to_mont())
            .collect();

        let reps = 5u128;
        let mut ser = OpenTimings::default();
        let mut par = OpenTimings::default();
        // Warmup.
        {
            let mut t = Transcript::new();
            t.append_u8_slice(&root.0, HASH_SIZE);
            let mut warm = OpenTimings::default();
            DeepFoldMamaBearProver::<SEF3>::open_par_profiled(
                &pp, &[&prover], point.clone(), &mut t, &mut warm,
            );
        }
        for _ in 0..reps {
            let mut t = Transcript::new();
            t.append_u8_slice(&root.0, HASH_SIZE);
            DeepFoldMamaBearProver::<SEF3>::open_profiled(
                &pp, &[&prover], point.clone(), &mut t, &mut ser,
            );
        }
        for _ in 0..reps {
            let mut t = Transcript::new();
            t.append_u8_slice(&root.0, HASH_SIZE);
            DeepFoldMamaBearProver::<SEF3>::open_par_profiled(
                &pp, &[&prover], point.clone(), &mut t, &mut par,
            );
        }
        let avg = |x: u128| x as f64 / reps as f64 / 1000.0;
        println!("\n=== open substages ext3 nv=20, {n_polys} polys (avg of {reps}, ms) ===");
        println!(
            "{:<18} {:>12} {:>12}",
            "substage", "serial", "par"
        );
        macro_rules! row {
            ($name:literal, $f:ident) => {
                println!("{:<18} {:>12.3} {:>12.3}", $name, avg(ser.$f), avg(par.$f));
            };
        }
        row!("combine_polys", combine_polys_us);
        row!("combine_subs", combine_subs_us);
        row!("mle_eval", mle_eval_us);
        row!("multilin_fold", multilin_fold_us);
        row!("split_fold", split_fold_us);
        row!("fri_fold", fri_fold_us);
        row!("fri_merkle", fri_merkle_us);
        row!("query_phase", query_phase_us);
        row!("TOTAL", total_us);
        println!(
            "fri_fold speedup = {:.2}x; fri_fold share of serial open = {:.1}%",
            ser.fri_fold_us as f64 / par.fri_fold_us.max(1) as f64,
            100.0 * ser.fri_fold_us as f64 / ser.total_us.max(1) as f64,
        );
    }
}
