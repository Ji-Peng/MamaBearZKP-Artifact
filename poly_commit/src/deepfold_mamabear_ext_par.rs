//! Parallel ext-field DeepFold commit (PEF2 / PEF3).
//!
//! Companion to `deepfold_mamabear_ext.rs` (serial). Provides a parallel
//! `DeepFoldMamaBearProverExt::new_par` that produces a bit-identical commit
//! to the serial path.
//!
//! Strategy (mirrors base `deepfold_mamabear_par::new_par`):
//!
//! 1. Build a flat task list of `(poly_idx, sub_k, write_offset)` tuples,
//!    one per sub-FFT.
//! 2. Allocate the output buffer with `Vec::with_capacity + set_len` (uninit)
//!    so we get the calloc-mmap fast path (no zero-fill, no `IsZero` slow
//!    path that would hit a `Vec<E::Packed>` newtype).
//! 3. Run all tasks via `par_iter().for_each_init` with a per-thread scratch
//!    `Vec<E::Packed>`. Each task strided-gathers its sub-poly into scratch
//!    and runs `E::fft_into_packed` directly into its disjoint output slot.
//! 4. Parallel round-0 leaf hashing.
//! 5. Parallel Merkle tree build.
//!
//! No new FFT kernel: a single sub-FFT is the same 8-lane SIMD `fft_into_packed`
//! used by serial. Parallelism is only at the (poly_idx, sub_k) task level.

use std::marker::PhantomData;
use std::sync::Arc;

use rayon::prelude::*;
use util::blake3_batch;
use util::merkle_tree_mamabear::{HASH_SIZE, MerkleTreeProverMB};

use arithmetic::fft_mamabear::MamaBearFFT;
// Re-export the subset of types we need from the serial module.
use crate::deepfold_mamabear_ext::{DeepFoldElement, DeepFoldMamaBearProverExt, InterpolateValueMBExt};
use crate::deepfold_mamabear::DeepFoldMamaBearParam;

/// `*mut T` shipped across thread boundaries as `usize`. Caller must ensure
/// every parallel task writes to a disjoint slice of the underlying buffer.
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

/// Read the logical scalar at `logical_idx` from a packed buffer. Same helper
/// as serial; duplicated here to keep this file self-contained without
/// importing private symbols from the serial module.
#[inline(always)]
fn read_logical_at<E: DeepFoldElement>(values_packed: &[E::Packed], logical_idx: usize) -> E {
    E::unpack_lane(values_packed[logical_idx >> 3], logical_idx & 7)
}

const ROUND0_LEAF_HASH_TARGET_BYTES: usize = 32 * 1024;

#[inline(always)]
fn round0_leaf_hash_batch_size_packed(leaf_bytes: usize, pairs_per_block: usize) -> usize {
    let target = (ROUND0_LEAF_HASH_TARGET_BYTES / leaf_bytes).max(1);
    let aligned = (target / pairs_per_block) * pairs_per_block;
    aligned.max(pairs_per_block)
}

/// Hash one batch of leaves. Mirrors serial `round0_leaf_hashes_from_pair_major_packed`'s
/// inner-batch loop. Pure function; no shared mutation outside `out_hashes`.
#[allow(clippy::too_many_arguments)]
fn hash_one_batch_packed<E: DeepFoldElement>(
    values_packed: &[E::Packed],
    segment_count: usize,
    eval_len: usize,
    pairs_per_block: usize,
    leaf_bytes: usize,
    batch_start: usize,
    active_leaves: usize,
    batch_leaf_bytes: &mut [u8],
    out_hashes: &mut [[u8; HASH_SIZE]],
) {
    let pair_bytes = 2 * E::SIZE;

    for segment in 0..segment_count {
        let segment_logical_base = segment * eval_len;
        let segment_offset_in_leaf = segment * pair_bytes;

        for block_start in (0..active_leaves).step_by(pairs_per_block) {
            let src_leaf = batch_start + block_start;
            let src_base_logical = segment_logical_base + src_leaf * 2;

            for lane in 0..pairs_per_block {
                let x = read_logical_at::<E>(values_packed, src_base_logical + lane);
                let nx = read_logical_at::<E>(
                    values_packed,
                    src_base_logical + lane + pairs_per_block,
                );
                let dst_leaf_idx = block_start + lane;
                let dst_offset = dst_leaf_idx * leaf_bytes + segment_offset_in_leaf;
                E::write_pair_bytes(
                    &mut batch_leaf_bytes[dst_offset..dst_offset + pair_bytes],
                    x,
                    nx,
                );
            }
        }
    }

    let active_bytes = active_leaves * leaf_bytes;
    blake3_batch::hash_leaves_batch_flat(
        &batch_leaf_bytes[..active_bytes],
        active_leaves,
        leaf_bytes,
        &mut out_hashes[..active_leaves],
    );
}

/// Parallel round-0 leaf hashing for ext-field commits. Bit-identical to the
/// serial `round0_leaf_hashes_from_pair_major_packed`. Each batch hashes a
/// disjoint range of leaves; rayon only kicks in when there are at least two
/// batches (otherwise the serial path avoids thread overhead).
fn round0_leaf_hashes_par_packed<E: DeepFoldElement>(
    values_packed: &[E::Packed],
    leaf_size: usize,
) -> Vec<[u8; HASH_SIZE]> {
    assert_eq!(leaf_size % 2, 0, "round0 leaf size must contain x/nx pairs");
    let segment_count = leaf_size / 2;
    assert!(segment_count > 0);
    let total_logical = values_packed.len() * 8;
    assert_eq!(total_logical % segment_count, 0);

    let eval_len = total_logical / segment_count;
    assert_eq!(eval_len % 2, 0);
    let leaf_count = eval_len / 2;
    assert!(leaf_count.is_power_of_two());

    let pairs_per_block = MamaBearFFT::pair_slots_per_block_for_pair_count(leaf_count);
    let leaf_bytes = leaf_size * E::SIZE;
    let batch_leaf_count = round0_leaf_hash_batch_size_packed(leaf_bytes, pairs_per_block);

    let batch_ranges: Vec<(usize, usize)> = (0..leaf_count)
        .step_by(batch_leaf_count)
        .map(|batch_start| {
            let active_leaves = (leaf_count - batch_start).min(batch_leaf_count);
            (batch_start, active_leaves)
        })
        .collect();

    let mut leaf_hashes = vec![[0u8; HASH_SIZE]; leaf_count];

    // Single batch: serial fast path -- saves rayon scheduling overhead.
    if batch_ranges.len() <= 1 {
        for &(batch_start, active_leaves) in &batch_ranges {
            let mut batch_leaf_bytes = vec![0u8; active_leaves * leaf_bytes];
            hash_one_batch_packed::<E>(
                values_packed,
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

    // Multi-batch: each batch writes a disjoint range. Use raw mut pointer
    // shipped via SendAddr so rayon's `par_iter().for_each` can dispatch.
    let leaf_hashes_addr = SendAddr::from_ptr(leaf_hashes.as_mut_ptr());

    batch_ranges.par_iter().for_each(|&(batch_start, active_leaves)| {
        let mut batch_leaf_bytes = vec![0u8; active_leaves * leaf_bytes];
        let out: &mut [[u8; HASH_SIZE]] = unsafe {
            std::slice::from_raw_parts_mut(
                leaf_hashes_addr.as_ptr::<[u8; HASH_SIZE]>().add(batch_start),
                active_leaves,
            )
        };
        hash_one_batch_packed::<E>(
            values_packed,
            segment_count,
            eval_len,
            pairs_per_block,
            leaf_bytes,
            batch_start,
            active_leaves,
            &mut batch_leaf_bytes,
            out,
        );
    });

    leaf_hashes
}

// ---------------------------------------------------------------------------
// Public parallel constructor on `DeepFoldMamaBearProverExt`.
//
// Bit-identical output to `DeepFoldMamaBearProverExt::new`. Parallelism scope:
//   - Sub-FFT tasks (poly_idx, sub_k) flat-tiled across rayon workers.
//   - Round-0 leaf hashing: per-batch parallel.
//   - Merkle tree build: `MerkleTreeProverMB::from_leaf_hashes_par`.
// ---------------------------------------------------------------------------

impl<F, E> DeepFoldMamaBearProverExt<F, E>
where
    E: DeepFoldElement,
{
    /// Parallel commit, **normal-form (non-Mont) input**. Same semantics as
    /// `new`, just rayon-parallelised. **Mont-form input?** Use
    /// [`new_par_from_mont`](Self::new_par_from_mont) instead.
    pub fn new_par(
        pp: &DeepFoldMamaBearParam,
        polys_packed: &[Arc<Vec<E::Packed>>],
    ) -> DeepFoldMamaBearProverExt<F, E> {
        Self::new_par_internal::<false>(pp, polys_packed)
    }

    /// Parallel commit, **Montgomery-form input**. Same as `new_par` but skips
    /// the per-element `to_montgomery()` step at the FFT entry. Use this when
    /// feeding batch-inversion output directly
    /// into commit.
    pub fn new_par_from_mont(
        pp: &DeepFoldMamaBearParam,
        polys_packed_mont: &[Arc<Vec<E::Packed>>],
    ) -> DeepFoldMamaBearProverExt<F, E> {
        Self::new_par_internal::<true>(pp, polys_packed_mont)
    }

    fn new_par_internal<const SRC_IS_MONT: bool>(
        pp: &DeepFoldMamaBearParam,
        polys_packed: &[Arc<Vec<E::Packed>>],
    ) -> DeepFoldMamaBearProverExt<F, E> {
        let fft = &pp.fft_groups[0];
        let split_level = pp.split_level;
        let sub_count = 1usize << split_level;
        let eval_len = fft.size();
        let eval_len_packed = eval_len / 8;
        assert!(
            split_level >= 3,
            "DeepFoldMamaBearProverExt::new_par requires pp.split_level >= 3"
        );
        assert!(!polys_packed.is_empty(), "polys_packed must be non-empty");

        let k_polys = polys_packed.len();
        let poly_packed_len = polys_packed[0].len();
        for p in polys_packed.iter() {
            assert_eq!(p.len(), poly_packed_len, "all polys_packed must have equal length");
        }
        let raw_len = poly_packed_len * 8;
        assert_eq!(raw_len % sub_count, 0);
        let raw_len_per_sub = raw_len / sub_count;
        assert_eq!(raw_len_per_sub % 8, 0);
        let raw_len_per_sub_packed = raw_len_per_sub / 8;
        let sub_count_packed = sub_count / 8;

        let total_packed = k_polys * sub_count * eval_len_packed;

        // Allocate uninitialized: each task writes its `eval_len_packed`-sized
        // slot before any reader (leaf-hash) touches it. Avoids the
        // newtype-`IsZero`-slow-path that `vec![E::PACKED_ZERO_NORMAL; n]`
        // would hit for a user-defined newtype.
        let mut sub_evals_packed: Vec<E::Packed> = Vec::with_capacity(total_packed);
        unsafe {
            sub_evals_packed.set_len(total_packed);
        }

        // Build (poly_idx, sub_k, write_offset) task list.
        struct FftTask {
            poly_idx: usize,
            sub_k: usize,
            write_offset: usize,
        }

        let mut tasks: Vec<FftTask> = Vec::with_capacity(k_polys * sub_count);
        let mut write_offset = 0usize;
        for poly_idx in 0..k_polys {
            for k in 0..sub_count {
                tasks.push(FftTask {
                    poly_idx,
                    sub_k: k,
                    write_offset,
                });
                write_offset += eval_len_packed;
            }
        }

        // Disjoint slot writes: each task writes
        // sub_evals_packed[write_offset..write_offset + eval_len_packed],
        // and the offsets are unique (poly_idx, sub_k) -> distinct ranges.
        let sub_evals_addr = SendAddr::from_ptr(sub_evals_packed.as_mut_ptr());

        tasks.par_iter().for_each_init(
            || {
                // Per-worker scratch: enough for one strided-gather sub-poly.
                vec![E::PACKED_ZERO_NORMAL; raw_len_per_sub_packed]
            },
            |scratch, task| {
                let poly = &polys_packed[task.poly_idx];
                let k_pbase = task.sub_k / 8;
                let k_lane = task.sub_k % 8;

                // Strided gather (matches serial `gather_strided_lanes`).
                for j in 0..raw_len_per_sub_packed {
                    let block_base = k_pbase + j * 8 * sub_count_packed;
                    let lanes: [E; 8] = std::array::from_fn(|l| {
                        E::unpack_lane(poly[block_base + l * sub_count_packed], k_lane)
                    });
                    scratch[j] = E::pack_lanes(lanes);
                }

                // Disjoint write into the global output.
                let dest: &mut [E::Packed] = unsafe {
                    std::slice::from_raw_parts_mut(
                        sub_evals_addr.as_ptr::<E::Packed>().add(task.write_offset),
                        eval_len_packed,
                    )
                };
                if SRC_IS_MONT {
                    E::fft_into_packed_mont(fft, &scratch[..raw_len_per_sub_packed], dest);
                } else {
                    E::fft_into_packed(fft, &scratch[..raw_len_per_sub_packed], dest);
                }
            },
        );

        let leaf_size = 2 * sub_count * k_polys;
        let value_packed = Arc::new(sub_evals_packed);
        let leaf_hashes = round0_leaf_hashes_par_packed::<E>(&value_packed, leaf_size);
        let merkle_tree = MerkleTreeProverMB::from_leaf_hashes_par(&leaf_hashes);
        let interpolation = InterpolateValueMBExt::<E>::from_fft_pair_major_parts_packed_with_tree(
            Arc::clone(&value_packed),
            leaf_size,
            merkle_tree,
        );

        DeepFoldMamaBearProverExt {
            interpolation,
            sub_evals_mont_packed: value_packed,
            poly_packed: polys_packed.to_vec(),
            _phantom: PhantomData,
        }
    }
}

// ---------------------------------------------------------------------------
// Parallel ext combine kernels (PEF * PEF Karatsuba accumulate).
//
// Mirrors `combine_pef2_packed_mont` / `combine_pef3_packed_mont` in the
// serial module, but parallelises the per-output-block loop via
// `(0..len_packed).into_par_iter()`. Powers precomputation stays serial
// (small, O(n_polys) muls).
// ---------------------------------------------------------------------------

use arithmetic::field::mamabear::{
    MamaBearScalar, MamaBearScalarExt3, P, PackedMamaBearAVX512,
    PackedMamaBearAVX512Ext3,
};

#[inline]
pub(crate) fn combine_pef3_packed_mont_par<const SRC_IS_MONT: bool>(
    polys_packed: &[&[PackedMamaBearAVX512Ext3]],
    r_mont: MamaBearScalarExt3,
) -> Vec<PackedMamaBearAVX512Ext3> {
    let n_polys = polys_packed.len();
    assert!(n_polys >= 1, "combine_pef3_par: at least one poly required");
    let len_packed = polys_packed[0].len();
    debug_assert!(polys_packed.iter().all(|p| p.len() == len_packed));

    let one_mont = MamaBearScalarExt3::from(MamaBearScalar(1)).to_montgomery();
    let mut ascending: Vec<MamaBearScalarExt3> = Vec::with_capacity(n_polys);
    ascending.push(one_mont);
    for i in 1..n_polys {
        ascending.push(ascending[i - 1] * r_mont);
    }
    let mut powers_scalar: Vec<MamaBearScalarExt3> = ascending.into_iter().rev().collect();
    if !SRC_IS_MONT {
        for p in powers_scalar.iter_mut() {
            *p = p.to_montgomery();
        }
    }
    let powers_packed: Vec<PackedMamaBearAVX512Ext3> = powers_scalar
        .iter()
        .map(|p| {
            PackedMamaBearAVX512Ext3::new(
                PackedMamaBearAVX512::broadcast(p.c0.0),
                PackedMamaBearAVX512::broadcast(p.c1.0),
                PackedMamaBearAVX512::broadcast(p.c2.0),
            )
        })
        .collect();

    let zero_mont_pbf = PackedMamaBearAVX512::broadcast(P);
    let zero_mont_pef = PackedMamaBearAVX512Ext3::new(zero_mont_pbf, zero_mont_pbf, zero_mont_pbf);

    (0..len_packed)
        .into_par_iter()
        .map(|k| {
            crate::deepfold_mamabear_ext::accumulate_combine_block_4way::<
                PackedMamaBearAVX512Ext3,
            >(&powers_packed, polys_packed, k, zero_mont_pef)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Parallel ext open: parallel combines + parallel main fold loop + parallel
// grinding. Uses `open_main_fold_after_combine_par` from base (extracted
// alongside the serial `open_main_fold_after_combine` per HC-3 -- both
// extractions are pure refactors, base machine code unchanged).
// ---------------------------------------------------------------------------

use crate::deepfold_mamabear::{
    open_main_fold_after_combine_par, DeepFoldExtField, OpenTimings,
};
use util::fiat_shamir::Transcript;

impl<E> DeepFoldMamaBearProverExt<E, E>
where
    E: DeepFoldExtField + DeepFoldElement,
    // Same type-equality bound as the serial impl in deepfold_mamabear_ext.rs:
    // forces the two associated types to unify so the helper handoff is a
    // no-op type-system identity (no unsafe transmute).
    E: DeepFoldExtField<PackedExt = <E as DeepFoldElement>::Packed>,
{
    /// Parallel open. Identical transcript output to serial `open`.
    pub fn open_par(
        pp: &DeepFoldMamaBearParam,
        provers: &[&Self],
        point_mont: Vec<E>,
        transcript: &mut Transcript,
    ) {
        let mut timings = OpenTimings::default();
        Self::open_inner_par(pp, provers, point_mont, transcript, &mut timings, false);
    }

    fn open_inner_par(
        pp: &DeepFoldMamaBearParam,
        provers: &[&Self],
        point_mont: Vec<E>,
        transcript: &mut Transcript,
        timings: &mut OpenTimings,
        record: bool,
    ) {
        let split_level = pp.split_level;
        let sub_count = 1usize << split_level;
        let eval_len_packed = pp.fft_groups[0].size() / 8;

        let r_raw: E = transcript.challenge_f();
        let r_mont = r_raw.to_mont();

        // -- combine_polys (parallel) --
        let poly_slices: Vec<&[<E as DeepFoldElement>::Packed]> = provers
            .iter()
            .flat_map(|p| p.poly_packed.iter().map(|v| v.as_slice()))
            .collect();
        let poly_evals_packed: Vec<<E as DeepFoldElement>::Packed> =
            <E as DeepFoldElement>::combine_packed_mont_par(&poly_slices, r_mont, false);

        // -- combine_subs (parallel) --
        let chunk_len_packed = sub_count * eval_len_packed;
        let sub_slices: Vec<&[<E as DeepFoldElement>::Packed]> = provers
            .iter()
            .flat_map(|prover| {
                (0..prover.poly_packed.len()).map(move |poly_idx| {
                    let start = poly_idx * chunk_len_packed;
                    &prover.sub_evals_mont_packed[start..start + chunk_len_packed]
                })
            })
            .collect();
        let sub_evals_packed: Vec<<E as DeepFoldElement>::Packed> =
            <E as DeepFoldElement>::combine_packed_mont_par(&sub_slices, r_mont, true);

        // -- main fold loop (PARALLEL; uses base par helper) --
        // Type-equality bound on the impl unifies `<E as DeepFoldElement>::Packed`
        // with `<E as DeepFoldExtField>::PackedExt`, so the helper handoff is
        // a direct identity (no transmute).
        let (mut leaf_indices, fri_results) = open_main_fold_after_combine_par::<E>(
            pp,
            poly_evals_packed,
            sub_evals_packed,
            &point_mont,
            transcript,
            timings,
            record,
        );

        // -- query  (serial; same as serial open) --
        let query = provers
            .iter()
            .map(|j| j.interpolation.query(&leaf_indices))
            .collect::<Vec<_>>();
        for q in query {
            transcript.append_u8_slice(&q.0, q.0.len());
            for j in q.1 {
                transcript.append_f(j);
            }
        }
        for k in 0..fri_results.len() {
            leaf_indices = leaf_indices.iter().map(|v| *v >> 1).collect();
            leaf_indices.sort();
            leaf_indices.dedup();
            let query = fri_results[k].query(&leaf_indices);
            transcript.append_u8_slice(&query.0, query.0.len());
            for j in query.1 {
                transcript.append_f(j);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests: parallel result must be byte-identical to serial.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use arithmetic::field::mamabear::{
        MamaBearScalarExt3, PackedMamaBearAVX512,
        PackedMamaBearAVX512Ext3,
    };
    use rand::rngs::SmallRng;
    use rand::{RngCore, SeedableRng};

    fn make_rng(seed: u64) -> SmallRng {
        SmallRng::seed_from_u64(seed)
    }

    /// Run on a 16 MB test thread AND in a rayon pool whose worker threads
    /// also have 16 MB stacks. The serial wrapper only sized the test thread;
    /// rayon worker threads default to 2 MB which overflows on debug-mode
    /// PEF3 fused FFT (see serial test module rationale). We need both.
    fn run_with_large_stack<F: FnOnce() + Send + 'static>(f: F) {
        std::thread::Builder::new()
            .stack_size(16 * 1024 * 1024)
            .spawn(move || {
                let pool = rayon::ThreadPoolBuilder::new()
                    .stack_size(16 * 1024 * 1024)
                    .build()
                    .expect("rayon large-stack pool");
                pool.install(f);
            })
            .expect("spawn large-stack test thread")
            .join()
            .expect("test thread panicked");
    }

    fn random_pef3_vec(n_packed: usize, rng: &mut impl RngCore) -> Vec<PackedMamaBearAVX512Ext3> {
        let p = arithmetic::field::mamabear::P;
        let mut out = Vec::with_capacity(n_packed);
        for _ in 0..n_packed {
            let mut a0 = [0u64; 8];
            let mut a1 = [0u64; 8];
            let mut a2 = [0u64; 8];
            for i in 0..8 {
                a0[i] = rng.next_u64() % p;
                a1[i] = rng.next_u64() % p;
                a2[i] = rng.next_u64() % p;
            }
            out.push(PackedMamaBearAVX512Ext3::new(
                PackedMamaBearAVX512::from_array(a0),
                PackedMamaBearAVX512::from_array(a1),
                PackedMamaBearAVX512::from_array(a2),
            ));
        }
        out
    }

    fn test_param(variable_num: usize) -> DeepFoldMamaBearParam {
        DeepFoldMamaBearParam::new_default(variable_num, 3, 8)
    }

    /// Parallel commit produces the same Merkle root as serial.
    #[test]
    fn ext_prover_new_par_pef3_matches_serial() {
        run_with_large_stack(|| {
        let mut rng = make_rng(0x8888);
        let pp = test_param(10);
        let n = 1usize << pp.variable_num;
        let n_packed = n / 8;
        let polys = vec![
            Arc::new(random_pef3_vec(n_packed, &mut rng)),
            Arc::new(random_pef3_vec(n_packed, &mut rng)),
        ];

        let serial =
            DeepFoldMamaBearProverExt::<MamaBearScalarExt3, MamaBearScalarExt3>::new(&pp, &polys);
        let parallel =
            DeepFoldMamaBearProverExt::<MamaBearScalarExt3, MamaBearScalarExt3>::new_par(
                &pp, &polys,
            );

        assert_eq!(serial.commit().0, parallel.commit().0);
        assert_eq!(serial.sub_evals_mont_packed.len(), parallel.sub_evals_mont_packed.len());
        for i in 0..serial.sub_evals_mont_packed.len() {
            let s = serial.sub_evals_mont_packed[i];
            let p = parallel.sub_evals_mont_packed[i];
            for lane in 0..8 {
                let s_e = MamaBearScalarExt3::unpack_lane(s, lane);
                let p_e = MamaBearScalarExt3::unpack_lane(p, lane);
                assert_eq!(s_e.c0.0, p_e.c0.0);
                assert_eq!(s_e.c1.0, p_e.c1.0);
                assert_eq!(s_e.c2.0, p_e.c2.0);
            }
        }
        });
    }

}
