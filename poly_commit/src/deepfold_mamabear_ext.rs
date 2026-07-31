//! Extension-field DeepFold prover (PEF3).
//!
//! Companion to `deepfold_mamabear.rs`. Provides an ext-field commit path
//! whose input lives in `Arc<Vec<E::Packed>>` (PEF3) end-to-end --
//! batch_invert -> commit -> open -- with no logical-scalar bouncing
//! Re-uses the base ext-FFT kernel from
//! `arithmetic::fft_mamabear_ext` and the existing `MerkleTreeProverMB`.
//!
//! This module currently provides `DeepFoldElement`, `InterpolateValueMBExt`,
//! leaf hashing, `DeepFoldMamaBearProverExt::new`, and `commit`. Extension
//! open/verify and mixed-view batch opening are implemented separately.

use std::marker::PhantomData;
use std::sync::Arc;

use arithmetic::fft_mamabear::MamaBearFFT;
use arithmetic::fft_mamabear_ext::{
    fft_into_packed_pef3, fft_into_packed_pef3_mont,
};
use arithmetic::field::{Field, as_bytes_vec};
use arithmetic::field::mamabear::{
    LazyReduction, MamaBearScalar, MamaBearScalarExt3, P,
    PackedMamaBearAVX512, PackedMamaBearAVX512Ext3,
};
use util::blake3_batch;
use util::fiat_shamir::{Proof, Transcript};
use util::merkle_tree_mamabear::{HASH_SIZE, MerkleTreeProverMB, MerkleTreeVerifierMB};

use crate::deepfold::MerkleRoot;
use crate::deepfold_mamabear::{
    open_main_fold_after_combine, DeepFoldExtField, DeepFoldMamaBearParam, OpenTimings,
};

// ---------------------------------------------------------------------------
// DeepFoldElement: trait abstracting the per-leaf serialisation, packed-block
// view, and FFT entry point for base / ext element types.
//
// A scope: only the extension-field type (Ext3) needs an impl.
// A `MamaBearScalar` impl will be added in when mixed-view batch open
// is wired in -- at that point the existing base prover gets an adapter via
// the `DeepFoldCommitView` trait, with `MamaBearScalar` filling the same
// `DeepFoldElement` slot as the ext element.
// ---------------------------------------------------------------------------

pub trait DeepFoldElement: Copy + Send + Sync + 'static {
    /// Byte width of one element when serialised into a Merkle leaf.
    /// Ext3 = 21, base = 7.
    const SIZE: usize;

    /// Number of base-field limbs. 1 = base, 3 = Ext3.
    const BASE_LIMBS: usize;

    /// 8-lane packed SIMD form. `repr(C)` lane layout is required so
    /// `unpack_lane` / `pack_lanes` can address the limbs directly.
    type Packed: Copy + Send + Sync + Default + 'static;

    /// Normal-form (NOT Montgomery) zero used to zero-pad a fresh
    /// `Vec<Self::Packed>` buffer before the FFT lifts it into Mont.
    const PACKED_ZERO_NORMAL: Self::Packed;

    /// Extract the `lane`-th logical element from a packed block. `lane` in
    /// `0..8`. Reads via the underlying `array: [u64; 8]` union view -- safe
    /// because each base limb is `#[repr(transparent)]` over `u64`.
    fn unpack_lane(packed: Self::Packed, lane: usize) -> Self;

    /// Pack 8 logical elements into one packed block (lane `i` <- `lanes[i]`).
    fn pack_lanes(lanes: [Self; 8]) -> Self::Packed;

    /// Serialise a (`x`, `nx`) pair into `dst[..2 * SIZE]` little-endian.
    /// Used by the round-0 leaf hash.
    fn write_pair_bytes(dst: &mut [u8], x: Self, nx: Self);

    /// Run a forward FFT on a packed buffer. Mirrors the base
    /// `MamaBearFFT::fft_into` signature but in packed form. **Input is
    /// normal-form (non-Montgomery)**; the FFT internally lifts it to Mont
    /// as the first step. Use `fft_into_packed_mont` if the caller already
    /// has Mont-form input (e.g., output of `batch_invert_pef*_in_place`).
    fn fft_into_packed(fft: &MamaBearFFT, raw: &[Self::Packed], buf: &mut [Self::Packed]);

    /// Same as `fft_into_packed`, but the input is **already in Montgomery
    /// form**. Skips the per-element `to_montgomery` step (compile-time
    /// elided via the const-generic `SRC_IS_MONT` flag in the underlying
    /// FFT kernel). Use this entry to avoid a redundant `from_mont -> to_mont`
    /// round-trip when feeding the output of `batch_invert_pef*_in_place`
    /// (or any other Mont-form producer) directly into commit.
    fn fft_into_packed_mont(
        fft: &MamaBearFFT,
        raw_mont: &[Self::Packed],
        buf: &mut [Self::Packed],
    );

    /// Compute `out[k] = sum_i powers_mont[i] * polys_packed[i][k]` where
    /// `powers_mont[i] = r_mont^(N-1-i)` (descending Horner). All output
    /// blocks are in Montgomery form, each component reduced to
    /// `[0, 2.0001P)` (i.e. the output range of `reduce_fast`). The flag
    /// `src_is_mont` controls power preconditioning:
    ///
    /// - `false`: input polys are normal-form (non-Mont). Powers are pre-
    ///   doubled (`R^2 * r^k`) so `mont_mul(R^2 * r^k, p_normal) = mont(r^k * p)`.
    /// - `true`: input polys are Mont-form. Powers carry one R factor.
    ///
    /// Mirrors `combine_opt_ext3_inner` semantics on `deepfold_mamabear.rs`,
    /// but for *packed extension* inputs (PEF * PEF) instead of base * ext.
    fn combine_packed_mont(
        polys_packed: &[&[Self::Packed]],
        r_mont: Self,
        src_is_mont: bool,
    ) -> Vec<Self::Packed>;

    /// Parallel variant of `combine_packed_mont`. Default impl delegates to
    /// the serial path; concrete impls override with rayon-parallel kernels
    /// (see `deepfold_mamabear_ext_par`). Bit-identical output to serial.
    fn combine_packed_mont_par(
        polys_packed: &[&[Self::Packed]],
        r_mont: Self,
        src_is_mont: bool,
    ) -> Vec<Self::Packed> {
        Self::combine_packed_mont(polys_packed, r_mont, src_is_mont)
    }
}

// ---------------------------------------------------------------------------
// DeepFoldElement impl for Ext3.
// ---------------------------------------------------------------------------

impl DeepFoldElement for MamaBearScalarExt3 {
    const SIZE: usize = 21; // 3 * MamaBearScalar::SIZE
    const BASE_LIMBS: usize = 3;
    type Packed = PackedMamaBearAVX512Ext3;
    const PACKED_ZERO_NORMAL: Self::Packed = PackedMamaBearAVX512Ext3::ZERO_NORMAL;

    #[inline(always)]
    fn unpack_lane(packed: Self::Packed, lane: usize) -> Self {
        debug_assert!(lane < 8);
        let c0 = unsafe { packed.c0.array[lane] };
        let c1 = unsafe { packed.c1.array[lane] };
        let c2 = unsafe { packed.c2.array[lane] };
        Self {
            c0: MamaBearScalar(c0),
            c1: MamaBearScalar(c1),
            c2: MamaBearScalar(c2),
        }
    }

    #[inline(always)]
    fn pack_lanes(lanes: [Self; 8]) -> Self::Packed {
        let mut a0 = [0u64; 8];
        let mut a1 = [0u64; 8];
        let mut a2 = [0u64; 8];
        for i in 0..8 {
            a0[i] = lanes[i].c0.0;
            a1[i] = lanes[i].c1.0;
            a2[i] = lanes[i].c2.0;
        }
        PackedMamaBearAVX512Ext3::new(
            PackedMamaBearAVX512::from_array(a0),
            PackedMamaBearAVX512::from_array(a1),
            PackedMamaBearAVX512::from_array(a2),
        )
    }

    #[inline(always)]
    fn write_pair_bytes(dst: &mut [u8], x: Self, nx: Self) {
        const SCALAR_SIZE: usize = MamaBearScalar::SIZE;
        let x_c0 = x.c0.0.to_le_bytes();
        let x_c1 = x.c1.0.to_le_bytes();
        let x_c2 = x.c2.0.to_le_bytes();
        let nx_c0 = nx.c0.0.to_le_bytes();
        let nx_c1 = nx.c1.0.to_le_bytes();
        let nx_c2 = nx.c2.0.to_le_bytes();
        dst[0..SCALAR_SIZE].copy_from_slice(&x_c0[..SCALAR_SIZE]);
        dst[SCALAR_SIZE..2 * SCALAR_SIZE].copy_from_slice(&x_c1[..SCALAR_SIZE]);
        dst[2 * SCALAR_SIZE..3 * SCALAR_SIZE].copy_from_slice(&x_c2[..SCALAR_SIZE]);
        dst[3 * SCALAR_SIZE..4 * SCALAR_SIZE].copy_from_slice(&nx_c0[..SCALAR_SIZE]);
        dst[4 * SCALAR_SIZE..5 * SCALAR_SIZE].copy_from_slice(&nx_c1[..SCALAR_SIZE]);
        dst[5 * SCALAR_SIZE..6 * SCALAR_SIZE].copy_from_slice(&nx_c2[..SCALAR_SIZE]);
    }

    #[inline(always)]
    fn fft_into_packed(fft: &MamaBearFFT, raw: &[Self::Packed], buf: &mut [Self::Packed]) {
        fft_into_packed_pef3(fft, raw, buf);
    }

    #[inline(always)]
    fn fft_into_packed_mont(
        fft: &MamaBearFFT,
        raw_mont: &[Self::Packed],
        buf: &mut [Self::Packed],
    ) {
        fft_into_packed_pef3_mont(fft, raw_mont, buf);
    }

    #[inline]
    fn combine_packed_mont(
        polys_packed: &[&[Self::Packed]],
        r_mont: Self,
        src_is_mont: bool,
    ) -> Vec<Self::Packed> {
        if src_is_mont {
            combine_pef3_packed_mont::<true>(polys_packed, r_mont)
        } else {
            combine_pef3_packed_mont::<false>(polys_packed, r_mont)
        }
    }

    #[inline]
    fn combine_packed_mont_par(
        polys_packed: &[&[Self::Packed]],
        r_mont: Self,
        src_is_mont: bool,
    ) -> Vec<Self::Packed> {
        if src_is_mont {
            crate::deepfold_mamabear_ext_par::combine_pef3_packed_mont_par::<true>(
                polys_packed,
                r_mont,
            )
        } else {
            crate::deepfold_mamabear_ext_par::combine_pef3_packed_mont_par::<false>(
                polys_packed,
                r_mont,
            )
        }
    }
}

// ---------------------------------------------------------------------------
// Round-0 leaf hash for ext (mirrors `round0_leaf_hashes_from_pair_major_values`
// in `deepfold_mamabear.rs`, but reads from a packed buffer and serialises with
// `E::SIZE` byte-width pair writes).
// ---------------------------------------------------------------------------

const ROUND0_LEAF_HASH_TARGET_BYTES: usize = 32 * 1024;

#[inline(always)]
fn round0_leaf_hash_batch_size_packed(leaf_bytes: usize, pairs_per_block: usize) -> usize {
    let target = (ROUND0_LEAF_HASH_TARGET_BYTES / leaf_bytes).max(1);
    let aligned = (target / pairs_per_block) * pairs_per_block;
    aligned.max(pairs_per_block)
}

/// Read the logical scalar at `logical_idx` from a packed buffer.
#[inline(always)]
fn read_logical_at<E: DeepFoldElement>(values_packed: &[E::Packed], logical_idx: usize) -> E {
    E::unpack_lane(values_packed[logical_idx >> 3], logical_idx & 7)
}

fn round0_leaf_hashes_from_pair_major_packed<E: DeepFoldElement>(
    values_packed: &[E::Packed],
    leaf_size: usize,
) -> Vec<[u8; HASH_SIZE]> {
    assert_eq!(leaf_size % 2, 0, "round0 leaf size must contain x/nx pairs");
    let segment_count = leaf_size / 2;
    assert!(segment_count > 0, "round0 segment count must be non-zero");
    let total_logical = values_packed.len() * 8;
    assert_eq!(total_logical % segment_count, 0);

    let eval_len = total_logical / segment_count;
    assert_eq!(eval_len % 2, 0, "round0 eval_len must be even");

    let leaf_count = eval_len / 2;
    assert!(leaf_count.is_power_of_two(), "leaf_count must be power of 2");
    let pairs_per_block = MamaBearFFT::pair_slots_per_block_for_pair_count(leaf_count);

    let pair_bytes = 2 * E::SIZE;
    let leaf_bytes = leaf_size * E::SIZE;
    let batch_leaf_count = round0_leaf_hash_batch_size_packed(leaf_bytes, pairs_per_block);

    let mut batch_leaf_bytes = vec![0u8; batch_leaf_count * leaf_bytes];
    let mut leaf_hashes = vec![[0u8; HASH_SIZE]; leaf_count];

    for batch_start in (0..leaf_count).step_by(batch_leaf_count) {
        let active_leaves = (leaf_count - batch_start).min(batch_leaf_count);
        let active_bytes = active_leaves * leaf_bytes;

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

        blake3_batch::hash_leaves_batch_flat(
            &batch_leaf_bytes[..active_bytes],
            active_leaves,
            leaf_bytes,
            &mut leaf_hashes[batch_start..batch_start + active_leaves],
        );
    }

    leaf_hashes
}

// ---------------------------------------------------------------------------
// InterpolateValueMBExt: ext analogue of base `InterpolateValueMB`.
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct InterpolateValueMBExt<E: DeepFoldElement> {
    /// Pair-major blocked, Montgomery-form FFT evaluations across all
    /// (poly, sub-poly) segments. Length = `sub_count * eval_len_packed * K`.
    pub value_packed: Arc<Vec<E::Packed>>,
    /// Logical leaf size = `2 * sub_count * K`.
    leaf_size: usize,
    merkle_tree: MerkleTreeProverMB,
}

impl<E: DeepFoldElement> InterpolateValueMBExt<E> {
    pub fn from_fft_pair_major_parts_packed(
        value_packed: Arc<Vec<E::Packed>>,
        leaf_size: usize,
    ) -> Self {
        let leaf_hashes = round0_leaf_hashes_from_pair_major_packed::<E>(&value_packed, leaf_size);
        let merkle_tree = MerkleTreeProverMB::from_leaf_hashes(&leaf_hashes);
        Self {
            value_packed,
            leaf_size,
            merkle_tree,
        }
    }

    pub fn from_fft_pair_major_parts_packed_with_tree(
        value_packed: Arc<Vec<E::Packed>>,
        leaf_size: usize,
        merkle_tree: MerkleTreeProverMB,
    ) -> Self {
        Self {
            value_packed,
            leaf_size,
            merkle_tree,
        }
    }

    pub fn leave_num(&self) -> usize {
        self.merkle_tree.leave_num()
    }

    pub fn commit(&self) -> [u8; HASH_SIZE] {
        self.merkle_tree.commit()
    }

    /// Pair-major blocked query: returns `(merkle_proof_bytes, scalar_values_in_x_then_nx_order)`.
    /// The scalar order matches the base path so the verifier sees the same shape.
    pub fn query(&self, leaf_indices: &[usize]) -> (Vec<u8>, Vec<E>) {
        let len = self.merkle_tree.leave_num();
        assert_eq!(len * self.leaf_size, self.value_packed.len() * 8);
        let segment_len = len * 2; // logical
        let segment_count = self.leaf_size / 2;

        let mut proof_values: Vec<E> = Vec::with_capacity(leaf_indices.len() * self.leaf_size);
        for segment in 0..segment_count {
            let base = segment * segment_len;
            for &leaf_idx in leaf_indices {
                let (x_pos, _) = MamaBearFFT::pair_storage_positions_for_pair_count(leaf_idx, len);
                proof_values.push(read_logical_at::<E>(&self.value_packed, base + x_pos));
            }
            for &leaf_idx in leaf_indices {
                let (_, nx_pos) = MamaBearFFT::pair_storage_positions_for_pair_count(leaf_idx, len);
                proof_values.push(read_logical_at::<E>(&self.value_packed, base + nx_pos));
            }
        }

        let proof_bytes = self.merkle_tree.open(leaf_indices);
        (proof_bytes, proof_values)
    }
}

// ---------------------------------------------------------------------------
// DeepFoldMamaBearProverExt<F, E>: ext-field commit prover.
//
// Companion to `DeepFoldMamaBearProver<F>`. Holds the same kind of
// `(interpolation, sub_evals_mont_packed, polys_packed)` triple but everywhere
// in PEF (E::Packed) form.
//
// A scope: `new` + `commit`. `open` is added in B.
// ---------------------------------------------------------------------------

pub struct DeepFoldMamaBearProverExt<F, E>
where
    E: DeepFoldElement,
{
    pub interpolation: InterpolateValueMBExt<E>,
    /// Alias of `interpolation.value_packed` -- shared via `Arc::clone` for
    /// O(1) cloning. The open  reads this for combine_subs.
    pub(crate) sub_evals_mont_packed: Arc<Vec<E::Packed>>,
    /// Original input polys (kept for the `combine_polys` step in open).
    /// Each entry = one whole poly's packed coefficients.
    pub(crate) poly_packed: Vec<Arc<Vec<E::Packed>>>,
    pub(crate) _phantom: PhantomData<F>,
}

impl<F, E> DeepFoldMamaBearProverExt<F, E>
where
    E: DeepFoldElement,
{
    /// Build the ext prover from a slice of packed polys, **normal-form
    /// (non-Montgomery) input**. Each `polys_packed[i]` is a single polynomial
    /// of logical length `polys_packed[i].len * 8`. All polys must have the
    /// same length.
    ///
    /// Layout of the produced `sub_evals_mont_packed`:
    /// ```text
    /// for poly_idx in 0..K:
    ///     for k in 0..sub_count:
    ///         sub_evals_mont_packed[poly_idx*sub_count*eval_len_packed + k*eval_len_packed
    ///                                .. + eval_len_packed]
    ///             = FFT(sub_poly_k of poly[poly_idx])
    /// ```
    /// where `sub_poly_k of poly[i]` is `poly[i][k + j*sub_count]` for j in 0..raw_len_per_sub.
    /// Pair-major blocked Montgomery, identical layout to the base prover.
    ///
    /// **Mont-form input?** Use [`new_from_mont`](Self::new_from_mont) instead
    /// to skip the redundant `to_montgomery` step.
    pub fn new(pp: &DeepFoldMamaBearParam, polys_packed: &[Arc<Vec<E::Packed>>]) -> Self {
        Self::new_internal::<false>(pp, polys_packed)
    }

    /// Same as [`new`](Self::new), but the input polys are **already in
    /// Montgomery form** (e.g., output of a batch-inversion kernel).
    /// Skips the per-element `to_montgomery` step at the FFT entry,
    /// compile-time eliminated via const-generic. Saves one `to_mont` sweep
    /// per polynomial vs. `caller doing from_mont then new`.
    ///
    /// Output layout is identical to `new` (pair-major blocked Mont).
    pub fn new_from_mont(
        pp: &DeepFoldMamaBearParam,
        polys_packed_mont: &[Arc<Vec<E::Packed>>],
    ) -> Self {
        Self::new_internal::<true>(pp, polys_packed_mont)
    }

    fn new_internal<const SRC_IS_MONT: bool>(
        pp: &DeepFoldMamaBearParam,
        polys_packed: &[Arc<Vec<E::Packed>>],
    ) -> Self {
        let fft = &pp.fft_groups[0];
        let split_level = pp.split_level;
        let sub_count = 1usize << split_level;
        let eval_len = fft.size();
        let eval_len_packed = eval_len / 8;
        assert!(
            split_level >= 3,
            "DeepFoldMamaBearProverExt requires pp.split_level >= 3 (sub_count multiple of 8)"
        );
        assert!(!polys_packed.is_empty(), "polys_packed must be non-empty");

        let k_polys = polys_packed.len();
        let poly_packed_len = polys_packed[0].len();
        for p in polys_packed.iter() {
            assert_eq!(
                p.len(),
                poly_packed_len,
                "all polys_packed entries must have equal length"
            );
        }
        let raw_len = poly_packed_len * 8; // logical
        assert_eq!(
            raw_len % sub_count,
            0,
            "raw_len must be divisible by sub_count"
        );
        let raw_len_per_sub = raw_len / sub_count; // logical per sub-poly
        assert_eq!(
            raw_len_per_sub % 8,
            0,
            "raw_len_per_sub must be divisible by 8"
        );
        let raw_len_per_sub_packed = raw_len_per_sub / 8;
        let sub_count_packed = sub_count / 8;

        let total_packed = k_polys * sub_count * eval_len_packed;
        let mut sub_evals_packed: Vec<E::Packed> = vec![E::PACKED_ZERO_NORMAL; total_packed];

        // Scratch buffer for one sub-poly's gathered coefficients.
        let mut scratch: Vec<E::Packed> = vec![E::PACKED_ZERO_NORMAL; raw_len_per_sub_packed];

        for poly_idx in 0..k_polys {
            let poly_packed = &polys_packed[poly_idx];
            for k in 0..sub_count {
                // sub_k[i] = poly_logical[k + i * sub_count]
                //         = poly_packed[k_pbase + i * sub_count_packed].lane[k_lane]
                // where k_pbase = k / 8, k_lane = k % 8.
                let k_pbase = k / 8;
                let k_lane = k % 8;

                gather_strided_lanes::<E>(
                    poly_packed,
                    k_pbase,
                    k_lane,
                    sub_count_packed,
                    &mut scratch,
                );

                let off = poly_idx * sub_count * eval_len_packed + k * eval_len_packed;
                if SRC_IS_MONT {
                    E::fft_into_packed_mont(
                        fft,
                        &scratch,
                        &mut sub_evals_packed[off..off + eval_len_packed],
                    );
                } else {
                    E::fft_into_packed(
                        fft,
                        &scratch,
                        &mut sub_evals_packed[off..off + eval_len_packed],
                    );
                }
            }
        }

        let leaf_size = 2 * sub_count * k_polys;
        let value_packed = Arc::new(sub_evals_packed);
        let interpolation = InterpolateValueMBExt::<E>::from_fft_pair_major_parts_packed(
            Arc::clone(&value_packed),
            leaf_size,
        );

        Self {
            interpolation,
            sub_evals_mont_packed: value_packed,
            poly_packed: polys_packed.to_vec(),
            _phantom: PhantomData,
        }
    }

    pub fn commit(&self) -> MerkleRoot {
        MerkleRoot(self.interpolation.commit())
    }
}

/// Gather the strided sub-poly into a contiguous packed scratch buffer.
///
/// For sub-poly index `k` with `k_pbase = k / 8`, `k_lane = k % 8`, we have
/// `sub_k[i] = poly_packed[k_pbase + i * sub_count_packed].lane[k_lane]`.
/// To repack 8 logical sub_k values into one E::Packed block, we read 8
/// source PEF blocks (at strided positions) and extract `k_lane` from each.
///
/// A: simple lane-by-lane via `unpack_lane` / `pack_lanes`. An 8x8
/// SIMD transpose can replace this in a future iteration when the gather
/// cost shows up in profiling. For sub_count = 8 (default), this gather
/// runs `raw_len_per_sub_packed * sub_count` times; for nv = 21 it's
/// `2^15 * 8 = 2^18` calls per poly, manageable.
fn gather_strided_lanes<E: DeepFoldElement>(
    poly_packed: &[E::Packed],
    k_pbase: usize,
    k_lane: usize,
    sub_count_packed: usize,
    scratch: &mut [E::Packed],
) {
    let n = scratch.len();
    for j in 0..n {
        let block_base = k_pbase + j * 8 * sub_count_packed;
        let lanes: [E; 8] = std::array::from_fn(|l| {
            E::unpack_lane(poly_packed[block_base + l * sub_count_packed], k_lane)
        });
        scratch[j] = E::pack_lanes(lanes);
    }
}

// ---------------------------------------------------------------------------
// Ext combine kernels (PEF * PEF Karatsuba accumulate).
//
// Mirrors `combine_opt_ext3_inner` semantics in `deepfold_mamabear.rs`, but
// for *packed extension* inputs (PEF * PEF, NOT base * ext). Per
// multiplication: 6 mont_mul (Ext3), vs the base path's 3. Output range per
// component: `[0, 2.0001P)` (`reduce_fast` per output block).
//
// 4-way ILP unroll on the inner accumulator: pair-then-tree adds break the
// `acc.lazy_add(prod1).lazy_add(prod2)...` serial dependency chain into two
// independent partial sums per quad, matching the `combine_opt_ext3_inner`
// pattern in the base path. This wins meaningfully when `n_polys >= 4`
// (typical mixed-batch open with base + ext lifts).
//
// Range: `prod` per component in [0, 2.0001P) (PEF Mul output via internal
// `reduce_fast`); accumulator after K-poly sum < (1 + 2.0001*K)*P, well
// below 2^64 for any realistic K (~30).
// ---------------------------------------------------------------------------

use std::ops::Mul;

/// Per-output-block 4-way ILP accumulator. Generic over `T` so the same body
/// monomorphizes for PEF3 (and also serves the par variants in
/// `deepfold_mamabear_ext_par.rs`).
#[inline(always)]
pub(crate) fn accumulate_combine_block_4way<T>(
    powers_packed: &[T],
    polys_packed: &[&[T]],
    k: usize,
    zero: T,
) -> T
where
    T: Copy + Mul<Output = T> + LazyReduction,
{
    let n_polys = polys_packed.len();
    let mut acc = zero;
    let mut j = 0;
    // 4-way unroll: independent prods, tree-reduced pair adds, single chain into acc.
    while j + 4 <= n_polys {
        let prod1 = powers_packed[j] * polys_packed[j][k];
        let prod2 = powers_packed[j + 1] * polys_packed[j + 1][k];
        let prod3 = powers_packed[j + 2] * polys_packed[j + 2][k];
        let prod4 = powers_packed[j + 3] * polys_packed[j + 3][k];
        let s12 = prod1.lazy_add(prod2);
        let s34 = prod3.lazy_add(prod4);
        acc = acc.lazy_add(s12).lazy_add(s34);
        j += 4;
    }
    // Tail: at most one of {3-way, 2-way} runs (since the 4-way loop drains
    // n_polys mod 4), then optionally one trailing 1-way.
    if j + 3 <= n_polys {
        let prod1 = powers_packed[j] * polys_packed[j][k];
        let prod2 = powers_packed[j + 1] * polys_packed[j + 1][k];
        let prod3 = powers_packed[j + 2] * polys_packed[j + 2][k];
        let s12 = prod1.lazy_add(prod2);
        acc = acc.lazy_add(s12).lazy_add(prod3);
        j += 3;
    } else if j + 2 <= n_polys {
        let prod1 = powers_packed[j] * polys_packed[j][k];
        let prod2 = powers_packed[j + 1] * polys_packed[j + 1][k];
        let s12 = prod1.lazy_add(prod2);
        acc = acc.lazy_add(s12);
        j += 2;
    }
    if j < n_polys {
        let prod = powers_packed[j] * polys_packed[j][k];
        acc = acc.lazy_add(prod);
    }
    acc.reduce_fast()
}

#[inline]
fn combine_pef3_packed_mont<const SRC_IS_MONT: bool>(
    polys_packed: &[&[PackedMamaBearAVX512Ext3]],
    r_mont: MamaBearScalarExt3,
) -> Vec<PackedMamaBearAVX512Ext3> {
    let n_polys = polys_packed.len();
    assert!(n_polys >= 1, "combine_pef3: at least one poly required");
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

    let mut out: Vec<PackedMamaBearAVX512Ext3> = Vec::with_capacity(len_packed);
    for k in 0..len_packed {
        out.push(accumulate_combine_block_4way::<PackedMamaBearAVX512Ext3>(
            &powers_packed,
            polys_packed,
            k,
            zero_mont_pef,
        ));
    }
    out
}

// ---------------------------------------------------------------------------
// DeepFoldMamaBearProverExt::open (single-view ext open).
//
// B scope: same-element-type open, i.e. `DeepFoldMamaBearProverExt<E, E>`.
// Mixed-view open (base + ext together) is via `DeepFoldCommitView`.
// ---------------------------------------------------------------------------

impl<E> DeepFoldMamaBearProverExt<E, E>
where
    E: DeepFoldExtField + DeepFoldElement,
    // Type-equality bound: forces `<E as DeepFoldExtField>::PackedExt` and
    // `<E as DeepFoldElement>::Packed` to be the same concrete type at the
    // type-system level. True by construction for Ext3 (resolves to PEF3),
    // but Rust does not infer this without an explicit bound. Letting us
    // drop the `unsafe { mem::transmute }` bridge that formerly bridged the
    // two associated types (formerly妥协 7).
    E: DeepFoldExtField<PackedExt = <E as DeepFoldElement>::Packed>,
{
    /// Driver: build timings, dispatch to `open_inner`. Mirrors the base
    /// `DeepFoldMamaBearProver::open` API (single-prover variant -- multi-view
    /// goes through 's `DeepFoldCommitView`).
    pub fn open(
        pp: &DeepFoldMamaBearParam,
        provers: &[&Self],
        point_mont: Vec<E>,
        transcript: &mut Transcript,
    ) {
        let mut timings = OpenTimings::default();
        Self::open_inner(pp, provers, point_mont, transcript, &mut timings, false);
    }

    fn open_inner(
        pp: &DeepFoldMamaBearParam,
        provers: &[&Self],
        point_mont: Vec<E>,
        transcript: &mut Transcript,
        _timings: &mut OpenTimings,
        _record: bool,
    ) {
        let split_level = pp.split_level;
        let sub_count = 1usize << split_level;
        let eval_len_packed = pp.fft_groups[0].size() / 8;

        let r_raw: E = transcript.challenge_f();
        let r_mont = r_raw.to_mont();

        // -- combine_polys (ext, normal-form input) --
        // For each prover, gather per-poly packed slices. Each `prover.poly_packed[k]`
        // is `Arc<Vec<E::Packed>>`; we view `.as_slice` for the combine kernel.
        let poly_slices: Vec<&[<E as DeepFoldElement>::Packed]> = provers
            .iter()
            .flat_map(|p| p.poly_packed.iter().map(|v| v.as_slice()))
            .collect();
        let poly_evals_packed: Vec<<E as DeepFoldElement>::Packed> =
            <E as DeepFoldElement>::combine_packed_mont(&poly_slices, r_mont, false);

        // -- combine_subs (ext, Mont-form input from FFT output) --
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
            <E as DeepFoldElement>::combine_packed_mont(&sub_slices, r_mont, true);

        // -- main fold loop + grind + sample query indices (shared with base) --
        // Thanks to the `<E as DeepFoldExtField>::PackedExt = <E as DeepFoldElement>::Packed`
        // bound on the impl, `Vec<<E as DeepFoldElement>::Packed>` and
        // `Vec<<E as DeepFoldExtField>::PackedExt>` are the same type and pass
        // directly to the helper -- no transmute needed (formerly妥协 7).
        let (mut leaf_indices, fri_results) = open_main_fold_after_combine::<E>(
            pp,
            poly_evals_packed,
            sub_evals_packed,
            &point_mont,
            transcript,
            _timings,
            _record,
        );

        // -- query  --
        let query = provers
            .iter()
            .map(|j| j.interpolation.query(&leaf_indices))
            .collect::<Vec<_>>();
        for q in query {
            transcript.append_u8_slice(&q.0, q.0.len());
            for j in q.1 {
                // E = F here, so `j: E` is the field element to append.
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
// DeepFoldMamaBearVerifierExt<F, E>: ext-field verifier (single-view, E = F).
//
// Mirrors `DeepFoldMamaBearVerifier::verify_inner` with these substitutions:
//   - Round-0 fat-leaf proof values are read as `E` (instead of `MamaBearScalar`)
//     using `proof.get_next_and_step::<E>`. Each E takes E::SIZE bytes
//     (21) instead of 7.
//   - Leaf bytes for Merkle re-computation are built via `as_bytes_vec::<E>(...)`
//     (E::SIZE per element), matching the prover's `E::write_pair_bytes` layout
//     by virtue of `MamaBearScalarExt3::serialize_into` writing the same
//     `[c0_le7, c1_le7, ...]` byte stream as `write_pair_bytes`.
//   - Combine accumulator drops `F::from_base_mont(pv[q])`; instead `pv[q]`
//     is already `F` (E = F single-view constraint).
//   - Subsequent FRI fold + consistency check is byte-identical to base.
//
// scope: single-view (one prover) ext verifier. Mixed batch with
// base + ext views together is a follow-up (/ view trait refactor).
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct DeepFoldMamaBearVerifierExt<F, E>
where
    F: DeepFoldExtField,
    E: DeepFoldElement,
{
    pub(crate) commit: MerkleTreeVerifierMB,
    pub(crate) poly_num: usize,
    _phantom: PhantomData<(F, E)>,
}

impl<F, E> DeepFoldMamaBearVerifierExt<F, E>
where
    F: DeepFoldExtField,
    E: DeepFoldElement,
{
    pub fn new(pp: &DeepFoldMamaBearParam, commit: MerkleRoot, poly_num: usize) -> Self {
        Self {
            commit: MerkleTreeVerifierMB::new(pp.fft_groups[0].size() / 2, commit.0),
            poly_num,
            _phantom: PhantomData,
        }
    }
}

impl<E> DeepFoldMamaBearVerifierExt<E, E>
where
    E: DeepFoldExtField + DeepFoldElement + Field,
{
    /// Verify a single-view ext open transcript. Returns `true` iff the
    /// proof is valid for the given commit + claimed evals at `point`.
    pub fn verify(
        pp: &DeepFoldMamaBearParam,
        verifiers: Vec<&Self>,
        point: Vec<E>,
        evals: Vec<Vec<E>>,
        transcript: &mut Transcript,
        proof: &mut Proof,
    ) -> bool {
        let split_level = pp.split_level;
        let sub_count = 1usize << split_level;

        // ---- Phases 1+2+3 (extracted; shared with base verifier) ----
        let mut timings = crate::deepfold_mamabear::VerifyTimings::default();
        let state = match crate::deepfold_mamabear::verify_pre_round0_fold_check::<E>(
            pp,
            point,
            evals,
            transcript,
            proof,
            &mut timings,
            false,
        ) {
            Some(s) => s,
            None => return false,
        };
        let r_mont = state.r_mont;
        let eval_mont = state.eval_mont;
        let challenges_mont = state.challenges_mont;
        let commits = state.commits;
        let leaf_indices = state.leaf_indices;
        let mut query_results: Vec<crate::deepfold_mamabear::QueryResultMB<E>> = vec![];
        let indices = leaf_indices.clone();

        // ---- Fat-leaf Merkle verify + per-prover regroup ----
        {
            let mut poly_values: Vec<Vec<E>> = vec![];
            for verifier in verifiers.iter() {
                let fat_leaf_size = 2 * sub_count * verifier.poly_num;
                let proof_bytes = proof.get_next_slice(verifier.commit.proof_length(&leaf_indices));
                let proof_values: Vec<E> = (0..leaf_indices.len() * fat_leaf_size)
                    .map(|_| proof.get_next_and_step::<E>())
                    .collect();
                transcript.append_u8_slice(&proof_bytes, proof_bytes.len());
                for k in &proof_values {
                    transcript.append_f(*k);
                }

                // Group by (poly, sub_k): see `DeepFoldMamaBearVerifier::verify_inner`
                // for the slot layout. Slot 2k = x for sub-poly k, slot 2k+1 = nx.
                for p in 0..verifier.poly_num {
                    for k in 0..sub_count {
                        let slot_j = (p * sub_count + k) * 2;
                        let slot_jh = (p * sub_count + k) * 2 + 1;
                        let mut vals = Vec::with_capacity(leaf_indices.len() * 2);
                        for q in 0..leaf_indices.len() {
                            vals.push(proof_values[slot_j * leaf_indices.len() + q]);
                        }
                        for q in 0..leaf_indices.len() {
                            vals.push(proof_values[slot_jh * leaf_indices.len() + q]);
                        }
                        poly_values.push(vals);
                    }
                }

                // Merkle verify: rebuild leaves from (slot, query) -> E values.
                // Layout follows base verifier: leaf at query q contains
                // `fat_leaf_size` E values in slot order.
                let mut base_leaves: Vec<Vec<u8>> = Vec::with_capacity(leaf_indices.len());
                for q in 0..leaf_indices.len() {
                    let leaf_vals: Vec<E> = (0..fat_leaf_size)
                        .map(|s| proof_values[s * leaf_indices.len() + q])
                        .collect();
                    base_leaves.push(as_bytes_vec::<E>(&leaf_vals));
                }
                if !verifier.commit.verify(&proof_bytes, &leaf_indices, &base_leaves) {
                    return false;
                }
            }

            // ---- Combine sub-poly values with r + local split fold ----
            let num_queries = leaf_indices.len();
            let total_sub_groups = poly_values.len();

            let mut combined_j: Vec<Vec<E>> = vec![Vec::new(); sub_count];
            let mut combined_jh: Vec<Vec<E>> = vec![Vec::new(); sub_count];
            for k in 0..sub_count {
                combined_j[k] = vec![E::zero().to_mont(); num_queries];
                combined_jh[k] = vec![E::zero().to_mont(); num_queries];
            }

            // E = F (single-view constraint), so pv[q] is already F. No
            // `from_base_mont` lifting needed (vs base verifier).
            for group_idx in 0..(total_sub_groups / sub_count) {
                for k in 0..sub_count {
                    let pv = &poly_values[group_idx * sub_count + k];
                    for q in 0..num_queries {
                        combined_j[k][q] = combined_j[k][q] * r_mont + pv[q];
                        combined_jh[k][q] = combined_jh[k][q] * r_mont + pv[num_queries + q];
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

            query_results.push(crate::deepfold_mamabear::QueryResultMB {
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
        }

        // ---- Phases 5 + 6 (extracted; shared with base verifier, ~110 LoC dedup) ----
        let mut timings = crate::deepfold_mamabear::VerifyTimings::default();
        crate::deepfold_mamabear::verify_after_round0_fri_check::<E>(
            pp,
            leaf_indices,
            indices,
            query_results,
            &challenges_mont,
            eval_mont,
            &commits,
            transcript,
            proof,
            &mut timings,
            false,
        )
    }
}

// ---------------------------------------------------------------------------
// Tests (A: deterministic commit, leaf hash sanity).
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::SmallRng;
    use rand::{RngCore, SeedableRng};

    fn make_rng(seed: u64) -> SmallRng {
        SmallRng::seed_from_u64(seed)
    }

    /// Run the closure on a fresh thread with a 16 MB stack. The fused dense3
    /// kernel in `arithmetic::fft_mamabear_ext` returns 8-tuples of PEF
    /// (1024 / 1536 bytes) which inline cleanly in release but balloon the
    /// stack frame at `-Copt-level=0`. Default test thread stack is 2 MB.
    fn run_with_large_stack<F: FnOnce() + Send + 'static>(f: F) {
        std::thread::Builder::new()
            .stack_size(16 * 1024 * 1024)
            .spawn(f)
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

    /// Round-trip pack/unpack: `pack_lanes(unpack_lane * 8) == identity`.
    #[test]
    fn pack_unpack_lane_round_trip_pef3() {
        let mut rng = make_rng(0x2222);
        let blocks = random_pef3_vec(16, &mut rng);
        for block in &blocks {
            let lanes: [MamaBearScalarExt3; 8] =
                std::array::from_fn(|l| MamaBearScalarExt3::unpack_lane(*block, l));
            let repacked = MamaBearScalarExt3::pack_lanes(lanes);
            for l in 0..8 {
                let a = MamaBearScalarExt3::unpack_lane(*block, l);
                let b = MamaBearScalarExt3::unpack_lane(repacked, l);
                assert_eq!(a.c0.0, b.c0.0);
                assert_eq!(a.c1.0, b.c1.0);
                assert_eq!(a.c2.0, b.c2.0);
            }
        }
    }

    /// Build a `DeepFoldMamaBearParam` for testing. We use small
    /// `variable_num` so the test runs in milliseconds. Code rate 1/8
    /// (default) and split_level >= 3 (also default).
    fn test_param(variable_num: usize) -> DeepFoldMamaBearParam {
        // code_rate_log = 3, query_num = 8 (irrelevant for commit-only tests)
        DeepFoldMamaBearParam::new_default(variable_num, 3, 8)
    }

    /// Determinism: building the same prover twice produces the same root.
    #[test]
    fn ext_prover_commit_pef3_deterministic() {
        run_with_large_stack(|| {
        let mut rng = make_rng(0xBABA);
        let pp = test_param(10);
        let n = 1usize << pp.variable_num;
        let n_packed = n / 8;
        let polys = vec![Arc::new(random_pef3_vec(n_packed, &mut rng))];

        let p1 = DeepFoldMamaBearProverExt::<MamaBearScalarExt3, MamaBearScalarExt3>::new(
            &pp, &polys,
        );
        let r1 = p1.commit();

        let p2 = DeepFoldMamaBearProverExt::<MamaBearScalarExt3, MamaBearScalarExt3>::new(
            &pp, &polys,
        );
        let r2 = p2.commit();

        assert_eq!(r1.0, r2.0, "ext PEF3 commit must be deterministic");
        });
    }

    /// `new_from_mont` produces a byte-identical commit to "from_mont then
    /// new" (the conventional path that a caller would
    /// otherwise need to follow). This validates that the const-generic
    /// `SRC_IS_MONT == true` path through ext FFT is consistent with the
    /// normal path, and that it skips exactly the right amount of work
    /// (one `to_montgomery` per loaded element, no functional change).
    #[test]
    fn ext_prover_new_from_mont_pef3_matches_from_mont_then_new() {
        run_with_large_stack(|| {
        let mut rng = make_rng(0x4F4E);
        let pp = test_param(10);
        let n = 1usize << pp.variable_num;
        let n_packed = n / 8;

        let polys_mont: Vec<Arc<Vec<PackedMamaBearAVX512Ext3>>> = (0..2)
            .map(|_| {
                let raw = random_pef3_vec(n_packed, &mut rng);
                let mont: Vec<PackedMamaBearAVX512Ext3> =
                    raw.iter().map(|v| v.to_montgomery()).collect();
                Arc::new(mont)
            })
            .collect();

        let polys_via_from_mont: Vec<Arc<Vec<PackedMamaBearAVX512Ext3>>> = polys_mont
            .iter()
            .map(|arc| {
                let normal: Vec<PackedMamaBearAVX512Ext3> =
                    arc.iter().map(|v| v.from_montgomery()).collect();
                Arc::new(normal)
            })
            .collect();
        let prover_normal = DeepFoldMamaBearProverExt::<
            MamaBearScalarExt3,
            MamaBearScalarExt3,
        >::new(&pp, &polys_via_from_mont);
        let prover_mont = DeepFoldMamaBearProverExt::<
            MamaBearScalarExt3,
            MamaBearScalarExt3,
        >::new_from_mont(&pp, &polys_mont);
        assert_eq!(prover_normal.commit().0, prover_mont.commit().0);
        });
    }

    /// end-to-end round-trip: commit + open + verify with a single
    /// ext prover. Validates that ext open produces a transcript the ext
    /// verifier accepts. Smaller log_order keeps test runtime down (Ext3
    /// verifier proof bytes scale with the 21-byte element width).
    #[test]
    fn ext_prover_open_pef3_verify_round_trip() {
        run_with_large_stack(|| {
        use arithmetic::poly::MultiLinearPoly;
        use util::fiat_shamir::Transcript;
        let mut rng = make_rng(0xFADE_BABE);
        let pp = test_param(8);
        let n = 1usize << pp.variable_num;
        let n_packed = n / 8;

        let k_polys = 1usize;
        let polys_logical: Vec<Vec<MamaBearScalarExt3>> = (0..k_polys)
            .map(|_| {
                (0..n)
                    .map(|_| MamaBearScalarExt3 {
                        c0: MamaBearScalar(rng.next_u64() % arithmetic::field::mamabear::P),
                        c1: MamaBearScalar(rng.next_u64() % arithmetic::field::mamabear::P),
                        c2: MamaBearScalar(rng.next_u64() % arithmetic::field::mamabear::P),
                    })
                    .collect::<Vec<_>>()
            })
            .collect();

        let polys_packed: Vec<Arc<Vec<PackedMamaBearAVX512Ext3>>> = polys_logical
            .iter()
            .map(|p| {
                let mut blocks: Vec<PackedMamaBearAVX512Ext3> = Vec::with_capacity(n_packed);
                for chunk_start in (0..n).step_by(8) {
                    let lanes: [MamaBearScalarExt3; 8] =
                        std::array::from_fn(|l| p[chunk_start + l]);
                    blocks.push(MamaBearScalarExt3::pack_lanes(lanes));
                }
                Arc::new(blocks)
            })
            .collect();

        let prover = DeepFoldMamaBearProverExt::<MamaBearScalarExt3, MamaBearScalarExt3>::new(
            &pp, &polys_packed,
        );
        let commit = prover.commit();

        let point: Vec<MamaBearScalarExt3> = (0..pp.variable_num)
            .map(|_| {
                MamaBearScalarExt3::from(MamaBearScalar(
                    rng.next_u64() % arithmetic::field::mamabear::P,
                ))
            })
            .collect();
        let point_mont: Vec<MamaBearScalarExt3> = point.iter().map(|v| v.to_mont()).collect();
        let polys_logical_mont: Vec<Vec<MamaBearScalarExt3>> = polys_logical
            .iter()
            .map(|p| p.iter().map(|v| v.to_mont()).collect())
            .collect();
        let claim_evals_mont: Vec<MamaBearScalarExt3> = polys_logical_mont
            .iter()
            .map(|p| MultiLinearPoly::eval_multilinear_ext(p, &point_mont))
            .collect();

        let mut prover_transcript = Transcript::new();
        prover_transcript.append_u8_slice(&commit.0, HASH_SIZE);
        for ev in &claim_evals_mont {
            prover_transcript.append_f(*ev);
        }
        DeepFoldMamaBearProverExt::<MamaBearScalarExt3, MamaBearScalarExt3>::open(
            &pp,
            &[&prover],
            point_mont.clone(),
            &mut prover_transcript,
        );

        let mut proof_bytes = util::fiat_shamir::Proof::default();
        proof_bytes.bytes = prover_transcript.proof.bytes.clone();
        let _commit_bytes = proof_bytes.get_next_slice(HASH_SIZE);
        let mut verifier_transcript = Transcript::new();
        verifier_transcript.append_u8_slice(&commit.0, HASH_SIZE);

        let mut claim_from_proof: Vec<Vec<MamaBearScalarExt3>> = vec![Vec::with_capacity(k_polys)];
        for _ in 0..k_polys {
            let v: MamaBearScalarExt3 = proof_bytes.get_next_and_step();
            verifier_transcript.append_f(v);
            claim_from_proof[0].push(v);
        }

        let verifier = DeepFoldMamaBearVerifierExt::<MamaBearScalarExt3, MamaBearScalarExt3>::new(
            &pp,
            commit,
            k_polys,
        );
        let ok = DeepFoldMamaBearVerifierExt::<MamaBearScalarExt3, MamaBearScalarExt3>::verify(
            &pp,
            vec![&verifier],
            point,
            claim_from_proof,
            &mut verifier_transcript,
            &mut proof_bytes,
        );
        assert!(ok, "ext PEF3 round-trip verify failed");
        });
    }

    /// Query returns `leaf_indices.len * leaf_size` scalars. Sanity check.
    #[test]
    fn ext_prover_query_returns_correct_count() {
        run_with_large_stack(|| {
        let mut rng = make_rng(0x7777);
        let pp = test_param(10);
        let n = 1usize << pp.variable_num;
        let n_packed = n / 8;
        let polys = vec![
            Arc::new(random_pef3_vec(n_packed, &mut rng)),
            Arc::new(random_pef3_vec(n_packed, &mut rng)),
            Arc::new(random_pef3_vec(n_packed, &mut rng)),
        ];

        let prover = DeepFoldMamaBearProverExt::<MamaBearScalarExt3, MamaBearScalarExt3>::new(
            &pp, &polys,
        );
        let leaf_count = prover.interpolation.leave_num();
        let leaf_indices = vec![0usize, 1, leaf_count / 2, leaf_count - 1];

        let (proof_bytes, values) = prover.interpolation.query(&leaf_indices);
        assert_eq!(values.len(), leaf_indices.len() * prover.interpolation.leaf_size);
        assert!(!proof_bytes.is_empty(), "Merkle proof must contain sibling hashes");
        });
    }
}
