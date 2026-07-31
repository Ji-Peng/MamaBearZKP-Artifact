///! DeepFold: Parameterized t-level FRI split for MamaBear field.
///!
///! Extends V1 with split-level parameter (t=0..4):
///! - t=0: identical to V1 (no split)
///! - t>0: split coefficients into 2^t sub-polynomials, each FFT'd on domain N/2^t
///!   - Fat-leaf Merkle tree: 2^{t+1} values per leaf
///!   - First t FRI rounds become cheap pointwise linear combinations (no intermediate Merkle)
///!   - Standard FRI fold resumes from round t
///!
///! All internal computation is in Montgomery form.
use std::{collections::HashMap, sync::Arc, time::Instant};

use arithmetic::{
    fft_mamabear::MamaBearFFT,
    field::{
        as_bytes_vec,
        mamabear::{
            LazyReduction, MamaBearScalar, MamaBearScalarExt3,
            PackedExtensionField, PackedMamaBearAVX512,
            PackedMamaBearAVX512Ext3, P,
        },
        Field,
    },
};
use util::{
    blake3_batch,
    fiat_shamir::Proof,
    merkle_tree_mamabear::{MerkleTreeProverMB, MerkleTreeVerifierMB, HASH_SIZE},
};

#[cfg(test)]
use super::CommitmentSerde;
use crate::deepfold::MerkleRoot;
use crate::Transcript;

// --- Parameters ---

/// Default split level used by `DeepFoldMamaBearParam::new_default`.
///
/// The "split level" t controls how many variables are committed as a single
/// batched sub-FFT before classical FRI folding begins: round 0 folds t+1
/// variables into a fat Merkle leaf (2^{t+1} values per leaf) and collapses
/// the first t FRI rounds into cheap pointwise linear combinations with no
/// intermediate Merkle tree; standard FRI fold then resumes from round t.
///
/// Two competing per-query effects determine the proof-size optimum:
/// - Round-0 leaf inflates from `2 * K * fe_base` (standard) to
///   `2^(t+1) * K * fe_base` (split), scaling linearly with the number of
///   polynomials K committed into the round-0 tree.
/// - Auth-path savings come from "t absorbed middle trees + round-0 tree
///   shortened by t levels", with byte count proportional to `log n * h`
///   and independent of K.
///
/// Since the cost is O(K) while the benefit is O(1) in K, the proof-size
/// optimum shifts toward smaller t as K grows. For HyperPlonk as used here
/// (K = 7 across two provers: 4 preprocessed polys + 3 witness polys;
/// n = 2^21, 32 queries, 32-byte hashes), `t = 3` gives the minimum proof
/// size, saving roughly 23 KB over `t = 0`. Going higher regresses quickly
/// because of the 2^(t+1) leaf term — at `t = 5` the fat-leaf inflation
/// actually overtakes the auth-path savings and proof size ends up larger
/// than `t = 0`. Older single-poly analyses that suggest `t = 4..6` is
/// optimal silently assume K = 1 and do not apply in the HyperPlonk
/// setting.
///
/// `t = 3` is also near the prover-time sweet spot (roughly 3-4x over
/// `t = 0`), so no separate throughput-vs-size knob is needed.
pub const DEFAULT_SPLIT_LEVEL: usize = 3;

#[derive(Clone)]
pub struct DeepFoldMamaBearParam {
    /// FFT groups for standard FRI fold rounds (rounds split_level..variable_num-1).
    /// fft_groups[k] has domain size N/2^{split_level+k}.
    /// fft_groups[0] also serves as the sub-FFT for commit.
    pub fft_groups: Vec<MamaBearFFT>,
    pub variable_num: usize,
    pub query_num: usize,
    pub split_level: usize,
    /// FRI grinding (proof-of-work) bits applied before the query-index draw.
    /// 0 disables grinding entirely (no transcript bytes appended), preserving
    /// byte-identical transcripts for callers that don't opt in.
    pub grinding_bits: u32,
}

impl DeepFoldMamaBearParam {
    /// Create parameters with explicit split level. Grinding is disabled by
    /// default; set `param.grinding_bits` directly to enable.
    pub fn new(
        variable_num: usize,
        code_rate_log: usize,
        query_num: usize,
        split_level: usize,
    ) -> Self {
        assert!(
            split_level <= variable_num,
            "split_level must be <= variable_num"
        );
        let log_n = variable_num + code_rate_log;
        assert!(log_n >= split_level, "domain too small for split_level");

        // fft_groups[k] = MamaBearFFT(log_n - split_level - k) for k = 0..variable_num-split_level
        let num_standard_rounds = variable_num - split_level;
        let mut fft_groups = Vec::with_capacity(num_standard_rounds);
        for k in 0..num_standard_rounds {
            fft_groups.push(MamaBearFFT::new((log_n - split_level - k) as u32));
        }

        // Precompute chirp twiddle tables for fft_groups[0] (the commit FFT with
        // zero-padded input). Only this group sees zero-padded FFTs; the rest are
        // used for dense FRI fold rounds.
        if !fft_groups.is_empty() && fft_groups[0].log_order >= 6 {
            fft_groups[0].precompute_chirp_prefix3();
        }

        Self {
            fft_groups,
            variable_num,
            query_num,
            split_level,
            grinding_bits: 0,
        }
    }

    /// Create parameters with DEFAULT_SPLIT_LEVEL (=3).
    pub fn new_default(variable_num: usize, code_rate_log: usize, query_num: usize) -> Self {
        Self::new(variable_num, code_rate_log, query_num, DEFAULT_SPLIT_LEVEL)
    }
}

// --- Interpolation value with Merkle tree ---

#[derive(Clone, Copy)]
enum InterpolationLayout {
    LeafSlotMajor,
    FftPairMajorBlocked,
}

#[derive(Clone)]
pub struct InterpolateValueMB<F: Field> {
    /// Shared via `Arc<Vec<F>>` so construction from a freshly built `Vec<F>`
    /// is a zero-copy `Arc::new` rather than the `Arc::<[F]>::from(vec)` memcpy.
    pub value: Arc<Vec<F>>,
    leaf_size: usize,
    merkle_tree: MerkleTreeProverMB,
    layout: InterpolationLayout,
}

impl<F: Field> InterpolateValueMB<F> {
    pub fn new(value: Vec<F>, leaf_size: usize) -> Self {
        let value: Arc<Vec<F>> = Arc::new(value);
        let len = value.len() / leaf_size;
        let merkle_tree = MerkleTreeProverMB::new(
            (0..len)
                .map(|i| {
                    as_bytes_vec::<F>(
                        &(0..leaf_size)
                            .map(|j| value[len * j + i])
                            .collect::<Vec<_>>(),
                    )
                })
                .collect(),
        );
        Self {
            value,
            leaf_size,
            merkle_tree,
            layout: InterpolationLayout::LeafSlotMajor,
        }
    }

    pub fn leave_num(&self) -> usize {
        self.merkle_tree.leave_num()
    }

    pub fn commit(&self) -> [u8; HASH_SIZE] {
        self.merkle_tree.commit()
    }

    pub fn query(&self, leaf_indices: &[usize]) -> (Vec<u8>, Vec<F>) {
        let len = self.merkle_tree.leave_num();
        assert_eq!(len * self.leaf_size, self.value.len());
        let proof_values = match self.layout {
            InterpolationLayout::LeafSlotMajor => (0..self.leaf_size)
                .flat_map(|i| {
                    leaf_indices
                        .iter()
                        .map(|j| self.value[*j + i * len])
                        .collect::<Vec<_>>()
                })
                .collect(),
            InterpolationLayout::FftPairMajorBlocked => {
                let segment_len = len * 2;
                let segment_count = self.leaf_size / 2;
                let mut proof_values = Vec::with_capacity(leaf_indices.len() * self.leaf_size);

                for segment in 0..segment_count {
                    let base = segment * segment_len;
                    for &leaf_idx in leaf_indices {
                        let (x_pos, _) =
                            MamaBearFFT::pair_storage_positions_for_pair_count(leaf_idx, len);
                        proof_values.push(self.value[base + x_pos]);
                    }
                    for &leaf_idx in leaf_indices {
                        let (_, nx_pos) =
                            MamaBearFFT::pair_storage_positions_for_pair_count(leaf_idx, len);
                        proof_values.push(self.value[base + nx_pos]);
                    }
                }

                proof_values
            }
        };
        let proof_bytes = self.merkle_tree.open(leaf_indices);
        (proof_bytes, proof_values)
    }
}

impl InterpolateValueMB<MamaBearScalar> {
    pub(crate) fn from_fft_pair_major_parts(
        value: Arc<Vec<MamaBearScalar>>,
        leaf_size: usize,
    ) -> Self {
        let merkle_tree = round0_pair_major_merkle_tree_from_values(&value, leaf_size);
        Self::from_fft_pair_major_parts_with_tree(value, leaf_size, merkle_tree)
    }

    pub(crate) fn from_fft_pair_major_parts_with_tree(
        value: Arc<Vec<MamaBearScalar>>,
        leaf_size: usize,
        merkle_tree: MerkleTreeProverMB,
    ) -> Self {
        Self {
            value,
            leaf_size,
            merkle_tree,
            layout: InterpolationLayout::FftPairMajorBlocked,
        }
    }
}

// --- FRI fold result ---

#[derive(Clone)]
pub(crate) struct FriFoldResult<F: DeepFoldExtField> {
    /// FRI fold result kept in packed form so the next round can consume it
    /// without re-packing. Logical length = `value_packed.len() * 8`.
    pub(crate) value_packed: Vec<F::PackedExt>,
    /// Canonicalized scalar copy used by the Merkle query path.
    /// Reordered as `[evens..., odds...]` so a query at index i can fetch
    /// both `value[2i]` and `value[2i+1]` via simple offset arithmetic.
    pub(crate) value_query_mont: Vec<F>,
    pub(crate) merkle_tree: MerkleTreeProverMB,
}

impl<F: DeepFoldExtField> FriFoldResult<F> {
    /// Build a FRI fold result from packed Mont values. Walks each packed block
    /// once to produce the canonical scalar buffer needed by the query path,
    /// then takes ownership of the packed buffer for the next round.
    /// Parallel version of `from_packed_values`. Below a conservative size
    /// threshold it falls back to the serial constructor (round overhead
    /// would dominate). Above the threshold it:
    /// 1. parallel-unpacks `value_packed` into the canonical scalar buffer,
    /// 2. parallel reorders evens-then-odds into `value_query_mont`,
    /// 3. parallel-hashes the leaves and builds the Merkle tree via
    ///    `MerkleTreeProverMB::from_leaf_hashes_par`, which itself
    ///    parallelizes the `build_parents` step.
    pub(crate) fn from_packed_values_par(value_packed: Vec<F::PackedExt>) -> Self {
        crate::deepfold_mamabear_par::fri_fold_result_from_packed_values_par::<F>(
            value_packed,
        )
    }

    pub(crate) fn from_packed_values(value_packed: Vec<F::PackedExt>) -> Self {
        let len = value_packed.len() * 8;
        let leave_num = len / 2;

        // Single unpack pass: write canonical scalars in their natural order.
        let mut canonical_by_position: Vec<F> = Vec::with_capacity(len);
        unsafe {
            canonical_by_position.set_len(len);
        }
        let mut tmp = [F::default(); 8];
        for (block_idx, block) in value_packed.iter().enumerate() {
            block.unpack_into_slice(&mut tmp);
            let off = block_idx * 8;
            for k in 0..8 {
                canonical_by_position[off + k] = tmp[k].reduce_canonical();
            }
        }

        // Reorder evens-then-odds for the query path.
        let mut value_query_mont = Vec::with_capacity(len);
        value_query_mont.extend(canonical_by_position.iter().step_by(2).copied());
        value_query_mont.extend(canonical_by_position.iter().skip(1).step_by(2).copied());

        let merkle_tree = MerkleTreeProverMB::new(
            (0..leave_num)
                .map(|i| as_bytes_vec::<F>(&[value_query_mont[i], value_query_mont[i + leave_num]]))
                .collect(),
        );

        FriFoldResult {
            value_packed,
            value_query_mont,
            merkle_tree,
        }
    }

    pub(crate) fn commit(&self) -> [u8; HASH_SIZE] {
        self.merkle_tree.commit()
    }

    pub(crate) fn query(&self, leaf_indices: &[usize]) -> (Vec<u8>, Vec<F>) {
        let leave_num = self.merkle_tree.leave_num();
        let proof_values = leaf_indices
            .iter()
            .map(|&j| self.value_query_mont[j])
            .chain(
                leaf_indices
                    .iter()
                    .map(|&j| self.value_query_mont[j + leave_num]),
            )
            .collect();
        let proof_bytes = self.merkle_tree.open(leaf_indices);
        (proof_bytes, proof_values)
    }
}

// --- DeepFold trait for Ext3 (extension-field) abstraction ---

/// Montgomery form of 0. Canonical is 0 (after `to_montgomery` does
/// `con_sub_xp(1)` on `mont_mul(0, R²) = P`). Use 0 — never `P` — so that
/// downstream `Sub` / `Neg` (both require operand < P) stay sound.
pub(crate) const ZERO_MONT: MamaBearScalar = MamaBearScalar(0);

pub trait DeepFoldExtField:
    Field<BaseField = MamaBearScalar> + LazyReduction + From<MamaBearScalar> + Send + Sync
{
    /// SIMD-packed counterpart of `Self`. 8 logical lanes per packed block.
    /// Named `PackedExt` (rather than `Packed`) to avoid colliding with
    /// `MamaBearExtConfig::Packed` in the hyperplonk crate, since the open()
    /// caller types satisfy both traits and the lookup needs to be unambiguous.
    type PackedExt: PackedExtensionField<ScalarExt = Self> + Copy + Send + Sync;

    fn to_mont(self) -> Self;
    fn from_mont(self) -> Self;
    fn reduce_canonical(self) -> Self;

    /// Create extension field element from a base-field value already in Montgomery form.
    /// Sets higher components to zero-in-Montgomery (= P, not 0).
    fn from_base_mont(base_mont: MamaBearScalar) -> Self;

    /// Bulk broadcast base-field Montgomery values to extension field.
    /// Default: scalar fallback. Ext3 overrides with AVX-512 SIMD.
    fn from_base_mont_vec(src: &[MamaBearScalar]) -> Vec<Self> {
        src.iter().map(|&v| Self::from_base_mont(v)).collect()
    }

    /// Splat the Montgomery zero (= P literal) across every component of one packed block.
    fn packed_zero_mont() -> Self::PackedExt;

    /// Combine: out_packed[k] (8 lanes) =
    ///     sum_{i=0..n_polys} r^i * lift(base_polys[i][8k..8k+8])
    /// where each base poly is in **Montgomery** form. The output stays packed end-to-end
    /// (no per-lane unpack), so the round loop can consume it directly. Each input slice
    /// must have the same length, divisible by 8.
    ///
    /// 4-way ILP unrolled, uses `mul_base_elem` (PEF×PBF cheap path).
    fn combine_opt_mont_base_to_ext_packed_mont(
        base_polys: &[&[MamaBearScalar]],
        r_mont: Self,
    ) -> Vec<Self::PackedExt>;

    /// Same as above but the base values are in **normal** (non-Montgomery) form;
    /// the precomputed powers absorb an extra R factor so `mul_base_elem` still
    /// produces output in Montgomery form. Mirrors the open()'s `combine_polys`.
    fn combine_opt_normal_base_to_ext_packed_mont(
        base_polys: &[&[MamaBearScalar]],
        r_mont: Self,
    ) -> Vec<Self::PackedExt>;

    /// One level of in-place multilinear fold. Requires `poly_evals_packed.len() >= 2`
    /// (i.e. logical length >= 16). Halves `poly_evals_packed.len()` in place.
    /// The very last 3 fold rounds (logical len 8 → 4 → 2 → 1) happen in scalar
    /// at the call site after unpacking the lone remaining block.
    fn fold_multilinear_packed(
        poly_evals_packed: &mut Vec<Self::PackedExt>,
        challenge_packed: Self::PackedExt,
    );

    /// Evaluate a multilinear (in packed Montgomery form) at `point` (scalar Mont).
    /// Folds in packed form until 1 block remains, then unpacks and finishes the
    /// last 3 levels in scalar. Returns the result in scalar Mont form.
    fn eval_multilinear_packed(poly_evals_packed: &[Self::PackedExt], point: &[Self]) -> Self;

    /// In-place packed split-fold: fold pairs of sub-polynomials pointwise.
    /// `sub_evals_packed` is laid out as `num_subs` consecutive blocks of length
    /// `eval_len_packed`. eval_len is always >= 8 in the shipped configurations.
    fn fold_sub_polys_packed(
        sub_evals_packed: &mut [Self::PackedExt],
        eval_len_packed: usize,
        num_subs: usize,
        challenge_packed: Self::PackedExt,
    );

    /// Parallel version of `combine_opt_mont_base_to_ext_packed_mont`.
    /// Default: falls back to serial.
    fn combine_opt_mont_base_to_ext_packed_mont_par(
        base_polys: &[&[MamaBearScalar]],
        r_mont: Self,
    ) -> Vec<Self::PackedExt> {
        Self::combine_opt_mont_base_to_ext_packed_mont(base_polys, r_mont)
    }

    /// Parallel version of `combine_opt_normal_base_to_ext_packed_mont`.
    /// Default: falls back to serial.
    fn combine_opt_normal_base_to_ext_packed_mont_par(
        base_polys: &[&[MamaBearScalar]],
        r_mont: Self,
    ) -> Vec<Self::PackedExt> {
        Self::combine_opt_normal_base_to_ext_packed_mont(base_polys, r_mont)
    }

    /// Parallel version of `fold_sub_polys_packed`. Takes `src` and `dst`
    /// as separate `&mut Vec` references so the caller can pre-allocate a
    /// ping-pong scratch buffer once and reuse it across FRI rounds.
    /// Writes the new (half-sized) values into `dst`, then swaps `src`
    /// and `dst` so `src` holds the folded output on return and `dst` is
    /// reusable scratch for the next round. The in-place collect variant
    /// was measured to regress split_fold by ~70% at nv=23 due to
    /// per-round `Vec` alloc/drop (mmap/munmap) overhead — hence the
    /// ping-pong design.
    /// Default: delegates to the serial slice variant on `src` directly.
    fn fold_sub_polys_packed_par(
        src: &mut Vec<Self::PackedExt>,
        dst: &mut Vec<Self::PackedExt>,
        eval_len_packed: usize,
        num_subs: usize,
        challenge_packed: Self::PackedExt,
    ) {
        let _ = dst;
        Self::fold_sub_polys_packed(
            src.as_mut_slice(),
            eval_len_packed,
            num_subs,
            challenge_packed,
        );
    }

    /// Parallel version of `fold_multilinear_packed`. Takes `src` and
    /// `dst` as separate `&mut Vec` so the caller can pre-allocate a
    /// ping-pong scratch once and reuse it across all fold rounds.
    /// Writes the new (half-sized) values into `dst`, then swaps
    /// `src`/`dst` so on return `src` holds the folded output and `dst`
    /// is available scratch for the next round. An earlier in-place
    /// wrapper allocated a fresh Vec per call and paid mmap + first-touch
    /// on every round, which silently wiped out the kernel's real 4-14x
    /// parallelism speedup.
    /// Default: delegates to the serial in-place slice variant on `src`.
    fn fold_multilinear_packed_par(
        src: &mut Vec<Self::PackedExt>,
        dst: &mut Vec<Self::PackedExt>,
        challenge_packed: Self::PackedExt,
    ) {
        let _ = dst;
        Self::fold_multilinear_packed(src, challenge_packed);
    }

    /// Parallel version of `eval_multilinear_packed`.
    /// Default: falls back to serial.
    fn eval_multilinear_packed_par(
        poly_evals_packed: &[Self::PackedExt],
        point: &[Self],
    ) -> Self {
        Self::eval_multilinear_packed(poly_evals_packed, point)
    }

    /// FRI fold round 0: input uses pair-major blocked layout `[x0..x7, nx0..nx7]`.
    /// Requires `last_packed.len() >= 2` and even (== 2 * packed_pair_count).
    fn evaluate_next_domain_first_round_packed(
        last_packed: &[Self::PackedExt],
        fft: &MamaBearFFT,
        challenge_packed: Self::PackedExt,
    ) -> Vec<Self::PackedExt>;

    /// FRI fold rounds ≥ 1: input uses pair-adjacent layout (x, nx, x, nx, ...).
    fn evaluate_next_domain_packed(
        last_packed: &[Self::PackedExt],
        fft: &MamaBearFFT,
        challenge_packed: Self::PackedExt,
    ) -> Vec<Self::PackedExt>;

    /// Parallel version of `evaluate_next_domain_first_round_packed`. Each
    /// output chunk depends only on `last_packed[2*chunk]`,
    /// `last_packed[2*chunk+1]`, and a per-chunk twiddle, so the loop is
    /// embarrassingly data-parallel and byte-identical to serial.
    /// Default: falls back to the serial kernel (non-par callers unaffected).
    fn evaluate_next_domain_first_round_packed_par(
        last_packed: &[Self::PackedExt],
        fft: &MamaBearFFT,
        challenge_packed: Self::PackedExt,
    ) -> Vec<Self::PackedExt> {
        Self::evaluate_next_domain_first_round_packed(last_packed, fft, challenge_packed)
    }

    /// Parallel version of `evaluate_next_domain_packed`.
    /// Default: falls back to the serial kernel.
    fn evaluate_next_domain_packed_par(
        last_packed: &[Self::PackedExt],
        fft: &MamaBearFFT,
        challenge_packed: Self::PackedExt,
    ) -> Vec<Self::PackedExt> {
        Self::evaluate_next_domain_packed(last_packed, fft, challenge_packed)
    }
}

// --- Short aliases ---
type PBF = PackedMamaBearAVX512;
type PEF3 = PackedMamaBearAVX512Ext3;

const ROUND0_PAIR_BYTES: usize = 2 * MamaBearScalar::SIZE;
const ROUND0_LEAF_HASH_TARGET_BYTES: usize = 32 * 1024;

#[inline(always)]
fn write_round0_pair_bytes(dst: &mut [u8], x: MamaBearScalar, nx: MamaBearScalar) {
    let x_bytes = x.0.to_le_bytes();
    let nx_bytes = nx.0.to_le_bytes();
    dst[..MamaBearScalar::SIZE].copy_from_slice(&x_bytes[..MamaBearScalar::SIZE]);
    dst[MamaBearScalar::SIZE..ROUND0_PAIR_BYTES].copy_from_slice(&nx_bytes[..MamaBearScalar::SIZE]);
}

#[inline(always)]
fn round0_leaf_hash_batch_size(leaf_bytes: usize, pairs_per_block: usize) -> usize {
    let target = (ROUND0_LEAF_HASH_TARGET_BYTES / leaf_bytes).max(1);
    let aligned = (target / pairs_per_block) * pairs_per_block;
    aligned.max(pairs_per_block)
}

fn round0_leaf_hashes_from_pair_major_values(
    values_mont: &[MamaBearScalar],
    leaf_size: usize,
) -> Vec<[u8; HASH_SIZE]> {
    assert_eq!(leaf_size % 2, 0, "round0 leaf size must contain x/nx pairs");

    let segment_count = leaf_size / 2;
    assert!(segment_count > 0, "round0 segment count must be non-zero");
    assert_eq!(values_mont.len() % segment_count, 0);

    let eval_len = values_mont.len() / segment_count;
    assert_eq!(eval_len % 2, 0, "round0 eval_len must be even");

    let leaf_count = eval_len / 2;
    let pairs_per_block = MamaBearFFT::pair_slots_per_block_for_pair_count(leaf_count);
    let leaf_bytes = leaf_size * MamaBearScalar::SIZE;
    let batch_leaf_count = round0_leaf_hash_batch_size(leaf_bytes, pairs_per_block);

    let mut batch_leaf_bytes = vec![0u8; batch_leaf_count * leaf_bytes];
    let mut leaf_hashes = vec![[0u8; HASH_SIZE]; leaf_count];

    for batch_start in (0..leaf_count).step_by(batch_leaf_count) {
        let active_leaves = (leaf_count - batch_start).min(batch_leaf_count);
        let active_bytes = active_leaves * leaf_bytes;

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

        blake3_batch::hash_leaves_batch_flat(
            &batch_leaf_bytes[..active_bytes],
            active_leaves,
            leaf_bytes,
            &mut leaf_hashes[batch_start..batch_start + active_leaves],
        );
    }

    leaf_hashes
}

fn round0_pair_major_merkle_tree_from_values(
    values_mont: &[MamaBearScalar],
    leaf_size: usize,
) -> MerkleTreeProverMB {
    let leaf_hashes = round0_leaf_hashes_from_pair_major_values(values_mont, leaf_size);
    MerkleTreeProverMB::from_leaf_hashes(&leaf_hashes)
}

// --- Ext3 implementation ---

impl DeepFoldExtField for MamaBearScalarExt3 {
    type PackedExt = PEF3;

    #[inline(always)]
    fn to_mont(self) -> Self {
        self.to_montgomery()
    }
    #[inline(always)]
    fn from_mont(self) -> Self {
        self.from_montgomery()
    }
    #[inline(always)]
    fn from_base_mont(base_mont: MamaBearScalar) -> Self {
        Self {
            c0: base_mont,
            c1: ZERO_MONT,
            c2: ZERO_MONT,
        }
    }

    /// Bulk broadcast base->Ext3: write [val, P, P] triples.
    fn from_base_mont_vec(src: &[MamaBearScalar]) -> Vec<Self> {
        let n = src.len();
        let mut dst = Vec::<Self>::with_capacity(n);
        unsafe {
            dst.set_len(n);
        }

        let src_ptr = src.as_ptr() as *const u64;
        let dst_ptr = dst.as_mut_ptr() as *mut u64;
        let p = P;

        for i in 0..n {
            unsafe {
                let d = dst_ptr.add(i * 3);
                *d = *src_ptr.add(i);
                *d.add(1) = p;
                *d.add(2) = p;
            }
        }
        dst
    }

    #[inline(always)]
    fn reduce_canonical(self) -> Self {
        Self {
            c0: self.c0.reduce(),
            c1: self.c1.reduce(),
            c2: self.c2.reduce(),
        }
    }

    #[inline(always)]
    fn packed_zero_mont() -> Self::PackedExt {
        PEF3::new(PBF::broadcast(P), PBF::broadcast(P), PBF::broadcast(P))
    }

    fn combine_opt_mont_base_to_ext_packed_mont(
        base_polys: &[&[MamaBearScalar]],
        r_mont: Self,
    ) -> Vec<Self::PackedExt> {
        combine_opt_ext3_inner::<true>(base_polys, r_mont)
    }

    fn combine_opt_normal_base_to_ext_packed_mont(
        base_polys: &[&[MamaBearScalar]],
        r_mont: Self,
    ) -> Vec<Self::PackedExt> {
        combine_opt_ext3_inner::<false>(base_polys, r_mont)
    }

    fn combine_opt_mont_base_to_ext_packed_mont_par(
        base_polys: &[&[MamaBearScalar]],
        r_mont: Self,
    ) -> Vec<Self::PackedExt> {
        crate::deepfold_mamabear_par::combine_opt_ext3_inner_par::<true>(base_polys, r_mont)
    }

    fn combine_opt_normal_base_to_ext_packed_mont_par(
        base_polys: &[&[MamaBearScalar]],
        r_mont: Self,
    ) -> Vec<Self::PackedExt> {
        crate::deepfold_mamabear_par::combine_opt_ext3_inner_par::<false>(base_polys, r_mont)
    }

    fn fold_sub_polys_packed_par(
        src: &mut Vec<Self::PackedExt>,
        dst: &mut Vec<Self::PackedExt>,
        eval_len_packed: usize,
        num_subs: usize,
        challenge_packed: Self::PackedExt,
    ) {
        crate::deepfold_mamabear_par::fold_sub_polys_packed_ext3_par(
            src,
            dst,
            eval_len_packed,
            num_subs,
            challenge_packed,
        );
    }

    fn fold_multilinear_packed_par(
        src: &mut Vec<Self::PackedExt>,
        dst: &mut Vec<Self::PackedExt>,
        challenge_packed: Self::PackedExt,
    ) {
        crate::deepfold_mamabear_par::fold_multilinear_packed_ext3_par(
            src,
            dst,
            challenge_packed,
        );
    }

    fn eval_multilinear_packed_par(
        poly_evals_packed: &[Self::PackedExt],
        point: &[Self],
    ) -> Self {
        crate::deepfold_mamabear_par::eval_multilinear_packed_ext3_par(
            poly_evals_packed,
            point,
        )
    }

    fn fold_multilinear_packed(
        poly_evals_packed: &mut Vec<Self::PackedExt>,
        challenge_packed: Self::PackedExt,
    ) {
        fold_multilinear_packed_ext3(poly_evals_packed, challenge_packed);
    }

    fn eval_multilinear_packed(poly_evals_packed: &[Self::PackedExt], point: &[Self]) -> Self {
        eval_multilinear_packed_ext3(poly_evals_packed, point)
    }

    fn fold_sub_polys_packed(
        sub_evals_packed: &mut [Self::PackedExt],
        eval_len_packed: usize,
        num_subs: usize,
        challenge_packed: Self::PackedExt,
    ) {
        let new_count = num_subs / 2;
        for m in 0..new_count {
            let src0 = 2 * m * eval_len_packed;
            let src1 = (2 * m + 1) * eval_len_packed;
            let dst = m * eval_len_packed;
            for j in 0..eval_len_packed {
                let v0 = sub_evals_packed[src0 + j];
                let v1 = sub_evals_packed[src1 + j];
                let diff = v1 - v0;
                sub_evals_packed[dst + j] = v0 + challenge_packed * diff;
            }
        }
    }

    fn evaluate_next_domain_first_round_packed(
        last_packed: &[Self::PackedExt],
        fft: &MamaBearFFT,
        challenge_packed: Self::PackedExt,
    ) -> Vec<Self::PackedExt> {
        evaluate_next_domain_first_round_packed_ext3(last_packed, fft, challenge_packed)
    }

    fn evaluate_next_domain_packed(
        last_packed: &[Self::PackedExt],
        fft: &MamaBearFFT,
        challenge_packed: Self::PackedExt,
    ) -> Vec<Self::PackedExt> {
        evaluate_next_domain_packed_ext3(last_packed, fft, challenge_packed)
    }

    fn evaluate_next_domain_first_round_packed_par(
        last_packed: &[Self::PackedExt],
        fft: &MamaBearFFT,
        challenge_packed: Self::PackedExt,
    ) -> Vec<Self::PackedExt> {
        crate::deepfold_mamabear_par::evaluate_next_domain_first_round_packed_ext3_par(
            last_packed,
            fft,
            challenge_packed,
        )
    }

    fn evaluate_next_domain_packed_par(
        last_packed: &[Self::PackedExt],
        fft: &MamaBearFFT,
        challenge_packed: Self::PackedExt,
    ) -> Vec<Self::PackedExt> {
        crate::deepfold_mamabear_par::evaluate_next_domain_packed_ext3_par(
            last_packed,
            fft,
            challenge_packed,
        )
    }
}

/// SIMD combine_opt kernel for Ext3. PEF3::mul_base_elem is 3 PBF mont_muls
/// (one per coefficient).
#[inline(always)]
pub(crate) fn combine_opt_ext3_inner<const BASE_IS_MONT: bool>(
    base_polys: &[&[MamaBearScalar]],
    r_mont: MamaBearScalarExt3,
) -> Vec<PEF3> {
    let n_polys = base_polys.len();
    assert!(n_polys >= 1);
    let len = base_polys[0].len();
    debug_assert!(base_polys.iter().all(|p| p.len() == len));
    debug_assert_eq!(len % 8, 0, "combine_opt requires 8-aligned input length");

    let chunks_packed = len / 8;
    let mut out_packed = Vec::<PEF3>::with_capacity(chunks_packed);

    // powers[i] = r^(n-1-i): descending Horner so the i-th input poly multiplies
    // by `r^(n-1-i)`, matching the original scalar Horner and the verifier's
    // descending eval combine.
    let mut ascending: Vec<MamaBearScalarExt3> = Vec::with_capacity(n_polys);
    ascending.push(MamaBearScalarExt3::from(MamaBearScalar(1)).to_montgomery());
    for i in 1..n_polys {
        let next = ascending[i - 1] * r_mont;
        ascending.push(next);
    }
    let mut powers_scalar: Vec<MamaBearScalarExt3> = ascending.into_iter().rev().collect();
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

    for chunk in 0..chunks_packed {
        let off = chunk * 8;

        let mut acc = PEF3::new(zero_mont_pbf, zero_mont_pbf, zero_mont_pbf);

        // 4-way ILP unroll with tree-reduced adds to keep partial sums packed
        // longer before the periodic reduce_fast.
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

        out_packed.push(acc.reduce_fast());
    }

    out_packed
}

/// Test-accessible wrapper for `combine_opt_ext3_inner`.
#[cfg(test)]
pub fn combine_opt_ext3_inner_test<const BASE_IS_MONT: bool>(
    base_polys: &[&[MamaBearScalar]],
    r_mont: MamaBearScalarExt3,
) -> Vec<PEF3> {
    combine_opt_ext3_inner::<BASE_IS_MONT>(base_polys, r_mont)
}

/// In-place packed multilinear fold for Ext3. Requires `>= 2` packed blocks.
pub(crate) fn fold_multilinear_packed_ext3(poly_evals_packed: &mut Vec<PEF3>, challenge_packed: PEF3) {
    let new_len = poly_evals_packed.len() / 2;
    debug_assert!(
        new_len >= 1,
        "fold_multilinear_packed requires >= 2 packed blocks"
    );

    const EVEN_IDX: [u64; 8] = [0, 2, 4, 6, 8, 10, 12, 14];
    const ODD_IDX: [u64; 8] = [1, 3, 5, 7, 9, 11, 13, 15];

    for i in 0..new_len {
        let lo = poly_evals_packed[2 * i];
        let hi = poly_evals_packed[2 * i + 1];

        let even_c0 = lo.c0.permute2(hi.c0, EVEN_IDX);
        let even_c1 = lo.c1.permute2(hi.c1, EVEN_IDX);
        let even_c2 = lo.c2.permute2(hi.c2, EVEN_IDX);
        let odd_c0 = lo.c0.permute2(hi.c0, ODD_IDX);
        let odd_c1 = lo.c1.permute2(hi.c1, ODD_IDX);
        let odd_c2 = lo.c2.permute2(hi.c2, ODD_IDX);

        let evens = PEF3::new(even_c0, even_c1, even_c2);
        let odds = PEF3::new(odd_c0, odd_c1, odd_c2);

        let diff = odds - evens;
        let folded = evens + challenge_packed * diff;
        poly_evals_packed[i] = folded;
    }
    poly_evals_packed.truncate(new_len);
}

/// Out-of-place packed multilinear fold for Ext3. Used by
/// `eval_multilinear_packed_ext3` for the first round.
pub(crate) fn fold_multilinear_packed_ext3_out_of_place(
    src: &[PEF3],
    dst: &mut [PEF3],
    challenge_packed: PEF3,
) {
    debug_assert!(src.len() >= 2);
    debug_assert_eq!(dst.len(), src.len() / 2);

    const EVEN_IDX: [u64; 8] = [0, 2, 4, 6, 8, 10, 12, 14];
    const ODD_IDX: [u64; 8] = [1, 3, 5, 7, 9, 11, 13, 15];

    for i in 0..dst.len() {
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
        let folded = evens + challenge_packed * diff;
        dst[i] = folded;
    }
}

pub(crate) fn eval_multilinear_packed_ext3(
    poly_evals_packed: &[PEF3],
    point: &[MamaBearScalarExt3],
) -> MamaBearScalarExt3 {
    let mut idx = 0;

    // Single packed block: skip the round loop.
    if poly_evals_packed.len() < 2 {
        let mut tail = [MamaBearScalarExt3::default(); 8];
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

    // First round: out-of-place into a fresh half-size buffer. Avoids the
    // full `to_vec` clone of `poly_evals_packed`. PEF3 does not impl IsZero,
    // so use `with_capacity + set_len`; the writer below covers every slot.
    let first_new_len = poly_evals_packed.len() / 2;
    let mut scratch: Vec<PEF3> = Vec::with_capacity(first_new_len);
    unsafe {
        scratch.set_len(first_new_len);
    }
    let r0 = point[idx];
    let r0_packed = PEF3::new(PBF::from(r0.c0.0), PBF::from(r0.c1.0), PBF::from(r0.c2.0));
    fold_multilinear_packed_ext3_out_of_place(poly_evals_packed, &mut scratch, r0_packed);
    idx += 1;

    while scratch.len() >= 2 {
        let r = point[idx];
        let r_packed = PEF3::new(PBF::from(r.c0.0), PBF::from(r.c1.0), PBF::from(r.c2.0));
        fold_multilinear_packed_ext3(&mut scratch, r_packed);
        idx += 1;
    }
    let mut tail = [MamaBearScalarExt3::default(); 8];
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

pub(crate) fn evaluate_next_domain_first_round_packed_ext3(
    last_packed: &[PEF3],
    fft: &MamaBearFFT,
    challenge_packed: PEF3,
) -> Vec<PEF3> {
    let len = fft.size();
    let pair_count = len / 2;
    debug_assert_eq!(last_packed.len(), len / 8);

    let inv_2_mont = MamaBearScalar::inv_2().to_montgomery();
    let inv_2_packed = PBF::from(inv_2_mont.0);

    let packed_pairs = pair_count / 8;
    let mut result_packed = Vec::with_capacity(packed_pairs);

    for chunk in 0..packed_pairs {
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
        result_packed.push(PEF3::new(
            scaled.c0.con_sub_xp(1),
            scaled.c1.con_sub_xp(1),
            scaled.c2.con_sub_xp(1),
        ));
    }

    debug_assert_eq!(pair_count % 8, 0);
    result_packed
}

pub(crate) fn evaluate_next_domain_packed_ext3(
    last_packed: &[PEF3],
    fft: &MamaBearFFT,
    challenge_packed: PEF3,
) -> Vec<PEF3> {
    let len = fft.size();
    let pair_count = len / 2;
    debug_assert_eq!(last_packed.len(), len / 8);

    let inv_2_mont = MamaBearScalar::inv_2().to_montgomery();
    let inv_2_packed = PBF::from(inv_2_mont.0);

    let packed_pairs = pair_count / 8;
    let mut result_packed = Vec::with_capacity(packed_pairs);

    const EVEN_IDX: [u64; 8] = [0, 2, 4, 6, 8, 10, 12, 14];
    const ODD_IDX: [u64; 8] = [1, 3, 5, 7, 9, 11, 13, 15];

    for chunk in 0..packed_pairs {
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
        result_packed.push(PEF3::new(
            scaled.c0.con_sub_xp(1),
            scaled.c1.con_sub_xp(1),
            scaled.c2.con_sub_xp(1),
        ));
    }

    debug_assert_eq!(pair_count % 8, 0);
    result_packed
}

// --- Prover ---

#[derive(Clone)]
pub struct DeepFoldMamaBearProver<F: DeepFoldExtField> {
    pub interpolation: InterpolateValueMB<MamaBearScalar>,
    /// Sub-polynomial FFT outputs in canonical Montgomery form (base field).
    /// Layout: round-0 FFT pair-major blocked storage, one full eval_len segment per sub-FFT.
    /// Shared with `interpolation.value` so round-0 query does not duplicate storage.
    /// Stored as `Arc<Vec<...>>` (instead of `Arc<[...]>`) so `Arc::new(vec)` is a
    /// zero-copy move, avoiding the per-construction `total_elems * 8B` memcpy.
    pub(crate) sub_evals_mont: Arc<Vec<MamaBearScalar>>,
    pub(crate) poly: Vec<Vec<MamaBearScalar>>,
    pub(crate) _phantom: std::marker::PhantomData<F>,
}

impl<F: DeepFoldExtField> DeepFoldMamaBearProver<F> {
    /// Create a new prover. Splits coefficients into 2^t sub-polynomials and FFTs each.
    ///
    /// Optimizations vs naive approach:
    /// - Stores FFT outputs in Montgomery form (`sub_evals_mont`) for `open` to use directly
    /// - Builds round-0 leaf hashes directly from pair-major FFT storage, without a full leaf buffer
    pub fn new(pp: &DeepFoldMamaBearParam, poly: &[&[MamaBearScalar]]) -> Self {
        let fft = &pp.fft_groups[0];
        let sub_count = 1usize << pp.split_level; // e.g., 8 for split_level=3
        let eval_len = fft.size(); // 2^{logn - split_level}, where logn=nv+code_rate_log
        let leaf_size = 2 * sub_count * poly.len(); // 2: (x, nx) pairs
        let total_elems = sub_count * eval_len * poly.len(); // logn * 3 for three witness polys

        // Single allocation; `fft_into` writes directly into the destination slot.
        // Uninit alloc: every slot is fully written by `fft_into` below. This
        // avoids the IsZero slow path (MamaBearScalar does not impl the std
        // internal IsZero trait), which would otherwise pay a single-threaded
        // fill loop + serial first-touch page faults for the full `total_elems`.
        let mut sub_evals_mont: Vec<MamaBearScalar> = Vec::with_capacity(total_elems);
        unsafe { sub_evals_mont.set_len(total_elems); }

        // Reusable strided-gather scratch. Loop below always writes `len <= eval_len`
        // slots before reading them, so uninit is safe here too.
        let mut sub_coeffs: Vec<MamaBearScalar> = Vec::with_capacity(eval_len);
        unsafe { sub_coeffs.set_len(eval_len); }

        let mut write_offset = 0usize;
        for coeffs in poly.iter() {
            for k in 0..sub_count {
                // Split coefficients: sub_k[i] = coeffs[k + i * 2^t]
                // Strided gather with sequential writes to reused buffer.
                let mut len = 0usize;
                let mut idx = k;
                while idx < coeffs.len() {
                    sub_coeffs[len] = coeffs[idx];
                    len += 1;
                    idx += sub_count;
                }

                // FFT directly into the next slot of `sub_evals_mont`.
                fft.fft_into(
                    &sub_coeffs[..len],
                    &mut sub_evals_mont[write_offset..write_offset + eval_len],
                );
                write_offset += eval_len;
            }
        }

        // Arc::new is a zero-copy move (no fresh slice allocation + memcpy that
        // `Arc::<[T]>::from(vec)` would do).
        let sub_evals_mont: Arc<Vec<MamaBearScalar>> = Arc::new(sub_evals_mont);
        let interpolation =
            InterpolateValueMB::from_fft_pair_major_parts(sub_evals_mont.clone(), leaf_size);

        DeepFoldMamaBearProver {
            interpolation,
            sub_evals_mont,
            poly: poly.iter().map(|p| p.to_vec()).collect(),
            _phantom: std::marker::PhantomData,
        }
    }

    /// Profiling variant of `new`: same result, but accumulates per-substage wall-clock
    /// time into `timings`. Mirrors the body of `new` exactly — any divergence is a bug.
    pub fn new_profiled(
        pp: &DeepFoldMamaBearParam,
        poly: &[&[MamaBearScalar]],
        timings: &mut NewTimings,
    ) -> Self {
        use std::time::Instant;

        let total_start = Instant::now();

        let fft = &pp.fft_groups[0];
        let sub_count = 1usize << pp.split_level;
        let eval_len = fft.size();
        let leaf_size = 2 * sub_count * poly.len();
        let total_elems = sub_count * eval_len * poly.len();

        let t0 = Instant::now();
        // Uninit alloc: mirrors `new()`. See the detailed comment there for why
        // this is both safe (every slot is overwritten) and materially faster
        // than `vec![MamaBearScalar(0); ..]` on this newtype.
        let mut sub_evals_mont: Vec<MamaBearScalar> = Vec::with_capacity(total_elems);
        unsafe { sub_evals_mont.set_len(total_elems); }
        let mut sub_coeffs: Vec<MamaBearScalar> = Vec::with_capacity(eval_len);
        unsafe { sub_coeffs.set_len(eval_len); }
        timings.alloc_us += t0.elapsed().as_micros();

        let mut write_offset = 0usize;
        for coeffs in poly.iter() {
            for k in 0..sub_count {
                let t_split = Instant::now();
                let mut len = 0usize;
                let mut idx = k;
                while idx < coeffs.len() {
                    sub_coeffs[len] = coeffs[idx];
                    len += 1;
                    idx += sub_count;
                }
                timings.split_us += t_split.elapsed().as_micros();

                let t_fft = Instant::now();
                fft.fft_into(
                    &sub_coeffs[..len],
                    &mut sub_evals_mont[write_offset..write_offset + eval_len],
                );
                timings.fft_us += t_fft.elapsed().as_micros();

                write_offset += eval_len;
                // `append` substage no longer exists; the FFT writes directly into place.
            }
        }

        let t_arc = Instant::now();
        let sub_evals_mont: Arc<Vec<MamaBearScalar>> = Arc::new(sub_evals_mont);
        timings.arc_convert_us += t_arc.elapsed().as_micros();

        let t_leaf = Instant::now();
        let leaf_hashes = round0_leaf_hashes_from_pair_major_values(&sub_evals_mont, leaf_size);
        timings.leaf_hash_us += t_leaf.elapsed().as_micros();

        let t_tree = Instant::now();
        let merkle_tree = MerkleTreeProverMB::from_leaf_hashes(&leaf_hashes);
        timings.merkle_tree_us += t_tree.elapsed().as_micros();

        let t_wrap = Instant::now();
        let interpolation = InterpolateValueMB::from_fft_pair_major_parts_with_tree(
            sub_evals_mont.clone(),
            leaf_size,
            merkle_tree,
        );
        let prover = DeepFoldMamaBearProver {
            interpolation,
            sub_evals_mont,
            poly: poly.iter().map(|p| p.to_vec()).collect(),
            _phantom: std::marker::PhantomData,
        };
        timings.wrap_us += t_wrap.elapsed().as_micros();

        timings.total_us += total_start.elapsed().as_micros();
        prover
    }

    pub fn commit(&self) -> MerkleRoot {
        MerkleRoot(self.interpolation.commit())
    }

    pub fn open(
        pp: &DeepFoldMamaBearParam,
        provers: &[&Self],
        point_mont: Vec<F>,
        transcript: &mut Transcript,
    ) {
        let mut t = OpenTimings::default();
        Self::open_inner(pp, provers, point_mont, transcript, &mut t, false);
    }

    /// Profiling variant of `open`. Mirrors `open` and accumulates per-substage
    /// wall-clock time into `timings`. The `Instant::now()` calls add a small
    /// nonzero overhead — only use for profiling.
    pub fn open_profiled(
        pp: &DeepFoldMamaBearParam,
        provers: &[&Self],
        point_mont: Vec<F>,
        transcript: &mut Transcript,
        timings: &mut OpenTimings,
    ) {
        Self::open_inner(pp, provers, point_mont, transcript, timings, true);
    }

    /// Parallel open — uses parallel combine for combine_polys and combine_subs.
    /// The FRI fold loop remains serial (Fiat-Shamir dependency).
    pub fn open_par(
        pp: &DeepFoldMamaBearParam,
        provers: &[&Self],
        point_mont: Vec<F>,
        transcript: &mut Transcript,
    ) {
        let mut t = OpenTimings::default();
        Self::open_inner_par(pp, provers, point_mont, transcript, &mut t, false);
    }

    /// Profiling variant of `open_par`.
    pub fn open_par_profiled(
        pp: &DeepFoldMamaBearParam,
        provers: &[&Self],
        point_mont: Vec<F>,
        transcript: &mut Transcript,
        timings: &mut OpenTimings,
    ) {
        Self::open_inner_par(pp, provers, point_mont, transcript, timings, true);
    }

    /// Shared open implementation. When `record == true`, accumulates per-substage
    /// timings; when false, the time-tracking calls become near no-ops. The two
    /// public entry points (`open`, `open_profiled`) only differ in this flag.
    fn open_inner(
        pp: &DeepFoldMamaBearParam,
        provers: &[&Self],
        point_mont: Vec<F>,
        transcript: &mut Transcript,
        timings: &mut OpenTimings,
        record: bool,
    ) {
        // Helper macro: only call Instant::now() / record when timing is on,
        // so the no-record path stays free of syscall overhead.
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
        let eval_len = pp.fft_groups[0].size();

        let r_raw: F = transcript.challenge_f();
        let r_mont = r_raw.to_mont();

        // -- combine_polys --
        let t0 = now!();
        let poly_slices: Vec<&[MamaBearScalar]> = provers
            .iter()
            .flat_map(|p| p.poly.iter().map(|v| v.as_slice()))
            .collect();
        let poly_evals_packed: Vec<F::PackedExt> =
            F::combine_opt_normal_base_to_ext_packed_mont(&poly_slices, r_mont);
        tick!(t0, combine_polys_us);

        // -- combine_subs --
        let chunk_len = sub_count * eval_len;
        let t0 = now!();
        let sub_slices: Vec<&[MamaBearScalar]> = provers
            .iter()
            .flat_map(|prover| {
                (0..prover.poly.len()).map(move |poly_idx| {
                    let start = poly_idx * chunk_len;
                    &prover.sub_evals_mont[start..start + chunk_len]
                })
            })
            .collect();
        let sub_evals_packed: Vec<F::PackedExt> =
            F::combine_opt_mont_base_to_ext_packed_mont(&sub_slices, r_mont);
        tick!(t0, combine_subs_us);

        // -- main fold loop + grind + sample query indices (extracted) --
        let (mut leaf_indices, fri_results) = open_main_fold_after_combine::<F>(
            pp,
            poly_evals_packed,
            sub_evals_packed,
            &point_mont,
            transcript,
            timings,
            record,
        );

        // -- query phase --
        let t0 = now!();
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
        tick!(t0, query_phase_us);

        if let Some(t) = total_t0 {
            timings.total_us += t.elapsed().as_micros();
        }
    }

    /// Parallel open implementation — same as `open_inner` but calls
    /// `_par` combine methods for combine_polys and combine_subs.
    fn open_inner_par(
        pp: &DeepFoldMamaBearParam,
        provers: &[&Self],
        point_mont: Vec<F>,
        transcript: &mut Transcript,
        timings: &mut OpenTimings,
        record: bool,
    ) {
        macro_rules! now {
            () => {
                if record { Some(Instant::now()) } else { None }
            };
        }
        macro_rules! tick {
            ($t:ident, $field:ident) => {
                if let Some(t) = $t { timings.$field += t.elapsed().as_micros(); }
            };
        }

        let total_t0 = now!();
        let split_level = pp.split_level;
        let sub_count = 1usize << split_level;
        let eval_len = pp.fft_groups[0].size();

        let r_raw: F = transcript.challenge_f();
        let r_mont = r_raw.to_mont();

        // -- combine_polys (PARALLEL) --
        let t0 = now!();
        let poly_slices: Vec<&[MamaBearScalar]> = provers
            .iter()
            .flat_map(|p| p.poly.iter().map(|v| v.as_slice()))
            .collect();
        let poly_evals_packed: Vec<F::PackedExt> =
            F::combine_opt_normal_base_to_ext_packed_mont_par(&poly_slices, r_mont);
        tick!(t0, combine_polys_us);

        // -- combine_subs (PARALLEL) --
        let chunk_len = sub_count * eval_len;
        let t0 = now!();
        let sub_slices: Vec<&[MamaBearScalar]> = provers
            .iter()
            .flat_map(|prover| {
                (0..prover.poly.len()).map(move |poly_idx| {
                    let start = poly_idx * chunk_len;
                    &prover.sub_evals_mont[start..start + chunk_len]
                })
            })
            .collect();
        let sub_evals_packed: Vec<F::PackedExt> =
            F::combine_opt_mont_base_to_ext_packed_mont_par(&sub_slices, r_mont);
        tick!(t0, combine_subs_us);

        // -- main fold loop + grind + sample query indices (extracted) --
        let (mut leaf_indices, fri_results) = open_main_fold_after_combine_par::<F>(
            pp,
            poly_evals_packed,
            sub_evals_packed,
            &point_mont,
            transcript,
            timings,
            record,
        );

        // -- query phase --
        let t0 = now!();
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
        tick!(t0, query_phase_us);

        if let Some(t) = total_t0 {
            timings.total_us += t.elapsed().as_micros();
        }
    }
}

/// Run the post-combine portion of `open_inner` (multilinear fold + split
/// fold + FRI fold + grinding + sample query indices). Shared between the
/// base `open_inner` and the ext-field
/// `DeepFoldMamaBearProverExt::open_inner` (in `deepfold_mamabear_ext.rs`).
///
/// Inputs: `poly_evals_packed` and `sub_evals_packed` are the post-combine
/// ext-packed buffers produced by the appropriate `combine_*` kernel
/// (base->ext for `DeepFoldMamaBearProver`, ext->ext for the ext prover).
///
/// Returns: `(leaf_indices, fri_results)`. `leaf_indices` are the round-0
/// query indices sampled from the transcript after grinding; the caller
/// must drive the actual leaf-byte query via its own
/// `interpolation.query(...)` and the FRI sibling-query loop (since round-0
/// leaf bytes differ per element-type byte width and FRI results are owned
/// here for the caller to read).
///
/// This extraction is the **single allowed structural change** to
/// `deepfold_mamabear.rs` per HC-3 -- a pure refactor of `open_inner`.
/// LLVM inlines this back at -Copt-level=3, so base machine code is
/// unchanged. Verified via cargo test (no regression in any existing test).
// ===================== DeepFold DEEP (out-of-domain) binding =====================
//
// The bare Merkle root does not bind the committed vector to a unique polynomial in
// the list-decoding regime. DeepFold adds an out-of-domain challenge `alpha` and the
// evaluation `c = f^(0)(alpha)`, plus a fresh per-round point `alpha_i`, so that (Thm 4,
// Lemma 7) two distinct list-decoded codewords cannot survive. We keep `c` and all
// per-round DEEP elements in the EVALUATION PROOF (not in the commitment struct): since
// `alpha` is Fiat-Shamir-derived from the already-absorbed `rt_0`, binding only needs
// the committed vector fixed before `alpha`.
//
// These three helpers are shared verbatim by the serial and parallel open kernels; the
// ONLY difference is the `packed_eval` closure (`eval_multilinear_packed` vs its `_par`
// twin, which is byte-identical by its serial-fallback contract), so the two opens stay
// bit-for-bit equal (the R7 byte-identity rule).

/// Evaluate the current combined multilinear at `pt`, dispatching between the packed
/// representation and the post-transition scalar tail exactly as the original z-line did.
#[inline]
fn deep_eval_dispatch<F: DeepFoldExtField>(
    poly_evals_tail: &Option<[F; 8]>,
    poly_evals_tail_len: usize,
    pt: &[F],
    packed_eval: &impl Fn(&[F]) -> F,
) -> F {
    if let Some(ref tail) = poly_evals_tail {
        eval_multilinear_scalar_tail(&tail[..poly_evals_tail_len], pt)
    } else {
        packed_eval(pt)
    }
}

// Test-only forge hook for the DeepFold OOD (out-of-domain) `c = f^(0)(alpha)`
// claim. When set nonzero, `deep_open_init` appends `c' = c + DEEP_C_OFFSET`
// (Montgomery form) instead of the honest `c`, decoupling the OOD claim from the
// committed polynomial. Used by `deepfold_mamabear_rejects_decoupled_deep_c_ext3`
// to prove the DEEP terminal check actually binds. Zero (honest) in every
// non-PoC path, and BOTH the thread-local and its read below are
// `#[cfg(test)]`-gated, so they vanish entirely from non-test builds (ZERO
// production impact). Mirrors the Goldilocks-generic `deep_c_offset` param in
// `crate::deepfold::open_inner`.
#[cfg(test)]
thread_local! {
    pub(crate) static DEEP_C_OFFSET: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// Prover setup: draw the OOD challenge alpha, append c = f^(0)(alpha), and return the
/// initial DEEP point set [z, alpha_vec]. `alpha`/`alpha_vec` are Montgomery form (so the
/// packed/tail eval helpers consume them directly). `poly_evals_tail` is None here (the
/// full poly), so `c` is evaluated via `packed_eval`.
#[inline]
fn deep_open_init<F: DeepFoldExtField>(
    variable_num: usize,
    point_mont: &[F],
    poly_evals_tail: &Option<[F; 8]>,
    poly_evals_tail_len: usize,
    transcript: &mut Transcript,
    packed_eval: &impl Fn(&[F]) -> F,
) -> Vec<Vec<F>> {
    let alpha = transcript.challenge_f::<F>().to_mont();
    let alpha_vec = crate::deepfold::deep_power_vector(alpha, variable_num);
    let c = deep_eval_dispatch(poly_evals_tail, poly_evals_tail_len, &alpha_vec, packed_eval);
    // DeepFold OOD-c soundness PoC hook: in `#[cfg(test)]` builds a nonzero
    // `DEEP_C_OFFSET` forges c' = c + offset (Montgomery form) — a
    // transcript-consistent but decoupled OOD claim the verifier's DEEP terminal
    // check must reject. Compiled out entirely in non-test builds.
    #[cfg(test)]
    let c = {
        let off = DEEP_C_OFFSET.with(|v| v.get());
        if off != 0 {
            c + F::from(MamaBearScalar(off)).to_mont()
        } else {
            c
        }
    };
    transcript.append_f(c.reduce_canonical());
    vec![point_mont.to_vec(), alpha_vec]
}

/// Prover per-round sends. (a) For every round but the last, introduce the fresh point
/// alpha_i and send its seed f^(i)(alpha_i) (the DEEP evaluation the protocol adds each
/// round; it has no carried claim, so it is sent explicitly). (b) For every active point
/// (z first, then alpha, then the fresh alpha_j's) send the line value at head+1; with the
/// carried claim at `head` this fixes the round's degree-1 line. Head-drop is done by the
/// caller after the fold.
#[inline]
fn deep_open_round<F: DeepFoldExtField>(
    i: usize,
    variable_num: usize,
    active: &mut Vec<Vec<F>>,
    poly_evals_tail: &Option<[F; 8]>,
    poly_evals_tail_len: usize,
    one_mont: F,
    transcript: &mut Transcript,
    packed_eval: &impl Fn(&[F]) -> F,
) {
    if i < variable_num - 1 {
        let alpha_i = transcript.challenge_f::<F>().to_mont();
        let w_i = crate::deepfold::deep_power_vector(alpha_i, variable_num - i);
        let seed = deep_eval_dispatch(poly_evals_tail, poly_evals_tail_len, &w_i, packed_eval);
        transcript.append_f(seed.reduce_canonical());
        active.push(w_i);
    }
    for w in active.iter() {
        let mut off = w.clone();
        off[0] = off[0] + one_mont;
        let e = deep_eval_dispatch(poly_evals_tail, poly_evals_tail_len, &off, packed_eval);
        transcript.append_f(e.reduce_canonical());
    }
}

pub(crate) fn open_main_fold_after_combine<F: DeepFoldExtField>(
    pp: &DeepFoldMamaBearParam,
    mut poly_evals_packed: Vec<F::PackedExt>,
    mut sub_evals_packed: Vec<F::PackedExt>,
    point_mont: &[F],
    transcript: &mut Transcript,
    timings: &mut OpenTimings,
    record: bool,
) -> (Vec<usize>, Vec<FriFoldResult<F>>) {
    macro_rules! now {
        () => {
            if record { Some(Instant::now()) } else { None }
        };
    }
    macro_rules! tick {
        ($t:ident, $field:ident) => {
            if let Some(t) = $t {
                timings.$field += t.elapsed().as_micros();
            }
        };
    }

    let split_level = pp.split_level;
    let sub_count = 1usize << split_level;
    let eval_len = pp.fft_groups[0].size();
    let eval_len_packed = eval_len / 8;

    let mut fri_results: Vec<FriFoldResult<F>> = vec![];
    let one_mont = F::from(MamaBearScalar(1)).to_mont();

    let mut poly_evals_logical_len = poly_evals_packed.len() * 8;
    let mut poly_evals_tail: Option<[F; 8]> = None;
    let mut poly_evals_tail_len: usize = 0;

    // DeepFold OOD binding — draw alpha, append c = f^(0)(alpha), seed the DEEP point
    // set [z, alpha_vec]. The z-point is active[0] (the old z-line); alpha is active[1].
    let t_init = now!();
    let mut active = deep_open_init(
        pp.variable_num,
        point_mont,
        &poly_evals_tail,
        poly_evals_tail_len,
        transcript,
        &|pt| F::eval_multilinear_packed(&poly_evals_packed, pt),
    );
    tick!(t_init, mle_eval_us);

    // -- main fold loop (mirrors base open_inner exactly) --
    let mut current_sub_count = sub_count;
    for i in 0..pp.variable_num {
        // DeepFold OOD binding — per-round sends for the growing point set: the fresh
        // alpha_i seed (all but the last round) plus one off-value per active point. This
        // replaces the single old z-line send; the z-point's off (active[0]) is
        // byte-identical to the old z-line element.
        let t0 = now!();
        deep_open_round(
            i,
            pp.variable_num,
            &mut active,
            &poly_evals_tail,
            poly_evals_tail_len,
            one_mont,
            transcript,
            &|pt| F::eval_multilinear_packed(&poly_evals_packed, pt),
        );
        tick!(t0, mle_eval_us);

        let challenge_raw: F = transcript.challenge_f();
        let challenge_mont = challenge_raw.to_mont();
        let challenge_packed =
            <F::PackedExt as PackedExtensionField>::broadcast_scalar(challenge_mont);

        let t0 = now!();
        if poly_evals_tail.is_some() {
            let tail = poly_evals_tail.as_mut().unwrap();
            let new_len = poly_evals_tail_len / 2;
            for j in 0..new_len {
                let v0 = tail[2 * j];
                let v1 = tail[2 * j + 1];
                tail[j] = v0 + challenge_mont * (v1 - v0);
            }
            poly_evals_tail_len = new_len;
            poly_evals_logical_len = new_len;
        } else if poly_evals_packed.len() >= 2 {
            F::fold_multilinear_packed(&mut poly_evals_packed, challenge_packed);
            poly_evals_logical_len /= 2;
        } else {
            debug_assert_eq!(poly_evals_packed.len(), 1);
            debug_assert_eq!(poly_evals_logical_len, 8);
            let mut tail = [F::default(); 8];
            poly_evals_packed[0].unpack_into_slice(&mut tail);
            let new_len = 4;
            for j in 0..new_len {
                let v0 = tail[2 * j];
                let v1 = tail[2 * j + 1];
                tail[j] = v0 + challenge_mont * (v1 - v0);
            }
            poly_evals_tail = Some(tail);
            poly_evals_tail_len = new_len;
            poly_evals_logical_len = new_len;
        }
        tick!(t0, multilin_fold_us);

        if i < split_level {
            let t0 = now!();
            F::fold_sub_polys_packed(
                &mut sub_evals_packed,
                eval_len_packed,
                current_sub_count,
                challenge_packed,
            );
            current_sub_count /= 2;
            tick!(t0, split_fold_us);
        } else {
            let fft_idx = i - split_level;
            let last_packed: &[F::PackedExt] = if fft_idx == 0 {
                &sub_evals_packed[0..eval_len_packed]
            } else {
                &fri_results[fft_idx - 1].value_packed
            };

            let t0 = now!();
            let next_packed: Vec<F::PackedExt> = if fft_idx == 0 {
                F::evaluate_next_domain_first_round_packed(
                    last_packed,
                    &pp.fft_groups[fft_idx],
                    challenge_packed,
                )
            } else {
                F::evaluate_next_domain_packed(
                    last_packed,
                    &pp.fft_groups[fft_idx],
                    challenge_packed,
                )
            };
            tick!(t0, fri_fold_us);

            let t0 = now!();
            if i < pp.variable_num - 1 {
                let fri_result = FriFoldResult::from_packed_values(next_packed);
                transcript.append_u8_slice(&fri_result.commit(), HASH_SIZE);
                fri_results.push(fri_result);
            } else {
                let mut tail = [F::default(); 8];
                next_packed[0].unpack_into_slice(&mut tail);
                transcript.append_f(tail[0].reduce_canonical());
            }
            tick!(t0, fri_merkle_us);
        }

        // DeepFold OOD binding — head-drop: variable i is folded into r_i, so every active
        // point loses its leading coordinate (its claim moves onto the shorter tail).
        for w in active.iter_mut() {
            w.remove(0);
        }
    }
    let _ = poly_evals_logical_len; // silence warning when not in tail mode

    // -- grind + sample query indices --
    transcript.grind(pp.grinding_bits);
    let mut leaf_indices = transcript.challenge_usizes(pp.query_num);
    let fat_domain = pp.fft_groups[0].size();
    leaf_indices = leaf_indices.iter_mut().map(|v| *v % (fat_domain >> 1)).collect();
    leaf_indices.sort();
    leaf_indices.dedup();

    (leaf_indices, fri_results)
}

/// Parallel companion of `open_main_fold_after_combine`. Runs the post-combine
/// fold + FRI + grinding portion of `open_inner_par` using the rayon-aware
/// trait variants (`F::eval_multilinear_packed_par`, `fold_multilinear_packed_par`,
/// `fold_sub_polys_packed_par`, `FriFoldResult::from_packed_values_par`,
/// `transcript.grind_par`). Allocates the two ping-pong scratch buffers
/// (poly_evals_scratch, sub_evals_scratch) internally to amortise their
/// first-touch cost across all rounds (critical for the parallel
/// fold_multilinear_packed_par 4-14x speedup, see comment on the inline
/// version in `open_inner_par`).
///
/// Bit-identical transcript output to the serial `open_main_fold_after_combine`.
pub(crate) fn open_main_fold_after_combine_par<F: DeepFoldExtField>(
    pp: &DeepFoldMamaBearParam,
    mut poly_evals_packed: Vec<F::PackedExt>,
    mut sub_evals_packed: Vec<F::PackedExt>,
    point_mont: &[F],
    transcript: &mut Transcript,
    timings: &mut OpenTimings,
    record: bool,
) -> (Vec<usize>, Vec<FriFoldResult<F>>) {
    macro_rules! now {
        () => {
            if record { Some(Instant::now()) } else { None }
        };
    }
    macro_rules! tick {
        ($t:ident, $field:ident) => {
            if let Some(t) = $t {
                timings.$field += t.elapsed().as_micros();
            }
        };
    }

    let split_level = pp.split_level;
    let sub_count = 1usize << split_level;
    let eval_len = pp.fft_groups[0].size();
    let eval_len_packed = eval_len / 8;

    let mut fri_results: Vec<FriFoldResult<F>> = vec![];
    let one_mont = F::from(MamaBearScalar(1)).to_mont();

    let mut poly_evals_logical_len = poly_evals_packed.len() * 8;
    let mut poly_evals_tail: Option<[F; 8]> = None;
    let mut poly_evals_tail_len: usize = 0;

    // Pre-allocate ping-pong scratch buffers (see `open_inner_par`'s inline
    // version for the allocation rationale: amortises mmap + first-touch cost
    // across all fold rounds).
    let mut sub_evals_scratch: Vec<F::PackedExt> = Vec::with_capacity(sub_evals_packed.len());
    unsafe {
        sub_evals_scratch.set_len(sub_evals_packed.len());
    }
    let mut poly_evals_scratch: Vec<F::PackedExt> = Vec::with_capacity(poly_evals_packed.len());
    unsafe {
        poly_evals_scratch.set_len(poly_evals_packed.len());
    }

    // DeepFold OOD binding — draw alpha, append c = f^(0)(alpha), seed the DEEP point
    // set [z, alpha_vec]. Byte-identical to the serial open by the R7 rule.
    let t_init = now!();
    let mut active = deep_open_init(
        pp.variable_num,
        point_mont,
        &poly_evals_tail,
        poly_evals_tail_len,
        transcript,
        &|pt| F::eval_multilinear_packed_par(&poly_evals_packed, pt),
    );
    tick!(t_init, mle_eval_us);

    let mut current_sub_count = sub_count;
    for i in 0..pp.variable_num {
        // DeepFold OOD binding — per-round sends (fresh alpha_i seed + one off per active
        // point). Uses the _par packed eval; byte-identical to serial by the R7 rule.
        let t0 = now!();
        deep_open_round(
            i,
            pp.variable_num,
            &mut active,
            &poly_evals_tail,
            poly_evals_tail_len,
            one_mont,
            transcript,
            &|pt| F::eval_multilinear_packed_par(&poly_evals_packed, pt),
        );
        tick!(t0, mle_eval_us);

        let challenge_raw: F = transcript.challenge_f();
        let challenge_mont = challenge_raw.to_mont();
        let challenge_packed =
            <F::PackedExt as PackedExtensionField>::broadcast_scalar(challenge_mont);

        let t0 = now!();
        if poly_evals_tail.is_some() {
            let tail = poly_evals_tail.as_mut().unwrap();
            let new_len = poly_evals_tail_len / 2;
            for j in 0..new_len {
                let v0 = tail[2 * j];
                let v1 = tail[2 * j + 1];
                tail[j] = v0 + challenge_mont * (v1 - v0);
            }
            poly_evals_tail_len = new_len;
            poly_evals_logical_len = new_len;
        } else if poly_evals_packed.len() >= 2 {
            F::fold_multilinear_packed_par(
                &mut poly_evals_packed,
                &mut poly_evals_scratch,
                challenge_packed,
            );
            poly_evals_logical_len /= 2;
        } else {
            debug_assert_eq!(poly_evals_packed.len(), 1);
            debug_assert_eq!(poly_evals_logical_len, 8);
            let mut tail = [F::default(); 8];
            poly_evals_packed[0].unpack_into_slice(&mut tail);
            let new_len = 4;
            for j in 0..new_len {
                let v0 = tail[2 * j];
                let v1 = tail[2 * j + 1];
                tail[j] = v0 + challenge_mont * (v1 - v0);
            }
            poly_evals_tail = Some(tail);
            poly_evals_tail_len = new_len;
            poly_evals_logical_len = new_len;
        }
        tick!(t0, multilin_fold_us);

        if i < split_level {
            let t0 = now!();
            F::fold_sub_polys_packed_par(
                &mut sub_evals_packed,
                &mut sub_evals_scratch,
                eval_len_packed,
                current_sub_count,
                challenge_packed,
            );
            current_sub_count /= 2;
            tick!(t0, split_fold_us);
        } else {
            let fft_idx = i - split_level;
            let last_packed: &[F::PackedExt] = if fft_idx == 0 {
                &sub_evals_packed[0..eval_len_packed]
            } else {
                &fri_results[fft_idx - 1].value_packed
            };
            let t0 = now!();
            let next_packed: Vec<F::PackedExt> = if fft_idx == 0 {
                F::evaluate_next_domain_first_round_packed_par(
                    last_packed,
                    &pp.fft_groups[fft_idx],
                    challenge_packed,
                )
            } else {
                F::evaluate_next_domain_packed_par(
                    last_packed,
                    &pp.fft_groups[fft_idx],
                    challenge_packed,
                )
            };
            tick!(t0, fri_fold_us);

            let t0 = now!();
            if i < pp.variable_num - 1 {
                let fri_result = FriFoldResult::from_packed_values_par(next_packed);
                transcript.append_u8_slice(&fri_result.commit(), HASH_SIZE);
                fri_results.push(fri_result);
            } else {
                let mut tail = [F::default(); 8];
                next_packed[0].unpack_into_slice(&mut tail);
                transcript.append_f(tail[0].reduce_canonical());
            }
            tick!(t0, fri_merkle_us);
        }

        // DeepFold OOD binding — head-drop every active point after folding variable i.
        for w in active.iter_mut() {
            w.remove(0);
        }
    }
    let _ = poly_evals_logical_len;

    // Parallel grinding + sample query indices.
    transcript.grind_par(pp.grinding_bits);
    let mut leaf_indices = transcript.challenge_usizes(pp.query_num);
    let fat_domain = pp.fft_groups[0].size();
    leaf_indices = leaf_indices.iter_mut().map(|v| *v % (fat_domain >> 1)).collect();
    leaf_indices.sort();
    leaf_indices.dedup();

    (leaf_indices, fri_results)
}

/// Evaluate a small (≤ 8 lane) multilinear in scalar form. Used for the
/// post-tail-switch rounds in `open_inner`.
#[inline]
fn eval_multilinear_scalar_tail<F: DeepFoldExtField>(poly_evals: &[F], point: &[F]) -> F {
    let mut scratch: Vec<F> = poly_evals.to_vec();
    for r in point.iter() {
        let new_len = scratch.len() / 2;
        for j in 0..new_len {
            let v0 = scratch[2 * j];
            let v1 = scratch[2 * j + 1];
            scratch[j] = v0 + *r * (v1 - v0);
        }
        scratch.truncate(new_len);
    }
    scratch[0]
}

/// Per-substage wall-clock breakdown for `DeepFoldMamaBearProver::new_profiled`.
/// All values are accumulated microseconds.
#[derive(Clone, Debug, Default)]
pub struct NewTimings {
    pub total_us: u128,
    /// Up-front buffer allocations: `sub_evals_mont` (total_elems * 8B), scratch `sub_coeffs` / `fft_buf`.
    pub alloc_us: u128,
    /// Strided gather: building each sub-polynomial coefficient slice via the stride-`sub_count` scan.
    /// Summed over all (poly, sub_k) pairs.
    pub split_us: u128,
    /// `fft.fft_into(sub_coeffs, fft_buf)` — the per sub-poly FFT in Mont form, pair-major blocked.
    pub fft_us: u128,
    /// Extending `sub_evals_mont` with the per-sub FFT output slice.
    pub append_us: u128,
    /// Converting the owning `Vec<MamaBearScalar>` into the shared `Arc<[MamaBearScalar]>`.
    pub arc_convert_us: u128,
    /// `round0_leaf_hashes_from_pair_major_values` — build leaf-level Blake3 hashes (dominates merkle cost).
    pub leaf_hash_us: u128,
    /// `MerkleTreeProverMB::from_leaf_hashes` — internal node hashing up the tree.
    pub merkle_tree_us: u128,
    /// `from_fft_pair_major_parts_with_tree` wrap (plus cloning `poly` slices into owned `Vec`s).
    pub wrap_us: u128,
}

/// Per-substage wall-clock breakdown for `DeepFoldMamaBearProver::open_profiled`.
/// All values are accumulated microseconds; multiple calls add up.
#[derive(Clone, Debug, Default)]
pub struct OpenTimings {
    pub total_us: u128,
    /// Horner combine of poly coefficients across all (prover, poly) pairs into `poly_evals`.
    pub combine_polys_us: u128,
    /// Horner combine of FFT sub-evaluations across all (prover, poly) pairs into `sub_evals`.
    pub combine_subs_us: u128,
    /// Sum over rounds: MLE evaluation at the shifted point (`eval_multilinear_ext_mont`).
    pub mle_eval_us: u128,
    /// Sum over rounds: in-place multilinear fold by the round challenge.
    pub multilin_fold_us: u128,
    /// Sum over split-fold rounds: `fold_sub_polys_*` (base->ext on first call when applicable).
    pub split_fold_us: u128,
    /// Sum over FRI rounds: `evaluate_next_domain_mont(_first_round)`.
    pub fri_fold_us: u128,
    /// Sum over FRI rounds: Merkle tree build / final value append (`from_mont_values` + `commit`).
    pub fri_merkle_us: u128,
    /// Query collection (Merkle openings + value appends), post-fold-loop.
    pub query_phase_us: u128,
}

/// Per-substage wall-clock breakdown for `DeepFoldMamaBearVerifier::verify_profiled`.
/// All values are accumulated microseconds.
#[derive(Clone, Debug, Default)]
pub struct VerifyTimings {
    pub total_us: u128,
    /// Challenge loop + fold-consistency equation (Mont-form eval update, per-round Merkle root read).
    pub fold_check_us: u128,
    /// `transcript.verify_grind` (single BLAKE3 PoW verification).
    pub grinding_us: u128,
    /// Query index derivation (`challenge_usizes`, sort, dedup).
    pub query_prep_us: u128,
    /// Initial fat-leaf Merkle verify + per-prover leaf regrouping.
    pub fat_merkle_us: u128,
    /// Local split-fold (rounds 0..split_level) over `combined_j` / `combined_jh`.
    pub split_fold_us: u128,
    /// Per-round Merkle path verification for the standard FRI query phase (rounds split_level+1..variable_num-1).
    pub std_fri_merkle_us: u128,
    /// FRI fold + consistency check loop (rounds split_level..variable_num).
    pub fri_folds_us: u128,
}

// --- Verifier ---

#[derive(Clone)]
pub struct DeepFoldMamaBearVerifier<F: DeepFoldExtField> {
    pub(crate) commit: MerkleTreeVerifierMB,
    pub(crate) poly_num: usize,
    _phantom: std::marker::PhantomData<F>,
}

impl<F: DeepFoldExtField> DeepFoldMamaBearVerifier<F> {
    pub fn new(pp: &DeepFoldMamaBearParam, commit: MerkleRoot, poly_num: usize) -> Self {
        // Fat leaf Merkle tree: leave_num = fat_domain / 2 = fft_groups[0].size() / 2
        Self {
            commit: MerkleTreeVerifierMB::new(pp.fft_groups[0].size() / 2, commit.0),
            poly_num,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Number of fixed polynomials this commitment covers (the `poly_num`
    /// passed to [`new`](Self::new)). Read-only accessor for callers outside
    /// this crate (e.g. structural setup tests).
    #[inline]
    pub fn poly_num(&self) -> usize {
        self.poly_num
    }

    /// The committed Merkle root bytes (32-byte BLAKE3 root). Read-only
    /// accessor for callers outside this crate (e.g. determinism checks).
    #[inline]
    pub fn root_bytes(&self) -> [u8; HASH_SIZE] {
        self.commit.merkle_root
    }

    pub fn verify(
        pp: &DeepFoldMamaBearParam,
        verifiers: Vec<&Self>,
        point: Vec<F>,
        evals: Vec<Vec<F>>,
        transcript: &mut Transcript,
        proof: &mut Proof,
    ) -> bool {
        let mut t = VerifyTimings::default();
        Self::verify_inner(pp, verifiers, point, evals, transcript, proof, &mut t, false)
    }

    /// Profiling variant of `verify`. Accumulates per-substage wall-clock time
    /// into `timings`; `Instant::now()` overhead means this should only be used
    /// for profiling, not in production verification.
    pub fn verify_profiled(
        pp: &DeepFoldMamaBearParam,
        verifiers: Vec<&Self>,
        point: Vec<F>,
        evals: Vec<Vec<F>>,
        transcript: &mut Transcript,
        proof: &mut Proof,
        timings: &mut VerifyTimings,
    ) -> bool {
        Self::verify_inner(pp, verifiers, point, evals, transcript, proof, timings, true)
    }

    fn verify_inner(
        pp: &DeepFoldMamaBearParam,
        verifiers: Vec<&Self>,
        point: Vec<F>,
        evals: Vec<Vec<F>>,
        transcript: &mut Transcript,
        proof: &mut Proof,
        timings: &mut VerifyTimings,
        record: bool,
    ) -> bool {
        use std::time::Instant;
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

        // --- Phases 1 + 2 + 3 (extracted to verify_pre_round0_fold_check) ---
        let state = match verify_pre_round0_fold_check::<F>(
            pp,
            point,
            evals,
            transcript,
            proof,
            timings,
            record,
        ) {
            Some(s) => s,
            None => {
                tick!(total_t0, total_us);
                return false;
            }
        };
        let r_mont = state.r_mont;
        let eval_mont = state.eval_mont;
        let challenges_mont = state.challenges_mont;
        let commits = state.commits;
        let leaf_indices = state.leaf_indices;
        let mut query_results: Vec<QueryResultMB<F>> = vec![];
        let indices = leaf_indices.clone();
        let fat_domain = pp.fft_groups[0].size(); // phase 4 (round-0 fat-leaf verify) needs this

        {
            // --- Fat-leaf Merkle verify + per-prover regroup ---
            let fat_t0 = now!();
            let mut poly_values: Vec<Vec<MamaBearScalar>> = vec![];
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

                // Group values by polynomial (sub_count groups per poly)
                // proof_values layout from query(): for each slot s in 0..fat_leaf_size,
                //   then for each query index → proof_values[s * num_queries + q]
                // Regroup into poly_values: [poly_p_sub_k values]
                for p in 0..verifier.poly_num {
                    for k in 0..sub_count {
                        let slot_j = (p * sub_count + k) * 2; // slot for sub_k at j
                        let slot_jh = (p * sub_count + k) * 2 + 1; // slot for sub_k at j+half
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

                // (2026-04-26): replace `.unwrap` + `assert!`
                // with clean `Option`/`bool` propagation. Tampered proofs
                // now reject via `return false` instead of panic; missing
                // base_query_values entries also reject cleanly. The OLD
                // soundness path relied on panic propagation through
                // catch_unwind in tests; that's not safe in production.
                let base_query_values: HashMap<usize, MamaBearScalar> = proof_values
                    .iter()
                    .copied()
                    .enumerate()
                    .map(|(idx, x)| {
                        (
                            leaf_indices[idx % leaf_indices.len()]
                                + (fat_domain / 2) * (idx / leaf_indices.len()),
                            x,
                        )
                    })
                    .collect();
                let mut base_leaves: Vec<Vec<u8>> =
                    Vec::with_capacity(leaf_indices.len());
                for i in leaf_indices.iter() {
                    let mut row: Vec<MamaBearScalar> = Vec::with_capacity(fat_leaf_size);
                    for j in 0..fat_leaf_size {
                        match base_query_values.get(&(i + j * (fat_domain / 2))) {
                            Some(v) => row.push(*v),
                            None => {
                                tick!(fat_t0, fat_merkle_us);
                                tick!(total_t0, total_us);
                                return false;
                            }
                        }
                    }
                    base_leaves.push(as_bytes_vec(&row));
                }
                if !verifier.commit.verify(&proof_bytes, &leaf_indices, &base_leaves) {
                    tick!(fat_t0, fat_merkle_us);
                    tick!(total_t0, total_us);
                    return false;
                }
            }
            tick!(fat_t0, fat_merkle_us);

            // Combine sub-poly values with r and do local split fold
            // poly_values: for each (prover, poly, sub_k): [vals_at_j..., vals_at_j+half...]
            // Total entries: total_polys * sub_count vectors, each of size 2 * num_queries

            let split_t0 = now!();
            let num_queries = leaf_indices.len();
            let total_sub_groups = poly_values.len(); // sum(poly_num) * sub_count

            // Combine with r: for each sub-poly k and query q, compute extension field value
            let mut combined_j: Vec<Vec<F>> = vec![Vec::new(); sub_count];
            let mut combined_jh: Vec<Vec<F>> = vec![Vec::new(); sub_count];

            for k in 0..sub_count {
                combined_j[k] = vec![F::zero().to_mont(); num_queries];
                combined_jh[k] = vec![F::zero().to_mont(); num_queries];
            }

            // Accumulate: iterate over (prover, poly) groups in the same order as the prover
            // poly_values is grouped as: [prover0_poly0_sub0, prover0_poly0_sub1, ..., prover0_poly1_sub0, ...]
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

            // Local split fold verification (rounds 0..split_level-1)
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

        // --- Phases 5 + 6 (extracted to verify_after_round0_fri_check) ---
        let result = verify_after_round0_fri_check::<F>(
            pp,
            leaf_indices,
            indices,
            query_results,
            &challenges_mont,
            eval_mont,
            &commits,
            transcript,
            proof,
            timings,
            record,
        );
        tick!(total_t0, total_us);
        result
    }
}

/// State returned by `verify_pre_round0_fold_check` for downstream phases
/// (round-0 fat-leaf verify + post-round-0 FRI check).
pub(crate) struct VerifyPreRound0State<F: DeepFoldExtField> {
    pub r_mont: F,
    pub eval_mont: F,
    pub challenges_mont: Vec<F>,
    pub commits: Vec<MerkleTreeVerifierMB>,
    pub leaf_indices: Vec<usize>,
}

/// Run the verifier's pre-round-0 phases (1+2+3 of `verify_inner`):
///   1. Fold consistency check (variable_num rounds, reads MLE eval + maybe
///      Merkle commit per round, final eval check on last round)
///   2. Grinding (PoW) verification
///   3. Query index derivation (sample, dedup)
///
/// Pure F-typed; shared between base `DeepFoldMamaBearVerifier::verify_inner`
/// and ext `DeepFoldMamaBearVerifierExt::verify`. Returns `None` if any
/// check fails (final-eval mismatch, grinding fail).
// ===================== DeepFold DEEP binding — verifier helpers =====================
//
// Mirror the prover-side deep_open_* helpers. Shared verbatim by the serial
// `verify_pre_round0_fold_check` and the parallel inline verifier in
// `deepfold_mamabear_par.rs`, so the two verifiers stay in lock-step. All values are
// Montgomery form; `reduce_canonical` only canonicalizes the representative (it is not a
// domain switch).

/// Draw alpha, read c = f^(0)(alpha), absorb it, and return the alpha-family DEEP point set
/// [(alpha_vec, c)] as (current-vector, carried-claim). The z-point is tracked by the
/// caller as `eval_mont` (the combined claim y).
#[inline]
pub(crate) fn deep_verify_init<F: DeepFoldExtField>(
    variable_num: usize,
    transcript: &mut Transcript,
    proof: &mut Proof,
) -> Vec<(Vec<F>, F)> {
    let alpha = transcript.challenge_f::<F>().to_mont();
    let alpha_vec = crate::deepfold::deep_power_vector(alpha, variable_num);
    let c = proof.get_next_and_step::<F>();
    transcript.append_f(c);
    vec![(alpha_vec, c)]
}

/// Per-round reads (mirror of deep_open_round). Read the fresh alpha_i seed (all but the
/// last round) and push it, then read the z-point off and every alpha-family off in the
/// exact order the prover sent them, absorbing each. Returns (z_off, alpha_family_offs).
#[inline]
pub(crate) fn deep_verify_reads<F: DeepFoldExtField>(
    i: usize,
    variable_num: usize,
    deep: &mut Vec<(Vec<F>, F)>,
    transcript: &mut Transcript,
    proof: &mut Proof,
) -> (F, Vec<F>) {
    if i < variable_num - 1 {
        let alpha_i = transcript.challenge_f::<F>().to_mont();
        let w_i = crate::deepfold::deep_power_vector(alpha_i, variable_num - i);
        let seed = proof.get_next_and_step::<F>();
        transcript.append_f(seed);
        deep.push((w_i, seed));
    }
    let next_eval_mont = proof.get_next_and_step::<F>();
    transcript.append_f(next_eval_mont);
    let mut deep_offs = Vec::with_capacity(deep.len());
    for _ in 0..deep.len() {
        let off = proof.get_next_and_step::<F>();
        transcript.append_f(off);
        deep_offs.push(off);
    }
    (next_eval_mont, deep_offs)
}

/// Per-round claim update (mirror of the prover fold + head-drop). Propagate each
/// alpha-family line to the fold challenge r_i and drop the leading coordinate.
#[inline]
pub(crate) fn deep_verify_update<F: DeepFoldExtField>(
    deep: &mut Vec<(Vec<F>, F)>,
    deep_offs: &[F],
    challenge_mont: F,
) {
    for (k, (w, claim)) in deep.iter_mut().enumerate() {
        let head = w[0];
        *claim = *claim + (challenge_mont - head) * (deep_offs[k] - *claim);
        w.remove(0);
    }
}

/// Terminal check: every alpha-family chain must have converged to the same f^(mu) as the
/// z-point (`eval_mont`). A codeword-swapping prover cannot satisfy this (Lemma 7).
#[inline]
pub(crate) fn deep_verify_terminal_ok<F: DeepFoldExtField>(
    deep: &[(Vec<F>, F)],
    eval_mont: F,
) -> bool {
    let target = eval_mont.reduce_canonical();
    deep.iter()
        .all(|(_, claim)| claim.reduce_canonical() == target)
}

pub(crate) fn verify_pre_round0_fold_check<F: DeepFoldExtField>(
    pp: &DeepFoldMamaBearParam,
    point: Vec<F>,
    evals: Vec<Vec<F>>,
    transcript: &mut Transcript,
    proof: &mut Proof,
    timings: &mut VerifyTimings,
    record: bool,
) -> Option<VerifyPreRound0State<F>> {
    macro_rules! now {
        () => {
            if record { Some(Instant::now()) } else { None }
        };
    }
    macro_rules! tick {
        ($t:ident, $field:ident) => {
            if let Some(t) = $t {
                timings.$field += t.elapsed().as_micros();
            }
        };
    }

    let split_level = pp.split_level;

    // --- Phase 1: Fold consistency check ---
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
    let mut commits = vec![];

    // DeepFold OOD binding — draw alpha, read c = f^(0)(alpha), seed the alpha-family DEEP
    // point set. `eval_mont` above is the z-point's carried claim y. See add/mul.
    let mut deep = deep_verify_init::<F>(pp.variable_num, transcript, proof);

    for i in 0..point.len() {
        // DeepFold OOD binding — read the fresh alpha_i seed and every off-value in the
        // prover's order (z off first, then the alpha-family offs), then draw r_i.
        let (next_eval_mont, deep_offs) =
            deep_verify_reads::<F>(i, pp.variable_num, &mut deep, transcript, proof);
        let challenge_raw: F = transcript.challenge_f();
        let challenge_mont = challenge_raw.to_mont();

        // z-line claim update (byte-identical to the old z-line).
        eval_mont = eval_mont + (challenge_mont - point_mont[i]) * (next_eval_mont - eval_mont);
        challenges_mont.push(challenge_mont);
        // alpha-family claim updates + head-drop.
        deep_verify_update::<F>(&mut deep, &deep_offs, challenge_mont);

        if i < split_level {
            // Split fold round: no Merkle commitment
        } else if i < pp.variable_num - 1 {
            // Standard FRI round: read Merkle commitment
            let merkle_root = proof.get_next_hash();
            transcript.append_u8_slice(&merkle_root, HASH_SIZE);
            commits.push(MerkleTreeVerifierMB::new(
                pp.fft_groups[i - split_level + 1].size() / 2,
                merkle_root,
            ));
        } else {
            // Last round: the z-point terminal must equal the prover's final scalar, AND
            // (the DEEP binding) every alpha-family chain must have converged to the same
            // f^(mu). A codeword-swapping prover cannot satisfy all of these (Lemma 7).
            let final_mont = proof.get_next_and_step::<F>();
            transcript.append_f(final_mont);
            if final_mont.reduce_canonical() != eval_mont.reduce_canonical() {
                tick!(fold_t0, fold_check_us);
                return None;
            }
            if !deep_verify_terminal_ok::<F>(&deep, eval_mont) {
                tick!(fold_t0, fold_check_us);
                return None;
            }
        }
    }
    tick!(fold_t0, fold_check_us);

    // --- Phase 2: Grinding (PoW) verification ---
    let grind_t0 = now!();
    let grind_ok = transcript.verify_grind(proof, pp.grinding_bits);
    tick!(grind_t0, grinding_us);
    if !grind_ok {
        return None;
    }

    // --- Phase 3: Query index derivation ---
    let qprep_t0 = now!();
    let mut leaf_indices = transcript.challenge_usizes(pp.query_num);
    let fat_domain = pp.fft_groups[0].size();
    leaf_indices = leaf_indices
        .iter_mut()
        .map(|v| *v % (fat_domain >> 1))
        .collect();
    leaf_indices.sort();
    leaf_indices.dedup();
    tick!(qprep_t0, query_prep_us);

    Some(VerifyPreRound0State {
        r_mont,
        eval_mont,
        challenges_mont,
        commits,
        leaf_indices,
    })
}

/// Run the standard FRI Merkle verification + FRI fold consistency check
/// (phases 5 + 6 of `verify_inner`). Shared between base
/// `DeepFoldMamaBearVerifier::verify_inner` and ext
/// `DeepFoldMamaBearVerifierExt::verify` -- the round-0 fat-leaf phase
/// stays type-specific (different leaf byte width / proof_value type),
/// but everything after it operates purely on `F` and is identical.
///
/// Returns `true` iff all FRI checks pass. May panic via
/// `QueryResultMB::verify_merkle_tree` if a Merkle proof is malformed
/// (matches base verifier behaviour).
pub(crate) fn verify_after_round0_fri_check<F: DeepFoldExtField>(
    pp: &DeepFoldMamaBearParam,
    mut leaf_indices: Vec<usize>,
    mut indices: Vec<usize>,
    mut query_results: Vec<QueryResultMB<F>>,
    challenges_mont: &[F],
    eval_mont: F,
    commits: &[MerkleTreeVerifierMB],
    transcript: &mut Transcript,
    proof: &mut Proof,
    timings: &mut VerifyTimings,
    record: bool,
) -> bool {
    macro_rules! now {
        () => {
            if record { Some(Instant::now()) } else { None }
        };
    }
    macro_rules! tick {
        ($t:ident, $field:ident) => {
            if let Some(t) = $t {
                timings.$field += t.elapsed().as_micros();
            }
        };
    }

    let split_level = pp.split_level;

    // --- Standard FRI query Merkle verification (rounds split_level+1..variable_num-1) ---
    let stdm_t0 = now!();
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
        // (2026-04-26): propagate the merkle verify result
        // upward instead of ignoring it. Earlier code called
        // `query.verify_merkle_tree(...);` and threw away the bool, relying
        // on `assert!(res)` *inside* `verify_merkle_tree` to panic on a
        // bad proof. After the inner function returns `false`
        // cleanly, so the caller MUST check + propagate. Soundness bug
        // surfaced by a byte-identity/tamper regression test that expects
        // verify to reject tampered proofs.
        if !query.verify_merkle_tree(&leaf_indices, 2, &commits[k]) {
            tick!(stdm_t0, std_fri_merkle_us);
            return false;
        }
        query_results.push(query);
    }
    drop(leaf_indices);
    tick!(stdm_t0, std_fri_merkle_us);

    // --- FRI fold + consistency check (rounds split_level..variable_num) ---
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
                    return false;
                }
            } else {
                let check = new_v.mul_base_elem(inv_2_mont);
                if check.reduce_canonical() != eval_mont.reduce_canonical() {
                    tick!(folds_t0, fri_folds_us);
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
    true
}

// --- Query result ---

#[derive(Clone)]
pub struct QueryResultMB<F: Field> {
    pub proof_bytes: Vec<u8>,
    pub proof_values: HashMap<usize, F>,
}

impl<F: Field> QueryResultMB<F> {
    pub fn verify_merkle_tree(
        &self,
        leaf_indices: &[usize],
        leaf_size: usize,
        merkle_verifier: &MerkleTreeVerifierMB,
    ) -> bool {
        debug_assert_eq!(
            leaf_size, 2,
            "query results currently assume adjacent-pair leaves"
        );
        // (2026-04-26): use `match` over `proof_values.get` so a
        // missing or corrupt entry returns `false` instead of panicking via
        // `.unwrap()`. Earlier code also wrapped the merkle verify result in
        // `assert!(res)`; a malicious / malformed proof would crash the
        // verifier process. Now we cleanly return false to give the caller a
        // chance to handle the failure (e.g. log + reject).
        let mut leaves: Vec<Vec<u8>> = Vec::with_capacity(leaf_indices.len());
        for i in leaf_indices.iter() {
            let mut row: Vec<F> = Vec::with_capacity(leaf_size);
            for j in 0..leaf_size {
                match self.proof_values.get(&(2 * i + j)) {
                    Some(v) => row.push(*v),
                    None => return false,
                }
            }
            leaves.push(as_bytes_vec(&row));
        }
        merkle_verifier.verify(&self.proof_bytes, leaf_indices, &leaves)
    }
}

// --- Helper: multilinear extension evaluation in Montgomery form ---

/// Reference multilinear extension evaluation, scalar-only.
///
/// Used by tests to compute the expected value before calling `open` /
/// `verify`. Not on the prover hot path (the prover uses
/// `eval_multilinear_packed`).
#[allow(dead_code)]
pub(crate) fn eval_multilinear_ext_mont<F: DeepFoldExtField>(poly_evals: &[F], point: &[F]) -> F {
    let mut scratch = poly_evals.to_vec();
    for r in point.iter() {
        let new_len = scratch.len() / 2;
        for j in 0..new_len {
            let v0 = scratch[2 * j];
            let v1 = scratch[2 * j + 1];
            scratch[j] = v0 + *r * (v1 - v0);
        }
        scratch.truncate(new_len);
    }
    scratch[0]
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::SmallRng;
    use rand::{RngCore, SeedableRng};

    fn test_prove_verify_ext3(nv: usize, split_level: usize) {
        let mut rng = SmallRng::seed_from_u64(1);
        let code_rate_log = 3;
        let query_num = 30;

        let pp = DeepFoldMamaBearParam::new(nv, code_rate_log, query_num, split_level);

        let poly_evals: Vec<MamaBearScalar> = (0..1 << nv)
            .map(|_| MamaBearScalar(rng.next_u64() % P))
            .collect();

        let point: Vec<MamaBearScalarExt3> = (0..nv)
            .map(|_| MamaBearScalarExt3 {
                c0: MamaBearScalar(rng.next_u64() % P),
                c1: MamaBearScalar(rng.next_u64() % P),
                c2: MamaBearScalar(rng.next_u64() % P),
            })
            .collect();

        let poly_evals_ext: Vec<MamaBearScalarExt3> = poly_evals
            .iter()
            .map(|x| MamaBearScalarExt3::from(*x).to_montgomery())
            .collect();
        let point_mont: Vec<MamaBearScalarExt3> = point.iter().map(|x| x.to_montgomery()).collect();
        let eval_mont = eval_multilinear_ext_mont(&poly_evals_ext, &point_mont);
        let eval = eval_mont.reduce_canonical();

        let mut transcript = Transcript::new();
        let prover = DeepFoldMamaBearProver::<MamaBearScalarExt3>::new(&pp, &[&poly_evals]);
        let commitment = prover.commit();
        let mut buffer = vec![0u8; HASH_SIZE];
        commitment.serialize_into(&mut buffer);
        transcript.append_u8_slice(&buffer, HASH_SIZE);
        transcript.append_f(eval);
        DeepFoldMamaBearProver::open(&pp, &[&prover], point_mont.clone(), &mut transcript);
        let mut proof = transcript.proof;

        let commitment = MerkleRoot::deserialize_from(&mut proof, nv, 1);
        let mut transcript = Transcript::new();
        let mut buffer = vec![0u8; HASH_SIZE];
        commitment.serialize_into(&mut buffer);
        transcript.append_u8_slice(&buffer, HASH_SIZE);
        let verifier = DeepFoldMamaBearVerifier::<MamaBearScalarExt3>::new(&pp, commitment, 1);
        let eval_from_proof: MamaBearScalarExt3 = proof.get_next_and_step();
        transcript.append_f(eval_from_proof);
        let ok = DeepFoldMamaBearVerifier::verify(
            &pp,
            vec![&verifier],
            point,
            vec![vec![eval_from_proof]],
            &mut transcript,
            &mut proof,
        );
        assert!(
            ok,
            "Ext3 verify failed (nv={}, split={})",
            nv, split_level
        );
    }

    #[test]
    fn test_zero_mont_assumption() {
        // After the `to_montgomery` canonicalization fix, the Mont form of 0
        // is 0 (not P). `ZERO_MONT` is aligned to this — use 0 everywhere.
        let zero = MamaBearScalar(0);
        let zero_mont = zero.to_montgomery();
        eprintln!("MamaBearScalar(0).to_montgomery() = {}", zero_mont.0);
        assert_eq!(zero_mont.0, 0, "Mont(0) should be canonical 0");
        assert_eq!(ZERO_MONT.0, 0, "ZERO_MONT should be canonical 0");

        // `from_base_mont` via `MamaBearScalarExt3::from + to_montgomery` must
        // match a direct build with `{ c0: raw.to_mont(), c1: c2: ZERO_MONT }`.
        let raw = MamaBearScalar(12345);
        let via_old = MamaBearScalarExt3::from(raw).to_montgomery();
        let mont_val = raw.to_montgomery();
        let via_new = MamaBearScalarExt3 {
            c0: mont_val,
            c1: ZERO_MONT,
            c2: ZERO_MONT,
        };
        assert_eq!(via_old.c0.0, via_new.c0.0);
        assert_eq!(via_old.c1.0, via_new.c1.0);
        assert_eq!(via_old.c2.0, via_new.c2.0);
    }

    // --- Ext3 tests for each split level ---

    #[test]
    fn test_ext3_split0() {
        test_prove_verify_ext3(12, 0);
    }
    #[test]
    fn test_ext3_split1() {
        test_prove_verify_ext3(12, 1);
    }
    #[test]
    fn test_ext3_split2() {
        test_prove_verify_ext3(12, 2);
    }
    #[test]
    fn test_ext3_split3() {
        test_prove_verify_ext3(12, 3);
    }
    #[test]
    fn test_ext3_split4() {
        test_prove_verify_ext3(12, 4);
    }

    // --- Larger NV test ---

    #[test]
    fn test_ext3_nv16_split3() {
        test_prove_verify_ext3(16, 3);
    }

    // --- DeepFold OOD (DEEP) soundness PoC: decoupled c must REJECT ---

    /// Soundness PoC — the DeepFold out-of-domain (DEEP) binding must REJECT a
    /// decoupled OOD claim `c`. A malicious prover appends a WRONG
    /// `c' = f^(0)(alpha) + 1` (via the `#[cfg(test)]` `DEEP_C_OFFSET` forge hook
    /// in `deep_open_init`) but is otherwise honest and transcript-consistent, so
    /// the z-line fold + FRI/Merkle query checks all pass. The alpha-family DEEP
    /// chain, seeded at the wrong `c'`, drifts and its terminal no longer equals
    /// `f^(mu)`, so the verifier's DEEP terminal check (`deep_verify_terminal_ok`)
    /// must reject. Without the OOD mechanism (the old z-line variant) this forged
    /// proof would be ACCEPTED — this is the evidence the fix binds on the
    /// PRODUCTION MamaBear backend (the Goldilocks-generic twin is
    /// `crate::deepfold` `pc_rejects_decoupled_deep_c`).
    #[test]
    fn deepfold_mamabear_rejects_decoupled_deep_c_ext3() {
        let nv = 12usize;
        let split_level = 2usize;
        let code_rate_log = 3;
        let query_num = 30;
        let pp = DeepFoldMamaBearParam::new(nv, code_rate_log, query_num, split_level);

        let mut rng = SmallRng::seed_from_u64(1);
        let poly_evals: Vec<MamaBearScalar> = (0..1 << nv)
            .map(|_| MamaBearScalar(rng.next_u64() % P))
            .collect();
        let point: Vec<MamaBearScalarExt3> = (0..nv)
            .map(|_| MamaBearScalarExt3 {
                c0: MamaBearScalar(rng.next_u64() % P),
                c1: MamaBearScalar(rng.next_u64() % P),
                c2: MamaBearScalar(rng.next_u64() % P),
            })
            .collect();
        let poly_evals_ext: Vec<MamaBearScalarExt3> = poly_evals
            .iter()
            .map(|x| MamaBearScalarExt3::from(*x).to_montgomery())
            .collect();
        let point_mont: Vec<MamaBearScalarExt3> =
            point.iter().map(|x| x.to_montgomery()).collect();
        let eval_mont = eval_multilinear_ext_mont(&poly_evals_ext, &point_mont);
        let eval = eval_mont.reduce_canonical();

        // One open+verify round-trip forging the OOD claim with `offset` (0 = honest).
        let run = |offset: u64| -> bool {
            // Prove.
            let mut transcript = Transcript::new();
            let prover = DeepFoldMamaBearProver::<MamaBearScalarExt3>::new(&pp, &[&poly_evals]);
            let commitment = prover.commit();
            let mut buffer = vec![0u8; HASH_SIZE];
            commitment.serialize_into(&mut buffer);
            transcript.append_u8_slice(&buffer, HASH_SIZE);
            transcript.append_f(eval);
            DEEP_C_OFFSET.with(|v| v.set(offset));
            DeepFoldMamaBearProver::open(&pp, &[&prover], point_mont.clone(), &mut transcript);
            DEEP_C_OFFSET.with(|v| v.set(0)); // reset so verify (and later tests) are honest
            let mut proof = transcript.proof;

            // Verify.
            let commitment = MerkleRoot::deserialize_from(&mut proof, nv, 1);
            let mut transcript = Transcript::new();
            let mut buffer = vec![0u8; HASH_SIZE];
            commitment.serialize_into(&mut buffer);
            transcript.append_u8_slice(&buffer, HASH_SIZE);
            let verifier = DeepFoldMamaBearVerifier::<MamaBearScalarExt3>::new(&pp, commitment, 1);
            let eval_from_proof: MamaBearScalarExt3 = proof.get_next_and_step();
            transcript.append_f(eval_from_proof);
            DeepFoldMamaBearVerifier::verify(
                &pp,
                vec![&verifier],
                point.clone(),
                vec![vec![eval_from_proof]],
                &mut transcript,
                &mut proof,
            )
        };

        // Positive control: the honest OOD claim (offset 0) ACCEPTS.
        assert!(run(0), "honest OOD c must verify (positive control)");
        // Soundness: the forged c' = c + 1 (decoupled from the committed poly) REJECTS.
        assert!(
            !run(1),
            "verifier MUST reject a decoupled OOD claim c (DEEP binding failure)"
        );
    }

    // --- FRI grinding: end-to-end prove/verify with PoW enabled ---

    fn test_prove_verify_ext3_with_grinding(nv: usize, split_level: usize, grinding_bits: u32) {
        let mut rng = SmallRng::seed_from_u64(1);
        let code_rate_log = 3;
        let query_num = 30;

        let mut pp = DeepFoldMamaBearParam::new(nv, code_rate_log, query_num, split_level);
        pp.grinding_bits = grinding_bits;

        let poly_evals: Vec<MamaBearScalar> = (0..1 << nv)
            .map(|_| MamaBearScalar(rng.next_u64() % P))
            .collect();

        let point: Vec<MamaBearScalarExt3> = (0..nv)
            .map(|_| MamaBearScalarExt3 {
                c0: MamaBearScalar(rng.next_u64() % P),
                c1: MamaBearScalar(rng.next_u64() % P),
                c2: MamaBearScalar(rng.next_u64() % P),
            })
            .collect();

        let poly_evals_ext: Vec<MamaBearScalarExt3> = poly_evals
            .iter()
            .map(|x| MamaBearScalarExt3::from(*x).to_montgomery())
            .collect();
        let point_mont: Vec<MamaBearScalarExt3> = point.iter().map(|x| x.to_montgomery()).collect();
        let eval_mont = eval_multilinear_ext_mont(&poly_evals_ext, &point_mont);
        let eval = eval_mont.reduce_canonical();

        let mut transcript = Transcript::new();
        let prover = DeepFoldMamaBearProver::<MamaBearScalarExt3>::new(&pp, &[&poly_evals]);
        let commitment = prover.commit();
        let mut buffer = vec![0u8; HASH_SIZE];
        commitment.serialize_into(&mut buffer);
        transcript.append_u8_slice(&buffer, HASH_SIZE);
        transcript.append_f(eval);
        DeepFoldMamaBearProver::open(&pp, &[&prover], point_mont.clone(), &mut transcript);
        let mut proof = transcript.proof;

        let commitment = MerkleRoot::deserialize_from(&mut proof, nv, 1);
        let mut transcript = Transcript::new();
        let mut buffer = vec![0u8; HASH_SIZE];
        commitment.serialize_into(&mut buffer);
        transcript.append_u8_slice(&buffer, HASH_SIZE);
        let verifier = DeepFoldMamaBearVerifier::<MamaBearScalarExt3>::new(&pp, commitment, 1);
        let eval_from_proof: MamaBearScalarExt3 = proof.get_next_and_step();
        transcript.append_f(eval_from_proof);
        let ok = DeepFoldMamaBearVerifier::verify(
            &pp,
            vec![&verifier],
            point,
            vec![vec![eval_from_proof]],
            &mut transcript,
            &mut proof,
        );
        assert!(
            ok,
            "Ext3 verify (grinding) failed (nv={}, split={}, bits={})",
            nv, split_level, grinding_bits
        );
    }

    /// User-requested smoke test: Ext3 end-to-end with 20-bit grinding.
    #[test]
    fn test_prove_verify_ext3_grinding20() {
        test_prove_verify_ext3_with_grinding(14, 5, 20);
    }

    /// (2026-04-26): `QueryResultMB::verify_merkle_tree` must
    /// return `false` cleanly when fed a malformed `QueryResultMB` (e.g.
    /// missing proof_values entries), NOT panic via `.unwrap()` on the
    /// HashMap lookup.
    ///
    /// Earlier code had two production-path crash points:
    ///   1. `proof_values.get(...).unwrap()` — panics on missing index
    ///   2. `assert!(res)` after merkle verify — panics on Merkle mismatch
    ///
    /// Both replaced with clean Err/false propagation. This test
    /// constructs an empty `QueryResultMB` (no proof_values, no
    /// proof_bytes) and confirms the function returns `false` instead of
    /// panicking.
    #[test]
    fn test_query_result_verify_returns_false_on_missing_entries() {
        use util::merkle_tree_mamabear::{MerkleTreeVerifierMB, HASH_SIZE};

        // Empty query result: no proof bytes, no proof_values.
        let empty_qr = QueryResultMB::<MamaBearScalar> {
            proof_bytes: Vec::new(),
            proof_values: std::collections::HashMap::new(),
        };

        // Construct a verifier referencing some plausible root (zeros).
        // We don't need a "real" tree: just confirm `verify_merkle_tree`
        // doesn't panic on a bad QueryResultMB.
        let zero_root: [u8; HASH_SIZE] = [0u8; HASH_SIZE];
        let verifier = MerkleTreeVerifierMB::new(8usize, zero_root);

        // leaf_indices points to entries that don't exist in proof_values
        // → first .get() in our fixed code returns None → return false.
        let leaf_indices = vec![0, 1, 2];

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            empty_qr.verify_merkle_tree(&leaf_indices, 2, &verifier)
        }));

        match result {
            Ok(false) => { /* expected: clean rejection, no panic */ }
            Ok(true) => panic!("empty QueryResultMB unexpectedly verified true"),
            Err(_) => panic!(
                "verify_merkle_tree panicked on missing entries — Phase E #3 \
                 unwrap/assert removal not effective"
            ),
        }
    }

    /// User-requested stress test: Ext3 end-to-end with 32-bit grinding.
    /// `#[ignore]` because the parallel grind takes several seconds; run via
    /// `cargo test -- --ignored`. This also exercises the `mask == u32::MAX`
    /// boundary (check becomes "first LE u32 word of BLAKE3 output == 0").
    #[test]
    #[ignore]
    fn test_prove_verify_ext3_grinding32() {
        test_prove_verify_ext3_with_grinding(14, 5, 32);
    }

    #[test]
    fn test_round0_leaf_hashes_match_flat_leaf_serialization() {
        fn build_round0_leaf_flat(values_mont: &[MamaBearScalar], leaf_size: usize) -> Vec<u8> {
            let segment_count = leaf_size / 2;
            let eval_len = values_mont.len() / segment_count;
            let leaf_count = eval_len / 2;
            let leaf_bytes = leaf_size * MamaBearScalar::SIZE;
            let mut leaf_flat = vec![0u8; leaf_count * leaf_bytes];

            for segment in 0..segment_count {
                let segment_values = &values_mont[segment * eval_len..(segment + 1) * eval_len];
                let leaf_offset = segment * ROUND0_PAIR_BYTES;
                let pair_count = leaf_count;
                let pairs_per_block = MamaBearFFT::pair_slots_per_block_for_pair_count(pair_count);

                if pairs_per_block == 8 {
                    for base in (0..segment_values.len()).step_by(16) {
                        let pair_base = base >> 1;
                        for lane in 0..8 {
                            let leaf_idx = pair_base + lane;
                            let dst_off = leaf_idx * leaf_bytes + leaf_offset;
                            write_round0_pair_bytes(
                                &mut leaf_flat[dst_off..dst_off + ROUND0_PAIR_BYTES],
                                segment_values[base + lane],
                                segment_values[base + 8 + lane],
                            );
                        }
                    }
                } else {
                    for pair_index in 0..pair_count {
                        let (x_pos, nx_pos) = MamaBearFFT::pair_storage_positions_for_pair_count(
                            pair_index, pair_count,
                        );
                        let dst_off = pair_index * leaf_bytes + leaf_offset;
                        write_round0_pair_bytes(
                            &mut leaf_flat[dst_off..dst_off + ROUND0_PAIR_BYTES],
                            segment_values[x_pos],
                            segment_values[nx_pos],
                        );
                    }
                }
            }

            leaf_flat
        }

        for (leaf_count, leaf_size) in [(4usize, 4usize), (32usize, 64usize)] {
            let segment_count = leaf_size / 2;
            let eval_len = leaf_count * 2;
            let mut rng = SmallRng::seed_from_u64(((leaf_count as u64) << 8) | leaf_size as u64);

            let values_mont: Vec<MamaBearScalar> = (0..segment_count * eval_len)
                .map(|_| MamaBearScalar(rng.next_u64() % P).to_montgomery().reduce())
                .collect();

            let leaf_bytes = leaf_size * MamaBearScalar::SIZE;
            let leaf_flat = build_round0_leaf_flat(&values_mont, leaf_size);
            let mut flat_hashes = vec![[0u8; HASH_SIZE]; leaf_count];
            blake3_batch::hash_leaves_batch_flat(
                &leaf_flat,
                leaf_count,
                leaf_bytes,
                &mut flat_hashes,
            );

            let blocked_hashes = round0_leaf_hashes_from_pair_major_values(&values_mont, leaf_size);
            assert_eq!(
                flat_hashes, blocked_hashes,
                "leaf hash mismatch for leaf_count={leaf_count}, leaf_size={leaf_size}"
            );

            let flat_tree =
                MerkleTreeProverMB::from_flat_leaves(&leaf_flat, leaf_count, leaf_bytes);
            let blocked_tree = MerkleTreeProverMB::from_leaf_hashes(&blocked_hashes);
            assert_eq!(flat_tree.commit(), blocked_tree.commit());
            assert_eq!(
                flat_tree.open(&[0, leaf_count - 1]),
                blocked_tree.open(&[0, leaf_count - 1])
            );
        }
    }

    // --- Proof size + timing analysis ---

    fn measure_ext3(
        nv: usize,
        split_level: usize,
    ) -> (usize, std::time::Duration, std::time::Duration) {
        use std::time::Instant;

        let mut rng = SmallRng::seed_from_u64(42);
        let code_rate_log = 3;
        let query_num = 34;

        let pp = DeepFoldMamaBearParam::new(nv, code_rate_log, query_num, split_level);

        let poly_evals: Vec<MamaBearScalar> = (0..1 << nv)
            .map(|_| MamaBearScalar(rng.next_u64() % P))
            .collect();
        let point: Vec<MamaBearScalarExt3> = (0..nv)
            .map(|_| MamaBearScalarExt3 {
                c0: MamaBearScalar(rng.next_u64() % P),
                c1: MamaBearScalar(rng.next_u64() % P),
                c2: MamaBearScalar(rng.next_u64() % P),
            })
            .collect();

        let poly_evals_ext: Vec<MamaBearScalarExt3> = poly_evals
            .iter()
            .map(|x| MamaBearScalarExt3::from(*x).to_montgomery())
            .collect();
        let point_mont: Vec<MamaBearScalarExt3> = point.iter().map(|x| x.to_montgomery()).collect();
        let eval_mont = eval_multilinear_ext_mont(&poly_evals_ext, &point_mont);
        let eval = eval_mont.reduce_canonical();

        // Commit
        let t0 = Instant::now();
        let prover = DeepFoldMamaBearProver::<MamaBearScalarExt3>::new(&pp, &[&poly_evals]);
        let commitment = prover.commit();
        let commit_time = t0.elapsed();

        // Open
        let mut transcript = Transcript::new();
        let mut buffer = vec![0u8; HASH_SIZE];
        commitment.serialize_into(&mut buffer);
        transcript.append_u8_slice(&buffer, HASH_SIZE);
        transcript.append_f(eval);

        let t1 = Instant::now();
        DeepFoldMamaBearProver::open(&pp, &[&prover], point_mont.clone(), &mut transcript);
        let open_time = t1.elapsed();

        let proof_size = transcript.proof.bytes.len();
        (proof_size, commit_time, open_time)
    }

    /// Prints a split-level cost table; asserts nothing. Kept as an opt-in
    /// measurement harness, like the `microbench_*` tests below.
    ///
    /// `#[ignore]` is load-bearing, not cosmetic: the sweep commits at nv up to
    /// 24 across seven split levels, so its resident set runs to tens of GiB.
    /// Left enabled it is the one test in this crate that a machine with less
    /// memory than the reference box cannot finish, and it fails by SIGKILL --
    /// which takes the whole test binary down and reads as a broken artifact
    /// rather than as one oversized measurement. Run it explicitly with
    /// `cargo test --release -p poly_commit --lib -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn analysis_split_level_sweep() {
        eprintln!("\n{}", "=".repeat(90));
        eprintln!(
            "  Split-Level Sweep: Proof Size + Prover Time (Ext3, query_num=34, code_rate=3)"
        );
        eprintln!("{}", "=".repeat(90));

        for nv in [18, 20, 22, 24] {
            let max_split = nv.min(6);
            eprintln!("\n  NV={nv}:");
            eprintln!("  {:-<80}", "");
            eprintln!(
                "  {:>7} | {:>12} | {:>12} | {:>12} | {:>12} | {:>8}",
                "split", "proof (KB)", "commit (ms)", "open (ms)", "total (ms)", "vs s=0"
            );

            let mut base_total = 0.0f64;

            for split in 0..=max_split {
                let (proof_size, commit_time, open_time) = measure_ext3(nv, split);
                let commit_ms = commit_time.as_secs_f64() * 1000.0;
                let open_ms = open_time.as_secs_f64() * 1000.0;
                let total_ms = commit_ms + open_ms;
                let proof_kb = proof_size as f64 / 1024.0;

                if split == 0 {
                    base_total = total_ms;
                }
                let speedup = if base_total > 0.0 {
                    base_total / total_ms
                } else {
                    1.0
                };

                eprintln!(
                    "  {:>7} | {:>10.1} KB | {:>10.1} ms | {:>10.1} ms | {:>10.1} ms | {:>6.2}x",
                    split, proof_kb, commit_ms, open_ms, total_ms, speedup
                );
            }
        }
        eprintln!();
    }

    // The detailed open/commit breakdown that used to live here was tied to the
    // pre-rewrite scalar pipeline (`fold_multilinear_mont`,
    // `evaluate_next_domain_mont`, `FriFoldResult::from_mont_values`, etc.).
    // The same end-to-end profile is now produced by the
    // `profile_hp_df_mamabear` binary in `hyperplonk/src/bin/`,
    // which times the open path via the `OpenTimings` substages and runs
    // through the actual production code path.
}
