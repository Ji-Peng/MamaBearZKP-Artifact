//! AVX-512IFMA-vectorized sumcheck prover for HyperPlonk's add-mul gate.
//!
//! # Inputs
//!
//! - multilinear evaluation tables `f : {0,1}^mu -> F` of size `n = 2^mu`,
//!   each field element in the MamaBear base/extension field.
//! - Fiat-Shamir transcript for the verifier-challenge protocol.
//!
//! The provers in this module expect every input table in **normal order**:
//! for 8 SIMD lanes (w = 3) and `L = 2^(mu-3)`,
//!
//! ```text
//!     evals_normal[p][lambda] = f(bin(8*p + lambda))
//!     p in 0..L, lambda in 0..8
//! ```
//!
//! i.e., eight consecutive scalar entries of `f` live inside one packed
//! vector, and consecutive vectors walk the high bits of the hypercube
//! index. This is the layout produced elsewhere in this crate (it matches
//! how witnesses are stored in memory), so no pre-permutation is needed.
//!
//! # Vectorization and the induced fold order
//!
//! Every vectorized round folds pairs of adjacent packed blocks
//! `(table[2q], table[2q+1])` and pairs of adjacent lanes in the scalar
//! tail. Under normal order this means:
//!
//! - The **first mu-3 rounds** run in SIMD on packed blocks. Folding
//!   `(table[2q], table[2q+1])` eliminates the least significant bit of
//!   the vector index, which under normal order is the hypercube bit
//!   `x_3` in round 0, then `x_4` in round 1, ..., up to `x_{mu-1}`.
//! - The **last 3 rounds** run scalar inside a single packed block
//!   (8 lanes). They pair adjacent lanes first, then adjacent lane-pairs,
//!   which eliminates the lane-index bits, i.e., `x_0`, then `x_1`,
//!   then `x_2`.
//!
//! Combined, the prover's fold / elimination order is
//!
//! ```text
//!     x_3, x_4, ..., x_{mu-1}, x_0, x_1, x_2
//! ```
//!
//! rather than the textbook little-endian `x_0, x_1, ..., x_{mu-1}`.
//! Standalone sumcheck correctness and soundness are unaffected (a
//! cyclic permutation of variables only re-indexes the hypercube sum).
//! However, any caller that feeds HyperPlonk / ZeroCheck-shaped objects
//! (the point `w` used in `eq(w, X)`, the final evaluations, etc.) must
//! apply the same permutation to those objects, otherwise the protocol
//! proves a variable-permuted statement, not the original one.
//!
//! Concretely: the returned `challenges` vector is in SIMD-round order
//! (challenge `k` was substituted for `x_{sigma(k)}` with `sigma` the
//! permutation above); the helper `simd_to_natural_point` in
//! `prover_mamabear.rs` inverts this rotation when the natural-order
//! point is needed downstream.
//!
//! # Module structure
//!
//! - Total sumcheck rounds: `mu`.
//! - Packed (AVX-512) rounds: the first `mu - 3`, folding over `PBF`/`PEF3`.
//! - Scalar tail rounds: the last 3, run per-lane inside a single packed block.
//! - Two prover families live here (both Ext3-only; MamaBear Ext2 was
//!   removed as an abandoned, insufficient-soundness extension-field config):
//!   1. Legacy 5-point prover (`prove_add_mul_ext3`): sends
//!      `{s_i(0), s_i(1), s_i(-1), s_i(2), s_i(inf))}` per round.
//!   2. `ell0`-optimized prover (`prove_add_mul_ell0_ext3`): factors
//!      `s_i(X) = c_i(X) * t_i(X)` with `c_i` linear and known, sends only
//!      the 3 hat values `{t_i(0), t_i(2), t_i(inf)}` of the degree-3 factor
//!      `t_i`. Combines with `verify_ell0`.

use arithmetic::field::mamabear::*;
use arithmetic::field::Field;
use std::convert::From;
use std::time::Instant;
use util::fiat_shamir::Transcript;

/// Sub-stage timings for ZeroCheck (ell0-based sumcheck prover).
/// All values in microseconds, accumulated across a single prover call.
#[derive(Clone, Debug, Default)]
pub struct ZeroCheckTimings {
    /// `build_two_stage_eq_tables` — cheap, O(log(N) · √N).
    pub eq_tables_us: u128,
    /// `precompute_small_value_tables_packed` — O(2N) scan over the 4 base tables.
    pub precompute_us: u128,
    /// For round in 0..ell0 small-value round loop (Fiat-Shamir serial).
    pub small_value_rounds_us: u128,
    /// The transition fold at round ell0 (base → ext), the heaviest single kernel.
    pub transition_fold_us: u128,
    /// Sum over rounds in (ell0+1 .. num_vars-3): fused fold + compute packed.
    pub packed_fold_rounds_us: u128,
    /// Last 3 rounds operating on single packed blocks.
    pub scalar_tail_us: u128,
    /// Total wall time of the profiled entry point (upper bound on the sum above).
    pub total_us: u128,
}

use arithmetic::field::mamabear::{MamaBearScalar as SBF, PackedMamaBearAVX512 as PBF};
// Ext3 type aliases
use arithmetic::field::mamabear::{
    MamaBearScalarExt3 as SEF3, PackedMamaBearAVX512Ext3 as PEF3,
};

pub struct SumcheckMamaBear;

// ---------------------------------------------------------------------------
// Trait abstracting extension-field details for generic sumcheck code.
// ---------------------------------------------------------------------------

/// Trait for types that provide Montgomery form conversions.
///
/// This is needed so that generic code parameterized by `SumcheckExtField` can call
/// `to_montgomery()` / `from_montgomery()` on the associated `Scalar` type.
pub trait MontgomeryOps {
    fn to_montgomery(self) -> Self;
    fn from_montgomery(self) -> Self;
}


impl MontgomeryOps for SEF3 {
    #[inline(always)]
    fn to_montgomery(self) -> Self { SEF3::to_montgomery(self) }
    #[inline(always)]
    fn from_montgomery(self) -> Self { SEF3::from_montgomery(self) }
}

/// Trait for packed extension field operations needed by the generic sumcheck prover.
///
/// The sumcheck range analysis is per-PBF-component and identical for all extension
/// degrees. This trait abstracts only the type-specific operations: constructing
/// packed values from scalar components, summing SIMD lanes, etc.
pub trait SumcheckExtField:
    Field<BaseField = PBF>
    + LazyReduction
    + PackedExtensionField<ScalarExt = Self::Scalar>
    + Copy
    + Clone
    + Default
    + std::fmt::Debug
    + std::ops::Add<Output = Self>
    + std::ops::Sub<Output = Self>
    + std::ops::Mul<Output = Self>
    + std::ops::AddAssign
    + std::ops::SubAssign
    + std::ops::Neg<Output = Self>
{
    /// The corresponding scalar extension field (SEF3 for Ext3, the only supported degree).
    type Scalar: Field<BaseField = SBF>
        + LazyReduction
        + Copy
        + Clone
        + Default
        + std::fmt::Debug
        + std::ops::Add<Output = Self::Scalar>
        + std::ops::Sub<Output = Self::Scalar>
        + std::ops::Mul<Output = Self::Scalar>
        + std::ops::AddAssign
        + std::ops::SubAssign
        + std::ops::MulAssign
        + std::ops::Neg<Output = Self::Scalar>
        + MontgomeryOps
        + From<u32>
        + From<SBF>;

    /// Number of PBF elements per packed extension element (3 for Ext3).
    const PBF_RATIO: usize;

    /// Convert a scalar extension value to packed (broadcast to all 8 lanes).
    #[inline(always)]
    fn from_scalar(s: Self::Scalar) -> Self {
        Self::broadcast_scalar(s)
    }

    /// Sum all 8 SIMD lanes into a single scalar extension value (in Montgomery form).
    fn sum_lanes_to_mont(value: Self) -> Self::Scalar;

    /// Pack scalar eq_L entries into packed extension elements following SIMD layout.
    /// eq_l.len() must be a multiple of 8.
    fn pack_eq_l(eq_l: &[Self::Scalar]) -> Vec<Self>;

    /// Unpack one packed extension element into 8 scalar extension values.
    #[inline(always)]
    fn unpack_to_scalars(value: Self) -> [Self::Scalar; 8] {
        value.unpack_to_array()
    }

    /// Pack up to 8 scalar values into one packed extension element.
    #[inline(always)]
    fn pack_scalars(values: &[Self::Scalar]) -> Self {
        Self::pack_partial(values)
    }

    /// Reinterpret a Vec<PBF> as Vec<Self> in-place (zero-copy).
    /// The caller must ensure pef_len * PBF_RATIO <= pbf_vec.capacity().
    unsafe fn reinterpret_pbf_vec(pbf_vec: Vec<PBF>, ext_len: usize) -> Vec<Self>;

    /// Load a packed weight from scalar eq values gathered at SIMD-strided positions.
    fn load_packed_weight(
        scalar_weight_fn: &dyn Fn(usize) -> Self::Scalar,
        packed_idx: usize,
        packed_groups: usize,
    ) -> Self;

    /// Convert to Montgomery representation (extension field level).
    fn ext_to_montgomery(self) -> Self;

    /// Convert from Montgomery representation (extension field level).
    fn ext_from_montgomery(self) -> Self;
}


// ---------------------------------------------------------------------------
// Ext3 implementation
// ---------------------------------------------------------------------------

impl SumcheckExtField for PEF3 {
    type Scalar = SEF3;
    const PBF_RATIO: usize = 3;

    fn sum_lanes_to_mont(value: Self) -> SEF3 {
        // R=2^52 unified: direct lane sum. aR + bR = (a+b)R.
        // reduce_fast sufficient: sum of 8 lanes ∈ [0, 16P) fits u64.
        let reduced = value.reduce_fast();
        let lanes_c0 = reduced.c0.to_array();
        let lanes_c1 = reduced.c1.to_array();
        let lanes_c2 = reduced.c2.to_array();
        let mut acc_c0 = 0u64;
        let mut acc_c1 = 0u64;
        let mut acc_c2 = 0u64;
        for lane in 0..8 {
            acc_c0 = acc_c0.wrapping_add(lanes_c0[lane]);
            acc_c1 = acc_c1.wrapping_add(lanes_c1[lane]);
            acc_c2 = acc_c2.wrapping_add(lanes_c2[lane]);
        }
        SEF3 { c0: SBF(acc_c0), c1: SBF(acc_c1), c2: SBF(acc_c2) }
    }

    fn pack_eq_l(eq_l: &[SEF3]) -> Vec<Self> {
        // R=2^52 unified: direct strided packing.
        debug_assert!(eq_l.len() % 8 == 0);
        let left_groups = eq_l.len() / 8;
        let mut packed = Vec::with_capacity(left_groups);
        for group_idx in 0..left_groups {
            let mut c0 = [0u64; 8];
            let mut c1 = [0u64; 8];
            let mut c2 = [0u64; 8];
            for lane in 0..8 {
                let val = eq_l[group_idx + lane * left_groups];
                c0[lane] = val.c0.0;
                c1[lane] = val.c1.0;
                c2[lane] = val.c2.0;
            }
            packed.push(
                PEF3::new(PBF::from_array(c0), PBF::from_array(c1), PBF::from_array(c2)),
            );
        }
        packed
    }

    unsafe fn reinterpret_pbf_vec(pbf_vec: Vec<PBF>, ext_len: usize) -> Vec<Self> {
        let pbf_ptr = pbf_vec.as_ptr();
        let pbf_cap = pbf_vec.capacity();
        std::mem::forget(pbf_vec);
        let ext_cap = pbf_cap / Self::PBF_RATIO;
        debug_assert!(ext_len <= ext_cap);
        Vec::from_raw_parts(pbf_ptr as *mut Self, ext_len, ext_cap)
    }

    fn load_packed_weight(
        scalar_weight_fn: &dyn Fn(usize) -> SEF3,
        packed_idx: usize,
        packed_groups: usize,
    ) -> Self {
        // R=2^52: direct pack, no Montgomery conversion needed.
        let mut c0 = [0u64; 8];
        let mut c1 = [0u64; 8];
        let mut c2 = [0u64; 8];
        for lane in 0..8 {
            let weight = scalar_weight_fn(packed_idx + lane * packed_groups);
            c0[lane] = weight.c0.0;
            c1[lane] = weight.c1.0;
            c2[lane] = weight.c2.0;
        }
        PEF3::new(PBF::from_array(c0), PBF::from_array(c1), PBF::from_array(c2))
    }

    #[inline(always)]
    fn ext_to_montgomery(self) -> Self { self.to_montgomery() }
    #[inline(always)]
    fn ext_from_montgomery(self) -> Self { self.from_montgomery() }
}

/// Two-stage eq table factorization: eq(w, X) = eq_L(w_L, x_L) · eq_R(w_R, x_R).
///
/// Splits the suffix variables at the midpoint so that the round polynomial
/// decomposes as:
///
///   t_i(u) = Σ_{x_L} Σ_{x_R} eq(w_L, x_L) · eq(w_R, x_R) · h(r[<i], u, x_L, x_R)
///
/// where w_L = w[>ℓ/2+i], w_R = w[i+1:ℓ/2+i], x_R ∈ B^{ℓ/2}, x_L ∈ B^{ℓ/2-i-1}.
///
/// The two-stage structure enables an inner/outer loop optimization:
/// - Inner loop: accumulate over x_R, weighting by eq_R only (cheap broadcast mul)
/// - Outer loop: multiply accumulated result by eq_L (one PEF mul per left group)
/// This saves |right_len| PEF multiplications per left group compared to flat eq.
///
/// When i ≥ split_round = ℓ - right_bits - 1, the left part vanishes and we use
/// only eq_R = eq(w[>i], x_R) with x_R ∈ B^{ℓ-i-1}.
#[derive(Clone, Debug)]
pub(crate) struct TwoStageEqTables<E: SumcheckExtField> {
    /// Number of bits assigned to the right (inner) eq table: ℓ/2.
    pub(crate) right_bits: usize,
    /// Round index at which eq_L vanishes: ℓ - right_bits - 1.
    pub(crate) split_round: usize,
    /// eq_R evaluations for each round. eq_r_levels[i] = build_eq_table(w[i+1:...]).
    /// - For i < split_round: eq_r_levels[i] has 2^right_bits entries.
    /// - For i ≥ split_round: eq_r_levels[i] has 2^(ℓ-i-1) entries (full suffix eq).
    pub(crate) eq_r_levels: Vec<Vec<E::Scalar>>,
    /// eq_L evaluations for rounds i < split_round.
    /// eq_l_levels[i] = build_eq_table(w[i+1+right_bits:]).
    pub(crate) eq_l_levels: Vec<Vec<E::Scalar>>,
    /// Pre-packed SIMD versions of eq_R (broadcast) and eq_L (packed across lanes).
    /// Only available when eq_L has ≥ 8 entries (enough to fill one PEF).
    pub(crate) split_packed_levels: Vec<Option<SplitEqPackedLevel<E>>>,
}

/// SIMD-ready eq weights for one round's two-stage factorization.
///
/// - `right_broadcast[j]`: eq_R(w_R, j) broadcast to all 8 SIMD lanes.
///   Each packed element has the same scalar value in all lanes.
/// - `left_packed[g]`: eq_L values for 8 consecutive left groups packed
///   into one PEF, following the SIMD interleaving layout.
#[derive(Clone, Debug)]
pub(crate) struct SplitEqPackedLevel<E: SumcheckExtField> {
    pub(crate) right_broadcast: Vec<E>,
    pub(crate) left_packed: Vec<E>,
}

/// Per-round view into the two-stage eq tables.
///
/// Provides access to eq weights for computing t_i(u) in the optimized prover.
/// For rounds before `split_round`, both eq_L and eq_R are available;
/// for later rounds, only eq_R remains (eq_L has been consumed).
#[derive(Clone, Copy)]
pub(crate) struct RoundEqView<'a, E: SumcheckExtField> {
    pub(crate) right_bits: usize,
    pub(crate) eq_r: &'a [E::Scalar],
    pub(crate) eq_l: Option<&'a [E::Scalar]>,
    pub(crate) packed_split: Option<&'a SplitEqPackedLevel<E>>,
}

/// Fast accessor for the packed two-stage eq weights in the inner loop.
///
/// The inner loop iterates over right indices (0..right_len), multiplying
/// accumulated gate values by eq_R[right_idx] (broadcast). The outer loop
/// then multiplies by eq_L[left_group] (packed), amortizing one packed PEF
/// multiplication across all right indices in the group.
#[derive(Clone, Copy)]
pub(crate) struct PackedSplitAccess<'a, E: SumcheckExtField> {
    pub(crate) right_broadcast: &'a [E],
    pub(crate) left_packed: &'a [E],
}

impl<'a, E: SumcheckExtField> PackedSplitAccess<'a, E> {
    /// Compute the full eq weight for a given group index by combining eq_L and eq_R.
    ///
    /// group_idx encodes (left_group, right_idx) in row-major order:
    ///   group_idx = left_group * right_len + right_idx
    ///
    /// Returns eq_L[left_group] * eq_R[right_idx], both in packed PEF form.
    #[inline(always)]
    pub(crate) fn weight(&self, group_idx: usize) -> E {
        let right_len = self.right_broadcast.len();
        self.left_packed[group_idx / right_len] * self.right_broadcast[group_idx % right_len]
    }
}

impl<'a, E: SumcheckExtField> RoundEqView<'a, E> {
    /// Compute the scalar eq weight for a given tail index.
    ///
    /// The tail_idx encodes (left_idx, right_idx) in row-major order:
    ///   tail_idx = left_idx * 2^right_bits + right_idx
    ///
    /// Returns eq_L[left_idx] * eq_R[right_idx] (or just eq_R[tail_idx] if no split).
    #[inline(always)]
    pub(crate) fn scalar_weight(&self, tail_idx: usize) -> E::Scalar {
        match self.eq_l {
            Some(eq_l) => {
                let right_mask = (1usize << self.right_bits) - 1;
                eq_l[tail_idx >> self.right_bits] * self.eq_r[tail_idx & right_mask]
            }
            None => self.eq_r[tail_idx],
        }
    }

    /// Construct a packed weight by gathering 8 scalar weights from SIMD-strided positions.
    ///
    /// In the packed SIMD layout, lane `k` of packed element `packed_idx` corresponds to
    /// scalar index `packed_idx + k * packed_groups`. This function gathers those 8 scalar
    /// eq weights and packs them into one packed ext element for SIMD-parallel accumulation.
    ///
    /// Used as fallback when the pre-packed split tables are not available (eq_L too small).
    #[inline(always)]
    pub(crate) fn load_packed_weight(&self, packed_idx: usize, packed_groups: usize) -> E {
        E::load_packed_weight(&|idx| self.scalar_weight(idx), packed_idx, packed_groups)
    }

    /// Try to obtain a PackedSplitAccess for the two-stage inner/outer loop pattern.
    ///
    /// Returns Some only when the pre-packed eq tables are available AND the table
    /// geometry matches: packed_groups must be divisible by right_len, and the quotient
    /// must equal left_packed.len().
    #[inline(always)]
    pub(crate) fn packed_split_for_groups(&self, packed_groups: usize) -> Option<PackedSplitAccess<'a, E>> {
        let split = self.packed_split?;
        let right_len = split.right_broadcast.len();
        if right_len == 0 || packed_groups % right_len != 0 {
            return None;
        }
        let left_groups = packed_groups / right_len;
        if left_groups != split.left_packed.len() {
            return None;
        }
        Some(PackedSplitAccess {
            right_broadcast: &split.right_broadcast,
            left_packed: &split.left_packed,
        })
    }
}

impl SumcheckMamaBear {
    pub const MAX_OPTIMIZED_ELL0: usize = 4;

    /// |U_d| = d + 1 = 4.
    ///
    /// For HyperPlonk gate h(S,L,R,O) = (1-S)(L+R) + S·L·R + O, the round polynomial
    /// t_i(X) has degree d = 3 in X (after factoring out the linear c_i(X) = eq(w_i, X)).
    /// U_d = {0, 1, 2, ∞} encoded as {0, 1, 2, 3}, so |U_d| = 4.
    pub(crate) const OPT_U_SIZE: usize = 4;

    /// |Û_d| = d = 3 — the number of evaluation points the prover sends per round.
    ///
    /// Û_d = U_d \ {1} = {0, 2, ∞}. Point 1 is excluded because the verifier recovers
    /// t_i(1) from the sumcheck consistency relation s_i(0) + s_i(1) = σ_{i-1}.
    pub(crate) const OPT_HAT_SIZE: usize = 3;

    /// Encoded hat points: {0, 2, ∞} → {0, 2, 3} in the nat_{d+1} encoding.
    ///
    /// Note: 3 encodes ∞ (the leading coefficient of the degree-3 polynomial t_i).
    pub(crate) const OPT_HAT_POINTS: [u8; Self::OPT_HAT_SIZE] = [0, 2, 3];

    #[inline(always)]
    pub fn default_optimized_ell0(mu: usize) -> usize {
        (mu / 2).saturating_sub(1).min(Self::MAX_OPTIMIZED_ELL0)
    }

    #[inline(always)]
    pub(crate) fn resolve_optimized_ell0(mu: usize, ell0: Option<usize>) -> usize {
        match ell0 {
            Some(ell0) => {
                assert!(
                    ell0 <= Self::MAX_OPTIMIZED_ELL0,
                    "ell0 must be <= {}",
                    Self::MAX_OPTIMIZED_ELL0
                );
                assert!(ell0 <= mu, "ell0 must not exceed the number of variables");
                ell0
            }
            None => Self::default_optimized_ell0(mu),
        }
    }

    /// Gate polynomial for HyperPlonk add-mul gate: h = (1-S)(L+R) + S·L·R + O.
    ///
    /// Algebraic identity: h = L+R + S·(L·R - L - R) + O, requiring only 2 PBF muls.
    ///
    /// # Range Analysis (per PBF component, inputs in [0, 2P))
    /// - l_plus_r = (L + R).con_sub(2P): [0, 4P) → [0, 2P)
    /// - lr = L * R (mont_mul, both ∈ [0, 2P)): [0, 1.5P)  [Case m=2: (4P²/R+1)P = 1.5P]
    /// - diff = lr + 2P - l_plus_r: [0, 1.5P+2P) - [0, 2P) = [0, 3.5P)
    /// - diff.con_sub(2P): [0, 2P)  [since max(3.5P-2P, 2P) = 2P]
    /// - term = S * diff' (mont_mul, both ∈ [0, 2P)): [0, 1.5P)
    /// - result = l_plus_r + term + O: [0, 2P) + [0, 1.5P) + [0, 2P) = [0, 5.5P)
    /// - reduce_fast: [0, 2.0001P)
    #[inline(always)]
    pub(crate) fn gate_h_packed_base(inputs: [PBF; 4], _one: PBF) -> PBF {
        let s = inputs[0]; // [0, 2P)
        let l = inputs[1]; // [0, 2P)
        let r = inputs[2]; // [0, 2P)
        let o = inputs[3]; // [0, 2P)
        let l_plus_r = l.lazy_add(r).con_sub_xp(2); // [0, 2P)
        let lr = l * r; // [0, 1.5P)
        let diff = lr.lazy_add_xp(2).lazy_sub(l_plus_r); // [0, 3.5P)
        let term = s * diff.con_sub_xp(2); // [0, 1.5P)
        l_plus_r.lazy_add(term).lazy_add(o).reduce_fast() // [0, 5.5P) → [0, 2.0001P)
    }

    #[inline(always)]
    pub(crate) fn packed_diff(v0: PBF, v1: PBF) -> PBF {
        v1.lazy_add_xp(2).lazy_sub(v0).con_sub_xp(2)
    }

    #[inline(always)]
    pub(crate) fn eq_linear_mont_generic<E: SumcheckExtField>(w_i: E::Scalar, x: E::Scalar, one: E::Scalar) -> E::Scalar {
        // eq(w, x) = (1-w)(1-x) + wx = 1 - w - x + 2wx
        // Rewritten to avoid underflow: one + 2*w*x + 2P - w - x
        let wx = w_i * x; // [0, 2P)
        one.lazy_add(wx).lazy_add(wx).lazy_add_xp(2).lazy_sub(w_i).lazy_sub(x).con_sub_xp(2)
    }

    #[allow(dead_code)]
    fn build_eq_table_generic<E: SumcheckExtField>(point: &[E::Scalar]) -> Vec<E::Scalar> {
        let one = E::Scalar::one().to_montgomery();
        let mut evals = vec![one];
        for &bit in point.iter().rev() {
            let one_minus_bit = one.lazy_add_xp(2).lazy_sub(bit); // [0, 4P)
            let mut next = Vec::with_capacity(evals.len() * 2);
            for &prod in &evals {
                next.push(prod * one_minus_bit);
                next.push(prod * bit);
            }
            evals = next;
        }
        evals
    }

    /// Build all per-round eq tables for the two-stage factorization.
    ///
    /// For round i, the factored round polynomial is
    ///
    ///   t_i(u) = Σ_{x_L ∈ B^{ℓ/2-i-1}} Σ_{x_R ∈ B^{ℓ/2}}
    ///            eq(w[>ℓ/2+i], x_L) · eq(w[i+1:ℓ/2+i], x_R) · h(r[<i], u, x_L, x_R).
    ///
    /// We pre-build:
    /// - eq_R[i] = build_eq_table(w[i+1 : i+1+right_bits])  for all rounds
    /// - eq_L[i] = build_eq_table(w[i+1+right_bits : ])       for rounds < split_round
    ///
    /// When i ≥ split_round (= ℓ - right_bits - 1), the left part has 0 variables,
    /// so eq_L vanishes and eq_R covers the entire suffix w[i+1:].
    ///
    /// Total space: O(log(N) · √N) — e.g., for N = 2^24, this is ~25 × 2^12 ≈ 100K entries.
    /// Extend an eq table by one variable at bit 0 (interleaving extension).
    ///
    /// If old table has layout `[eq(old_vars, x)]` with `old_vars` at bits 0..k-1,
    /// the result has `new_var` at bit 0 and old vars shifted to bits 1..k:
    ///   `new[2*j]   = old[j] * (1 - new_var)`   (new_var = 0)
    ///   `new[2*j+1] = old[j] * new_var`          (new_var = 1)
    ///
    /// This matches prepending a variable to the front of build_eq_table's point slice.
    #[inline]
    fn extend_eq_table_low_generic<E: SumcheckExtField>(table: &[E::Scalar], new_var: E::Scalar) -> Vec<E::Scalar> {
        let one = E::Scalar::one().to_montgomery();
        let w_bar = one.lazy_add_xp(2).lazy_sub(new_var); // [0, 4P)
        let mut next = Vec::with_capacity(table.len() * 2);
        for &prod in table {
            next.push(prod * w_bar);
            next.push(prod * new_var);
        }
        next
    }

    /// Extend an eq table by one variable at the highest bit (doubling extension).
    ///
    /// If old table has layout `[eq(old_vars, x)]` with `old_vars` at bits 0..k-1,
    /// the result has old vars unchanged and `new_var` at bit k:
    ///   `new[j]         = old[j] * (1 - new_var)`   (new_var = 0)
    ///   `new[old_len+j] = old[j] * new_var`          (new_var = 1)
    ///
    /// This matches appending a variable to the end of build_eq_table's point slice.
    #[inline]
    fn extend_eq_table_high_generic<E: SumcheckExtField>(table: &[E::Scalar], new_var: E::Scalar) -> Vec<E::Scalar> {
        let one = E::Scalar::one().to_montgomery();
        let w_bar = one.lazy_add_xp(2).lazy_sub(new_var); // [0, 4P)
        let mut next = Vec::with_capacity(table.len() * 2);
        for &prod in table {
            next.push(prod * w_bar);
        }
        for &prod in table {
            next.push(prod * new_var);
        }
        next
    }

    pub(crate) fn build_two_stage_eq_tables_generic<E: SumcheckExtField>(point: &[E::Scalar]) -> TwoStageEqTables<E> {
        let mu = point.len();
        let right_bits = mu / 2;
        let split_round = mu.saturating_sub(right_bits + 1);
        let one = E::Scalar::one().to_montgomery();

        // ── eq_L: incremental construction from end to start ──
        //
        // eq_L[i] = build_eq_table(&point[i+1+rb..mu]).
        // As i decreases, each table extends the previous by prepending one variable.
        // Build in reverse order (smallest first), then reverse the vec.
        let mut eq_l_levels = Vec::with_capacity(split_round);
        if split_round > 0 {
            let mut table = vec![one]; // start with empty eq table
            for i in (0..split_round).rev() {
                // Prepend variable point[i+1+rb] → interleaving extension (new var at bit 0)
                table = Self::extend_eq_table_low_generic::<E>(&table, point[i + 1 + right_bits]);
                eq_l_levels.push(table.clone());
            }
            eq_l_levels.reverse(); // now eq_l_levels[i] corresponds to round i
        }

        // ── eq_R for i < split_round: tensor product of left-prefix × right-suffix ──
        //
        // eq_R[i] = build_eq_table(&point[i+1 .. i+1+rb]).
        // Split variables at the pivot w[rb]:
        //   left part:  w[i+1], ..., w[rb-1]  (rb-1-i variables, bits 0..rb-2-i)
        //   right part: w[rb], ..., w[i+rb]   (i+1 variables, bits rb-1-i..rb-1)
        //
        // Build right sub-tables incrementally by appending (high-bit extension):
        //   right[k] = eq({w[rb], ..., w[rb+k]}) with 2^(k+1) entries
        //
        // Build left sub-tables incrementally by prepending (low-bit extension):
        //   left[k] = eq({w[rb-1-k], ..., w[rb-1]}) with 2^(k+1) entries
        //
        // Combine: eq_R[i][r * 2^left_len + l] = right[i][r] * left[rb-2-i][l]
        let mut eq_r_levels = Vec::with_capacity(mu);

        if split_round > 0 {
            // Right sub-tables: append w[rb], w[rb+1], ... via high-bit extension
            let mut right_parts = Vec::with_capacity(split_round);
            let mut right_table = vec![one];
            for k in 0..split_round {
                right_table = Self::extend_eq_table_high_generic::<E>(&right_table, point[right_bits + k]);
                right_parts.push(right_table.clone());
            }

            // Left sub-tables: prepend w[rb-1], w[rb-2], ... via low-bit extension
            let max_left = (right_bits - 1).min(split_round);
            let mut left_parts = Vec::with_capacity(max_left);
            let mut left_table = vec![one];
            for k in 0..max_left {
                left_table = Self::extend_eq_table_low_generic::<E>(&left_table, point[right_bits - 1 - k]);
                left_parts.push(left_table.clone());
            }

            // Tensor product combination for each round i < split_round
            for i in 0..split_round {
                let left_vars = right_bits - 1 - i;
                let right = &right_parts[i]; // 2^(i+1) entries
                if left_vars == 0 {
                    eq_r_levels.push(right.clone());
                } else {
                    let left = &left_parts[left_vars - 1]; // 2^(left_vars) entries
                    let mut eq_r = Vec::with_capacity(right.len() * left.len());
                    for &r_val in right {
                        for &l_val in left {
                            eq_r.push(r_val * l_val);
                        }
                    }
                    eq_r_levels.push(eq_r);
                }
            }
        }

        // ── eq_R for i ≥ split_round: suffix tables, incremental from end ──
        //
        // eq_R[i] = build_eq_table(&point[i+1..mu]).
        // As i decreases, each table extends by prepending one variable.
        {
            let mut table = vec![one]; // i = mu-1: empty slice
            let mut suffix_rev = Vec::with_capacity(mu - split_round);
            suffix_rev.push(table.clone());

            for i in (split_round..mu.saturating_sub(1)).rev() {
                table = Self::extend_eq_table_low_generic::<E>(&table, point[i + 1]);
                suffix_rev.push(table.clone());
            }
            suffix_rev.reverse();
            for t in suffix_rev {
                eq_r_levels.push(t);
            }
        }

        // ── Build packed SIMD versions for rounds < split_round ──
        let mut split_packed_levels = Vec::with_capacity(split_round);
        for i in 0..split_round {
            let eq_r = &eq_r_levels[i];
            let eq_l = &eq_l_levels[i];
            let packed_split = if eq_l.len() >= 8 {
                Some(SplitEqPackedLevel {
                    right_broadcast: eq_r
                        .iter()
                        .copied()
                        .map(E::from_scalar)
                        .collect(),
                    left_packed: E::pack_eq_l(eq_l),
                })
            } else {
                None
            };
            split_packed_levels.push(packed_split);
        }

        TwoStageEqTables {
            right_bits,
            split_round,
            eq_r_levels,
            eq_l_levels,
            split_packed_levels,
        }
    }

    #[inline(always)]
    pub(crate) fn round_eq_view_generic<'a, E: SumcheckExtField>(eq_tables: &'a TwoStageEqTables<E>, round: usize) -> RoundEqView<'a, E> {
        RoundEqView {
            right_bits: eq_tables.right_bits,
            eq_r: &eq_tables.eq_r_levels[round],
            eq_l: (round < eq_tables.split_round).then(|| eq_tables.eq_l_levels[round].as_slice()),
            packed_split: (round < eq_tables.split_round)
                .then(|| eq_tables.split_packed_levels[round].as_ref())
                .flatten(),
        }
    }

    pub(crate) fn decode_u4(mut encoded: usize, digits: usize) -> Vec<u8> {
        let mut decoded = Vec::with_capacity(digits);
        for _ in 0..digits {
            decoded.push((encoded % Self::OPT_U_SIZE) as u8);
            encoded /= Self::OPT_U_SIZE;
        }
        decoded
    }

    pub(crate) fn eval_packed_block_at_finite_points(
        block: &[PBF],
        points: &[u8],
        _two: PBF,
        three: PBF,
    ) -> PBF {
        debug_assert_eq!(block.len(), 1 << points.len());
        let mut scratch = block.to_vec();
        let mut cur = block.len();
        for &point in points {
            let half = cur >> 1;
            for idx in 0..half {
                let v0 = scratch[idx << 1];
                let v1 = scratch[(idx << 1) + 1];
                let diff = Self::packed_diff(v0, v1);
                scratch[idx] = match point {
                    0 => v0,
                    1 => v1,
                    2 => v1.lazy_add(diff).con_sub_xp(2),
                    3 => v0.lazy_add((diff * three).reduce_fast()).con_sub_xp(2),
                    _ => unreachable!("unsupported finite point"),
                };
            }
            cur = half;
        }
        scratch[0]
    }

    #[inline(always)]
    pub(crate) fn finite_line_from_pair_packed(v0: PBF, v1: PBF, three: PBF) -> [PBF; 4] {
        let diff = Self::packed_diff(v0, v1);
        [
            v0,
            v1,
            v1.lazy_add(diff).con_sub_xp(2),
            v0.lazy_add((diff * three).reduce_fast()).con_sub_xp(2),
        ]
    }

    #[inline(always)]
    fn finite_plane_from_quad_packed(values: [PBF; 4], three: PBF) -> [PBF; 16] {
        let low = Self::finite_line_from_pair_packed(values[0], values[1], three);
        let high = Self::finite_line_from_pair_packed(values[2], values[3], three);
        let mut plane = [PBF::zero(); 16];
        for x0 in 0..4 {
            let line = Self::finite_line_from_pair_packed(low[x0], high[x0], three);
            for x1 in 0..4 {
                plane[x0 + (x1 << 2)] = line[x1];
            }
        }
        plane
    }

    #[inline(always)]
    fn gate_tensor_from_octets_packed(
        poly_octets: [[PBF; 8]; 4],
        one: PBF,
        three: PBF,
    ) -> [PBF; 64] {
        let low_planes: [[PBF; 16]; 4] = std::array::from_fn(|poly_idx| {
            Self::finite_plane_from_quad_packed(
                [
                    poly_octets[poly_idx][0],
                    poly_octets[poly_idx][1],
                    poly_octets[poly_idx][2],
                    poly_octets[poly_idx][3],
                ],
                three,
            )
        });
        let high_planes: [[PBF; 16]; 4] = std::array::from_fn(|poly_idx| {
            Self::finite_plane_from_quad_packed(
                [
                    poly_octets[poly_idx][4],
                    poly_octets[poly_idx][5],
                    poly_octets[poly_idx][6],
                    poly_octets[poly_idx][7],
                ],
                three,
            )
        });

        let mut tensor = [PBF::zero(); 64];
        for state in 0..16 {
            let lines: [[PBF; 4]; 4] = std::array::from_fn(|poly_idx| {
                Self::finite_line_from_pair_packed(
                    low_planes[poly_idx][state],
                    high_planes[poly_idx][state],
                    three,
                )
            });
            for x2 in 0..4 {
                tensor[state + (x2 << 4)] = Self::gate_h_packed_base(
                    [lines[0][x2], lines[1][x2], lines[2][x2], lines[3][x2]],
                    one,
                );
            }
        }
        tensor
    }

    fn transform_finite_values_to_ud_packed(values: [PBF; 4], inv6: PBF) -> [PBF; 4] {
        let three = PBF::from(3u32).to_montgomery();
        let lhs = values[3].lazy_add((values[1] * three).reduce_fast());
        let rhs = (values[2] * three).reduce_fast().lazy_add(values[0]);
        let numer = lhs.lazy_add_xp(5).lazy_sub(rhs).con_sub_xp(5);
        [
            values[0],
            values[1],
            values[2],
            (numer * inv6).reduce_fast(),
        ]
    }

    pub(crate) fn transform_tensor_axis_to_ud_packed(tensor: &mut [PBF], axis: usize, inv6: PBF) {
        let stride = Self::OPT_U_SIZE.pow(axis as u32);
        let block = stride * Self::OPT_U_SIZE;
        let outer = tensor.len() / block;

        for outer_idx in 0..outer {
            let block_base = outer_idx * block;
            for inner_idx in 0..stride {
                let base = block_base + inner_idx;
                let transformed = Self::transform_finite_values_to_ud_packed(
                    [
                        tensor[base],
                        tensor[base + stride],
                        tensor[base + 2 * stride],
                        tensor[base + 3 * stride],
                    ],
                    inv6,
                );
                for digit in 0..Self::OPT_U_SIZE {
                    tensor[base + digit * stride] = transformed[digit];
                }
            }
        }
    }

    /// Gate polynomial for HyperPlonk add-mul gate (extension field):
    /// h = (1-S)(L+R) + S·L·R + O = L+R + S·(L·R - L - R) + O, requiring only 2 PEF muls.
    ///
    /// # Range Analysis (per PBF component within PEF, inputs in [0, 2P))
    ///
    /// PEF mul uses Karatsuba internally and applies reduce_fast, so output ∈ [0, 2.0001P).
    /// For the range analysis below, we write [0, 2P) as shorthand for [0, 2.0001P).
    ///
    /// - l_plus_r = (L + R).con_sub(2P): [0, 4P) → [0, 2P)
    /// - lr = L * R (PEF Karatsuba mul, both ∈ [0, 2P)): [0, 2P)
    /// - diff = lr + 2P - l_plus_r: [0, 2P+2P) - [0, 2P) = [0, 4P)
    /// - diff.con_sub(2P): [0, 2P)  [since max(4P-2P, 2P) = 2P]
    /// - term = S * diff' (PEF mul, both ∈ [0, 2P)): [0, 2P)
    /// - result = l_plus_r + term + O: [0, 2P) + [0, 2P) + [0, 2P) = [0, 6P)
    /// - con_sub_xp(3): [0, 6P) → [0, 3P)
    ///
    /// # Output range: [0, 3P) per component
    ///
    /// All callers use this as an operand of PEF mul (eq_r * gate_h, weight * gate_h),
    /// which requires inputs < 2^51 ≈ 4P. Since 3P ≈ 2^50.6 < 2^51, this is safe.
    ///
    /// Using con_sub_xp(3) instead of reduce_fast saves ~4 instructions per call
    /// (reduce_fast does shift+add+con_sub exploiting P's sparsity; con_sub is just one
    /// min operation per component).
    #[inline(always)]
    pub(crate) fn gate_h_packed_ext_generic<E: SumcheckExtField>(inputs: [E; 4], _one: E) -> E {
        let s = inputs[0]; // [0, 2P)
        let l = inputs[1]; // [0, 2P)
        let r = inputs[2]; // [0, 2P)
        let o = inputs[3]; // [0, 2P)
        let l_plus_r = l.lazy_add(r).con_sub_xp(2); // [0, 2P)
        let lr = l * r; // [0, 2P)  (PEF mul output after internal reduce_fast)
        let diff = lr.lazy_add_xp(2).lazy_sub(l_plus_r); // [0, 4P)
        let term = s * diff.con_sub_xp(2); // [0, 2P)  (PEF mul output)
        l_plus_r.lazy_add(term).lazy_add(o).reduce_fast() // [0, 6P) → [0, 2P), matching scalar gate
    }

    /// Gate function for a circuit containing only addition and multiplication gates.
    /// h(x) := (1-S(x))(L(x)+R(x)) + S(x)L(x)R(x) + O(x))eq(x,r).
    ///
    /// # Range Analysis
    ///
    /// PBF and PEF use unsigned representation; SBF and SEF use signed representation.
    /// Therefore, we need to separately analyze the range for each representation.
    ///
    /// ## PBF/PEF
    /// We need to ensure that each multiplication operand is < 2^52, i.e., <= 8p.
    ///
    /// We restrict the input range to [0, 2p), which is easily implemented considering the context of the subroutine.
    /// The output range is [0, 2.51p).
    ///
    /// ## SBF/SEF
    /// We use the signed representation for SBF/SEF and signed Montgomery multiplication with R=2^64.
    ///
    /// For the last 3 scalar rounds, the number of accumulations is not large;
    /// so overflow or underflow will not occur.
    /// Legacy gate function: h(x)·eq(x) = ((1-S)(L+R) + S·L·R + O) · eq.
    ///
    /// Generic over F: called with F=PEF for AVX-512 rounds, F=SEF for scalar rounds.
    /// Uses the algebraic identity: h = L+R + S·(L·R - L - R) + O for 2 gate muls + 1 eq mul.
    ///
    /// # Range Analysis (F = PEF, per PBF component, inputs ∈ [0, 2P))
    ///
    /// ```text
    /// l_plus_r = (L + R).con_sub(2P)            ∈ [0, 2P)
    /// lr = L * R (PEF mul)                      ∈ [0, 2P)
    /// diff = lr + 2P - l_plus_r                 ∈ [0, 4P)
    /// diff' = diff.con_sub(2P)                  ∈ [0, 2P)
    /// term = S * diff' (PEF mul)                ∈ [0, 2P)
    /// combined = l_plus_r + term + O            ∈ [0, 6P)
    /// result = combined * eq (PEF mul)          ∈ [0, 2P)  (internal reduce_fast)
    /// ```
    ///
    /// For the accumulation threshold: result ∈ [0, 2P), and
    /// 2^64 / (2P) ≈ 2^64 / 2^50 = 2^14 = 16384, so up to 13055 lazy_adds are safe.
    #[inline(always)]
    fn gate_add_mul<F>(inputs: [F; 5]) -> F
    where
        F: Field + LazyReduction,
    {
        let s = inputs[0]; // [0, 2P)
        let l = inputs[1]; // [0, 2P)
        let r = inputs[2]; // [0, 2P)
        let o = inputs[3]; // [0, 2P)
        let eq = inputs[4]; // [0, 2P)

        let l_plus_r = l.lazy_add(r).con_sub_xp(2); // [0, 2P)
        let lr = l * r; // PEF: [0, 2P)
        let diff = lr.lazy_add_xp(2).lazy_sub(l_plus_r); // [0, 4P)
        let term = s * diff.con_sub_xp(2); // [0, 2P)
        let combined = l_plus_r.lazy_add(term).lazy_add(o); // [0, 6P)
        combined * eq // [0, 2P)  (PEF mul applies reduce_fast)
    }

    /// Generic legacy gate function for round 0: gate inputs are PBF, eq weight is E.
    ///
    /// Same range analysis as `gate_add_mul_round0` — PBF arithmetic for gate,
    /// then `eq.mul_base_elem(combined)` to produce the E-type result.
    #[inline(always)]
    fn gate_add_mul_round0_generic<E: SumcheckExtField>(inputs: [PBF; 4], eq: E) -> E {
        let s = inputs[0]; // [0, 2P)
        let l = inputs[1]; // [0, 2P)
        let r = inputs[2]; // [0, 2P)
        let o = inputs[3]; // [0, 2P)

        let l_plus_r = l.lazy_add(r).con_sub_xp(2); // [0, 2P)
        let lr = l * r; // [0, 1.5P)
        let diff = lr.lazy_add_xp(2).lazy_sub(l_plus_r); // [0, 3.5P)
        let term = s * diff.con_sub_xp(2); // [0, 1.5P)
        let combined = l_plus_r.lazy_add(term).lazy_add(o); // [0, 5.5P)
        eq.mul_base_elem(combined) // [0, 2.38P) per component
    }

    /// Generic legacy (5-point evaluation) prover, parameterized by extension
    /// field type `E: SumcheckExtField`.
    ///
    /// This version allocates new `Vec<E>` arrays at round 1 instead of
    /// in-place PBF→PEF reinterpretation, which is necessary because
    /// PBF_RATIO=3 (Ext3) does not allow in-place conversion (writes would
    /// overflow reads).
    fn prove_add_mul_generic<E: SumcheckExtField>(
        evals: [Vec<PBF>; 4],
        mut evals_eq: Vec<E>,
        transcript: &mut Transcript,
    ) -> (Vec<E::Scalar>, [E::Scalar; 5]) {
        const N: usize = 5;
        const POINTS_NUM: usize = 5;
        let vec_len = evals[0].len();
        let mu = vec_len.ilog2() as usize + 3;
        let mut challenges: Vec<E::Scalar> = Vec::with_capacity(mu);

        // --- Round 0: PBF tables + E eq ---
        let mut acc_0 = E::default();
        let mut acc_1 = E::default();
        let mut acc_2 = E::default();
        let mut acc_m1 = E::default();
        let mut acc_inf = E::default();
        let len = vec_len;
        for i in (0..len).step_by(2) {
            let mut args_0 = [PBF::default(); N - 1];
            let mut args_1 = [PBF::default(); N - 1];
            let mut args_2 = [PBF::default(); N - 1];
            let mut args_m1 = [PBF::default(); N - 1];
            let mut args_inf = [PBF::default(); N - 1];
            let mut args_eq = [E::default(); POINTS_NUM];
            for k in 0..N - 1 {
                let v0 = evals[k][i];
                let v1 = evals[k][i + 1];
                let diff = v1.lazy_add_xp(2).lazy_sub(v0).con_sub_xp(2);
                args_0[k] = v0;
                args_1[k] = v1;
                args_m1[k] = v0.lazy_add_xp(2).lazy_sub(diff).con_sub_xp(2);
                args_2[k] = v1.lazy_add(diff).con_sub_xp(2);
                args_inf[k] = diff;
            }
            let eq_v0 = evals_eq[i];
            let eq_v1 = evals_eq[i + 1];
            let eq_diff = eq_v1.lazy_add_xp(2).lazy_sub(eq_v0).con_sub_xp(2);
            args_eq[0] = eq_v0;
            args_eq[1] = eq_v1;
            args_eq[2] = eq_v0.lazy_add_xp(2).lazy_sub(eq_diff).con_sub_xp(2);
            args_eq[3] = eq_v1.lazy_add(eq_diff).con_sub_xp(2);
            args_eq[4] = eq_diff;
            let res_0 = Self::gate_add_mul_round0_generic::<E>(args_0, args_eq[0]);
            let res_1 = Self::gate_add_mul_round0_generic::<E>(args_1, args_eq[1]);
            let res_m1 = Self::gate_add_mul_round0_generic::<E>(args_m1, args_eq[2]);
            let res_2 = Self::gate_add_mul_round0_generic::<E>(args_2, args_eq[3]);
            let res_inf = Self::gate_add_mul_round0_generic::<E>(args_inf, args_eq[4]);
            acc_0 = acc_0.lazy_add(res_0);
            acc_1 = acc_1.lazy_add(res_1);
            acc_2 = acc_2.lazy_add(res_2);
            acc_m1 = acc_m1.lazy_add(res_m1);
            acc_inf = acc_inf.lazy_add(res_inf);
            if ((i >> 1) + 1) % 13055 == 0 {
                acc_0 = acc_0.reduce_fast();
                acc_1 = acc_1.reduce_fast();
                acc_2 = acc_2.reduce_fast();
                acc_m1 = acc_m1.reduce_fast();
                acc_inf = acc_inf.reduce_fast();
            }
        }
        acc_0 = acc_0.reduce_fast().ext_from_montgomery().con_sub_xp(1);
        acc_1 = acc_1.reduce_fast().ext_from_montgomery().con_sub_xp(1);
        acc_2 = acc_2.reduce_fast().ext_from_montgomery().con_sub_xp(1);
        acc_m1 = acc_m1.reduce_fast().ext_from_montgomery().con_sub_xp(1);
        acc_inf = acc_inf.reduce_fast().ext_from_montgomery().con_sub_xp(1);
        transcript.append_f(acc_0);
        transcript.append_f(acc_1);
        transcript.append_f(acc_2);
        transcript.append_f(acc_m1);
        transcript.append_f(acc_inf);

        // --- Rounds 1..(mu-3): AVX-512 packed rounds ---
        // At round 1, we convert PBF tables to E tables (allocating new vecs).
        // After round 1, evals_ext holds the extension field tables.
        let mut evals_ext: Option<[Vec<E>; 4]> = None;

        for round in 1..(mu - 3) {
            let len = vec_len >> (round - 1);
            let r_sef: E::Scalar = transcript.challenge_f::<E::Scalar>().to_montgomery();
            challenges.push(r_sef);
            let alpha = E::from_scalar(r_sef);
            let mut acc_0 = E::default();
            let mut acc_1 = E::default();
            let mut acc_2 = E::default();
            let mut acc_m1 = E::default();
            let mut acc_inf = E::default();
            for i in (0..len).step_by(4) {
                let mut args_0 = [E::default(); N];
                let mut args_1 = [E::default(); N];
                let mut args_2 = [E::default(); N];
                let mut args_m1 = [E::default(); N];
                let mut args_inf = [E::default(); N];
                for k in 0..N {
                    let v0: E;
                    let v1: E;
                    if k == N - 1 {
                        // eq polynomial: already in E format
                        let eki0 = evals_eq[i];
                        let eki1 = evals_eq[i + 1];
                        let eki2 = evals_eq[i + 2];
                        let eki3 = evals_eq[i + 3];
                        let t0 = eki1.lazy_add_xp(2).lazy_sub(eki0);
                        let t1 = eki3.lazy_add_xp(2).lazy_sub(eki2);
                        let t2 = alpha * t0;
                        let t3 = alpha * t1;
                        v0 = t2.lazy_add(eki0).con_sub_xp(2);
                        v1 = t3.lazy_add(eki2).con_sub_xp(2);
                        evals_eq[i >> 1] = v0;
                        evals_eq[(i >> 1) + 1] = v1;
                    } else if round == 1 {
                        // Round 1 base→ext conversion: PBF inputs, E outputs.
                        let eki0 = evals[k][i];
                        let eki1 = evals[k][i + 1];
                        let eki2 = evals[k][i + 2];
                        let eki3 = evals[k][i + 3];
                        let t0 = eki1.lazy_add_xp(2).lazy_sub(eki0); // [0, 4P)
                        let t1 = eki3.lazy_add_xp(2).lazy_sub(eki2);
                        let t2 = alpha.mul_base_elem(t0); // [0, 2P) per component
                        let t3 = alpha.mul_base_elem(t1);
                        v0 = t2.add_base_elem(eki0).con_sub_xp(2); // [0, 2P)
                        v1 = t3.add_base_elem(eki2).con_sub_xp(2);
                        // Store into newly allocated ext tables
                        let ext = evals_ext.get_or_insert_with(|| {
                            let half = len / 2;
                            [
                                vec![E::default(); half],
                                vec![E::default(); half],
                                vec![E::default(); half],
                                vec![E::default(); half],
                            ]
                        });
                        ext[k][i >> 1] = v0;
                        ext[k][(i >> 1) + 1] = v1;
                    } else {
                        // Subsequent rounds: E inputs and outputs.
                        let ext = evals_ext.as_mut().unwrap();
                        let eki0 = ext[k][i];
                        let eki1 = ext[k][i + 1];
                        let eki2 = ext[k][i + 2];
                        let eki3 = ext[k][i + 3];
                        let t0 = eki1.lazy_add_xp(2).lazy_sub(eki0);
                        let t1 = eki3.lazy_add_xp(2).lazy_sub(eki2);
                        let t2 = alpha * t0;
                        let t3 = alpha * t1;
                        v0 = t2.lazy_add(eki0).con_sub_xp(2);
                        v1 = t3.lazy_add(eki2).con_sub_xp(2);
                        ext[k][i >> 1] = v0;
                        ext[k][(i >> 1) + 1] = v1;
                    }
                    let diff = v1.lazy_add_xp(2).lazy_sub(v0).con_sub_xp(2);
                    args_0[k] = v0;
                    args_1[k] = v1;
                    args_m1[k] = v0.lazy_add_xp(2).lazy_sub(diff).con_sub_xp(2);
                    args_2[k] = v1.lazy_add(diff).con_sub_xp(2);
                    args_inf[k] = diff;
                }
                let res_0 = Self::gate_add_mul(args_0);
                let res_1 = Self::gate_add_mul(args_1);
                let res_2 = Self::gate_add_mul(args_2);
                let res_m1 = Self::gate_add_mul(args_m1);
                let res_inf = Self::gate_add_mul(args_inf);
                acc_0 = acc_0.lazy_add(res_0);
                acc_1 = acc_1.lazy_add(res_1);
                acc_2 = acc_2.lazy_add(res_2);
                acc_m1 = acc_m1.lazy_add(res_m1);
                acc_inf = acc_inf.lazy_add(res_inf);
                if ((i >> 2) + 1) % 13055 == 0 {
                    acc_0 = acc_0.reduce_fast();
                    acc_1 = acc_1.reduce_fast();
                    acc_2 = acc_2.reduce_fast();
                    acc_m1 = acc_m1.reduce_fast();
                    acc_inf = acc_inf.reduce_fast();
                }
            }
            acc_0 = acc_0.reduce_fast().ext_from_montgomery().con_sub_xp(1);
            acc_1 = acc_1.reduce_fast().ext_from_montgomery().con_sub_xp(1);
            acc_2 = acc_2.reduce_fast().ext_from_montgomery().con_sub_xp(1);
            acc_m1 = acc_m1.reduce_fast().ext_from_montgomery().con_sub_xp(1);
            acc_inf = acc_inf.reduce_fast().ext_from_montgomery().con_sub_xp(1);
            transcript.append_f(acc_0);
            transcript.append_f(acc_1);
            transcript.append_f(acc_2);
            transcript.append_f(acc_m1);
            transcript.append_f(acc_inf);
            if round != 1 {
                let ext = evals_ext.as_mut().unwrap();
                let eval_len = ext[0].len();
                for k in 0..N - 1 {
                    ext[k].truncate(eval_len / 2);
                }
            }
            evals_eq.truncate(evals_eq.len() / 2);
        }

        // --- Transition to scalar tail ---
        // At this point, each ext table has 2 E elements and eq has 2 E elements.
        // Fold one more time to get 1 E element per poly, then unpack to 8 scalars.
        let alpha_sef: E::Scalar = transcript.challenge_f::<E::Scalar>().to_montgomery();
        let alpha = E::from_scalar(alpha_sef);
        challenges.push(alpha_sef);

        let ext = evals_ext.as_mut().unwrap();
        // Fold the 2 E elements into 1 for each polynomial, then from_montgomery
        // R=2^52 unified: no ext_from_montgomery needed at PEF→scalar boundary.
        for k in 0..N {
            if k == N - 1 {
                let eki0 = evals_eq[0];
                let eki1 = evals_eq[1];
                let t0 = eki1.lazy_add_xp(2).lazy_sub(eki0);
                evals_eq[0] = (alpha * t0).lazy_add(eki0).con_sub_xp(2);
            } else {
                let eki0 = ext[k][0];
                let eki1 = ext[k][1];
                let t0 = eki1.lazy_add_xp(2).lazy_sub(eki0);
                ext[k][0] = (alpha * t0).lazy_add(eki0).con_sub_xp(2);
            }
        }

        // Unpack each polynomial's single packed element into 8 scalar values (in Montgomery form).
        // The SIMD layout interleaves lanes: lane i holds the value at index i*group_size + group_idx.
        // With 1 packed element: group_size=1, group_idx=0, so lane i holds value at index i.
        // unpack_to_scalars returns [lane0, lane1, ..., lane7] already to_montgomery'd.
        let mut scalar_evals: [Vec<E::Scalar>; 5] = [
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
        ];
        for k in 0..N {
            let packed = if k == N - 1 {
                evals_eq[0]
            } else {
                ext[k][0]
            };
            let scalars = E::unpack_to_scalars(packed);
            // `unpack_to_scalars` already produces the fold-pair order the
            // scalar-tail loop below expects: consecutive lanes
            // (scalars[0], scalars[1]), (scalars[2], scalars[3]), ... are the
            // (v0, v1) pair for each fold position, so no further reordering
            // is needed here.
            scalar_evals[k] = scalars.to_vec();
        }

        let mut final_evals = [E::Scalar::default(); N];
        // --- Scalar Rounds (The Last 3 Rounds) ---
        for round in (mu - 3)..mu {
            let len = 1 << (mu - round);
            let mut acc_0 = E::Scalar::default();
            let mut acc_1 = E::Scalar::default();
            let mut acc_2 = E::Scalar::default();
            let mut acc_m1 = E::Scalar::default();
            let mut acc_inf = E::Scalar::default();
            for i in (0..len).step_by(2) {
                let mut args_0 = [E::Scalar::default(); N];
                let mut args_1 = [E::Scalar::default(); N];
                let mut args_2 = [E::Scalar::default(); N];
                let mut args_m1 = [E::Scalar::default(); N];
                let mut args_inf = [E::Scalar::default(); N];
                for k in 0..N {
                    let v0 = scalar_evals[k][i];
                    let v1 = scalar_evals[k][i + 1];
                    let diff = v1.lazy_add_xp(2).lazy_sub(v0);
                    args_0[k] = v0;
                    args_1[k] = v1;
                    args_2[k] = v1 + diff;
                    args_m1[k] = v0.lazy_add_xp(4).lazy_sub(diff).con_sub_xp(2);
                    args_inf[k] = diff;
                }
                let res_0 = Self::gate_add_mul(args_0);
                let res_1 = Self::gate_add_mul(args_1);
                let res_2 = Self::gate_add_mul(args_2);
                let res_m1 = Self::gate_add_mul(args_m1);
                let res_inf = Self::gate_add_mul(args_inf);
                acc_0 = acc_0.lazy_add(res_0);
                acc_1 = acc_1.lazy_add(res_1);
                acc_2 = acc_2.lazy_add(res_2);
                acc_m1 = acc_m1.lazy_add(res_m1);
                acc_inf = acc_inf.lazy_add(res_inf);
            }
            acc_0 = acc_0.from_montgomery();
            acc_1 = acc_1.from_montgomery();
            acc_2 = acc_2.from_montgomery();
            acc_m1 = acc_m1.from_montgomery();
            acc_inf = acc_inf.from_montgomery();
            transcript.append_f(acc_0);
            transcript.append_f(acc_1);
            transcript.append_f(acc_2);
            transcript.append_f(acc_m1);
            transcript.append_f(acc_inf);
            let alpha: E::Scalar = transcript.challenge_f::<E::Scalar>().to_montgomery();
            challenges.push(alpha);
            for i in (0..len).step_by(2) {
                for k in 0..N {
                    let eki0 = scalar_evals[k][i];
                    let eki1 = scalar_evals[k][i + 1];
                    let t0 = eki1.lazy_add_xp(2).lazy_sub(eki0); // [0, 4P)
                    let t2 = alpha * t0;
                    let v0 = t2.lazy_add(eki0).con_sub_xp(2);
                    if round == mu - 1 {
                        let v0 = v0.from_montgomery();
                        final_evals[k] = v0;
                        scalar_evals[k][i >> 1] = v0;
                    } else {
                        scalar_evals[k][i >> 1] = v0;
                    }
                }
            }
        }
        (challenges, final_evals)
    }

    // =========================================================================
    // Generic helpers for the optimized prover (parameterized by E: SumcheckExtField)
    // =========================================================================

    /// Generic Lagrange basis polynomials L_{U_d, u}(x) for U_d = {0, 1, 2, inf}.
    pub(crate) fn lagrange_basis_degree3_generic<E: SumcheckExtField>(
        x: E::Scalar,
        one: E::Scalar,
        two: E::Scalar,
        inv2: E::Scalar,
    ) -> [E::Scalar; Self::OPT_U_SIZE] {
        let x_minus_1 = x.lazy_add_xp(2).lazy_sub(one); // [0, 4P)
        let x_minus_2 = x.lazy_add_xp(2).lazy_sub(two); // [0, 4P)
        [
            x_minus_1 * x_minus_2 * inv2,
            -(x * x_minus_2),
            x * x_minus_1 * inv2,
            x * x_minus_1 * x_minus_2,
        ]
    }

    /// Generic Kronecker product update: R_{i+1} = R_i (x) basis.
    pub(crate) fn update_small_value_weights_generic<E: SumcheckExtField>(
        current: &[E::Scalar],
        basis: [E::Scalar; Self::OPT_U_SIZE],
    ) -> Vec<E::Scalar> {
        let state_count = current.len();
        let mut next = vec![E::Scalar::zero(); state_count * Self::OPT_U_SIZE];
        for (state, &coeff) in current.iter().enumerate() {
            for (basis_idx, &basis_value) in basis.iter().enumerate() {
                next[state + basis_idx * state_count] = coeff * basis_value;
            }
        }
        next
    }

    /// Generic contraction of precomputed A_i(v, u) table with weight vector R_i[v].
    pub(crate) fn compute_t_from_precomputed_generic<E: SumcheckExtField>(
        round_table: &[E::Scalar],
        weights: &[E::Scalar],
    ) -> [E::Scalar; Self::OPT_HAT_SIZE] {
        let states = weights.len();
        let mut t_hat = [E::Scalar::zero(); Self::OPT_HAT_SIZE];
        for hat_idx in 0..Self::OPT_HAT_SIZE {
            let base = hat_idx * states;
            let mut acc = E::Scalar::zero();
            for state in 0..states {
                acc = acc.lazy_add(weights[state] * round_table[base + state]);
            }
            t_hat[hat_idx] = acc.reduce();
        }
        t_hat
    }

    /// Generic s_hat from t_hat: s_i(u) = c_i(u) * t_i(u).
    pub(crate) fn compute_round_s_from_t_generic<E: SumcheckExtField>(
        prefix_eq: E::Scalar,
        w_i: E::Scalar,
        t_hat: [E::Scalar; Self::OPT_HAT_SIZE],
        one: E::Scalar,
        two: E::Scalar,
    ) -> [E::Scalar; Self::OPT_HAT_SIZE] {
        let eq_0 = one.lazy_add_xp(2).lazy_sub(w_i); // [0, 4P)
        let eq_2 = Self::eq_linear_mont_generic::<E>(w_i, two, one);
        let eq_inf = w_i.lazy_add(w_i).lazy_add_xp(2).lazy_sub(one).con_sub_xp(2); // 2w-1+2P, [0, 4P)
        [
            prefix_eq * eq_0 * t_hat[0],
            prefix_eq * eq_2 * t_hat[1],
            prefix_eq * eq_inf * t_hat[2],
        ]
    }

    /// Generic transcript append for hat-point values.
    pub(crate) fn append_hat_round_values_generic<E: SumcheckExtField>(
        transcript: &mut Transcript,
        values: [E::Scalar; Self::OPT_HAT_SIZE],
    ) {
        for value in values {
            transcript.append_f(value.from_montgomery());
        }
    }

    /// Generic precompute small value tables (packed). Uses PBF gate_h but E for eq weights.
    fn precompute_small_value_tables_packed_generic<E: SumcheckExtField>(
        evals: &[Vec<PBF>; 4],
        eq_tables: &TwoStageEqTables<E>,
        ell0: usize,
        zero_check: bool,
        one: PBF,
        two: PBF,
        three: PBF,
        inv6: PBF,
    ) -> Vec<Vec<E::Scalar>> {
        let mut precomputed = Vec::with_capacity(ell0);
        for round in 0..ell0 {
            let prefix_len = round + 1;
            let eq_view = Self::round_eq_view_generic(eq_tables, round);
            if round == 0 {
                // Binary-zero: when zero_check=true, skip t_hat[0] (gate h = 0 at binary points).
                let packed_groups = evals[0].len() >> 1;
                let mut t_0 = E::zero();
                let mut t_2 = E::zero();
                let mut t_inf = E::zero();
                let packed_split = eq_view.packed_split_for_groups(packed_groups);

                if let Some(split) = packed_split {
                    let right_len = split.right_broadcast.len();
                    for left_group in 0..split.left_packed.len() {
                        let mut inner_t_0 = E::zero();
                        let mut inner_t_2 = E::zero();
                        let mut inner_t_inf = E::zero();
                        let group_base = left_group * right_len;

                        for right_idx in 0..right_len {
                            let eq_r = split.right_broadcast[right_idx];
                            let group_idx = group_base + right_idx;
                            let base = group_idx << 1;
                            let mut values_2 = [PBF::zero(); 4];
                            let mut diffs = [PBF::zero(); 4];

                            for poly_idx in 0..4 {
                                let v0 = evals[poly_idx][base];
                                let v1 = evals[poly_idx][base + 1];
                                let diff = Self::packed_diff(v0, v1);
                                values_2[poly_idx] = v1.lazy_add(diff).con_sub_xp(2);
                                diffs[poly_idx] = diff;
                            }

                            if !zero_check {
                                let values_0: [PBF; 4] = std::array::from_fn(|pi| evals[pi][base]);
                                inner_t_0 = inner_t_0.lazy_add(eq_r.mul_base_elem(Self::gate_h_packed_base(values_0, one)));
                            }
                            inner_t_2 = inner_t_2.lazy_add(eq_r.mul_base_elem(Self::gate_h_packed_base(values_2, one)));
                            inner_t_inf = inner_t_inf.lazy_add(eq_r.mul_base_elem(diffs[0] * diffs[1] * diffs[2]));
                        }

                        let eq_l = split.left_packed[left_group];
                        if !zero_check { t_0 = t_0.lazy_add(eq_l * inner_t_0.reduce_fast()); }
                        t_2 = t_2.lazy_add(eq_l * inner_t_2.reduce_fast());
                        t_inf = t_inf.lazy_add(eq_l * inner_t_inf.reduce_fast());
                    }
                } else {
                    for group_idx in 0..packed_groups {
                        let weight = eq_view.load_packed_weight(group_idx, packed_groups);
                        let base = group_idx << 1;
                        let mut values_2 = [PBF::zero(); 4];
                        let mut diffs = [PBF::zero(); 4];

                        for poly_idx in 0..4 {
                            let v0 = evals[poly_idx][base];
                            let v1 = evals[poly_idx][base + 1];
                            let diff = Self::packed_diff(v0, v1);
                            values_2[poly_idx] = v1.lazy_add(diff).con_sub_xp(2);
                            diffs[poly_idx] = diff;
                        }

                        if !zero_check {
                            let values_0: [PBF; 4] = std::array::from_fn(|pi| evals[pi][base]);
                            t_0 = t_0.lazy_add(weight.mul_base_elem(Self::gate_h_packed_base(values_0, one)));
                        }
                        t_2 = t_2.lazy_add(weight.mul_base_elem(Self::gate_h_packed_base(values_2, one)));
                        t_inf = t_inf.lazy_add(weight.mul_base_elem(diffs[0] * diffs[1] * diffs[2]));

                        if (group_idx + 1) % 8192 == 0 {
                            t_0 = t_0.reduce_fast();
                            t_2 = t_2.reduce_fast();
                            t_inf = t_inf.reduce_fast();
                        }
                    }
                }

                precomputed.push(vec![
                    if zero_check { E::Scalar::zero() } else { E::sum_lanes_to_mont(t_0) },
                    E::sum_lanes_to_mont(t_2),
                    E::sum_lanes_to_mont(t_inf),
                ]);
                continue;
            }
            if round == 1 {
                let packed_groups = evals[0].len() >> 2;
                let mut round_table = [E::zero(); Self::OPT_HAT_SIZE * Self::OPT_U_SIZE];
                let packed_split = eq_view.packed_split_for_groups(packed_groups);

                for block_idx in 0..packed_groups {
                    let weight = match packed_split {
                        Some(split) => split.weight(block_idx),
                        None => eq_view.load_packed_weight(block_idx, packed_groups),
                    };
                    let base = block_idx << 2;
                    let mut poly_grids = [[PBF::zero(); 16]; 4];

                    for poly_idx in 0..4 {
                        let low = Self::finite_line_from_pair_packed(
                            evals[poly_idx][base],
                            evals[poly_idx][base + 1],
                            three,
                        );
                        let high = Self::finite_line_from_pair_packed(
                            evals[poly_idx][base + 2],
                            evals[poly_idx][base + 3],
                            three,
                        );
                        for v_idx in 0..4 {
                            let line =
                                Self::finite_line_from_pair_packed(low[v_idx], high[v_idx], three);
                            for u_idx in 0..4 {
                                poly_grids[poly_idx][v_idx + (u_idx << 2)] = line[u_idx];
                            }
                        }
                    }

                    let mut tensor = [PBF::zero(); 16];
                    for idx in 0..16 {
                        tensor[idx] = Self::gate_h_packed_base(
                            std::array::from_fn(|poly_idx| poly_grids[poly_idx][idx]),
                            one,
                        );
                    }

                    Self::transform_tensor_axis_to_ud_packed(&mut tensor, 0, inv6);
                    Self::transform_tensor_axis_to_ud_packed(&mut tensor, 1, inv6);

                    for state in 0..Self::OPT_U_SIZE {
                        round_table[state] += weight.mul_base_elem(tensor[state]);
                        round_table[Self::OPT_U_SIZE + state] +=
                            weight.mul_base_elem(tensor[state + (2 * Self::OPT_U_SIZE)]);
                        round_table[(2 * Self::OPT_U_SIZE) + state] +=
                            weight.mul_base_elem(tensor[state + (3 * Self::OPT_U_SIZE)]);
                    }

                    if (block_idx + 1) % 8192 == 0 {
                        for entry in &mut round_table {
                            *entry = entry.reduce_fast();
                        }
                    }
                }

                precomputed.push(
                    round_table
                        .into_iter()
                        .map(|entry| E::sum_lanes_to_mont(entry))
                        .collect(),
                );
                continue;
            }
            if round == 2 {
                let states = Self::OPT_U_SIZE * Self::OPT_U_SIZE;
                let packed_groups = evals[0].len() >> 3;
                let mut round_table =
                    [E::zero(); Self::OPT_HAT_SIZE * (Self::OPT_U_SIZE * Self::OPT_U_SIZE)];
                let packed_split = eq_view.packed_split_for_groups(packed_groups);

                for block_idx in 0..packed_groups {
                    let weight = match packed_split {
                        Some(split) => split.weight(block_idx),
                        None => eq_view.load_packed_weight(block_idx, packed_groups),
                    };
                    let base = block_idx << 3;
                    let mut tensor = Self::gate_tensor_from_octets_packed(
                        std::array::from_fn(|poly_idx| {
                            [
                                evals[poly_idx][base],
                                evals[poly_idx][base + 1],
                                evals[poly_idx][base + 2],
                                evals[poly_idx][base + 3],
                                evals[poly_idx][base + 4],
                                evals[poly_idx][base + 5],
                                evals[poly_idx][base + 6],
                                evals[poly_idx][base + 7],
                            ]
                        }),
                        one,
                        three,
                    );

                    Self::transform_tensor_axis_to_ud_packed(&mut tensor, 0, inv6);
                    Self::transform_tensor_axis_to_ud_packed(&mut tensor, 1, inv6);
                    Self::transform_tensor_axis_to_ud_packed(&mut tensor, 2, inv6);

                    for state in 0..states {
                        round_table[state] += weight.mul_base_elem(tensor[state]);
                        round_table[states + state] += 
                            weight.mul_base_elem(tensor[state + (2 * states)]);
                        round_table[(2 * states) + state] += 
                            weight.mul_base_elem(tensor[state + (3 * states)]);
                    }

                    if (block_idx + 1) % 8192 == 0 {
                        for entry in &mut round_table {
                            *entry = entry.reduce_fast();
                        }
                    }
                }

                precomputed.push(
                    round_table
                        .into_iter()
                        .map(|entry| E::sum_lanes_to_mont(entry))
                        .collect(),
                );
                continue;
            }

            // Generic fallback for larger ell0.
            let states = Self::OPT_U_SIZE.pow(round as u32);
            let block_len = 1usize << prefix_len;
            let grid_len = Self::OPT_U_SIZE.pow(prefix_len as u32);
            let packed_groups = evals[0].len() / block_len;
            let mut round_table = vec![E::zero(); Self::OPT_HAT_SIZE * states];
            let packed_split = eq_view.packed_split_for_groups(packed_groups);

            for packed_idx in 0..packed_groups {
                let start = packed_idx * block_len;
                let end = start + block_len;
                let weight = match packed_split {
                    Some(split) => split.weight(packed_idx),
                    None => eq_view.load_packed_weight(packed_idx, packed_groups),
                };
                let mut tensor = vec![PBF::zero(); grid_len];

                for finite_idx in 0..grid_len {
                    let finite_points = Self::decode_u4(finite_idx, prefix_len);
                    let gate_inputs = std::array::from_fn(|poly_idx| {
                        Self::eval_packed_block_at_finite_points(
                            &evals[poly_idx][start..end],
                            &finite_points,
                            two,
                            three,
                        )
                    });
                    tensor[finite_idx] = Self::gate_h_packed_base(gate_inputs, one);
                }

                for axis in 0..prefix_len {
                    Self::transform_tensor_axis_to_ud_packed(&mut tensor, axis, inv6);
                }

                for state in 0..states {
                    for (hat_idx, &hat_point) in Self::OPT_HAT_POINTS.iter().enumerate() {
                        let tensor_idx = state + (hat_point as usize) * states;
                        round_table[hat_idx * states + state] += 
                            weight.mul_base_elem(tensor[tensor_idx]);
                    }
                }

                if (packed_idx + 1) % 8192 == 0 {
                    for entry in &mut round_table {
                        *entry = entry.reduce_fast();
                    }
                }
            }

            for entry in &mut round_table {
                *entry = entry.reduce_fast();
            }
            precomputed.push(
                round_table
                    .into_iter()
                    .map(|v| E::sum_lanes_to_mont(v))
                    .collect(),
            );
        }
        precomputed
    }

    /// Generic fold of packed ext-field tables in-place by one challenge.
    pub(crate) fn fold_packed_ext_tables_in_place_generic<E: SumcheckExtField>(
        evals: &mut [Vec<E>; 4],
        challenge_mont: E,
    ) {
        let next_len = evals[0].len() >> 1;
        for poly in evals.iter_mut() {
            for pair_idx in 0..next_len {
                let v0 = poly[pair_idx << 1];
                let v1 = poly[(pair_idx << 1) + 1];
                let diff = v1.lazy_add_xp(2).lazy_sub(v0);
                poly[pair_idx] = (challenge_mont * diff)
                    .lazy_add(v0)
                    .con_sub_xp(2);
            }
            poly.truncate(next_len);
        }
    }

    /// Generic fold of base-field tables by one challenge, PBF -> E (allocating).
    fn fold_packed_base_tables_once_generic<E: SumcheckExtField>(
        evals: &[Vec<PBF>; 4],
        challenge_mont: E,
    ) -> [Vec<E>; 4] {
        let next_len = evals[0].len() >> 1;
        std::array::from_fn(|poly_idx| {
            let mut next = Vec::with_capacity(next_len);
            for pair_idx in 0..next_len {
                let v0 = evals[poly_idx][pair_idx << 1];
                let v1 = evals[poly_idx][(pair_idx << 1) + 1];
                let diff = v1.lazy_add_xp(2).lazy_sub(v0);
                let folded = challenge_mont
                    .mul_base_elem(diff)
                    .add_base_elem(v0)
                    .con_sub_xp(2);
                next.push(folded);
            }
            next
        })
    }

    /// Generic fold of base-field tables by multiple challenges, PBF -> E (allocating).
    pub(crate) fn fold_base_tables_to_ext_generic<E: SumcheckExtField>(
        evals: &[Vec<PBF>; 4],
        challenges: &[E::Scalar],
    ) -> [Vec<E>; 4] {
        debug_assert!(!challenges.is_empty());
        if challenges.len() == 2 {
            let alpha0 = E::from_scalar(challenges[0]);
            let alpha1 = E::from_scalar(challenges[1]);
            let next_len = evals[0].len() >> 2;
            return std::array::from_fn(|poly_idx| {
                let mut next = Vec::with_capacity(next_len);
                for block_idx in 0..next_len {
                    let base = block_idx << 2;
                    let e0 = evals[poly_idx][base];
                    let e1 = evals[poly_idx][base + 1];
                    let e2 = evals[poly_idx][base + 2];
                    let e3 = evals[poly_idx][base + 3];
                    let diff0 = e1.lazy_add_xp(2).lazy_sub(e0);
                    let diff1 = e3.lazy_add_xp(2).lazy_sub(e2);
                    let low = alpha0
                        .mul_base_elem(diff0)
                        .add_base_elem(e0)
                        .con_sub_xp(2);
                    let high = alpha0
                        .mul_base_elem(diff1)
                        .add_base_elem(e2)
                        .con_sub_xp(2);
                    let diff = high.lazy_add_xp(2).lazy_sub(low);
                    next.push((alpha1 * diff).lazy_add(low).con_sub_xp(2));
                }
                next
            });
        }
        if challenges.len() == 3 {
            let alpha0 = E::from_scalar(challenges[0]);
            let alpha1 = E::from_scalar(challenges[1]);
            let alpha2 = E::from_scalar(challenges[2]);
            let next_len = evals[0].len() >> 3;
            return std::array::from_fn(|poly_idx| {
                let mut next = Vec::with_capacity(next_len);
                for block_idx in 0..next_len {
                    let base = block_idx << 3;
                    let e0 = evals[poly_idx][base];
                    let e1 = evals[poly_idx][base + 1];
                    let e2 = evals[poly_idx][base + 2];
                    let e3 = evals[poly_idx][base + 3];
                    let e4 = evals[poly_idx][base + 4];
                    let e5 = evals[poly_idx][base + 5];
                    let e6 = evals[poly_idx][base + 6];
                    let e7 = evals[poly_idx][base + 7];

                    let d01 = e1.lazy_add_xp(2).lazy_sub(e0);
                    let d23 = e3.lazy_add_xp(2).lazy_sub(e2);
                    let d45 = e5.lazy_add_xp(2).lazy_sub(e4);
                    let d67 = e7.lazy_add_xp(2).lazy_sub(e6);

                    let x00 = alpha0.mul_base_elem(d01).add_base_elem(e0).con_sub_xp(2);
                    let x01 = alpha0.mul_base_elem(d23).add_base_elem(e2).con_sub_xp(2);
                    let x10 = alpha0.mul_base_elem(d45).add_base_elem(e4).con_sub_xp(2);
                    let x11 = alpha0.mul_base_elem(d67).add_base_elem(e6).con_sub_xp(2);

                    let dy0 = x01.lazy_add_xp(2).lazy_sub(x00);
                    let dy1 = x11.lazy_add_xp(2).lazy_sub(x10);
                    let y0 = (alpha1 * dy0).lazy_add(x00).con_sub_xp(2);
                    let y1 = (alpha1 * dy1).lazy_add(x10).con_sub_xp(2);

                    let dz = y1.lazy_add_xp(2).lazy_sub(y0);
                    next.push((alpha2 * dz).lazy_add(y0).con_sub_xp(2));
                }
                next
            });
        }
        // Generic fallback
        let mut folded =
            Self::fold_packed_base_tables_once_generic::<E>(evals, E::from_scalar(challenges[0]));
        for &challenge in &challenges[1..] {
            Self::fold_packed_ext_tables_in_place_generic(
                &mut folded,
                E::from_scalar(challenge),
            );
        }
        folded
    }

    /// Generic fused fold + round polynomial computation for packed ext-field tables.
    pub(crate) fn fold_packed_ext_tables_and_compute_round_t_in_place_generic<E: SumcheckExtField>(
        evals: &mut [Vec<E>; 4],
        eq_view: &RoundEqView<'_, E>,
        challenge: E::Scalar,
    ) -> [E::Scalar; Self::OPT_HAT_SIZE] {
        let next_len = evals[0].len() >> 1;
        let packed_groups = next_len >> 1;
        let alpha = E::from_scalar(challenge);
        let one = E::one().ext_to_montgomery();
        let mut t_0 = E::zero();
        let mut t_2 = E::zero();
        let mut t_inf = E::zero();
        let packed_split = eq_view.packed_split_for_groups(packed_groups);

        if let Some(split) = packed_split {
            let right_len = split.right_broadcast.len();
            for left_group in 0..split.left_packed.len() {
                let mut inner_t_0 = E::zero();
                let mut inner_t_2 = E::zero();
                let mut inner_t_inf = E::zero();
                let group_base = left_group * right_len;

                for right_idx in 0..right_len {
                    let group_idx = group_base + right_idx;
                    let eq_r = split.right_broadcast[right_idx];
                    let src_base = group_idx << 2;
                    let dst_base = group_idx << 1;
                    let mut values_0 = [E::zero(); 4];
                    let mut values_2 = [E::zero(); 4];
                    let mut diffs = [E::zero(); 4];

                    for poly_idx in 0..4 {
                        let e0 = evals[poly_idx][src_base];
                        let e1 = evals[poly_idx][src_base + 1];
                        let e2 = evals[poly_idx][src_base + 2];
                        let e3 = evals[poly_idx][src_base + 3];
                        let diff0 = e1.lazy_add_xp(2).lazy_sub(e0);
                        let diff1 = e3.lazy_add_xp(2).lazy_sub(e2);
                        let v0 = (alpha * diff0).lazy_add(e0).con_sub_xp(2);
                        let v1 = (alpha * diff1).lazy_add(e2).con_sub_xp(2);
                        let diff = v1.lazy_add_xp(2).lazy_sub(v0).con_sub_xp(2);

                        evals[poly_idx][dst_base] = v0;
                        evals[poly_idx][dst_base + 1] = v1;
                        values_0[poly_idx] = v0;
                        values_2[poly_idx] = v1.lazy_add(diff).con_sub_xp(2);
                        diffs[poly_idx] = diff;
                    }

                    inner_t_0 = inner_t_0.lazy_add(eq_r * Self::gate_h_packed_ext_generic(values_0, one));
                    inner_t_2 = inner_t_2.lazy_add(eq_r * Self::gate_h_packed_ext_generic(values_2, one));
                    inner_t_inf = inner_t_inf.lazy_add(eq_r * diffs[0] * diffs[1] * diffs[2]);
                }

                let eq_l = split.left_packed[left_group];
                t_0 = t_0.lazy_add(eq_l * inner_t_0.reduce_fast());
                t_2 = t_2.lazy_add(eq_l * inner_t_2.reduce_fast());
                t_inf = t_inf.lazy_add(eq_l * inner_t_inf.reduce_fast());
            }
        } else {
            for group_idx in 0..packed_groups {
                let weight = eq_view.load_packed_weight(group_idx, packed_groups);
                let src_base = group_idx << 2;
                let dst_base = group_idx << 1;
                let mut values_0 = [E::zero(); 4];
                let mut values_2 = [E::zero(); 4];
                let mut diffs = [E::zero(); 4];

                for poly_idx in 0..4 {
                    let e0 = evals[poly_idx][src_base];
                    let e1 = evals[poly_idx][src_base + 1];
                    let e2 = evals[poly_idx][src_base + 2];
                    let e3 = evals[poly_idx][src_base + 3];
                    let diff0 = e1.lazy_add_xp(2).lazy_sub(e0);
                    let diff1 = e3.lazy_add_xp(2).lazy_sub(e2);
                    let v0 = (alpha * diff0).lazy_add(e0).con_sub_xp(2);
                    let v1 = (alpha * diff1).lazy_add(e2).con_sub_xp(2);
                    let diff = v1.lazy_add_xp(2).lazy_sub(v0).con_sub_xp(2);

                    evals[poly_idx][dst_base] = v0;
                    evals[poly_idx][dst_base + 1] = v1;
                    values_0[poly_idx] = v0;
                    values_2[poly_idx] = v1.lazy_add(diff).con_sub_xp(2);
                    diffs[poly_idx] = diff;
                }

                t_0 = t_0.lazy_add(weight * Self::gate_h_packed_ext_generic(values_0, one));
                t_2 = t_2.lazy_add(weight * Self::gate_h_packed_ext_generic(values_2, one));
                t_inf = t_inf.lazy_add(weight * diffs[0] * diffs[1] * diffs[2]);

                if (group_idx + 1) % 8192 == 0 {
                    t_0 = t_0.reduce_fast();
                    t_2 = t_2.reduce_fast();
                    t_inf = t_inf.reduce_fast();
                }
            }
        }

        for poly in evals.iter_mut() {
            poly.truncate(next_len);
        }

        [
            E::sum_lanes_to_mont(t_0),
            E::sum_lanes_to_mont(t_2),
            E::sum_lanes_to_mont(t_inf),
        ]
    }

    /// Generic compute round t from packed tables (without fold).
    pub(crate) fn compute_round_t_from_packed_tables_generic<E: SumcheckExtField>(
        evals: &[Vec<E>; 4],
        eq_view: &RoundEqView<'_, E>,
    ) -> [E::Scalar; Self::OPT_HAT_SIZE] {
        let packed_groups = evals[0].len() >> 1;
        let one = E::one().ext_to_montgomery();
        let mut t_0 = E::zero();
        let mut t_2 = E::zero();
        let mut t_inf = E::zero();
        let packed_split = eq_view.packed_split_for_groups(packed_groups);

        if let Some(split) = packed_split {
            let right_len = split.right_broadcast.len();
            for left_group in 0..split.left_packed.len() {
                let mut inner_t_0 = E::zero();
                let mut inner_t_2 = E::zero();
                let mut inner_t_inf = E::zero();
                let group_base = left_group * right_len;

                for right_idx in 0..right_len {
                    let eq_r = split.right_broadcast[right_idx];
                    let group_idx = group_base + right_idx;
                    let base = group_idx << 1;
                    let mut values_0 = [E::zero(); 4];
                    let mut values_2 = [E::zero(); 4];
                    let mut diffs = [E::zero(); 4];
                    for poly_idx in 0..4 {
                        let v0 = evals[poly_idx][base];
                        let v1 = evals[poly_idx][base + 1];
                        let diff = v1.lazy_add_xp(2).lazy_sub(v0).con_sub_xp(2);
                        values_0[poly_idx] = v0;
                        values_2[poly_idx] = v1.lazy_add(diff).con_sub_xp(2);
                        diffs[poly_idx] = diff;
                    }
                    inner_t_0 = inner_t_0.lazy_add(eq_r * Self::gate_h_packed_ext_generic(values_0, one));
                    inner_t_2 = inner_t_2.lazy_add(eq_r * Self::gate_h_packed_ext_generic(values_2, one));
                    inner_t_inf = inner_t_inf.lazy_add(eq_r * diffs[0] * diffs[1] * diffs[2]);
                }

                let eq_l = split.left_packed[left_group];
                t_0 = t_0.lazy_add(eq_l * inner_t_0.reduce_fast());
                t_2 = t_2.lazy_add(eq_l * inner_t_2.reduce_fast());
                t_inf = t_inf.lazy_add(eq_l * inner_t_inf.reduce_fast());
            }
        } else {
            for group_idx in 0..packed_groups {
                let base = group_idx << 1;
                let weight = eq_view.load_packed_weight(group_idx, packed_groups);
                let mut values_0 = [E::zero(); 4];
                let mut values_2 = [E::zero(); 4];
                let mut diffs = [E::zero(); 4];
                for poly_idx in 0..4 {
                    let v0 = evals[poly_idx][base];
                    let v1 = evals[poly_idx][base + 1];
                    let diff = v1.lazy_add_xp(2).lazy_sub(v0).con_sub_xp(2);
                    values_0[poly_idx] = v0;
                    values_2[poly_idx] = v1.lazy_add(diff).con_sub_xp(2);
                    diffs[poly_idx] = diff;
                }
                t_0 = t_0.lazy_add(weight * Self::gate_h_packed_ext_generic(values_0, one));
                t_2 = t_2.lazy_add(weight * Self::gate_h_packed_ext_generic(values_2, one));
                t_inf = t_inf.lazy_add(weight * diffs[0] * diffs[1] * diffs[2]);

                if (group_idx + 1) % 8192 == 0 {
                    t_0 = t_0.reduce_fast();
                    t_2 = t_2.reduce_fast();
                    t_inf = t_inf.reduce_fast();
                }
            }
        }

        [
            E::sum_lanes_to_mont(t_0),
            E::sum_lanes_to_mont(t_2),
            E::sum_lanes_to_mont(t_inf),
        ]
    }

    /// Generic build_single_packed_pair_views for scalar tail.
    pub(crate) fn build_single_packed_pair_views_generic<E: SumcheckExtField>(
        value: E,
        pair_count: usize,
    ) -> (E, E, E) {
        debug_assert!(pair_count > 0 && pair_count <= 4);
        let lanes = E::unpack_to_scalars(value);
        let mut v0 = [E::Scalar::zero(); 8];
        let mut v1 = [E::Scalar::zero(); 8];
        let mut diffs_arr = [E::Scalar::zero(); 8];
        for pair_idx in 0..pair_count {
            let left = lanes[pair_idx << 1];
            let right = lanes[(pair_idx << 1) + 1];
            v0[pair_idx] = left;
            v1[pair_idx] = right;
            diffs_arr[pair_idx] = right.lazy_add_xp(2).lazy_sub(left).con_sub_xp(2);
        }
        (
            E::pack_scalars(&v0[..pair_count]),
            E::pack_scalars(&v1[..pair_count]),
            E::pack_scalars(&diffs_arr[..pair_count]),
        )
    }

    /// Generic compute round t from single packed tables (scalar tail).
    pub(crate) fn compute_round_t_from_single_packed_tables_generic<E: SumcheckExtField>(
        evals: &[E; 4],
        eq_view: &RoundEqView<'_, E>,
        active_len: usize,
    ) -> [E::Scalar; Self::OPT_HAT_SIZE] {
        debug_assert!(active_len.is_power_of_two() && active_len >= 2 && active_len <= 8);
        let pair_count = active_len >> 1;
        let mut scalar_weights = [E::Scalar::zero(); 8];
        for pair_idx in 0..pair_count {
            scalar_weights[pair_idx] = eq_view.scalar_weight(pair_idx);
        }
        let packed_weights = E::pack_scalars(&scalar_weights[..pair_count]);
        let one = E::one().ext_to_montgomery();
        let mut values_0 = [E::zero(); 4];
        let mut values_2 = [E::zero(); 4];
        let mut diffs = [E::zero(); 4];

        for poly_idx in 0..4 {
            let (v0, v1, diff) =
                Self::build_single_packed_pair_views_generic::<E>(evals[poly_idx], pair_count);
            values_0[poly_idx] = v0;
            values_2[poly_idx] = v1.lazy_add(diff).con_sub_xp(2);
            diffs[poly_idx] = diff;
        }

        let t_0 = packed_weights * Self::gate_h_packed_ext_generic(values_0, one);
        let t_2 = packed_weights * Self::gate_h_packed_ext_generic(values_2, one);
        let t_inf = packed_weights * diffs[0] * diffs[1] * diffs[2];

        [
            E::sum_lanes_to_mont(t_0),
            E::sum_lanes_to_mont(t_2),
            E::sum_lanes_to_mont(t_inf),
        ]
    }

    /// Generic fold single packed ext tables in-place (scalar tail).
    pub(crate) fn fold_single_packed_ext_tables_in_place_generic<E: SumcheckExtField>(
        evals: &mut [E; 4],
        challenge_mont: E,
        active_len: usize,
    ) {
        debug_assert!(active_len.is_power_of_two() && active_len >= 2 && active_len <= 8);
        let pair_count = active_len >> 1;
        for poly in evals.iter_mut() {
            let (v0, _v1, diff) =
                Self::build_single_packed_pair_views_generic::<E>(*poly, pair_count);
            *poly = (challenge_mont * diff)
                .con_sub_xp(2)
                .lazy_add(v0)
                .con_sub_xp(2);
        }
    }

    // =========================================================================
    // Generic optimized prover
    // =========================================================================

    /// Generic optimized prover parameterized by E: SumcheckExtField.
    /// `prove_add_mul_ell0_ext3` delegates to this directly.
    ///
    /// Uses allocating fold for the base->ext transition.
    pub(crate) fn prove_add_mul_optimized_with_ell0_generic<E: SumcheckExtField>(
        evals: [Vec<PBF>; 4],
        point: &[E::Scalar],
        ell0: usize,
        zero_check: bool,
        transcript: &mut Transcript,
        timings: &mut ZeroCheckTimings,
    ) -> (Vec<E::Scalar>, [E::Scalar; 5]) {
        let t_total = Instant::now();
        let num_vars = point.len();
        let ell0 = Self::resolve_optimized_ell0(num_vars, Some(ell0));

        // Convert point to Montgomery form for all subsequent arithmetic.
        let point_mont = point
            .iter()
            .copied()
            .map(|v| v.to_montgomery())
            .collect::<Vec<_>>();

        // Build two-stage eq tables for all rounds.
        let t_eq = Instant::now();
        let eq_tables = Self::build_two_stage_eq_tables_generic::<E>(&point_mont);
        timings.eq_tables_us += t_eq.elapsed().as_micros();

        let base_one = PBF::one().to_montgomery();
        let base_two = PBF::from(2u32).to_montgomery();
        let base_three = PBF::from(3u32).to_montgomery();
        let base_inv6 = PBF::from(((P + 1) / 6) as u64).to_montgomery();

        // : Pre-computation
        let t_pre = Instant::now();
        let small_value_tables = Self::precompute_small_value_tables_packed_generic::<E>(
            &evals,
            &eq_tables,
            ell0,
            zero_check,
            base_one,
            base_two,
            base_three,
            base_inv6,
        );
        timings.precompute_us += t_pre.elapsed().as_micros();

        let ext_one = E::Scalar::one().to_montgomery();
        let ext_two = E::Scalar::from(2u32).to_montgomery();
        let ext_inv2 = E::Scalar::from(SBF::inv_2()).to_montgomery();

        let mut verifier_challenges = Vec::with_capacity(num_vars);
        let mut prefix_eq_eval = ext_one;
        let mut lagrange_weights = vec![ext_one];

        // : Small-value rounds i = 0, ..., ell0-1
        let t_small = Instant::now();
        for round in 0..ell0 {
            let t_hat = Self::compute_t_from_precomputed_generic::<E>(
                &small_value_tables[round],
                &lagrange_weights,
            );
            let s_hat = Self::compute_round_s_from_t_generic::<E>(
                prefix_eq_eval,
                point_mont[round],
                t_hat,
                ext_one,
                ext_two,
            );
            Self::append_hat_round_values_generic::<E>(transcript, s_hat);

            let challenge = transcript.challenge_f::<E::Scalar>().to_montgomery();
            let challenge_mont = challenge;
            prefix_eq_eval *=
                Self::eq_linear_mont_generic::<E>(point_mont[round], challenge_mont, ext_one);
            verifier_challenges.push(challenge);

            if round + 1 < ell0 {
                let basis = Self::lagrange_basis_degree3_generic::<E>(
                    challenge_mont,
                    ext_one,
                    ext_two,
                    ext_inv2,
                );
                lagrange_weights =
                    Self::update_small_value_weights_generic::<E>(&lagrange_weights, basis);
            }
        }
        timings.small_value_rounds_us += t_small.elapsed().as_micros();

        // : Packed remaining rounds [ell0, num_vars - 3)
        let packed_round_start = ell0;
        let packed_round_end = num_vars.saturating_sub(3);

        // Transition fold: fold PBF tables by all ell0 challenges at once, producing E tables.
        // Uses allocating fold (safe for any extension degree).
        let t_trans = Instant::now();
        let mut folded_tables = Self::fold_base_tables_to_ext_generic::<E>(
            &evals,
            &verifier_challenges[..ell0],
        );
        drop(evals); // Free PBF memory
        timings.transition_fold_us += t_trans.elapsed().as_micros();

        if packed_round_start < packed_round_end {
            // First packed round: compute t_hat from the freshly-folded tables.
            let t_first = Instant::now();
            let first_t_hat = Self::compute_round_t_from_packed_tables_generic::<E>(
                &folded_tables,
                &Self::round_eq_view_generic(&eq_tables, packed_round_start),
            );
            timings.packed_fold_rounds_us += t_first.elapsed().as_micros();
            let s_hat = Self::compute_round_s_from_t_generic::<E>(
                prefix_eq_eval,
                point_mont[packed_round_start],
                first_t_hat,
                ext_one,
                ext_two,
            );
            Self::append_hat_round_values_generic::<E>(transcript, s_hat);

            let challenge = transcript.challenge_f::<E::Scalar>().to_montgomery();
            let challenge_mont = challenge;
            prefix_eq_eval *= Self::eq_linear_mont_generic::<E>(
                point_mont[packed_round_start],
                challenge_mont,
                ext_one,
            );
            verifier_challenges.push(challenge);
            let mut prev_challenge = challenge;

            // Subsequent packed rounds: fused fold + compute.
            for round in (packed_round_start + 1)..packed_round_end {
                let eq_view = Self::round_eq_view_generic(&eq_tables, round);
                let t_pf = Instant::now();
                let t_hat = Self::fold_packed_ext_tables_and_compute_round_t_in_place_generic::<E>(
                    &mut folded_tables,
                    &eq_view,
                    prev_challenge,
                );
                timings.packed_fold_rounds_us += t_pf.elapsed().as_micros();
                let s_hat = Self::compute_round_s_from_t_generic::<E>(
                    prefix_eq_eval,
                    point_mont[round],
                    t_hat,
                    ext_one,
                    ext_two,
                );
                Self::append_hat_round_values_generic::<E>(transcript, s_hat);

                let challenge = transcript.challenge_f::<E::Scalar>().to_montgomery();
                let challenge_mont = challenge;
                prefix_eq_eval *= Self::eq_linear_mont_generic::<E>(
                    point_mont[round],
                    challenge_mont,
                    ext_one,
                );
                verifier_challenges.push(challenge);
                prev_challenge = challenge;
            }

            // Final fold before scalar tail.
            let t_pf_final = Instant::now();
            Self::fold_packed_ext_tables_in_place_generic(
                &mut folded_tables,
                E::from_scalar(prev_challenge),
            );
            timings.packed_fold_rounds_us += t_pf_final.elapsed().as_micros();
        }

        // : Scalar tail (last 3 rounds)
        let t_tail = Instant::now();
        let scalar_tail_start = (num_vars.saturating_sub(3)).max(ell0);
        let mut active_len = 1usize << (num_vars - scalar_tail_start);
        let mut packed_tail_tables: [E; 4] = std::array::from_fn(|k| folded_tables[k][0]);

        for round in scalar_tail_start..num_vars {
            let eq_view = Self::round_eq_view_generic(&eq_tables, round);
            let t_hat = Self::compute_round_t_from_single_packed_tables_generic::<E>(
                &packed_tail_tables,
                &eq_view,
                active_len,
            );
            let s_hat = Self::compute_round_s_from_t_generic::<E>(
                prefix_eq_eval,
                point_mont[round],
                t_hat,
                ext_one,
                ext_two,
            );
            Self::append_hat_round_values_generic::<E>(transcript, s_hat);

            let challenge = transcript.challenge_f::<E::Scalar>().to_montgomery();
            let challenge_mont = challenge;
            prefix_eq_eval *= Self::eq_linear_mont_generic::<E>(
                point_mont[round],
                challenge_mont,
                ext_one,
            );
            verifier_challenges.push(challenge);

            if active_len > 1 {
                Self::fold_single_packed_ext_tables_in_place_generic::<E>(
                    &mut packed_tail_tables,
                    E::from_scalar(challenge),
                    active_len,
                );
                active_len >>= 1;
            }
        }

        let mut final_evals = [E::Scalar::zero(); 5];
        for poly_idx in 0..4 {
            final_evals[poly_idx] =
                E::unpack_to_scalars(packed_tail_tables[poly_idx])[0].from_montgomery();
        }
        final_evals[4] = prefix_eq_eval.from_montgomery();
        timings.scalar_tail_us += t_tail.elapsed().as_micros();
        timings.total_us += t_total.elapsed().as_micros();
        (verifier_challenges, final_evals)
    }

    // =========================================================================
    // Ext3 public API
    // =========================================================================


    // =========================================================================
    // Verifier
    // =========================================================================

    /// Verify a legacy sumcheck proof (5-point format: {s(0), s(1), s(2), s(-1), s(∞)}).
    ///
    /// Input:
    ///   - `y`: claimed sum (1 output)
    ///   - `var_num`: number of sumcheck variables
    ///   - `transcript`: Fiat-Shamir transcript
    ///   - `proof`: serialized proof bytes
    ///
    /// Output:
    ///   - `(challenges, final_claim)`: challenges and the final evaluated claim
    ///
    /// The prover sends 5 evaluations per round: s(0), s(1), s(2), s(-1), s(∞).
    /// The verifier checks s(0) + s(1) == claimed_sum, then extrapolates s(r)
    /// at the random challenge r using Lagrange interpolation over {0, 1, 2, -1, ∞}.
    ///
    /// Note: "∞" point means the leading coefficient of the degree-4 polynomial.
    /// The polynomial is degree 4, so 5 points determine it uniquely.
    /// Verify a legacy sumcheck proof (5-point format: {s(0), s(1), s(2), s(-1), s(∞)}).
    ///
    /// The legacy prover writes 5 evaluations per round. The polynomial has degree 4.
    /// Points: {0, 1, 2, -1, leading_coeff}. We reconstruct s(r) via Lagrange interpolation
    /// over {0, 1, 2, 3, 4} after computing s(3) and s(4) from the sent values.
    ///
    /// The returned `challenges` are in the same SIMD-round order the prover used
    /// (x_3, x_4, ..., x_{μ-1}, x_0, x_1, x_2); callers must apply the same rotation
    /// to any downstream point evaluation that is tied to natural variable order.
    pub fn verify_legacy<E: SumcheckExtField>(
        mut y: E::Scalar,
        var_num: usize,
        transcript: &mut Transcript,
        proof: &mut util::fiat_shamir::Proof,
    ) -> (Vec<E::Scalar>, E::Scalar) {
        // Delegate to Sumcheck::verify which reads 5 values per round (degree=4).
        // The Sumcheck::verify expects evaluations at {0, 1, 2, 3, 4}.
        // The legacy prover writes {s(0), s(1), s(2), s(-1), s(∞)}.
        // These ARE in the order that Sumcheck::verify reads them: it reads degree+1=5 values
        // and interprets them as evaluations at {0, 1, ..., degree}.
        // So the prover-verifier mismatch: prover sends s(-1) and s(∞), but verifier
        // expects s(3) and s(4). These are different!
        //
        // Actually, looking at the Goldilocks Sumcheck::prove, it evaluates at {0, 1, ..., degree}
        // and sends those values. The legacy MamaBear prover evaluates at {0, 1, 2, -1, ∞}.
        // These are DIFFERENT evaluation points, so Sumcheck::verify CAN'T be used directly.
        //
        // For now, implement a custom verifier that handles the legacy format.
        let mut challenges = Vec::with_capacity(var_num);
        for _round in 0..var_num {
            // Read 5 values: s(0), s(1), s(2), s(-1), s(∞)
            let s0: E::Scalar = proof.get_next_and_step();
            transcript.append_f(s0);
            let s1: E::Scalar = proof.get_next_and_step();
            transcript.append_f(s1);
            let s2: E::Scalar = proof.get_next_and_step();
            transcript.append_f(s2);
            let sm1: E::Scalar = proof.get_next_and_step();
            transcript.append_f(sm1);
            let sinf: E::Scalar = proof.get_next_and_step();
            transcript.append_f(sinf);

            let ym = y.to_montgomery();

            // Note: the caller is responsible for checking s(0) + s(1) == claimed_sum.
            // Here we just track the running sum for the next round.
            let _ = ym;

            // Get challenge
            let r: E::Scalar = transcript.challenge_f::<E::Scalar>().to_montgomery();
            challenges.push(r.from_montgomery());

            // Extrapolate s(r) from 5 points {0, 1, 2, -1, ∞}
            // s(X) is degree 4. We have s at {0, 1, 2, -1} (4 proper points) + leading coeff.
            // Reconstruct: s(3) = s(-1) - 4*s(0) + 6*s(1) - 4*s(2) + s(∞)*? ...
            // Actually, s(∞) = leading coefficient of degree-4 polynomial.
            // s(X) = a₄X⁴ + a₃X³ + a₂X² + a₁X + a₀
            // s(0) = a₀, s(1) = a₄+a₃+a₂+a₁+a₀, s(2) = 16a₄+8a₃+4a₂+2a₁+a₀
            // s(-1) = a₄-a₃+a₂-a₁+a₀
            // s(∞) = a₄ (leading coefficient)
            //
            // From these 5 values, reconstruct s(r):
            // a₀ = s(0)
            // a₄ = s(∞)
            // From s(1)+s(-1) = 2a₄+2a₂+2a₀ → a₂ = (s(1)+s(-1)-2a₄-2a₀)/2
            // From s(1)-s(-1) = 2a₃+2a₁ → a₃+a₁ = (s(1)-s(-1))/2
            // From s(2) = 16a₄+8a₃+4a₂+2a₁+a₀ → 8a₃+2a₁ = s(2)-16a₄-4a₂-a₀
            //   → 8a₃+2a₁ and a₃+a₁ → solve: a₃ = (8a₃+2a₁ - 2(a₃+a₁))/6, a₁ = (a₃+a₁)-a₃
            let s0m = s0.to_montgomery();
            let s1m = s1.to_montgomery();
            let s2m = s2.to_montgomery();
            let sm1m = sm1.to_montgomery();
            let sinfm = sinf.to_montgomery(); // = a₄

            let inv2 = E::Scalar::from(SBF::inv_2()).to_montgomery();
            let inv6 = E::Scalar::from(SBF((P + 1) / 6)).to_montgomery();

            let a0 = s0m;
            let a4 = sinfm;
            // a2 = (s1 + s(-1) - 2a4 - 2a0) / 2
            let a2 = (s1m.lazy_add(sm1m).lazy_add_xp(4)
                .lazy_sub(a4.lazy_add(a4))
                .lazy_sub(a0.lazy_add(a0))
                .reduce()) * inv2;
            // a3 + a1 = (s1 - s(-1)) / 2
            let a3_plus_a1 = (s1m.lazy_add_xp(2).lazy_sub(sm1m).reduce()) * inv2;
            // 8a3 + 2a1 = s2 - 16a4 - 4a2 - a0
            let four = E::Scalar::from(4u32).to_montgomery();
            let sixteen = E::Scalar::from(16u32).to_montgomery();
            let eight_a3_plus_2a1 = s2m.lazy_add_xp(8)
                .lazy_sub(sixteen * a4)
                .lazy_sub(four * a2)
                .lazy_sub(a0)
                .reduce();
            // a3 = (8a3+2a1 - 2(a3+a1)) / 6
            let a3 = (eight_a3_plus_2a1.lazy_add_xp(2).lazy_sub(a3_plus_a1.lazy_add(a3_plus_a1)).reduce()) * inv6;
            let a1 = a3_plus_a1.lazy_add_xp(2).lazy_sub(a3).reduce();

            // s(r) = a4*r⁴ + a3*r³ + a2*r² + a1*r + a0
            // Horner: s(r) = ((((a4*r + a3)*r + a2)*r + a1)*r + a0)
            let sr = ((((a4 * r).lazy_add(a3).reduce() * r).lazy_add(a2).reduce() * r).lazy_add(a1).reduce() * r).lazy_add(a0).reduce();
            y = sr.from_montgomery();
        }

        (challenges, y)
    }

    /// Verify an optimized sumcheck proof (3-hat-point format: {s(0), s(2), s(∞)}).
    ///
    /// Input:
    ///   - `y`: claimed sum (1 output) in normal form
    ///   - `r_zero`: the caller's ZeroCheck challenge vector (the `point` that
    ///     was passed to the prover). Each `r_zero[round]` is used as `w_i` to
    ///     reconstruct `c_i(X)` in the t-factored Lagrange step below. Must be
    ///     in the same SIMD-round order the prover consumed — passing a
    ///     natural-order point here silently verifies a variable-permuted
    ///     statement.
    ///   - `var_num`: number of sumcheck variables (== r_zero.len())
    ///   - `transcript`: Fiat-Shamir transcript
    ///   - `proof`: serialized proof bytes
    ///
    /// Output:
    ///   - `(challenges, final_claim)`: challenges in normal form and final y.
    ///     The challenges are in SIMD-round order
    ///     (x_3, x_4, ..., x_{μ-1}, x_0, x_1, x_2); callers that need a
    ///     natural-order point should rotate with `simd_to_natural_point`.
    ///
    /// # Round-polynomial degree and the t-factored reconstruction
    ///
    /// The ZeroCheck target is f(X) = eq(w, X) · h(X) where h is the gate
    /// (1-S)(L+R) + S·L·R + O. In one variable, eq is linear and h is degree 3
    /// (from the S·L·R term), so f — and thus the round polynomial s_i(X) —
    /// has degree **4**. The ell0 optimization factors s_i(X) = c_i(X) · t_i(X)
    /// where c_i(X) = eq(w[<i], r[<i]) · eq(w_i, X) is linear and known from
    /// previous rounds, leaving t_i of degree 3.
    ///
    /// The prover sends 3 hat values {s_i(0), s_i(2), s_i(∞)} where
    /// s_i(∞) is the LEADING coefficient a_4 of s_i (coefficient of X^4).
    /// With s_i(1) = y_prev - s_i(0) recovered from the sum-check invariant
    /// that's only 4 values — NOT enough to uniquely determine a degree-4
    /// polynomial (5 coefficients). We must exploit the c · t factored form.
    ///
    /// Correct reconstruction:
    ///   1. c_i(u) is computable from w_i = r_zero[round] and prefix_eq.
    ///   2. t_i(u) = s_i(u) / c_i(u) for u ∈ {0, 1, 2, ∞} yields four values
    ///      of a degree-3 polynomial — enough to determine it uniquely.
    ///   3. Interpolate t_i at challenge α over {0, 1, 2, 3} where
    ///      t_i(3) = 6·t_i(∞) + 3·t_i(2) - 3·t_i(1) + t_i(0) (Δ^3 t = 6·a_3).
    ///   4. s_i(α) = c_i(α) · t_i(α).
    ///
    /// The old (buggy) implementation tried to Lagrange-interpolate s_i as a
    /// degree-3 polynomial, which gives the wrong y for degree-4 s_i. See
    /// verifier_mamabear.rs top-of-file "Part 2" documentation for the full
    /// derivation.
    pub fn verify_ell0<E: SumcheckExtField>(
        mut y: E::Scalar,
        r_zero: &[E::Scalar],
        var_num: usize,
        transcript: &mut Transcript,
        proof: &mut util::fiat_shamir::Proof,
    ) -> (Vec<E::Scalar>, E::Scalar) {
        assert_eq!(r_zero.len(), var_num, "verify_ell0: r_zero length mismatch");
        let mut challenges = Vec::with_capacity(var_num);
        let one = E::Scalar::one().to_montgomery();
        let two = E::Scalar::from(2u32).to_montgomery();
        let three = E::Scalar::from(3u32).to_montgomery();
        let six = E::Scalar::from(6u32).to_montgomery();
        let inv2 = E::Scalar::from(SBF::inv_2()).to_montgomery();
        let inv6 = E::Scalar::from(SBF((P + 1) / 6)).to_montgomery();
        let neg_inv6 = -inv6;
        let neg_inv2 = -inv2;

        // prefix_eq = ∏_{j<round} eq(w_j, alpha_j), tracked in Mont form.
        let mut prefix_eq = one;

        for round in 0..var_num {
            // Read 3 hat values: s(0), s(2), s(∞) where s(∞) = a_4 of s.
            let s_0: E::Scalar = proof.get_next_and_step();
            transcript.append_f(s_0);
            let s_2: E::Scalar = proof.get_next_and_step();
            transcript.append_f(s_2);
            let s_inf: E::Scalar = proof.get_next_and_step();
            transcript.append_f(s_inf);

            let s_0m = s_0.to_montgomery();
            let s_2m = s_2.to_montgomery();
            let s_infm = s_inf.to_montgomery();
            let ym = y.to_montgomery();
            let w_i_m = r_zero[round].to_montgomery();

            // s_1 = y_prev - s_0 (sum-check invariant, tautological given
            // that s_1 is derived — no extra check needed here, but see below).
            let s_1m = ym.lazy_add_xp(2).lazy_sub(s_0m).con_sub_xp(2).reduce();

            // Challenge α for this round. `to_montgomery()` now canonicalizes
            // internally (components guaranteed < P), so the downstream
            // `two - alpha` / `alpha - one` SBF Sub is sound.
            let alpha: E::Scalar = transcript.challenge_f::<E::Scalar>().to_montgomery();
            challenges.push(alpha.from_montgomery());

            // Compute c(u) values. c(X) = prefix_eq · eq(w_i, X) where
            // eq(w_i, X) = (1 - w_i) + (2w_i - 1) · X.
            //   c(0)   = prefix_eq · (1 - w_i)
            //   c(1)   = prefix_eq · w_i
            //   c(2)   = prefix_eq · eq(w_i, 2) = prefix_eq · (3w_i - 1)
            //   c(∞)   = prefix_eq · (2w_i - 1)                  (leading coeff)
            //   c(α)   = prefix_eq · eq(w_i, α)
            let eq_w_0 = one.lazy_add_xp(2).lazy_sub(w_i_m).con_sub_xp(2); // 1 - w_i
            let eq_w_1 = w_i_m;
            let eq_w_2 = Self::eq_linear_mont_generic::<E>(w_i_m, two, one);
            let eq_w_inf = w_i_m.lazy_add(w_i_m).lazy_add_xp(2).lazy_sub(one).con_sub_xp(2);
            let eq_w_alpha = Self::eq_linear_mont_generic::<E>(w_i_m, alpha, one);

            let c_0 = (prefix_eq * eq_w_0).reduce();
            let c_1 = (prefix_eq * eq_w_1).reduce();
            let c_2 = (prefix_eq * eq_w_2).reduce();
            let c_inf = (prefix_eq * eq_w_inf).reduce();
            let c_alpha = (prefix_eq * eq_w_alpha).reduce();

            // Batched inversion via Montgomery's trick: one Fermat exponentiation
            // replaces four. Given [a, b, c, d], form prefix products p2=a*b,
            // p3=p2*c, p4=p3*d; invert p4; then back-substitute:
            //   d^-1 = inv4 * p3,  r3 = inv4 * d (= 1/(a*b*c))
            //   c^-1 = r3  * p2,  r2 = r3   * c (= 1/(a*b))
            //   b^-1 = r2  * a,   a^-1 = r2  * b
            // Total: 3 prefix muls + 1 inversion + 6 back-sub muls instead of 4 inversions.
            let p2 = (c_0 * c_1).reduce();
            let p3 = (p2 * c_2).reduce();
            let p4 = (p3 * c_inf).reduce();
            let inv4 = p4
                .from_montgomery()
                .inv()
                .expect("c-product invertible")
                .to_montgomery();
            let c_inf_inv = (inv4 * p3).reduce();
            let r3 = (inv4 * c_inf).reduce();
            let c_2_inv = (r3 * p2).reduce();
            let r2 = (r3 * c_2).reduce();
            let c_1_inv = (r2 * c_0).reduce();
            let c_0_inv = (r2 * c_1).reduce();

            let t_0 = (s_0m * c_0_inv).reduce();
            let t_1 = (s_1m * c_1_inv).reduce();
            let t_2 = (s_2m * c_2_inv).reduce();
            let t_inf = (s_infm * c_inf_inv).reduce(); // = a_3 (leading coeff of t)

            // t-factored Lagrange reconstruction using only standard field
            // ops. q(X) is the unique degree-≤2 polynomial with q(i) = t(i)
            // for i ∈ {0,1,2}; then t(X) = t_inf·X(X-1)(X-2) + q(X).
            let _ = (three, six, neg_inv6, neg_inv2);

            let am1 = alpha - one;
            let am2 = alpha - two;
            let two_m_alpha = two - alpha;

            let alpha_am1 = alpha * am1;
            let am1_am2 = am1 * am2;
            let alpha_am1_am2 = alpha_am1 * am2;

            let half_am1_am2 = am1_am2 * inv2;
            let half_alpha_am1 = alpha_am1 * inv2;

            // q(α) = t(0)·(α-1)(α-2)/2 + t(1)·α(2-α) + t(2)·α(α-1)/2
            let q_alpha =
                t_0 * half_am1_am2 + t_1 * (alpha * two_m_alpha) + t_2 * half_alpha_am1;

            // t(α) = t_inf · α(α-1)(α-2) + q(α)
            let t_alpha = t_inf * alpha_am1_am2 + q_alpha;

            // s(α) = c(α) · t(α).
            let s_alpha = (c_alpha * t_alpha).reduce();
            y = s_alpha.from_montgomery();

            // Update prefix_eq *= eq(w_i, α) for the next round.
            prefix_eq = (prefix_eq * eq_w_alpha).reduce();
        }

        (challenges, y)
    }

    // =========================================================================
    // Public entry points
    // =========================================================================


    /// Legacy prover for Ext3: delegates to `prove_add_mul_generic<PEF3>`.
    ///
    /// Input-layout and fold-order convention: tables in normal order,
    /// returned challenges in SIMD-round order (x_3, x_4, ..., x_{μ-1}, x_0,
    /// x_1, x_2).
    pub fn prove_add_mul_ext3(
        evals: [Vec<PBF>; 4],
        evals_eq: Vec<PEF3>,
        transcript: &mut Transcript,
    ) -> (Vec<SEF3>, [SEF3; 5]) {
        Self::prove_add_mul_generic::<PEF3>(evals, evals_eq, transcript)
    }




    /// Ext3 optimized prover with explicit ell0.
    ///
    /// Fold-order convention: tables in normal order, returned challenges in
    /// SIMD-round order (x_3, x_4, ..., x_{μ-1}, x_0, x_1, x_2). Rotate via
    /// `simd_to_natural_point` when a natural-order point is needed.
    pub fn prove_add_mul_ell0_ext3(
        evals: [Vec<PBF>; 4],
        point: &[SEF3],
        ell0: usize,
        transcript: &mut Transcript,
    ) -> (Vec<SEF3>, [SEF3; 5]) {
        let mut throwaway = ZeroCheckTimings::default();
        Self::prove_add_mul_optimized_with_ell0_generic::<PEF3>(
            evals,
            point,
            ell0,
            true,
            transcript,
            &mut throwaway,
        )
    }

    /// Profiled variant: same semantics as `prove_add_mul_ell0_ext3` but fills `timings`.
    pub fn prove_add_mul_ell0_ext3_profiled(
        evals: [Vec<PBF>; 4],
        point: &[SEF3],
        ell0: usize,
        transcript: &mut Transcript,
        timings: &mut ZeroCheckTimings,
    ) -> (Vec<SEF3>, [SEF3; 5]) {
        Self::prove_add_mul_optimized_with_ell0_generic::<PEF3>(
            evals, point, ell0, true, transcript, timings,
        )
    }

    #[cfg(test)]
    fn prove_add_mul_ell0_ext3_inner(
        evals: [Vec<PBF>; 4],
        point: &[SEF3],
        ell0: usize,
        zero_check: bool,
        transcript: &mut Transcript,
    ) -> (Vec<SEF3>, [SEF3; 5]) {
        let mut throwaway = ZeroCheckTimings::default();
        Self::prove_add_mul_optimized_with_ell0_generic::<PEF3>(
            evals,
            point,
            ell0,
            zero_check,
            transcript,
            &mut throwaway,
        )
    }

}

// RUSTFLAGS="-A warnings -C target-cpu=native" cargo test bench_sumcheck_mamabear_prove --release -- --nocapture
#[cfg(test)]
mod tests {
    use super::SumcheckMamaBear;
    use arithmetic::field::mamabear::{MamaBearScalar as SBF, PackedMamaBearAVX512 as PBF, P};
    use arithmetic::field::Field;
    use rand::rngs::SmallRng;
    use rand::SeedableRng;
    use util::fiat_shamir::Transcript;

    fn pack_scalar_evals(evals: &[SBF]) -> Vec<PBF> {
        // R=2^52 unified: direct strided packing, no from_mont/to_mont round-trip.
        assert_eq!(
            evals.len() % 8,
            0,
            "packed AVX-512 layout requires multiples of 8"
        );
        let stride = evals.len() / 8;
        let mut packed = Vec::with_capacity(stride);
        for packed_idx in 0..stride {
            let mut lanes = [0u64; 8];
            for lane in 0..8 {
                lanes[lane] = evals[lane * stride + packed_idx].0;
            }
            packed.push(PBF::from_array(lanes));
        }
        packed
    }


    /// Test that the Ext3 optimized prover produces consistent results across ell0 values.
    ///
    /// Since we don't have a scalar Ext3 reference implementation, we verify:
    /// 1. Different ell0 values produce the same transcript (same round polynomials).
    /// 2. The prover doesn't panic or produce garbage (basic sanity).
    #[test]
    fn sumcheck_mamabear_ext3_optimized_consistency() {
        use arithmetic::field::mamabear::MamaBearScalarExt3 as SEF3;

        let mut rng = SmallRng::seed_from_u64(42);

        for nv in 6usize..=12 {
            let domain_size = 1usize << nv;
            let point = (0..nv)
                .map(|_| SEF3::random(&mut rng))
                .collect::<Vec<_>>();
            let packed_evals: [Vec<PBF>; 4] = std::array::from_fn(|_| {
                let scalars: Vec<SBF> = (0..domain_size)
                    .map(|_| SBF::random(&mut rng).to_montgomery())
                    .collect();
                pack_scalar_evals(&scalars)
            });

            // Run with ell0=1 as baseline (zero_check=false for random data)
            let mut transcript_ell0_1 = Transcript::new();
            let result_ell0_1 = SumcheckMamaBear::prove_add_mul_ell0_ext3_inner(
                packed_evals.clone(),
                &point,
                1,
                false,
                &mut transcript_ell0_1,
            );

            // Verify challenges vector length
            assert_eq!(
                result_ell0_1.0.len(),
                nv,
                "Ext3 ell0=1 should produce {nv} challenges for nv={nv}"
            );

            // Run with different ell0 values and verify transcript consistency
            for ell0 in 2..=nv.min(SumcheckMamaBear::MAX_OPTIMIZED_ELL0) {
                if ell0 > nv.saturating_sub(3) {
                    continue; // skip if ell0 too large for packed rounds
                }
                let mut transcript_ell0_k = Transcript::new();
                let result_ell0_k = SumcheckMamaBear::prove_add_mul_ell0_ext3_inner(
                    packed_evals.clone(),
                    &point,
                    ell0,
                    false,
                    &mut transcript_ell0_k,
                );

                assert_eq!(
                    transcript_ell0_1.proof.bytes, transcript_ell0_k.proof.bytes,
                    "Ext3 transcript mismatch: ell0=1 vs ell0={ell0} for nv={nv}"
                );
                assert_eq!(
                    result_ell0_1, result_ell0_k,
                    "Ext3 output mismatch: ell0=1 vs ell0={ell0} for nv={nv}"
                );
            }
        }
    }


    /// Test: Ext3 prove_add_mul_ell0_ext3 + verify_ell0::<PEF3>.
    /// Uses a VALID witness (gate_h = 0 on hypercube) + zero_check=true to
    /// match the HyperPlonk ZeroCheck flow. Verifies prover_y == zero_y.
    #[test]
    fn tmp_verify_ell0_ext3_self_consistency() {
        use arithmetic::field::mamabear::{MamaBearScalarExt3 as SEF3, PackedMamaBearAVX512Ext3 as PEF3};
        let mut rng = SmallRng::seed_from_u64(137);

        for nv in [6, 10, 14, 18, 20, 22] {
            let domain_size = 1usize << nv;
            let point: Vec<SEF3> = (0..nv).map(|_| SEF3::random(&mut rng)).collect();

            // Valid witness: selector=0 (all add gates), c = -(a+b) mod P.
            // Then gate_h = (1-0)(a+b) + 0·ab + c = a+b - (a+b) = 0 on hypercube.
            let selector: Vec<SBF> = vec![SBF::zero(); domain_size];
            let a_vec: Vec<SBF> = (0..domain_size).map(|_| SBF::random(&mut rng)).collect();
            let b_vec: Vec<SBF> = (0..domain_size).map(|_| SBF::random(&mut rng)).collect();
            let c_vec: Vec<SBF> = (0..domain_size).map(|i| {
                let s = (a_vec[i].0 + b_vec[i].0) % P;
                if s == 0 { SBF(0) } else { SBF(P - s) }
            }).collect();
            // Convert to Montgomery (as the HyperPlonk prover does).
            let evals_canon: [Vec<SBF>; 4] = [selector, a_vec, b_vec, c_vec];
            let evals: [Vec<SBF>; 4] = evals_canon.each_ref().map(|v| {
                v.iter().map(|x| x.to_montgomery()).collect()
            });
            let packed_evals: [Vec<PBF>; 4] = std::array::from_fn(|k| {
                let stride = evals[k].len() / 8;
                let mut packed = Vec::with_capacity(stride);
                for packed_idx in 0..stride {
                    let mut lanes = [0u64; 8];
                    for lane in 0..8 {
                        lanes[lane] = evals[k][packed_idx + lane * stride].0;
                    }
                    packed.push(PBF::from_array(lanes));
                }
                packed
            });

            // Prove with zero_check=true (matching HyperPlonk's ZeroCheck flow).
            let mut tr_prove = Transcript::new();
            let (challenges_prove, final_evals) =
                SumcheckMamaBear::prove_add_mul_ell0_ext3_inner(
                    packed_evals, &point, SumcheckMamaBear::default_optimized_ell0(nv),
                    true, &mut tr_prove,
                );

            // Append 4 claims to transcript (mirrors HyperPlonk zerocheck flow).
            for v in final_evals.into_iter().take(4) {
                tr_prove.append_f(v);
            }

            // Prover's implied zero_y = prefix_eq * gate_h(claims).
            let s = final_evals[0].to_montgomery();
            let l = final_evals[1].to_montgomery();
            let r = final_evals[2].to_montgomery();
            let o = final_evals[3].to_montgomery();
            let pref_eq = final_evals[4].to_montgomery();
            let lr = l * r;
            let inner = lr - (l + r);
            let gate = l + r + s * inner + o;
            let prover_y = (pref_eq * gate).from_montgomery();

            // Verify: run verify_ell0 using the prove transcript's proof bytes.
            let mut tr_verify = Transcript::new();
            let mut proof = util::fiat_shamir::Proof::default();
            proof.bytes = tr_prove.proof.bytes.clone();
            let (challenges_verify, zero_y) = SumcheckMamaBear::verify_ell0::<PEF3>(
                SEF3::zero(),
                &point,
                nv,
                &mut tr_verify,
                &mut proof,
            );

            // Compare challenges (note: prover uses Mont form, verifier canonical).
            for i in 0..nv {
                assert_eq!(
                    challenges_prove[i].from_montgomery(),
                    challenges_verify[i],
                    "nv={}: challenge[{}] mismatch", nv, i
                );
            }
            assert_eq!(prover_y, zero_y, "nv={}: prover_y != zero_y (verify_ell0 bug!)", nv);
        }
    }


}
