//! Degree-generic, column-generic ZeroCheck sumcheck prover.
//!
//! This is a copy-and-generalize of the add/mul optimized prover
//! `SumcheckMamaBear::prove_add_mul_optimized_with_ell0_generic` (see
//! `mamabear/sumcheck.rs`), lifted along two compile-time axes:
//!
//! - `const D: usize` — the per-round gate degree (excluding the linear `eq`/Lagrange
//!   factor `c_i(X)`). The protocol degree is `D + 1`. The round polynomial `t_i(X)`
//!   (after factoring out `c_i`) has degree `D`, so the prover sends `D` hat points.
//! - `const NUM_COLS: usize` — the number of evaluation columns (selectors + wires).
//!
//! The gate identity itself is abstracted behind the `ZeroCheckGate` trait, so the same
//! generic prover can serve any degree-`D`, `NUM_COLS`-column gate. `AddMulD3` is the
//! sanity instance: it reproduces the legacy add/mul gate
//! `h = L+R + S*(L*R - L-R) + O` over the 4 columns `[S, L, R, O]`, and
//! `prove_zero_check_generic::<E, AddMulD3, 3, 4>` is **byte-identical**
//! to `prove_add_mul_ell0_ext3`.
//!
//! # Encoding of the grid `U_d`
//!
//! Each round variable is interpolated on the grid `U_d = {0, 1, ..., D-1, ∞}`,
//! encoded as slots `{0, 1, ..., D-1, D}` (slot `D` carries the leading coefficient,
//! i.e. the value at `∞`). `U_SIZE = D + 1`.
//!
//! The prover sends `HAT_SIZE = D` evaluations per round, at the hat points
//! `Û_d = U_d \ {1} = {0, 2, 3, ..., D-1, ∞}` (slot encoding `[0, 2, 3, ..., D-1, D]`).
//! Point `1` is skipped: the verifier recovers `s_i(1)` from `s_i(0) + s_i(1) = σ_{i-1}`.
//! For `D=3` the hat points are `{0, 2, ∞}`; for `D=5`, `{0, 2, 3, 4, ∞}`.
//!
//! # Performance discipline
//!
//! This prover follows the same 10-point checklist as the add/mul prover: three phases
//! (ell0 small-value precompute / fused packed rounds / single-block scalar tail),
//! `mul_base_elem` for base×ext products, two-stage eq tables, `[0, 2P)` lazy invariant
//! with documented ranges, and a strictly separate `_par` mirror (`zerocheck_generic_par.rs`).
//! The fused fold-plus-round-poly kernel scans each source quad once.
//!
//! The leading coefficient (the `∞` hat point) is computed gate-specifically via
//! `ZeroCheckGate::h_leading_*` (a homogeneous degree-`D` form in the per-column slopes)
//! rather than by a generic `(D+1)`-point finite difference. For `AddMulD3` this is
//! `diff_S * diff_L * diff_R`, exactly the legacy fused-kernel expression, which keeps the
//! `D=3` hot path's instruction count identical to the hand-written prover.

use arithmetic::field::mamabear::*;
use arithmetic::field::Field;
use util::fiat_shamir::Transcript;

use crate::sumcheck_mamabear::{
    MontgomeryOps, RoundEqView, SumcheckExtField, SumcheckMamaBear, TwoStageEqTables,
    ZeroCheckTimings,
};

use arithmetic::field::mamabear::{MamaBearScalar as SBF, PackedMamaBearAVX512 as PBF};

/// Maximum supported `U_SIZE = D + 1`. Caps `D <= 7`.
pub(crate) const MAX_U: usize = 8;

// ===========================================================================
// The gate trait
// ===========================================================================

/// A degree-`D`, `NUM_COLS`-column ZeroCheck gate identity.
///
/// The prover only ever evaluates `h` on real field points (never on the boolean
/// hypercube alone), so the implementation must be a pure polynomial evaluation —
/// **no dynamic dispatch, no branches on data** — to keep AVX-512 codegen intact.
///
/// The `h_*` methods evaluate the gate at a finite point (the columns are given as the
/// already-interpolated field values). The `h_leading_*` methods return the leading
/// `X^D` coefficient of `h(col_0(X), ..., col_{NC-1}(X))` given the per-column slopes
/// `diff_c = col_c(1) - col_c(0)`; this is the value at the `∞` hat point.
pub trait ZeroCheckGate<E: SumcheckExtField, const D: usize, const NUM_COLS: usize> {
    /// The gate degree (per round variable), excluding the linear `c_i(X)` factor.
    const DEGREE: usize = D;

    /// Gate `h` on packed base-field columns (used by the small-value precompute).
    fn h_packed_base(inputs: [PBF; NUM_COLS], one: PBF) -> PBF;
    /// Gate `h` on packed ext-field columns (used by the packed/tail rounds).
    fn h_packed_ext(inputs: [E; NUM_COLS], one: E) -> E;
    /// Gate `h` on scalar ext-field columns (used by the scalar fallback / verifier-side checks).
    fn h_scalar_ext(inputs: [E::Scalar; NUM_COLS], one: E::Scalar) -> E::Scalar;

    /// Leading `X^D` coefficient of `h` given per-column slopes, packed base field.
    fn h_leading_packed_base(diffs: [PBF; NUM_COLS], one: PBF) -> PBF;
    /// Leading `X^D` coefficient of `h` given per-column slopes, packed ext field.
    fn h_leading_packed_ext(diffs: [E; NUM_COLS], one: E) -> E;
    /// Leading `X^D` coefficient of `h` given per-column slopes, scalar ext field.
    fn h_leading_scalar_ext(diffs: [E::Scalar; NUM_COLS], one: E::Scalar) -> E::Scalar;
}

// ===========================================================================
// AddMulD3 — the sanity gate (oracle target for the byte-identity test)
// ===========================================================================

/// The legacy add/mul gate `h = (1-S)(L+R) + S·L·R + O`, columns `[S, L, R, O]`, degree 3.
///
/// Delegates `h_packed_base` / `h_packed_ext` to the legacy implementations so the generic
/// `D=3` path is byte-identical to `prove_add_mul_ell0_ext{2,3}`. The leading coefficient
/// is `diff_S * diff_L * diff_R` (the `X^3` term of `S·L·R`), exactly the legacy fused
/// kernel's `diffs[0]*diffs[1]*diffs[2]`.
pub struct AddMulD3;

impl<E: SumcheckExtField> ZeroCheckGate<E, 3, 4> for AddMulD3 {
    #[inline(always)]
    fn h_packed_base(inputs: [PBF; 4], one: PBF) -> PBF {
        SumcheckMamaBear::gate_h_packed_base(inputs, one)
    }
    #[inline(always)]
    fn h_packed_ext(inputs: [E; 4], one: E) -> E {
        SumcheckMamaBear::gate_h_packed_ext_generic(inputs, one)
    }
    #[inline(always)]
    fn h_scalar_ext(inputs: [E::Scalar; 4], _one: E::Scalar) -> E::Scalar {
        // h = L+R + S*(L*R - L - R) + O, mirroring gate_h_scalar_generic.
        let s = inputs[0];
        let l = inputs[1];
        let r = inputs[2];
        let o = inputs[3];
        let l_plus_r = l.lazy_add(r).con_sub_xp(2); // [0, 2P)
        let lr = l * r; // [0, 2P)
        let diff = lr.lazy_add_xp(2).lazy_sub(l_plus_r); // [0, 4P)
        let term = s * diff.con_sub_xp(2); // [0, 2P)
        l_plus_r.lazy_add(term).lazy_add(o).con_sub_xp(3) // [0, 3P)
    }
    #[inline(always)]
    fn h_leading_packed_base(diffs: [PBF; 4], _one: PBF) -> PBF {
        diffs[0] * diffs[1] * diffs[2]
    }
    #[inline(always)]
    fn h_leading_packed_ext(diffs: [E; 4], _one: E) -> E {
        diffs[0] * diffs[1] * diffs[2]
    }
    #[inline(always)]
    fn h_leading_scalar_ext(diffs: [E::Scalar; 4], _one: E::Scalar) -> E::Scalar {
        diffs[0] * diffs[1] * diffs[2]
    }
}

// ===========================================================================
// Compile-time-constant gate constants (finite-point and finite-difference)
// ===========================================================================

/// Precomputed base-field constants for the small-value tensor evaluation.
///
/// All entries are in Montgomery form. Only indices `0..=D` are populated.
pub(crate) struct GateConstsBase {
    /// `point_mont[p] = Mont(p)` for `p in 0..=D` (used to evaluate at finite point `p`).
    pub(crate) point_mont: [PBF; MAX_U],
    /// `fd[k] = (-1)^(D-k) * C(D, k)` for `k in 0..=D`, in field form (sign baked in).
    pub(crate) fd: [PBF; MAX_U],
    /// `inv_fact = (D!)^{-1}`.
    pub(crate) inv_fact: PBF,
    /// `Mont(1)` at the base-field level.
    pub(crate) one: PBF,
}

#[inline]
fn binomial(n: usize, k: usize) -> u64 {
    let mut num: u128 = 1;
    let mut den: u128 = 1;
    let mut i = 0;
    while i < k {
        num *= (n - i) as u128;
        den *= (i + 1) as u128;
        i += 1;
    }
    (num / den) as u64
}

/// Build the degree-`D` base-field gate constants.
pub(crate) fn gate_consts_base<const D: usize>() -> GateConstsBase {
    assert!(D >= 1 && D < MAX_U, "ZeroCheck degree D must satisfy 1 <= D <= {}", MAX_U - 1);
    let mut point_mont = [PBF::zero(); MAX_U];
    let mut fd = [PBF::zero(); MAX_U];
    for p in 0..=D {
        point_mont[p] = PBF::from(p as u64).to_montgomery();
        let cbin = binomial(D, p);
        // (-1)^(D-p) * C(D,p) as a normal-form value in [0, P).
        let signed = if (D - p) % 2 == 0 { cbin } else { P - cbin };
        fd[p] = PBF::from(signed).to_montgomery();
    }
    // (D!)^{-1} via the scalar field inverse, broadcast to all lanes in Montgomery form.
    let fact_d: u64 = (1..=D as u64).product();
    let inv_fact_normal = SBF::from(fact_d).inv().expect("D! invertible mod P").0;
    let inv_fact = PBF::from(inv_fact_normal).to_montgomery();
    GateConstsBase {
        point_mont,
        fd,
        inv_fact,
        one: PBF::one().to_montgomery(),
    }
}

// ===========================================================================
// Degree-generic Lagrange basis (grid {0, ..., D-1, ∞})
// ===========================================================================

/// Lagrange basis on the grid `U_d = {0, 1, ..., D-1, ∞}`, evaluated at `x` (Mont form).
///
/// Returns `[L_0(x), ..., L_{D-1}(x), L_∞(x)]` in the first `D+1` slots of a `MAX_U` array:
/// - For finite node `j in 0..D-1`: `L_j(x) = ∏_{k≠j, k<D} (x-k) / ∏_{k≠j, k<D} (j-k)`
///   (the degree-`(D-1)` Lagrange basis over the `D` finite nodes `{0, ..., D-1}`).
/// - `L_∞(x) = ∏_{k=0}^{D-1} (x-k)` (the monic degree-`D` poly vanishing at all finite
///   nodes; it contributes only the leading coefficient).
///
/// Fully unrolled over the const `D` (the products and the per-node denominator inverse are
/// const-trip loops that LLVM unrolls). Only called `ell0` times, so the per-node inversion
/// is negligible.
pub(crate) fn lagrange_basis_degree_d_generic<E: SumcheckExtField, const D: usize>(
    x: E::Scalar,
    one: E::Scalar,
) -> [E::Scalar; MAX_U] {
    let mut out = [E::Scalar::zero(); MAX_U];
    // node[k] = Mont(k) for k in 0..D.
    let mut node = [E::Scalar::zero(); MAX_U];
    for k in 0..D {
        node[k] = E::Scalar::from(k as u32).to_montgomery();
    }
    // x_minus[k] = x - node[k] in [0, ...] (field Sub keeps a valid representative).
    let mut x_minus = [E::Scalar::zero(); MAX_U];
    for k in 0..D {
        x_minus[k] = x - node[k];
    }
    // L_∞ = ∏_{k<D} (x - k).
    let mut linf = one;
    for k in 0..D {
        linf = linf * x_minus[k];
    }
    out[D] = linf;
    // Finite nodes.
    for j in 0..D {
        // numerator = ∏_{k≠j} (x - k)
        let mut numer = one;
        for k in 0..D {
            if k != j {
                numer = numer * x_minus[k];
            }
        }
        // denom = ∏_{k≠j} (node[j] - node[k])
        let mut denom = one;
        for k in 0..D {
            if k != j {
                denom = denom * (node[j] - node[k]);
            }
        }
        let inv_denom = denom
            .from_montgomery()
            .inv()
            .expect("Lagrange denominator invertible")
            .to_montgomery();
        out[j] = numer * inv_denom;
    }
    out
}

// ===========================================================================
// Small-value weight (Kronecker) update + contraction
// ===========================================================================

/// `R_{i+1} = R_i ⊗ basis`. `basis` is valid in slots `0..D+1`.
pub(crate) fn update_small_value_weights<E: SumcheckExtField, const D: usize>(
    current: &[E::Scalar],
    basis: &[E::Scalar; MAX_U],
) -> Vec<E::Scalar> {
    let u_size = D + 1;
    let state_count = current.len();
    let mut next = vec![E::Scalar::zero(); state_count * u_size];
    for (state, &coeff) in current.iter().enumerate() {
        for basis_idx in 0..u_size {
            next[state + basis_idx * state_count] = coeff * basis[basis_idx];
        }
    }
    next
}

/// `t_i(u) = Σ_v R_i[v] · A_i(v, u)` for each hat point `u`.
pub(crate) fn compute_t_from_precomputed<E: SumcheckExtField, const D: usize>(
    round_table: &[E::Scalar],
    weights: &[E::Scalar],
) -> [E::Scalar; D] {
    let states = weights.len();
    let mut t_hat = [E::Scalar::zero(); D];
    for hat_idx in 0..D {
        let base = hat_idx * states;
        let mut acc = E::Scalar::zero();
        for state in 0..states {
            acc = acc.lazy_add(weights[state] * round_table[base + state]);
        }
        t_hat[hat_idx] = acc.reduce();
    }
    t_hat
}

/// `s_i(u) = c_i(u) · t_i(u)` at each hat point. `c_i(X) = prefix_eq · eq(w_i, X)`.
///
/// Hat slot layout: `hat_idx 0 -> point 0`, `hat_idx D-1 -> ∞`, otherwise finite point
/// `hat_idx + 1` (so `hat_idx 1 -> 2`, `hat_idx 2 -> 3`, ...).
pub(crate) fn compute_round_s_from_t<E: SumcheckExtField, const D: usize>(
    prefix_eq: E::Scalar,
    w_i: E::Scalar,
    t_hat: [E::Scalar; D],
    one: E::Scalar,
    finite_point_ext: &[E::Scalar; MAX_U],
) -> [E::Scalar; D] {
    let mut s = [E::Scalar::zero(); D];
    for hat_idx in 0..D {
        let eq_val = if hat_idx == 0 {
            // eq(w_i, 0) = 1 - w_i.
            one.lazy_add_xp(2).lazy_sub(w_i) // [0, 4P)
        } else if hat_idx == D - 1 {
            // eq(w_i, ∞) = leading coeff = 2 w_i - 1.
            w_i.lazy_add(w_i).lazy_add_xp(2).lazy_sub(one).con_sub_xp(2) // [0, 4P)
        } else {
            // eq(w_i, p), finite point p = hat_idx + 1.
            SumcheckMamaBear::eq_linear_mont_generic::<E>(w_i, finite_point_ext[hat_idx + 1], one)
        };
        s[hat_idx] = prefix_eq * eq_val * t_hat[hat_idx];
    }
    s
}

/// Append the `D` hat values to the transcript (in canonical normal form).
pub(crate) fn append_hat_round_values<E: SumcheckExtField, const D: usize>(
    transcript: &mut Transcript,
    values: [E::Scalar; D],
) {
    for value in values {
        transcript.append_f(value.from_montgomery());
    }
}

// ===========================================================================
// Finite-point tensor evaluation + axis transform (base field, precompute)
// ===========================================================================

/// Build the `(D+1)^prefix_len` finite grid of one column from its `2^prefix_len` boolean
/// block, in `O(grid_len)` (each cell written once via per-axis finite-line extension).
///
/// Layout: `grid[idx]` with `idx = Σ_a digit_a · (D+1)^a` (axis 0 lowest), `digit_a ∈ 0..=D`.
/// Boolean block layout: `block[bx]` with `bx = Σ_a bit_a · 2^a` (axis 0 = least significant
/// bit), matching `eval_packed_block_at_finite_points`'s fold order in the legacy prover.
pub(crate) fn build_finite_grid_base<const D: usize>(
    block: &[PBF],
    prefix_len: usize,
    consts: &GateConstsBase,
    grid: &mut [PBF],
) {
    let u = D + 1;
    // Step 1: scatter the boolean corners (digits 0/1) into the grid.
    for (bx, &bv) in block.iter().enumerate().take(1usize << prefix_len) {
        let mut gidx = 0usize;
        let mut s = 1usize;
        for a in 0..prefix_len {
            gidx += ((bx >> a) & 1) * s;
            s *= u;
        }
        grid[gidx] = bv;
    }
    // Step 2: extend each axis from the boolean digits {0,1} to the full finite range {0..D}.
    // When extending axis `a`, axes `< a` are already full (range `u`) and axes `> a` are
    // still boolean (digits 0/1); a finite line along axis `a` only needs its digits 0 and 1.
    for a in 0..prefix_len {
        let stride = u.pow(a as u32);
        let n_high = 1usize << (prefix_len - 1 - a); // boolean combos of axes > a
        for low in 0..stride {
            for high in 0..n_high {
                let mut high_off = 0usize;
                let mut hs = stride * u; // u^(a+1)
                for b in 0..(prefix_len - 1 - a) {
                    high_off += ((high >> b) & 1) * hs;
                    hs *= u;
                }
                let base = low + high_off;
                let v0 = grid[base];
                let v1 = grid[base + stride];
                let diff = SumcheckMamaBear::packed_diff(v0, v1); // [0, 2P)
                for d in 2..u {
                    // f(d) = v0 + d·diff (multilinear extension is linear in this variable).
                    grid[base + d * stride] = v0
                        .lazy_add((diff * consts.point_mont[d]).reduce_fast())
                        .con_sub_xp(2); // [0, 2P)
                }
            }
        }
    }
}

/// Transform one tensor axis from the finite-evaluation basis `{0, 1, ..., D}` to the
/// `U_d` basis `{0, 1, ..., D-1, ∞}`. Slots `0..D` (the finite values `0..D-1`) are kept
/// as-is; slot `D` is overwritten with the leading coefficient `Δ^D / D!`.
pub(crate) fn transform_tensor_axis_to_ud_base<const D: usize>(
    tensor: &mut [PBF],
    axis: usize,
    consts: &GateConstsBase,
) {
    let u_size = D + 1;
    let stride = u_size.pow(axis as u32);
    let block = stride * u_size;
    let outer = tensor.len() / block;
    for outer_idx in 0..outer {
        let block_base = outer_idx * block;
        for inner_idx in 0..stride {
            let base = block_base + inner_idx;
            // leading = (Σ_{k=0}^{D} fd[k] · f_k) · inv_fact.
            let mut acc = PBF::zero();
            for k in 0..u_size {
                acc = acc.lazy_add(consts.fd[k] * tensor[base + k * stride]); // sum < 1.5P·(D+1) < 2^64
            }
            let leading = (acc.reduce_fast() * consts.inv_fact).reduce_fast(); // [0, 2.0001P)
            tensor[base + D * stride] = leading;
        }
    }
}

// ===========================================================================
// Per-group hat computation (ext / base field)
// ===========================================================================

/// Compute the `D` hat values for one packed group from per-column `v0` and slope `diff`
/// (ext field). Finite hat points `{0, 2, 3, ..., D-1}` are gate evaluations; the `∞` hat
/// is the gate-specific leading coefficient `h_leading`.
#[inline(always)]
pub(crate) fn compute_group_hats_ext<E: SumcheckExtField, G: ZeroCheckGate<E, D, NUM_COLS>, const D: usize, const NUM_COLS: usize>(
    v0s: &[E; NUM_COLS],
    diffs: &[E; NUM_COLS],
    one: E,
) -> [E; D] {
    let mut hats = [E::zero(); D];
    // hat 0: finite point 0.
    let vals0: [E; NUM_COLS] = std::array::from_fn(|c| v0s[c]);
    hats[0] = G::h_packed_ext(vals0, one);
    // hats 1..D-1: finite points 2, 3, ..., D-1 via running additions of diff.
    if D > 2 {
        // running[c] starts at point 1 (= v0 + diff).
        let mut running: [E; NUM_COLS] =
            std::array::from_fn(|c| v0s[c].lazy_add(diffs[c]).con_sub_xp(2));
        for hat_idx in 1..(D - 1) {
            for c in 0..NUM_COLS {
                running[c] = running[c].lazy_add(diffs[c]).con_sub_xp(2); // advance one finite point
            }
            let vals: [E; NUM_COLS] = std::array::from_fn(|c| running[c]);
            hats[hat_idx] = G::h_packed_ext(vals, one);
        }
    }
    // hat D-1: ∞ (leading coefficient).
    hats[D - 1] = G::h_leading_packed_ext(*diffs, one);
    hats
}

/// Base-field analogue of `compute_group_hats_ext` (used by the round-0 precompute fast path).
#[inline(always)]
pub(crate) fn compute_group_hats_base<E: SumcheckExtField, G: ZeroCheckGate<E, D, NUM_COLS>, const D: usize, const NUM_COLS: usize>(
    v0s: &[PBF; NUM_COLS],
    diffs: &[PBF; NUM_COLS],
    one: PBF,
) -> [PBF; D] {
    let mut hats = [PBF::zero(); D];
    let vals0: [PBF; NUM_COLS] = std::array::from_fn(|c| v0s[c]);
    hats[0] = G::h_packed_base(vals0, one);
    if D > 2 {
        let mut running: [PBF; NUM_COLS] =
            std::array::from_fn(|c| v0s[c].lazy_add(diffs[c]).con_sub_xp(2));
        for hat_idx in 1..(D - 1) {
            for c in 0..NUM_COLS {
                running[c] = running[c].lazy_add(diffs[c]).con_sub_xp(2);
            }
            let vals: [PBF; NUM_COLS] = std::array::from_fn(|c| running[c]);
            hats[hat_idx] = G::h_packed_base(vals, one);
        }
    }
    hats[D - 1] = G::h_leading_packed_base(*diffs, one);
    hats
}

// ===========================================================================
// Small-value precompute (base field, three rounds + generic fallback)
// ===========================================================================

/// Build the small-value tables `A_i(v, u)` for `i in 0..ell0`.
///
/// Round 0 uses a fast diff-based path (matching the legacy add/mul prover); rounds `>= 1`
/// use the generic finite-point tensor + axis transform. The result `small_value_tables[i]`
/// has `HAT_SIZE · U_SIZE^i = D · (D+1)^i` entries.
#[allow(clippy::too_many_arguments)]
pub(crate) fn precompute_small_value_tables<E: SumcheckExtField, G: ZeroCheckGate<E, D, NUM_COLS>, const D: usize, const NUM_COLS: usize>(
    evals: &[Vec<PBF>; NUM_COLS],
    eq_tables: &TwoStageEqTables<E>,
    ell0: usize,
    zero_check: bool,
    consts: &GateConstsBase,
) -> Vec<Vec<E::Scalar>> {
    let u_size = D + 1;
    let hat_size = D;
    let one = consts.one;
    let mut precomputed: Vec<Vec<E::Scalar>> = Vec::with_capacity(ell0);

    for round in 0..ell0 {
        let prefix_len = round + 1;
        let eq_view = SumcheckMamaBear::round_eq_view_generic(eq_tables, round);

        if round == 0 {
            // Fast path: one diff per column, gate at the finite hat points + leading coeff.
            let packed_groups = evals[0].len() >> 1;
            let mut t_acc = [E::zero(); D];
            let packed_split = eq_view.packed_split_for_groups(packed_groups);

            if let Some(split) = packed_split {
                let right_len = split.right_broadcast.len();
                for left_group in 0..split.left_packed.len() {
                    let mut inner = [E::zero(); D];
                    let group_base = left_group * right_len;
                    for right_idx in 0..right_len {
                        let eq_r = split.right_broadcast[right_idx];
                        let group_idx = group_base + right_idx;
                        let base = group_idx << 1;
                        let mut v0s = [PBF::zero(); NUM_COLS];
                        let mut diffs = [PBF::zero(); NUM_COLS];
                        for c in 0..NUM_COLS {
                            let v0 = evals[c][base];
                            let v1 = evals[c][base + 1];
                            v0s[c] = v0;
                            diffs[c] = SumcheckMamaBear::packed_diff(v0, v1);
                        }
                        let hats = compute_group_hats_base::<E, G, D, NUM_COLS>(&v0s, &diffs, one);
                        for k in 0..hat_size {
                            if zero_check && k == 0 {
                                continue;
                            }
                            inner[k] = inner[k].lazy_add(eq_r.mul_base_elem(hats[k]));
                        }
                    }
                    let eq_l = split.left_packed[left_group];
                    for k in 0..hat_size {
                        if zero_check && k == 0 {
                            continue;
                        }
                        t_acc[k] = t_acc[k].lazy_add(eq_l * inner[k].reduce_fast());
                    }
                }
            } else {
                for group_idx in 0..packed_groups {
                    let weight = eq_view.load_packed_weight(group_idx, packed_groups);
                    let base = group_idx << 1;
                    let mut v0s = [PBF::zero(); NUM_COLS];
                    let mut diffs = [PBF::zero(); NUM_COLS];
                    for c in 0..NUM_COLS {
                        let v0 = evals[c][base];
                        let v1 = evals[c][base + 1];
                        v0s[c] = v0;
                        diffs[c] = SumcheckMamaBear::packed_diff(v0, v1);
                    }
                    let hats = compute_group_hats_base::<E, G, D, NUM_COLS>(&v0s, &diffs, one);
                    for k in 0..hat_size {
                        if zero_check && k == 0 {
                            continue;
                        }
                        t_acc[k] = t_acc[k].lazy_add(weight.mul_base_elem(hats[k]));
                    }
                    if (group_idx + 1) % 8192 == 0 {
                        for k in 0..hat_size {
                            t_acc[k] = t_acc[k].reduce_fast();
                        }
                    }
                }
            }

            let mut row = Vec::with_capacity(hat_size);
            for k in 0..hat_size {
                if zero_check && k == 0 {
                    row.push(E::Scalar::zero());
                } else {
                    row.push(E::sum_lanes_to_mont(t_acc[k]));
                }
            }
            precomputed.push(row);
            continue;
        }

        // Generic finite-point tensor path for rounds >= 1.
        let states = u_size.pow(round as u32);
        let block_len = 1usize << prefix_len;
        let grid_len = u_size.pow(prefix_len as u32);
        let packed_groups = evals[0].len() / block_len;
        let mut round_table = vec![E::zero(); hat_size * states];
        let packed_split = eq_view.packed_split_for_groups(packed_groups);

        // Scratch buffers hoisted out of the group loop (no per-group allocation).
        // `poly_grids[c]` holds column c's full finite grid, built once per group in
        // O(grid_len) (vs O(grid_len · block_len) if we re-evaluated per grid point).
        let mut tensor = vec![PBF::zero(); grid_len];
        let mut poly_grids: [Vec<PBF>; NUM_COLS] =
            std::array::from_fn(|_| vec![PBF::zero(); grid_len]);

        for packed_idx in 0..packed_groups {
            let start = packed_idx * block_len;
            let end = start + block_len;
            let weight = match packed_split {
                Some(split) => split.weight(packed_idx),
                None => eq_view.load_packed_weight(packed_idx, packed_groups),
            };

            for c in 0..NUM_COLS {
                build_finite_grid_base::<D>(
                    &evals[c][start..end],
                    prefix_len,
                    consts,
                    &mut poly_grids[c],
                );
            }
            for finite_idx in 0..grid_len {
                let gate_inputs: [PBF; NUM_COLS] =
                    std::array::from_fn(|c| poly_grids[c][finite_idx]);
                tensor[finite_idx] = G::h_packed_base(gate_inputs, one);
            }

            for axis in 0..prefix_len {
                transform_tensor_axis_to_ud_base::<D>(&mut tensor, axis, consts);
            }

            for state in 0..states {
                for hat_idx in 0..hat_size {
                    let slot = if hat_idx == 0 { 0 } else { hat_idx + 1 };
                    let tensor_idx = state + slot * states;
                    round_table[hat_idx * states + state] = round_table[hat_idx * states + state]
                        .lazy_add(weight.mul_base_elem(tensor[tensor_idx]));
                }
            }

            if (packed_idx + 1) % 8192 == 0 {
                for entry in &mut round_table {
                    *entry = entry.reduce_fast();
                }
            }
        }

        precomputed.push(
            round_table
                .into_iter()
                .map(|entry| E::sum_lanes_to_mont(entry.reduce_fast()))
                .collect(),
        );
    }

    precomputed
}

// ===========================================================================
// Folds (base -> ext transition, ext in-place, single-block tail)
// ===========================================================================

/// Fold `NUM_COLS` base-field tables by all `challenges`, producing ext tables (allocating).
///
/// Sequential fold: one base->ext pass by `challenges[0]` (via `mul_base_elem`), then one
/// ext in-place fold per remaining challenge. `challenges[0]` eliminates the innermost
/// (adjacent-pair) bit, matching the add/mul prover's fold order, so the folded tables are
/// mod-P identical to `fold_base_tables_to_ext_generic`.
pub(crate) fn fold_base_tables_to_ext<E: SumcheckExtField, const NUM_COLS: usize>(
    evals: &[Vec<PBF>; NUM_COLS],
    challenges: &[E::Scalar],
) -> [Vec<E>; NUM_COLS] {
    debug_assert!(!challenges.is_empty());
    // Fused two-challenge path (the common ell0=2 case): one pass, single allocation.
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
                let low = alpha0.mul_base_elem(diff0).add_base_elem(e0).con_sub_xp(2);
                let high = alpha0.mul_base_elem(diff1).add_base_elem(e2).con_sub_xp(2);
                let diff = high.lazy_add_xp(2).lazy_sub(low);
                next.push((alpha1 * diff).lazy_add(low).con_sub_xp(2));
            }
            next
        });
    }
    // Fused three-challenge path (ell0=3).
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
    // Generic sequential path (ell0 = 1 or >= 4): one base->ext pass, then ext folds.
    let alpha0 = E::from_scalar(challenges[0]);
    let next_len = evals[0].len() >> 1;
    let mut folded: [Vec<E>; NUM_COLS] = std::array::from_fn(|poly_idx| {
        let mut next = Vec::with_capacity(next_len);
        for pair_idx in 0..next_len {
            let e0 = evals[poly_idx][pair_idx << 1];
            let e1 = evals[poly_idx][(pair_idx << 1) + 1];
            let diff = e1.lazy_add_xp(2).lazy_sub(e0); // [0, 4P)
            next.push(alpha0.mul_base_elem(diff).add_base_elem(e0).con_sub_xp(2)); // [0, 2P)
        }
        next
    });
    for &challenge in &challenges[1..] {
        fold_ext_tables_in_place::<E, NUM_COLS>(&mut folded, E::from_scalar(challenge));
    }
    folded
}

/// Fold `NUM_COLS` ext tables in-place by one challenge.
pub(crate) fn fold_ext_tables_in_place<E: SumcheckExtField, const NUM_COLS: usize>(
    evals: &mut [Vec<E>; NUM_COLS],
    challenge_mont: E,
) {
    let next_len = evals[0].len() >> 1;
    for poly in evals.iter_mut() {
        for pair_idx in 0..next_len {
            let v0 = poly[pair_idx << 1];
            let v1 = poly[(pair_idx << 1) + 1];
            let diff = v1.lazy_add_xp(2).lazy_sub(v0); // [0, 4P)
            poly[pair_idx] = (challenge_mont * diff).lazy_add(v0).con_sub_xp(2); // [0, 2P)
        }
        poly.truncate(next_len);
    }
}

/// Fold a single packed block (scalar tail) in-place by one challenge.
pub(crate) fn fold_single_packed_in_place<E: SumcheckExtField, const NUM_COLS: usize>(
    evals: &mut [E; NUM_COLS],
    challenge_mont: E,
    active_len: usize,
) {
    let pair_count = active_len >> 1;
    for poly in evals.iter_mut() {
        let (v0, _v1, diff) =
            SumcheckMamaBear::build_single_packed_pair_views_generic::<E>(*poly, pair_count);
        *poly = (challenge_mont * diff).con_sub_xp(2).lazy_add(v0).con_sub_xp(2);
    }
}

// ===========================================================================
// Fused fold + round-poly kernel, first packed round, scalar tail
// ===========================================================================

/// Fused single-scan fold + round-poly for the packed ext tables. Halves each table in
/// place and returns the `D` hat values `t_i(u)`.
pub(crate) fn fold_and_compute_round_t<E: SumcheckExtField, G: ZeroCheckGate<E, D, NUM_COLS>, const D: usize, const NUM_COLS: usize>(
    evals: &mut [Vec<E>; NUM_COLS],
    eq_view: &RoundEqView<'_, E>,
    challenge: E::Scalar,
) -> [E::Scalar; D] {
    let next_len = evals[0].len() >> 1;
    let packed_groups = next_len >> 1;
    let alpha = E::from_scalar(challenge);
    let one = E::one().ext_to_montgomery();
    let mut t_acc = [E::zero(); D];
    let packed_split = eq_view.packed_split_for_groups(packed_groups);

    if let Some(split) = packed_split {
        let right_len = split.right_broadcast.len();
        for left_group in 0..split.left_packed.len() {
            let mut inner = [E::zero(); D];
            let group_base = left_group * right_len;
            for right_idx in 0..right_len {
                let group_idx = group_base + right_idx;
                let eq_r = split.right_broadcast[right_idx];
                let src_base = group_idx << 2;
                let dst_base = group_idx << 1;
                let mut v0s = [E::zero(); NUM_COLS];
                let mut diffs = [E::zero(); NUM_COLS];
                for c in 0..NUM_COLS {
                    let e0 = evals[c][src_base];
                    let e1 = evals[c][src_base + 1];
                    let e2 = evals[c][src_base + 2];
                    let e3 = evals[c][src_base + 3];
                    let diff0 = e1.lazy_add_xp(2).lazy_sub(e0); // [0, 4P)
                    let diff1 = e3.lazy_add_xp(2).lazy_sub(e2); // [0, 4P)
                    let v0 = (alpha * diff0).lazy_add(e0).con_sub_xp(2); // [0, 2P)
                    let v1 = (alpha * diff1).lazy_add(e2).con_sub_xp(2); // [0, 2P)
                    let diff = v1.lazy_add_xp(2).lazy_sub(v0).con_sub_xp(2); // [0, 2P)
                    evals[c][dst_base] = v0;
                    evals[c][dst_base + 1] = v1;
                    v0s[c] = v0;
                    diffs[c] = diff;
                }
                let hats = compute_group_hats_ext::<E, G, D, NUM_COLS>(&v0s, &diffs, one);
                for k in 0..D {
                    inner[k] = inner[k].lazy_add(eq_r * hats[k]);
                }
            }
            let eq_l = split.left_packed[left_group];
            for k in 0..D {
                t_acc[k] = t_acc[k].lazy_add(eq_l * inner[k].reduce_fast());
            }
        }
    } else {
        for group_idx in 0..packed_groups {
            let weight = eq_view.load_packed_weight(group_idx, packed_groups);
            let src_base = group_idx << 2;
            let dst_base = group_idx << 1;
            let mut v0s = [E::zero(); NUM_COLS];
            let mut diffs = [E::zero(); NUM_COLS];
            for c in 0..NUM_COLS {
                let e0 = evals[c][src_base];
                let e1 = evals[c][src_base + 1];
                let e2 = evals[c][src_base + 2];
                let e3 = evals[c][src_base + 3];
                let diff0 = e1.lazy_add_xp(2).lazy_sub(e0);
                let diff1 = e3.lazy_add_xp(2).lazy_sub(e2);
                let v0 = (alpha * diff0).lazy_add(e0).con_sub_xp(2);
                let v1 = (alpha * diff1).lazy_add(e2).con_sub_xp(2);
                let diff = v1.lazy_add_xp(2).lazy_sub(v0).con_sub_xp(2);
                evals[c][dst_base] = v0;
                evals[c][dst_base + 1] = v1;
                v0s[c] = v0;
                diffs[c] = diff;
            }
            let hats = compute_group_hats_ext::<E, G, D, NUM_COLS>(&v0s, &diffs, one);
            for k in 0..D {
                t_acc[k] = t_acc[k].lazy_add(weight * hats[k]);
            }
            if (group_idx + 1) % 8192 == 0 {
                for k in 0..D {
                    t_acc[k] = t_acc[k].reduce_fast();
                }
            }
        }
    }

    for poly in evals.iter_mut() {
        poly.truncate(next_len);
    }

    let mut out = [E::Scalar::zero(); D];
    for k in 0..D {
        out[k] = E::sum_lanes_to_mont(t_acc[k]);
    }
    out
}

/// Compute the `D` hat values from already-folded ext tables (no fold; the first packed round).
pub(crate) fn compute_round_t<E: SumcheckExtField, G: ZeroCheckGate<E, D, NUM_COLS>, const D: usize, const NUM_COLS: usize>(
    evals: &[Vec<E>; NUM_COLS],
    eq_view: &RoundEqView<'_, E>,
) -> [E::Scalar; D] {
    let packed_groups = evals[0].len() >> 1;
    let one = E::one().ext_to_montgomery();
    let mut t_acc = [E::zero(); D];
    let packed_split = eq_view.packed_split_for_groups(packed_groups);

    if let Some(split) = packed_split {
        let right_len = split.right_broadcast.len();
        for left_group in 0..split.left_packed.len() {
            let mut inner = [E::zero(); D];
            let group_base = left_group * right_len;
            for right_idx in 0..right_len {
                let eq_r = split.right_broadcast[right_idx];
                let group_idx = group_base + right_idx;
                let base = group_idx << 1;
                let mut v0s = [E::zero(); NUM_COLS];
                let mut diffs = [E::zero(); NUM_COLS];
                for c in 0..NUM_COLS {
                    let v0 = evals[c][base];
                    let v1 = evals[c][base + 1];
                    v0s[c] = v0;
                    diffs[c] = v1.lazy_add_xp(2).lazy_sub(v0).con_sub_xp(2);
                }
                let hats = compute_group_hats_ext::<E, G, D, NUM_COLS>(&v0s, &diffs, one);
                for k in 0..D {
                    inner[k] = inner[k].lazy_add(eq_r * hats[k]);
                }
            }
            let eq_l = split.left_packed[left_group];
            for k in 0..D {
                t_acc[k] = t_acc[k].lazy_add(eq_l * inner[k].reduce_fast());
            }
        }
    } else {
        for group_idx in 0..packed_groups {
            let base = group_idx << 1;
            let weight = eq_view.load_packed_weight(group_idx, packed_groups);
            let mut v0s = [E::zero(); NUM_COLS];
            let mut diffs = [E::zero(); NUM_COLS];
            for c in 0..NUM_COLS {
                let v0 = evals[c][base];
                let v1 = evals[c][base + 1];
                v0s[c] = v0;
                diffs[c] = v1.lazy_add_xp(2).lazy_sub(v0).con_sub_xp(2);
            }
            let hats = compute_group_hats_ext::<E, G, D, NUM_COLS>(&v0s, &diffs, one);
            for k in 0..D {
                t_acc[k] = t_acc[k].lazy_add(weight * hats[k]);
            }
            if (group_idx + 1) % 8192 == 0 {
                for k in 0..D {
                    t_acc[k] = t_acc[k].reduce_fast();
                }
            }
        }
    }

    let mut out = [E::Scalar::zero(); D];
    for k in 0..D {
        out[k] = E::sum_lanes_to_mont(t_acc[k]);
    }
    out
}

/// Compute the `D` hat values from a single packed block (scalar tail).
pub(crate) fn compute_round_t_single<E: SumcheckExtField, G: ZeroCheckGate<E, D, NUM_COLS>, const D: usize, const NUM_COLS: usize>(
    evals: &[E; NUM_COLS],
    eq_view: &RoundEqView<'_, E>,
    active_len: usize,
) -> [E::Scalar; D] {
    let pair_count = active_len >> 1;
    let mut scalar_weights = [E::Scalar::zero(); 8];
    for (pair_idx, slot) in scalar_weights.iter_mut().enumerate().take(pair_count) {
        *slot = eq_view.scalar_weight(pair_idx);
    }
    let packed_weights = E::pack_scalars(&scalar_weights[..pair_count]);
    let one = E::one().ext_to_montgomery();

    let mut v0s = [E::zero(); NUM_COLS];
    let mut diffs = [E::zero(); NUM_COLS];
    for c in 0..NUM_COLS {
        let (v0, _v1, diff) =
            SumcheckMamaBear::build_single_packed_pair_views_generic::<E>(evals[c], pair_count);
        v0s[c] = v0;
        diffs[c] = diff;
    }
    let hats = compute_group_hats_ext::<E, G, D, NUM_COLS>(&v0s, &diffs, one);

    let mut out = [E::Scalar::zero(); D];
    for k in 0..D {
        out[k] = E::sum_lanes_to_mont(packed_weights * hats[k]);
    }
    out
}

// ===========================================================================
// The generic ZeroCheck prover
// ===========================================================================

/// Resolve `ell0`, clamped so the packed path is always used (no scalar fallback here).
#[inline]
pub fn default_optimized_ell0_generic<const D: usize>(mu: usize) -> usize {
    let cap = if D <= 5 { 4 } else { 3 };
    (mu / 2).saturating_sub(1).min(cap)
}

#[inline]
pub(crate) fn resolve_ell0<const D: usize>(mu: usize, ell0: usize) -> usize {
    let cap = if D <= 5 { 4 } else { 3 };
    ell0.min(cap).min(mu.saturating_sub(3)).min(mu)
}

/// Generic degree-`D`, `NUM_COLS`-column ZeroCheck prover.
///
/// `evals` are the `NUM_COLS` packed base-field columns (Montgomery form, normal SIMD order).
/// `point` is the ZeroCheck challenge vector (normal form; converted internally to Mont).
/// `zero_check = true` skips the round-0 hat point 0 (gate is 0 on the boolean hypercube).
///
/// Returns `(challenges, col_claims, eq_claim)`: the per-round challenges (SIMD-round order,
/// Mont form), the `NUM_COLS` final column evaluations, and the final `eq` product
/// (all normal form). The 's `[E::Scalar; NUM_COLS + 1]` is `col_claims` followed by
/// `eq_claim`.
pub fn prove_zero_check_generic<E: SumcheckExtField, G: ZeroCheckGate<E, D, NUM_COLS>, const D: usize, const NUM_COLS: usize>(
    evals: [Vec<PBF>; NUM_COLS],
    point: &[E::Scalar],
    ell0: usize,
    zero_check: bool,
    transcript: &mut Transcript,
) -> (Vec<E::Scalar>, [E::Scalar; NUM_COLS], E::Scalar) {
    let mut throwaway = ZeroCheckTimings::default();
    prove_zero_check_generic_profiled::<E, G, D, NUM_COLS>(
        evals, point, ell0, zero_check, transcript, &mut throwaway,
    )
}

/// Profiled variant of `prove_zero_check_generic` (fills `timings`).
pub fn prove_zero_check_generic_profiled<E: SumcheckExtField, G: ZeroCheckGate<E, D, NUM_COLS>, const D: usize, const NUM_COLS: usize>(
    evals: [Vec<PBF>; NUM_COLS],
    point: &[E::Scalar],
    ell0: usize,
    zero_check: bool,
    transcript: &mut Transcript,
    timings: &mut ZeroCheckTimings,
) -> (Vec<E::Scalar>, [E::Scalar; NUM_COLS], E::Scalar) {
    use std::time::Instant;
    let t_total = Instant::now();
    let num_vars = point.len();
    let ell0 = resolve_ell0::<D>(num_vars, ell0);

    let point_mont: Vec<E::Scalar> = point.iter().copied().map(|v| v.to_montgomery()).collect();

    let t_eq = Instant::now();
    let eq_tables = SumcheckMamaBear::build_two_stage_eq_tables_generic::<E>(&point_mont);
    timings.eq_tables_us += t_eq.elapsed().as_micros();

    let consts = gate_consts_base::<D>();

    // a: precompute small-value tables.
    let t_pre = Instant::now();
    let small_value_tables = precompute_small_value_tables::<E, G, D, NUM_COLS>(
        &evals, &eq_tables, ell0, zero_check, &consts,
    );
    timings.precompute_us += t_pre.elapsed().as_micros();

    let ext_one = E::Scalar::one().to_montgomery();
    let mut finite_point_ext = [E::Scalar::zero(); MAX_U];
    for (p, slot) in finite_point_ext.iter_mut().enumerate() {
        *slot = E::Scalar::from(p as u32).to_montgomery();
    }

    let mut verifier_challenges = Vec::with_capacity(num_vars);
    let mut prefix_eq_eval = ext_one;
    let mut lagrange_weights = vec![ext_one];

    // b: small-value rounds.
    let t_small = Instant::now();
    for round in 0..ell0 {
        let t_hat = compute_t_from_precomputed::<E, D>(&small_value_tables[round], &lagrange_weights);
        let s_hat = compute_round_s_from_t::<E, D>(
            prefix_eq_eval,
            point_mont[round],
            t_hat,
            ext_one,
            &finite_point_ext,
        );
        append_hat_round_values::<E, D>(transcript, s_hat);

        let challenge = transcript.challenge_f::<E::Scalar>().to_montgomery();
        prefix_eq_eval *=
            SumcheckMamaBear::eq_linear_mont_generic::<E>(point_mont[round], challenge, ext_one);
        verifier_challenges.push(challenge);

        if round + 1 < ell0 {
            let basis = lagrange_basis_degree_d_generic::<E, D>(challenge, ext_one);
            lagrange_weights = update_small_value_weights::<E, D>(&lagrange_weights, &basis);
        }
    }
    timings.small_value_rounds_us += t_small.elapsed().as_micros();

    // packed rounds [ell0, num_vars - 3).
    let packed_round_start = ell0;
    let packed_round_end = num_vars.saturating_sub(3);

    let t_trans = Instant::now();
    let mut folded_tables =
        fold_base_tables_to_ext::<E, NUM_COLS>(&evals, &verifier_challenges[..ell0]);
    drop(evals);
    timings.transition_fold_us += t_trans.elapsed().as_micros();

    if packed_round_start < packed_round_end {
        let t_first = Instant::now();
        let first_t_hat = compute_round_t::<E, G, D, NUM_COLS>(
            &folded_tables,
            &SumcheckMamaBear::round_eq_view_generic(&eq_tables, packed_round_start),
        );
        timings.packed_fold_rounds_us += t_first.elapsed().as_micros();
        let s_hat = compute_round_s_from_t::<E, D>(
            prefix_eq_eval,
            point_mont[packed_round_start],
            first_t_hat,
            ext_one,
            &finite_point_ext,
        );
        append_hat_round_values::<E, D>(transcript, s_hat);

        let challenge = transcript.challenge_f::<E::Scalar>().to_montgomery();
        prefix_eq_eval *= SumcheckMamaBear::eq_linear_mont_generic::<E>(
            point_mont[packed_round_start],
            challenge,
            ext_one,
        );
        verifier_challenges.push(challenge);
        let mut prev_challenge = challenge;

        for round in (packed_round_start + 1)..packed_round_end {
            let eq_view = SumcheckMamaBear::round_eq_view_generic(&eq_tables, round);
            let t_pf = Instant::now();
            let t_hat = fold_and_compute_round_t::<E, G, D, NUM_COLS>(
                &mut folded_tables,
                &eq_view,
                prev_challenge,
            );
            timings.packed_fold_rounds_us += t_pf.elapsed().as_micros();
            let s_hat = compute_round_s_from_t::<E, D>(
                prefix_eq_eval,
                point_mont[round],
                t_hat,
                ext_one,
                &finite_point_ext,
            );
            append_hat_round_values::<E, D>(transcript, s_hat);

            let challenge = transcript.challenge_f::<E::Scalar>().to_montgomery();
            prefix_eq_eval *=
                SumcheckMamaBear::eq_linear_mont_generic::<E>(point_mont[round], challenge, ext_one);
            verifier_challenges.push(challenge);
            prev_challenge = challenge;
        }

        let t_pf_final = Instant::now();
        fold_ext_tables_in_place::<E, NUM_COLS>(&mut folded_tables, E::from_scalar(prev_challenge));
        timings.packed_fold_rounds_us += t_pf_final.elapsed().as_micros();
    }

    // scalar tail (last 3 rounds).
    let t_tail = Instant::now();
    let scalar_tail_start = (num_vars.saturating_sub(3)).max(ell0);
    let mut active_len = 1usize << (num_vars - scalar_tail_start);
    let mut packed_tail_tables: [E; NUM_COLS] = std::array::from_fn(|c| folded_tables[c][0]);

    for round in scalar_tail_start..num_vars {
        let eq_view = SumcheckMamaBear::round_eq_view_generic(&eq_tables, round);
        let t_hat = compute_round_t_single::<E, G, D, NUM_COLS>(&packed_tail_tables, &eq_view, active_len);
        let s_hat = compute_round_s_from_t::<E, D>(
            prefix_eq_eval,
            point_mont[round],
            t_hat,
            ext_one,
            &finite_point_ext,
        );
        append_hat_round_values::<E, D>(transcript, s_hat);

        let challenge = transcript.challenge_f::<E::Scalar>().to_montgomery();
        prefix_eq_eval *=
            SumcheckMamaBear::eq_linear_mont_generic::<E>(point_mont[round], challenge, ext_one);
        verifier_challenges.push(challenge);

        if active_len > 1 {
            fold_single_packed_in_place::<E, NUM_COLS>(
                &mut packed_tail_tables,
                E::from_scalar(challenge),
                active_len,
            );
            active_len >>= 1;
        }
    }

    let mut col_claims = [E::Scalar::zero(); NUM_COLS];
    for c in 0..NUM_COLS {
        col_claims[c] = E::unpack_to_scalars(packed_tail_tables[c])[0].from_montgomery();
    }
    let eq_claim = prefix_eq_eval.from_montgomery();
    timings.scalar_tail_us += t_tail.elapsed().as_micros();
    timings.total_us += t_total.elapsed().as_micros();
    (verifier_challenges, col_claims, eq_claim)
}

// ===========================================================================
// The generic ZeroCheck verifier
// ===========================================================================

/// Verify a generic degree-`D` ZeroCheck proof produced by `prove_zero_check_generic`.
///
/// `y` is the claimed sum (normal form; `0` for a genuine ZeroCheck). `r_zero` is the
/// prover's `point` (SIMD-round order). Returns `(challenges, final_claim)`, the latter
/// being `s(α)` of the last round (caller multiplies by the column claims / eq to finish
/// the ZeroCheck — see the add/mul verifier).
pub fn verify_zero_check_generic<E: SumcheckExtField, const D: usize>(
    mut y: E::Scalar,
    r_zero: &[E::Scalar],
    var_num: usize,
    transcript: &mut Transcript,
    proof: &mut util::fiat_shamir::Proof,
) -> (Vec<E::Scalar>, E::Scalar) {
    assert_eq!(r_zero.len(), var_num, "verify_zero_check_generic: r_zero length mismatch");
    let one = E::Scalar::one().to_montgomery();
    let mut finite_point = [E::Scalar::zero(); MAX_U];
    for (p, slot) in finite_point.iter_mut().enumerate() {
        *slot = E::Scalar::from(p as u32).to_montgomery();
    }

    let mut challenges = Vec::with_capacity(var_num);
    let mut prefix_eq = one;

    for round in 0..var_num {
        // Read the D hat values s(0), s(2), ..., s(D-1), s(∞).
        let mut s_sent = [E::Scalar::zero(); D];
        for slot in s_sent.iter_mut() {
            let v: E::Scalar = proof.get_next_and_step();
            transcript.append_f(v);
            *slot = v.to_montgomery();
        }
        let ym = y.to_montgomery();
        let w_i = r_zero[round].to_montgomery();

        let alpha: E::Scalar = transcript.challenge_f::<E::Scalar>().to_montgomery();
        challenges.push(alpha.from_montgomery());

        // s(1) = y_prev - s(0).
        let s_1 = ym.lazy_add_xp(2).lazy_sub(s_sent[0]).con_sub_xp(2).reduce();

        // Reconstruct s at the full grid {0, 1, ..., D-1, ∞}.
        // slot j (j < D) holds s(j); the last holds s(∞).
        let mut s_full = [E::Scalar::zero(); MAX_U];
        s_full[0] = s_sent[0]; // s(0)
        s_full[1] = s_1; // s(1)
        for slot in 2..D {
            s_full[slot] = s_sent[slot - 1]; // s(2..D-1)
        }
        s_full[D] = s_sent[D - 1]; // s(∞)

        // c at the grid: c(p) = prefix_eq · eq(w_i, p); c(∞) = prefix_eq · (2 w_i - 1).
        let mut c = [E::Scalar::zero(); MAX_U];
        c[0] = (prefix_eq * one.lazy_add_xp(2).lazy_sub(w_i)).reduce(); // c(0)
        c[1] = (prefix_eq * w_i).reduce(); // c(1)
        for slot in 2..D {
            let eq_p = SumcheckMamaBear::eq_linear_mont_generic::<E>(w_i, finite_point[slot], one);
            c[slot] = (prefix_eq * eq_p).reduce();
        }
        let eq_inf = w_i.lazy_add(w_i).lazy_add_xp(2).lazy_sub(one).con_sub_xp(2);
        c[D] = (prefix_eq * eq_inf).reduce(); // c(∞)

        // t(p) = s(p) / c(p) for each grid point.
        let mut t = [E::Scalar::zero(); MAX_U];
        for slot in 0..=D {
            let c_inv = c[slot]
                .from_montgomery()
                .inv()
                .expect("c value invertible")
                .to_montgomery();
            t[slot] = (s_full[slot] * c_inv).reduce();
        }

        // t(α) = Σ_{j<D} t(j) · L_j(α) + t(∞) · L_∞(α).
        let basis = lagrange_basis_degree_d_generic::<E, D>(alpha, one);
        let mut t_alpha = E::Scalar::zero();
        for slot in 0..D {
            t_alpha = t_alpha.lazy_add(t[slot] * basis[slot]);
        }
        t_alpha = t_alpha.lazy_add(t[D] * basis[D]).reduce();

        // s(α) = c(α) · t(α).
        let eq_alpha = SumcheckMamaBear::eq_linear_mont_generic::<E>(w_i, alpha, one);
        let c_alpha = (prefix_eq * eq_alpha).reduce();
        let s_alpha = (c_alpha * t_alpha).reduce();
        y = s_alpha.from_montgomery();

        prefix_eq = (prefix_eq * eq_alpha).reduce();
    }

    (challenges, y)
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use arithmetic::field::mamabear::{
        MamaBearScalar as SBF, MamaBearScalarExt3 as SEF3,
        PackedMamaBearAVX512 as PBF, PackedMamaBearAVX512Ext3 as PEF3,
    };
    use rand::rngs::SmallRng;
    use rand::SeedableRng;
    use util::fiat_shamir::Transcript;

    fn pack_scalar_evals(evals: &[SBF]) -> Vec<PBF> {
        assert_eq!(evals.len() % 8, 0);
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

    fn random_cols(nv: usize, rng: &mut SmallRng) -> [Vec<PBF>; 4] {
        let domain = 1usize << nv;
        let mut scalar: [Vec<SBF>; 4] = std::array::from_fn(|_| Vec::with_capacity(domain));
        for col in scalar.iter_mut() {
            for _ in 0..domain {
                col.push(SBF::random(&mut *rng).to_montgomery());
            }
        }
        scalar.each_ref().map(|poly| pack_scalar_evals(poly))
    }

    #[test]
    fn zerocheck_generic_vs_legacy_ext3() {
        let mut rng = SmallRng::seed_from_u64(0xB17E_1DE3);
        for nv in 16usize..=22 {
            let point: Vec<SEF3> = (0..nv).map(|_| SEF3::random(&mut rng)).collect();
            let cols = random_cols(nv, &mut rng);
            for ell0 in 1..=4usize.min(nv.saturating_sub(3)) {
                let mut t_legacy = Transcript::new();
                let legacy = SumcheckMamaBear::prove_add_mul_ell0_ext3(
                    cols.clone(),
                    &point,
                    ell0,
                    &mut t_legacy,
                );
                let mut t_gen = Transcript::new();
                let generic = prove_zero_check_generic::<PEF3, AddMulD3, 3, 4>(
                    cols.clone(),
                    &point,
                    ell0,
                    true,
                    &mut t_gen,
                );
                assert_eq!(
                    t_legacy.proof.bytes, t_gen.proof.bytes,
                    "ext3 proof bytes differ at nv={nv}, ell0={ell0}"
                );
                assert_eq!(legacy.0, generic.0, "ext3 challenges differ nv={nv} ell0={ell0}");
                for c in 0..4 {
                    assert_eq!(legacy.1[c], generic.1[c], "ext3 col {c} claim differs nv={nv} ell0={ell0}");
                }
                assert_eq!(legacy.1[4], generic.2, "ext3 eq claim differs nv={nv} ell0={ell0}");
            }
        }
    }

    // -----------------------------------------------------------------------
    // Exercise the const-generic axes beyond D=3/NUM_COLS=4 with a synthetic gate.
    // -----------------------------------------------------------------------

    /// Degree-3, 8-column smoke gate:
    /// h = c0·c1·c2 + c3 + c4 + c5 + c6 + c7. Leading coeff = diff0·diff1·diff2.
    struct EightColSmokeGate;
    impl<E: SumcheckExtField> ZeroCheckGate<E, 3, 8> for EightColSmokeGate {
        fn h_packed_base(i: [PBF; 8], _o: PBF) -> PBF {
            (i[0] * i[1] * i[2])
                .lazy_add(i[3]).lazy_add(i[4]).lazy_add(i[5]).lazy_add(i[6]).lazy_add(i[7])
                .reduce_fast()
        }
        fn h_packed_ext(i: [E; 8], _o: E) -> E {
            (i[0] * i[1] * i[2])
                .lazy_add(i[3]).lazy_add(i[4]).lazy_add(i[5]).lazy_add(i[6]).lazy_add(i[7])
                .reduce_fast()
        }
        fn h_scalar_ext(i: [E::Scalar; 8], _o: E::Scalar) -> E::Scalar {
            (i[0] * i[1] * i[2])
                .lazy_add(i[3]).lazy_add(i[4]).lazy_add(i[5]).lazy_add(i[6]).lazy_add(i[7])
                .reduce()
        }
        fn h_leading_packed_base(d: [PBF; 8], _o: PBF) -> PBF { d[0] * d[1] * d[2] }
        fn h_leading_packed_ext(d: [E; 8], _o: E) -> E { d[0] * d[1] * d[2] }
        fn h_leading_scalar_ext(d: [E::Scalar; 8], _o: E::Scalar) -> E::Scalar { d[0] * d[1] * d[2] }
    }

    fn random_cols_n<const NC: usize>(nv: usize, rng: &mut SmallRng) -> [Vec<PBF>; NC] {
        let domain = 1usize << nv;
        let mut scalar: [Vec<SBF>; NC] = std::array::from_fn(|_| Vec::with_capacity(domain));
        for col in scalar.iter_mut() {
            for _ in 0..domain {
                col.push(SBF::random(&mut *rng).to_montgomery());
            }
        }
        scalar.each_ref().map(|poly| pack_scalar_evals(poly))
    }

    /// Checks that `prove_zero_check_generic::<E, _, 3, 8>` monomorphizes and the
    /// prover/verifier round structure (D=3 hats/round, 8 columns) is self-consistent.
    #[test]
    fn zerocheck_generic_d3_8col_ext3() {
        let mut rng = SmallRng::seed_from_u64(0x803);
        for nv in 14usize..=18 {
            let point: Vec<SEF3> = (0..nv).map(|_| SEF3::random(&mut rng)).collect();
            let cols = random_cols_n::<8>(nv, &mut rng);
            let mut tp = Transcript::new();
            let (pc, _cc, _eq) = prove_zero_check_generic::<PEF3, EightColSmokeGate, 3, 8>(
                cols, &point, 2, true, &mut tp,
            );
            assert_eq!(tp.proof.bytes.len(), nv * 3 * SEF3::SIZE, "D3 must carry 3 hats/round nv={nv}");
            let mut proof = tp.proof.clone();
            let mut tv = Transcript::new();
            let (vc, _y) =
                verify_zero_check_generic::<PEF3, 3>(SEF3::zero(), &point, nv, &mut tv, &mut proof);
            let pc_norm: Vec<SEF3> = pc.iter().map(|c| c.from_montgomery()).collect();
            assert_eq!(pc_norm, vc, "D3 8-col prover/verifier challenge mismatch nv={nv}");
        }
    }

}

