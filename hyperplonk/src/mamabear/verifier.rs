use arithmetic::{
    field::{
        mamabear::{MamaBearScalar, P},
        Field,
    },
};
use poly_commit::{
    deepfold::MerkleRoot,
    deepfold_mamabear::{self, DeepFoldMamaBearParam, DeepFoldMamaBearVerifier},
    CommitmentSerde,
};
use util::fiat_shamir::{Proof, Transcript};

use crate::{
    prodcheck_mamabear_perwire::ProdEqCheckMamaBearPerWire,
    prover_mamabear::{simd_to_natural_point, MamaBearExtConfig, VerifierKeyMamaBear},
    sumcheck_mamabear::SumcheckMamaBear,
};

const NUM_WIRES: usize = 3;
const ID_SHIFT_1: u64 = 1 << 29;
const ID_SHIFT_2: u64 = 1 << 30;

pub struct VerifierMamaBear<F: MamaBearExtConfig> {
    pub verifier_key: VerifierKeyMamaBear<F>,
}

/// Per-substage wall-clock breakdown for `VerifierMamaBear::verify_profiled`.
/// All values are accumulated microseconds. `pcs_verify_breakdown` is the
/// DeepFold verify sub-breakdown for the `pcs_verify_us` bucket.
#[derive(Clone, Debug, Default)]
pub struct VerifyTimings {
    pub total_us: u128,
    /// Commit deserialize + `transcript.append_u8_slice` + `DeepFoldMamaBearVerifier::new`.
    pub setup_us: u128,
    /// `r_zero` challenge draw (nv challenges for ZeroCheck).
    pub sc_chal_us: u128,
    /// `SumcheckMamaBear::verify_ell0` — 4·nv scalar Fermat inversions live inside.
    pub zero_chk_us: u128,
    /// main-gate identity check: `eval_eq_mont(r_zero, sc_pt)` + gate composition vs `zero_y`.
    pub c1_us: u128,
    /// ProductCheck setup challenges `prod_r0` / `prod_r1`.
    pub prod_chal_us: u128,
    /// `ProdEqCheckMamaBearPerWire::verify` sumcheck verification.
    pub prod_chk_us: u128,
    /// ProductCheck reconstruction check: per-wire final-claim check (uses `eval_identical_mont`).
    pub c2_us: u128,
    /// `rho` challenge + Horner `initial_y` for final-reduce sumcheck.
    pub rho_init_y_us: u128,
    /// `verify_final_reduce_sumcheck` (degree-2 Newton-form interpolation for nv rounds × 2 trees).
    pub final_rsc_us: u128,
    /// final-reduce tree reconstruction check: two `eval_eq_mont` calls + composition check for both trees.
    pub c4_us: u128,
    /// `DeepFoldMamaBearVerifier::verify` total; substages in `pcs_verify_breakdown`.
    pub pcs_verify_us: u128,
    pub pcs_verify_breakdown: deepfold_mamabear::VerifyTimings,
}

// =========================================================================================
// FULL PROTOCOL CONSISTENCY — main-gate identity, ProductCheck reconstruction, final-reduce sumcheck, final-reduce tree reconstruction all enforced
// =========================================================================================
//
// The verifier now enforces all four algebraic consistency equations. Each check uses the
// variable-permutation fixes and Lagrange reconstruction derived in the
// documentation block at the bottom of this file. In summary:
//
//   * Part 2 fix:  SumcheckMamaBear::verify_ell0 takes the ZC challenge
//                  vector `r_zero` and reconstructs t_i(α) via the t-factored
//                  form t(α) = t_inf · α(α-1)(α-2) + q(α), where q is the
//                  unique degree-≤2 polynomial through (t(0), t(1), t(2))
//                  and t_inf is the prover's leading-coefficient hat value.
//                  The factor α(α-1)(α-2) vanishes at the interpolation
//                  nodes, so this correctly accounts for the degree-3
//                  structure of t. All field operations use standard
//                  Add/Sub/Mul — using a 4-point Lagrange with Newton-derived
//                  t(3) produced subtle Mont-arithmetic-bias discrepancies
//                  (the 4-pt and 3-pt+leading formulas should be algebraically
//                  equivalent but differed numerically; 3-pt+leading is the
//                  robust choice).
//
//   * Part 3 fix:  prove_final_reduce_sumcheck permutes `sumcheck_point` and
//                  `prod_point` to natural order via `simd_to_natural_point`
//                  before building eq tables. The initial sum then equals
//                     tree0_MLE(natural_point_zc) = Horner(claim_s, claim_w0, …)
//                  matching the verifier's claim-based initial_y. Because σ^{-1}
//                  is applied identically to both eq-point sides,
//                  eq(natural_zc, natural_fr) = eq(sumcheck_point, point_fr)
//                  in the verifier's final-reduce tree reconstruction check.
//
//   * Part 4 fix:  the prover permutes prod_point / point_fr through
//                  `simd_to_natural_point` before passing them to
//                  `eval_multilinear_base_packed` (mid-stage + final openings)
//                  AND to the DeepFold PCS open. The verifier mirrors this by
//                  calling PCS verify on the same natural point.
//
//   * Bias fix:    residual mont_mul(0, 0) = P bias from packed-SIMD
//                  accumulation is neutralised by explicit `.reduce()` before
//                  every `assert_eq!` / `from_mont()` comparison.
//
// With these in place, all existing tests pass with main-gate identity..final-reduce tree reconstruction asserting
// actively. A malicious prover corrupting any of:
//   - zero-check claim openings at the reduced point (main-gate identity)
//   - witness / permutation MLE openings at prod_point (ProductCheck reconstruction)
//   - final-reduce per-round polynomials (final-reduce sumcheck)
//   - final polynomial openings at point_fr (final-reduce tree reconstruction)
// will now cause the verifier to fail.
//
// Note: enabling main-gate identity also surfaced a latent bug in the `run_mamabear_snark`
// test helper, which constructed witnesses using `MamaBearScalar::*` (=
// mont_mul, producing x·y/R) rather than raw modular multiplication. The test
// was vacuously accepted when main-gate identity was disabled; it now uses explicit u128-based
// modular mul to build a valid satisfying witness.
//
// =========================================================================================

/// Round-by-round verifier for the final-reduce sumcheck.
///
/// The final-reduce's round polynomial `s_t(X) = tree_t(X) · eq(natural_pt_*, X)`
/// has degree **2** (both factors are linear), so the prover sends 3 hat values
/// per tree {s_t(0), s_t(1), s_t(2)} and the verifier:
///   1. asserts the sum-check invariant `s_t(0) + s_t(1) == y_prev[t]`
///   2. Lagrange-interpolates s_t at the round challenge r via Newton form
///      (degree 2): `s(r) = s(0) + r·(s(1)-s(0)) + r(r-1)/2·(s(2) - 2s(1) + s(0))`
///   3. returns `y_final` so the caller can run the final composition check final-reduce tree reconstruction.
fn verify_final_reduce_sumcheck<F: MamaBearExtConfig>(
    initial_y: [F; 2],
    nv: usize,
    transcript: &mut Transcript,
    proof: &mut Proof,
) -> (Vec<F>, [F; 2]) {
    let mut point = Vec::with_capacity(nv);
    let one_m = F::one().to_mont();
    let inv2_m = F::from(MamaBearScalar((P + 1) / 2)).to_mont();

    // Track y in Mont form across rounds (starts from initial_y in normal form).
    let mut y_m: [F; 2] = [initial_y[0].to_mont(), initial_y[1].to_mont()];

    for round in 0..nv {
        // Read 3 hat values per tree. Prover sends them in normal form.
        let mut s_m: [[F; 3]; 2] = [[F::zero(); 3], [F::zero(); 3]];
        for t in 0..2 {
            for h in 0..3 {
                let v: F = proof.get_next_and_step();
                transcript.append_f(v);
                s_m[t][h] = v.to_mont();
            }
        }

        // Round challenge α (stored in normal form in `point` so the caller
        // can plug it directly into eq / MLE routines that expect normal form).
        let alpha_m: F = transcript.challenge_f::<F>().to_mont();
        point.push(alpha_m.from_mont());

        // Precompute α(α-1) and (α-1) shared across the two trees' interp.
        let alpha_minus_1 = alpha_m.lazy_add_xp(2).lazy_sub(one_m).con_sub_xp(2); // [0, 2P)
        let alpha_times_alpha_minus_1 = (alpha_m * alpha_minus_1).reduce();       // [0, P)
        let half_alpha_alpha_m1 = (alpha_times_alpha_minus_1 * inv2_m).reduce();  // α(α-1)/2 ∈ [0, P)

        for t in 0..2 {
            let s0 = s_m[t][0]; // [0, 2P)
            let s1 = s_m[t][1];
            let s2 = s_m[t][2];

            // ── (1) Sum-check invariant at this round: s(0) + s(1) == y_prev.
            // Compare canonical normal-form values.
            let lhs = s0.lazy_add(s1).reduce().from_mont();
            let rhs = y_m[t].reduce().from_mont();
            assert_eq!(
                lhs, rhs,
                "final-reduce sumcheck round {} tree {} invariant failed",
                round, t
            );

            // ── (2) Newton-form interpolation at α over {0, 1, 2} (degree 2).
            //   s(α) = s(0) + α·(s(1)-s(0)) + α(α-1)/2 · (s(2) - 2s(1) + s(0))
            let d1 = s1.lazy_add_xp(2).lazy_sub(s0).con_sub_xp(2); // s(1) - s(0) ∈ [0, 2P)
            let d2 = s2
                .lazy_add_xp(4)
                .lazy_sub(s1)
                .lazy_sub(s1)
                .lazy_add(s0)
                .reduce(); // s(2) - 2s(1) + s(0) ∈ [0, P)

            let lin = (alpha_m * d1).reduce();                   // α·d1 ∈ [0, P)
            let quad = (half_alpha_alpha_m1 * d2).reduce();       // α(α-1)/2·d2 ∈ [0, P)
            y_m[t] = s0.lazy_add(lin).lazy_add(quad).reduce();    // ∈ [0, P)
        }
    }

    // Return y in normal form for the caller's convenience.
    (point, [y_m[0].from_mont(), y_m[1].from_mont()])
}

fn eval_eq_mont<F: MamaBearExtConfig>(r: &[F], point: &[F]) -> F {
    let one = F::one().to_mont();
    let mut result = one;
    for idx in 0..r.len() {
        result = (result
            * SumcheckMamaBear::eq_linear_mont_generic::<F::Packed>(r[idx], point[idx], one)
                .reduce())
            .reduce();
    }
    result
}

fn eval_identical_mont<F: MamaBearExtConfig>(point: &[F], offset: MamaBearScalar) -> F {
    let mut res = F::from_base_mont(offset.to_montgomery());
    let mut coeff = MamaBearScalar::one().to_montgomery();
    for (idx, value) in point.iter().enumerate() {
        if idx == 0 {
            res = res + *value;
        } else {
            coeff = coeff + coeff;
            res = res + value.mul_base_elem(coeff);
        }
    }
    res.reduce()
}

impl<F: MamaBearExtConfig> VerifierMamaBear<F> {
    pub fn verify(&self, pp: &DeepFoldMamaBearParam, nv: usize, proof: Proof) -> bool {
        let mut t = VerifyTimings::default();
        self.verify_inner(pp, nv, proof, &mut t, false, false)
    }

    /// Profiling variant of `verify`. Accumulates per-substage wall-clock time
    /// into `timings`; `Instant::now()` overhead means this should only be used
    /// for profiling, not in production verification.
    pub fn verify_profiled(
        &self,
        pp: &DeepFoldMamaBearParam,
        nv: usize,
        proof: Proof,
        timings: &mut VerifyTimings,
    ) -> bool {
        self.verify_inner(pp, nv, proof, timings, true, false)
    }

    /// Shared inner implementation. `record` toggles profiling; `par_pcs`
    /// dispatches to `DeepFoldMamaBearVerifier::verify_par{,_profiled}`
    /// instead of the serial path at the PCS step. The par entry points live
    /// in `hyperplonk::verifier_mamabear_par` to keep the per-file layout
    /// mirroring the serial/par split used by sumcheck / prover / prodcheck.
    pub(crate) fn verify_inner(
        &self,
        pp: &DeepFoldMamaBearParam,
        nv: usize,
        mut proof: Proof,
        timings: &mut VerifyTimings,
        record: bool,
        par_pcs: bool,
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

        let setup_t0 = now!();
        let mut transcript = Transcript::new();
        let commit = MerkleRoot::deserialize_from(&mut proof, nv, NUM_WIRES);
        let mut buffer = vec![0u8; MerkleRoot::size(nv, NUM_WIRES)];
        commit.serialize_into(&mut buffer);
        transcript.append_u8_slice(&buffer, buffer.len());
        let witness_pc = DeepFoldMamaBearVerifier::<F>::new(pp, commit, NUM_WIRES);
        tick!(setup_t0, setup_us);

        // ── ZeroCheck challenges (r_zero) — kept to drive verify_ell0's
        // t-factored reconstruction and to evaluate eq(r_zero, sumcheck_point)
        // for the main-gate identity gate check.
        let sc_chal_t0 = now!();
        let r_zero: Vec<F> = (0..nv).map(|_| transcript.challenge_f::<F>()).collect();
        tick!(sc_chal_t0, sc_chal_us);

        let zc_t0 = now!();
        let (sumcheck_point, zero_y) = SumcheckMamaBear::verify_ell0::<F::Packed>(
            F::zero(),
            &r_zero,
            nv,
            &mut transcript,
            &mut proof,
        );
        tick!(zc_t0, zero_chk_us);
        let sumcheck_point_mont = sumcheck_point
            .iter()
            .copied()
            .map(F::to_mont)
            .collect::<Vec<_>>();
        let r_zero_mont = r_zero.iter().copied().map(F::to_mont).collect::<Vec<_>>();

        let claim_s: F = proof.get_next_and_step();
        transcript.append_f(claim_s);
        let claim_w0: F = proof.get_next_and_step();
        transcript.append_f(claim_w0);
        let claim_w1: F = proof.get_next_and_step();
        transcript.append_f(claim_w1);
        let claim_w2: F = proof.get_next_and_step();
        transcript.append_f(claim_w2);
        let claim_s_m = claim_s.to_mont();
        let claim_w0_m = claim_w0.to_mont();
        let claim_w1_m = claim_w1.to_mont();
        let claim_w2_m = claim_w2.to_mont();

        // ── main-gate identity: ZeroCheck final-claim check.
        //
        // Derivation: the prover's ZeroCheck sumcheck operates on
        //   f(x) = h(S(x), L(x), R(x), O(x)) · eq_prover(r_zero, x)
        // where eq_prover(r_zero, x) = ∏_i eq(r_zero[i], x_{σ(i)}) is the
        // eq factor AS IMPLEMENTED — each round i multiplies by
        // eq(r_zero_natural[i], ·) onto the variable being folded that round
        // (which is x_{σ(i)} under the SIMD Theorem-2 permutation). This
        // still has the ZeroCheck property (f = 0 on hypercube ⟺ h = 0 on
        // hypercube since eq is nonzero), so the protocol is sound.
        //
        // After the full sumcheck reduces all variables, the final y equals
        //   y_final = (∏_i eq(r_zero[i], r_i)) · h(claims at reduced point)
        //           = eval_eq(r_zero, sumcheck_point) · gate(claim_s, ...)
        // where sumcheck_point[i] = r_i (round order) and the eq is taken
        // pointwise (NOT through simd_to_natural — because the prover's eq
        // factor is already written in that non-standard permuted form).
        //
        // verify_ell0 faithfully reproduces this by tracking prefix_eq =
        // ∏_{j<i} eq(r_zero[j], r_j) and factoring s_i(X) = c_i(X) · t_i(X)
        // with c_i(X) = prefix_eq · eq(r_zero[i], X). At the last round,
        // t_{nv-1}(r_{nv-1}) = h(claims) and zero_y is the product of both
        // factors.
        let c1_t0 = now!();
        {
            let eq_val = eval_eq_mont::<F>(&r_zero_mont, &sumcheck_point_mont);
            // gate_h = L + R + S·(L·R − L − R) + O.
            let l_plus_r = claim_w0_m + claim_w1_m;
            let lr = (claim_w0_m * claim_w1_m).reduce();
            let inner = (lr.lazy_add_xp(2).lazy_sub(l_plus_r).con_sub_xp(2)).reduce();
            let term = (claim_s_m * inner).reduce();
            let gate_val = (l_plus_r + term + claim_w2_m).reduce();
            let expected = (eq_val * gate_val).reduce();
            assert_eq!(
                expected.from_mont(),
                zero_y,
                "C1: ZeroCheck final-claim mismatch"
            );
        }
        tick!(c1_t0, c1_us);

        let prod_chal_t0 = now!();
        let prod_r0: F = transcript.challenge_f();
        let prod_r1: F = transcript.challenge_f();
        tick!(prod_chal_t0, prod_chal_us);

        let prod_chk_t0 = now!();
        let (prod_point, prod_y, lambda) =
            ProdEqCheckMamaBearPerWire::verify::<F::Packed>(nv, &mut transcript, &mut proof);
        tick!(prod_chk_t0, prod_chk_us);
        let prod_point_mont = prod_point
            .iter()
            .copied()
            .map(F::to_mont)
            .collect::<Vec<_>>();
        // The per-wire ProductCheck sumcheck output is in SIMD round order;
        // prover-side mid-stage openings were evaluated at the natural-order
        // reduced point. Compute the corresponding natural point here so id_g
        // / σ_g evaluations match the prover's witness_eval[g] / perm_eval[g].
        let prod_point_natural_mont = simd_to_natural_point(&prod_point_mont);

        let witness_eval = [0; NUM_WIRES].map(|_| {
            let value: F = proof.get_next_and_step();
            transcript.append_f(value);
            value
        });
        let perm_eval = [0; NUM_WIRES].map(|_| {
            let value: F = proof.get_next_and_step();
            transcript.append_f(value);
            value
        });
        let witness_eval_m = witness_eval.map(F::to_mont);
        let perm_eval_m = perm_eval.map(F::to_mont);

        // ── ProductCheck reconstruction: Per-wire ProductCheck final-claim check.
        //   prod_y[0] ?= Σ_g λ^g · (γ + witness_eval[g] + β · id_g(nat_pc))
        //   prod_y[1] ?= Σ_g λ^g · (γ + witness_eval[g] + β · σ_g(nat_pc) = perm_eval[g])
        // γ = prod_r0, β = prod_r1 (ProductCheck setup challenges).
        let c2_t0 = now!();
        {
            let prod_r0_m = prod_r0.to_mont();
            let prod_r1_m = prod_r1.to_mont();
            let lambda_m = lambda.to_mont();
            let lambda2_m = (lambda_m * lambda_m).reduce();
            let lambda_w: [F; 3] = [F::one().to_mont(), lambda_m, lambda2_m];

            let offsets = [
                MamaBearScalar::zero(),
                MamaBearScalar::from(ID_SHIFT_1),
                MamaBearScalar::from(ID_SHIFT_2),
            ];
            let id_evals: [F; 3] =
                offsets.map(|off| eval_identical_mont::<F>(&prod_point_natural_mont, off));

            for tree in 0..2 {
                let mut sum = F::zero();
                for g in 0..NUM_WIRES {
                    let third = if tree == 0 { id_evals[g] } else { perm_eval_m[g] };
                    let inner =
                        ((prod_r0_m + witness_eval_m[g]) + (prod_r1_m * third).reduce()).reduce();
                    sum = sum.lazy_add((lambda_w[g] * inner).reduce()).reduce();
                }
                // Reduce to canonical form to neutralize any residual mont_mul(0,0)=P
                // bias from the prover's packed-SIMD accumulation on inactive lanes.
                assert_eq!(
                    sum.reduce().from_mont(),
                    prod_y[tree],
                    "C2: ProductCheck tree {} final-claim mismatch",
                    tree
                );
            }
        }
        tick!(c2_t0, c2_us);

        // Setup the final-reduce sumcheck. Initial_y[0] for tree0 must equal
        // tree0_MLE at the point where the prover built the eq table — which,
        // after the Part 3 prover-side permutation, is natural_point_zc.
        // Horner over claim_s, claim_w0, claim_w1, claim_w2 gives exactly this.
        let rho_t0 = now!();
        let rho: F = transcript.challenge_f();
        let rho_m = rho.to_mont();
        let initial_y = [
            (claim_s_m
                + (rho_m
                    * (claim_w0_m + (rho_m * (claim_w1_m + (rho_m * claim_w2_m).reduce())).reduce()))
                .reduce())
            .reduce()
            .from_mont(),
            (perm_eval_m[0]
                + (rho_m
                    * (perm_eval_m[1]
                        + (rho_m
                            * (perm_eval_m[2]
                                + (rho_m
                                    * (witness_eval_m[0]
                                        + (rho_m
                                            * (witness_eval_m[1]
                                                + (rho_m * witness_eval_m[2]).reduce()))
                                        .reduce()))
                                .reduce()))
                        .reduce()))
                .reduce())
            .reduce()
            .from_mont(),
        ];
        tick!(rho_t0, rho_init_y_us);

        // The final-reduce sumcheck's per-round invariants are enforced inside
        // verify_final_reduce_sumcheck.
        let final_rsc_t0 = now!();
        let (point_fr, y_final) =
            verify_final_reduce_sumcheck::<F>(initial_y, nv, &mut transcript, &mut proof);
        tick!(final_rsc_t0, final_rsc_us);
        let point_fr_mont = point_fr.iter().copied().map(F::to_mont).collect::<Vec<_>>();

        let final_selector: F = proof.get_next_and_step();
        transcript.append_f(final_selector);
        let final_perm = [0; NUM_WIRES].map(|_| {
            let value: F = proof.get_next_and_step();
            transcript.append_f(value);
            value
        });
        let final_witness = [0; NUM_WIRES].map(|_| {
            let value: F = proof.get_next_and_step();
            transcript.append_f(value);
            value
        });
        let final_selector_m = final_selector.to_mont();
        let final_perm_m = final_perm.map(F::to_mont);
        let final_witness_m = final_witness.map(F::to_mont);

        // ── final-reduce tree reconstruction: Final-reduce sumcheck composition check.
        //   y_final[0] ?= Horner(selector, w0, w1, w2; ρ) · eq(natural_pt_zc, natural_pt_fr)
        //   y_final[1] ?= Horner(p0, p1, p2, w0, w1, w2; ρ) · eq(natural_pt_pc, natural_pt_fr)
        //
        // Since the PROVER applied simd_to_natural_point to BOTH its eq-table
        // argument (sumcheck_point / prod_point) AND implicitly to point_fr via
        // how the SIMD sumcheck consumes them, the σ^{-1} rotations cancel and
        // eq(natural_pt_*, natural_pt_fr) = eq(sumcheck_point, point_fr). See
        // Part 3 of the top-of-file documentation block.
        let c4_t0 = now!();
        {
            let eq_zc = eval_eq_mont::<F>(&sumcheck_point_mont, &point_fr_mont);
            let eq_pc = eval_eq_mont::<F>(&prod_point_mont, &point_fr_mont);

            // Tree 0: selector + ρ·(w0 + ρ·(w1 + ρ·w2))
            let w_inner = (final_witness_m[0]
                + (rho_m * (final_witness_m[1] + (rho_m * final_witness_m[2]).reduce())).reduce())
            .reduce();
            let tree0_eval = (final_selector_m + (rho_m * w_inner).reduce()).reduce();
            let rhs0 = (tree0_eval * eq_zc).reduce();
            assert_eq!(
                rhs0.from_mont(),
                y_final[0],
                "C4: final-reduce tree 0 composition mismatch"
            );

            // Tree 1: perm[0] + ρ·(perm[1] + ρ·(perm[2] + ρ·(w0 + ρ·(w1 + ρ·w2))))
            let perm_inner = (final_perm_m[0]
                + (rho_m
                    * (final_perm_m[1]
                        + (rho_m * (final_perm_m[2] + (rho_m * w_inner).reduce())).reduce()))
                .reduce())
            .reduce();
            let rhs1 = (perm_inner * eq_pc).reduce();
            assert_eq!(
                rhs1.from_mont(),
                y_final[1],
                "C4: final-reduce tree 1 composition mismatch"
            );
        }
        tick!(c4_t0, c4_us);

        // PCS verification: the prover permuted point_fr via simd_to_natural_point
        // before opening, so the verifier must do the same to pass a matching point
        // to the DeepFold commitment verifier. The evals we read from the proof
        // above are at this natural point.
        let point_fr_natural: Vec<F> = simd_to_natural_point(&point_fr);

        let pcs_t0 = now!();
        let verifiers = vec![&self.verifier_key.commitment, &witness_pc];
        let evals = vec![
            vec![final_selector_m, final_perm_m[0], final_perm_m[1], final_perm_m[2]],
            vec![final_witness_m[0], final_witness_m[1], final_witness_m[2]],
        ];
        let pcs_ok = match (par_pcs, record) {
            (false, true) => DeepFoldMamaBearVerifier::verify_profiled(
                pp,
                verifiers,
                point_fr_natural,
                evals,
                &mut transcript,
                &mut proof,
                &mut timings.pcs_verify_breakdown,
            ),
            (false, false) => DeepFoldMamaBearVerifier::verify(
                pp,
                verifiers,
                point_fr_natural,
                evals,
                &mut transcript,
                &mut proof,
            ),
            (true, true) => DeepFoldMamaBearVerifier::verify_par_profiled(
                pp,
                verifiers,
                point_fr_natural,
                evals,
                &mut transcript,
                &mut proof,
                &mut timings.pcs_verify_breakdown,
            ),
            (true, false) => DeepFoldMamaBearVerifier::verify_par(
                pp,
                verifiers,
                point_fr_natural,
                evals,
                &mut transcript,
                &mut proof,
            ),
        };
        tick!(pcs_t0, pcs_verify_us);
        tick!(total_t0, total_us);
        pcs_ok
    }
}

// =========================================================================================
// Derivation retained here so the verifier's consistency equations are self-contained.
// =========================================================================================
//
// Part 1 — σ permutation (Theorem 2): normal-layout SIMD sumcheck eliminates
// variables in order x_w, x_{w+1}, …, x_{nv-1}, x_0, …, x_{w-1} (W=8, w=3).
// So challenges in round-order (sumcheck_point, prod_point, point_fr) satisfy
// point[i] = value of x_{σ(i)}; converting to natural "point[k] = x_k" requires
// a right-rotation by w: natural_point[k] = simd_point[σ^{-1}(k)] where
// σ^{-1}(k) = (k - w) mod nv. This is exactly `simd_to_natural_point` above.
//
// Part 2 — verify_ell0 degree: the ZeroCheck round polynomial is degree 4
// (= eq · h, eq degree 1, h degree 3 from S·L·R). With only 3 hat values
// sent by the prover, the verifier MUST exploit the factored form
// s_i = c_i · t_i where c_i is linear and t_i is degree 3. That factored
// reconstruction is now in place in `SumcheckMamaBear::verify_ell0`.
//
// Part 3 — final-reduce eq point: the prover previously built its eq table
// from the raw (SIMD-order) sumcheck_point / prod_point, making the initial
// sum `tree_t_MLE` at a different semantic point than where the claims live
// (natural_point_*). The fix: permute to natural order in the prover before
// build_eq_table_packed. With this, the eq point matches claim semantics and
// the verifier's initial_y from Horner over claims is correct. Additionally,
// eq(natural_zc, natural_fr) = eq(sumcheck_point, point_fr) (permutations
// cancel), so the verifier's final-reduce tree reconstruction eq call stays cheap.
//
// Part 4 — per-wire ProductCheck: prod_point is in SIMD round order. The
// prover now calls `eval_multilinear_base_packed` with
// `simd_to_natural_point(prod_point)` so witness_eval/perm_eval are at the
// natural reduced point, and the verifier uses the same natural point for
// id_g / σ_g evaluation in ProductCheck reconstruction.
//
// Bias — residual mont_mul(0,0)=P bias from packed-SIMD accumulation is
// neutralised by applying `.reduce()` to all assertion operands before the
// `from_mont()` / equality comparison.
// =========================================================================================
