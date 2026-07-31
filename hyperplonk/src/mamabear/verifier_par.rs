//! Parallel (rayon) variants of the HyperPlonk verifier.
//!
//! This file mirrors the structure of `hyperplonk::sumcheck_mamabear_par` /
//! `hyperplonk::prover_mamabear_par` relative to their serial files: the
//! serial implementation (the `verify_inner` orchestrator) lives in
//! `verifier_mamabear.rs`; this file attaches `verify_par` /
//! `verify_par_profiled` to the same `VerifierMamaBear` type via a separate
//! `impl` block.
//!
//! The only substantive difference between serial and parallel verify is the
//! PCS check: the serial path calls `DeepFoldMamaBearVerifier::verify{,_profiled}`
//! and the parallel path calls
//! `DeepFoldMamaBearVerifier::verify_par{,_profiled}`. The HyperPlonk-level
//! substages (ZeroCheck verify, ProductCheck verify, final-reduce sumcheck
//! verify, main-gate identity..final-reduce tree reconstruction) are dominated by Fiat-Shamir sequential dependencies and
//! remain serial. `verify_inner` in `verifier_mamabear.rs` accepts a
//! `par_pcs: bool` flag for this dispatch so the orchestrator stays single-source.

use util::fiat_shamir::Proof;

use poly_commit::deepfold_mamabear::DeepFoldMamaBearParam;

use crate::prover_mamabear::MamaBearExtConfig;
use crate::verifier_mamabear::{VerifierMamaBear, VerifyTimings};

impl<F: MamaBearExtConfig> VerifierMamaBear<F> {
    /// Parallel variant of `verify` — all HyperPlonk substages remain serial
    /// (Fiat-Shamir chain), but the DeepFold PCS verification (Merkle-path
    /// heavy) runs on `DeepFoldMamaBearVerifier::verify_par`.
    pub fn verify_par(&self, pp: &DeepFoldMamaBearParam, nv: usize, proof: Proof) -> bool {
        let mut t = VerifyTimings::default();
        self.verify_inner(pp, nv, proof, &mut t, false, true)
    }

    /// Profiling variant of `verify_par`.
    pub fn verify_par_profiled(
        &self,
        pp: &DeepFoldMamaBearParam,
        nv: usize,
        proof: Proof,
        timings: &mut VerifyTimings,
    ) -> bool {
        self.verify_inner(pp, nv, proof, timings, true, true)
    }
}

// =========================================================================================
// Correctness tests — verify_par on a fresh proof AND verify_par on a proof that
// `verify` accepts must agree. Both variants must reach the same bool for a valid
// proof.
// =========================================================================================

#[cfg(test)]
mod tests {
    use arithmetic::field::mamabear::{MamaBearScalar, MamaBearScalarExt3, P};
    use arithmetic::field::Field;
    use poly_commit::deepfold_mamabear::DeepFoldMamaBearParam;
    use rand::rngs::SmallRng;
    use rand::SeedableRng;

    use crate::circuit::Circuit;
    use crate::prover_mamabear::{
        setup_mamabear, AlignedPoly, MamaBearExtConfig, ProverMamaBear,
    };
    use crate::verifier_mamabear::VerifierMamaBear;

    fn run_verify_par_matches_serial<F: MamaBearExtConfig>()
    where
        F::Packed: Send + Sync,
        F: Send + Sync,
    {
        let mut rng = SmallRng::seed_from_u64(0xA51CE_u64);
        let nv = 10usize;
        let num_gates = 1usize << nv;
        let circuit = Circuit::<F> {
            permutation: [
                (0..num_gates).map(|x| MamaBearScalar::from(x as u64)).collect(),
                (0..num_gates)
                    .map(|x| MamaBearScalar::from((x + (1 << 29)) as u64))
                    .collect(),
                (0..num_gates)
                    .map(|x| MamaBearScalar::from((x + (1 << 30)) as u64))
                    .collect(),
            ],
            selector: (0..num_gates)
                .map(|x| MamaBearScalar::from((x & 1) as u64))
                .collect(),
        };
        let pp = DeepFoldMamaBearParam::new_default(nv, 3, 34);
        let (pk, vk) = setup_mamabear::<F>(&circuit, &pp);
        let prover = ProverMamaBear { prover_key: pk };
        let verifier = VerifierMamaBear { verifier_key: vk };

        // Build a valid witness: c = -((1-s)(a+b) + s*a*b) mod P with raw mul.
        let a = (0..num_gates)
            .map(|_| MamaBearScalar::random(&mut rng))
            .collect::<Vec<_>>();
        let b = (0..num_gates)
            .map(|_| MamaBearScalar::random(&mut rng))
            .collect::<Vec<_>>();
        let raw_mul = |x: u64, y: u64| ((x as u128 * y as u128) % (P as u128)) as u64;
        let c = (0..num_gates)
            .map(|i| {
                let s = circuit.selector[i].0;
                let a_i = a[i].0;
                let b_i = b[i].0;
                let a_plus_b = (a_i + b_i) % P;
                let one_minus_s = (P + 1 - s) % P;
                let term1 = raw_mul(one_minus_s, a_plus_b);
                let s_a_b = raw_mul(raw_mul(s, a_i), b_i);
                let expr = (term1 + s_a_b) % P;
                let neg = if expr == 0 { 0 } else { P - expr };
                MamaBearScalar(neg)
            })
            .collect::<Vec<_>>();

        let witness = [
            AlignedPoly::from_sbf(&a),
            AlignedPoly::from_sbf(&b),
            AlignedPoly::from_sbf(&c),
        ];
        let proof_serial = prover.prove(&pp, nv, witness.clone());
        let proof_parallel = prover.prove_par(&pp, nv, witness);

        // Prover must produce identical byte-level proofs in serial and parallel
        // modes (design invariant of the rest of the codebase).
        assert_eq!(
            proof_serial.bytes, proof_parallel.bytes,
            "serial and parallel prover must emit identical proofs"
        );

        // All four verify entry points must accept a valid proof.
        assert!(verifier.verify(&pp, nv, proof_serial.clone()));
        assert!(verifier.verify_par(&pp, nv, proof_serial.clone()));
        assert!(verifier.verify(&pp, nv, proof_parallel.clone()));
        assert!(verifier.verify_par(&pp, nv, proof_parallel));
    }

    #[test]
    fn verify_par_matches_serial_ext3() {
        run_verify_par_matches_serial::<MamaBearScalarExt3>();
    }
}
