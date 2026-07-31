// Source files are physically grouped into baseline/ circuits/ mamabear/
// subfolders for readability, but the public module names are kept flat via
// #[path] so the module graph remains unchanged.

// baseline/ — Goldilocks generic PIOP reference
#[path = "baseline/circuit.rs"]   pub mod circuit;
#[path = "baseline/prover.rs"]    pub mod prover;
#[path = "baseline/sumcheck.rs"]  pub mod sumcheck;
#[path = "baseline/verifier.rs"]  pub mod verifier;
#[path = "baseline/prodcheck.rs"] pub mod prodcheck;

// circuits/ — circuit definitions (witness builders, gate identities).
#[path = "circuits/builder.rs"]       pub mod circuit_builder;
#[cfg(target_arch = "x86_64")] #[path = "circuits/boyar_peralta.rs"] pub mod boyar_peralta;
#[cfg(target_arch = "x86_64")] #[path = "circuits/sha256.rs"]        pub mod sha256_circuit;
#[cfg(target_arch = "x86_64")] #[path = "circuits/aes.rs"]           pub mod aes_circuit;
#[cfg(target_arch = "x86_64")] #[path = "circuits/blake3.rs"]        pub mod blake3_circuit;
#[cfg(target_arch = "x86_64")] #[path = "circuits/poseidon2.rs"]     pub mod poseidon2_circuit;
#[cfg(target_arch = "x86_64")] #[path = "circuits/keccakf.rs"]       pub mod keccakf_circuit;

// mamabear/ — MamaBear add+mul optimized prover stack.
// AVX-512IFMA + DeepFold MamaBear PCS; x86_64-only.
#[cfg(target_arch = "x86_64")] #[path = "mamabear/prover.rs"]                pub mod prover_mamabear;
#[cfg(target_arch = "x86_64")] #[path = "mamabear/prover_par.rs"]            pub mod prover_mamabear_par;
#[cfg(target_arch = "x86_64")] #[path = "mamabear/sumcheck.rs"]              pub mod sumcheck_mamabear;
#[cfg(target_arch = "x86_64")] #[path = "mamabear/sumcheck_par.rs"]          pub mod sumcheck_mamabear_par;
#[cfg(target_arch = "x86_64")] #[path = "mamabear/zerocheck_generic.rs"]     pub mod zerocheck_generic_mamabear;
#[cfg(target_arch = "x86_64")] #[path = "mamabear/zerocheck_generic_par.rs"] pub mod zerocheck_generic_mamabear_par;
#[cfg(target_arch = "x86_64")] #[path = "mamabear/verifier.rs"]              pub mod verifier_mamabear;
#[cfg(target_arch = "x86_64")] #[path = "mamabear/verifier_par.rs"]          pub mod verifier_mamabear_par;
#[cfg(target_arch = "x86_64")] #[path = "mamabear/prodcheck_perwire.rs"]     pub mod prodcheck_mamabear_perwire;
#[cfg(target_arch = "x86_64")] #[path = "mamabear/prodcheck_perwire_par.rs"] pub mod prodcheck_mamabear_perwire_par;

#[cfg(all(test, target_arch = "x86_64"))]
mod tests {
    use arithmetic::{
        field::{
            goldilocks64::{Goldilocks64, Goldilocks64Ext},
            mamabear::{MamaBearScalar, MamaBearScalarExt3},
            Field,
        },
        mul_group::Radix2Group,
    };
    use poly_commit::nil::{NilPcProver, NilPcVerifier};
    use poly_commit::deepfold_mamabear::DeepFoldMamaBearParam;
    use rand::rngs::SmallRng;
    use rand::SeedableRng;

    use crate::{
        circuit::Circuit,
        prover::Prover,
        prover_mamabear::{setup_mamabear, ProverMamaBear},
        verifier::Verifier,
        verifier_mamabear::VerifierMamaBear,
    };

    #[test]
    fn snark() {
        let mut rng = SmallRng::seed_from_u64(1);
        let nv = 13u32;
        let num_gates = 1u32 << nv;
        let mock_circuit = Circuit::<Goldilocks64Ext> {
            permutation: [
                (0..num_gates).map(|x| x.into()).collect(),
                (0..num_gates).map(|x| (x + (1 << 29)).into()).collect(),
                (0..num_gates).map(|x| (x + (1 << 30)).into()).collect(),
            ],
            selector: (0..num_gates).map(|x| (x & 1).into()).collect(),
        };

        let mut mult_subgroups = vec![Radix2Group::<Goldilocks64>::new(nv + 2)];
        for i in 1..nv as usize {
            mult_subgroups.push(mult_subgroups[i - 1].exp(2));
        }
        let (pk, vk) = mock_circuit.setup::<NilPcProver<_>, NilPcVerifier<_>>(&(), &());
        let prover = Prover { prover_key: pk };
        let verifier = Verifier { verifier_key: vk };
        let a = (0..num_gates)
            .map(|_| Goldilocks64::random(&mut rng))
            .collect::<Vec<_>>();
        let b = (0..num_gates)
            .map(|_| Goldilocks64::random(&mut rng))
            .collect::<Vec<_>>();
        let c = (0..num_gates)
            .map(|i| {
                let i = i as usize;
                let s = mock_circuit.selector[i];
                -((Goldilocks64::one() - s) * (a[i] + b[i]) + s * a[i] * b[i])
            })
            .collect();
        let proof = prover.prove(&(), nv as usize, [a, b, c]);
        assert!(verifier.verify(&(), nv as usize, proof));
    }

    fn run_mamabear_snark<F: crate::prover_mamabear::MamaBearExtConfig>() {
        let mut rng = SmallRng::seed_from_u64(7);
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

        let a = (0..num_gates)
            .map(|_| MamaBearScalar::random(&mut rng))
            .collect::<Vec<_>>();
        let b = (0..num_gates)
            .map(|_| MamaBearScalar::random(&mut rng))
            .collect::<Vec<_>>();
        use arithmetic::field::mamabear::P;
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
        use crate::prover_mamabear::AlignedPoly;
        let proof = prover.prove(&pp, nv, [
            AlignedPoly::from_sbf(&a),
            AlignedPoly::from_sbf(&b),
            AlignedPoly::from_sbf(&c),
        ]);
        assert!(verifier.verify(&pp, nv, proof));
    }

    #[test]
    fn snark_mamabear_ext3() {
        run_mamabear_snark::<MamaBearScalarExt3>();
    }

    fn run_mamabear_par_vs_serial<F: crate::prover_mamabear::MamaBearExtConfig>()
    where
        <F as crate::prover_mamabear::MamaBearExtConfig>::Packed: Send + Sync,
        F: Send + Sync,
    {
        let mut rng = SmallRng::seed_from_u64(42);
        let nv = 10usize;
        let num_gates = 1usize << nv;
        let circuit = Circuit::<F> {
            permutation: [
                (0..num_gates).map(|x| MamaBearScalar::from(x as u64)).collect(),
                (0..num_gates).map(|x| MamaBearScalar::from((x + (1 << 29)) as u64)).collect(),
                (0..num_gates).map(|x| MamaBearScalar::from((x + (1 << 30)) as u64)).collect(),
            ],
            selector: (0..num_gates).map(|x| MamaBearScalar::from((x & 1) as u64)).collect(),
        };
        let pp = DeepFoldMamaBearParam::new_default(nv, 3, 34);
        let (pk, vk) = setup_mamabear::<F>(&circuit, &pp);
        let prover = ProverMamaBear { prover_key: pk };
        let verifier = VerifierMamaBear { verifier_key: vk };

        let a: Vec<_> = (0..num_gates).map(|_| MamaBearScalar::random(&mut rng)).collect();
        let b: Vec<_> = (0..num_gates).map(|_| MamaBearScalar::random(&mut rng)).collect();
        use arithmetic::field::mamabear::P;
        let raw_mul = |x: u64, y: u64| ((x as u128 * y as u128) % (P as u128)) as u64;
        let c: Vec<_> = (0..num_gates).map(|i| {
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
        }).collect();

        use crate::prover_mamabear::AlignedPoly;
        let witness_serial = [
            AlignedPoly::from_sbf(&a),
            AlignedPoly::from_sbf(&b),
            AlignedPoly::from_sbf(&c),
        ];
        let witness_par = [
            AlignedPoly::from_sbf(&a),
            AlignedPoly::from_sbf(&b),
            AlignedPoly::from_sbf(&c),
        ];

        let proof_serial = prover.prove(&pp, nv, witness_serial);
        let proof_par = prover.prove_par(&pp, nv, witness_par);
        assert_eq!(
            proof_serial.bytes, proof_par.bytes,
            "Proof bytes mismatch: serial vs parallel"
        );
        assert!(verifier.verify(&pp, nv, proof_serial.clone()), "serial proof failed verify");
        assert!(verifier.verify(&pp, nv, proof_par.clone()), "parallel proof failed verify");
    }

    #[test]
    fn snark_mamabear_par_ext3() {
        run_mamabear_par_vs_serial::<MamaBearScalarExt3>();
    }
}
