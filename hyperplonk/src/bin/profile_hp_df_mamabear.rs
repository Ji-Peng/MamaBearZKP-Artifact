// x86_64-only profiling/size utility (AVX-512 MamaBear types). On other
// architectures this is a stub `fn main()` so `cargo bench` / `cargo test` --
// which compile every bin in the package -- still succeed; the real body
// below is x86_64-gated and unchanged.
#[cfg(not(target_arch = "x86_64"))]
fn main() {}
#[cfg(target_arch = "x86_64")]
fn main() {
    x86_impl::main();
}
#[cfg(target_arch = "x86_64")]
mod x86_impl {
//! Profile the `ProverMamaBear::prove` + `VerifierMamaBear::verify` paths
//! stage-by-stage.
//!
//! Command:
//!     RUSTFLAGS="-C target-cpu=native" cargo run -p hyperplonk --release \
//!         --bin profile_hp_df_mamabear
//!
//! The binary mirrors the current `prove` / `prove_par` implementations in
//! `hyperplonk/src/prover_mamabear{,_par}.rs` and the `verify` implementation
//! in `hyperplonk/src/verifier_mamabear.rs`, wrapping each stage with a
//! wall-clock timer. Setup is executed once per (variant, nv) outside the
//! measured region.
//!
//! FRI security level (MamaBear is Ext3-only):
//!   Ext3 -> PROV128 (s=88 queries, grinding = 16)

use std::time::Instant;

use arithmetic::field::{
    mamabear::{MamaBearScalar, MamaBearScalarExt3, P},
    Field,
};
use hyperplonk::{
    circuit::Circuit,
    prodcheck_mamabear_perwire::{ProdCheckTimings, ProdEqCheckMamaBearPerWire},
    prover_mamabear::{
        build_productcheck_inputs, prove_final_reduce_sumcheck, setup_mamabear, AlignedPoly,
        MamaBearExtConfig, ProverMamaBear,
    },
    sumcheck_mamabear::ZeroCheckTimings,
    verifier_mamabear::{VerifierMamaBear, VerifyTimings as HpVerifyTimings},
};
use poly_commit::{
    deepfold::MerkleRoot,
    deepfold_mamabear::{
        self, DeepFoldMamaBearParam, DeepFoldMamaBearProver, NewTimings, OpenTimings,
    },
    CommitmentSerde,
};
use rand::rngs::SmallRng;
use rand::SeedableRng;
use util::fiat_shamir::Transcript;
use util::params::{
    mamabear::{GRINDING_BITS_EXT3_PROV_QUERY128, QUERY_NUM_PROV_QUERY128},
    CODE_RATE_LOG,
};

type SBF = MamaBearScalar;
type SEF3 = MamaBearScalarExt3;

const NUM_WIRES: usize = 3;

/// FRI parameters per extension-field variant. Security footing matches
/// `hp_df_mamabear_sha256.rs`:
///   Ext3 -> PROV128 (R=3, s=88, grinding=16)
fn fri_params_for(variant_name: &str) -> (usize, u32) {
    match variant_name {
        "Ext3" => (QUERY_NUM_PROV_QUERY128, GRINDING_BITS_EXT3_PROV_QUERY128),
        other => panic!(
            "fri_params_for: unsupported variant {other:?}; MamaBear only supports Ext3 \
             (Ext2 has been removed, insufficient soundness)"
        ),
    }
}

#[derive(Clone, Debug, Default)]
struct StageTimes {
    total_us: u128,
    witness_pc_new_us: u128,
    witness_commit_us: u128,
    commit_serialize_us: u128,
    witness_to_mont_us: u128,
    witness_pack_us: u128,
    sumcheck_challenge_us: u128,
    zero_check_us: u128,
    zero_evals_append_us: u128,
    prod_challenge_us: u128,
    product_check_us: u128,
    mid_claims_us: u128,
    final_reduce_sumcheck_us: u128,
    final_claims_us: u128,
    pcs_open_us: u128,
    pcs_open_breakdown: OpenTimings,
    wpc_new_breakdown: NewTimings,
    prod_breakdown: ProdCheckTimings,
    zero_check_breakdown: ZeroCheckTimings,
    proof_bytes: usize,
}

#[derive(Clone, Debug, Default)]
struct VerifyTimes {
    total_us: u128,
    setup_us: u128,
    sc_chal_us: u128,
    zero_chk_us: u128,
    c1_us: u128,
    prod_chal_us: u128,
    prod_chk_us: u128,
    c2_us: u128,
    rho_init_y_us: u128,
    final_rsc_us: u128,
    c4_us: u128,
    pcs_verify_us: u128,
    pcs_verify_breakdown: deepfold_mamabear::VerifyTimings,
}


#[inline]
fn measure<T, F: FnOnce() -> T>(f: F) -> (T, u128) {
    let t0 = Instant::now();
    let out = f();
    (out, t0.elapsed().as_micros())
}

fn canonical_witness<F: MamaBearExtConfig>(circuit: &Circuit<F>, nv: usize, seed: u64) -> [Vec<SBF>; NUM_WIRES] {
    // HyperPlonk gate identity: h = (1-S)(L+R) + S*L*R + O must vanish on the
    // hypercube, so c = -((1-s)(a+b) + s*a*b) mod P. `MamaBearScalar::Mul` is
    // `mont_mul` (produces x*y/R), which is wrong for a RAW product — build
    // the witness with u128-based modular multiplication to match the verifier.
    let mut rng = SmallRng::seed_from_u64(seed);
    let num_gates = 1usize << nv;
    let a = (0..num_gates).map(|_| SBF::random(&mut rng)).collect::<Vec<_>>();
    let b = (0..num_gates).map(|_| SBF::random(&mut rng)).collect::<Vec<_>>();
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
    [a, b, c]
}

fn profile_prove<F: MamaBearExtConfig>(
    prover: &ProverMamaBear<F>,
    pp: &DeepFoldMamaBearParam,
    nv: usize,
    mut witness: [AlignedPoly; NUM_WIRES],
) -> StageTimes {
    let mut stage = StageTimes::default();
    let total_start = Instant::now();

    let mut transcript = Transcript::new();

    let mut new_timings = NewTimings::default();
    let (witness_pc, elapsed) = measure(|| {
        DeepFoldMamaBearProver::<F>::new_profiled(
            pp,
            &[witness[0].as_sbf(), witness[1].as_sbf(), witness[2].as_sbf()],
            &mut new_timings,
        )
    });
    stage.witness_pc_new_us = elapsed;
    stage.wpc_new_breakdown = new_timings;

    let (commit, elapsed) = measure(|| witness_pc.commit());
    stage.witness_commit_us = elapsed;

    let ((), elapsed) = measure(|| {
        let mut buffer = vec![0u8; MerkleRoot::size(nv, NUM_WIRES)];
        commit.serialize_into(&mut buffer);
        transcript.append_u8_slice(&buffer, buffer.len());
    });
    stage.commit_serialize_us = elapsed;

    let ((), elapsed) = measure(|| {
        for poly in witness.iter_mut() {
            poly.to_montgomery_in_place();
        }
    });
    stage.witness_to_mont_us = elapsed;

    let ((), elapsed) = measure(|| {});
    stage.witness_pack_us = elapsed;

    let (sumcheck_r, elapsed) = measure(|| {
        (0..nv)
            .map(|_| transcript.challenge_f::<F>())
            .collect::<Vec<_>>()
    });
    stage.sumcheck_challenge_us = elapsed;

    let mut zc_timings = ZeroCheckTimings::default();
    let zc_evals = [
        prover.prover_key.selector.as_pbf().to_vec(),
        witness[0].as_pbf().to_vec(),
        witness[1].as_pbf().to_vec(),
        witness[2].as_pbf().to_vec(),
    ];
    let ((sumcheck_point, zero_evals), elapsed) = measure(|| {
        F::prove_zero_check_profiled(
            zc_evals,
            &sumcheck_r,
            &mut transcript,
            &mut zc_timings,
        )
    });
    stage.zero_check_us = elapsed;
    stage.zero_check_breakdown = zc_timings;

    let ((), elapsed) = measure(|| {
        for value in zero_evals.into_iter().take(4) {
            transcript.append_f(value);
        }
    });
    stage.zero_evals_append_us = elapsed;

    let ((prod_r0, prod_r1), elapsed) = measure(|| {
        let r0: F = transcript.challenge_f();
        let r1: F = transcript.challenge_f();
        (r0, r1)
    });
    stage.prod_challenge_us = elapsed;

    let (prod_point, elapsed) = measure(|| {
        let mut prod_timings = ProdCheckTimings::default();
        let t_inputs = std::time::Instant::now();
        let inputs = build_productcheck_inputs::<F>(
            &witness,
            &prover.prover_key.identical,
            &prover.prover_key.permutation,
            prod_r0.to_mont(),
            prod_r1.to_mont(),
        );
        prod_timings.build_inputs_us = t_inputs.elapsed().as_micros();
        let t_prove = std::time::Instant::now();
        let pt = ProdEqCheckMamaBearPerWire::prove_profiled::<F::Packed>(
            inputs,
            &mut transcript,
            &mut prod_timings,
        );
        // round_misc = prove_profiled_total - (everything else measured inside it)
        let prove_us = t_prove.elapsed().as_micros();
        let accounted = prod_timings.tree_build_us
            + prod_timings.top_extract_us
            + prod_timings.eq_tables_us
            + prod_timings.round_t_us
            + prod_timings.fold_us
            + prod_timings.inreg_fold_us
            + prod_timings.scalar_tail_us;
        prod_timings.round_misc_us = prove_us.saturating_sub(accounted);
        stage.prod_breakdown = prod_timings;
        pt
    });
    stage.product_check_us = elapsed;

    let ((), elapsed) = measure(|| {
        for poly in witness.iter() {
            transcript.append_f(
                F::eval_multilinear_base_packed(poly.as_pbf(), &prod_point[..nv]).from_mont(),
            );
        }
        for poly in prover.prover_key.permutation.iter() {
            transcript.append_f(
                F::eval_multilinear_base_packed(poly.as_pbf(), &prod_point[..nv]).from_mont(),
            );
        }
    });
    stage.mid_claims_us = elapsed;

    let evals: [&AlignedPoly; 7] = [
        &prover.prover_key.selector,
        &prover.prover_key.permutation[0],
        &prover.prover_key.permutation[1],
        &prover.prover_key.permutation[2],
        &witness[0],
        &witness[1],
        &witness[2],
    ];
    let (point_mont, elapsed) = measure(|| {
        prove_final_reduce_sumcheck::<F>(
            evals,
            &sumcheck_point,
            &prod_point[..nv],
            &mut transcript,
        )
    });
    stage.final_reduce_sumcheck_us = elapsed;

    let ((), elapsed) = measure(|| {
        transcript.append_f(
            F::eval_multilinear_base_packed(prover.prover_key.selector.as_pbf(), &point_mont).from_mont(),
        );
        for poly in prover.prover_key.permutation.iter() {
            transcript
                .append_f(F::eval_multilinear_base_packed(poly.as_pbf(), &point_mont).from_mont());
        }
        for poly in witness.iter() {
            transcript
                .append_f(F::eval_multilinear_base_packed(poly.as_pbf(), &point_mont).from_mont());
        }
    });
    stage.final_claims_us = elapsed;

    let mut open_timings = OpenTimings::default();
    let ((), elapsed) = measure(|| {
        DeepFoldMamaBearProver::open_profiled(
            pp,
            &[&prover.prover_key.commitments, &witness_pc],
            point_mont,
            &mut transcript,
            &mut open_timings,
        );
    });
    stage.pcs_open_us = elapsed;
    stage.pcs_open_breakdown = open_timings;

    stage.proof_bytes = transcript.proof.bytes.len();
    stage.total_us = total_start.elapsed().as_micros();
    stage
}

/// Parallel-prover profiling — mirrors `profile_prove` but uses parallel
/// components for wpc_new, prod_chk, and pcs_open.
fn profile_prove_par<F: MamaBearExtConfig>(
    prover: &ProverMamaBear<F>,
    pp: &DeepFoldMamaBearParam,
    nv: usize,
    mut witness: [AlignedPoly; NUM_WIRES],
) -> StageTimes
where
    F::Packed: Send + Sync,
    F: Send + Sync,
{
    use hyperplonk::prodcheck_mamabear_perwire_par::ProdEqCheckMamaBearPerWirePar;
    use hyperplonk::prover_mamabear_par::build_productcheck_inputs_par;
    use poly_commit::deepfold_mamabear_par;

    let mut stage = StageTimes::default();
    let total_start = Instant::now();

    let mut transcript = Transcript::new();

    // Parallel commit
    let mut new_timings = NewTimings::default();
    let (witness_pc, elapsed) = measure(|| {
        deepfold_mamabear_par::new_par_profiled::<F>(
            pp,
            &[witness[0].as_sbf(), witness[1].as_sbf(), witness[2].as_sbf()],
            &mut new_timings,
        )
    });
    stage.witness_pc_new_us = elapsed;
    stage.wpc_new_breakdown = new_timings;

    let (commit, elapsed) = measure(|| witness_pc.commit());
    stage.witness_commit_us = elapsed;

    let ((), elapsed) = measure(|| {
        let mut buffer = vec![0u8; MerkleRoot::size(nv, NUM_WIRES)];
        commit.serialize_into(&mut buffer);
        transcript.append_u8_slice(&buffer, buffer.len());
    });
    stage.commit_serialize_us = elapsed;

    let ((), elapsed) = measure(|| {
        for poly in witness.iter_mut() {
            poly.to_montgomery_in_place();
        }
    });
    stage.witness_to_mont_us = elapsed;

    let ((), elapsed) = measure(|| {});
    stage.witness_pack_us = elapsed;

    let (sumcheck_r, elapsed) = measure(|| {
        (0..nv)
            .map(|_| transcript.challenge_f::<F>())
            .collect::<Vec<_>>()
    });
    stage.sumcheck_challenge_us = elapsed;

    // ZeroCheck — parallel variant.
    let mut zc_timings = ZeroCheckTimings::default();
    let zc_evals = [
        prover.prover_key.selector.as_pbf().to_vec(),
        witness[0].as_pbf().to_vec(),
        witness[1].as_pbf().to_vec(),
        witness[2].as_pbf().to_vec(),
    ];
    let ((sumcheck_point, zero_evals), elapsed) = measure(|| {
        F::prove_zero_check_par_profiled(
            zc_evals,
            &sumcheck_r,
            &mut transcript,
            &mut zc_timings,
        )
    });
    stage.zero_check_us = elapsed;
    stage.zero_check_breakdown = zc_timings;

    let ((), elapsed) = measure(|| {
        for value in zero_evals.into_iter().take(4) {
            transcript.append_f(value);
        }
    });
    stage.zero_evals_append_us = elapsed;

    let ((prod_r0, prod_r1), elapsed) = measure(|| {
        let r0: F = transcript.challenge_f();
        let r1: F = transcript.challenge_f();
        (r0, r1)
    });
    stage.prod_challenge_us = elapsed;

    // Parallel ProductCheck
    let (prod_point, elapsed) = measure(|| {
        let mut prod_timings = ProdCheckTimings::default();
        let t_inputs = std::time::Instant::now();
        let inputs = build_productcheck_inputs_par::<F>(
            &witness,
            &prover.prover_key.identical,
            &prover.prover_key.permutation,
            prod_r0.to_mont(),
            prod_r1.to_mont(),
        );
        prod_timings.build_inputs_us = t_inputs.elapsed().as_micros();
        let t_prove = std::time::Instant::now();
        let pt = ProdEqCheckMamaBearPerWirePar::prove_profiled::<F::Packed>(
            inputs,
            &mut transcript,
            &mut prod_timings,
        );
        let prove_us = t_prove.elapsed().as_micros();
        let accounted = prod_timings.tree_build_us
            + prod_timings.top_extract_us
            + prod_timings.eq_tables_us
            + prod_timings.round_t_us
            + prod_timings.fold_us
            + prod_timings.inreg_fold_us
            + prod_timings.scalar_tail_us;
        prod_timings.round_misc_us = prove_us.saturating_sub(accounted);
        stage.prod_breakdown = prod_timings;
        pt
    });
    stage.product_check_us = elapsed;

    let ((), elapsed) = measure(|| {
        let mid_polys: [&[_]; 6] = [
            witness[0].as_pbf(),
            witness[1].as_pbf(),
            witness[2].as_pbf(),
            prover.prover_key.permutation[0].as_pbf(),
            prover.prover_key.permutation[1].as_pbf(),
            prover.prover_key.permutation[2].as_pbf(),
        ];
        let mid_evals = hyperplonk::prover_mamabear_par::eval_multilinear_base_packed_batch_par::<F>(
            &mid_polys,
            &prod_point[..nv],
        );
        for v in &mid_evals {
            transcript.append_f(v.from_mont());
        }
    });
    stage.mid_claims_us = elapsed;

    let evals: [&AlignedPoly; 7] = [
        &prover.prover_key.selector,
        &prover.prover_key.permutation[0],
        &prover.prover_key.permutation[1],
        &prover.prover_key.permutation[2],
        &witness[0],
        &witness[1],
        &witness[2],
    ];
    // final_rsc: parallel variant
    let (point_mont, elapsed) = measure(|| {
        hyperplonk::prover_mamabear_par::prove_final_reduce_sumcheck_par::<F>(
            evals,
            &sumcheck_point,
            &prod_point[..nv],
            &mut transcript,
        )
    });
    stage.final_reduce_sumcheck_us = elapsed;

    let ((), elapsed) = measure(|| {
        let final_polys: [&[_]; 7] = [
            prover.prover_key.selector.as_pbf(),
            prover.prover_key.permutation[0].as_pbf(),
            prover.prover_key.permutation[1].as_pbf(),
            prover.prover_key.permutation[2].as_pbf(),
            witness[0].as_pbf(),
            witness[1].as_pbf(),
            witness[2].as_pbf(),
        ];
        let final_evals = hyperplonk::prover_mamabear_par::eval_multilinear_base_packed_batch_par::<F>(
            &final_polys,
            &point_mont,
        );
        for v in &final_evals {
            transcript.append_f(v.from_mont());
        }
    });
    stage.final_claims_us = elapsed;

    // Parallel pcs_open
    let mut open_timings = OpenTimings::default();
    let ((), elapsed) = measure(|| {
        DeepFoldMamaBearProver::open_par_profiled(
            pp,
            &[&prover.prover_key.commitments, &witness_pc],
            point_mont,
            &mut transcript,
            &mut open_timings,
        );
    });
    stage.pcs_open_us = elapsed;
    stage.pcs_open_breakdown = open_timings;

    stage.proof_bytes = transcript.proof.bytes.len();
    stage.total_us = total_start.elapsed().as_micros();
    stage
}

fn build_circuit<F: MamaBearExtConfig>(nv: usize) -> Circuit<F> {
    let num_gates = 1usize << nv;
    Circuit::<F> {
        permutation: [
            (0..num_gates).map(|x| SBF::from(x as u64)).collect(),
            (0..num_gates)
                .map(|x| SBF::from((x + (1 << 29)) as u64))
                .collect(),
            (0..num_gates)
                .map(|x| SBF::from((x + (1 << 30)) as u64))
                .collect(),
        ],
        selector: (0..num_gates)
            .map(|x| SBF::from((x & 1) as u64))
            .collect(),
    }
}

fn run_variant<F: MamaBearExtConfig>(variant_name: &str, nv_range: &[usize], reps_table: &[(usize, usize)]) -> Vec<(usize, StageTimes, VerifyTimes)>
where
    F::Packed: Send + Sync,
    F: Send + Sync,
{
    let parallel_mode = std::env::var("PROFILE_PAR").ok().map_or(false, |v| v == "1");
    let mut results = Vec::new();
    for &nv in nv_range {
        let reps = reps_table
            .iter()
            .find(|(max_nv, _)| nv <= *max_nv)
            .map(|(_, r)| *r)
            .unwrap_or(1);

        let circuit = build_circuit::<F>(nv);
        let (query_num, grinding_bits) = fri_params_for(variant_name);
        let mut pp = DeepFoldMamaBearParam::new_default(nv, CODE_RATE_LOG, query_num);
        pp.grinding_bits = grinding_bits;
        let (pk, vk) = setup_mamabear::<F>(&circuit, &pp);
        let prover = ProverMamaBear { prover_key: pk };
        let verifier = VerifierMamaBear { verifier_key: vk };

        let mut acc = StageTimes::default();
        let mut v_acc = VerifyTimes::default();
        for rep in 0..reps {
            let raw_witness = canonical_witness(&circuit, nv, 0xC0FFEE_u64.wrapping_add(rep as u64));
            let witness_prove = raw_witness.clone().map(|p| AlignedPoly::from_sbf(&p));
            let stages = if parallel_mode {
                profile_prove_par(&prover, &pp, nv, witness_prove)
            } else {
                profile_prove(&prover, &pp, nv, witness_prove)
            };

            // Produce a real proof + profile verify. The proof must match what
            // the current `prove()` produces (not the inline profile_prove body)
            // so the verifier can check it; if the two drift, this will assert.
            let witness_verify = raw_witness.map(|p| AlignedPoly::from_sbf(&p));
            let proof = if parallel_mode {
                prover.prove_par(&pp, nv, witness_verify)
            } else {
                prover.prove(&pp, nv, witness_verify)
            };
            let mut vt = HpVerifyTimings::default();
            let ok = if parallel_mode {
                verifier.verify_par_profiled(&pp, nv, proof, &mut vt)
            } else {
                verifier.verify_profiled(&pp, nv, proof, &mut vt)
            };
            assert!(ok, "verify failed at {} nv={}", variant_name, nv);
            accumulate_verify(&mut v_acc, &vt);
            acc.total_us += stages.total_us;
            acc.witness_pc_new_us += stages.witness_pc_new_us;
            acc.witness_commit_us += stages.witness_commit_us;
            acc.commit_serialize_us += stages.commit_serialize_us;
            acc.witness_to_mont_us += stages.witness_to_mont_us;
            acc.witness_pack_us += stages.witness_pack_us;
            acc.sumcheck_challenge_us += stages.sumcheck_challenge_us;
            acc.zero_check_us += stages.zero_check_us;
            acc.zero_evals_append_us += stages.zero_evals_append_us;
            acc.prod_challenge_us += stages.prod_challenge_us;
            acc.product_check_us += stages.product_check_us;
            acc.mid_claims_us += stages.mid_claims_us;
            acc.final_reduce_sumcheck_us += stages.final_reduce_sumcheck_us;
            acc.final_claims_us += stages.final_claims_us;
            acc.pcs_open_us += stages.pcs_open_us;
            acc.pcs_open_breakdown.total_us += stages.pcs_open_breakdown.total_us;
            acc.pcs_open_breakdown.combine_polys_us += stages.pcs_open_breakdown.combine_polys_us;
            acc.pcs_open_breakdown.combine_subs_us += stages.pcs_open_breakdown.combine_subs_us;
            acc.pcs_open_breakdown.mle_eval_us += stages.pcs_open_breakdown.mle_eval_us;
            acc.pcs_open_breakdown.multilin_fold_us += stages.pcs_open_breakdown.multilin_fold_us;
            acc.pcs_open_breakdown.split_fold_us += stages.pcs_open_breakdown.split_fold_us;
            acc.pcs_open_breakdown.fri_fold_us += stages.pcs_open_breakdown.fri_fold_us;
            acc.pcs_open_breakdown.fri_merkle_us += stages.pcs_open_breakdown.fri_merkle_us;
            acc.pcs_open_breakdown.query_phase_us += stages.pcs_open_breakdown.query_phase_us;
            acc.wpc_new_breakdown.total_us += stages.wpc_new_breakdown.total_us;
            acc.wpc_new_breakdown.alloc_us += stages.wpc_new_breakdown.alloc_us;
            acc.wpc_new_breakdown.split_us += stages.wpc_new_breakdown.split_us;
            acc.wpc_new_breakdown.fft_us += stages.wpc_new_breakdown.fft_us;
            acc.wpc_new_breakdown.append_us += stages.wpc_new_breakdown.append_us;
            acc.wpc_new_breakdown.arc_convert_us += stages.wpc_new_breakdown.arc_convert_us;
            acc.wpc_new_breakdown.leaf_hash_us += stages.wpc_new_breakdown.leaf_hash_us;
            acc.wpc_new_breakdown.merkle_tree_us += stages.wpc_new_breakdown.merkle_tree_us;
            acc.wpc_new_breakdown.wrap_us += stages.wpc_new_breakdown.wrap_us;
            acc.prod_breakdown.build_inputs_us += stages.prod_breakdown.build_inputs_us;
            acc.prod_breakdown.tree_build_us += stages.prod_breakdown.tree_build_us;
            acc.prod_breakdown.top_extract_us += stages.prod_breakdown.top_extract_us;
            acc.prod_breakdown.eq_tables_us += stages.prod_breakdown.eq_tables_us;
            acc.prod_breakdown.round_t_us += stages.prod_breakdown.round_t_us;
            acc.prod_breakdown.fold_us += stages.prod_breakdown.fold_us;
            acc.prod_breakdown.inreg_fold_us += stages.prod_breakdown.inreg_fold_us;
            acc.prod_breakdown.scalar_tail_us += stages.prod_breakdown.scalar_tail_us;
            acc.prod_breakdown.round_misc_us += stages.prod_breakdown.round_misc_us;
            acc.zero_check_breakdown.eq_tables_us += stages.zero_check_breakdown.eq_tables_us;
            acc.zero_check_breakdown.precompute_us += stages.zero_check_breakdown.precompute_us;
            acc.zero_check_breakdown.small_value_rounds_us += stages.zero_check_breakdown.small_value_rounds_us;
            acc.zero_check_breakdown.transition_fold_us += stages.zero_check_breakdown.transition_fold_us;
            acc.zero_check_breakdown.packed_fold_rounds_us += stages.zero_check_breakdown.packed_fold_rounds_us;
            acc.zero_check_breakdown.scalar_tail_us += stages.zero_check_breakdown.scalar_tail_us;
            acc.zero_check_breakdown.total_us += stages.zero_check_breakdown.total_us;
            acc.proof_bytes = stages.proof_bytes;
        }
        let reps_u128 = reps as u128;
        acc.total_us /= reps_u128;
        acc.witness_pc_new_us /= reps_u128;
        acc.witness_commit_us /= reps_u128;
        acc.commit_serialize_us /= reps_u128;
        acc.witness_to_mont_us /= reps_u128;
        acc.witness_pack_us /= reps_u128;
        acc.sumcheck_challenge_us /= reps_u128;
        acc.zero_check_us /= reps_u128;
        acc.zero_evals_append_us /= reps_u128;
        acc.prod_challenge_us /= reps_u128;
        acc.product_check_us /= reps_u128;
        acc.mid_claims_us /= reps_u128;
        acc.final_reduce_sumcheck_us /= reps_u128;
        acc.final_claims_us /= reps_u128;
        acc.pcs_open_us /= reps_u128;
        acc.pcs_open_breakdown.total_us /= reps_u128;
        acc.pcs_open_breakdown.combine_polys_us /= reps_u128;
        acc.pcs_open_breakdown.combine_subs_us /= reps_u128;
        acc.pcs_open_breakdown.mle_eval_us /= reps_u128;
        acc.pcs_open_breakdown.multilin_fold_us /= reps_u128;
        acc.pcs_open_breakdown.split_fold_us /= reps_u128;
        acc.pcs_open_breakdown.fri_fold_us /= reps_u128;
        acc.pcs_open_breakdown.fri_merkle_us /= reps_u128;
        acc.pcs_open_breakdown.query_phase_us /= reps_u128;
        acc.wpc_new_breakdown.total_us /= reps_u128;
        acc.wpc_new_breakdown.alloc_us /= reps_u128;
        acc.wpc_new_breakdown.split_us /= reps_u128;
        acc.wpc_new_breakdown.fft_us /= reps_u128;
        acc.wpc_new_breakdown.append_us /= reps_u128;
        acc.wpc_new_breakdown.arc_convert_us /= reps_u128;
        acc.wpc_new_breakdown.leaf_hash_us /= reps_u128;
        acc.wpc_new_breakdown.merkle_tree_us /= reps_u128;
        acc.wpc_new_breakdown.wrap_us /= reps_u128;
        acc.prod_breakdown.build_inputs_us /= reps_u128;
        acc.prod_breakdown.tree_build_us /= reps_u128;
        acc.prod_breakdown.top_extract_us /= reps_u128;
        acc.prod_breakdown.eq_tables_us /= reps_u128;
        acc.prod_breakdown.round_t_us /= reps_u128;
        acc.prod_breakdown.fold_us /= reps_u128;
        acc.prod_breakdown.inreg_fold_us /= reps_u128;
        acc.prod_breakdown.scalar_tail_us /= reps_u128;
        acc.prod_breakdown.round_misc_us /= reps_u128;
        acc.zero_check_breakdown.eq_tables_us /= reps_u128;
        acc.zero_check_breakdown.precompute_us /= reps_u128;
        acc.zero_check_breakdown.small_value_rounds_us /= reps_u128;
        acc.zero_check_breakdown.transition_fold_us /= reps_u128;
        acc.zero_check_breakdown.packed_fold_rounds_us /= reps_u128;
        acc.zero_check_breakdown.scalar_tail_us /= reps_u128;
        acc.zero_check_breakdown.total_us /= reps_u128;
        divide_verify(&mut v_acc, reps_u128);
        println!(
            "{} nv={} reps={} prove={:.2}ms verify={:.2}ms proof={}B",
            variant_name,
            nv,
            reps,
            acc.total_us as f64 / 1000.0,
            v_acc.total_us as f64 / 1000.0,
            acc.proof_bytes
        );
        results.push((nv, acc, v_acc));
    }
    results
}

fn accumulate_verify(acc: &mut VerifyTimes, s: &HpVerifyTimings) {
    acc.total_us += s.total_us;
    acc.setup_us += s.setup_us;
    acc.sc_chal_us += s.sc_chal_us;
    acc.zero_chk_us += s.zero_chk_us;
    acc.c1_us += s.c1_us;
    acc.prod_chal_us += s.prod_chal_us;
    acc.prod_chk_us += s.prod_chk_us;
    acc.c2_us += s.c2_us;
    acc.rho_init_y_us += s.rho_init_y_us;
    acc.final_rsc_us += s.final_rsc_us;
    acc.c4_us += s.c4_us;
    acc.pcs_verify_us += s.pcs_verify_us;
    acc.pcs_verify_breakdown.total_us += s.pcs_verify_breakdown.total_us;
    acc.pcs_verify_breakdown.fold_check_us += s.pcs_verify_breakdown.fold_check_us;
    acc.pcs_verify_breakdown.grinding_us += s.pcs_verify_breakdown.grinding_us;
    acc.pcs_verify_breakdown.query_prep_us += s.pcs_verify_breakdown.query_prep_us;
    acc.pcs_verify_breakdown.fat_merkle_us += s.pcs_verify_breakdown.fat_merkle_us;
    acc.pcs_verify_breakdown.split_fold_us += s.pcs_verify_breakdown.split_fold_us;
    acc.pcs_verify_breakdown.std_fri_merkle_us += s.pcs_verify_breakdown.std_fri_merkle_us;
    acc.pcs_verify_breakdown.fri_folds_us += s.pcs_verify_breakdown.fri_folds_us;
}

fn divide_verify(acc: &mut VerifyTimes, n: u128) {
    acc.total_us /= n;
    acc.setup_us /= n;
    acc.sc_chal_us /= n;
    acc.zero_chk_us /= n;
    acc.c1_us /= n;
    acc.prod_chal_us /= n;
    acc.prod_chk_us /= n;
    acc.c2_us /= n;
    acc.rho_init_y_us /= n;
    acc.final_rsc_us /= n;
    acc.c4_us /= n;
    acc.pcs_verify_us /= n;
    acc.pcs_verify_breakdown.total_us /= n;
    acc.pcs_verify_breakdown.fold_check_us /= n;
    acc.pcs_verify_breakdown.grinding_us /= n;
    acc.pcs_verify_breakdown.query_prep_us /= n;
    acc.pcs_verify_breakdown.fat_merkle_us /= n;
    acc.pcs_verify_breakdown.split_fold_us /= n;
    acc.pcs_verify_breakdown.std_fri_merkle_us /= n;
    acc.pcs_verify_breakdown.fri_folds_us /= n;
}

/// One row (per nv) of a stage/share table pair. `total_us` is the
/// denominator used for the share computation; `items` holds raw us values
/// in the canonical order (parallel to the `names` vector used to build
/// the layout).
#[derive(Clone, Debug)]
struct TableRow {
    nv: usize,
    total_us: u128,
    items: Vec<u128>,
}

/// Sort + collapse layout shared by a (stage ms, share %) table pair.
///
/// `visible_idx` lists the canonical item indices in descending max-nv share
/// order. `collapsed_idx` lists items whose share at the largest nv is below
/// the threshold (default 1%). Both printers iterate `visible_idx` so the
/// two tables always agree on column order.
#[derive(Clone, Debug)]
struct TableLayout {
    visible_idx: Vec<usize>,
    collapsed_idx: Vec<usize>,
    names: Vec<&'static str>,
}

impl TableLayout {
    fn has_other(&self) -> bool {
        !self.collapsed_idx.is_empty()
    }

    fn collapsed_sum(&self, row: &TableRow) -> u128 {
        self.collapsed_idx.iter().map(|&i| row.items[i]).sum()
    }
}

fn build_layout(
    names: Vec<&'static str>,
    rows: &[TableRow],
    threshold_pct: f64,
) -> TableLayout {
    assert!(!rows.is_empty());
    let ref_row = rows.iter().max_by_key(|r| r.nv).unwrap();
    let denom = ref_row.total_us.max(1) as f64;
    let mut ordered: Vec<(usize, f64)> = (0..names.len())
        .map(|i| (i, ref_row.items[i] as f64 / denom * 100.0))
        .collect();
    ordered.sort_by(|a, b| {
        b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut visible_idx = Vec::new();
    let mut collapsed_idx = Vec::new();
    for (i, pct) in ordered {
        if pct >= threshold_pct {
            visible_idx.push(i);
        } else {
            collapsed_idx.push(i);
        }
    }
    TableLayout {
        visible_idx,
        collapsed_idx,
        names,
    }
}

fn print_ms_table(
    heading: &str,
    subheading: &str,
    layout: &TableLayout,
    rows: &[TableRow],
    proof_bytes: Option<&[usize]>,
) {
    println!("\n### {} {} (ms)", heading, subheading);
    let mut header = String::from("| nv | total |");
    let mut sep = String::from("| --- | ---: |");
    if proof_bytes.is_some() {
        header.push_str(" proof B |");
        sep.push_str(" ---: |");
    }
    for &i in &layout.visible_idx {
        header.push_str(&format!(" {} |", layout.names[i]));
        sep.push_str(" ---: |");
    }
    if layout.has_other() {
        header.push_str(" other |");
        sep.push_str(" ---: |");
    }
    println!("{}", header);
    println!("{}", sep);
    for (row_idx, r) in rows.iter().enumerate() {
        let mut line = format!("| {} | {:.2} |", r.nv, r.total_us as f64 / 1000.0);
        if let Some(pbs) = proof_bytes {
            line.push_str(&format!(" {} |", pbs[row_idx]));
        }
        for &i in &layout.visible_idx {
            line.push_str(&format!(" {:.2} |", r.items[i] as f64 / 1000.0));
        }
        if layout.has_other() {
            line.push_str(&format!(
                " {:.2} |",
                layout.collapsed_sum(r) as f64 / 1000.0
            ));
        }
        println!("{}", line);
    }
}

fn print_share_table(
    heading: &str,
    subheading: &str,
    layout: &TableLayout,
    rows: &[TableRow],
) {
    println!("\n### {} {} share (%)", heading, subheading);
    let mut header = String::from("| nv |");
    let mut sep = String::from("| --- |");
    for &i in &layout.visible_idx {
        header.push_str(&format!(" {} |", layout.names[i]));
        sep.push_str(" ---: |");
    }
    if layout.has_other() {
        header.push_str(" other |");
        sep.push_str(" ---: |");
    }
    println!("{}", header);
    println!("{}", sep);
    for r in rows {
        let t = r.total_us.max(1) as f64;
        let mut line = format!("| {} |", r.nv);
        for &i in &layout.visible_idx {
            line.push_str(&format!(" {:.1} |", r.items[i] as f64 / t * 100.0));
        }
        if layout.has_other() {
            line.push_str(&format!(
                " {:.1} |",
                layout.collapsed_sum(r) as f64 / t * 100.0
            ));
        }
        println!("{}", line);
    }
}

fn print_layout_legend(layout: &TableLayout) {
    if layout.has_other() {
        let names: Vec<&'static str> = layout
            .collapsed_idx
            .iter()
            .map(|&i| layout.names[i])
            .collect();
        println!("other includes: {}", names.join(", "));
    }
}

fn build_main_stage_model(
    results: &[(usize, StageTimes, VerifyTimes)],
) -> (Vec<&'static str>, Vec<TableRow>, Vec<usize>) {
    let names = vec![
        "wpc_new",
        "wpc_commit",
        "commit_ser",
        "w_to_mont",
        "w_pack",
        "sc_chal",
        "zero_chk",
        "zero_ap",
        "prod_chal",
        "prod_chk",
        "mid_clm",
        "final_rsc",
        "final_clm",
        "pcs_open",
    ];
    let mut rows = Vec::with_capacity(results.len());
    let mut proofs = Vec::with_capacity(results.len());
    for (nv, st, _) in results {
        rows.push(TableRow {
            nv: *nv,
            total_us: st.total_us,
            items: vec![
                st.witness_pc_new_us,
                st.witness_commit_us,
                st.commit_serialize_us,
                st.witness_to_mont_us,
                st.witness_pack_us,
                st.sumcheck_challenge_us,
                st.zero_check_us,
                st.zero_evals_append_us,
                st.prod_challenge_us,
                st.product_check_us,
                st.mid_claims_us,
                st.final_reduce_sumcheck_us,
                st.final_claims_us,
                st.pcs_open_us,
            ],
        });
        proofs.push(st.proof_bytes);
    }
    (names, rows, proofs)
}

fn build_open_model(
    results: &[(usize, StageTimes, VerifyTimes)],
) -> (Vec<&'static str>, Vec<TableRow>) {
    let names = vec![
        "combine_polys",
        "combine_subs",
        "mle_eval",
        "mlin_fold",
        "split_fold",
        "fri_fold",
        "fri_mkl",
        "query",
    ];
    let rows = results
        .iter()
        .map(|(nv, st, _)| {
            let b = &st.pcs_open_breakdown;
            TableRow {
                nv: *nv,
                total_us: b.total_us,
                items: vec![
                    b.combine_polys_us,
                    b.combine_subs_us,
                    b.mle_eval_us,
                    b.multilin_fold_us,
                    b.split_fold_us,
                    b.fri_fold_us,
                    b.fri_merkle_us,
                    b.query_phase_us,
                ],
            }
        })
        .collect();
    (names, rows)
}

fn build_wpc_new_model(
    results: &[(usize, StageTimes, VerifyTimes)],
) -> (Vec<&'static str>, Vec<TableRow>) {
    let names = vec![
        "alloc",
        "split",
        "fft",
        "append",
        "arc",
        "leaf_hash",
        "mkl_tree",
        "wrap",
    ];
    let rows = results
        .iter()
        .map(|(nv, st, _)| {
            let b = &st.wpc_new_breakdown;
            TableRow {
                nv: *nv,
                total_us: b.total_us,
                items: vec![
                    b.alloc_us,
                    b.split_us,
                    b.fft_us,
                    b.append_us,
                    b.arc_convert_us,
                    b.leaf_hash_us,
                    b.merkle_tree_us,
                    b.wrap_us,
                ],
            }
        })
        .collect();
    (names, rows)
}

fn build_prod_model(
    results: &[(usize, StageTimes, VerifyTimes)],
) -> (Vec<&'static str>, Vec<TableRow>) {
    let names = vec![
        "build_inputs",
        "tree_build",
        "top_extract",
        "eq_tables",
        "round_t",
        "fold",
        "inreg_fold",
        "scalar_tail",
        "round_misc",
    ];
    let rows = results
        .iter()
        .map(|(nv, st, _)| {
            let b = &st.prod_breakdown;
            TableRow {
                nv: *nv,
                total_us: st.product_check_us,
                items: vec![
                    b.build_inputs_us,
                    b.tree_build_us,
                    b.top_extract_us,
                    b.eq_tables_us,
                    b.round_t_us,
                    b.fold_us,
                    b.inreg_fold_us,
                    b.scalar_tail_us,
                    b.round_misc_us,
                ],
            }
        })
        .collect();
    (names, rows)
}

fn build_zero_chk_model(
    results: &[(usize, StageTimes, VerifyTimes)],
) -> (Vec<&'static str>, Vec<TableRow>) {
    let names = vec![
        "eq_tables",
        "precompute",
        "small_val",
        "trans_fold",
        "pack_fold",
        "scalar_tail",
        "round_misc",
    ];
    let rows = results
        .iter()
        .map(|(nv, st, _)| {
            let b = &st.zero_check_breakdown;
            let timed_sum = b.eq_tables_us
                + b.precompute_us
                + b.small_value_rounds_us
                + b.transition_fold_us
                + b.packed_fold_rounds_us
                + b.scalar_tail_us;
            let round_misc_us = st.zero_check_us.saturating_sub(timed_sum);
            TableRow {
                nv: *nv,
                total_us: st.zero_check_us,
                items: vec![
                    b.eq_tables_us,
                    b.precompute_us,
                    b.small_value_rounds_us,
                    b.transition_fold_us,
                    b.packed_fold_rounds_us,
                    b.scalar_tail_us,
                    round_misc_us,
                ],
            }
        })
        .collect();
    (names, rows)
}

fn build_verify_stage_model(
    results: &[(usize, StageTimes, VerifyTimes)],
) -> (Vec<&'static str>, Vec<TableRow>) {
    // Names are short labels matching a legend emitted by the caller.
    let names = vec![
        "setup",
        "sc_chal",
        "zero_chk",
        "c1",
        "prod_chal",
        "prod_chk",
        "c2",
        "rho_iy",
        "final_rsc",
        "c4",
        "pcs_verify",
    ];
    let rows = results
        .iter()
        .map(|(nv, _, v)| TableRow {
            nv: *nv,
            total_us: v.total_us,
            items: vec![
                v.setup_us,
                v.sc_chal_us,
                v.zero_chk_us,
                v.c1_us,
                v.prod_chal_us,
                v.prod_chk_us,
                v.c2_us,
                v.rho_init_y_us,
                v.final_rsc_us,
                v.c4_us,
                v.pcs_verify_us,
            ],
        })
        .collect();
    (names, rows)
}

fn build_pcs_verify_model(
    results: &[(usize, StageTimes, VerifyTimes)],
) -> (Vec<&'static str>, Vec<TableRow>) {
    let names = vec![
        "fold_check",
        "grinding",
        "query_prep",
        "fat_mkl",
        "split_fold",
        "std_fri_mkl",
        "fri_folds",
    ];
    let rows = results
        .iter()
        .map(|(nv, _, v)| {
            let b = &v.pcs_verify_breakdown;
            TableRow {
                nv: *nv,
                total_us: v.pcs_verify_us,
                items: vec![
                    b.fold_check_us,
                    b.grinding_us,
                    b.query_prep_us,
                    b.fat_merkle_us,
                    b.split_fold_us,
                    b.std_fri_merkle_us,
                    b.fri_folds_us,
                ],
            }
        })
        .collect();
    (names, rows)
}

fn print_main_stage_pair(variant: &str, results: &[(usize, StageTimes, VerifyTimes)]) {
    let (names, rows, proofs) = build_main_stage_model(results);
    let layout = build_layout(names, &rows, 1.0);
    print_ms_table(variant, "stage", &layout, &rows, Some(&proofs));
    print_share_table(variant, "stage", &layout, &rows);
    print_layout_legend(&layout);
}

fn print_sub_pair(
    variant: &str,
    subheading: &str,
    (names, rows): (Vec<&'static str>, Vec<TableRow>),
) {
    let layout = build_layout(names, &rows, 1.0);
    print_ms_table(variant, subheading, &layout, &rows, None);
    print_share_table(variant, subheading, &layout, &rows);
    print_layout_legend(&layout);
}

fn print_verify_pair(variant: &str, results: &[(usize, StageTimes, VerifyTimes)]) {
    let (names, rows) = build_verify_stage_model(results);
    let layout = build_layout(names, &rows, 1.0);
    print_ms_table(variant, "verify stage", &layout, &rows, None);
    print_share_table(variant, "verify stage", &layout, &rows);
    print_layout_legend(&layout);
}

fn print_all_tables(variant: &str, results: &[(usize, StageTimes, VerifyTimes)]) {
    print_main_stage_pair(variant, results);
    print_sub_pair(variant, "pcs_open substages", build_open_model(results));
    print_sub_pair(variant, "wpc_new substages", build_wpc_new_model(results));
    print_sub_pair(variant, "prod_chk substages", build_prod_model(results));
    print_sub_pair(variant, "zero_chk substages", build_zero_chk_model(results));
    print_verify_pair(variant, results);
    print_sub_pair(variant, "pcs_verify substages", build_pcs_verify_model(results));
}

pub fn main() {
    let nv_range: Vec<usize> = std::env::var("NV_RANGE")
        .ok()
        .and_then(|s| {
            let parts: Vec<&str> = s.split(',').collect();
            if parts.len() == 2 {
                let lo: usize = parts[0].trim().parse().ok()?;
                let hi: usize = parts[1].trim().parse().ok()?;
                Some((lo..=hi).collect())
            } else {
                None
            }
        })
        .unwrap_or_else(|| (18..=24).collect());
    let reps_table: Vec<(usize, usize)> = vec![(20, 2), (usize::MAX, 1)];

    println!("# HyperPlonk DeepFold MamaBear profile");
    println!("NV range: {:?}", nv_range);
    println!(
        "FRI security: Ext3 -> PROV128 (queries={}, grinding={})",
        QUERY_NUM_PROV_QUERY128, GRINDING_BITS_EXT3_PROV_QUERY128
    );
    println!();
    println!("Verify stage legend:");
    println!("- setup       : commit deserialize + transcript append + DeepFold verifier setup");
    println!("- sc_chal     : ZeroCheck r_zero challenge draws (nv challenges)");
    println!("- zero_chk    : SumcheckMamaBear::verify_ell0 (4*nv scalar Fermat inversions inside)");
    println!("- c1          : C1 check (eq(r_zero, sc_pt) * gate composition) vs zero_y");
    println!("- prod_chal   : ProductCheck setup challenges prod_r0 / prod_r1");
    println!("- prod_chk    : ProdEqCheckMamaBearPerWire::verify (sumcheck verification)");
    println!("- c2          : C2 per-wire claim check (uses eval_identical_mont)");
    println!("- rho_iy      : rho challenge + Horner initial_y for final-reduce");
    println!("- final_rsc   : verify_final_reduce_sumcheck (degree-2 Newton interp, 2 trees * nv rounds)");
    println!("- c4          : C4 composition check (2 x eval_eq_mont + tree composition)");
    println!("- pcs_verify  : DeepFoldMamaBearVerifier::verify (Merkle + FRI fold check)");
    println!();
    println!("pcs_verify substages:");
    println!("- fold_check  : fold consistency loop (challenge + eval-update per round)");
    println!("- grinding    : transcript.verify_grind (BLAKE3 PoW, Ext3 only)");
    println!("- query_prep  : query index derivation (challenge_usizes + sort + dedup)");
    println!("- fat_mkl     : initial fat-leaf Merkle verify + per-prover regroup");
    println!("- split_fold  : local split-fold (rounds 0..split_level) over combined_j/jh");
    println!("- std_fri_mkl : per-round standard-FRI Merkle path verification");
    println!("- fri_folds   : FRI fold + consistency loop (rounds split_level..nv)");
    println!();

    println!("\n## Ext3 (PROV128)");
    let ext3_results = run_variant::<SEF3>("Ext3", &nv_range, &reps_table);
    print_all_tables("Ext3", &ext3_results);

    // Parallel comparison if PARALLEL=1
    if std::env::var("PARALLEL").ok().map_or(false, |v| v == "1") {
        println!("\n## Parallel Comparison");
        print_par_comparison::<SEF3>("Ext3", &nv_range, &reps_table);
    }
}

fn print_par_comparison<F: MamaBearExtConfig>(variant_name: &str, nv_range: &[usize], reps_table: &[(usize, usize)])
where
    F::Packed: Send + Sync,
    F: Send + Sync,
{
    use hyperplonk::prover_mamabear::AlignedPoly;

    println!("\n### {} serial vs parallel (ms)", variant_name);
    println!("| nv | serial | parallel | speedup |");
    println!("| --- | ---: | ---: | ---: |");

    for &nv in nv_range {
        let reps = reps_table
            .iter()
            .find(|(max_nv, _)| nv <= *max_nv)
            .map(|(_, r)| *r)
            .unwrap_or(1);

        let circuit = build_circuit::<F>(nv);
        let (query_num, grinding_bits) = fri_params_for(variant_name);
        let mut pp = DeepFoldMamaBearParam::new_default(nv, CODE_RATE_LOG, query_num);
        pp.grinding_bits = grinding_bits;
        let (pk, _vk) = setup_mamabear::<F>(&circuit, &pp);
        let prover = ProverMamaBear { prover_key: pk };

        let mut serial_us = 0u128;
        let mut par_us = 0u128;
        for rep in 0..reps {
            let raw_witness = canonical_witness(&circuit, nv, 0xC0FFEE_u64.wrapping_add(rep as u64));

            // Serial
            let witness_s = raw_witness.clone().map(|p| AlignedPoly::from_sbf(&p));
            let t0 = Instant::now();
            let proof_s = prover.prove(&pp, nv, witness_s);
            serial_us += t0.elapsed().as_micros();

            // Parallel
            let witness_p = raw_witness.map(|p| AlignedPoly::from_sbf(&p));
            let t0 = Instant::now();
            let proof_p = prover.prove_par(&pp, nv, witness_p);
            par_us += t0.elapsed().as_micros();

            assert_eq!(proof_s.bytes, proof_p.bytes, "Proof mismatch at nv={}", nv);
        }

        let serial_ms = serial_us as f64 / (reps as f64 * 1000.0);
        let par_ms = par_us as f64 / (reps as f64 * 1000.0);
        let speedup = serial_ms / par_ms;
        println!("| {} | {:.2} | {:.2} | {:.2}x |", nv, serial_ms, par_ms, speedup);
    }
}
}
