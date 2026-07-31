//! SHA-256 circuit generator using the [`CircuitBuilder`].
//!
//! Implements the SHA-256 compression function as an arithmetic circuit over
//! MamaBear add/mul gates. All bitwise operations (XOR, AND, NOT, rotations)
//! are expressed via the bit-level gadgets in [`CircuitBuilder`].
//!
//! # Gate counts (per SHA-256 block)
//!
//! | Component                 | Gates     |
//! |--------------------------|-----------|
//! | 64 compression rounds    | ~207,360  |
//! | 48 message schedule      | ~74,112   |
//! | Input decomposition      | ~1,024    |
//! | Final hash addition      | ~2,752    |
//! | **Total**                | **~285K** |

use arithmetic::field::{
    babybear::{BabyBearExt4, BabyBearField},
    goldilocks64::{Goldilocks64, Goldilocks64Ext},
};
// MamaBear is x86_64-only (its `CircuitBuilder::build` is x86-gated), so the
// MamaBear-typed entry point is gated below and these imports are x86-only too
// (`Field` is only named by that entry point's bound). The gate-emission body
// (`sha256_one_block`) is field-agnostic and the Goldilocks/BabyBear builders go
// through `build_generic`, so they reach aarch64 and provide a cross-field
// baseline.
#[cfg(target_arch = "x86_64")]
use arithmetic::field::{mamabear::MamaBearScalar, Field};
use crate::circuit::Circuit;
use crate::circuit_builder::{CircuitBuilder, WireId};

/// SHA-256 round constants K[0..63].
const K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
    0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
    0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
    0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

/// SHA-256 initial hash values H[0..7].
const H_INIT: [u32; 8] = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];

/// A 32-bit word represented as 32 boolean WireIds (LSB first).
type Word32 = [WireId; 32];

/// Build a SHA-256 circuit that computes `num_blocks` independent compression
/// blocks, padded to `2^target_nv` total gates.
///
/// Each block uses a fixed input message (zeroed for benchmarking purposes).
/// The circuit structure (gate count, wiring) is identical regardless of input.
///
/// x86_64-only: `CircuitBuilder::build` needs the non-canonical
/// `MamaBearScalar(v)` constructor. The body is unchanged and stays
/// byte-identical; see the Goldilocks/BabyBear twins below for other fields.
#[cfg(target_arch = "x86_64")]
pub fn build_sha256_circuit<F: Field<BaseField = MamaBearScalar>>(
    num_blocks: usize,
    target_nv: usize,
) -> (Circuit<F>, [Vec<MamaBearScalar>; 3]) {
    let mut builder = CircuitBuilder::new_mamabear();

    for block_idx in 0..num_blocks {
        sha256_one_block(&mut builder, block_idx as u32);
    }

    builder.pad_to_nv(target_nv);
    builder.build()
}

/// Goldilocks variant: same SHA-256 circuit but emitted as
/// `Circuit<Goldilocks64Ext>` with `[Vec<Goldilocks64>; 3]` witness.
pub fn build_sha256_circuit_goldilocks(
    num_blocks: usize,
    target_nv: usize,
) -> (Circuit<Goldilocks64Ext>, [Vec<Goldilocks64>; 3]) {
    let mut builder = CircuitBuilder::new_goldilocks();

    for block_idx in 0..num_blocks {
        sha256_one_block(&mut builder, block_idx as u32);
    }

    builder.pad_to_nv(target_nv);
    builder.build_generic::<Goldilocks64Ext, Goldilocks64>()
}

/// BabyBear variant: same SHA-256 circuit but emitted as
/// `Circuit<BabyBearExt4>` with `[Vec<BabyBearField>; 3]` witness.
pub fn build_sha256_circuit_babybear(
    num_blocks: usize,
    target_nv: usize,
) -> (Circuit<BabyBearExt4>, [Vec<BabyBearField>; 3]) {
    let mut builder = CircuitBuilder::new_babybear();

    for block_idx in 0..num_blocks {
        sha256_one_block(&mut builder, block_idx as u32);
    }

    builder.pad_to_nv(target_nv);
    builder.build_generic::<BabyBearExt4, BabyBearField>()
}

/// Raw gate count for a single SHA-256 compression block (no padding), as
/// emitted by [`sha256_one_block`] on a fresh [`CircuitBuilder`]. Includes the
/// two constant-pool setup gates that [`CircuitBuilder::new`] allocates.
pub fn sha256_gates_per_block() -> usize {
    let mut b = CircuitBuilder::new();
    sha256_one_block(&mut b, 0);
    b.num_gates()
}

/// Compute one SHA-256 compression block inside the circuit builder.
/// Uses `block_seed` to generate a deterministic but non-trivial input message.
fn sha256_one_block(b: &mut CircuitBuilder, block_seed: u32) {
    // 1. Generate 16 input words (deterministic from seed).
    let mut w: Vec<Word32> = Vec::with_capacity(64);
    for i in 0..16u32 {
        // Simple deterministic message: each word depends on seed and index.
        let val = block_seed
            .wrapping_mul(0x9E3779B9)
            .wrapping_add(i.wrapping_mul(0x517CC1B7));
        w.push(b.decompose_u32(val));
    }

    // 2. Message schedule: W[16..63].
    for i in 16..64 {
        // sigma0(W[i-15]) = ROTR(7) XOR ROTR(18) XOR SHR(3)
        let s0 = b.sigma_word32(&w[i - 15], 7, 18, 3, true);
        // sigma1(W[i-2]) = ROTR(17) XOR ROTR(19) XOR SHR(10)
        let s1 = b.sigma_word32(&w[i - 2], 17, 19, 10, true);
        // W[i] = W[i-16] + sigma0(W[i-15]) + W[i-7] + sigma1(W[i-2])
        let t1 = b.add_mod32(&w[i - 16], &s0);
        let t2 = b.add_mod32(&t1, &w[i - 7]);
        let wi = b.add_mod32(&t2, &s1);
        w.push(wi);
    }

    // 3. Initialize working variables from H_INIT.
    let mut state: [Word32; 8] = std::array::from_fn(|i| b.decompose_u32(H_INIT[i]));

    // 4. Compression rounds (64 rounds).
    for i in 0..64 {
        let [a, b_st, c, d, e, f, g, h] = state;

        // Sigma1(e) = ROTR(6) XOR ROTR(11) XOR ROTR(25)
        let sigma1 = b.sigma_word32(&e, 6, 11, 25, false);
        // Ch(e, f, g)
        let ch = b.ch_word32(&e, &f, &g);
        // Sigma0(a) = ROTR(2) XOR ROTR(13) XOR ROTR(22)
        let sigma0 = b.sigma_word32(&a, 2, 13, 22, false);
        // Maj(a, b, c)
        let maj = b.maj_word32(&a, &b_st, &c);

        // K[i] as circuit constant word
        let ki = b.decompose_u32(K[i]);

        // T1 = h + Sigma1 + Ch + K[i] + W[i]
        let t1 = b.add_mod32(&h, &sigma1);
        let t1 = b.add_mod32(&t1, &ch);
        let t1 = b.add_mod32(&t1, &ki);
        let t1 = b.add_mod32(&t1, &w[i]);

        // T2 = Sigma0 + Maj
        let t2 = b.add_mod32(&sigma0, &maj);

        // Update state
        state = [
            b.add_mod32(&t1, &t2), // a = T1 + T2
            a,                       // b = old a
            b_st,                    // c = old b
            c,                       // d = old c
            b.add_mod32(&d, &t1),   // e = d + T1
            e,                       // f = old e
            f,                       // g = old f
            g,                       // h = old g
        ];
    }

    // 5. Add the compressed chunk to the initial hash value.
    let h_init: [Word32; 8] = std::array::from_fn(|i| b.decompose_u32(H_INIT[i]));
    for i in 0..8 {
        let _final_word = b.add_mod32(&state[i], &h_init[i]);
        // The final hash words are computed but not explicitly output-constrained
        // (for benchmarking, we only need the circuit to be well-formed).
    }
}

/// Compute the expected SHA-256 hash of a message block (for verification).
/// Uses the same deterministic message generation as `sha256_one_block`.
pub fn sha256_reference(block_seed: u32) -> [u32; 8] {
    // Generate message words
    let mut w = [0u32; 64];
    for i in 0..16 {
        w[i] = block_seed
            .wrapping_mul(0x9E3779B9)
            .wrapping_add((i as u32).wrapping_mul(0x517CC1B7));
    }
    for i in 16..64 {
        let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
        let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16]
            .wrapping_add(s0)
            .wrapping_add(w[i - 7])
            .wrapping_add(s1);
    }

    let mut h = H_INIT;
    let mut state = h;
    for i in 0..64 {
        let [a, b, c, d, e, f, g, hh] = state;
        let sigma1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
        let ch = (e & f) ^ ((!e) & g);
        let sigma0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
        let maj = (a & b) ^ (a & c) ^ (b & c);
        let t1 = hh
            .wrapping_add(sigma1)
            .wrapping_add(ch)
            .wrapping_add(K[i])
            .wrapping_add(w[i]);
        let t2 = sigma0.wrapping_add(maj);
        state = [
            t1.wrapping_add(t2),
            a,
            b,
            c,
            d.wrapping_add(t1),
            e,
            f,
            g,
        ];
    }
    for i in 0..8 {
        h[i] = h[i].wrapping_add(state[i]);
    }
    h
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

// x86-gated as a whole: the module imports the MamaBear prover/verifier stack
// (`prover_mamabear`, `verifier_mamabear`, `deepfold_mamabear`) at module scope,
// all of which are x86_64-only. On x86 every test below is unchanged.
#[cfg(all(test, target_arch = "x86_64"))]
mod tests {
    use super::*;
    use arithmetic::field::mamabear::MamaBearScalarExt3;
    use crate::prover_mamabear::{setup_mamabear, AlignedPoly, MamaBearExtConfig, ProverMamaBear};
    use crate::verifier_mamabear::VerifierMamaBear;
    use poly_commit::deepfold_mamabear::DeepFoldMamaBearParam;

    #[test]
    fn sha256_reference_sanity() {
        // Quick sanity check that the reference implementation runs.
        let h = sha256_reference(0);
        // Just check it's not all zeros.
        assert!(h.iter().any(|&v| v != 0));
    }

    #[test]
    fn sha256_circuit_gate_count() {
        let mut b = CircuitBuilder::new();
        sha256_one_block(&mut b, 0);
        let n = b.num_gates();
        eprintln!("SHA-256 one block: {} gates", n);
        // Should be roughly 250K-350K gates.
        assert!(n > 100_000, "Too few gates: {}", n);
        assert!(n < 500_000, "Too many gates: {}", n);
    }

    /// Helper: run one SHA-256 block through the serial prover and verifier.
    fn sha256_mamabear_serial<F: MamaBearExtConfig>() {
        let target_nv = 20usize;
        let (circuit, witness) = build_sha256_circuit::<F>(1, target_nv);

        let pp = DeepFoldMamaBearParam::new_default(target_nv, 3, 34);
        let (pk, vk) = setup_mamabear::<F>(&circuit, &pp);
        let prover = ProverMamaBear { prover_key: pk };
        let verifier = VerifierMamaBear { verifier_key: vk };

        let witness_ap = [
            AlignedPoly::from_sbf(&witness[0]),
            AlignedPoly::from_sbf(&witness[1]),
            AlignedPoly::from_sbf(&witness[2]),
        ];
        let proof = prover.prove(&pp, target_nv, witness_ap);
        assert!(verifier.verify(&pp, target_nv, proof));
    }

    /// Build a SHA-256 circuit and witness whose per-block input messages
    /// are offset by `seed`. Uses the same `num_blocks` / `pad_to_nv`
    /// recipe as [`build_sha256_circuit`] but feeds each block a distinct
    /// `block_seed = seed * num_blocks + block_idx` so two different
    /// `seed` values produce genuinely different circuits (different
    /// input-word constants baked into the compression gates).
    fn build_sha256_circuit_seeded<F: Field<BaseField = MamaBearScalar>>(
        num_blocks: usize,
        target_nv: usize,
        seed: u32,
    ) -> (Circuit<F>, [Vec<MamaBearScalar>; 3]) {
        let mut builder = CircuitBuilder::new_mamabear();
        for block_idx in 0..num_blocks {
            let block_seed = seed
                .wrapping_mul(num_blocks as u32)
                .wrapping_add(block_idx as u32);
            sha256_one_block(&mut builder, block_seed);
        }
        builder.pad_to_nv(target_nv);
        builder.build()
    }

    /// Cross-check `{serial, parallel}` prover x `{verify, verify_par}`
    /// verifier on the SHA-256 circuit, sweeping over a range of `nv` and
    /// multiple circuit seeds per nv.
    ///
    /// At each `(nv, seed)` we check four invariants:
    /// 1. Parallel prover emits byte-identical proof as the serial prover.
    /// 2. Both proofs are accepted by the serial verifier (`verify`).
    /// 3. Both proofs are accepted by the parallel verifier (`verify_par`),
    ///    i.e. `verify_par` agrees with `verify` on valid proofs.
    /// 4. A proof mutated inside the Merkle-path region MUST be rejected
    ///    by both verifiers, i.e. `verify_par` also agrees with `verify`
    ///    on invalid proofs (soundness of the parallel Merkle-path
    ///    verification path).
    ///
    /// `num_blocks` at each `nv` is computed from `SHA256_GATES_PER_BLOCK`
    /// the same way `hp_df_mamabear_sha256.rs` does, so the circuit size
    /// actually scales with `nv` rather than always being a single block
    /// padded to `2^nv`.
    fn sha256_mamabear_par<F: MamaBearExtConfig>()
    where
        F::Packed: Send + Sync,
        F: Send + Sync,
    {
        /// Matches `hp_df_mamabear_sha256.rs::SHA256_GATES_PER_BLOCK`.
        const SHA256_GATES_PER_BLOCK: usize = 302_000;
        const NV_MIN: usize = 18;
        const NV_MAX: usize = 22;
        const SEEDS_PER_NV: u32 = 5;

        fn min_nv_for(gates: usize) -> usize {
            assert!(gates > 0);
            (usize::BITS - (gates - 1).leading_zeros()) as usize
        }

        let min_nv = min_nv_for(SHA256_GATES_PER_BLOCK);

        for target_nv in NV_MIN..=NV_MAX {
            if target_nv < min_nv {
                eprintln!(
                    "[sha256_mamabear_par] skip nv={target_nv}: SHA-256 needs >= 2^{min_nv} gates for 1 block"
                );
                continue;
            }
            let total_gates = 1usize << target_nv;
            let num_blocks = (total_gates / SHA256_GATES_PER_BLOCK).max(1);

            for seed in 0..SEEDS_PER_NV {
                let (circuit, witness) =
                    build_sha256_circuit_seeded::<F>(num_blocks, target_nv, seed);

                let pp = DeepFoldMamaBearParam::new_default(target_nv, 3, 34);
                let (pk, vk) = setup_mamabear::<F>(&circuit, &pp);
                let prover = ProverMamaBear { prover_key: pk };
                let verifier = VerifierMamaBear { verifier_key: vk };

                let witness_ap_serial = [
                    AlignedPoly::from_sbf(&witness[0]),
                    AlignedPoly::from_sbf(&witness[1]),
                    AlignedPoly::from_sbf(&witness[2]),
                ];
                let witness_ap_par = [
                    AlignedPoly::from_sbf(&witness[0]),
                    AlignedPoly::from_sbf(&witness[1]),
                    AlignedPoly::from_sbf(&witness[2]),
                ];

                let proof_serial = prover.prove(&pp, target_nv, witness_ap_serial);
                let proof_par = prover.prove_par(&pp, target_nv, witness_ap_par);

                // Invariant 1: prover determinism across serial and parallel.
                assert_eq!(
                    proof_serial.bytes, proof_par.bytes,
                    "serial vs parallel proof mismatch at nv={} seed={} blocks={}",
                    target_nv, seed, num_blocks
                );

                // Invariants 2+3: all four {serial, parallel} x
                // {verify, verify_par} combinations must accept the valid
                // proof.
                let case = format!("nv={} seed={} blocks={}", target_nv, seed, num_blocks);
                assert!(
                    verifier.verify(&pp, target_nv, proof_serial.clone()),
                    "serial verify rejected serial proof at {}",
                    case
                );
                assert!(
                    verifier.verify(&pp, target_nv, proof_par.clone()),
                    "serial verify rejected parallel proof at {}",
                    case
                );
                assert!(
                    verifier.verify_par(&pp, target_nv, proof_serial.clone()),
                    "parallel verify rejected serial proof at {}",
                    case
                );
                assert!(
                    verifier.verify_par(&pp, target_nv, proof_par.clone()),
                    "parallel verify rejected parallel proof at {}",
                    case
                );

                // Invariant 4 (soundness): a proof with a flipped byte
                // inside the Merkle-path region MUST be rejected by both
                // verifiers. We flip at `bytes.len() / 2`, which lands
                // inside the PCS open() payload rather than the
                // circuit-commit header. Using `std::panic::catch_unwind`
                // because the DeepFold verifier signals invalid Merkle
                // proofs via `assert!` panics inside
                // `QueryResultMB::verify_merkle_tree` /
                // `MerkleTreeVerifierMB::verify` rather than a `false`
                // return. "Not accepted" covers both `false` and panic.
                let mut flipped = proof_par.clone();
                let flip_idx = flipped.bytes.len() / 2;
                flipped.bytes[flip_idx] ^= 0xAA;

                let pp_ref = &pp;
                let verifier_ref = &verifier;
                let flipped_s = flipped.clone();
                let flipped_p = flipped.clone();
                let serial_rejects =
                    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        verifier_ref.verify(pp_ref, target_nv, flipped_s)
                    }))
                    .map(|ok| !ok)
                    .unwrap_or(true);
                let par_rejects = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    verifier_ref.verify_par(pp_ref, target_nv, flipped_p)
                }))
                .map(|ok| !ok)
                .unwrap_or(true);
                assert!(
                    serial_rejects,
                    "serial verify unexpectedly accepted tampered proof at {}",
                    case
                );
                assert!(
                    par_rejects,
                    "parallel verify unexpectedly accepted tampered proof at {} (soundness bug)",
                    case
                );
            }
        }
    }

    #[test]
    fn sha256_mamabear_ext3_proves_and_verifies() {
        sha256_mamabear_serial::<MamaBearScalarExt3>();
    }

    #[test]
    fn sha256_mamabear_ext3_par_proves_and_verifies() {
        sha256_mamabear_par::<MamaBearScalarExt3>();
    }

    /// In-process micro-bench: compare `verify` vs `verify_par` median
    /// latency on a real SHA-256 circuit, at each (query_num, nv) combination.
    ///
    /// Why this test exists: rayon's `par_iter` pays ~10-20 us of scheduling
    /// overhead per dispatch, which can dominate when the task count is low
    /// or per-task work is small ("naive parallelism" regression). For the
    /// verify path we want to make sure that stays benign across realistic
    /// configurations. This test alternates `verify` / `verify_par` calls in
    /// the same process (so rayon's thread pool is warm for both timings)
    /// and reports the median over `BENCH_SAMPLES` samples.
    ///
    /// Run:
    /// ```bash
    /// RUSTFLAGS="-C target-cpu=native" cargo test -p hyperplonk --lib --release \
    ///     sha256_circuit::tests::microbench_sha256_verify_serial_vs_par \
    ///     -- --ignored --nocapture
    /// ```
    ///
    /// Env vars:
    /// - `BENCH_NV_MIN` / `BENCH_NV_MAX`: nv range (default 19..=22, the
    ///   SHA-256 circuit needs >= 19 to fit 1 block).
    /// - `BENCH_WARMUP`: warmup iterations for the rayon pool (default 3).
    /// - `BENCH_SAMPLES`: measured samples per cell (default 9).
    #[test]
    #[ignore]
    fn microbench_sha256_verify_serial_vs_par() {
        use std::time::Instant;

        fn env_usize(key: &str, default: usize) -> usize {
            std::env::var(key)
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(default)
        }

        fn min_nv_for(gates: usize) -> usize {
            assert!(gates > 0);
            (usize::BITS - (gates - 1).leading_zeros()) as usize
        }

        fn run_variant<F: MamaBearExtConfig>(
            variant_name: &str,
            query_num: usize,
            grinding_bits: u32,
            nv_range: &[usize],
            warmup: usize,
            samples: usize,
        ) where
            F::Packed: Send + Sync,
            F: Send + Sync,
        {
            let gates_per_block = sha256_gates_per_block();
            let min_nv = min_nv_for(gates_per_block);

            println!(
                "\n## {} (queries={}, grinding={})",
                variant_name, query_num, grinding_bits
            );
            println!("| nv | blocks | serial ms | parallel ms | par/ser | winner |");
            println!("| --- | ---: | ---: | ---: | ---: | --- |");

            for &nv in nv_range {
                if nv < min_nv {
                    eprintln!(
                        "[microbench] skip nv={nv}: SHA-256 needs >= 2^{min_nv} gates for 1 block"
                    );
                    continue;
                }
                let num_blocks = ((1usize << nv) / gates_per_block).max(1);
                let (circuit, witness_raw) = build_sha256_circuit::<F>(num_blocks, nv);
                let mut pp = DeepFoldMamaBearParam::new_default(nv, 3, query_num);
                pp.grinding_bits = grinding_bits;
                let (pk, vk) = setup_mamabear::<F>(&circuit, &pp);
                let prover = ProverMamaBear { prover_key: pk };
                let verifier = VerifierMamaBear { verifier_key: vk };

                let witness = [
                    AlignedPoly::from_sbf(&witness_raw[0]),
                    AlignedPoly::from_sbf(&witness_raw[1]),
                    AlignedPoly::from_sbf(&witness_raw[2]),
                ];
                let proof = prover.prove_par(&pp, nv, witness);

                // Warm rayon thread pool + icache.
                for _ in 0..warmup {
                    assert!(verifier.verify_par(&pp, nv, proof.clone()));
                }

                // Alternate serial / parallel samples so any residual cache
                // state affects both equally.
                let mut s_us: Vec<u128> = Vec::with_capacity(samples);
                let mut p_us: Vec<u128> = Vec::with_capacity(samples);
                for _ in 0..samples {
                    let p1 = proof.clone();
                    let t0 = Instant::now();
                    let ok = verifier.verify(&pp, nv, p1);
                    s_us.push(t0.elapsed().as_micros());
                    assert!(ok);

                    let p2 = proof.clone();
                    let t0 = Instant::now();
                    let ok = verifier.verify_par(&pp, nv, p2);
                    p_us.push(t0.elapsed().as_micros());
                    assert!(ok);
                }
                s_us.sort();
                p_us.sort();
                let s_med = s_us[samples / 2] as f64 / 1000.0;
                let p_med = p_us[samples / 2] as f64 / 1000.0;
                let ratio = p_med / s_med;
                let winner = if p_med < s_med * 0.95 {
                    "par"
                } else if p_med > s_med * 1.05 {
                    "serial"
                } else {
                    "tie"
                };
                println!(
                    "| {} | {} | {:.2} | {:.2} | {:.2}x | {} |",
                    nv, num_blocks, s_med, p_med, ratio, winner
                );
            }
        }

        let nv_min = env_usize("BENCH_NV_MIN", 19);
        let nv_max = env_usize("BENCH_NV_MAX", 22);
        let warmup = env_usize("BENCH_WARMUP", 3);
        let samples = env_usize("BENCH_SAMPLES", 9);
        let nv_range: Vec<usize> = (nv_min..=nv_max).collect();

        println!(
            "# SHA-256 verify vs verify_par (same process, warm rayon pool, median of {} samples after {} warmups)",
            samples, warmup
        );
        println!("NV range: {:?}", nv_range);

        // Ext3: PROV128.
        run_variant::<MamaBearScalarExt3>(
            "Ext3 PROV128",
            88,
            16,
            &nv_range,
            warmup,
            samples,
        );
    }

    #[test]
    fn sha256_circuit_goldilocks_builds() {
        // Just ensure the Goldilocks variant produces a well-formed circuit.
        let target_nv = 20;
        let (circuit, witness) = build_sha256_circuit_goldilocks(1, target_nv);
        assert_eq!(circuit.selector.len(), 1 << target_nv);
        assert_eq!(witness[0].len(), 1 << target_nv);
    }

    #[test]
    fn sha256_circuit_goldilocks_proves_and_verifies() {
        use arithmetic::{
            field::goldilocks64::{Goldilocks64, Goldilocks64Ext},
            mul_group::Radix2Group,
        };
        use crate::{prover::Prover, verifier::Verifier};
        use poly_commit::deepfold::{DeepFoldParam, DeepFoldProver, DeepFoldVerifier};

        let target_nv = 20usize;
        let (circuit, witness) = build_sha256_circuit_goldilocks(1, target_nv);

        let mut mult_subgroups = vec![Radix2Group::<Goldilocks64>::new((target_nv + 2) as u32)];
        for i in 1..target_nv {
            mult_subgroups.push(mult_subgroups[i - 1].exp(2));
        }
        let pp = DeepFoldParam::<Goldilocks64Ext> {
            mult_subgroups,
            variable_num: target_nv,
            query_num: 45,
        };
        let (pk, vk) = circuit.setup::<DeepFoldProver<_>, DeepFoldVerifier<_>>(&pp, &pp);
        let prover = Prover { prover_key: pk };
        let verifier = Verifier { verifier_key: vk };
        let proof = prover.prove(&pp, target_nv, witness);
        assert!(verifier.verify(&pp, target_nv, proof));
    }

    #[test]
    fn sha256_circuit_babybear_builds() {
        let target_nv = 20;
        let (circuit, witness) = build_sha256_circuit_babybear(1, target_nv);
        assert_eq!(circuit.selector.len(), 1 << target_nv);
        assert_eq!(witness[0].len(), 1 << target_nv);
    }

    #[test]
    fn sha256_circuit_babybear_proves_and_verifies() {
        use arithmetic::{
            field::babybear::{BabyBearExt4, BabyBearField},
            mul_group::Radix2Group,
        };
        use crate::{prover::Prover, verifier::Verifier};
        use poly_commit::deepfold::{DeepFoldParam, DeepFoldProver, DeepFoldVerifier};

        let target_nv = 20usize;
        let (circuit, witness) = build_sha256_circuit_babybear(1, target_nv);

        let mut mult_subgroups = vec![Radix2Group::<BabyBearField>::new((target_nv + 2) as u32)];
        for i in 1..target_nv {
            mult_subgroups.push(mult_subgroups[i - 1].exp(2));
        }
        let pp = DeepFoldParam::<BabyBearExt4> {
            mult_subgroups,
            variable_num: target_nv,
            query_num: 45,
        };
        let (pk, vk) = circuit.setup::<DeepFoldProver<_>, DeepFoldVerifier<_>>(&pp, &pp);
        let prover = Prover { prover_key: pk };
        let verifier = Verifier { verifier_key: vk };
        let proof = prover.prove(&pp, target_nv, witness);
        assert!(verifier.verify(&pp, target_nv, proof));
    }
}

// Deliberately NOT x86-gated (unlike the typed test module above): this touches
// no field type, and the claim it protects is what lets an x86-only
// `circuit_size` run stand in for any cross-architecture build.
#[cfg(test)]
mod prime_invariance_pin {
    use super::*;
    use crate::circuit_builder::{BABYBEAR_P, GOLDILOCKS_P, MAMABEAR_P};

    /// A bit-level circuit's gate count does not depend on the prime.
    ///
    /// This is the load-bearing claim that lets an x86-only `circuit_size` run
    /// (which builds with `CircuitBuilder::new()` = MamaBear) report gate counts
    /// that are valid for any target field. That holds here because SHA-256's
    /// constants are spec-fixed, so the builder's `const_pool` dedups identically
    /// under any prime.
    ///
    /// It does NOT hold universally -- Poseidon2 derives its constants from the
    /// field and the gate count can vary with the diagonal draw (see
    /// `poseidon2_circuit::poseidon2_gate_counts_are_field_invariant`).
    /// So this pin is what separates "reuse is sound" from "reuse is assumed".
    #[test]
    fn sha256_gate_count_is_prime_invariant() {
        let count = |p| {
            let mut b = CircuitBuilder::new_with_prime(p);
            sha256_one_block(&mut b, 0);
            b.num_gates()
        };
        let mama = count(MAMABEAR_P);
        assert_eq!(mama, 301_634, "the committed results/ row");
        for (label, p) in [
            ("goldilocks", GOLDILOCKS_P),
            ("babybear", BABYBEAR_P),
        ] {
            assert_eq!(count(p), mama, "sha256 gate count must not depend on the prime ({label})");
        }
    }
}
