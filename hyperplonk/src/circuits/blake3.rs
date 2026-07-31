//! BLAKE3 permutation circuit generator using the [`CircuitBuilder`].
//!
//! Implements the BLAKE3 7-round permutation as an arithmetic circuit over
//! add/mul gates. All bitwise operations (XOR, rotations) are expressed via
//! the bit-level gadgets in [`CircuitBuilder`]. This mirrors
//! [`sha256_circuit`](crate::sha256_circuit) in structure and is the HyperPlonk
//! counterpart of plonky3's `p3-blake3-air::Blake3Air` (7-round permutation,
//! no feed-forward XOR) so the two benchmarks share a directly comparable
//! "one BLAKE3 permutation" unit of work.
//!
//! # Gate counts (per BLAKE3 permutation, measured via
//! `blake3_circuit_gate_count` test)
//!
//! | Component                       | Ops                   |
//! |---------------------------------|-----------------------|
//! | 7 rounds x 8 G x 6 ADDs mod 2^32| 336 x 32-bit ADDs     |
//! | 7 rounds x 8 G x 4 XORs         | 224 x 32-bit XORs     |
//! | Message permutation / rotations | free (wire rewiring)  |
//! | 16 message-word decomposition   | 16 x 32 boolean wires |
//! | 16 state-word decomposition     | 16 x 32 boolean wires |
//!
//! Message schedule is a static wire permutation and therefore costs zero
//! gates; this is the main reason a BLAKE3 permutation is cheaper per call
//! than a SHA-256 compression block of the same round count per G.

use arithmetic::field::{
    babybear::{BabyBearExt4, BabyBearField},
    goldilocks64::{Goldilocks64, Goldilocks64Ext},
};
// MamaBear is x86_64-only (its `CircuitBuilder::build` is x86-gated), so the
// MamaBear-typed entry point is gated below and these imports are x86-only too
// (`Field` is only named by that entry point's bound). The gate-emission body
// (`blake3_one_permutation`) is field-agnostic and the Goldilocks/BabyBear
// builders go through `build_generic`, so they reach aarch64 and provide a
// cross-field baseline.
#[cfg(target_arch = "x86_64")]
use arithmetic::field::{mamabear::MamaBearScalar, Field};
use crate::circuit::Circuit;
use crate::circuit_builder::{CircuitBuilder, WireId};

/// BLAKE3 initial hash values (identical to the SHA-256 / BLAKE2s IV).
const IV: [u32; 8] = [
    0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
    0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19,
];

/// BLAKE3 message-word permutation applied between consecutive rounds.
const MSG_PERMUTATION: [usize; 16] =
    [2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8];

/// A 32-bit word represented as 32 boolean WireIds (LSB first).
type Word32 = [WireId; 32];

/// Build a BLAKE3 circuit that executes `num_perms` independent BLAKE3
/// permutations (7 rounds each), padded to `2^target_nv` total gates.
///
/// Each permutation uses a deterministic (but non-trivial) initial state
/// derived from its index. The circuit structure (gate count, wiring) is
/// identical regardless of input.
///
/// x86_64-only: `CircuitBuilder::build` needs the non-canonical
/// `MamaBearScalar(v)` constructor. The body is unchanged and stays
/// byte-identical; see the Goldilocks/BabyBear twins below for other fields.
#[cfg(target_arch = "x86_64")]
pub fn build_blake3_circuit<F: Field<BaseField = MamaBearScalar>>(
    num_perms: usize,
    target_nv: usize,
) -> (Circuit<F>, [Vec<MamaBearScalar>; 3]) {
    let mut builder = CircuitBuilder::new_mamabear();

    for perm_idx in 0..num_perms {
        blake3_one_permutation(&mut builder, perm_idx as u32);
    }

    builder.pad_to_nv(target_nv);
    builder.build()
}

/// Goldilocks variant: same BLAKE3 circuit emitted as
/// `Circuit<Goldilocks64Ext>` with `[Vec<Goldilocks64>; 3]` witness.
pub fn build_blake3_circuit_goldilocks(
    num_perms: usize,
    target_nv: usize,
) -> (Circuit<Goldilocks64Ext>, [Vec<Goldilocks64>; 3]) {
    let mut builder = CircuitBuilder::new_goldilocks();

    for perm_idx in 0..num_perms {
        blake3_one_permutation(&mut builder, perm_idx as u32);
    }

    builder.pad_to_nv(target_nv);
    builder.build_generic::<Goldilocks64Ext, Goldilocks64>()
}

/// BabyBear variant: same BLAKE3 circuit emitted as
/// `Circuit<BabyBearExt4>` with `[Vec<BabyBearField>; 3]` witness.
pub fn build_blake3_circuit_babybear(
    num_perms: usize,
    target_nv: usize,
) -> (Circuit<BabyBearExt4>, [Vec<BabyBearField>; 3]) {
    let mut builder = CircuitBuilder::new_babybear();

    for perm_idx in 0..num_perms {
        blake3_one_permutation(&mut builder, perm_idx as u32);
    }

    builder.pad_to_nv(target_nv);
    builder.build_generic::<BabyBearExt4, BabyBearField>()
}

/// Deterministic 16-word message derived from `seed`. Same LCG pattern as
/// [`crate::sha256_circuit::sha256_one_block`] so numbers are non-trivial but
/// reproducible.
#[inline]
fn derive_message(seed: u32) -> [u32; 16] {
    let mut m = [0u32; 16];
    for i in 0..16u32 {
        m[i as usize] = seed
            .wrapping_mul(0x9E3779B9)
            .wrapping_add(i.wrapping_mul(0x517CC1B7));
    }
    m
}

/// Deterministic initial state: `h[0..8] || IV[0..4] || t_lo || t_hi ||
/// block_len || flags`. Values are arbitrary for benchmark purposes since the
/// circuit structure does not depend on them.
#[inline]
fn derive_initial_state(seed: u32) -> [u32; 16] {
    // Chaining value: deterministic from seed, disjoint from message LCG.
    let h: [u32; 8] = std::array::from_fn(|i| {
        seed.wrapping_mul(0xC2B2AE35)
            .wrapping_add((i as u32).wrapping_mul(0x27D4EB2F))
    });
    // t = seed as 64-bit counter, block_len = 64, flags = CHUNK_START |
    // CHUNK_END | ROOT = 0x0B. These specific values never affect gate count.
    let t_lo = seed;
    let t_hi = 0u32;
    let block_len = 64u32;
    let flags = 0x0Bu32;
    [
        h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7],
        IV[0], IV[1], IV[2], IV[3],
        t_lo, t_hi, block_len, flags,
    ]
}

/// Apply [`MSG_PERMUTATION`] to a 16-word message. Zero-gate: pure wire
/// relabelling since `Word32` is `Copy`.
#[inline]
fn permute_message(m: &[Word32; 16]) -> [Word32; 16] {
    std::array::from_fn(|i| m[MSG_PERMUTATION[i]])
}

/// The BLAKE3 `G` quarter-round on four 32-bit state slots, mixing in two
/// message words. Operations per call: 6 x `add_mod32`, 4 x `xor_word32`,
/// 4 x `rotr32` (free).
#[allow(clippy::too_many_arguments)]
fn g(
    b: &mut CircuitBuilder,
    st: &mut [Word32; 16],
    ai: usize, bi: usize, ci: usize, di: usize,
    mx: &Word32, my: &Word32,
) {
    let mut a = st[ai];
    let mut bw = st[bi];
    let mut c = st[ci];
    let mut d = st[di];

    // a = a + b + mx
    a = b.add_mod32(&a, &bw);
    a = b.add_mod32(&a, mx);
    // d = (d XOR a) >>> 16
    let dx = b.xor_word32(&d, &a);
    d = b.rotr32(&dx, 16);
    // c = c + d
    c = b.add_mod32(&c, &d);
    // b = (b XOR c) >>> 12
    let bx = b.xor_word32(&bw, &c);
    bw = b.rotr32(&bx, 12);

    // a = a + b + my
    a = b.add_mod32(&a, &bw);
    a = b.add_mod32(&a, my);
    // d = (d XOR a) >>> 8
    let dx2 = b.xor_word32(&d, &a);
    d = b.rotr32(&dx2, 8);
    // c = c + d
    c = b.add_mod32(&c, &d);
    // b = (b XOR c) >>> 7
    let bx2 = b.xor_word32(&bw, &c);
    bw = b.rotr32(&bx2, 7);

    st[ai] = a;
    st[bi] = bw;
    st[ci] = c;
    st[di] = d;
}

/// Raw gate count for a single BLAKE3 permutation (no padding), as emitted
/// by [`blake3_one_permutation`] on a fresh [`CircuitBuilder`]. Includes the
/// two constant-pool setup gates that [`CircuitBuilder::new`] allocates.
pub fn blake3_gates_per_permutation() -> usize {
    let mut b = CircuitBuilder::new();
    blake3_one_permutation(&mut b, 0);
    b.num_gates()
}

/// Emit one BLAKE3 permutation (7 rounds, no feed-forward) into the builder.
/// Matches the trace semantics of plonky3's `p3-blake3-air::Blake3Air`.
fn blake3_one_permutation(b: &mut CircuitBuilder, seed: u32) {
    // 1. Decompose 16 message words (deterministic from seed).
    let msg_vals = derive_message(seed);
    let mut m: [Word32; 16] = std::array::from_fn(|i| b.decompose_u32(msg_vals[i]));

    // 2. Decompose initial 16-word state.
    let state_vals = derive_initial_state(seed);
    let mut state: [Word32; 16] = std::array::from_fn(|i| b.decompose_u32(state_vals[i]));

    // 3. Seven full rounds (Blake3Air::generate_trace_rows_for_perm).
    for round in 0..7 {
        // Column step: four G's acting on columns of the 4x4 state.
        g(b, &mut state, 0, 4,  8, 12, &m[0],  &m[1]);
        g(b, &mut state, 1, 5,  9, 13, &m[2],  &m[3]);
        g(b, &mut state, 2, 6, 10, 14, &m[4],  &m[5]);
        g(b, &mut state, 3, 7, 11, 15, &m[6],  &m[7]);
        // Diagonal step: four G's acting on diagonals.
        g(b, &mut state, 0, 5, 10, 15, &m[8],  &m[9]);
        g(b, &mut state, 1, 6, 11, 12, &m[10], &m[11]);
        g(b, &mut state, 2, 7,  8, 13, &m[12], &m[13]);
        g(b, &mut state, 3, 4,  9, 14, &m[14], &m[15]);

        // Apply MSG_PERMUTATION for the next round (BLAKE3 spec). No-op after
        // the last round since its result is unused.
        if round < 6 {
            m = permute_message(&m);
        }
    }

    // Final state held in `state` (not output-constrained). No feed-forward,
    // matching plonky3's Blake3Air.
    let _ = state;
}

/// Native reference: compute one BLAKE3 permutation in plain `u32` arithmetic.
/// Used by tests to cross-check the circuit semantics.
pub fn blake3_reference(seed: u32) -> [u32; 16] {
    let mut state = derive_initial_state(seed);
    let mut m = derive_message(seed);

    for round in 0..7 {
        // Columns
        ref_g(&mut state, 0, 4,  8, 12, m[0],  m[1]);
        ref_g(&mut state, 1, 5,  9, 13, m[2],  m[3]);
        ref_g(&mut state, 2, 6, 10, 14, m[4],  m[5]);
        ref_g(&mut state, 3, 7, 11, 15, m[6],  m[7]);
        // Diagonals
        ref_g(&mut state, 0, 5, 10, 15, m[8],  m[9]);
        ref_g(&mut state, 1, 6, 11, 12, m[10], m[11]);
        ref_g(&mut state, 2, 7,  8, 13, m[12], m[13]);
        ref_g(&mut state, 3, 4,  9, 14, m[14], m[15]);

        if round < 6 {
            let permuted: [u32; 16] = std::array::from_fn(|i| m[MSG_PERMUTATION[i]]);
            m = permuted;
        }
    }
    state
}

#[inline]
fn ref_g(st: &mut [u32; 16], ai: usize, bi: usize, ci: usize, di: usize, mx: u32, my: u32) {
    st[ai] = st[ai].wrapping_add(st[bi]).wrapping_add(mx);
    st[di] = (st[di] ^ st[ai]).rotate_right(16);
    st[ci] = st[ci].wrapping_add(st[di]);
    st[bi] = (st[bi] ^ st[ci]).rotate_right(12);
    st[ai] = st[ai].wrapping_add(st[bi]).wrapping_add(my);
    st[di] = (st[di] ^ st[ai]).rotate_right(8);
    st[ci] = st[ci].wrapping_add(st[di]);
    st[bi] = (st[bi] ^ st[ci]).rotate_right(7);
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
    fn blake3_reference_sanity() {
        let s = blake3_reference(0);
        assert!(s.iter().any(|&v| v != 0));
        // Determinism.
        assert_eq!(s, blake3_reference(0));
        assert_ne!(s, blake3_reference(1));
    }

    /// Builds one BLAKE3 permutation and prints the real gate count. The
    /// bench files consume this number as `BLAKE3_GATES_PER_PERMUTATION`.
    #[test]
    fn blake3_circuit_gate_count() {
        let mut b = CircuitBuilder::new();
        blake3_one_permutation(&mut b, 0);
        let n = b.num_gates();
        eprintln!("BLAKE3 one permutation: {} gates", n);
        // Loose bound: the real number is expected in the 80K-250K range.
        assert!(n > 50_000, "Too few gates: {}", n);
        assert!(n < 400_000, "Too many gates: {}", n);
    }

    /// Helper: run one BLAKE3 permutation through the serial prover / verifier.
    fn blake3_mamabear_serial<F: MamaBearExtConfig>() {
        let target_nv = 20usize;
        let (circuit, witness) = build_blake3_circuit::<F>(1, target_nv);

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

    /// Helper: run one BLAKE3 permutation through the parallel prover and
    /// verifier, and also check parallel proof bytes match the serial ones.
    fn blake3_mamabear_par<F: MamaBearExtConfig>()
    where
        F::Packed: Send + Sync,
        F: Send + Sync,
    {
        let target_nv = 20usize;
        let (circuit, witness) = build_blake3_circuit::<F>(1, target_nv);

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

        assert!(verifier.verify(&pp, target_nv, proof_serial.clone()));
        assert!(verifier.verify(&pp, target_nv, proof_par.clone()));

        assert_eq!(
            proof_serial.bytes, proof_par.bytes,
            "serial vs parallel proof mismatch"
        );
    }

    #[test]
    fn blake3_mamabear_ext3_proves_and_verifies() {
        blake3_mamabear_serial::<MamaBearScalarExt3>();
    }

    #[test]
    fn blake3_mamabear_ext3_par_proves_and_verifies() {
        blake3_mamabear_par::<MamaBearScalarExt3>();
    }
}
