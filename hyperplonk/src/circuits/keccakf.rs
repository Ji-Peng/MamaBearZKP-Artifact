//! Keccak-f[1600] permutation circuit generator using the
//! [`CircuitBuilder`].
//!
//! Implements the FIPS 202 Keccak-f[1600] permutation (24 rounds over a
//! 5x5x64-bit state) as an arithmetic circuit over the HyperPlonk add/mul
//! gate. Matches the specification in NIST FIPS 202 / the Keccak reference
//! documents, and is field-agnostic (same circuit compiles over MamaBear /
//! BabyBear / Goldilocks).
//!
//! # Gate counts (per Keccak-f[1600] permutation, measured via
//! `keccakf_circuit_gate_count` test)
//!
//! | Component              | Bit ops / round  | Notes                 |
//! |------------------------|:----------------:|-----------------------|
//! | theta (parity + xor)   | ~3200 XORs       | dominated by applying D |
//! | rho + pi               | 0 gates          | free (index remap)    |
//! | chi                    | 1600 bits x 7    | AND + NOT + XOR per bit |
//! | iota (round constant)  | 64 XORs          | lane 0                |
//!
//! Expected total: ~550-700K gates per permutation, which makes Keccak-f
//! the largest of the currently-supported circuits (needs `nv >= 20` to
//! fit one permutation).

use arithmetic::field::{
    babybear::{BabyBearExt4, BabyBearField},
    goldilocks64::{Goldilocks64, Goldilocks64Ext},
    mamabear::MamaBearScalar,
    Field,
};
use crate::circuit::Circuit;
use crate::circuit_builder::{CircuitBuilder, WireId};

pub const KECCAK_ROUNDS: usize = 24;

/// Keccak rotation offsets r[x][y]; state is indexed as `state[x][y]`.
/// Access as `KECCAK_RHO_OFFSETS[x][y]`. Values follow FIPS 202 Table 2.
pub const KECCAK_RHO_OFFSETS: [[u32; 5]; 5] = [
    // y = 0       y = 1        y = 2      y = 3      y = 4
    [   0,          36,          3,         41,        18],  // x = 0
    [   1,          44,         10,         45,         2],  // x = 1
    [  62,           6,         43,         15,        61],  // x = 2
    [  28,          55,         25,         21,        56],  // x = 3
    [  27,          20,         39,          8,        14],  // x = 4
];

/// FIPS 202 Keccak-f[1600] round constants.
pub const KECCAK_ROUND_CONSTANTS: [u64; KECCAK_ROUNDS] = [
    0x0000_0000_0000_0001, 0x0000_0000_0000_8082, 0x8000_0000_0000_808A,
    0x8000_0000_8000_8000, 0x0000_0000_0000_808B, 0x0000_0000_8000_0001,
    0x8000_0000_8000_8081, 0x8000_0000_0000_8009, 0x0000_0000_0000_008A,
    0x0000_0000_0000_0088, 0x0000_0000_8000_8009, 0x0000_0000_8000_000A,
    0x0000_0000_8000_808B, 0x8000_0000_0000_008B, 0x8000_0000_0000_8089,
    0x8000_0000_0000_8003, 0x8000_0000_0000_8002, 0x8000_0000_0000_0080,
    0x0000_0000_0000_800A, 0x8000_0000_8000_000A, 0x8000_0000_8000_8081,
    0x8000_0000_0000_8080, 0x0000_0000_8000_0001, 0x8000_0000_8000_8008,
];

type Lane = [WireId; 64];
type StateWires = [[Lane; 5]; 5]; // state[x][y]

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Build a Keccak-f[1600] circuit that computes `num_perms` independent
/// permutations (24 rounds each), padded to `2^target_nv` total gates.
pub fn build_keccakf_circuit<F: Field<BaseField = MamaBearScalar>>(
    num_perms: usize,
    target_nv: usize,
) -> (Circuit<F>, [Vec<MamaBearScalar>; 3]) {
    let mut b = CircuitBuilder::new_mamabear();
    for perm_idx in 0..num_perms {
        keccakf_one_permutation(&mut b, perm_idx as u32);
    }
    b.pad_to_nv(target_nv);
    b.build()
}

pub fn build_keccakf_circuit_goldilocks(
    num_perms: usize,
    target_nv: usize,
) -> (Circuit<Goldilocks64Ext>, [Vec<Goldilocks64>; 3]) {
    let mut b = CircuitBuilder::new_goldilocks();
    for perm_idx in 0..num_perms {
        keccakf_one_permutation(&mut b, perm_idx as u32);
    }
    b.pad_to_nv(target_nv);
    b.build_generic::<Goldilocks64Ext, Goldilocks64>()
}

pub fn build_keccakf_circuit_babybear(
    num_perms: usize,
    target_nv: usize,
) -> (Circuit<BabyBearExt4>, [Vec<BabyBearField>; 3]) {
    let mut b = CircuitBuilder::new_babybear();
    for perm_idx in 0..num_perms {
        keccakf_one_permutation(&mut b, perm_idx as u32);
    }
    b.pad_to_nv(target_nv);
    b.build_generic::<BabyBearExt4, BabyBearField>()
}

/// Raw gate count for one Keccak-f[1600] permutation (no padding).
pub fn keccakf_gates_per_permutation() -> usize {
    let mut b = CircuitBuilder::new();
    keccakf_one_permutation(&mut b, 0);
    b.num_gates()
}

// ---------------------------------------------------------------------------
// Circuit body
// ---------------------------------------------------------------------------

/// Decompose 25 initial lanes (deterministic from `seed`) and emit one
/// 24-round Keccak-f[1600] permutation into the builder.
fn keccakf_one_permutation(b: &mut CircuitBuilder, seed: u32) {
    let initial = keccakf_initial_state(seed);
    // Decompose each lane into 64 boolean WireIds.
    let mut state: StateWires =
        std::array::from_fn(|x| std::array::from_fn(|y| b.decompose_u64(initial[x][y])));

    for round in 0..KECCAK_ROUNDS {
        theta(b, &mut state);
        rho_pi(&mut state);
        chi(b, &mut state);
        iota(b, &mut state, round);
    }

    // Final state lives on the last 25*64 wires; not output-constrained
    // (the benchmark only needs well-formedness of the circuit).
    let _ = state;
}

/// theta: A[x, y] ^= C[x-1] ^ ROT(C[x+1], 1) where C[x] = XOR over y of A[x, y].
fn theta(b: &mut CircuitBuilder, state: &mut StateWires) {
    // C[x] = state[x][0] XOR ... XOR state[x][4]
    let mut c: [Lane; 5] = [[WireId::a(0); 64]; 5];
    for x in 0..5 {
        let mut acc = state[x][0];
        for y in 1..5 {
            acc = b.xor_word64(&acc, &state[x][y]);
        }
        c[x] = acc;
    }
    // D[x] = C[x-1] XOR ROTL(C[x+1], 1)
    let mut d: [Lane; 5] = [[WireId::a(0); 64]; 5];
    for x in 0..5 {
        let xm1 = (x + 4) % 5;
        let xp1 = (x + 1) % 5;
        let rotated = b.rotl64(&c[xp1], 1);
        d[x] = b.xor_word64(&c[xm1], &rotated);
    }
    // Apply: state[x][y] ^= D[x]
    for x in 0..5 {
        for y in 0..5 {
            state[x][y] = b.xor_word64(&state[x][y], &d[x]);
        }
    }
}

/// Combined rho + pi: new_state[y][2x+3y mod 5] = ROT(state[x][y], r[x][y]).
/// FREE — no gates, only index remapping and lane rotation (also free).
fn rho_pi(state: &mut StateWires) {
    let mut new_state: StateWires = [[[WireId::a(0); 64]; 5]; 5];
    for x in 0..5 {
        for y in 0..5 {
            let n = KECCAK_RHO_OFFSETS[x][y] as usize;
            // Rotate the lane left by n bits (rotl is free).
            let rotated = rotl64_inline(&state[x][y], n);
            // Map to new position: (x', y') = (y, (2x + 3y) mod 5).
            let nx = y;
            let ny = (2 * x + 3 * y) % 5;
            new_state[nx][ny] = rotated;
        }
    }
    *state = new_state;
}

/// In-lane left rotation for StateWires (free; just index remap). We
/// duplicate `CircuitBuilder::rotl64` here because it takes `&self` and
/// operates on a bare `[WireId; 64]`; this inline version avoids
/// requiring mutable access during `rho_pi`.
fn rotl64_inline(bits: &Lane, n: usize) -> Lane {
    let n = n % 64;
    let mut out = [WireId::a(0); 64];
    for i in 0..64 {
        out[i] = bits[(i + 64 - n) % 64];
    }
    out
}

/// chi: A[x, y] ^= (NOT A[x+1, y]) AND A[x+2, y].
/// Per bit: compute `c - bc` (= (NOT b) AND c) then XOR with a. 7 gates
/// per bit: 1 mul + 1 add + 1 negate + 4 for xor_bit.
fn chi(b: &mut CircuitBuilder, state: &mut StateWires) {
    let mut new_state: StateWires = [[[WireId::a(0); 64]; 5]; 5];
    for y in 0..5 {
        for x in 0..5 {
            let a = state[x][y];
            let bb = state[(x + 1) % 5][y];
            let cc = state[(x + 2) % 5][y];
            new_state[x][y] = chi_lane(b, &a, &bb, &cc);
        }
    }
    *state = new_state;
}

/// Compute a_i XOR ((NOT b_i) AND c_i) for all 64 bits of a lane.
fn chi_lane(b: &mut CircuitBuilder, a: &Lane, bb: &Lane, cc: &Lane) -> Lane {
    let mut out = [WireId::a(0); 64];
    for i in 0..64 {
        // (NOT b) AND c = c - b*c.
        let neg_bc = b.mul_gate(bb[i], cc[i]);           // -(b*c)
        let neg_t = b.add_gate(cc[i], neg_bc);           // -(c + (-(bc))) = bc - c
        let t = b.negate(neg_t);                         // c - bc  = (NOT b) AND c
        out[i] = b.xor_bit(a[i], t);
    }
    out
}

/// iota: A[0, 0] ^= RC[round].
/// Per bit: if the RC bit is 0, no-op; if the RC bit is 1, xor with a
/// constant-1 wire (which is `xor_bit(a, one_const)` = NOT a; 4 gates
/// per such bit). Using the cheaper identity `a XOR 1 = 1 - a` via
/// `not_bit` = `xor_bit(a, one)` already in the builder.
fn iota(b: &mut CircuitBuilder, state: &mut StateWires, round: usize) {
    let rc = KECCAK_ROUND_CONSTANTS[round];
    let lane = &mut state[0][0];
    for i in 0..64 {
        if (rc >> i) & 1 == 1 {
            lane[i] = b.not_bit(lane[i]);
        }
    }
}

// ---------------------------------------------------------------------------
// Initial state derivation + native reference
// ---------------------------------------------------------------------------

/// Derive a deterministic initial state from a seed. The circuit
/// structure (gate count, wiring) is independent of the input values.
pub fn keccakf_initial_state(seed: u32) -> [[u64; 5]; 5] {
    let mut state = [[0u64; 5]; 5];
    for x in 0..5 {
        for y in 0..5 {
            // Independent LCG per lane, same shape as other circuit
            // seed schemes in this module.
            let i = (x * 5 + y) as u32;
            let lo = seed
                .wrapping_mul(0x9E3779B9)
                .wrapping_add(i.wrapping_mul(0x517CC1B7));
            let hi = seed
                .wrapping_mul(0xC2B2AE35)
                .wrapping_add(i.wrapping_mul(0x27D4EB2F));
            state[x][y] = ((hi as u64) << 32) | (lo as u64);
        }
    }
    state
}

/// Native reference: compute one 24-round Keccak-f[1600] permutation in
/// plain `u64` arithmetic. Matches FIPS 202.
pub fn keccakf_reference(mut state: [[u64; 5]; 5]) -> [[u64; 5]; 5] {
    for round in 0..KECCAK_ROUNDS {
        // theta
        let mut c = [0u64; 5];
        for x in 0..5 {
            c[x] = state[x][0] ^ state[x][1] ^ state[x][2] ^ state[x][3] ^ state[x][4];
        }
        let mut d = [0u64; 5];
        for x in 0..5 {
            let xm1 = (x + 4) % 5;
            let xp1 = (x + 1) % 5;
            d[x] = c[xm1] ^ c[xp1].rotate_left(1);
        }
        for x in 0..5 {
            for y in 0..5 {
                state[x][y] ^= d[x];
            }
        }

        // rho + pi
        let mut new_state = [[0u64; 5]; 5];
        for x in 0..5 {
            for y in 0..5 {
                let n = KECCAK_RHO_OFFSETS[x][y];
                let rotated = state[x][y].rotate_left(n);
                let nx = y;
                let ny = (2 * x + 3 * y) % 5;
                new_state[nx][ny] = rotated;
            }
        }
        state = new_state;

        // chi
        let mut chi_state = [[0u64; 5]; 5];
        for y in 0..5 {
            for x in 0..5 {
                let a = state[x][y];
                let b_ = state[(x + 1) % 5][y];
                let c_ = state[(x + 2) % 5][y];
                chi_state[x][y] = a ^ ((!b_) & c_);
            }
        }
        state = chi_state;

        // iota
        state[0][0] ^= KECCAK_ROUND_CONSTANTS[round];
    }
    state
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use arithmetic::field::mamabear::MamaBearScalarExt3;
    use crate::prover_mamabear::{setup_mamabear, AlignedPoly, MamaBearExtConfig, ProverMamaBear};
    use crate::verifier_mamabear::VerifierMamaBear;
    use poly_commit::deepfold_mamabear::DeepFoldMamaBearParam;

    #[test]
    fn keccakf_reference_sanity() {
        let s0 = [[0u64; 5]; 5];
        let s1 = keccakf_reference(s0);
        assert_ne!(s1, s0, "permutation should not fix the zero state");
        // Determinism.
        assert_eq!(s1, keccakf_reference(s0));
    }

    /// Known-answer test against published Keccak-f[1600] output for the
    /// zero state. These 25 values are from the public Keccak reference
    /// (XKCP / tiny-keccak / standard test vectors); after one full
    /// Keccak-f[1600] run on the all-zero initial state, the state
    /// lanes are as shown below (indexed as `state[x][y]`).
    #[test]
    fn keccakf_known_answer_test() {
        // Well-known Keccak-f[1600] output for the all-zero initial
        // state. Values sourced from the tiny-keccak / XKCP reference
        // (KeccakF1600 single-invocation output, documented as the
        // "initial state after one permutation" in Keccak test fixtures).
        //
        // Laid out as `state[x + 5*y]`, LE-encoded u64.
        const EXPECTED_FLAT: [u64; 25] = [
            0xF1258F7940E1DDE7, 0x84D5CCF933C0478A, 0xD598261EA65AA9EE,
            0xBD1547306F80494D, 0x8B284E056253D057, 0xFF97A42D7F8E6FD4,
            0x90FEE5A0A44647C4, 0x8C5BDA0CD6192E76, 0xAD30A6F71B19059C,
            0x30935AB7D08FFC64, 0xEB5AA93F2317D635, 0xA9A6E6260D712103,
            0x81A57C16DBCF555F, 0x43B831CD0347C826, 0x01F22F1A11A5569F,
            0x05E5635A21D9AE61, 0x64BEFEF28CC970F2, 0x613670957BC46611,
            0xB87C5A554FD00ECB, 0x8C3EE88A1CCF32C8, 0x940C7922AE3A2614,
            0x1841F924A2C509E4, 0x16F53526E70465C2, 0x75F644E97F30A13B,
            0xEAF1FF7B5CECA249,
        ];
        let mut expected: [[u64; 5]; 5] = [[0; 5]; 5];
        for y in 0..5 {
            for x in 0..5 {
                expected[x][y] = EXPECTED_FLAT[x + 5 * y];
            }
        }

        let got = keccakf_reference([[0u64; 5]; 5]);
        assert_eq!(
            got, expected,
            "Keccak-f[1600] on zero state does not match published test vector"
        );
    }

    /// Cross-check the circuit output against the native reference for a
    /// handful of seeds. We extract the final-state wire values from the
    /// builder and compare.
    #[test]
    fn keccakf_circuit_vs_reference() {
        for seed in 0..3u32 {
            let initial = keccakf_initial_state(seed);
            let expected = keccakf_reference(initial);

            // Replay the circuit and read back the final-state wire values.
            let mut b = CircuitBuilder::new();
            let mut state: StateWires = std::array::from_fn(|x| {
                std::array::from_fn(|y| b.decompose_u64(initial[x][y]))
            });
            for round in 0..KECCAK_ROUNDS {
                theta(&mut b, &mut state);
                rho_pi(&mut state);
                chi(&mut b, &mut state);
                iota(&mut b, &mut state, round);
            }
            for x in 0..5 {
                for y in 0..5 {
                    let mut got: u64 = 0;
                    for i in 0..64 {
                        got |= (b.get_val(state[x][y][i]) as u64) << i;
                    }
                    assert_eq!(
                        got, expected[x][y],
                        "seed={} position=({},{}) mismatch",
                        seed, x, y
                    );
                }
            }
        }
    }

    #[test]
    fn keccakf_circuit_gate_count() {
        let n = keccakf_gates_per_permutation();
        eprintln!("Keccak-f[1600] one permutation: {} gates", n);
        assert!(n > 200_000, "too few gates: {}", n);
        assert!(n < 1_500_000, "too many gates: {}", n);
    }

    // --- MamaBear serial / parallel prove+verify -------------------------

    fn keccakf_mamabear_serial<F: MamaBearExtConfig>() {
        // Need nv >= min_nv_for(gates_per_perm). Use nv=20.
        let target_nv = 20usize;
        let gates = keccakf_gates_per_permutation();
        assert!(1usize << target_nv >= gates,
            "nv={target_nv} cannot fit one Keccak-f permutation ({gates} gates)");

        let (circuit, witness) = build_keccakf_circuit::<F>(1, target_nv);
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

    /// Mirrors `sha256_mamabear_par` structure: sweeps nv, multiple
    /// seeds, checks serial==parallel proof bytes, 4 verify combos, and
    /// soundness (flipped byte).
    fn keccakf_mamabear_par<F: MamaBearExtConfig>()
    where
        F::Packed: Send + Sync,
        F: Send + Sync,
    {
        const NV_MIN: usize = 20;
        const NV_MAX: usize = 21;
        const SEEDS_PER_NV: u32 = 2;

        let gates = keccakf_gates_per_permutation();
        let min_nv = {
            assert!(gates > 0);
            (usize::BITS - (gates - 1).leading_zeros()) as usize
        };

        for target_nv in NV_MIN..=NV_MAX {
            if target_nv < min_nv {
                eprintln!("[keccakf_mamabear_par] skip nv={target_nv} (min {min_nv})");
                continue;
            }
            let total_gates = 1usize << target_nv;
            let num_perms = (total_gates / gates).max(1);

            for _ in 0..SEEDS_PER_NV {
                let (circuit, witness) = build_keccakf_circuit::<F>(num_perms, target_nv);

                let pp = DeepFoldMamaBearParam::new_default(target_nv, 3, 34);
                let (pk, vk) = setup_mamabear::<F>(&circuit, &pp);
                let prover = ProverMamaBear { prover_key: pk };
                let verifier = VerifierMamaBear { verifier_key: vk };

                let witness_ap_s = [
                    AlignedPoly::from_sbf(&witness[0]),
                    AlignedPoly::from_sbf(&witness[1]),
                    AlignedPoly::from_sbf(&witness[2]),
                ];
                let witness_ap_p = [
                    AlignedPoly::from_sbf(&witness[0]),
                    AlignedPoly::from_sbf(&witness[1]),
                    AlignedPoly::from_sbf(&witness[2]),
                ];

                let proof_serial = prover.prove(&pp, target_nv, witness_ap_s);
                let proof_par = prover.prove_par(&pp, target_nv, witness_ap_p);

                assert_eq!(
                    proof_serial.bytes, proof_par.bytes,
                    "serial vs parallel proof mismatch at nv={target_nv}"
                );

                let case = format!("nv={target_nv} num_perms={num_perms}");
                assert!(verifier.verify(&pp, target_nv, proof_serial.clone()),
                    "serial verify rejected serial proof at {case}");
                assert!(verifier.verify(&pp, target_nv, proof_par.clone()),
                    "serial verify rejected parallel proof at {case}");
                assert!(verifier.verify_par(&pp, target_nv, proof_serial.clone()),
                    "parallel verify rejected serial proof at {case}");
                assert!(verifier.verify_par(&pp, target_nv, proof_par.clone()),
                    "parallel verify rejected parallel proof at {case}");

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
                let par_rejects =
                    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        verifier_ref.verify_par(pp_ref, target_nv, flipped_p)
                    }))
                    .map(|ok| !ok)
                    .unwrap_or(true);
                assert!(serial_rejects, "serial verify accepted tampered proof at {case}");
                assert!(par_rejects, "parallel verify accepted tampered proof at {case} (soundness bug)");
            }
        }
    }

    #[test]
    fn keccakf_mamabear_ext3_proves_and_verifies() {
        keccakf_mamabear_serial::<MamaBearScalarExt3>();
    }

    #[test]
    fn keccakf_mamabear_ext3_par_proves_and_verifies() {
        keccakf_mamabear_par::<MamaBearScalarExt3>();
    }

    // --- Goldilocks / BabyBear smoke tests -------------------------------

    #[test]
    fn keccakf_circuit_goldilocks_builds() {
        let target_nv = 20;
        let (circuit, witness) = build_keccakf_circuit_goldilocks(1, target_nv);
        assert_eq!(circuit.selector.len(), 1 << target_nv);
        assert_eq!(witness[0].len(), 1 << target_nv);
    }

    #[test]
    fn keccakf_circuit_goldilocks_proves_and_verifies() {
        use arithmetic::{
            field::goldilocks64::{Goldilocks64, Goldilocks64Ext},
            mul_group::Radix2Group,
        };
        use crate::{prover::Prover, verifier::Verifier};
        use poly_commit::deepfold::{DeepFoldParam, DeepFoldProver, DeepFoldVerifier};

        let target_nv = 20usize;
        let (circuit, witness) = build_keccakf_circuit_goldilocks(1, target_nv);

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
    fn keccakf_circuit_babybear_builds() {
        let target_nv = 20;
        let (circuit, witness) = build_keccakf_circuit_babybear(1, target_nv);
        assert_eq!(circuit.selector.len(), 1 << target_nv);
        assert_eq!(witness[0].len(), 1 << target_nv);
    }

    #[test]
    fn keccakf_circuit_babybear_proves_and_verifies() {
        use arithmetic::{
            field::babybear::{BabyBearExt4, BabyBearField},
            mul_group::Radix2Group,
        };
        use crate::{prover::Prover, verifier::Verifier};
        use poly_commit::deepfold::{DeepFoldParam, DeepFoldProver, DeepFoldVerifier};

        let target_nv = 20usize;
        let (circuit, witness) = build_keccakf_circuit_babybear(1, target_nv);

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
