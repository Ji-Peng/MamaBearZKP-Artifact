//! AES-128 circuit generator using the [`CircuitBuilder`] and
//! [`boyar_peralta::sbox`].
//!
//! Implements AES-128 encryption as an arithmetic circuit over MamaBear add/mul
//! gates. Uses the Boyar-Peralta S-box (~360 gates/byte) for SubBytes.
//!
//! # Gate counts (per AES-128 call)
//!
//! | Component                      | Gates   |
//! |-------------------------------|---------|
//! | SubBytes (10 rounds x 16)     | ~56,000 |
//! | MixColumns (9 rounds x 4 col) | ~16,700 |
//! | AddRoundKey (11 calls)        | ~5,600  |
//! | Key Schedule (10 rounds)      | ~19,400 |
//! | Input/key decomposition       | ~512    |
//! | **Total**                     | **~98K** |

use arithmetic::field::{
    babybear::{BabyBearExt4, BabyBearField},
    goldilocks64::{Goldilocks64, Goldilocks64Ext},
};
// MamaBear is x86_64-only (its `CircuitBuilder::build` is x86-gated), so the
// MamaBear-typed entry point is gated below and these imports are x86-only too
// (`Field` is only named by that entry point's bound). The gate-emission body
// (`aes128_one_call`) is field-agnostic and the Goldilocks/BabyBear builders go
// through `build_generic`, so they reach aarch64 and provide a cross-field
// baseline.
#[cfg(target_arch = "x86_64")]
use arithmetic::field::{mamabear::MamaBearScalar, Field};
use crate::circuit::Circuit;
use crate::circuit_builder::{CircuitBuilder, WireId};
use crate::boyar_peralta;

/// An AES byte: 8 boolean WireIds, MSB = index 0, LSB = index 7.
type Byte = [WireId; 8];

/// AES-128 state: 16 bytes arranged as a 4x4 column-major matrix.
/// state[col][row], each is a Byte.
type State = [[Byte; 4]; 4];

// ---------------------------------------------------------------------------
// AES round constants
// ---------------------------------------------------------------------------

const RCON: [u8; 10] = [0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1B, 0x36];

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Build an AES-128 circuit for `num_calls` independent encryptions,
/// padded to `2^target_nv` total gates.
///
/// x86_64-only: `CircuitBuilder::build` needs the non-canonical
/// `MamaBearScalar(v)` constructor. The body is unchanged and stays
/// byte-identical; see the Goldilocks/BabyBear twins below for other fields.
#[cfg(target_arch = "x86_64")]
pub fn build_aes128_circuit<F: Field<BaseField = MamaBearScalar>>(
    num_calls: usize,
    target_nv: usize,
) -> (Circuit<F>, [Vec<MamaBearScalar>; 3]) {
    let mut builder = CircuitBuilder::new_mamabear();

    for call_idx in 0..num_calls {
        aes128_one_call(&mut builder, call_idx as u32);
    }

    builder.pad_to_nv(target_nv);
    builder.build()
}

/// Goldilocks variant of [`build_aes128_circuit`].
pub fn build_aes128_circuit_goldilocks(
    num_calls: usize,
    target_nv: usize,
) -> (Circuit<Goldilocks64Ext>, [Vec<Goldilocks64>; 3]) {
    let mut builder = CircuitBuilder::new_goldilocks();

    for call_idx in 0..num_calls {
        aes128_one_call(&mut builder, call_idx as u32);
    }

    builder.pad_to_nv(target_nv);
    builder.build_generic::<Goldilocks64Ext, Goldilocks64>()
}

/// BabyBear variant of [`build_aes128_circuit`].
pub fn build_aes128_circuit_babybear(
    num_calls: usize,
    target_nv: usize,
) -> (Circuit<BabyBearExt4>, [Vec<BabyBearField>; 3]) {
    let mut builder = CircuitBuilder::new_babybear();

    for call_idx in 0..num_calls {
        aes128_one_call(&mut builder, call_idx as u32);
    }

    builder.pad_to_nv(target_nv);
    builder.build_generic::<BabyBearExt4, BabyBearField>()
}

// ---------------------------------------------------------------------------
// AES-128 encryption circuit
// ---------------------------------------------------------------------------

/// Encrypt one block with AES-128 inside the circuit builder.
/// Raw gate count for a single AES-128 encryption (no padding), as emitted by
/// [`aes128_one_call`] on a fresh [`CircuitBuilder`]. Includes the two
/// constant-pool setup gates that [`CircuitBuilder::new`] allocates.
pub fn aes128_gates_per_call() -> usize {
    let mut b = CircuitBuilder::new();
    aes128_one_call(&mut b, 0);
    b.num_gates()
}

fn aes128_one_call(b: &mut CircuitBuilder, seed: u32) {
    // Generate deterministic plaintext and key from seed.
    let pt = gen_aes_input(seed, 0);
    let key = gen_aes_input(seed, 1);

    // Decompose plaintext and key into bytes, then bits.
    let mut state = bytes_to_state(b, &pt);
    let key_bytes: [Byte; 16] = decompose_16_bytes(b, &key);

    // Key schedule: expand 128-bit key to 11 round keys (44 words).
    let round_keys = key_expansion(b, &key_bytes);

    // Initial AddRoundKey
    state = add_round_key(b, &state, &round_keys[0]);

    // Rounds 1-9 (full rounds with MixColumns)
    for round in 1..=9 {
        state = sub_bytes(b, &state);
        state = shift_rows(&state);
        state = mix_columns(b, &state);
        state = add_round_key(b, &state, &round_keys[round]);
    }

    // Round 10 (no MixColumns)
    state = sub_bytes(b, &state);
    state = shift_rows(&state);
    state = add_round_key(b, &state, &round_keys[10]);

    // state now holds the ciphertext (not explicitly output-constrained).
    let _ = state;
}

// ---------------------------------------------------------------------------
// Input generation
// ---------------------------------------------------------------------------

fn gen_aes_input(seed: u32, variant: u32) -> [u8; 16] {
    let mut out = [0u8; 16];
    for i in 0..16 {
        let v = seed
            .wrapping_mul(0x9E3779B9)
            .wrapping_add((i as u32).wrapping_mul(0x517CC1B7))
            .wrapping_add(variant.wrapping_mul(0x6A09E667));
        out[i] = (v >> 16) as u8;
    }
    out
}

fn decompose_byte(b: &mut CircuitBuilder, byte: u8) -> Byte {
    // MSB first: index 0 = bit 7, index 7 = bit 0
    std::array::from_fn(|i| {
        let bit = ((byte >> (7 - i)) & 1) as u64;
        b.alloc_input(bit)
    })
}

fn decompose_16_bytes(b: &mut CircuitBuilder, bytes: &[u8; 16]) -> [Byte; 16] {
    std::array::from_fn(|i| decompose_byte(b, bytes[i]))
}

fn bytes_to_state(b: &mut CircuitBuilder, bytes: &[u8; 16]) -> State {
    // AES state is column-major: byte[col*4 + row]
    let byte_wires = decompose_16_bytes(b, bytes);
    std::array::from_fn(|col| {
        std::array::from_fn(|row| byte_wires[col * 4 + row])
    })
}

// ---------------------------------------------------------------------------
// SubBytes
// ---------------------------------------------------------------------------

fn sub_bytes(b: &mut CircuitBuilder, state: &State) -> State {
    std::array::from_fn(|col| {
        std::array::from_fn(|row| boyar_peralta::sbox(b, state[col][row]))
    })
}

// ---------------------------------------------------------------------------
// ShiftRows (purely combinatorial — no gates)
// ---------------------------------------------------------------------------

fn shift_rows(state: &State) -> State {
    // Row 0: no shift
    // Row 1: shift left by 1
    // Row 2: shift left by 2
    // Row 3: shift left by 3
    std::array::from_fn(|col| {
        std::array::from_fn(|row| state[(col + row) % 4][row])
    })
}

// ---------------------------------------------------------------------------
// MixColumns
// ---------------------------------------------------------------------------

/// XOR two bytes (8-bit XOR).
fn xor_byte(b: &mut CircuitBuilder, a: &Byte, c: &Byte) -> Byte {
    std::array::from_fn(|i| b.xor_bit(a[i], c[i]))
}

/// xtime(a): multiply byte `a` by 0x02 in GF(2^8).
/// Left shift by 1, then XOR with 0x1B if MSB was set.
/// In bit terms (MSB-first indexing, index 0 = bit 7):
///   out[0] = a[1]        (bit 6 shifts up)
///   out[1] = a[2]
///   out[2] = a[3]
///   out[3] = a[4] XOR a[0]  (0x1B bit 4 = 1)
///   out[4] = a[5] XOR a[0]  (0x1B bit 3 = 1)
///   out[5] = a[6]
///   out[6] = a[7] XOR a[0]  (0x1B bit 1 = 1)
///   out[7] = a[0]            (0x1B bit 0 = 1)
fn xtime(b: &mut CircuitBuilder, a: &Byte) -> Byte {
    let msb = a[0]; // a[0] is bit 7 (MSB)
    [
        a[1],                       // bit 6
        a[2],                       // bit 5
        a[3],                       // bit 4
        b.xor_bit(a[4], msb),      // bit 3 XOR msb
        b.xor_bit(a[5], msb),      // bit 2 XOR msb
        a[6],                       // bit 1
        b.xor_bit(a[7], msb),      // bit 0 XOR msb
        msb,                        // feedback: msb goes to LSB
    ]
}

/// Multiply by 0x03 in GF(2^8): xtime(a) XOR a.
fn mul03(b: &mut CircuitBuilder, a: &Byte) -> Byte {
    let xt = xtime(b, a);
    xor_byte(b, &xt, a)
}

/// MixColumns for one column [s0, s1, s2, s3].
/// r0 = 02*s0 + 03*s1 + s2 + s3
/// r1 = s0 + 02*s1 + 03*s2 + s3
/// r2 = s0 + s1 + 02*s2 + 03*s3
/// r3 = 03*s0 + s1 + s2 + 02*s3
fn mix_one_column(b: &mut CircuitBuilder, col: &[Byte; 4]) -> [Byte; 4] {
    // Precompute xtime and mul03 for each byte
    let xt: [Byte; 4] = std::array::from_fn(|i| xtime(b, &col[i]));
    let m3: [Byte; 4] = std::array::from_fn(|i| mul03(b, &col[i]));

    // r0 = xt[0] + m3[1] + col[2] + col[3]
    let t0 = xor_byte(b, &xt[0], &m3[1]);
    let t1 = xor_byte(b, &col[2], &col[3]);
    let r0 = xor_byte(b, &t0, &t1);

    // r1 = col[0] + xt[1] + m3[2] + col[3]
    let t0 = xor_byte(b, &col[0], &xt[1]);
    let t1 = xor_byte(b, &m3[2], &col[3]);
    let r1 = xor_byte(b, &t0, &t1);

    // r2 = col[0] + col[1] + xt[2] + m3[3]
    let t0 = xor_byte(b, &col[0], &col[1]);
    let t1 = xor_byte(b, &xt[2], &m3[3]);
    let r2 = xor_byte(b, &t0, &t1);

    // r3 = m3[0] + col[1] + col[2] + xt[3]
    let t0 = xor_byte(b, &m3[0], &col[1]);
    let t1 = xor_byte(b, &col[2], &xt[3]);
    let r3 = xor_byte(b, &t0, &t1);

    [r0, r1, r2, r3]
}

fn mix_columns(b: &mut CircuitBuilder, state: &State) -> State {
    std::array::from_fn(|col| {
        let column = [state[col][0], state[col][1], state[col][2], state[col][3]];
        let mixed = mix_one_column(b, &column);
        mixed
    })
}

// ---------------------------------------------------------------------------
// AddRoundKey
// ---------------------------------------------------------------------------

/// XOR state with a round key (16 bytes).
fn add_round_key(b: &mut CircuitBuilder, state: &State, rk: &[Byte; 16]) -> State {
    std::array::from_fn(|col| {
        std::array::from_fn(|row| {
            xor_byte(b, &state[col][row], &rk[col * 4 + row])
        })
    })
}

// ---------------------------------------------------------------------------
// Key Schedule
// ---------------------------------------------------------------------------

/// XOR a byte with a constant byte (known at circuit construction time).
/// Only XOR bits where the constant bit is 1 (otherwise wire routing).
fn xor_byte_const(b: &mut CircuitBuilder, a: &Byte, constant: u8) -> Byte {
    std::array::from_fn(|i| {
        let cbit = ((constant >> (7 - i)) & 1) as u64;
        if cbit == 1 {
            b.not_bit(a[i]) // XOR with 1 = NOT
        } else {
            a[i] // XOR with 0 = identity
        }
    })
}

/// AES-128 key expansion: 128-bit key → 11 round keys (each 16 bytes).
fn key_expansion(b: &mut CircuitBuilder, key: &[Byte; 16]) -> [[Byte; 16]; 11] {
    // Key schedule works on 32-bit words (4 bytes each).
    // Initial key = 4 words.
    let mut w: Vec<[Byte; 4]> = Vec::with_capacity(44);

    // w[0..3] = initial key words
    for i in 0..4 {
        w.push([key[i * 4], key[i * 4 + 1], key[i * 4 + 2], key[i * 4 + 3]]);
    }

    // w[4..43] = expanded key words
    for i in 4..44 {
        let mut temp = w[i - 1];
        if i % 4 == 0 {
            // RotWord: rotate bytes left by 1
            temp = [temp[1], temp[2], temp[3], temp[0]];
            // SubWord: apply S-box to each byte
            temp = [
                boyar_peralta::sbox(b, temp[0]),
                boyar_peralta::sbox(b, temp[1]),
                boyar_peralta::sbox(b, temp[2]),
                boyar_peralta::sbox(b, temp[3]),
            ];
            // XOR with Rcon (only first byte)
            temp[0] = xor_byte_const(b, &temp[0], RCON[i / 4 - 1]);
        }
        // w[i] = w[i-4] XOR temp
        let prev = &w[i - 4];
        let word: [Byte; 4] = std::array::from_fn(|j| xor_byte(b, &prev[j], &temp[j]));
        w.push(word);
    }

    // Convert 44 words to 11 round keys (each 16 bytes).
    let mut round_keys = [[[WireId::a(0); 8]; 16]; 11];
    for rk_idx in 0..11 {
        for word_in_rk in 0..4 {
            let word = &w[rk_idx * 4 + word_in_rk];
            for byte_in_word in 0..4 {
                round_keys[rk_idx][word_in_rk * 4 + byte_in_word] = word[byte_in_word];
            }
        }
    }
    round_keys
}

// ---------------------------------------------------------------------------
// Reference implementation for testing
// ---------------------------------------------------------------------------

/// AES-128 encryption reference (for testing circuit correctness).
pub fn aes128_reference(plaintext: &[u8; 16], key: &[u8; 16]) -> [u8; 16] {
    let mut state = *plaintext;

    // Key expansion
    let mut w = [[0u8; 4]; 44];
    for i in 0..4 {
        w[i] = [key[4 * i], key[4 * i + 1], key[4 * i + 2], key[4 * i + 3]];
    }
    for i in 4..44 {
        let mut temp = w[i - 1];
        if i % 4 == 0 {
            temp = [temp[1], temp[2], temp[3], temp[0]]; // RotWord
            for b in temp.iter_mut() {
                *b = boyar_peralta::AES_SBOX[*b as usize];
            } // SubWord
            temp[0] ^= RCON[i / 4 - 1]; // Rcon
        }
        for j in 0..4 {
            w[i][j] = w[i - 4][j] ^ temp[j];
        }
    }

    // Extract round keys
    let rk = |round: usize| -> [u8; 16] {
        let mut k = [0u8; 16];
        for wi in 0..4 {
            for bi in 0..4 {
                k[wi * 4 + bi] = w[round * 4 + wi][bi];
            }
        }
        k
    };

    // AddRoundKey
    let add_rk = |s: &mut [u8; 16], k: &[u8; 16]| {
        for i in 0..16 {
            s[i] ^= k[i];
        }
    };

    // SubBytes
    let sub_bytes_ref = |s: &mut [u8; 16]| {
        for b in s.iter_mut() {
            *b = boyar_peralta::AES_SBOX[*b as usize];
        }
    };

    // ShiftRows (operates on column-major: state[col*4+row])
    let shift_rows_ref = |s: &mut [u8; 16]| {
        // Row 1: shift left by 1
        let t = s[1];
        s[1] = s[5];
        s[5] = s[9];
        s[9] = s[13];
        s[13] = t;
        // Row 2: shift left by 2
        let (t0, t1) = (s[2], s[6]);
        s[2] = s[10];
        s[6] = s[14];
        s[10] = t0;
        s[14] = t1;
        // Row 3: shift left by 3
        let t = s[15];
        s[15] = s[11];
        s[11] = s[7];
        s[7] = s[3];
        s[3] = t;
    };

    // MixColumns
    let xtime_ref = |a: u8| -> u8 {
        let r = (a as u16) << 1;
        if a & 0x80 != 0 {
            (r ^ 0x1B) as u8
        } else {
            r as u8
        }
    };
    let mix_columns_ref = |s: &mut [u8; 16]| {
        for col in 0..4 {
            let a = [s[col * 4], s[col * 4 + 1], s[col * 4 + 2], s[col * 4 + 3]];
            s[col * 4] = xtime_ref(a[0]) ^ xtime_ref(a[1]) ^ a[1] ^ a[2] ^ a[3];
            s[col * 4 + 1] = a[0] ^ xtime_ref(a[1]) ^ xtime_ref(a[2]) ^ a[2] ^ a[3];
            s[col * 4 + 2] = a[0] ^ a[1] ^ xtime_ref(a[2]) ^ xtime_ref(a[3]) ^ a[3];
            s[col * 4 + 3] = xtime_ref(a[0]) ^ a[0] ^ a[1] ^ a[2] ^ xtime_ref(a[3]);
        }
    };

    // Initial round key
    let k0 = rk(0);
    add_rk(&mut state, &k0);

    // Rounds 1-9
    for round in 1..=9 {
        sub_bytes_ref(&mut state);
        shift_rows_ref(&mut state);
        mix_columns_ref(&mut state);
        let ki = rk(round);
        add_rk(&mut state, &ki);
    }

    // Round 10
    sub_bytes_ref(&mut state);
    shift_rows_ref(&mut state);
    let k10 = rk(10);
    add_rk(&mut state, &k10);

    state
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
    fn aes128_reference_nist_vector() {
        // NIST FIPS 197 Appendix B test vector
        let key: [u8; 16] = [
            0x2b, 0x7e, 0x15, 0x16, 0x28, 0xae, 0xd2, 0xa6,
            0xab, 0xf7, 0x15, 0x88, 0x09, 0xcf, 0x4f, 0x3c,
        ];
        let pt: [u8; 16] = [
            0x32, 0x43, 0xf6, 0xa8, 0x88, 0x5a, 0x30, 0x8d,
            0x31, 0x31, 0x98, 0xa2, 0xe0, 0x37, 0x07, 0x34,
        ];
        let expected_ct: [u8; 16] = [
            0x39, 0x25, 0x84, 0x1d, 0x02, 0xdc, 0x09, 0xfb,
            0xdc, 0x11, 0x85, 0x97, 0x19, 0x6a, 0x0b, 0x32,
        ];
        let ct = aes128_reference(&pt, &key);
        assert_eq!(ct, expected_ct, "NIST test vector mismatch");
    }

    #[test]
    fn aes128_gate_count() {
        let mut b = CircuitBuilder::new();
        aes128_one_call(&mut b, 0);
        let n = b.num_gates();
        eprintln!("AES-128 one call: {} gates", n);
        assert!(n > 50_000, "Too few gates: {}", n);
        assert!(n < 200_000, "Too many gates: {}", n);
    }

    /// Helper: serial prove -> verify for 1 AES-128 call at nv=18.
    fn aes128_mamabear_serial<F: MamaBearExtConfig>() {
        let target_nv = 18usize;
        let (circuit, witness) = build_aes128_circuit::<F>(1, target_nv);

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

    /// Helper: parallel prove -> verify + serial/par proof-bytes equality check.
    fn aes128_mamabear_par<F: MamaBearExtConfig>()
    where
        F::Packed: Send + Sync,
        F: Send + Sync,
    {
        let target_nv = 18usize;
        let (circuit, witness) = build_aes128_circuit::<F>(1, target_nv);

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
    fn aes128_mamabear_ext3_proves_and_verifies() {
        aes128_mamabear_serial::<MamaBearScalarExt3>();
    }

    #[test]
    fn aes128_mamabear_ext3_par_proves_and_verifies() {
        aes128_mamabear_par::<MamaBearScalarExt3>();
    }

    #[test]
    fn aes128_circuit_goldilocks_builds() {
        let target_nv = 18;
        let (circuit, witness) = build_aes128_circuit_goldilocks(1, target_nv);
        assert_eq!(circuit.selector.len(), 1 << target_nv);
        assert_eq!(witness[0].len(), 1 << target_nv);
    }

    #[test]
    fn aes128_circuit_goldilocks_proves_and_verifies() {
        use arithmetic::{
            field::goldilocks64::{Goldilocks64, Goldilocks64Ext},
            mul_group::Radix2Group,
        };
        use crate::{prover::Prover, verifier::Verifier};
        use poly_commit::deepfold::{DeepFoldParam, DeepFoldProver, DeepFoldVerifier};

        let target_nv = 18usize;
        let (circuit, witness) = build_aes128_circuit_goldilocks(1, target_nv);

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
    fn aes128_circuit_babybear_builds() {
        let target_nv = 18;
        let (circuit, witness) = build_aes128_circuit_babybear(1, target_nv);
        assert_eq!(circuit.selector.len(), 1 << target_nv);
        assert_eq!(witness[0].len(), 1 << target_nv);
    }

    #[test]
    fn aes128_circuit_babybear_proves_and_verifies() {
        use arithmetic::{
            field::babybear::{BabyBearExt4, BabyBearField},
            mul_group::Radix2Group,
        };
        use crate::{prover::Prover, verifier::Verifier};
        use poly_commit::deepfold::{DeepFoldParam, DeepFoldProver, DeepFoldVerifier};

        let target_nv = 18usize;
        let (circuit, witness) = build_aes128_circuit_babybear(1, target_nv);

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
