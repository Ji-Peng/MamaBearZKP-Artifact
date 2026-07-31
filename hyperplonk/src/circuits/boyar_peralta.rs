//! Boyar-Peralta optimized Boolean circuit for the AES S-box.
//!
//! Implements the 113-gate circuit (32 AND + 81 XOR/XNOR) from:
//! Joan Boyar and René Peralta, "A small depth-16 circuit for the AES S-Box"
//! (IFIP Sec 2012). Circuit file: `SLP_AES_113.txt` from Peralta's page.
//!
//! Inputs: U0..U7 (8 bits of AES state byte, MSB=U0, LSB=U7).
//! Outputs: S0..S7 (8 bits of S-box output).
//!
//! # Sign-tracked lowering
//!
//! All intermediate wires are held in their **field-negated** form — a wire
//! that logically represents bit `v` carries the value `-v mod P` instead
//! of `+v`. This lets us:
//!
//! - Replace the 2-gate `and_bit` (`mul_gate` + `negate`) with a bare
//!   `mul_gate` (1 gate): `mul(-a, -b) = -((-a)(-b)) = -ab`, already in
//!   the desired NEG form.
//! - Replace the 4-gate `xor_bit` (`x + y - 2xy` chain) with the
//!   `(x - y)^2` identity:
//!   - Mixed-sign inputs (POS,NEG) → 2 gates: `add` gives `diff = x - y`
//!     directly, then `mul(diff, diff) = -(x - y)^2 = -XOR`.
//!   - Same-sign inputs → 3 gates: one extra `negate` to flip one input
//!     before the `add`.
//!
//! The outputs `s0..s7` are materialized back to positive booleans at the
//! end of the function. XNOR outputs are materialized via `negate` then
//! `not_bit` (fast path: 1 gate each).
//!
//! Measured cost: ~305 gates per S-box (down from 412 in the naive
//! encoding), with full exhaustive-test coverage preserved.

use crate::circuit_builder::{CircuitBuilder, WireId};

/// Compute the AES S-box on 8 boolean WireIds using the Boyar-Peralta circuit.
///
/// `input[0]` = U0 (MSB), `input[7]` = U7 (LSB).
/// Returns `output[0]` = S0 (MSB), `output[7]` = S7 (LSB).
pub fn sbox(b: &mut CircuitBuilder, input: [WireId; 8]) -> [WireId; 8] {
    let [u0, u1, u2, u3, u4, u5, u6, u7] = input;

    // Convention below: `_neg` suffix marks a wire whose value is `-bit`
    // in the field (POS wire is named without the suffix). Raw inputs
    // `u0..u7` are POS (no suffix).
    //
    // Helpers used:
    //   xor_same_sign_to_neg(a, b)  -> -XOR,  3 gates (both inputs same sign)
    //   xor_mixed_sign_to_neg(a, b) -> -XOR,  2 gates (opposite signs)
    //   mul_gate(a, b)              -> -(a*b), 1 gate (sign flips)

    // ===== Top linear transform (23 XOR gates) =====
    // First-tier XORs: two POS inputs → NEG output (3 gates each).
    let y14_neg = b.xor_same_sign_to_neg(u3, u5);
    let y13_neg = b.xor_same_sign_to_neg(u0, u6);
    let y9_neg = b.xor_same_sign_to_neg(u0, u3);
    let y8_neg = b.xor_same_sign_to_neg(u0, u5);
    let t0_neg = b.xor_same_sign_to_neg(u1, u2);
    // Mixed (NEG,POS) XORs at 2 gates each:
    let y1_neg = b.xor_mixed_sign_to_neg(t0_neg, u7);
    let y4_neg = b.xor_mixed_sign_to_neg(y1_neg, u3);
    // Same-sign (NEG,NEG) XOR at 3 gates:
    let y12_neg = b.xor_same_sign_to_neg(y13_neg, y14_neg);
    let y2_neg = b.xor_mixed_sign_to_neg(y1_neg, u0);
    let y5_neg = b.xor_mixed_sign_to_neg(y1_neg, u6);
    let y3_neg = b.xor_same_sign_to_neg(y5_neg, y8_neg);
    let t1_neg = b.xor_mixed_sign_to_neg(u4, y12_neg);
    let y15_neg = b.xor_mixed_sign_to_neg(t1_neg, u5);
    let y20_neg = b.xor_mixed_sign_to_neg(t1_neg, u1);
    let y6_neg = b.xor_mixed_sign_to_neg(y15_neg, u7);
    let y10_neg = b.xor_same_sign_to_neg(y15_neg, t0_neg);
    let y11_neg = b.xor_same_sign_to_neg(y20_neg, y9_neg);
    let y7_neg = b.xor_mixed_sign_to_neg(u7, y11_neg);
    let y17_neg = b.xor_same_sign_to_neg(y10_neg, y11_neg);
    let y19_neg = b.xor_same_sign_to_neg(y10_neg, y8_neg);
    let y16_neg = b.xor_same_sign_to_neg(t0_neg, y11_neg);
    let y21_neg = b.xor_same_sign_to_neg(y13_neg, y16_neg);
    let y18_neg = b.xor_mixed_sign_to_neg(u0, y16_neg);

    // ===== Shared nonlinear middle (32 AND + 26 XOR) =====
    // AND implemented via bare `mul_gate`. Output sign = -(in1_sign * in2_sign).
    // Inputs that are (NEG, NEG) produce NEG output (same as XOR outputs above).
    // Inputs that mix (NEG, POS) produce POS output.
    let t2_neg = b.mul_gate(y12_neg, y15_neg); // NEG, NEG -> NEG
    let t3_neg = b.mul_gate(y3_neg, y6_neg);   // NEG
    let t4_neg = b.xor_same_sign_to_neg(t3_neg, t2_neg);
    // t5 has mixed-sign inputs (y4 NEG, u7 POS), so its output is POS.
    let t5_pos = b.mul_gate(y4_neg, u7); // POS
    let t6_neg = b.xor_mixed_sign_to_neg(t5_pos, t2_neg);
    let t7_neg = b.mul_gate(y13_neg, y16_neg);
    let t8_neg = b.mul_gate(y5_neg, y1_neg);
    let t9_neg = b.xor_same_sign_to_neg(t8_neg, t7_neg);
    let t10_neg = b.mul_gate(y2_neg, y7_neg);
    let t11_neg = b.xor_same_sign_to_neg(t10_neg, t7_neg);
    let t12_neg = b.mul_gate(y9_neg, y11_neg);
    let t13_neg = b.mul_gate(y14_neg, y17_neg);
    let t14_neg = b.xor_same_sign_to_neg(t13_neg, t12_neg);
    let t15_neg = b.mul_gate(y8_neg, y10_neg);
    let t16_neg = b.xor_same_sign_to_neg(t15_neg, t12_neg);
    let t17_neg = b.xor_same_sign_to_neg(t4_neg, y20_neg);
    let t18_neg = b.xor_same_sign_to_neg(t6_neg, t16_neg);
    let t19_neg = b.xor_same_sign_to_neg(t9_neg, t14_neg);
    let t20_neg = b.xor_same_sign_to_neg(t11_neg, t16_neg);
    let t21_neg = b.xor_same_sign_to_neg(t17_neg, t14_neg);
    let t22_neg = b.xor_same_sign_to_neg(t18_neg, y19_neg);
    let t23_neg = b.xor_same_sign_to_neg(t19_neg, y21_neg);
    let t24_neg = b.xor_same_sign_to_neg(t20_neg, y18_neg);
    let t25_neg = b.xor_same_sign_to_neg(t21_neg, t22_neg);
    let t26_neg = b.mul_gate(t21_neg, t23_neg); // NEG, NEG -> NEG
    let t27_neg = b.xor_same_sign_to_neg(t24_neg, t26_neg);
    let t28_neg = b.mul_gate(t25_neg, t27_neg);
    let t29_neg = b.xor_same_sign_to_neg(t28_neg, t22_neg);
    let t30_neg = b.xor_same_sign_to_neg(t23_neg, t24_neg);
    let t31_neg = b.xor_same_sign_to_neg(t22_neg, t26_neg);
    let t32_neg = b.mul_gate(t31_neg, t30_neg);
    let t33_neg = b.xor_same_sign_to_neg(t32_neg, t24_neg);
    let t34_neg = b.xor_same_sign_to_neg(t23_neg, t33_neg);
    let t35_neg = b.xor_same_sign_to_neg(t27_neg, t33_neg);
    let t36_neg = b.mul_gate(t24_neg, t35_neg);
    let t37_neg = b.xor_same_sign_to_neg(t36_neg, t34_neg);
    let t38_neg = b.xor_same_sign_to_neg(t27_neg, t36_neg);
    let t39_neg = b.mul_gate(t29_neg, t38_neg);
    let t40_neg = b.xor_same_sign_to_neg(t25_neg, t39_neg);
    let t41_neg = b.xor_same_sign_to_neg(t40_neg, t37_neg);
    let t42_neg = b.xor_same_sign_to_neg(t29_neg, t33_neg);
    let t43_neg = b.xor_same_sign_to_neg(t29_neg, t40_neg);
    let t44_neg = b.xor_same_sign_to_neg(t33_neg, t37_neg);
    let t45_neg = b.xor_same_sign_to_neg(t42_neg, t41_neg);

    // ===== Multiply by bases (18 AND + 0 XOR) =====
    // All z_i = AND(NEG, NEG) -> NEG. Bare `mul_gate`, 1 gate each.
    // Exception: z2 and z11 use `u7` (POS) as one input, so those outputs
    // are POS. We handle them explicitly below.
    let z0_neg = b.mul_gate(t44_neg, y15_neg);
    let z1_neg = b.mul_gate(t37_neg, y6_neg);
    let z2_pos = b.mul_gate(t33_neg, u7); // NEG, POS -> POS
    let z3_neg = b.mul_gate(t43_neg, y16_neg);
    let z4_neg = b.mul_gate(t40_neg, y1_neg);
    let z5_neg = b.mul_gate(t29_neg, y7_neg);
    let z6_neg = b.mul_gate(t42_neg, y11_neg);
    let z7_neg = b.mul_gate(t45_neg, y17_neg);
    let z8_neg = b.mul_gate(t41_neg, y10_neg);
    let z9_neg = b.mul_gate(t44_neg, y12_neg);
    let z10_neg = b.mul_gate(t37_neg, y3_neg);
    let z11_neg = b.mul_gate(t33_neg, y4_neg);
    let z12_neg = b.mul_gate(t43_neg, y13_neg);
    let z13_neg = b.mul_gate(t40_neg, y5_neg);
    let z14_neg = b.mul_gate(t29_neg, y2_neg);
    let z15_neg = b.mul_gate(t42_neg, y9_neg);
    let z16_neg = b.mul_gate(t45_neg, y14_neg);
    let z17_neg = b.mul_gate(t41_neg, y8_neg);

    // ===== Bottom linear transform (28 XOR + 4 NOT for XNOR outputs) =====
    let tc1_neg = b.xor_same_sign_to_neg(z15_neg, z16_neg);
    let tc2_neg = b.xor_same_sign_to_neg(z10_neg, tc1_neg);
    let tc3_neg = b.xor_same_sign_to_neg(z9_neg, tc2_neg);
    // z2_pos paired with z0_neg is mixed-sign: 2 gates.
    let tc4_neg = b.xor_mixed_sign_to_neg(z2_pos, z0_neg);
    let tc5_neg = b.xor_same_sign_to_neg(z1_neg, z0_neg);
    let tc6_neg = b.xor_same_sign_to_neg(z3_neg, z4_neg);
    let tc7_neg = b.xor_same_sign_to_neg(z12_neg, tc4_neg);
    let tc8_neg = b.xor_same_sign_to_neg(z7_neg, tc6_neg);
    let tc9_neg = b.xor_same_sign_to_neg(z8_neg, tc7_neg);
    let tc10_neg = b.xor_same_sign_to_neg(tc8_neg, tc9_neg);
    let tc11_neg = b.xor_same_sign_to_neg(tc6_neg, tc5_neg);
    let tc12_neg = b.xor_same_sign_to_neg(z3_neg, z5_neg);
    let tc13_neg = b.xor_same_sign_to_neg(z13_neg, tc1_neg);
    let tc14_neg = b.xor_same_sign_to_neg(tc4_neg, tc12_neg);
    let s3_neg = b.xor_same_sign_to_neg(tc3_neg, tc11_neg);
    let tc16_neg = b.xor_same_sign_to_neg(z6_neg, tc8_neg);
    let tc17_neg = b.xor_same_sign_to_neg(z14_neg, tc10_neg);
    let tc18_neg = b.xor_same_sign_to_neg(tc13_neg, tc14_neg);
    let s7_pre_neg = b.xor_same_sign_to_neg(z12_neg, tc18_neg);
    let tc20_neg = b.xor_same_sign_to_neg(z15_neg, tc16_neg);
    let tc21_neg = b.xor_same_sign_to_neg(tc2_neg, z11_neg);
    let s0_neg = b.xor_same_sign_to_neg(tc3_neg, tc16_neg);
    let s6_pre_neg = b.xor_same_sign_to_neg(tc10_neg, tc18_neg);
    let s4_neg = b.xor_same_sign_to_neg(tc14_neg, s3_neg);
    let s1_pre_neg = b.xor_same_sign_to_neg(s3_neg, tc16_neg);
    let tc26_neg = b.xor_same_sign_to_neg(tc17_neg, tc20_neg);
    let s2_pre_neg = b.xor_same_sign_to_neg(tc26_neg, z17_neg);
    let s5_neg = b.xor_same_sign_to_neg(tc21_neg, tc17_neg);

    // ===== Output materialization =====
    // - Plain XOR outputs (s0, s3, s4, s5): one `negate` each to flip NEG -> POS.
    // - XNOR outputs (s7, s6, s1, s2): `negate` to POS XOR, then `not_bit`
    //   (fast 1-gate path via `add(x, P-1)`) to compute 1 - XOR = XNOR.
    let s0 = b.negate(s0_neg);
    let s3 = b.negate(s3_neg);
    let s4 = b.negate(s4_neg);
    let s5 = b.negate(s5_neg);
    let s7_pre = b.negate(s7_pre_neg);
    let s7 = b.not_bit(s7_pre);
    let s6_pre = b.negate(s6_pre_neg);
    let s6 = b.not_bit(s6_pre);
    let s1_pre = b.negate(s1_pre_neg);
    let s1 = b.not_bit(s1_pre);
    let s2_pre = b.negate(s2_pre_neg);
    let s2 = b.not_bit(s2_pre);

    [s0, s1, s2, s3, s4, s5, s6, s7]
}

// ---------------------------------------------------------------------------
// Reference AES S-box table for testing
// ---------------------------------------------------------------------------

/// The standard AES S-box lookup table.
#[rustfmt::skip]
pub const AES_SBOX: [u8; 256] = [
    0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
    0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
    0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
    0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
    0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
    0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
    0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
    0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
    0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
    0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
    0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
    0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
    0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
    0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
    0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
    0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16,
];

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::circuit_builder::CircuitBuilder;

    #[test]
    fn sbox_exhaustive() {
        let mut builder = CircuitBuilder::new();

        for byte_val in 0u8..=255 {
            // Decompose input byte to 8 bits (MSB first: input[0] = bit 7)
            let input: [WireId; 8] = std::array::from_fn(|i| {
                let bit = ((byte_val >> (7 - i)) & 1) as u64;
                builder.alloc_input(bit)
            });

            let output = sbox(&mut builder, input);

            // Reconstruct output byte
            let mut out_byte = 0u8;
            for i in 0..8 {
                let bit = builder.get_val(output[i]);
                assert!(bit <= 1, "S-box output bit is not boolean: {}", bit);
                out_byte |= (bit as u8) << (7 - i);
            }

            let expected = AES_SBOX[byte_val as usize];
            assert_eq!(
                out_byte, expected,
                "S-box mismatch for input 0x{:02x}: got 0x{:02x}, expected 0x{:02x}",
                byte_val, out_byte, expected
            );
        }
    }

    #[test]
    fn sbox_gate_count() {
        let mut builder = CircuitBuilder::new();
        let before = builder.num_gates();
        let input: [WireId; 8] = std::array::from_fn(|i| builder.alloc_input(i as u64 & 1));
        let _ = sbox(&mut builder, input);
        let gates = builder.num_gates() - before;
        eprintln!("S-box gates (including input alloc): {}", gates);
        // Should be ~400-600 gates per S-box (including input allocation overhead)
        assert!(gates < 1000, "Too many gates: {}", gates);
    }
}
