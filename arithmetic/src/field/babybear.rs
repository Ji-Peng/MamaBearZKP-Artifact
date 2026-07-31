//! Wrapper types for Plonky3's Baby Bear field, implementing our Field + FftField traits.
//!
//! Baby Bear: P = 2^31 - 2^27 + 1 = 2013265921 (31-bit prime, 2-adic order 27).
//! Extension: BinomialExtensionField<BabyBear, 4> (~124-bit security).
//!
//! Implementation note: this wrapper is a scalar implementation. It delegates every
//! operation to Plonky3's scalar `BabyBear` (= `MontyField31<BabyBearParameters>`)
//! and never constructs a `PackedBabyBearAVX512` value, so Plonky3's 16-lane AVX-512
//! intrinsic path is not reached through this wrapper. Inspecting the disassembly
//! (with `RUSTFLAGS="-C target-cpu=native"`) confirms this:
//!
//! - Multiplication-related ops (`Mul`, `MulAssign`, `square`, Montgomery reduce) are
//!   emitted as scalar `imul` chains against constants `0x37ffffe9` (MONTY_MU) and
//!   `0x78000001` (PRIME); no packed multiplication (`vpmuludq`/`vpmullq` on `zmm`
//!   in 16-lane form) appears from inside BabyBear arithmetic itself.
//! - Additive ops (`Add`, `Sub`, `Neg`) may occasionally be auto-vectorized by LLVM
//!   when the caller exposes enough independent parallel data (e.g. the bench
//!   harness's 10-way dependency chain). Any `vpaddd`/`vpsubd` seen there is a
//!   caller-side auto-vectorization of trivial u32 add/sub, not a call into
//!   Plonky3's AVX-512 implementation.
//!
//! This wrapper is deliberately scalar-only: it exists to give the BabyBear
//! baseline a uniform interface for cross-field comparison, not to compete with
//! Plonky3's own vectorised kernels, so adding SIMD here would only blur what
//! the baseline is measuring.

use p3_baby_bear::BabyBear;
use p3_field::extension::BinomialExtensionField;
use p3_field::{BasedVectorSpace, Field as P3Field, PrimeCharacteristicRing, PrimeField32};

use super::{FftField, Field};
use rand::RngCore;

/// Baby Bear prime: P = 2^31 - 2^27 + 1
pub const P: u32 = 2013265921;

// ─── BabyBearField (base field wrapper) ───────────────────────────────────────

#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[repr(transparent)]
pub struct BabyBearField(pub BabyBear);

impl std::ops::Neg for BabyBearField {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self { BabyBearField(-self.0) }
}

impl std::ops::Add for BabyBearField {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self { BabyBearField(self.0 + rhs.0) }
}

impl std::ops::AddAssign for BabyBearField {
    #[inline]
    fn add_assign(&mut self, rhs: Self) { self.0 += rhs.0; }
}

impl std::ops::Sub for BabyBearField {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self { BabyBearField(self.0 - rhs.0) }
}

impl std::ops::SubAssign for BabyBearField {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) { self.0 -= rhs.0; }
}

impl std::ops::Mul for BabyBearField {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self { BabyBearField(self.0 * rhs.0) }
}

impl std::ops::MulAssign for BabyBearField {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) { self.0 *= rhs.0; }
}

impl From<u32> for BabyBearField {
    fn from(value: u32) -> Self {
        BabyBearField(BabyBear::new(value % P))
    }
}

impl From<u64> for BabyBearField {
    fn from(value: u64) -> Self {
        BabyBearField(BabyBear::new((value % P as u64) as u32))
    }
}

impl Field for BabyBearField {
    const NAME: &'static str = "BabyBear";
    const SIZE: usize = 4;
    type BaseField = BabyBearField;

    #[inline]
    fn zero() -> Self { BabyBearField(BabyBear::ZERO) }

    #[inline]
    fn is_zero(&self) -> bool { self.0 == BabyBear::ZERO }

    #[inline]
    fn one() -> Self { BabyBearField(BabyBear::ONE) }

    fn random(mut rng: impl RngCore) -> Self {
        let v = rng.next_u32() % P;
        BabyBearField(BabyBear::new(v))
    }

    fn inv_2() -> Self {
        // (P + 1) / 2 = 1006632961
        BabyBearField(BabyBear::new(1006632961))
    }

    fn exp(&self, mut exponent: usize) -> Self {
        let mut res = Self::one();
        let mut t = *self;
        while exponent != 0 {
            if (exponent & 1) == 1 {
                res *= t;
            }
            t *= t;
            exponent >>= 1;
        }
        res
    }

    fn inv(&self) -> Option<Self> {
        self.0.try_inverse().map(BabyBearField)
    }

    fn add_base_elem(&self, rhs: Self::BaseField) -> Self {
        *self + rhs
    }

    fn add_assign_base_elem(&mut self, rhs: Self::BaseField) {
        *self += rhs;
    }

    fn mul_base_elem(&self, rhs: Self::BaseField) -> Self {
        *self * rhs
    }

    fn mul_assign_base_elem(&mut self, rhs: Self::BaseField) {
        *self *= rhs;
    }

    fn from_uniform_bytes(bytes: &[u8; 32]) -> Self {
        let v = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        BabyBearField(BabyBear::new(v % P))
    }

    fn serialize_into(&self, buffer: &mut [u8]) {
        let canonical = self.0.as_canonical_u32();
        buffer[..4].copy_from_slice(&canonical.to_le_bytes());
    }

    fn deserialize_from(buffer: &[u8]) -> Self {
        let v = u32::from_le_bytes([buffer[0], buffer[1], buffer[2], buffer[3]]);
        assert!(v < P);
        BabyBearField(BabyBear::new(v))
    }
}

/// Compute the 2^27-th primitive root of unity for Baby Bear.
/// Generator of multiplicative group is 31. P-1 = 2^27 * 15.
/// Root = 31^15 mod P.
#[cfg(test)]
fn compute_root_of_unity() -> BabyBear {
    let g = BabyBear::new(31);
    // 31^15 = 31^8 * 31^4 * 31^2 * 31^1
    let g2 = g * g;
    let g4 = g2 * g2;
    let g8 = g4 * g4;
    g8 * g4 * g2 * g
}

impl FftField for BabyBearField {
    const LOG_ORDER: u32 = 27;
    // Computed via compute_root_of_unity() = 31^15 mod P, then hardcoded.
    // The canonical value is 440564289; BabyBear::new(440564289) converts to Montgomery form.
    const ROOT_OF_UNITY: Self = BabyBearField(BabyBear::new(440564289));
    type FftBaseField = BabyBearField;
}

// ─── BabyBearExt4 (degree-4 extension wrapper) ───────────────────────────────

type P3Ext4 = BinomialExtensionField<BabyBear, 4>;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[repr(transparent)]
pub struct BabyBearExt4(pub P3Ext4);

impl std::ops::Neg for BabyBearExt4 {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self { BabyBearExt4(-self.0) }
}

impl std::ops::Add for BabyBearExt4 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self { BabyBearExt4(self.0 + rhs.0) }
}

impl std::ops::AddAssign for BabyBearExt4 {
    #[inline]
    fn add_assign(&mut self, rhs: Self) { self.0 += rhs.0; }
}

impl std::ops::Sub for BabyBearExt4 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self { BabyBearExt4(self.0 - rhs.0) }
}

impl std::ops::SubAssign for BabyBearExt4 {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) { self.0 -= rhs.0; }
}

impl std::ops::Mul for BabyBearExt4 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self { BabyBearExt4(self.0 * rhs.0) }
}

impl std::ops::MulAssign for BabyBearExt4 {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) { self.0 *= rhs.0; }
}

impl From<u32> for BabyBearExt4 {
    fn from(value: u32) -> Self {
        BabyBearExt4(P3Ext4::from(BabyBear::new(value % P)))
    }
}

impl From<BabyBearField> for BabyBearExt4 {
    fn from(value: BabyBearField) -> Self {
        BabyBearExt4(P3Ext4::from(value.0))
    }
}

/// Access the 4 base-field coefficients of the extension element.
fn ext4_coeffs(e: &P3Ext4) -> &[BabyBear] {
    e.as_basis_coefficients_slice()
}

/// Build an Ext4 from 4 base-field coefficients.
fn ext4_from_coeffs(c0: BabyBear, c1: BabyBear, c2: BabyBear, c3: BabyBear) -> P3Ext4 {
    let coeffs = [c0, c1, c2, c3];
    let mut i = 0;
    P3Ext4::from_basis_coefficients_fn(|_| {
        let c = coeffs[i];
        i += 1;
        c
    })
}

impl Field for BabyBearExt4 {
    const NAME: &'static str = "BabyBearExt4";
    const SIZE: usize = 16; // 4 * 4 bytes
    type BaseField = BabyBearField;

    #[inline]
    fn zero() -> Self { BabyBearExt4(P3Ext4::ZERO) }

    #[inline]
    fn is_zero(&self) -> bool { self.0 == P3Ext4::ZERO }

    #[inline]
    fn one() -> Self { BabyBearExt4(P3Ext4::ONE) }

    fn random(mut rng: impl RngCore) -> Self {
        let c0 = BabyBear::new(rng.next_u32() % P);
        let c1 = BabyBear::new(rng.next_u32() % P);
        let c2 = BabyBear::new(rng.next_u32() % P);
        let c3 = BabyBear::new(rng.next_u32() % P);
        BabyBearExt4(ext4_from_coeffs(c0, c1, c2, c3))
    }

    fn inv_2() -> Self {
        BabyBearExt4::from(BabyBearField::inv_2())
    }

    fn exp(&self, mut exponent: usize) -> Self {
        let mut res = Self::one();
        let mut t = *self;
        while exponent != 0 {
            if (exponent & 1) == 1 {
                res *= t;
            }
            t *= t;
            exponent >>= 1;
        }
        res
    }

    fn inv(&self) -> Option<Self> {
        self.0.try_inverse().map(BabyBearExt4)
    }

    fn add_base_elem(&self, rhs: Self::BaseField) -> Self {
        BabyBearExt4(self.0 + P3Ext4::from(rhs.0))
    }

    fn add_assign_base_elem(&mut self, rhs: Self::BaseField) {
        self.0 += P3Ext4::from(rhs.0);
    }

    fn mul_base_elem(&self, rhs: Self::BaseField) -> Self {
        // Multiply each component by the base element
        let coeffs = ext4_coeffs(&self.0);
        BabyBearExt4(ext4_from_coeffs(
            coeffs[0] * rhs.0,
            coeffs[1] * rhs.0,
            coeffs[2] * rhs.0,
            coeffs[3] * rhs.0,
        ))
    }

    fn mul_assign_base_elem(&mut self, rhs: Self::BaseField) {
        *self = self.mul_base_elem(rhs);
    }

    fn from_uniform_bytes(bytes: &[u8; 32]) -> Self {
        let c0 = BabyBear::new(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) % P);
        let c1 = BabyBear::new(u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]) % P);
        let c2 = BabyBear::new(u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) % P);
        let c3 = BabyBear::new(u32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]) % P);
        BabyBearExt4(ext4_from_coeffs(c0, c1, c2, c3))
    }

    fn serialize_into(&self, buffer: &mut [u8]) {
        let coeffs = ext4_coeffs(&self.0);
        for i in 0..4 {
            let canonical = coeffs[i].as_canonical_u32();
            buffer[i * 4..(i + 1) * 4].copy_from_slice(&canonical.to_le_bytes());
        }
    }

    fn deserialize_from(buffer: &[u8]) -> Self {
        let mut coeffs = [BabyBear::ZERO; 4];
        for i in 0..4 {
            let v = u32::from_le_bytes([
                buffer[i * 4],
                buffer[i * 4 + 1],
                buffer[i * 4 + 2],
                buffer[i * 4 + 3],
            ]);
            assert!(v < P);
            coeffs[i] = BabyBear::new(v);
        }
        BabyBearExt4(ext4_from_coeffs(coeffs[0], coeffs[1], coeffs[2], coeffs[3]))
    }
}

impl FftField for BabyBearExt4 {
    const LOG_ORDER: u32 = 27;
    // Embed the base-field root into Ext4 via unsafe transmute, since
    // BinomialExtensionField::new() is pub(crate) and cannot be called from outside.
    // BinomialExtensionField is #[repr(transparent)] over [BabyBear; 4].
    const ROOT_OF_UNITY: Self = {
        let arr: [BabyBear; 4] = [
            BabyBear::new(440564289),
            BabyBear::new(0),
            BabyBear::new(0),
            BabyBear::new(0),
        ];
        // SAFETY: BinomialExtensionField<BabyBear, 4> is #[repr(transparent)]
        // wrapping [BabyBear; 4] (with a zero-sized PhantomData).
        BabyBearExt4(unsafe { std::mem::transmute::<[BabyBear; 4], P3Ext4>(arr) })
    };
    type FftBaseField = BabyBearField;
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mul_group::Radix2Group;
    use rand::rngs::SmallRng;
    use rand::SeedableRng;

    #[test]
    fn babybear_root_of_unity_verify() {
        // Verify ROOT_OF_UNITY is correct: 31^15 mod P
        let root = compute_root_of_unity();
        let expected = BabyBear::new(440564289);
        assert_eq!(
            root.as_canonical_u32(),
            expected.as_canonical_u32(),
            "ROOT_OF_UNITY canonical value mismatch"
        );

        // Verify it has order exactly 2^27
        let r = BabyBearField::ROOT_OF_UNITY;
        let r_pow = r.exp(1 << 27);
        assert_eq!(r_pow, BabyBearField::one(), "ROOT_OF_UNITY^(2^27) != 1");

        // Verify it's primitive (not a root of smaller order)
        let r_half = r.exp(1 << 26);
        assert_ne!(r_half, BabyBearField::one(), "ROOT_OF_UNITY has order < 2^27");
    }

    #[test]
    fn babybear_basic_arithmetic() {
        let a = BabyBearField::from(7u32);
        let b = BabyBearField::from(11u32);

        // add
        let c = a + b;
        assert_eq!(c, BabyBearField::from(18u32));

        // sub
        let d = b - a;
        assert_eq!(d, BabyBearField::from(4u32));

        // mul
        let e = a * b;
        assert_eq!(e, BabyBearField::from(77u32));

        // neg
        let f = -a;
        assert_eq!(a + f, BabyBearField::zero());

        // inv
        let g = a.inv().unwrap();
        assert_eq!(a * g, BabyBearField::one());

        // zero inv
        assert!(BabyBearField::zero().inv().is_none());
    }

    #[test]
    fn babybear_field_axioms() {
        let mut rng = SmallRng::seed_from_u64(42);
        for _ in 0..100 {
            let a = BabyBearField::random(&mut rng);
            let b = BabyBearField::random(&mut rng);
            let c = BabyBearField::random(&mut rng);

            // Commutativity
            assert_eq!(a + b, b + a);
            assert_eq!(a * b, b * a);

            // Associativity
            assert_eq!((a + b) + c, a + (b + c));
            assert_eq!((a * b) * c, a * (b * c));

            // Distributivity
            assert_eq!(a * (b + c), a * b + a * c);

            // Identity
            assert_eq!(a + BabyBearField::zero(), a);
            assert_eq!(a * BabyBearField::one(), a);
        }
    }

    #[test]
    fn babybear_serialize_roundtrip() {
        let mut rng = SmallRng::seed_from_u64(42);
        for _ in 0..100 {
            let a = BabyBearField::random(&mut rng);
            let mut buf = [0u8; 4];
            a.serialize_into(&mut buf);
            let b = BabyBearField::deserialize_from(&buf);
            assert_eq!(a, b);
        }
    }

    #[test]
    fn babybear_ext4_basic() {
        let a = BabyBearExt4::from(7u32);
        let b = BabyBearExt4::from(11u32);

        // Embedding preserves arithmetic
        let c = a + b;
        assert_eq!(c, BabyBearExt4::from(18u32));

        let d = a * b;
        assert_eq!(d, BabyBearExt4::from(77u32));

        // inv
        let e = BabyBearExt4::random(&mut SmallRng::seed_from_u64(42));
        let f = e.inv().unwrap();
        assert_eq!(e * f, BabyBearExt4::one());
    }

    #[test]
    fn babybear_ext4_mul_base_elem() {
        let mut rng = SmallRng::seed_from_u64(42);
        for _ in 0..100 {
            let ext = BabyBearExt4::random(&mut rng);
            let base = BabyBearField::random(&mut rng);

            let via_trait = ext.mul_base_elem(base);
            let via_full = ext * BabyBearExt4::from(base);
            assert_eq!(via_trait, via_full);
        }
    }

    #[test]
    fn babybear_ext4_serialize_roundtrip() {
        let mut rng = SmallRng::seed_from_u64(42);
        for _ in 0..100 {
            let a = BabyBearExt4::random(&mut rng);
            let mut buf = [0u8; 16];
            a.serialize_into(&mut buf);
            let b = BabyBearExt4::deserialize_from(&buf);
            assert_eq!(a, b);
        }
    }

    #[test]
    fn babybear_ext4_field_axioms() {
        let mut rng = SmallRng::seed_from_u64(42);
        for _ in 0..50 {
            let a = BabyBearExt4::random(&mut rng);
            let b = BabyBearExt4::random(&mut rng);
            let c = BabyBearExt4::random(&mut rng);

            assert_eq!(a + b, b + a);
            assert_eq!(a * b, b * a);
            assert_eq!((a + b) + c, a + (b + c));
            assert_eq!((a * b) * c, a * (b * c));
            assert_eq!(a * (b + c), a * b + a * c);
            assert_eq!(a + BabyBearExt4::zero(), a);
            assert_eq!(a * BabyBearExt4::one(), a);
        }
    }

    #[test]
    fn babybear_ext4_fft_root() {
        let r = BabyBearExt4::ROOT_OF_UNITY;
        assert_eq!(r.exp(1 << 27), BabyBearExt4::one());
        assert_ne!(r.exp(1 << 26), BabyBearExt4::one());
    }

    #[test]
    fn babybear_fft_roundtrip() {
        // Small FFT round-trip using Radix2Group
        let log_n = 4u32;
        let group = Radix2Group::<BabyBearField>::new(log_n);
        let n = 1 << log_n;

        let mut rng = SmallRng::seed_from_u64(42);
        let original: Vec<BabyBearField> = (0..n).map(|_| BabyBearField::random(&mut rng)).collect();

        // FFT then IFFT
        let fft_result = group.fft(original.clone());
        let recovered = group.ifft(fft_result);

        assert_eq!(recovered, original, "FFT round-trip mismatch");
    }
}
