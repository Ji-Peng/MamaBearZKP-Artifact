use std::{
    fmt::Debug,
    ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

use ark_ec::pairing::Pairing;
use ark_ff::UniformRand;
use rand::RngCore;

pub mod babybear;
pub mod bn_254;
pub mod goldilocks64;
#[cfg(target_arch = "x86_64")]
pub mod mamabear;

pub trait Field:
    Copy
    + Clone
    + Debug
    + Default
    + PartialEq
    + From<u32>
    + From<Self::BaseField>
    + Neg<Output = Self>
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + AddAssign
    + SubAssign
    + MulAssign
{
    const NAME: &'static str;
    const SIZE: usize;
    type BaseField: Field;

    fn zero() -> Self;
    fn is_zero(&self) -> bool;
    fn one() -> Self;
    fn random(rng: impl RngCore) -> Self;
    fn square(&self) -> Self {
        self.clone() * self.clone()
    }
    fn inv_2() -> Self;
    fn double(&self) -> Self {
        self.clone() + self.clone()
    }
    fn exp(&self, exponent: usize) -> Self;
    fn inv(&self) -> Option<Self>;
    fn add_base_elem(&self, rhs: Self::BaseField) -> Self;
    fn add_assign_base_elem(&mut self, rhs: Self::BaseField);
    fn mul_base_elem(&self, rhs: Self::BaseField) -> Self;
    fn mul_assign_base_elem(&mut self, rhs: Self::BaseField);
    /// Canonicalize lazy/unreduced representations. Default is a no-op.
    /// Fields with lazy Montgomery arithmetic (e.g. MamaBear) override this to
    /// keep intermediate scratch values within their safe input range.
    #[inline(always)]
    fn reduce_mod_p(self) -> Self {
        self
    }
    fn from_uniform_bytes(bytes: &[u8; 32]) -> Self;
    /// Number of uniform transcript bytes this field wants for a Fiat-Shamir
    /// challenge draw. The default (32) reproduces the historical single-SHA-256-digest
    /// flow exactly. Fields whose prime is too wide for an unbiased draw from 32 bytes
    /// (e.g. p ~ 2^58 per component) override this to request more bytes
    /// so the transcript can gather multiple digests and draw with negligible bias.
    #[inline(always)]
    fn uniform_bytes_needed() -> usize {
        32
    }
    /// Wide Fiat-Shamir draw from `uniform_bytes_needed()` transcript bytes. The default
    /// copies up to the first 32 bytes into a `[u8; 32]` and delegates to
    /// `from_uniform_bytes`, so for any field that keeps `uniform_bytes_needed() == 32`
    /// this is byte-for-byte identical to the historical single-digest flow. Fields with
    /// a wider request override this to consume all the bytes (e.g. a `u128 % p` per
    /// component for a bias <= p / 2^128).
    #[inline(always)]
    fn from_uniform_bytes_wide(bytes: &[u8]) -> Self {
        let mut b = [0u8; 32];
        let n = bytes.len().min(32);
        b[..n].copy_from_slice(&bytes[..n]);
        Self::from_uniform_bytes(&b)
    }
    fn serialize_into(&self, buffer: &mut [u8]);
    fn deserialize_from(buffer: &[u8]) -> Self;
}

pub trait FftField: Field + From<Self::FftBaseField> {
    const LOG_ORDER: u32;
    const ROOT_OF_UNITY: Self;
    type FftBaseField: FftField<BaseField = Self::BaseField>;
}

/// Base scalars that expose an explicit canonical-integer construction boundary. The
/// circuit builders emit raw canonical `u64` witness values; this trait converts them into
/// the field representation (and back) without any implicit Montgomery choreography, so the
/// same builder can target different base primes by their canonical integers.
pub trait CanonicalBaseScalar: Field {
    /// The base prime; a canonical value satisfies `0 <= x < PRIME`.
    const PRIME: u64;
    /// Wrap a canonical integer in `[0, PRIME)` as a field element.
    fn from_canonical_u64(x: u64) -> Self;
    /// The canonical integer representative in `[0, PRIME)`.
    fn to_canonical_u64(&self) -> u64;
}

#[cfg(target_arch = "x86_64")]
impl CanonicalBaseScalar for mamabear::MamaBearScalar {
    const PRIME: u64 = mamabear::P;
    #[inline(always)]
    fn from_canonical_u64(x: u64) -> Self {
        mamabear::MamaBearScalar(x)
    }
    #[inline(always)]
    fn to_canonical_u64(&self) -> u64 {
        use mamabear::LazyReduction;
        self.reduce().0
    }
}

pub trait PairingField: Field {
    type E: Pairing;
    type G1: Into<<Self::E as Pairing>::G1Prepared> + UniformRand + Clone + Copy;
    type G2: Into<<Self::E as Pairing>::G2Prepared> + UniformRand + Clone + Copy;

    fn g1_mul(g1: Self::G1, x: Self) -> Self::G1;
    fn g2_mul(g2: Self::G2, x: Self) -> Self::G2;
}

pub fn batch_inverse<F: Field>(v: &mut [F]) {
    let mut aux = vec![v[0]];
    let len = v.len();
    for i in 1..len {
        aux.push(aux[i - 1] * v[i]);
    }
    let mut prod = aux[len - 1].inv().unwrap();
    for i in (1..len).rev() {
        (prod, v[i]) = (prod * v[i], prod * aux[i - 1]);
    }
    v[0] = prod;
}

pub fn as_bytes_vec<F: Field>(v: &[F]) -> Vec<u8> {
    let mut buffer = vec![0; F::SIZE * v.len()];
    let mut cnt = 0;
    for i in v.iter() {
        i.serialize_into(&mut buffer[cnt..cnt + F::SIZE]);
        cnt += F::SIZE;
    }
    buffer
}

// #[cfg(test)]
// mod tests {
//     use ark_ec::pairing::Pairing;
//     use ark_ff::UniformRand;
//     use rand::rngs::SmallRng;
//     use rand::SeedableRng;

//     use super::{bn_254::Bn254F, Field, PairingField};

//     #[test]
//     fn serialize() {
//         let mut rng = SmallRng::seed_from_u64(1);
//         for _ in 0..100 {
//             let f = Bn254F::random(&mut rng);
//             let mut buffer = [0u8; 64];
//             f.serialize_into(&mut buffer);
//             let g = Bn254F::deserialize_from(&buffer);
//             assert_eq!(f, g);
//         }
//     }

//     // fn pairing<F: PairingField>() {
//     //     let mut rng = SmallRng::seed_from_u64(1);
//     //     for _ in 0..10 {
//     //         let g1 = F::G1::rand(&mut rng);
//     //         let g2 = F::G2::rand(&mut rng);
//     //         let x = F::random(&mut rng);
//     //         assert_eq!(
//     //             F::E::pairing(F::g1_mul(g1, x), g2),
//     //             F::E::pairing(g1, F::g2_mul(g2, x))
//     //         );
//     //     }
//     // }

//     // #[test]
//     // fn pairing_test() {
//     //     pairing::<Bn254F>();
//     // }
// }
