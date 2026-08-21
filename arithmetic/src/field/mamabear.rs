#![allow(non_snake_case)]
#![allow(clippy::missing_safety_doc)]

///! Mama Bear Field Implementation with AVX-512IFMA Support
///!
///! Mama Bear is a prime field with modulus P = 2^49 - 2^34 + 1.
///! This implementation leverages AVX-512IFMA instructions for efficient vectorized arithmetic.
///!
///! The field multiplication is performed using Montgomery multiplication.
///! Lazy reduction techniques are employed to optimize addition and subtraction operations.
///!
///! (2^52 - 1) / P > 8, which is useful for range analysis when using lazy reduction.
use core::arch::x86_64::*;
use core::fmt::{self, Debug};
use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use rand::RngCore;

// --- Constants Definition (Mama Bear) ---
// P = 2^49 - 2^34 + 1. We use R = 2^52 for Montgomery multiplication with AVX-512IFMA.
pub const P: u64 = 562932773552129;

/// Lazy reduction property: allows the value to accumulate in multiple
/// additions without reduction, only performs fast reduction (inaccurate)
/// or full reduction (accurate) when necessary.
pub trait LazyReduction: Sized + Copy {
    /// Pure vector addition, without modulo reduction.
    /// Warning: The caller must ensure that the accumulated result does not overflow u64 (AVX-512 64-bit adder bit width).
    fn lazy_add(self, rhs: Self) -> Self;
    /// Add broadcasted multiples of P.
    fn lazy_add_xp(self, x: u8) -> Self;
    /// Pure vector subtraction, without modulo reduction.
    /// Warning: The caller must ensure that the accumulated result does not overflow u64 (AVX-512 64-bit adder bit width).
    fn lazy_sub(self, rhs: Self) -> Self;
    /// Conditional subtraction of x*P. This is useful when you don't want to perform reduction.
    fn con_sub_xp(self, x: u8) -> Self;
    /// Fast reduction, utilizing the sparsity of the modulus: `2^49 ≡ 2^34 - 1
    /// (mod P)`, so `x = x_lo + 2^49 * x_hi` folds to `x_lo + (2^34 - 1) * x_hi`
    /// in two shifts and a subtract.
    ///
    /// RANGE, stated exactly because it is *not* `[0, 2P)`: for an arbitrary
    /// `u64` input the image is `[0, 2^50 - 2^34 - 2^15]`, whose top element is
    /// `2P + (2^34 - 2^15 - 1)` — i.e. `reduce_fast` DOES exceed `2P`, though
    /// only for inputs `>= 2^64 - 2^49` (equivalently `x_hi == 2^15 - 1`, the
    /// largest possible high part). The bound honoured everywhere in this file
    /// is therefore `[0, 2.0001P)`, and `[0, 2^50)` is the convenient slack
    /// form of it. Any comment claiming `reduce_fast -> [0, 2P)` is wrong at
    /// the very top of the `u64` range; write `[0, 2^50)` instead.
    ///
    /// The `reduce_2p` / `reduce` ladder below is unaffected: one conditional
    /// subtraction of `P` maps `[0, 2P + P)` into `[0, 2P)`, and a second maps
    /// it into `[0, P)`, so both remain exact.
    fn reduce_fast(self) -> Self;
    /// Reduction to [0, 2p).
    fn reduce_2p(self) -> Self;
    /// Precise reduction, output range is [0, p).
    fn reduce(self) -> Self;
}

use super::Field;

// Unsigned representation with R=2^52, matching the AVX-512IFMA packed version.
// This allows zero-cost conversion between scalar and packed SIMD representations.
#[repr(transparent)]
#[derive(Clone, Copy, Default, PartialEq)]
pub struct MamaBearScalar(pub u64);

impl fmt::Debug for MamaBearScalar {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Mask for lower 52 bits.
const MASK52: u64 = (1u64 << 52) - 1;

impl MamaBearScalar {
    /// Convert normal representation to Montgomery form: x -> x * R mod P, where R = 2^52.
    /// Output is canonical (in [0, P)).
    ///
    /// `mont_mul` output is only guaranteed to be in [0, 1.5P) — NOT canonical.
    /// Without the trailing `con_sub_xp(1)`, callers that feed the result into
    /// `Sub` (rhs precondition: < P) or `Neg` (self precondition: < P) can get
    /// canonically-WRONG results when a component lands in [P, 1.5P). This bit
    /// us in `verify_ell0` at Ext3 for the SHA-256 nv=22 corner case; we now
    /// canonicalize at the source to make the contract honest.
    #[inline(always)]
    pub fn to_montgomery(self) -> Self {
        // R^2 % P, where R = 2^52
        const R2_MOD_P: u64 = 15393129233472;
        let t = Self::mont_mul(self.0, R2_MOD_P); // [0, 1.5P)
        Self(t.min(t.wrapping_sub(P))) // con_sub_xp(1): -> [0, P)
    }

    /// Convert Montgomery form to normal representation: x -> x * R^{-1} mod P.
    /// Output is in [0, P) — canonical representation.
    #[inline(always)]
    pub fn from_montgomery(self) -> Self {
        let t = Self(Self::mont_mul(self.0, 1)); // [0, 2P)
        Self(t.0.min(t.0.wrapping_sub(P))) // con_sub_xp(1): → [0, P)
    }

    /// Modular multiplication for raw-form values: (a * b) mod P.
    /// Internally converts to Montgomery, multiplies, converts back.
    #[inline(always)]
    pub fn raw_mul(self, rhs: Self) -> Self {
        // a_mont * b_mont = (a*R)(b*R)R^{-1} = a*b*R (Montgomery form of a*b)
        // from_montgomery then gives a*b
        Self(Self::mont_mul(self.to_montgomery().0, rhs.to_montgomery().0)).from_montgomery()
    }

    /// Montgomery Multiplication: (a * b * R^{-1}) mod P, where R = 2^52.
    ///
    /// Matches the AVX-512IFMA `mont_mul_avx512ifma` algorithm exactly.
    ///
    /// # Range analysis
    ///
    /// Same as the packed version:
    /// - For a*b in [0, p*R): result in [0, 2P).
    /// - For a*b in [0, m*p²): result in [0, m*p²/R + P).
    ///   - m=4 (both in [0,2P)): result in [0, 1.5P)
    ///   - m=16 (both in [0,4P)): result in [0, 3P)
    #[inline(always)]
    pub fn mont_mul(a: u64, b: u64) -> u64 {
        const PINV: u64 = 3940666853818369; // P^{-1} mod R
        let ab = a as u128 * b as u128;
        let c0 = (ab as u64) & MASK52;             // low 52 bits of a*b
        let c1 = (ab >> 52) as u64 + P;            // high bits + P
        let t0 = (c0 as u128 * PINV as u128) as u64 & MASK52; // low 52 bits of c0*pinv
        let t1 = (t0 as u128 * P as u128 >> 52) as u64;       // high bits of t0*P
        c1 - t1
    }
}

impl From<u32> for MamaBearScalar {
    fn from(x: u32) -> Self {
        Self(x as u64)
    }
}
impl From<u64> for MamaBearScalar {
    fn from(x: u64) -> Self {
        Self(x)
    }
}
impl Neg for MamaBearScalar {
    type Output = Self;
    /// Negate: (2P - self) reduced to canonical [0, P).
    ///
    /// Accepts non-canonical `self ∈ [0, 2P)` (e.g. the output of `Mul` /
    /// `mont_mul` / `reduce_fast` / `lazy_add*` chains). The older `P - self`
    /// form silently underflowed for `self ≥ P` and produced canonically-wrong
    /// results — see `test_neg_noncanonical_canonical`.
    #[inline(always)]
    fn neg(self) -> Self {
        // 2P - self fits u64 for self < 2P (2P - 0 = 2P; 2P - (2P-1) = 1).
        let diff = (2 * P).wrapping_sub(self.0);      // [1, 2P]
        let d1 = diff.min(diff.wrapping_sub(P));       // con_sub_xp(1): [0, P]
        Self(d1.min(d1.wrapping_sub(P)))               // handle the d1 == P boundary: [0, P)
    }
}
impl Add for MamaBearScalar {
    type Output = Self;
    /// Modular add: `(self + rhs).con_sub_xp(1)`. Matches
    /// `PackedMamaBearAVX512::add` exactly, which is why it is a single
    /// conditional subtraction rather than a full reduction.
    ///
    /// RANGE, and this is the asymmetry to watch: `Sub` and `Neg` accept
    /// non-canonical operands in `[0, 2P)` and return CANONICAL results, but
    /// `Add` does NOT preserve `[0, 2P)` — one `con_sub_xp(1)` on a sum in
    /// `[0, 4P)` only guarantees `[0, 3P)`. So `a + b` on two `[0, 2P)`
    /// operands can land at or above `2P`, and feeding that straight into `Neg`
    /// (or into `Sub` on either side) leaves the `2P` window those two require:
    /// their `wrapping_sub` then contributes `2^64`, which is not a multiple of
    /// `P` (`2^64 mod P = 2^34 - 2^15 - 1`), so the result is not congruent to
    /// the intended value at all. Canonical operands are fine — `a, b < P`
    /// gives a sum `< 2P` and hence a canonical result — the hazard is only for
    /// values arriving from a lazy chain. Insert `reduce_2p()` before negating
    /// or subtracting with the output of a chain of `Add`s.
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        let sum = self.0.wrapping_add(rhs.0);
        let sum_minus_p = sum.wrapping_sub(P);
        Self(sum.min(sum_minus_p))
    }
}
impl AddAssign for MamaBearScalar {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}
impl Sub for MamaBearScalar {
    type Output = Self;
    /// Modular sub reduced to canonical [0, P).
    ///
    /// Accepts non-canonical `self, rhs ∈ [0, 2P)` (e.g. the output of `Mul` /
    /// `mont_mul` / `reduce_fast` / `lazy_add*` chains). The older `self + P - rhs`
    /// form silently underflowed for `rhs ≥ P` and produced canonically-wrong
    /// results — that bit the Ext3 verifier at specific Fiat-Shamir challenges,
    /// see `test_sub_noncanonical_rhs_canonical`.
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        let tmp = self.0.wrapping_add(2 * P);          // self < 2P -> tmp < 4P
        let diff = tmp.wrapping_sub(rhs.0);             // rhs < 2P  -> diff in (0, 4P)
        let d = diff.min(diff.wrapping_sub(2 * P));     // con_sub_xp(2): [0, 2P)
        Self(d.min(d.wrapping_sub(P)))                  // con_sub_xp(1): [0, P)
    }
}
impl SubAssign for MamaBearScalar {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}
impl Mul for MamaBearScalar {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        Self(Self::mont_mul(self.0, rhs.0))
    }
}
impl MulAssign for MamaBearScalar {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl Field for MamaBearScalar {
    const NAME: &'static str = "MamaBear";
    const SIZE: usize = 7;
    type BaseField = Self;
    fn zero() -> Self {
        Self(0)
    }
    fn is_zero(&self) -> bool {
        self.0 == 0
    }
    /// The RAW (non-Montgomery) representative of 1 — NOT the identity of
    /// [`Mul`], which is Montgomery multiplication.
    ///
    /// This type's `u64` payload carries a Montgomery representative in the hot
    /// paths (`R = 2^52`), but `zero`/`one`/`random`/`inv_2` all produce RAW
    /// values, and the convention of which domain a given value lives in is
    /// carried by the call site, not by the type. The consequence is that this
    /// `Field` impl is NOT a lawful ring: `one() * x == mont_mul(1, x) ==
    /// x * R^{-1} mod P`, which differs from `x` for every nonzero canonical
    /// `x`. The multiplicative identity of `Mul` is `R mod P`.
    ///
    /// Every call site that needs a `Mul` identity must therefore write
    /// `Field::one().to_montgomery()` (or seed with `R mod P` directly, as
    /// [`Field::exp`] does). That obligation is enforced by review, not by the
    /// type system; see the `Radix2Group` note further down for a generic
    /// helper this rules out. Do not "simplify" a `one().to_montgomery()` to a
    /// bare `one()`.
    fn one() -> Self {
        Self(1)
    }
    fn random(mut rng: impl RngCore) -> Self {
        Self(rng.next_u64() % P)
    }
    fn inv_2() -> Self {
        Self((P + 1) / 2)
    }
    /// Binary exponentiation for raw-form values.
    /// Converts to Montgomery domain internally for correct mont_mul chaining,
    /// then converts back to raw form.
    fn exp(&self, mut e: usize) -> Self {
        // R mod P — the Montgomery representation of 1
        const R_MOD_P: u64 = (1u64 << 52) % P;
        let mut result = Self(R_MOD_P);
        let mut base = self.to_montgomery();
        while e > 0 {
            if e & 1 == 1 {
                result = Self(Self::mont_mul(result.0, base.0));
            }
            base = Self(Self::mont_mul(base.0, base.0));
            e >>= 1;
        }
        result.from_montgomery()
    }
    /// Modular inverse via Fermat's little theorem: a^{P-2} mod P.
    fn inv(&self) -> Option<Self> {
        if self.is_zero() {
            return None;
        }
        Some(self.exp(P as usize - 2))
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
    #[inline(always)]
    fn reduce_mod_p(self) -> Self {
        <Self as LazyReduction>::reduce(self)
    }
    /// Draw a field element from transcript bytes: one little-endian `u64`
    /// reduced mod `P`.
    ///
    /// BIAS, stated because a modular reduction of a power-of-two range is
    /// never exactly uniform: writing `2^64 = qP + r`, the total-variation
    /// distance from uniform is `r(P - r)/(2^64 P)`. Here
    /// `r = 2^64 mod P = 2^34 - 2^15 - 1`, which sits unusually high for a
    /// 49-bit modulus because `P = 2^49 - 2^34 + 1` is Solinas, giving
    /// TV ~ `2^-30.0` per component. That is nonzero but far below any
    /// soundness threshold this system claims, and it accumulates only
    /// linearly in the number of challenges drawn per proof.
    ///
    /// Widening the draw is the clean fix — reducing a 512-bit integer instead
    /// of a 64-bit one takes the per-component TV to about `2^-465` — but it
    /// moves every challenge value and hence every proof byte, so it is not
    /// applied on this artifact branch, whose recorded measurements are pinned
    /// to these exact bytes.
    fn from_uniform_bytes(b: &[u8; 32]) -> Self {
        let mut arr = [0u8; 8];
        arr.copy_from_slice(&b[0..8]);
        Self(u64::from_le_bytes(arr) % P)
    }
    /// Emit the low 7 bytes (56 bits) — the transcript / Merkle-leaf wire form.
    ///
    /// CONTRACT: `self` must be CANONICAL (`< P`). The format is injective on
    /// canonical values (`P < 2^49 < 2^56`) but not on the whole `u64`: the low
    /// 56 bits are kept and everything above is discarded, so e.g. `0` and
    /// `2^56` serialize identically. Serializing an unreduced accumulator would
    /// therefore weaken transcript BINDING, not merely canonicity — two
    /// distinct claimed values could hash to the same transcript bytes.
    ///
    /// Enforced by `debug_assert` rather than by an unconditional `reduce()`:
    /// this sits on the FRI Merkle-leaf hot path, and MamaBear's UNSIGNED
    /// representation makes canonicity a maintainable invariant of the callers.
    /// A field backend using a SIGNED representation could not rely on that —
    /// there `x` and `x - p` are both valid reps, so no caller-side invariant
    /// establishes uniqueness and the boundary must canonicalize itself.
    /// The invariant here was checked empirically by promoting this to a hard
    /// `assert` and running the end-to-end SNARK prove/verify suite, which
    /// reported zero violations.
    fn serialize_into(&self, buffer: &mut [u8]) {
        debug_assert!(
            self.0 < P,
            "MamaBearScalar::serialize_into on a non-canonical value ({}): the \
             7-byte wire format keeps only the low 56 bits and is injective on \
             [0, P) only — reduce() before crossing the byte boundary",
            self.0
        );
        buffer[0..7].copy_from_slice(&self.0.to_le_bytes()[..7]);
    }
    fn deserialize_from(b: &[u8]) -> Self {
        assert!(b.len() >= 7, "Buffer too small for MamaBearScalar");
        let mut arr = [0u8; 8];
        arr[..7].copy_from_slice(&b[0..7]);
        Self(u64::from_le_bytes(arr) % P)
    }
}

impl LazyReduction for MamaBearScalar {
    #[inline(always)]
    fn lazy_add(self, rhs: Self) -> Self {
        Self(self.0.wrapping_add(rhs.0))
    }
    #[inline(always)]
    fn lazy_add_xp(self, x: u8) -> Self {
        Self(self.0.wrapping_add(x as u64 * P))
    }
    #[inline(always)]
    fn lazy_sub(self, rhs: Self) -> Self {
        Self(self.0.wrapping_sub(rhs.0))
    }
    /// Conditional subtraction of x*P: min(self, self - x*P).
    /// If self < x*P, wrapping_sub produces a huge value, min picks self.
    #[inline(always)]
    fn con_sub_xp(self, x: u8) -> Self {
        let xp = x as u64 * P;
        let diff = self.0.wrapping_sub(xp);
        Self(self.0.min(diff))
    }
    /// Fast reduction using P = 2^49 - 2^34 + 1 sparsity.
    /// Input: [0, 2^64).
    /// The maximum value of output is less than < 2^{64-49+34} + 2^49 ~= 2.0001P
    /// Output: [0, 2.0001P).
    #[inline(always)]
    fn reduce_fast(self) -> Self {
        let x_hi = self.0 >> 49;
        let x_lo = self.0 & ((1u64 << 49) - 1);
        let t = (x_hi << 34).wrapping_sub(x_hi);
        Self(x_lo.wrapping_add(t))
    }
    /// Reduce to [0, 2P).
    #[inline(always)]
    fn reduce_2p(self) -> Self {
        self.reduce_fast().con_sub_xp(1)
    }
    /// Precise reduction to [0, P).
    #[inline(always)]
    fn reduce(self) -> Self {
        let r = self.reduce_2p();
        // One more conditional subtraction
        r.con_sub_xp(1)
    }
}

// --- FftField implementations ---
// P - 1 = 2^34 * 32767, so the 2-adic order is 34.
// ROOT_OF_UNITY = 3^((P-1)/2^34) mod P = 57971402726332 (primitive 2^34-th root of unity).
//
// NOTE: Radix2Group<MamaBearScalar> will NOT work correctly because Radix2Group::new
// uses F::one() (raw 1) as the seed and chains F::Mul (mont_mul), which is inconsistent
// for raw-form values. Use the dedicated MamaBear FFT module (fft_mamabear.rs) instead.
// These FftField impls exist for trait compatibility (e.g., DeepFoldParam<F: FftField>).

use super::FftField;

impl FftField for MamaBearScalar {
    const LOG_ORDER: u32 = 34;
    const ROOT_OF_UNITY: Self = MamaBearScalar(57971402726332);
    type FftBaseField = Self;
}

impl FftField for MamaBearScalarExt3 {
    const LOG_ORDER: u32 = 34;
    const ROOT_OF_UNITY: Self = MamaBearScalarExt3 {
        c0: MamaBearScalar(57971402726332),
        c1: MamaBearScalar(0),
        c2: MamaBearScalar(0),
    };
    type FftBaseField = MamaBearScalar;
}

// MamaBearScalarExt3

/// Cubic extension field over MamaBearScalar with irreducible polynomial x^3 - x - 1.
/// Elements are represented as a0 + a1*X + a2*X^2 where X^3 = X + 1.
#[derive(Clone, Copy, Default, PartialEq)]
#[repr(C)]
pub struct MamaBearScalarExt3 {
    pub c0: MamaBearScalar,
    pub c1: MamaBearScalar,
    pub c2: MamaBearScalar,
}

impl fmt::Debug for MamaBearScalarExt3 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {}, {})", self.c0.0, self.c1.0, self.c2.0)
    }
}

impl MamaBearScalarExt3 {
    #[inline(always)]
    pub fn to_montgomery(self) -> Self {
        Self {
            c0: self.c0.to_montgomery(),
            c1: self.c1.to_montgomery(),
            c2: self.c2.to_montgomery(),
        }
    }

    #[inline(always)]
    pub fn from_montgomery(self) -> Self {
        Self {
            c0: self.c0.from_montgomery(),
            c1: self.c1.from_montgomery(),
            c2: self.c2.from_montgomery(),
        }
    }
}

impl From<u32> for MamaBearScalarExt3 {
    fn from(x: u32) -> Self {
        Self {
            c0: MamaBearScalar::from(x),
            c1: MamaBearScalar::zero(),
            c2: MamaBearScalar::zero(),
        }
    }
}
impl From<u64> for MamaBearScalarExt3 {
    fn from(x: u64) -> Self {
        Self {
            c0: MamaBearScalar::from(x),
            c1: MamaBearScalar::zero(),
            c2: MamaBearScalar::zero(),
        }
    }
}
impl From<MamaBearScalar> for MamaBearScalarExt3 {
    fn from(s: MamaBearScalar) -> Self {
        Self {
            c0: s,
            c1: MamaBearScalar::zero(),
            c2: MamaBearScalar::zero(),
        }
    }
}
impl From<[MamaBearScalar; 3]> for MamaBearScalarExt3 {
    fn from(arr: [MamaBearScalar; 3]) -> Self {
        Self {
            c0: arr[0],
            c1: arr[1],
            c2: arr[2],
        }
    }
}
impl Neg for MamaBearScalarExt3 {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        Self {
            c0: -self.c0,
            c1: -self.c1,
            c2: -self.c2,
        }
    }
}
impl Add for MamaBearScalarExt3 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self {
            c0: self.c0 + rhs.c0,
            c1: self.c1 + rhs.c1,
            c2: self.c2 + rhs.c2,
        }
    }
}
impl Sub for MamaBearScalarExt3 {
    type Output = Self;
    /// Safe subtraction: delegates to MamaBearScalar::sub (which adds 2P).
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self {
            c0: self.c0 - rhs.c0,
            c1: self.c1 - rhs.c1,
            c2: self.c2 - rhs.c2,
        }
    }
}
impl Mul for MamaBearScalarExt3 {
    type Output = Self;
    /// Multiplication in F_p[X]/(X^3 - X - 1) using 6-multiplication Karatsuba.
    /// Matches the packed AVX-512 version's unsigned arithmetic pattern.
    ///
    /// Given a = a0 + a1·X + a2·X² and b = b0 + b1·X + b2·X², with X³ = X + 1:
    ///   c0 = v0 + v12
    ///   c1 = v01 + v12 + v2
    ///   c2 = v02 + v1 + v2
    ///
    /// # Range Analysis (unsigned, R = 2^52)
    ///
    /// Inputs per component: `[0, 4P)`, i.e. just under `2^51`. The pairwise
    /// `lazy_add`s below then reach `8P < 2^52`, which is the real ceiling —
    /// `mont_mul`'s `a*b` must stay inside `u128` and, in the packed twin, each
    /// operand must fit the 52-bit `madd52` window.
    ///
    /// `mont_mul(x, y) = floor(xy/R) + P - floor(mP/R)` with `m < R`, so it is
    /// bounded above by `xy/R + P` and below by `floor(xy/R)`. Hence:
    ///
    /// - `v0, v1, v2` (operands `< 4P`): `< 16P^2/R + P = 3P`.
    /// - `s01, s02, s12` (operands `< 8P`): `< 64P^2/R + P = 9P` — NOT `3P`.
    ///   The `3P` figure applies only to operands `< 4P`, and the `sij` inputs
    ///   are sums of two components.
    /// - `vij = sij + 4P - vi - vj`: upper bound `9P + 4P = 13P`, not `7P`.
    /// - `c0 = v0 + v12 < 16P`; `c1 = v01 + v12 + v2 < 29P`;
    ///   `c2 = v02 + v1 + v2 < 19P`. All are far under `2^64` (`29P < 2^54.9`),
    ///   so no accumulator wraps.
    /// - `reduce_fast` on any `u64` lands in `[0, 2^50)` — see the note on
    ///   `LazyReduction::reduce_fast`; it is `[0, 2.0001P)`, NOT `[0, 2P)`.
    ///
    /// WHY THE `+4P` CANNOT UNDERFLOW, which is the one step a worst-case bound
    /// gets wrong: naively `vi + vj` reaches `6P`, so `+4P` looks insufficient.
    /// It is sufficient because `sij` and `vi + vj` are correlated, not
    /// independent. With `x = a_i + a_j` and `y = b_i + b_j` we have
    /// `xy >= a_i b_i + a_j b_j`, so
    /// `sij >= floor(xy/R) >= floor(a_i b_i/R) + floor(a_j b_j/R)`, while
    /// `vi <= floor(a_i b_i/R) + P` and likewise for `vj`. Therefore
    /// `sij + 4P - vi - vj >= 2P > 0`. Any future edit that changes which
    /// products are subtracted must redo THIS argument, not the worst-case one.
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        let v0 = self.c0 * rhs.c0;
        let v1 = self.c1 * rhs.c1;
        let v2 = self.c2 * rhs.c2;

        let s01 = self.c0.lazy_add(self.c1) * rhs.c0.lazy_add(rhs.c1);
        let s02 = self.c0.lazy_add(self.c2) * rhs.c0.lazy_add(rhs.c2);
        let s12 = self.c1.lazy_add(self.c2) * rhs.c1.lazy_add(rhs.c2);

        // Add 4P before subtracting to prevent unsigned underflow. 4P suffices
        // by the correlation argument in the doc comment (sij >= floor(a_i b_i/R)
        // + floor(a_j b_j/R) while vi + vj <= that + 2P), NOT because vi + vj
        // happens to be small — worst-case it reaches 6P.
        // v12 = s12 + 4P - v1 - v2, in [2P, 13P)
        let v12 = s12.lazy_add_xp(4).lazy_sub(v1).lazy_sub(v2);
        let v01 = s01.lazy_add_xp(4).lazy_sub(v0).lazy_sub(v1);
        let v02 = s02.lazy_add_xp(4).lazy_sub(v0).lazy_sub(v2);

        // c0 = v0 + v12, in [0, 16P) → reduce_fast → [0, 2^50)
        let c0 = v0.lazy_add(v12).reduce_fast();
        // c1 = v01 + v12 + v2, in [0, 29P) → reduce_fast → [0, 2^50)
        let c1 = v01.lazy_add(v12).lazy_add(v2).reduce_fast();
        // c2 = v02 + v1 + v2, in [0, 19P) → reduce_fast → [0, 2^50)
        let c2 = v02.lazy_add(v1).lazy_add(v2).reduce_fast();

        Self { c0, c1, c2 }
    }
}
impl AddAssign for MamaBearScalarExt3 {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}
impl SubAssign for MamaBearScalarExt3 {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}
impl MulAssign for MamaBearScalarExt3 {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl Field for MamaBearScalarExt3 {
    const NAME: &'static str = "MamaBearExt3";
    const SIZE: usize = 21;
    type BaseField = MamaBearScalar;
    fn zero() -> Self {
        Self {
            c0: MamaBearScalar::zero(),
            c1: MamaBearScalar::zero(),
            c2: MamaBearScalar::zero(),
        }
    }
    fn is_zero(&self) -> bool {
        self.c0.is_zero() && self.c1.is_zero() && self.c2.is_zero()
    }
    fn one() -> Self {
        Self {
            c0: MamaBearScalar::one(),
            c1: MamaBearScalar::zero(),
            c2: MamaBearScalar::zero(),
        }
    }
    fn random(mut rng: impl RngCore) -> Self {
        Self {
            c0: MamaBearScalar::random(&mut rng),
            c1: MamaBearScalar::random(&mut rng),
            c2: MamaBearScalar::random(&mut rng),
        }
    }
    fn inv_2() -> Self {
        Self {
            c0: MamaBearScalar::inv_2(),
            c1: MamaBearScalar::zero(),
            c2: MamaBearScalar::zero(),
        }
    }
    /// Binary exponentiation in Ext3. Converts to Montgomery domain internally.
    fn exp(&self, mut e: usize) -> Self {
        let mut result = Self::one().to_montgomery();
        let mut base = self.to_montgomery();
        while e > 0 {
            if e & 1 == 1 {
                result = result * base;
            }
            base = base * base;
            e >>= 1;
        }
        result.from_montgomery()
    }
    /// Inverse in Ext3 via norm to base field.
    /// For X^3 = X + 1, the norm N(a) = a * a^P * a^{P^2} is a base field element.
    /// We compute a^{-1} = a^{P^3-2} / (using the formula for the norm map).
    /// Simpler: use the adjugate/norm formula directly.
    fn inv(&self) -> Option<Self> {
        // For a = c0 + c1*X + c2*X^2 with X^3 = X + 1:
        // Use exp-based approach: a^{-1} = a^{P^3-2}
        // Since P^3 doesn't fit in usize, compute via:
        //   a^{P-1} first (gives conjugate-like element)
        //   Then norm(a) = a * a^P * a^{P^2} which is in base field
        //   a^{-1} = (a^P * a^{P^2}) / norm(a)
        //
        // Alternatively, just compute norm directly and use adjugate.
        // For simplicity, use Frobenius: a^P maps X -> X^P.
        // This is complex, so use the direct algebraic formula.
        //
        // Direct approach: solve (c0+c1X+c2X^2)(d0+d1X+d2X^2) = 1 mod X^3-X-1
        // This gives a 3x3 linear system. Use Cramer's rule.
        //
        // For now, use iterated squaring with P as exponent to get a^{P-1},
        // then compute norm.

        // Simple approach: compute using raw_mul directly.
        // Matrix form of multiplication by a = (c0, c1, c2):
        //   [c0  c2  c1+c2]   [d0]   [r0]
        //   [c1  c0+c2  c1+c2] [d1] = [r1]  (wrong - let me derive correctly)
        //   [c2  c1  c0+c2]   [d2]   [r2]
        //
        // For the generic approach, use exp with large exponent.
        // P^3 - 2 doesn't fit in usize, but we can decompose:
        //   a^{-1} = (a^{P-1})^{(P^2+P+1)/(something)} ... too complex.
        //
        // Simplest correct approach: compute a^{P-1} using Frobenius,
        // then the norm, then divide.
        //
        // Fix: need raw_mul for Ext3 too...
        // Actually since exp returns raw values, and * does mont_mul, this is wrong.
        // Let me use a different approach: compute norm = a * a^P * a^{P^2} directly.

        // Frobenius: a^P for Ext3. Since exp works correctly:
        let a_p = self.exp(P as usize);       // a^P (raw form, correct)
        let a_p2 = a_p.exp(P as usize);       // a^{P^2} (raw form, correct)
        // norm = a * a^P * a^{P^2} — should be a base field element
        // Need raw Ext3 multiplication. Use to_montgomery + * + from_montgomery.
        let a_mont = self.to_montgomery();
        let ap_mont = a_p.to_montgomery();
        let ap2_mont = a_p2.to_montgomery();
        let norm_ext = (a_mont * ap_mont) * ap2_mont;
        let norm = norm_ext.from_montgomery();
        // norm should have c1 == 0 and c2 == 0 (it's in the base field)
        debug_assert!(norm.c1.reduce().0 == 0 || norm.c1.reduce().0 == P);
        debug_assert!(norm.c2.reduce().0 == 0 || norm.c2.reduce().0 == P);
        let norm_base = norm.c0.reduce();
        let norm_inv = norm_base.inv()?;
        // a^{-1} = a^{P} * a^{P^2} * norm^{-1}
        let conjugate = (ap_mont * ap2_mont).from_montgomery();
        Some(Self {
            c0: conjugate.c0.raw_mul(norm_inv),
            c1: conjugate.c1.raw_mul(norm_inv),
            c2: conjugate.c2.raw_mul(norm_inv),
        })
    }
    fn add_base_elem(&self, rhs: Self::BaseField) -> Self {
        Self {
            c0: self.c0 + rhs,
            c1: self.c1,
            c2: self.c2,
        }
    }
    fn add_assign_base_elem(&mut self, rhs: Self::BaseField) {
        self.c0 += rhs;
    }
    fn mul_base_elem(&self, rhs: Self::BaseField) -> Self {
        Self {
            c0: self.c0 * rhs,
            c1: self.c1 * rhs,
            c2: self.c2 * rhs,
        }
    }
    fn mul_assign_base_elem(&mut self, rhs: Self::BaseField) {
        self.c0 *= rhs;
        self.c1 *= rhs;
        self.c2 *= rhs;
    }
    #[inline(always)]
    fn reduce_mod_p(self) -> Self {
        Self {
            c0: <MamaBearScalar as LazyReduction>::reduce(self.c0),
            c1: <MamaBearScalar as LazyReduction>::reduce(self.c1),
            c2: <MamaBearScalar as LazyReduction>::reduce(self.c2),
        }
    }
    fn from_uniform_bytes(b: &[u8; 32]) -> Self {
        let mut arr0 = [0u8; 8];
        let mut arr1 = [0u8; 8];
        let mut arr2 = [0u8; 8];
        arr0.copy_from_slice(&b[0..8]);
        arr1.copy_from_slice(&b[8..16]);
        arr2.copy_from_slice(&b[16..24]);
        Self {
            c0: MamaBearScalar(u64::from_le_bytes(arr0) % P),
            c1: MamaBearScalar(u64::from_le_bytes(arr1) % P),
            c2: MamaBearScalar(u64::from_le_bytes(arr2) % P),
        }
    }
    fn serialize_into(&self, buffer: &mut [u8]) {
        self.c0.serialize_into(&mut buffer[0..7]);
        self.c1.serialize_into(&mut buffer[7..14]);
        self.c2.serialize_into(&mut buffer[14..21]);
    }
    fn deserialize_from(b: &[u8]) -> Self {
        assert!(b.len() >= 21, "Buffer too small for MamaBearScalarExt3");
        Self {
            c0: MamaBearScalar::deserialize_from(&b[0..7]),
            c1: MamaBearScalar::deserialize_from(&b[7..14]),
            c2: MamaBearScalar::deserialize_from(&b[14..21]),
        }
    }
}

impl LazyReduction for MamaBearScalarExt3 {
    #[inline(always)]
    fn lazy_add(self, rhs: Self) -> Self {
        Self {
            c0: self.c0.lazy_add(rhs.c0),
            c1: self.c1.lazy_add(rhs.c1),
            c2: self.c2.lazy_add(rhs.c2),
        }
    }
    #[inline(always)]
    fn lazy_add_xp(self, x: u8) -> Self {
        Self {
            c0: self.c0.lazy_add_xp(x),
            c1: self.c1.lazy_add_xp(x),
            c2: self.c2.lazy_add_xp(x),
        }
    }
    #[inline(always)]
    fn lazy_sub(self, rhs: Self) -> Self {
        Self {
            c0: self.c0.lazy_sub(rhs.c0),
            c1: self.c1.lazy_sub(rhs.c1),
            c2: self.c2.lazy_sub(rhs.c2),
        }
    }
    #[inline(always)]
    fn con_sub_xp(self, x: u8) -> Self {
        Self {
            c0: self.c0.con_sub_xp(x),
            c1: self.c1.con_sub_xp(x),
            c2: self.c2.con_sub_xp(x),
        }
    }
    #[inline(always)]
    fn reduce_fast(self) -> Self {
        Self {
            c0: self.c0.reduce_fast(),
            c1: self.c1.reduce_fast(),
            c2: self.c2.reduce_fast(),
        }
    }
    #[inline(always)]
    fn reduce_2p(self) -> Self {
        Self {
            c0: self.c0.reduce_2p(),
            c1: self.c1.reduce_2p(),
            c2: self.c2.reduce_2p(),
        }
    }
    #[inline(always)]
    fn reduce(self) -> Self {
        Self {
            c0: self.c0.reduce(),
            c1: self.c1.reduce(),
            c2: self.c2.reduce(),
        }
    }
}

// --- PackedMamaBearAVX512 Implementation ---

#[repr(C)]
#[derive(Copy, Clone)]
pub union PackedMamaBearAVX512 {
    pub array: [u64; 8],
    pub simd: __m512i,
}

// Ensure Debug prints the array view
impl Debug for PackedMamaBearAVX512 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        unsafe { f.debug_tuple("PackedMamaBear").field(&self.array).finish() }
    }
}

// Default to Zero
impl Default for PackedMamaBearAVX512 {
    fn default() -> Self {
        Self::zero()
    }
}

// Equality Check
impl PartialEq for PackedMamaBearAVX512 {
    fn eq(&self, other: &Self) -> bool {
        unsafe {
            // Compare 8x64 bits
            let cmp = _mm512_cmpeq_epu64_mask(self.simd, other.simd);
            cmp == 0xFF
        }
    }
}

impl PackedMamaBearAVX512 {
    #[inline(always)]
    pub fn broadcast(val: u64) -> Self {
        unsafe {
            Self {
                simd: _mm512_set1_epi64(val as i64),
            }
        }
    }

    #[inline(always)]
    pub fn from_vector(vec: __m512i) -> Self {
        Self { simd: vec }
    }

    #[inline(always)]
    pub fn to_vector(self) -> __m512i {
        unsafe { self.simd }
    }

    #[inline(always)]
    pub fn from_array(arr: [u64; 8]) -> Self {
        Self { array: arr }
    }

    #[inline(always)]
    pub fn to_array(self) -> [u64; 8] {
        unsafe { self.array }
    }

    #[inline(always)]
    pub fn load_scalar_slice(slice: &[MamaBearScalar]) -> Self {
        debug_assert!(slice.len() >= 8);
        unsafe { core::ptr::read_unaligned(slice.as_ptr() as *const Self) }
    }

    #[inline(always)]
    pub fn store_scalar_slice(slice: &mut [MamaBearScalar], value: Self) {
        debug_assert!(slice.len() >= 8);
        unsafe {
            core::ptr::write_unaligned(slice.as_mut_ptr() as *mut Self, value);
        }
    }

    #[inline(always)]
    pub fn permute(self, indices: [u64; 8]) -> Self {
        unsafe {
            let idx = Self::from_array(indices).simd;
            Self::from_vector(_mm512_permutexvar_epi64(idx, self.simd))
        }
    }

    #[inline(always)]
    pub fn permute2(self, other: Self, indices: [u64; 8]) -> Self {
        unsafe {
            let idx = Self::from_array(indices).simd;
            Self::from_vector(_mm512_permutex2var_epi64(self.simd, idx, other.simd))
        }
    }

    #[inline(always)]
    pub fn mask_blend(self, other: Self, mask: u8) -> Self {
        unsafe {
            Self::from_vector(_mm512_mask_blend_epi64(mask, self.simd, other.simd))
        }
    }

    /// Convert normal representation to Montgomery form: x -> x * R mod P
    /// Output is canonical per lane (each lane in [0, P)).
    ///
    /// See `MamaBearScalar::to_montgomery` for why canonicalizing here is
    /// required: downstream callers that feed the result into `Sub` / `Neg`
    /// silently produce canonically-wrong results when a lane lands in
    /// [P, 1.5P) — the output of `mont_mul_avx512ifma` without reduction.
    #[inline(always)]
    pub fn to_montgomery(self) -> Self {
        // R^2 % P, where R = 2^52
        const R2_MOD_P: u64 = 15393129233472;
        unsafe {
            let r2_mod_q = _mm512_set1_epi64(R2_MOD_P as i64);
            let mont = Self::mont_mul_avx512ifma(self.simd, r2_mod_q); // [0, 1.5P)
            // con_sub_xp(1): -> [0, P) per lane.
            let v_p = _mm512_set1_epi64(P as i64);
            let diff = _mm512_sub_epi64(mont, v_p);
            let canonical = _mm512_min_epu64(mont, diff);
            Self::from_vector(canonical)
        }
    }

    /// Convert Montgomery form to normal representation: x * R mod P -> x.
    /// Output is in [0, P) — canonical representation.
    #[inline(always)]
    pub fn from_montgomery(self) -> Self {
        unsafe {
            let one = _mm512_set1_epi64(1i64);
            let normal = Self::mont_mul_avx512ifma(self.simd, one); // [0, 2P)
            // con_sub_xp(1) to canonicalize to [0, P)
            let v_p = _mm512_set1_epi64(P as i64);
            let diff = _mm512_sub_epi64(normal, v_p);
            let canonical = _mm512_min_epu64(normal, diff);
            Self::from_vector(canonical)
        }
    }

    /// Montgomery Multiplication using AVX-512IFMA instructions.
    ///
    /// Computes (a * b * R^{-1}) mod p, where R = 2^52 and p is the Mama Bear modulus.
    ///
    /// # Range Analysis
    ///
    /// ## Inputs
    /// a, b in [0, 2^52)
    ///
    /// ## Outputs
    /// For a\*b in [0,p\*R), the result is in [0,2p), where p\*R > 8p.
    ///
    /// For a\*b in [p\*R, (R-1)^2], the result is in [0,p+R).
    ///
    /// For a\*b in [0, mp^2), the result is in [0, mp^2/R+p).
    /// For example:
    /// (1) m=16 -> result in [0, 3p).
    /// (2) m=18 -> result in [0, 3.25p).
    /// (3) m=32 -> result in [0, 5p).
    /// (4) m=9  -> result in [0, 2.13p).
    /// (5) m=3*2.13 -> result in [0, 1.8p).
    /// (6) m=4  -> result in [0, 1.5p).
    /// (7) m=2  -> result in [0, 1.25p).
    /// (8) m=3  -> result in [0, 1.38p).
    ///
    /// ## Cases
    /// Case 1: When calculating the multiplication gate, i.e., s\*l\*r, let's determine if we are allowed to directly compute by mont_mul(mont_mul(s,l),r) if each operand is in the range [0,3p). s\*l is in [0,9p^2), and the result obtained using mont_mul is in the range [0,2.13p). Therefore, we can directly use mont_mul without additional reduction.
    ///
    /// Case 2: s\*l\*r, if each operand is in the range [0,2p). s\*l is in [0,4p^2), and the result obtained using mont_mul is in the range [0,1.5p). Then, s\*l\*r using mont_mul will be in the range [0, 1.38p).
    #[inline(always)]
    unsafe fn mont_mul_avx512ifma(a: __m512i, b: __m512i) -> __m512i {
        // Modular inverse of P mod R
        const PINV: u64 = 3940666853818369;
        unsafe {
            let v_p = _mm512_set1_epi64(P as i64);
            let v_pinv = _mm512_set1_epi64(PINV as i64);
            let v_zero = _mm512_setzero_si512();
            // c0 = VMADD52L(0, a, b)
            let c0 = _mm512_madd52lo_epu64(v_zero, a, b);
            // c1 = VMADD52H(p, a, b), i.e., (a * b >> 52) + p
            let c1 = _mm512_madd52hi_epu64(v_p, a, b);
            // t0 = VMADD52L(0, c0, pinv), i.e., (c0 * pinv) & 0xFFFFFFFFFFFFF
            let t0 = _mm512_madd52lo_epu64(v_zero, c0, v_pinv);
            // t1 = VMADD52H(0, t0, p), i.e., (t0 * p >> 52)
            let t1 = _mm512_madd52hi_epu64(v_zero, t0, v_p);
            // r = c1 - t1
            let r = _mm512_sub_epi64(c1, t1);
            r
        }
    }
}

impl From<u32> for PackedMamaBearAVX512 {
    /// Converts a u32 to PackedMamaBearAVX512 in normal form.
    fn from(val: u32) -> Self {
        Self::broadcast(val as u64)
    }
}

impl From<u64> for PackedMamaBearAVX512 {
    /// Converts a u64 to PackedMamaBearAVX512 in normal form.
    fn from(val: u64) -> Self {
        Self::broadcast(val)
    }
}

impl From<MamaBearScalar> for PackedMamaBearAVX512 {
    fn from(s: MamaBearScalar) -> Self {
        Self::broadcast(s.0 as u64)
    }
}

impl LazyReduction for PackedMamaBearAVX512 {
    /// Standard 64-bit ADD. Caller must ensure no overflow (Result < 2^64)
    #[inline(always)]
    fn lazy_add(self, rhs: Self) -> Self {
        unsafe { Self::from_vector(_mm512_add_epi64(self.simd, rhs.simd)) }
    }

    /// Add broadcasted multiples of P.
    #[inline(always)]
    fn lazy_add_xp(self, x: u8) -> Self {
        unsafe {
            let v_xp = _mm512_set1_epi64(((x as u64) * P) as i64);
            let sum = _mm512_add_epi64(self.simd, v_xp);
            Self::from_vector(sum)
        }
    }

    /// Standard 64-bit SUB. Caller must ensure no underflow (self >= rhs)
    #[inline(always)]
    fn lazy_sub(self, rhs: Self) -> Self {
        unsafe { Self::from_vector(_mm512_sub_epi64(self.simd, rhs.simd)) }
    }

    /// Conditional subtraction of x*P.
    #[inline(always)]
    fn con_sub_xp(self, x: u8) -> Self {
        unsafe {
            let v_xp = _mm512_set1_epi64(((x as u64) * P) as i64);
            let t_minus_xp = _mm512_sub_epi64(self.simd, v_xp);
            let r = _mm512_min_epu64(self.simd, t_minus_xp);
            Self::from_vector(r)
        }
    }

    /// Fast reduction using the special modulus property.
    ///
    /// # Basic Idea
    /// P = 2^49 - 2^34 + 1; 2^49 = 2^34 - 1 mod P
    ///
    /// x = lo + hi * 2^49 = lo + hi * (2^34 - 1)
    ///
    /// # Range Analysis
    /// For x < 2^64: result <= (2^49-1)+(2^15-1)*(2^34-1) < 2^50 ≈ 2P+2^34 ≈ 2.0001P.
    ///
    /// For x: result < 2^49+(x/2^15).
    #[inline(always)]
    fn reduce_fast(self) -> Self {
        // Constants for Lazy Reduction: 2^49 = 2^34 - 1 (mod P)
        const REDUCE_COEFF: u64 = (1 << 34) - 1;
        unsafe {
            let v_mask_49 = _mm512_set1_epi64(((1u64 << 49) - 1) as i64);
            // 2^34 - 1
            let v_coeff = _mm512_set1_epi64(REDUCE_COEFF as i64);
            // Extract Low 49 bits: lo = x & mask
            let lo = _mm512_and_si512(self.simd, v_mask_49);
            // Extract High bits: hi = x >> 49
            let hi = _mm512_srli_epi64(self.simd, 49);
            // result = lo + hi * coeff. hi is at most (64-49)=15 bits. coeff is 34 bits. Product fits in 52 bits easily.
            // We can use _mm512_madd52lo_epu64(lo, hi, coeff) -> lo + hi * coeff
            let res = _mm512_madd52lo_epu64(lo, hi, v_coeff);
            Self::from_vector(res)
        }
    }

    /// Reduction to [0, 2p).
    #[inline(always)]
    fn reduce_2p(self) -> Self {
        unsafe {
            // t < 2P + 2^34
            let t = self.reduce_fast();
            let p = _mm512_set1_epi64(P as i64);
            let t_minus_p = _mm512_sub_epi64(t.simd, p);
            let r = _mm512_min_epu64(t.simd, t_minus_p);
            Self::from_vector(r)
        }
    }

    /// Full reduction to ensure result < P.
    ///
    /// # Basic Idea
    /// Use fast reduction first to get result < 2P + 2^34
    /// Then conditionally subtract P twice if necessary.
    ///
    /// # Range Analysis
    /// For x < 2^64: result < P
    #[inline(always)]
    fn reduce(self) -> Self {
        unsafe {
            // First do fast reduction to ensure we are in range [0, 2P+2^34)
            let fast = self.reduce_fast();
            let v_p = _mm512_set1_epi64(P as i64);
            // Conditional subtraction if >= P
            let t_minus_p = _mm512_sub_epi64(fast.simd, v_p);
            let r_0 = _mm512_min_epu64(fast.simd, t_minus_p);
            // Conditional subtraction if >= P
            let t_minus_p = _mm512_sub_epi64(r_0, v_p);
            let v_sub = _mm512_min_epu64(r_0, t_minus_p);
            Self::from_vector(v_sub)
        }
    }
}

impl Add for PackedMamaBearAVX512 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        unsafe {
            let sum = _mm512_add_epi64(self.simd, rhs.simd);
            // Strict modular addition: needs to reduce
            let v_p = _mm512_set1_epi64(P as i64);
            let t_minus_p = _mm512_sub_epi64(sum, v_p);
            let r = _mm512_min_epu64(sum, t_minus_p);
            Self::from_vector(r)
        }
    }
}

impl AddAssign for PackedMamaBearAVX512 {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl Sub for PackedMamaBearAVX512 {
    type Output = Self;
    /// Modular sub reduced to canonical [0, P) per lane.
    /// Tolerates non-canonical lhs/rhs ∈ [0, 2P) — matches the scalar
    /// `MamaBearScalar::Sub` contract.
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        unsafe {
            let v_p = _mm512_set1_epi64(P as i64);
            let v_2p = _mm512_set1_epi64((2 * P) as i64);
            let tmp = _mm512_add_epi64(self.simd, v_2p);       // self < 2P -> tmp < 4P
            let diff = _mm512_sub_epi64(tmp, rhs.simd);         // rhs < 2P -> diff in (0, 4P)
            let d = _mm512_min_epu64(diff, _mm512_sub_epi64(diff, v_2p));  // [0, 2P)
            let canonical = _mm512_min_epu64(d, _mm512_sub_epi64(d, v_p)); // [0, P)
            Self::from_vector(canonical)
        }
    }
}

impl SubAssign for PackedMamaBearAVX512 {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl Neg for PackedMamaBearAVX512 {
    type Output = Self;
    /// Negate reduced to canonical [0, P) per lane. Tolerates non-canonical
    /// `self ∈ [0, 2P)` — matches the scalar `MamaBearScalar::Neg` contract.
    #[inline(always)]
    fn neg(self) -> Self {
        unsafe {
            let v_p = _mm512_set1_epi64(P as i64);
            let v_2p = _mm512_set1_epi64((2 * P) as i64);
            let diff = _mm512_sub_epi64(v_2p, self.simd);                  // self < 2P -> [1, 2P]
            let d = _mm512_min_epu64(diff, _mm512_sub_epi64(diff, v_p));    // con_sub_xp(1): [0, P]
            let canonical = _mm512_min_epu64(d, _mm512_sub_epi64(d, v_p)); // handle d == P: [0, P)
            Self::from_vector(canonical)
        }
    }
}

impl Mul for PackedMamaBearAVX512 {
    type Output = Self;
    /// Multiplication uses Montgomery multiplication, so inputs must be in Montgomery form.
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        unsafe { Self::from_vector(Self::mont_mul_avx512ifma(self.simd, rhs.simd)) }
    }
}

impl MulAssign for PackedMamaBearAVX512 {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl Field for PackedMamaBearAVX512 {
    const NAME: &'static str = "PackedMamaBearAVX512";
    // An u64 element can hold 1 Mama Bear scalar. We consider SIMD packing here.
    const SIZE: usize = 64;
    type BaseField = MamaBearScalar;

    fn zero() -> Self {
        Self::broadcast(0)
    }
    fn is_zero(&self) -> bool {
        unsafe {
            let eq = _mm512_cmpeq_epu64_mask(self.simd, _mm512_setzero_si512());
            eq == 0xFF
        }
    }
    fn one() -> Self {
        Self::broadcast(1)
    }
    fn random(mut rng: impl RngCore) -> Self {
        let mut arr = [0u64; 8];
        for i in 0..8 {
            arr[i] = rng.next_u64() % P;
        }
        Self::from_array(arr)
    }
    /// Returns the multiplicative inverse of 2 in the field, (P + 1) / 2.
    fn inv_2() -> Self {
        Self::broadcast((P + 1) >> 1)
    }
    fn exp(&self, _exponent: usize) -> Self {
        unimplemented!()
    }
    fn inv(&self) -> Option<Self> {
        unimplemented!()
    }
    fn add_base_elem(&self, rhs: Self::BaseField) -> Self {
        *self + Self::from(rhs)
    }
    fn add_assign_base_elem(&mut self, rhs: Self::BaseField) {
        *self += Self::from(rhs)
    }
    fn mul_base_elem(&self, rhs: Self::BaseField) -> Self {
        *self * Self::from(rhs)
    }
    fn mul_assign_base_elem(&mut self, rhs: Self::BaseField) {
        *self *= Self::from(rhs)
    }
    /// Interpret the first 8 bytes as a u64 little-endian integer and reduce mod P.
    fn from_uniform_bytes(bytes: &[u8; 32]) -> Self {
        let ptr = bytes.as_ptr() as *const u64;
        let v = unsafe { ptr.read_unaligned() } as u64 % P;
        Self::from(v)
    }
    fn serialize_into(&self, buffer: &mut [u8]) {
        assert!(
            buffer.len() >= 64,
            "Buffer too small for PackedMamaBearAVX512"
        );
        unsafe {
            let arr = self.array;
            for i in 0..8 {
                buffer[i * 8..(i + 1) * 8].copy_from_slice(&arr[i].to_le_bytes());
            }
        }
    }
    fn deserialize_from(buffer: &[u8]) -> Self {
        assert!(
            buffer.len() >= 64,
            "Buffer too small for PackedMamaBearAVX512"
        );
        let ptr = buffer.as_ptr() as *const u64;
        let mut arr = [0u64; 8];
        unsafe {
            for i in 0..8 {
                arr[i] = ptr.add(i).read_unaligned() as u64;
                assert!(arr[i] < P, "Value out of range for Mama Bear field");
            }
        }
        Self::from_array(arr)
    }
}

// --- Extension Field Implementation (Degree 3) with irreducible polynomial x^3 - x - 1 ---
// X^3 = X + 1, X^4 = X^2 + X

#[derive(Clone, Copy, PartialEq, Default)]
#[repr(C)]
pub struct PackedMamaBearAVX512Ext3 {
    pub c0: PackedMamaBearAVX512,
    pub c1: PackedMamaBearAVX512,
    pub c2: PackedMamaBearAVX512,
}

impl Debug for PackedMamaBearAVX512Ext3 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        unsafe {
            f.debug_tuple("PackedMamaBearExt3")
                .field(&self.c0.array)
                .field(&self.c1.array)
                .field(&self.c2.array)
                .finish()
        }
    }
}

impl PackedMamaBearAVX512Ext3 {
    /// Normal-form (i.e. NOT Montgomery) zero, suitable for zero-padding the FFT
    /// input buffer before the in-place `to_montgomery` pass.
    pub const ZERO_NORMAL: Self = Self {
        c0: PackedMamaBearAVX512 { array: [0u64; 8] },
        c1: PackedMamaBearAVX512 { array: [0u64; 8] },
        c2: PackedMamaBearAVX512 { array: [0u64; 8] },
    };

    #[inline(always)]
    pub fn new(c0: PackedMamaBearAVX512, c1: PackedMamaBearAVX512, c2: PackedMamaBearAVX512) -> Self {
        Self { c0, c1, c2 }
    }

    #[inline(always)]
    pub fn to_montgomery(self) -> Self {
        Self {
            c0: self.c0.to_montgomery(),
            c1: self.c1.to_montgomery(),
            c2: self.c2.to_montgomery(),
        }
    }

    #[inline(always)]
    pub fn from_montgomery(self) -> Self {
        Self {
            c0: self.c0.from_montgomery(),
            c1: self.c1.from_montgomery(),
            c2: self.c2.from_montgomery(),
        }
    }
}

impl Add for PackedMamaBearAVX512Ext3 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self {
            c0: self.c0 + rhs.c0,
            c1: self.c1 + rhs.c1,
            c2: self.c2 + rhs.c2,
        }
    }
}

impl AddAssign for PackedMamaBearAVX512Ext3 {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        self.c0 += rhs.c0;
        self.c1 += rhs.c1;
        self.c2 += rhs.c2;
    }
}

impl Sub for PackedMamaBearAVX512Ext3 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self {
            c0: self.c0 - rhs.c0,
            c1: self.c1 - rhs.c1,
            c2: self.c2 - rhs.c2,
        }
    }
}

impl SubAssign for PackedMamaBearAVX512Ext3 {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) {
        self.c0 -= rhs.c0;
        self.c1 -= rhs.c1;
        self.c2 -= rhs.c2;
    }
}

impl Mul for PackedMamaBearAVX512Ext3 {
    type Output = Self;

    /// Multiplication in F_p[X]/(X^3 - X - 1) using 6-multiplication Karatsuba.
    ///
    /// X^3 = X + 1, so after schoolbook expansion and reduction:
    ///   c0 = v0 + v12
    ///   c1 = v01 + v12 + v2
    ///   c2 = v02 + v1 + v2
    ///
    /// # Range Analysis
    ///
    /// Input contract: every component `< 2^51` (just over `4P`). The pairwise
    /// `lazy_add`s below reach `2^52`, which is the hard ceiling — that is the
    /// widest operand `madd52` reads without silently dropping high bits.
    ///
    /// `mont_mul(x, y)` is bounded above by `xy/R + P` and below by
    /// `floor(xy/R)`, so at this contract:
    ///
    /// - `v0, v1, v2` (operands `< 4P`): `< 3P`.
    /// - `s01, s02, s12` (operands `< 8P`): `< 9P`.
    /// - `c0 = v0 + s12 + 4P - v1 - v2`: `[2P, 16P)`.
    /// - `c1 = s01 + s12 + 8P - v0 - 2*v1`: `[4P, 26P)`.
    /// - `c2 = s02 + v1 + 4P - v0`: `[2P, 16P)`.
    /// - `reduce_fast` on any `u64`: `[0, 2^50)`, i.e. `[0, 2.0001P)`.
    ///
    /// Every intermediate fits `u64` with room to spare (`26P < 2^54.7`), and
    /// each expression is evaluated additions-first, so no partial sum dips
    /// below zero either.
    ///
    /// The lower bounds need the SAME correlation argument as the scalar twin,
    /// because a worst-case bound makes the `+4P` look insufficient (`vi + vj`
    /// reaches `6P`). With `x = a_i + a_j`, `y = b_i + b_j` we have
    /// `xy >= a_i b_i + a_j b_j`, hence
    /// `sij >= floor(a_i b_i/R) + floor(a_j b_j/R) >= vi + vj - 2P`, so each
    /// `sij + 4P - vi - vj >= 2P`. Regrouping gives the three bounds above.
    ///
    /// HISTORY: this block previously read `v* < 1.5P`, `s* < 3P` and
    /// `c* < 14P`. Those are the figures for a `[0, 2P)` input contract and
    /// contradicted this block's own first line; the code is correct over the
    /// full `2^51` window, the arithmetic quoted for it was not.
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        // 6 base field multiplications
        let v0 = self.c0 * rhs.c0;                                        // [0, 3P)
        let v1 = self.c1 * rhs.c1;                                        // [0, 3P)
        let v2 = self.c2 * rhs.c2;                                        // [0, 3P)
        let s01 = self.c0.lazy_add(self.c1) * rhs.c0.lazy_add(rhs.c1);    // [0, 9P)
        let s02 = self.c0.lazy_add(self.c2) * rhs.c0.lazy_add(rhs.c2);    // [0, 9P)
        let s12 = self.c1.lazy_add(self.c2) * rhs.c1.lazy_add(rhs.c2);    // [0, 9P)

        let p4 = PackedMamaBearAVX512::broadcast(4 * P);
        let p8 = PackedMamaBearAVX512::broadcast(8 * P);

        // c0 = v0 + (s12 - v1 - v2)  = v0 + s12 + 4P - v1 - v2
        // range: [2P, 3P + 9P + 4P) = [2P, 16P); the lower bound is the
        // correlation argument in the doc comment, not 4P - max(v1 + v2).
        let c0 = v0.lazy_add(s12).lazy_add(p4).lazy_sub(v1).lazy_sub(v2).reduce_fast();

        // c1 = (s01 - v0 - v1) + (s12 - v1 - v2) + v2
        //    = s01 + s12 + 8P - v0 - 2*v1
        // range: [4P, 9P + 9P + 8P) = [4P, 26P)
        let c1 = s01.lazy_add(s12).lazy_add(p8).lazy_sub(v0).lazy_sub(v1).lazy_sub(v1).reduce_fast();

        // c2 = (s02 - v0 - v2) + v1 + v2
        //    = s02 + v1 + 4P - v0
        // range: [2P, 9P + 3P + 4P) = [2P, 16P)
        let c2 = s02.lazy_add(v1).lazy_add(p4).lazy_sub(v0).reduce_fast();

        Self { c0, c1, c2 }
    }
}

impl MulAssign for PackedMamaBearAVX512Ext3 {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl Neg for PackedMamaBearAVX512Ext3 {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        Self {
            c0: -self.c0,
            c1: -self.c1,
            c2: -self.c2,
        }
    }
}

impl LazyReduction for PackedMamaBearAVX512Ext3 {
    #[inline(always)]
    fn lazy_add(self, rhs: Self) -> Self {
        Self {
            c0: self.c0.lazy_add(rhs.c0),
            c1: self.c1.lazy_add(rhs.c1),
            c2: self.c2.lazy_add(rhs.c2),
        }
    }

    #[inline(always)]
    fn lazy_add_xp(self, x: u8) -> Self {
        Self {
            c0: self.c0.lazy_add_xp(x),
            c1: self.c1.lazy_add_xp(x),
            c2: self.c2.lazy_add_xp(x),
        }
    }

    #[inline(always)]
    fn lazy_sub(self, rhs: Self) -> Self {
        Self {
            c0: self.c0.lazy_sub(rhs.c0),
            c1: self.c1.lazy_sub(rhs.c1),
            c2: self.c2.lazy_sub(rhs.c2),
        }
    }

    #[inline(always)]
    fn con_sub_xp(self, x: u8) -> Self {
        Self {
            c0: self.c0.con_sub_xp(x),
            c1: self.c1.con_sub_xp(x),
            c2: self.c2.con_sub_xp(x),
        }
    }

    #[inline(always)]
    fn reduce_fast(self) -> Self {
        Self {
            c0: self.c0.reduce_fast(),
            c1: self.c1.reduce_fast(),
            c2: self.c2.reduce_fast(),
        }
    }

    #[inline(always)]
    fn reduce_2p(self) -> Self {
        Self {
            c0: self.c0.reduce_2p(),
            c1: self.c1.reduce_2p(),
            c2: self.c2.reduce_2p(),
        }
    }

    #[inline(always)]
    fn reduce(self) -> Self {
        Self {
            c0: self.c0.reduce(),
            c1: self.c1.reduce(),
            c2: self.c2.reduce(),
        }
    }
}

impl From<u32> for PackedMamaBearAVX512Ext3 {
    fn from(val: u32) -> Self {
        Self {
            c0: PackedMamaBearAVX512::from(val),
            c1: PackedMamaBearAVX512::zero(),
            c2: PackedMamaBearAVX512::zero(),
        }
    }
}

impl From<u64> for PackedMamaBearAVX512Ext3 {
    fn from(val: u64) -> Self {
        Self {
            c0: PackedMamaBearAVX512::from(val),
            c1: PackedMamaBearAVX512::zero(),
            c2: PackedMamaBearAVX512::zero(),
        }
    }
}

impl From<PackedMamaBearAVX512> for PackedMamaBearAVX512Ext3 {
    fn from(val: PackedMamaBearAVX512) -> Self {
        Self {
            c0: val,
            c1: PackedMamaBearAVX512::zero(),
            c2: PackedMamaBearAVX512::zero(),
        }
    }
}

impl From<MamaBearScalar> for PackedMamaBearAVX512Ext3 {
    fn from(s: MamaBearScalar) -> Self {
        Self {
            c0: PackedMamaBearAVX512::from(s),
            c1: PackedMamaBearAVX512::zero(),
            c2: PackedMamaBearAVX512::zero(),
        }
    }
}

impl From<MamaBearScalarExt3> for PackedMamaBearAVX512Ext3 {
    fn from(s: MamaBearScalarExt3) -> Self {
        Self {
            c0: PackedMamaBearAVX512::from(s.c0),
            c1: PackedMamaBearAVX512::from(s.c1),
            c2: PackedMamaBearAVX512::from(s.c2),
        }
    }
}

impl Field for PackedMamaBearAVX512Ext3 {
    const NAME: &'static str = "PackedMamaBearAVX512Ext3";
    const SIZE: usize = PackedMamaBearAVX512::SIZE * 3;
    type BaseField = PackedMamaBearAVX512;

    fn zero() -> Self {
        Self {
            c0: PackedMamaBearAVX512::zero(),
            c1: PackedMamaBearAVX512::zero(),
            c2: PackedMamaBearAVX512::zero(),
        }
    }
    fn is_zero(&self) -> bool {
        self.c0.is_zero() && self.c1.is_zero() && self.c2.is_zero()
    }
    fn one() -> Self {
        Self {
            c0: PackedMamaBearAVX512::one(),
            c1: PackedMamaBearAVX512::zero(),
            c2: PackedMamaBearAVX512::zero(),
        }
    }
    fn random(mut rng: impl RngCore) -> Self {
        Self {
            c0: PackedMamaBearAVX512::random(&mut rng),
            c1: PackedMamaBearAVX512::random(&mut rng),
            c2: PackedMamaBearAVX512::random(&mut rng),
        }
    }
    fn inv_2() -> Self {
        Self {
            c0: PackedMamaBearAVX512::inv_2(),
            c1: PackedMamaBearAVX512::zero(),
            c2: PackedMamaBearAVX512::zero(),
        }
    }
    fn exp(&self, _exponent: usize) -> Self {
        unimplemented!()
    }
    fn inv(&self) -> Option<Self> {
        unimplemented!()
    }
    fn add_base_elem(&self, rhs: Self::BaseField) -> Self {
        Self {
            c0: self.c0.lazy_add(rhs),
            c1: self.c1,
            c2: self.c2,
        }
    }
    fn add_assign_base_elem(&mut self, rhs: Self::BaseField) {
        *self = Self {
            c0: self.c0.lazy_add(rhs),
            c1: self.c1,
            c2: self.c2,
        }
    }
    fn mul_base_elem(&self, rhs: Self::BaseField) -> Self {
        Self {
            c0: self.c0 * rhs,
            c1: self.c1 * rhs,
            c2: self.c2 * rhs,
        }
    }
    fn mul_assign_base_elem(&mut self, rhs: Self::BaseField) {
        *self = Self {
            c0: self.c0 * rhs,
            c1: self.c1 * rhs,
            c2: self.c2 * rhs,
        }
    }
    fn from_uniform_bytes(bytes: &[u8; 32]) -> Self {
        let ptr = bytes.as_ptr() as *const u64;
        let v0 = unsafe { ptr.read_unaligned() } as u64 % P;
        let v1 = unsafe { ptr.add(1).read_unaligned() } as u64 % P;
        let v2 = unsafe { ptr.add(2).read_unaligned() } as u64 % P;
        Self {
            c0: PackedMamaBearAVX512::from(v0),
            c1: PackedMamaBearAVX512::from(v1),
            c2: PackedMamaBearAVX512::from(v2),
        }
    }
    fn serialize_into(&self, buffer: &mut [u8]) {
        assert!(
            buffer.len() >= 192,
            "Buffer too small for PackedMamaBearAVX512Ext3"
        );
        self.c0.serialize_into(&mut buffer[0..64]);
        self.c1.serialize_into(&mut buffer[64..128]);
        self.c2.serialize_into(&mut buffer[128..192]);
    }
    fn deserialize_from(buffer: &[u8]) -> Self {
        assert!(
            buffer.len() >= 192,
            "Buffer too short for PackedMamaBearAVX512Ext3"
        );
        let mut arr = [0u64; 24];
        let ptr = buffer.as_ptr() as *const u64;
        unsafe {
            for i in 0..24 {
                arr[i] = ptr.add(i).read_unaligned() as u64;
                assert!(arr[i] < P, "Value out of range for Mama Bear field");
            }
        }
        Self {
            c0: PackedMamaBearAVX512::from_array(arr[0..8].try_into().unwrap()),
            c1: PackedMamaBearAVX512::from_array(arr[8..16].try_into().unwrap()),
            c2: PackedMamaBearAVX512::from_array(arr[16..24].try_into().unwrap()),
        }
    }
}

pub trait PackedExtensionField: Copy {
    type ScalarExt: Field + Copy;

    /// Broadcast one scalar extension element to all SIMD lanes.
    fn broadcast_scalar(value: Self::ScalarExt) -> Self;

    /// Pack 8 consecutive scalar extension values in AoS layout into one packed SoA value.
    fn pack_slice_exact(slice: &[Self::ScalarExt]) -> Self;

    /// Pack up to 8 scalar extension values, zero-padding the inactive lanes.
    #[inline(always)]
    fn pack_partial(slice: &[Self::ScalarExt]) -> Self {
        debug_assert!(slice.len() <= 8);
        if slice.len() == 8 {
            return Self::pack_slice_exact(slice);
        }

        let mut padded = [Self::ScalarExt::zero(); 8];
        padded[..slice.len()].copy_from_slice(slice);
        Self::pack_slice_exact(&padded)
    }

    /// Unpack one packed SoA value back into 8 scalar extension values in AoS layout.
    fn unpack_to_array(self) -> [Self::ScalarExt; 8];

    /// Unpack into an existing slice.
    #[inline(always)]
    fn unpack_into_slice(self, slice: &mut [Self::ScalarExt]) {
        debug_assert!(slice.len() >= 8);
        slice[..8].copy_from_slice(&self.unpack_to_array());
    }

    /// Append all 8 unpacked values to a result buffer.
    #[inline(always)]
    fn unpack_append(self, result: &mut Vec<Self::ScalarExt>) {
        let scalars = self.unpack_to_array();
        result.extend_from_slice(&scalars);
    }
}

pub trait PackedExtensionPairStride: PackedExtensionField {
    /// Pack either the even pairs or odd pairs from a 16-element AoS slice.
    fn pack_strided_pair_slice(slice: &[Self::ScalarExt], odd: bool) -> Self;
}

impl PackedExtensionField for PackedMamaBearAVX512Ext3 {
    type ScalarExt = MamaBearScalarExt3;

    #[inline(always)]
    fn broadcast_scalar(value: Self::ScalarExt) -> Self {
        Self::from(value)
    }

    #[inline(always)]
    fn pack_slice_exact(slice: &[Self::ScalarExt]) -> Self {
        debug_assert!(slice.len() >= 8);
        let mut c0 = [0u64; 8];
        let mut c1 = [0u64; 8];
        let mut c2 = [0u64; 8];
        for (lane, value) in slice.iter().take(8).copied().enumerate() {
            c0[lane] = value.c0.0;
            c1[lane] = value.c1.0;
            c2[lane] = value.c2.0;
        }
        Self {
            c0: PackedMamaBearAVX512::from_array(c0),
            c1: PackedMamaBearAVX512::from_array(c1),
            c2: PackedMamaBearAVX512::from_array(c2),
        }
    }

    #[inline(always)]
    fn unpack_to_array(self) -> [Self::ScalarExt; 8] {
        let lanes_c0 = self.c0.to_array();
        let lanes_c1 = self.c1.to_array();
        let lanes_c2 = self.c2.to_array();
        let mut scalars = [MamaBearScalarExt3::zero(); 8];
        for lane in 0..8 {
            scalars[lane] = MamaBearScalarExt3 {
                c0: MamaBearScalar(lanes_c0[lane]),
                c1: MamaBearScalar(lanes_c1[lane]),
                c2: MamaBearScalar(lanes_c2[lane]),
            };
        }
        scalars
    }
}

impl PackedExtensionPairStride for PackedMamaBearAVX512Ext3 {
    #[inline(always)]
    fn pack_strided_pair_slice(slice: &[Self::ScalarExt], odd: bool) -> Self {
        debug_assert!(slice.len() >= 16);
        let mut c0 = [0u64; 8];
        let mut c1 = [0u64; 8];
        let mut c2 = [0u64; 8];
        let start = usize::from(odd);
        for lane in 0..8 {
            let value = slice[start + (lane << 1)];
            c0[lane] = value.c0.0;
            c1[lane] = value.c1.0;
            c2[lane] = value.c2.0;
        }
        Self {
            c0: PackedMamaBearAVX512::from_array(c0),
            c1: PackedMamaBearAVX512::from_array(c1),
            c2: PackedMamaBearAVX512::from_array(c2),
        }
    }
}

// RUSTFLAGS="-A warnings -C target-cpu=native" cargo test field::mamabear::tests --release -- --nocapture
#[cfg(test)]
mod tests {
    use super::*;
    type PMamaBear = PackedMamaBearAVX512;

    /// Regression: non-canonical rhs must not break Sub/Neg.
    ///
    /// `MamaBearScalar::Sub` (and per-component Ext{2,3}/packed analogues)
    /// historically assumed `rhs < P`. Functions like `mont_mul`, `Mul`,
    /// `reduce_fast`, `lazy_add*` routinely produce values ≥ P canonically
    /// equivalent to a smaller in-range value. If any such non-canonical
    /// `rhs` landed in `Sub`, the output was canonically WRONG. This test
    /// exercises that exact scenario at the scalar and packed levels.
    #[test]
    fn test_sub_noncanonical_rhs_canonical() {
        let self_val = MamaBearScalar(50);
        let canonical_rhs = MamaBearScalar(100);
        // Same canonical value (100) but non-canonical representation (P + 100).
        let noncanonical_rhs = MamaBearScalar(P + 100);
        let r1 = (self_val - canonical_rhs).0 % P;
        let r2 = (self_val - noncanonical_rhs).0 % P;
        assert_eq!(
            r1, r2,
            "Sub must give canonically-equivalent result for canonically-equivalent rhs"
        );
    }

    /// Regression: Neg must tolerate non-canonical self.
    #[test]
    fn test_neg_noncanonical_canonical() {
        let canonical = MamaBearScalar(100);
        // Same canonical (100), non-canonical representation (P + 100).
        let noncanonical = MamaBearScalar(P + 100);
        let r1 = (-canonical).0 % P;
        let r2 = (-noncanonical).0 % P;
        assert_eq!(r1, r2, "Neg must be canonically-correct for non-canonical self");
    }

    #[test]
    fn test_add_sub_basic() {
        let a = PMamaBear::from(10u64);
        let b = PMamaBear::from(20u64);
        let c = a + b;
        let expected = PMamaBear::from(30u64);
        assert_eq!(c, expected);

        let d = b - a;
        let expected = PMamaBear::from(10u64);
        assert_eq!(d, expected);

        let e = a - b;
        let expected = PMamaBear::from(P - 10);
        assert_eq!(e, expected);
    }

    #[test]
    fn test_neg_basic() {
        let a = PMamaBear::from(15u64);
        let b = -a;
        let expected = PMamaBear::from(P - 15);
        assert_eq!(b, expected);

        let zero = PMamaBear::zero();
        let neg_zero = -zero;
        assert_eq!(neg_zero, zero);
    }

    #[test]
    fn test_mul_basic() {
        let a = PMamaBear::from(3u64).to_montgomery();
        let b = PMamaBear::from(5u64);
        let c = (a * b).reduce();
        let expected = PMamaBear::from(15u64);
        assert_eq!(c, expected);
    }

    #[test]
    fn test_to_from_montgomery() {
        let a = PMamaBear::from(7u64);
        let mont = a.to_montgomery();
        let normal = mont.from_montgomery();
        assert_eq!(a, normal);
    }

    #[test]
    fn test_scalar_mont_roundtrip() {
        // mont_mul(0, x) = P (not 0) because c1 = (ab>>52) + P adds P unconditionally.
        // This means to_mont(from_mont(x)) ≡ x (mod P) but may differ by ±P.
        let test_vals: Vec<u64> = vec![0, 1, P - 1, P, P + 1, 2 * P - 1, 137438953464];
        for x in test_vals {
            let s = MamaBearScalar(x);
            let roundtrip = s.from_montgomery().to_montgomery();
            assert_eq!(
                roundtrip.0 % P, x % P,
                "Round-trip not mod-P equivalent for x={x}: got {}, expected {}",
                roundtrip.0 % P, x % P,
            );
        }
        // Verify: mont_mul(0, 1) = P (not 0). The raw mont_mul intentionally
        // biases its output by P; from_montgomery / to_montgomery canonicalize
        // that bias away via the final con_sub_xp(1).
        assert_eq!(MamaBearScalar::mont_mul(0, 1), P);
        // Verify: from_mont(0) = 0 (after con_sub_xp).
        assert_eq!(MamaBearScalar(0).from_montgomery().0, 0);
        // Verify: to_mont(0) = 0 (after con_sub_xp inside to_montgomery).
        assert_eq!(MamaBearScalar(0).to_montgomery().0, 0);
        // Verify: to_mont(from_mont(0)) = 0 (both operations canonicalize).
        assert_eq!(MamaBearScalar(0).from_montgomery().to_montgomery().0, 0);
    }

    #[test]
    fn test_lazy_add_sub() {
        let a = PMamaBear::from(P - 1);
        let b = PMamaBear::from(2u64);
        let sum = a.lazy_add(b);
        unsafe {
            let res = sum.array[0];
            println!("Lazy Add: {} + {} = {}", P - 1, 2, res);
            assert_eq!(res, P + 1);
        }

        let diff = sum.lazy_sub(b);
        unsafe {
            let res = diff.array[0];
            println!("Lazy Sub: {} - {} = {}", P + 1, 2, res);
            assert_eq!(res, P - 1);
        }
    }

    #[test]
    fn test_con_sub_xp() {
        let a = PMamaBear::from(P + 5);
        let reduced = a.con_sub_xp(1);
        unsafe {
            let res = reduced.array[0];
            println!("Con Sub P: {} - P = {}", P + 5, res);
            assert_eq!(res, 5);
        }

        let b = PMamaBear::from(P - 3);
        let reduced = b.con_sub_xp(1);
        unsafe {
            let res = reduced.array[0];
            println!("Con Sub P: {} - P = {}", P - 3, res);
            assert_eq!(res, P - 3);
        }

        let c = PMamaBear::from(P * 2 + 7);
        let reduced = c.con_sub_xp(2);
        unsafe {
            let res = reduced.array[0];
            println!("Con Sub 2P: {} - 2P = {}", P * 2 + 7, res);
            assert_eq!(res, 7);
        }

        let d = PMamaBear::from(P * 2 - 4);
        let reduced = d.con_sub_xp(1);
        unsafe {
            let res = reduced.array[0];
            println!("Con Sub P: {} - P = {}", P * 2 - 4, res);
            assert_eq!(res, P - 4);
        }
    }

    #[test]
    fn test_reduce_fast() {
        // Construct a value that needs reduction
        let val_u64 = 1u64 << 50;
        let a = PMamaBear::from(val_u64);
        let fast = a.reduce_fast();
        unsafe {
            let res = fast.array[0];
            println!("Raw: {}, Fast Reduced: {}", val_u64, res);
            assert!(res < 2 * P);
            // Check logic: 2^50 = 2 * 2^49 = 2 * (2^34 - 1) = 2^35 - 2 mod P
            // fast reduce logic:
            // lo = 0, hi = 2. res = 0 + 2 * (2^34 - 1) = 2^35 - 2.
            assert_eq!(res, (1 << 35) - 2);
        }

        // Another value: 2^64 - 1
        let val_u64 = u64::MAX;
        let a = PMamaBear::from(val_u64);
        let fast = a.reduce_fast();
        unsafe {
            let res = fast.array[0];
            println!("Raw: {}, Fast Reduced: {}", val_u64, res);
            assert!(res < 2 * P + (1 << 34));
            // Check logic: 2^64 - 1
            // lo = (2^49 - 1), hi = (2^15 - 1)
            // res = (2^49 - 1) + (2^15 - 1) * (2^34 - 1)
            // = (2^49 - 1) + (2^49 - 2^34 - 2^15 + 1)
            // = 1125882726940672 > 2p
            assert_eq!(res, 1125882726940672);
        }
    }

    #[test]
    fn test_reduce_2p() {
        // 2^64 - 1
        let val_u64 = u64::MAX;
        let a = PMamaBear::from(val_u64);
        let reduced = a.reduce_2p();
        unsafe {
            let res = reduced.array[0];
            println!("Raw: {}, reduce_2p: {}", val_u64, res);
            assert!(res < 2 * P);
            assert_eq!(res, 1125882726940672 - P);
        }
    }

    #[test]
    fn test_reduce() {
        // 2^64 - 1
        let val_u64 = u64::MAX;
        let a = PMamaBear::from(val_u64);
        let reduced = a.reduce();
        unsafe {
            let res = reduced.array[0];
            println!("Raw: {}, reduce: {}", val_u64, res);
            assert!(res < P);
            assert_eq!(res, 1125882726940672 - 2 * P);
        }
    }

    // --- Ext3 Tests ---
    type PMamaBearExt3 = PackedMamaBearAVX512Ext3;

    /// Schoolbook multiplication in F_p[X]/(X^3 - X - 1) for reference.
    /// Works on normal (non-Montgomery) u64 values.
    fn schoolbook_ext3_mul(a: [u64; 3], b: [u64; 3]) -> [u64; 3] {
        let p = P as u128;
        let d0 = (a[0] as u128 * b[0] as u128) % p;
        let d1 = ((a[0] as u128 * b[1] as u128) + (a[1] as u128 * b[0] as u128)) % p;
        let d2 = ((a[0] as u128 * b[2] as u128) + (a[1] as u128 * b[1] as u128) + (a[2] as u128 * b[0] as u128)) % p;
        let d3 = ((a[1] as u128 * b[2] as u128) + (a[2] as u128 * b[1] as u128)) % p;
        let d4 = (a[2] as u128 * b[2] as u128) % p;
        // X^3 = X + 1, X^4 = X^2 + X
        let c0 = (d0 + d3) % p;
        let c1 = (d1 + d3 + d4) % p;
        let c2 = (d2 + d4) % p;
        [c0 as u64, c1 as u64, c2 as u64]
    }

    #[test]
    fn test_mul_ext3_basic() {
        // a = 3 + 4X + 2X^2, b = 5 + 6X + 7X^2
        let a_vals: [u64; 3] = [3, 4, 2];
        let b_vals: [u64; 3] = [5, 6, 7];
        let expected = schoolbook_ext3_mul(a_vals, b_vals);

        let a = PMamaBearExt3::new(
            PMamaBear::from(a_vals[0]).to_montgomery(),
            PMamaBear::from(a_vals[1]).to_montgomery(),
            PMamaBear::from(a_vals[2]).to_montgomery(),
        );
        let b = PMamaBearExt3::new(
            PMamaBear::from(b_vals[0]),
            PMamaBear::from(b_vals[1]),
            PMamaBear::from(b_vals[2]),
        );

        let c = a * b;
        let c_reduced = PMamaBearExt3 {
            c0: c.c0.reduce(),
            c1: c.c1.reduce(),
            c2: c.c2.reduce(),
        };

        assert_eq!(c_reduced.c0, PMamaBear::from(expected[0]));
        assert_eq!(c_reduced.c1, PMamaBear::from(expected[1]));
        assert_eq!(c_reduced.c2, PMamaBear::from(expected[2]));
    }

    #[test]
    fn test_mul_ext3_scalar_basic() {
        let a_vals: [u64; 3] = [3, 4, 2];
        let b_vals: [u64; 3] = [5, 6, 7];
        let expected = schoolbook_ext3_mul(a_vals, b_vals);

        let a = MamaBearScalarExt3 {
            c0: MamaBearScalar(a_vals[0]).to_montgomery(),
            c1: MamaBearScalar(a_vals[1]).to_montgomery(),
            c2: MamaBearScalar(a_vals[2]).to_montgomery(),
        };
        let b = MamaBearScalarExt3 {
            c0: MamaBearScalar(b_vals[0]),
            c1: MamaBearScalar(b_vals[1]),
            c2: MamaBearScalar(b_vals[2]),
        };

        let c = (a * b).reduce();
        assert_eq!(c.c0.0 as u64, expected[0]);
        assert_eq!(c.c1.0 as u64, expected[1]);
        assert_eq!(c.c2.0 as u64, expected[2]);
    }

    #[test]
    fn test_ext3_scalar_vs_packed_random() {
        use rand::rngs::SmallRng;
        use rand::SeedableRng;
        let mut rng = SmallRng::seed_from_u64(42);

        for _ in 0..100 {
            // Generate random values in [0, P)
            let a_vals: [u64; 3] = [
                rng.next_u64() % P,
                rng.next_u64() % P,
                rng.next_u64() % P,
            ];
            let b_vals: [u64; 3] = [
                rng.next_u64() % P,
                rng.next_u64() % P,
                rng.next_u64() % P,
            ];
            let expected = schoolbook_ext3_mul(a_vals, b_vals);

            // Test packed version
            let a_packed = PMamaBearExt3::new(
                PMamaBear::from(a_vals[0]).to_montgomery(),
                PMamaBear::from(a_vals[1]).to_montgomery(),
                PMamaBear::from(a_vals[2]).to_montgomery(),
            );
            let b_packed = PMamaBearExt3::new(
                PMamaBear::from(b_vals[0]),
                PMamaBear::from(b_vals[1]),
                PMamaBear::from(b_vals[2]),
            );
            let c_packed = (a_packed * b_packed).reduce();
            unsafe {
                assert_eq!(c_packed.c0.array[0], expected[0], "packed c0 mismatch");
                assert_eq!(c_packed.c1.array[0], expected[1], "packed c1 mismatch");
                assert_eq!(c_packed.c2.array[0], expected[2], "packed c2 mismatch");
            }

            // Test scalar version
            let a_scalar = MamaBearScalarExt3 {
                c0: MamaBearScalar(a_vals[0]).to_montgomery(),
                c1: MamaBearScalar(a_vals[1]).to_montgomery(),
                c2: MamaBearScalar(a_vals[2]).to_montgomery(),
            };
            let b_scalar = MamaBearScalarExt3 {
                c0: MamaBearScalar(b_vals[0]),
                c1: MamaBearScalar(b_vals[1]),
                c2: MamaBearScalar(b_vals[2]),
            };
            let c_scalar = (a_scalar * b_scalar).reduce();
            assert_eq!(c_scalar.c0.0 as u64, expected[0], "scalar c0 mismatch");
            assert_eq!(c_scalar.c1.0 as u64, expected[1], "scalar c1 mismatch");
            assert_eq!(c_scalar.c2.0 as u64, expected[2], "scalar c2 mismatch");
        }
    }

    #[test]
    fn test_ext3_associativity() {
        use rand::rngs::SmallRng;
        use rand::SeedableRng;
        let mut rng = SmallRng::seed_from_u64(42);

        for _ in 0..50 {
            let a = PMamaBearExt3::random(&mut rng).to_montgomery();
            let b = PMamaBearExt3::random(&mut rng);
            let c = PMamaBearExt3::random(&mut rng);

            let ab_c = ((a * b) * c).reduce();
            let a_bc = (a * (b * c)).reduce();
            assert_eq!(ab_c, a_bc);
        }
    }

    #[test]
    fn test_ext3_distributivity() {
        use rand::rngs::SmallRng;
        use rand::SeedableRng;
        let mut rng = SmallRng::seed_from_u64(42);

        for _ in 0..50 {
            let a = PMamaBearExt3::random(&mut rng).to_montgomery();
            let b = PMamaBearExt3::random(&mut rng);
            let c = PMamaBearExt3::random(&mut rng);

            let a_bpc = (a * (b + c)).reduce();
            let ab_ac = ((a * b) + (a * c)).reduce();
            assert_eq!(a_bpc, ab_ac);
        }
    }

    #[test]
    fn test_ext3_one_mul_identity() {
        use rand::rngs::SmallRng;
        use rand::SeedableRng;
        let mut rng = SmallRng::seed_from_u64(42);

        // a in Montgomery form, one is raw 1.
        // mont_mul(a*R, 1) = a (plain), so result = a.from_montgomery()
        let a = PMamaBearExt3::random(&mut rng).to_montgomery();
        let one = PMamaBearExt3::one();
        let result = (a * one).reduce();
        let expected = a.from_montgomery(); // already [0, P) per component
        assert_eq!(result, expected);
    }

    // --- exp / inv / raw_mul tests ---

    #[test]
    fn test_scalar_exp_basic() {
        // 2^10 = 1024
        let two = MamaBearScalar(2);
        assert_eq!(two.exp(10).0, 1024);
        // 3^0 = 1
        assert_eq!(MamaBearScalar(3).exp(0).0, 1);
        // 3^1 = 3
        assert_eq!(MamaBearScalar(3).exp(1).0, 3);
        // 7^(P-1) = 1 (Fermat)
        let result = MamaBearScalar(7).exp(P as usize - 1);
        assert_eq!(result.0, 1, "Fermat's little theorem failed");
    }

    #[test]
    fn test_scalar_inv() {
        // 2 * inv(2) = 1
        let two = MamaBearScalar(2);
        let inv2 = two.inv().unwrap();
        let product = two.raw_mul(inv2);
        assert_eq!(product.0, 1, "2 * inv(2) should be 1, got {}", product.0);

        // inv_2() should equal inv(2)
        let inv_2 = MamaBearScalar::inv_2();
        assert_eq!(inv_2.0, inv2.0, "inv_2() != inv(2)");

        // Random values: a * a^{-1} = 1
        use rand::rngs::SmallRng;
        use rand::SeedableRng;
        let mut rng = SmallRng::seed_from_u64(123);
        for _ in 0..20 {
            let a = MamaBearScalar(rng.next_u64() % P);
            if a.is_zero() { continue; }
            let a_inv = a.inv().unwrap();
            let product = a.raw_mul(a_inv);
            assert_eq!(product.0, 1, "a * a^{{-1}} != 1 for a={}", a.0);
        }
    }

    #[test]
    fn test_scalar_raw_mul() {
        // 3 * 5 = 15
        let a = MamaBearScalar(3);
        let b = MamaBearScalar(5);
        assert_eq!(a.raw_mul(b).0, 15);
        // 0 * x = 0
        assert_eq!(MamaBearScalar(0).raw_mul(b).0, 0);
    }

    #[test]
    fn test_ext3_exp_inv() {
        use rand::rngs::SmallRng;
        use rand::SeedableRng;
        let mut rng = SmallRng::seed_from_u64(789);

        for _ in 0..5 {
            let a = MamaBearScalarExt3 {
                c0: MamaBearScalar(rng.next_u64() % P),
                c1: MamaBearScalar(rng.next_u64() % P),
                c2: MamaBearScalar(rng.next_u64() % P),
            };
            if a.c0.is_zero() && a.c1.is_zero() && a.c2.is_zero() { continue; }
            let a_inv = a.inv().unwrap();
            let a_mont = a.to_montgomery();
            let a_inv_mont = a_inv.to_montgomery();
            let product = (a_mont * a_inv_mont).from_montgomery(); // already [0, P)
            assert_eq!(product.c0.0, 1, "Ext3 inv c0 != 1: got {}", product.c0.0);
            assert_eq!(product.c1.0, 0, "Ext3 inv c1 != 0: got {}", product.c1.0);
            assert_eq!(product.c2.0, 0, "Ext3 inv c2 != 0: got {}", product.c2.0);
        }
    }

    #[test]
    fn test_ext3_inv_2() {
        let inv2 = MamaBearScalarExt3::inv_2();
        assert_eq!(inv2.c0.raw_mul(MamaBearScalar(2)).0, 1);
        assert_eq!(inv2.c1.0, 0);
        assert_eq!(inv2.c2.0, 0);
    }

    #[test]
    fn test_root_of_unity() {
        let root = MamaBearScalar::ROOT_OF_UNITY;
        // root^(2^34) = 1
        let r34 = root.exp(1usize << 34);
        assert_eq!(r34.0, 1, "root^(2^34) != 1");
        // root^(2^33) != 1 (primitive)
        let r33 = root.exp(1usize << 33);
        assert_ne!(r33.0, 1, "root^(2^33) == 1 (not primitive!)");
        // root^(2^33) should be P-1 = -1
        assert_eq!(r33.0, P - 1, "root^(2^33) should be -1");
    }
}
