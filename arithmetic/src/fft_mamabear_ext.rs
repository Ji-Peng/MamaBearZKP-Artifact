//! Extension-field MamaBear FFT kernels (PEF3) with the same lazy-reduction
//! range invariants and pair-major output layout as the base `fft_mamabear`
//! PBF FFT.
//!
//! # Why a separate file
//!
//! The base `MamaBearFFT::fft_into` and all its private helpers (`fft_in_place`,
//! `dif_packed_three_layer_no_shuffle`, `dif_packed_tail_three_layer_pairmajor`,
//! ...) operate on `&mut [MamaBearScalar]` and use `Self::load_packed` /
//! `Self::store_packed` to view 8 consecutive scalars as a `PackedMamaBearAVX512`
//! (PBF). They are private fns and may not be parameterised without changing
//! their signatures; per the project's HC-3 ("base path 0 modification") we
//! never touch them. This file holds a *separate* generic kernel parameterised
//! over a private trait `ExtFftElem`, with a monomorphic instantiation for
//! `PackedMamaBearAVX512Ext3`. The base PBF kernel is left bit-identical.
//!
//! # Layout convention
//!
//! All public entries take `&[T]` and `&mut [T]` where `T = PEF3`. `len * 8 ==
//! fft.size()`: each `T` element packs 8 logical extension scalars across 8
//! SIMD lanes (the standard `PackedExtensionField<ScalarExt = E>` semantics).
//! All offset arithmetic in the helpers below stays in *logical scalar units*
//! (`m0`, `m1`, `m2`, `j`, `k`, `twiddle_step`) -- exactly like the base
//! kernel -- and we use `block_at` / `store_block` to translate to
//! packed-block indexing at every load/store site.
//!
//! # Twiddle convention
//!
//! Twiddle vectors are read from `fft.elements_mont()` (a base-field slice).
//! Each butterfly multiplies a PEF lane vector by a PBF twiddle vector; this
//! is `(c0 * w_pbf, c1 * w_pbf, c2 * w_pbf)` -- 3 `mont_mul`s per butterfly,
//! NOT a Karatsuba (which would cost 6). See `ExtFftElem::mul_pbf`.
//!
//! # Iteration scope
//!
//! This iteration covers the **non-zero-padded** path (`fft_in_place_ext`):
//! when the caller's input length equals `fft.size()`, the buffer is filled
//! with the input, padded to `n` with `T::ZERO_NORMAL`, lifted to Montgomery,
//! and processed top-down with the same DIF layer dispatch as base. The
//! zero-padded fast paths (chirp prefix3 + fused dense3) will be added in a
//! follow-up iteration; their absence here only costs performance, not
//! correctness.

use std::ptr;

use crate::fft_mamabear::MamaBearFFT;
use crate::field::Field;
use crate::field::mamabear::{
    LazyReduction, MamaBearScalar, MamaBearScalarExt3, PackedMamaBearAVX512,
    PackedMamaBearAVX512Ext3,
};

// ---------------------------------------------------------------------------
// ExtFftElem: private trait abstracting PEF3 ops used by the kernel.
// ---------------------------------------------------------------------------

trait ExtFftElem: Copy + Sized {
    /// Normal-form (non-Montgomery) zero, used for zero-padding the input.
    const ZERO_NORMAL: Self;

    /// Logical scalar type for one lane of `Self`. For PEF3 -> Ext3.
    /// Used by the n < 64 small-FFT fallback to operate on unpacked logical
    /// scalar arrays.
    type Scalar: Copy
        + Default
        + LazyReduction
        + Field<BaseField = MamaBearScalar>;

    /// Componentwise `to_montgomery`.
    fn to_montgomery(self) -> Self;

    /// Componentwise `lazy_add`.
    fn lazy_add(self, rhs: Self) -> Self;

    /// Componentwise `lazy_sub`.
    fn lazy_sub(self, rhs: Self) -> Self;

    /// Componentwise `lazy_add_xp(x)`.
    fn lazy_add_xp(self, x: u8) -> Self;

    /// Componentwise `con_sub_xp(x)`.
    fn con_sub_xp(self, x: u8) -> Self;

    /// Componentwise full reduction to canonical form.
    fn reduce(self) -> Self;

    /// Multiply by a PBF (per-lane base-field twiddle) componentwise. The PBF
    /// is the standard FFT twiddle vector returned by `load_twiddle_vector_pbf`.
    fn mul_pbf(self, w: PackedMamaBearAVX512) -> Self;

    /// Componentwise `permute2` (lane permutation across two source vectors,
    /// `_mm512_permutex2var_epi64`). Used by the tail 3-layer pair-major step.
    fn permute2(self, other: Self, indices: [u64; 8]) -> Self;

    /// Unpack the 8 lanes of this packed block to logical scalar form. Used
    /// by the n < 64 small-FFT fallback.
    fn unpack_to_scalars(self) -> [Self::Scalar; 8];

    /// Build a packed block from 8 logical scalar values.
    fn pack_from_scalars(scalars: [Self::Scalar; 8]) -> Self;
}

impl ExtFftElem for PackedMamaBearAVX512Ext3 {
    const ZERO_NORMAL: Self = PackedMamaBearAVX512Ext3::ZERO_NORMAL;
    type Scalar = MamaBearScalarExt3;

    #[inline(always)]
    fn to_montgomery(self) -> Self {
        self.to_montgomery()
    }
    #[inline(always)]
    fn lazy_add(self, rhs: Self) -> Self {
        <Self as LazyReduction>::lazy_add(self, rhs)
    }
    #[inline(always)]
    fn lazy_sub(self, rhs: Self) -> Self {
        <Self as LazyReduction>::lazy_sub(self, rhs)
    }
    #[inline(always)]
    fn lazy_add_xp(self, x: u8) -> Self {
        <Self as LazyReduction>::lazy_add_xp(self, x)
    }
    #[inline(always)]
    fn con_sub_xp(self, x: u8) -> Self {
        <Self as LazyReduction>::con_sub_xp(self, x)
    }
    #[inline(always)]
    fn reduce(self) -> Self {
        <Self as LazyReduction>::reduce(self)
    }
    #[inline(always)]
    fn mul_pbf(self, w: PackedMamaBearAVX512) -> Self {
        // (c0 * w, c1 * w, c2 * w): three PBF mont_muls.
        Self {
            c0: self.c0 * w,
            c1: self.c1 * w,
            c2: self.c2 * w,
        }
    }
    #[inline(always)]
    fn permute2(self, other: Self, indices: [u64; 8]) -> Self {
        Self {
            c0: self.c0.permute2(other.c0, indices),
            c1: self.c1.permute2(other.c1, indices),
            c2: self.c2.permute2(other.c2, indices),
        }
    }
    #[inline(always)]
    fn unpack_to_scalars(self) -> [Self::Scalar; 8] {
        let arr0 = unsafe { self.c0.array };
        let arr1 = unsafe { self.c1.array };
        let arr2 = unsafe { self.c2.array };
        std::array::from_fn(|i| MamaBearScalarExt3 {
            c0: MamaBearScalar(arr0[i]),
            c1: MamaBearScalar(arr1[i]),
            c2: MamaBearScalar(arr2[i]),
        })
    }
    #[inline(always)]
    fn pack_from_scalars(scalars: [Self::Scalar; 8]) -> Self {
        let mut a0 = [0u64; 8];
        let mut a1 = [0u64; 8];
        let mut a2 = [0u64; 8];
        for i in 0..8 {
            a0[i] = scalars[i].c0.0;
            a1[i] = scalars[i].c1.0;
            a2[i] = scalars[i].c2.0;
        }
        Self {
            c0: PackedMamaBearAVX512::from_array(a0),
            c1: PackedMamaBearAVX512::from_array(a1),
            c2: PackedMamaBearAVX512::from_array(a2),
        }
    }
}

// ---------------------------------------------------------------------------
// Block / twiddle helpers (generic over T: ExtFftElem).
// ---------------------------------------------------------------------------

/// Read the packed block whose first lane sits at logical scalar offset `scalar_offset`.
/// `scalar_offset` must be a multiple of 8.
#[inline(always)]
fn block_at<T: ExtFftElem>(slice: &[T], scalar_offset: usize) -> T {
    debug_assert_eq!(scalar_offset & 7, 0);
    slice[scalar_offset >> 3]
}

#[inline(always)]
fn store_block<T: ExtFftElem>(slice: &mut [T], scalar_offset: usize, value: T) {
    debug_assert_eq!(scalar_offset & 7, 0);
    slice[scalar_offset >> 3] = value;
}

/// Load 8 base-field twiddles `omega^{(start+i)*step}` for `i = 0..8` into a PBF.
#[inline(always)]
fn load_twiddle_vector_pbf(
    elements_mont: &[MamaBearScalar],
    start: usize,
    twiddle_step: usize,
) -> PackedMamaBearAVX512 {
    let mut w_arr = [0u64; 8];
    for i in 0..8 {
        w_arr[i] = elements_mont[(start + i) * twiddle_step].0;
    }
    PackedMamaBearAVX512::from_array(w_arr)
}

// ---------------------------------------------------------------------------
// Butterfly + 3-layer prefix + tail (generic).
// ---------------------------------------------------------------------------

/// DIF butterfly on PEF: (u, v) ∈ [0,2P) per component, output (u+v) ∈ [0,2P)
/// after `con_sub_xp(2)`, output (u-v)*w ∈ [0, 1.75P) after `mul_pbf` (mont_mul
/// of u + 2P - v which is in [0, 4P) -- below 2^51 -- against w in [0, 1.5P]).
#[inline(always)]
fn dif_butterfly_pair_ext<T: ExtFftElem>(u: T, v: T, twiddle: PackedMamaBearAVX512) -> (T, T) {
    let out_plus = u.lazy_add(v).con_sub_xp(2); // [0, 2P) per component
    let diff = u.lazy_add_xp(2).lazy_sub(v); // [0, 4P) per component
    let out_minus = diff.mul_pbf(twiddle); // [0, 1.75P) per component
    (out_plus, out_minus)
}

#[inline(always)]
fn dif_butterfly_pair_final_reduce_ext<T: ExtFftElem>(
    u: T,
    v: T,
    twiddle: PackedMamaBearAVX512,
) -> (T, T) {
    let (out_plus, out_minus) = dif_butterfly_pair_ext(u, v, twiddle);
    (out_plus.reduce(), out_minus.reduce())
}

/// Register-only 3-layer DIF block on 8 PEF inputs. Mirrors
/// `dif_packed_three_layer_from_registers` in the base path, but each butterfly
/// routes through `mul_pbf` (componentwise PBF multiply).
#[inline(always)]
fn dif_three_layer_from_registers_ext<T: ExtFftElem>(
    p0: T,
    p1: T,
    p2: T,
    p3: T,
    p4: T,
    p5: T,
    p6: T,
    p7: T,
    tw_l0: PackedMamaBearAVX512,
    tw_l1: PackedMamaBearAVX512,
    tw_l2: PackedMamaBearAVX512,
    tw_l3: PackedMamaBearAVX512,
    tw_m0: PackedMamaBearAVX512,
    tw_m1: PackedMamaBearAVX512,
    tw_n: PackedMamaBearAVX512,
) -> (T, T, T, T, T, T, T, T) {
    let (a0, a4) = dif_butterfly_pair_ext(p0, p4, tw_l0);
    let (a1, a5) = dif_butterfly_pair_ext(p1, p5, tw_l1);
    let (a2, a6) = dif_butterfly_pair_ext(p2, p6, tw_l2);
    let (a3, a7) = dif_butterfly_pair_ext(p3, p7, tw_l3);

    let (b0, b2) = dif_butterfly_pair_ext(a0, a2, tw_m0);
    let (b1, b3) = dif_butterfly_pair_ext(a1, a3, tw_m1);
    let (b4, b6) = dif_butterfly_pair_ext(a4, a6, tw_m0);
    let (b5, b7) = dif_butterfly_pair_ext(a5, a7, tw_m1);

    let (r0, r1) = dif_butterfly_pair_ext(b0, b1, tw_n);
    let (r2, r3) = dif_butterfly_pair_ext(b2, b3, tw_n);
    let (r4, r5) = dif_butterfly_pair_ext(b4, b5, tw_n);
    let (r6, r7) = dif_butterfly_pair_ext(b6, b7, tw_n);

    (r0, r1, r2, r3, r4, r5, r6, r7)
}

/// One DIF layer at packed stride `m >= 8`. `coeff` is in *logical scalar units*
/// of length `n = coeff.len() * 8`.
#[inline]
fn dif_one_layer_no_shuffle_ext<T: ExtFftElem>(
    coeff: &mut [T],
    elements_mont: &[MamaBearScalar],
    m: usize,
    twiddle_step: usize,
) {
    let n = coeff.len() * 8;
    debug_assert!(m >= 8);
    debug_assert!(m & 7 == 0);
    for j in (0..n).step_by(m * 2) {
        for k in (0..m).step_by(8) {
            let w = load_twiddle_vector_pbf(elements_mont, k, twiddle_step);
            let u = block_at(coeff, j + k);
            let v = block_at(coeff, j + k + m);
            let (out_plus, out_minus) = dif_butterfly_pair_ext(u, v, w);
            store_block(coeff, j + k, out_plus);
            store_block(coeff, j + k + m, out_minus);
        }
    }
}

/// Two-layer fused DIF pass at stride `m0 >= 16` (so `m1 = m0/2 >= 8`).
#[inline]
fn dif_two_layer_no_shuffle_ext<T: ExtFftElem>(
    coeff: &mut [T],
    elements_mont: &[MamaBearScalar],
    m0: usize,
    twiddle_step0: usize,
) {
    let n = coeff.len() * 8;
    let m1 = m0 / 2;
    let twiddle_step1 = twiddle_step0 * 2;
    debug_assert!(m1 >= 8);

    for j in (0..n).step_by(m0 * 2) {
        for k in (0..m1).step_by(8) {
            let a = block_at(coeff, j + k);
            let b = block_at(coeff, j + k + m1);
            let c = block_at(coeff, j + k + m0);
            let d = block_at(coeff, j + k + m0 + m1);

            let w0_k = load_twiddle_vector_pbf(elements_mont, k, twiddle_step0);
            let w0_km1 = load_twiddle_vector_pbf(elements_mont, k + m1, twiddle_step0);
            let w1_k = load_twiddle_vector_pbf(elements_mont, k, twiddle_step1);

            let (u_ac, v_ac) = dif_butterfly_pair_ext(a, c, w0_k);
            let (u_bd, v_bd) = dif_butterfly_pair_ext(b, d, w0_km1);

            let (r0, r1) = dif_butterfly_pair_ext(u_ac, u_bd, w1_k);
            let (r2, r3) = dif_butterfly_pair_ext(v_ac, v_bd, w1_k);

            store_block(coeff, j + k, r0);
            store_block(coeff, j + k + m1, r1);
            store_block(coeff, j + k + m0, r2);
            store_block(coeff, j + k + m0 + m1, r3);
        }
    }
}

/// Three-layer fused DIF pass: `m0 >= 32` so `m2 = m0/4 >= 8`. Mirrors the base
/// `dif_packed_three_layer_no_shuffle` exactly, except every butterfly uses
/// componentwise PBF multiply (`mul_pbf`).
#[inline]
fn dif_three_layer_no_shuffle_ext<T: ExtFftElem>(
    coeff: &mut [T],
    elements_mont: &[MamaBearScalar],
    m0: usize,
    twiddle_step0: usize,
) {
    let n = coeff.len() * 8;
    let m1 = m0 / 2;
    let m2 = m0 / 4;
    let twiddle_step1 = twiddle_step0 * 2;
    let twiddle_step2 = twiddle_step0 * 4;
    debug_assert!(m2 >= 8);

    for j in (0..n).step_by(m0 * 2) {
        for k in (0..m2).step_by(8) {
            let p0 = block_at(coeff, j + k);
            let p1 = block_at(coeff, j + k + m2);
            let p2 = block_at(coeff, j + k + m1);
            let p3 = block_at(coeff, j + k + m1 + m2);
            let p4 = block_at(coeff, j + k + m0);
            let p5 = block_at(coeff, j + k + m0 + m2);
            let p6 = block_at(coeff, j + k + m0 + m1);
            let p7 = block_at(coeff, j + k + m0 + m1 + m2);

            let tw_l0 = load_twiddle_vector_pbf(elements_mont, k, twiddle_step0);
            let tw_l1 = load_twiddle_vector_pbf(elements_mont, k + m2, twiddle_step0);
            let tw_l2 = load_twiddle_vector_pbf(elements_mont, k + m1, twiddle_step0);
            let tw_l3 = load_twiddle_vector_pbf(elements_mont, k + m1 + m2, twiddle_step0);
            let tw_m0 = load_twiddle_vector_pbf(elements_mont, k, twiddle_step1);
            let tw_m1 = load_twiddle_vector_pbf(elements_mont, k + m2, twiddle_step1);
            let tw_n = load_twiddle_vector_pbf(elements_mont, k, twiddle_step2);

            let (r0, r1, r2, r3, r4, r5, r6, r7) = dif_three_layer_from_registers_ext(
                p0, p1, p2, p3, p4, p5, p6, p7, tw_l0, tw_l1, tw_l2, tw_l3, tw_m0, tw_m1, tw_n,
            );

            store_block(coeff, j + k, r0);
            store_block(coeff, j + k + m2, r1);
            store_block(coeff, j + k + m1, r2);
            store_block(coeff, j + k + m1 + m2, r3);
            store_block(coeff, j + k + m0, r4);
            store_block(coeff, j + k + m0 + m2, r5);
            store_block(coeff, j + k + m0 + m1, r6);
            store_block(coeff, j + k + m0 + m1 + m2, r7);
        }
    }
}

/// Tail-3-layer pair-major pass. Mirrors `dif_packed_tail_three_layer_pairmajor`.
/// Operates on 64-logical-scalar groups (= 8 PEF blocks per group) so a group's
/// 8 PEF inputs sit in 8 contiguous slots of `coeff`. Internally re-uses the
/// base `tail_twiddle_last3 / last2 / last1` PBF vectors (read-only via
/// `MamaBearFFT::tail_twiddle_lastN`) -- 0 duplicate state.
#[inline]
fn dif_tail_three_layer_pairmajor_ext<T: ExtFftElem>(coeff: &mut [T], fft: &MamaBearFFT) {
    debug_assert_eq!(coeff.len() & 7, 0);

    const SHUF4_LO: [u64; 8] = [0, 1, 2, 3, 8, 9, 10, 11];
    const SHUF4_HI: [u64; 8] = [4, 5, 6, 7, 12, 13, 14, 15];
    const SHUF2_LO: [u64; 8] = [0, 1, 8, 9, 4, 5, 12, 13];
    const SHUF2_HI: [u64; 8] = [2, 3, 10, 11, 6, 7, 14, 15];
    const SHUF1_LO: [u64; 8] = [0, 8, 2, 10, 4, 12, 6, 14];
    const SHUF1_HI: [u64; 8] = [1, 9, 3, 11, 5, 13, 7, 15];

    let tw3 = fft.tail_twiddle_last3();
    let tw2 = fft.tail_twiddle_last2();
    let tw1 = fft.tail_twiddle_last1();

    for group in coeff.chunks_exact_mut(8) {
        let mut p0 = group[0];
        let mut p1 = group[1];
        let mut p2 = group[2];
        let mut p3 = group[3];
        let mut p4 = group[4];
        let mut p5 = group[5];
        let mut p6 = group[6];
        let mut p7 = group[7];

        // Stage 1: shuf4 + butterfly + tw3 (4 register pairs)
        (p0, p1) = tail_stage_ext(p0, p1, SHUF4_LO, SHUF4_HI, tw3);
        (p2, p3) = tail_stage_ext(p2, p3, SHUF4_LO, SHUF4_HI, tw3);
        (p4, p5) = tail_stage_ext(p4, p5, SHUF4_LO, SHUF4_HI, tw3);
        (p6, p7) = tail_stage_ext(p6, p7, SHUF4_LO, SHUF4_HI, tw3);

        // Stage 2: shuf2 + butterfly + tw2
        (p0, p1) = tail_stage_ext(p0, p1, SHUF2_LO, SHUF2_HI, tw2);
        (p2, p3) = tail_stage_ext(p2, p3, SHUF2_LO, SHUF2_HI, tw2);
        (p4, p5) = tail_stage_ext(p4, p5, SHUF2_LO, SHUF2_HI, tw2);
        (p6, p7) = tail_stage_ext(p6, p7, SHUF2_LO, SHUF2_HI, tw2);

        // Stage 3: shuf1 + butterfly + tw1, with final reduce (each component < P)
        (p0, p1) = tail_stage_final_reduce_ext(p0, p1, SHUF1_LO, SHUF1_HI, tw1);
        (p2, p3) = tail_stage_final_reduce_ext(p2, p3, SHUF1_LO, SHUF1_HI, tw1);
        (p4, p5) = tail_stage_final_reduce_ext(p4, p5, SHUF1_LO, SHUF1_HI, tw1);
        (p6, p7) = tail_stage_final_reduce_ext(p6, p7, SHUF1_LO, SHUF1_HI, tw1);

        group[0] = p0;
        group[1] = p1;
        group[2] = p2;
        group[3] = p3;
        group[4] = p4;
        group[5] = p5;
        group[6] = p6;
        group[7] = p7;
    }
}

#[inline(always)]
fn tail_stage_ext<T: ExtFftElem>(
    left: T,
    right: T,
    shuffle_lo: [u64; 8],
    shuffle_hi: [u64; 8],
    twiddle: PackedMamaBearAVX512,
) -> (T, T) {
    let u = left.permute2(right, shuffle_lo);
    let v = left.permute2(right, shuffle_hi);
    dif_butterfly_pair_ext(u, v, twiddle)
}

#[inline(always)]
fn tail_stage_final_reduce_ext<T: ExtFftElem>(
    left: T,
    right: T,
    shuffle_lo: [u64; 8],
    shuffle_hi: [u64; 8],
    twiddle: PackedMamaBearAVX512,
) -> (T, T) {
    let u = left.permute2(right, shuffle_lo);
    let v = left.permute2(right, shuffle_hi);
    dif_butterfly_pair_final_reduce_ext(u, v, twiddle)
}

// ---------------------------------------------------------------------------
// Small-size scalar tail for `n < 64`. Mirrors base `dif_scalar_butterfly_layer`
// + `fft_in_place_small` + `reorder_pair_adjacent_to_pair_major_blocks_and_reduce_in_place`,
// but operates on the per-element-type Scalar (Ext3) instead of base.
//
// Strategy: unpack the PEF buffer to a `Vec<E::Scalar>`, run the layer loop in
// scalar form (one butterfly per output position, identical to base's stride
// < 8 path but for ext), reorder pair-adjacent -> pair-major + reduce, then
// repack to PEF. Performance is uncritical here -- this path only runs for
// `fft.size() < 64`, which is test-only territory; the production lookup
// scenario (nv >= 8) always has `fft.size() >= 256` and stays on the
// hot packed path.
// ---------------------------------------------------------------------------

/// DIF scalar butterfly layer (logical Vec<E::Scalar>). Same algorithm as base
/// `dif_scalar_butterfly_layer`, just with E componentwise instead of single base.
///
/// Range invariants per component (mirror of base):
/// - Inputs `u`, `v` in [0, 2P), twiddle `w` in [0, 2P)
/// - `u + v` in [0, 4P) -> `con_sub_xp(2)` -> [0, 2P)
/// - `diff = u + 2P - v` in [0, 4P)
/// - `diff.mul_base_elem(w)` componentwise mont_mul, output in [0, 1.75P) (per
///   base "mont_mul both inputs < 4P -> output < 3P" with w < 2P + diff < 4P)
#[inline]
fn dif_scalar_butterfly_layer_ext<E>(
    coeff: &mut [E],
    elements_mont: &[MamaBearScalar],
    m: usize,
    twiddle_step: usize,
) where
    E: Copy + LazyReduction + Field<BaseField = MamaBearScalar>,
{
    let n = coeff.len();
    for j in (0..n).step_by(m * 2) {
        for k in 0..m {
            let w = elements_mont[k * twiddle_step];
            let u = coeff[j + k];
            let v = coeff[j + k + m];
            coeff[j + k] = u.lazy_add(v).con_sub_xp(2);
            let diff = u.lazy_add_xp(2).lazy_sub(v);
            coeff[j + k + m] = diff.mul_base_elem(w);
        }
    }
}

/// Reorder pair-adjacent block layout to pair-major blocked + reduce each
/// element to canonical [0, P). Mirror of base
/// `reorder_pair_adjacent_to_pair_major_blocks_and_reduce_in_place`.
#[inline]
fn reorder_pair_adjacent_to_pair_major_blocks_and_reduce_ext<E>(coeff: &mut [E])
where
    E: Copy + Default + LazyReduction,
{
    let pair_count = coeff.len() >> 1;
    if pair_count == 0 {
        for value in coeff.iter_mut() {
            *value = value.reduce();
        }
        return;
    }
    let pairs_per_block = MamaBearFFT::pair_slots_per_block_for_pair_count(pair_count);
    let block_len = pairs_per_block * 2;
    let mut tmp: [E; 16] = [E::default(); 16];
    for block in coeff.chunks_exact_mut(block_len) {
        tmp[..block_len].copy_from_slice(block);
        for lane in 0..pairs_per_block {
            block[lane] = tmp[2 * lane].reduce();
            block[pairs_per_block + lane] = tmp[2 * lane + 1].reduce();
        }
    }
}

/// Small-FFT fallback: unpack -> per-layer scalar butterfly -> reorder to
/// pair-major -> repack. Handles any `n < 64` and any `start_layer < log_n`.
/// Output layout matches the packed path (pair-major blocked, each component
/// reduced to [0, P)).
#[inline]
fn fft_in_place_from_layer_small_ext<T: ExtFftElem>(
    coeff_packed: &mut [T],
    fft: &MamaBearFFT,
    start_layer: usize,
) {
    let n_packed = coeff_packed.len();
    let n = n_packed * 8;
    let log_n = fft.log_order as usize;
    debug_assert!(start_layer < log_n);

    // Unpack to Vec<E::Scalar>, length n.
    let mut scalar: Vec<T::Scalar> = Vec::with_capacity(n);
    for block in coeff_packed.iter() {
        let lanes = block.unpack_to_scalars();
        scalar.extend_from_slice(&lanes);
    }

    // Run scalar FFT layers.
    let elements_mont = fft.elements_mont();
    for layer in start_layer..log_n {
        let m = n >> (layer + 1);
        let twiddle_step = 1usize << layer;
        dif_scalar_butterfly_layer_ext::<T::Scalar>(&mut scalar, elements_mont, m, twiddle_step);
    }

    // pair-adjacent -> pair-major + canonicalise each component.
    reorder_pair_adjacent_to_pair_major_blocks_and_reduce_ext::<T::Scalar>(&mut scalar);

    // Repack to PEF.
    for (block_idx, block) in coeff_packed.iter_mut().enumerate() {
        let mut lanes: [T::Scalar; 8] = [T::Scalar::default(); 8];
        for lane in 0..8 {
            lanes[lane] = scalar[block_idx * 8 + lane];
        }
        *block = T::pack_from_scalars(lanes);
    }
}

// ---------------------------------------------------------------------------
// Generic top-level dispatch.
// ---------------------------------------------------------------------------

/// In-place DIF FFT on Montgomery-form PEF blocks (length `n / 8`). Output is
/// pair-major blocked with each component reduced to [0, P).
fn fft_in_place_ext<T: ExtFftElem>(coeff: &mut [T], fft: &MamaBearFFT) {
    let n_packed = coeff.len();
    let n = n_packed * 8;
    assert_eq!(n, fft.size(), "coeff length must equal fft.size() / 8");
    fft_in_place_from_layer_ext(coeff, fft, 0);
}

/// Continue an in-progress DIF pass from `start_layer` to the end (mirrors base
/// `fft_in_place_from_layer`). Used by the zero-padded path: the prefix3 chirp
/// expansion materialises layers 0..3 in one sweep, then this helper runs from
/// layer 3 (or 6 in the fused-dense3 case) to the end.
///
/// For `n < 64` we fall back to the per-layer scalar path (mirrors base
/// `fft_in_place_small` + the `n < 64` branch of base `fft_in_place_from_layer`):
/// unpack -> scalar butterflies -> pair-major reorder -> repack. The hot
/// production path (n >= 64) stays on the fully packed code below.
fn fft_in_place_from_layer_ext<T: ExtFftElem>(
    coeff: &mut [T],
    fft: &MamaBearFFT,
    start_layer: usize,
) {
    let n_packed = coeff.len();
    let n = n_packed * 8;
    let log_n = fft.log_order as usize;

    if start_layer >= log_n {
        // All layers done: just canonicalise each block.
        for v in coeff.iter_mut() {
            *v = v.reduce();
        }
        return;
    }

    if n < 64 {
        // Small-FFT fallback (formerly妥协 2: panic stub).
        fft_in_place_from_layer_small_ext(coeff, fft, start_layer);
        return;
    }

    let elements_mont = fft.elements_mont();
    let mut layer = start_layer;
    let mut remaining = log_n - start_layer;

    while remaining > 3 {
        let m0 = n >> (layer + 1);
        let twiddle_step0 = 1usize << layer;

        if remaining >= 6 {
            dif_three_layer_no_shuffle_ext(coeff, elements_mont, m0, twiddle_step0);
            layer += 3;
            remaining -= 3;
        } else if remaining == 5 {
            dif_two_layer_no_shuffle_ext(coeff, elements_mont, m0, twiddle_step0);
            layer += 2;
            remaining -= 2;
        } else {
            debug_assert_eq!(remaining, 4);
            dif_one_layer_no_shuffle_ext(coeff, elements_mont, m0, twiddle_step0);
            layer += 1;
            remaining -= 1;
        }
    }

    debug_assert_eq!(remaining, 3);
    dif_tail_three_layer_pairmajor_ext(coeff, fft);
}

// ---------------------------------------------------------------------------
// store_dif_packed_three_layer_outputs_ext: PEF analogue of base
// `store_dif_packed_three_layer_outputs`. Writes 8 PEF results back to
// positions (k, k+m2, k+m1, k+m1+m2, k+m0, k+m0+m2, k+m0+m1, k+m0+m1+m2)
// within the destination block.
// ---------------------------------------------------------------------------

#[inline(always)]
fn store_dif_packed_three_layer_outputs_ext<T: ExtFftElem>(
    coeff: &mut [T],
    k_logical: usize,
    m0: usize,
    m1: usize,
    m2: usize,
    outputs: (T, T, T, T, T, T, T, T),
) {
    let (r0, r1, r2, r3, r4, r5, r6, r7) = outputs;
    store_block(coeff, k_logical, r0);
    store_block(coeff, k_logical + m2, r1);
    store_block(coeff, k_logical + m1, r2);
    store_block(coeff, k_logical + m1 + m2, r3);
    store_block(coeff, k_logical + m0, r4);
    store_block(coeff, k_logical + m0 + m2, r5);
    store_block(coeff, k_logical + m0 + m1, r6);
    store_block(coeff, k_logical + m0 + m1 + m2, r7);
}

// ---------------------------------------------------------------------------
// prefix3 dense3 block builders -- iteration 1C kernels.
// Each builder takes 8 PEF raw inputs (already in Montgomery form), applies a
// per-position chirp (PBF twiddle, beta-dependent), then runs the dense 3-layer
// DIF butterfly. Two variants: (a) on-the-fly chirp from base twiddles, used
// when `chirp_prefix3` is not precomputed (large domains exceeding L2 cache);
// (b) chirp_table direct loads, used when precomputed.
//
// Range invariant: raw_i in [0, P) per component (after to_montgomery).
// chirp factors in [0, 1.5P]. mont_mul output in [0, 1.5P) per component.
// con_sub_xp(1) -> [0, P) per component, fed into dense3 layers as inputs.
// ---------------------------------------------------------------------------

#[inline(always)]
fn prefix3_dense3_block_from_base_twiddle_ext<T: ExtFftElem>(
    raw0: T,
    raw1: T,
    raw2: T,
    raw3: T,
    raw4: T,
    raw5: T,
    raw6: T,
    raw7: T,
    base_tw: PackedMamaBearAVX512,
    offset_m0: PackedMamaBearAVX512,
    offset_m1: PackedMamaBearAVX512,
    offset_m2: PackedMamaBearAVX512,
    tw_l0: PackedMamaBearAVX512,
    tw_l1: PackedMamaBearAVX512,
    tw_l2: PackedMamaBearAVX512,
    tw_l3: PackedMamaBearAVX512,
    tw_m0: PackedMamaBearAVX512,
    tw_m1: PackedMamaBearAVX512,
    tw_n: PackedMamaBearAVX512,
) -> (T, T, T, T, T, T, T, T) {
    // Chirp factors for the 8 input positions, computed via geometric chained
    // PBF mul (same as base path).
    //
    // base_tw  = omega^(beta * k)
    // tw_seg1  = omega^(beta * (k + m2))
    // tw_seg2  = omega^(beta * (k + m1))
    // tw_seg3  = omega^(beta * (k + m1 + m2))
    // tw_seg4  = omega^(beta * (k + m0))
    // tw_seg5  = omega^(beta * (k + m0 + m2))
    // tw_seg6  = omega^(beta * (k + m0 + m1))
    // tw_seg7  = omega^(beta * (k + m0 + m1 + m2))
    let tw_seg1 = (base_tw * offset_m2).con_sub_xp(1);
    let tw_seg2 = (base_tw * offset_m1).con_sub_xp(1);
    let tw_seg3 = (tw_seg2 * offset_m2).con_sub_xp(1);
    let tw_seg4 = (base_tw * offset_m0).con_sub_xp(1);
    let tw_seg5 = (tw_seg4 * offset_m2).con_sub_xp(1);
    let tw_seg6 = (tw_seg4 * offset_m1).con_sub_xp(1);
    let tw_seg7 = (tw_seg6 * offset_m2).con_sub_xp(1);

    let p0 = raw0.mul_pbf(base_tw).con_sub_xp(1);
    let p1 = raw1.mul_pbf(tw_seg1).con_sub_xp(1);
    let p2 = raw2.mul_pbf(tw_seg2).con_sub_xp(1);
    let p3 = raw3.mul_pbf(tw_seg3).con_sub_xp(1);
    let p4 = raw4.mul_pbf(tw_seg4).con_sub_xp(1);
    let p5 = raw5.mul_pbf(tw_seg5).con_sub_xp(1);
    let p6 = raw6.mul_pbf(tw_seg6).con_sub_xp(1);
    let p7 = raw7.mul_pbf(tw_seg7).con_sub_xp(1);

    dif_three_layer_from_registers_ext(
        p0, p1, p2, p3, p4, p5, p6, p7, tw_l0, tw_l1, tw_l2, tw_l3, tw_m0, tw_m1, tw_n,
    )
}

#[inline(always)]
fn prefix3_dense3_block_from_chirp_table_ext<T: ExtFftElem>(
    raw0: T,
    raw1: T,
    raw2: T,
    raw3: T,
    raw4: T,
    raw5: T,
    raw6: T,
    raw7: T,
    chirp0: PackedMamaBearAVX512,
    chirp1: PackedMamaBearAVX512,
    chirp2: PackedMamaBearAVX512,
    chirp3: PackedMamaBearAVX512,
    chirp4: PackedMamaBearAVX512,
    chirp5: PackedMamaBearAVX512,
    chirp6: PackedMamaBearAVX512,
    chirp7: PackedMamaBearAVX512,
    tw_l0: PackedMamaBearAVX512,
    tw_l1: PackedMamaBearAVX512,
    tw_l2: PackedMamaBearAVX512,
    tw_l3: PackedMamaBearAVX512,
    tw_m0: PackedMamaBearAVX512,
    tw_m1: PackedMamaBearAVX512,
    tw_n: PackedMamaBearAVX512,
) -> (T, T, T, T, T, T, T, T) {
    let p0 = raw0.mul_pbf(chirp0).con_sub_xp(1);
    let p1 = raw1.mul_pbf(chirp1).con_sub_xp(1);
    let p2 = raw2.mul_pbf(chirp2).con_sub_xp(1);
    let p3 = raw3.mul_pbf(chirp3).con_sub_xp(1);
    let p4 = raw4.mul_pbf(chirp4).con_sub_xp(1);
    let p5 = raw5.mul_pbf(chirp5).con_sub_xp(1);
    let p6 = raw6.mul_pbf(chirp6).con_sub_xp(1);
    let p7 = raw7.mul_pbf(chirp7).con_sub_xp(1);

    dif_three_layer_from_registers_ext(
        p0, p1, p2, p3, p4, p5, p6, p7, tw_l0, tw_l1, tw_l2, tw_l3, tw_m0, tw_m1, tw_n,
    )
}

// ---------------------------------------------------------------------------
// Fused dense3 prefix3 main path -- iteration 1C.
// Mirrors `fft_into_zero_padded_prefix3_fused_dense3_pair_major`. Combines chirp
// expansion (replacing global layers 0..2) and dense 3-layer butterfly (global
// layers 3..5) in a single register pass per chunk -- 8 PEF loads + 64 stores
// per outer iteration, no intermediate write-read of the M-element block.
// ---------------------------------------------------------------------------

#[inline]
fn fft_into_zero_padded_prefix3_fused_dense3_pair_major_ext<
    T: ExtFftElem,
    const SRC_IS_MONT: bool,
>(
    fft: &MamaBearFFT,
    raw: &[T],
    buf: &mut [T],
) {
    let n_packed = buf.len();
    let n = n_packed * 8;
    let raw_packed_len = raw.len();
    let raw_len_logical = raw_packed_len * 8;

    debug_assert_eq!(raw_len_logical * 8, n, "prefix_layers must equal 3");
    debug_assert!(
        raw_len_logical >= 64,
        "fused dense3 requires raw_len >= 64 (m2 >= 8)"
    );

    let m0 = raw_len_logical / 2;
    let m1 = raw_len_logical / 4;
    let m2 = raw_len_logical / 8;
    let twiddle_step0 = 1usize << 3;
    let twiddle_step1 = 1usize << 4;
    let twiddle_step2 = 1usize << 5;

    // Split buf into 8 equal-length packed blocks.
    let (block0, tail) = buf.split_at_mut(raw_packed_len);
    let (block1, tail) = tail.split_at_mut(raw_packed_len);
    let (block2, tail) = tail.split_at_mut(raw_packed_len);
    let (block3, tail) = tail.split_at_mut(raw_packed_len);
    let (block4, tail) = tail.split_at_mut(raw_packed_len);
    let (block5, tail) = tail.split_at_mut(raw_packed_len);
    let (block6, block7) = tail.split_at_mut(raw_packed_len);

    let elements_mont = fft.elements_mont();

    // Indices into raw[] for the 8 chunk positions, in *packed* units.
    let r_off_0 = 0usize;
    let r_off_1 = m2 / 8;
    let r_off_2 = m1 / 8;
    let r_off_3 = (m1 + m2) / 8;
    let r_off_4 = m0 / 8;
    let r_off_5 = (m0 + m2) / 8;
    let r_off_6 = (m0 + m1) / 8;
    let r_off_7 = (m0 + m1 + m2) / 8;

    if let Some(chirp_table) = fft.chirp_prefix3() {
        // ---- Chirp-table fast path: 7 betas * raw_packed_len entries each. ----
        // beta_idx 0..6 correspond to beta values 1..7. Block index -> beta_idx
        // via bit-reversed pair order: blocks 1..7 -> beta_vals {4,2,6,1,5,3,7}
        // -> beta_idxs {3,1,5,0,4,2,6}.
        let packed_count = raw_packed_len; // = M/8

        let c_off = [
            0usize, r_off_1, r_off_2, r_off_3, r_off_4, r_off_5, r_off_6, r_off_7,
        ];
        let bi: [usize; 7] = [3, 1, 5, 0, 4, 2, 6];
        let blocks: [&mut [T]; 7] =
            [block1, block2, block3, block4, block5, block6, block7];

        for k_packed in 0..(m2 / 8) {
            let k_logical = k_packed * 8;

            let raw0 = maybe_to_mont::<T, SRC_IS_MONT>(raw[k_packed + r_off_0]);
            let raw1 = maybe_to_mont::<T, SRC_IS_MONT>(raw[k_packed + r_off_1]);
            let raw2 = maybe_to_mont::<T, SRC_IS_MONT>(raw[k_packed + r_off_2]);
            let raw3 = maybe_to_mont::<T, SRC_IS_MONT>(raw[k_packed + r_off_3]);
            let raw4 = maybe_to_mont::<T, SRC_IS_MONT>(raw[k_packed + r_off_4]);
            let raw5 = maybe_to_mont::<T, SRC_IS_MONT>(raw[k_packed + r_off_5]);
            let raw6 = maybe_to_mont::<T, SRC_IS_MONT>(raw[k_packed + r_off_6]);
            let raw7 = maybe_to_mont::<T, SRC_IS_MONT>(raw[k_packed + r_off_7]);

            let tw_l0 = load_twiddle_vector_pbf(elements_mont, k_logical, twiddle_step0);
            let tw_l1 =
                load_twiddle_vector_pbf(elements_mont, k_logical + m2, twiddle_step0);
            let tw_l2 =
                load_twiddle_vector_pbf(elements_mont, k_logical + m1, twiddle_step0);
            let tw_l3 = load_twiddle_vector_pbf(
                elements_mont,
                k_logical + m1 + m2,
                twiddle_step0,
            );
            let tw_m0 = load_twiddle_vector_pbf(elements_mont, k_logical, twiddle_step1);
            let tw_m1 =
                load_twiddle_vector_pbf(elements_mont, k_logical + m2, twiddle_step1);
            let tw_n = load_twiddle_vector_pbf(elements_mont, k_logical, twiddle_step2);

            // Block 0: beta=0, identity (no chirp) -- raw values directly into dense3.
            let block0_out = dif_three_layer_from_registers_ext(
                raw0, raw1, raw2, raw3, raw4, raw5, raw6, raw7, tw_l0, tw_l1, tw_l2, tw_l3,
                tw_m0, tw_m1, tw_n,
            );
            store_dif_packed_three_layer_outputs_ext(
                block0, k_logical, m0, m1, m2, block0_out,
            );

            // Blocks 1..7 with chirp from precomputed table.
            for (blk_idx, &beta_idx) in bi.iter().enumerate() {
                let base = beta_idx * packed_count;
                let out = prefix3_dense3_block_from_chirp_table_ext(
                    raw0,
                    raw1,
                    raw2,
                    raw3,
                    raw4,
                    raw5,
                    raw6,
                    raw7,
                    chirp_table[base + k_packed + c_off[0]],
                    chirp_table[base + k_packed + c_off[1]],
                    chirp_table[base + k_packed + c_off[2]],
                    chirp_table[base + k_packed + c_off[3]],
                    chirp_table[base + k_packed + c_off[4]],
                    chirp_table[base + k_packed + c_off[5]],
                    chirp_table[base + k_packed + c_off[6]],
                    chirp_table[base + k_packed + c_off[7]],
                    tw_l0,
                    tw_l1,
                    tw_l2,
                    tw_l3,
                    tw_m0,
                    tw_m1,
                    tw_n,
                );
                store_dif_packed_three_layer_outputs_ext(
                    blocks[blk_idx],
                    k_logical,
                    m0,
                    m1,
                    m2,
                    out,
                );
            }
        }
    } else {
        // ---- On-the-fly chirp path: compute base_tw + offset_mX broadcasts ----
        let offset_m0_arr: [PackedMamaBearAVX512; 8] = std::array::from_fn(|beta| {
            PackedMamaBearAVX512::broadcast(elements_mont[beta * m0].0)
        });
        let offset_m1_arr: [PackedMamaBearAVX512; 8] = std::array::from_fn(|beta| {
            PackedMamaBearAVX512::broadcast(elements_mont[beta * m1].0)
        });
        let offset_m2_arr: [PackedMamaBearAVX512; 8] = std::array::from_fn(|beta| {
            PackedMamaBearAVX512::broadcast(elements_mont[beta * m2].0)
        });

        for k_packed in 0..(m2 / 8) {
            let k_logical = k_packed * 8;

            let raw0 = maybe_to_mont::<T, SRC_IS_MONT>(raw[k_packed + r_off_0]);
            let raw1 = maybe_to_mont::<T, SRC_IS_MONT>(raw[k_packed + r_off_1]);
            let raw2 = maybe_to_mont::<T, SRC_IS_MONT>(raw[k_packed + r_off_2]);
            let raw3 = maybe_to_mont::<T, SRC_IS_MONT>(raw[k_packed + r_off_3]);
            let raw4 = maybe_to_mont::<T, SRC_IS_MONT>(raw[k_packed + r_off_4]);
            let raw5 = maybe_to_mont::<T, SRC_IS_MONT>(raw[k_packed + r_off_5]);
            let raw6 = maybe_to_mont::<T, SRC_IS_MONT>(raw[k_packed + r_off_6]);
            let raw7 = maybe_to_mont::<T, SRC_IS_MONT>(raw[k_packed + r_off_7]);

            let tw_l0 = load_twiddle_vector_pbf(elements_mont, k_logical, twiddle_step0);
            let tw_l1 =
                load_twiddle_vector_pbf(elements_mont, k_logical + m2, twiddle_step0);
            let tw_l2 =
                load_twiddle_vector_pbf(elements_mont, k_logical + m1, twiddle_step0);
            let tw_l3 = load_twiddle_vector_pbf(
                elements_mont,
                k_logical + m1 + m2,
                twiddle_step0,
            );
            let tw_m0 = load_twiddle_vector_pbf(elements_mont, k_logical, twiddle_step1);
            let tw_m1 =
                load_twiddle_vector_pbf(elements_mont, k_logical + m2, twiddle_step1);
            let tw_n = load_twiddle_vector_pbf(elements_mont, k_logical, twiddle_step2);

            // Block 0: identity.
            let block0_out = dif_three_layer_from_registers_ext(
                raw0, raw1, raw2, raw3, raw4, raw5, raw6, raw7, tw_l0, tw_l1, tw_l2, tw_l3,
                tw_m0, tw_m1, tw_n,
            );
            store_dif_packed_three_layer_outputs_ext(
                block0, k_logical, m0, m1, m2, block0_out,
            );

            // Blocks 1..7: beta_vals (in block order) = {4, 2, 6, 1, 5, 3, 7}.
            // Inlined to match the base path's structure.
            macro_rules! compute_block {
                ($block:expr, $beta:expr) => {{
                    let base_tw = load_twiddle_vector_pbf(elements_mont, k_logical, $beta);
                    let out = prefix3_dense3_block_from_base_twiddle_ext(
                        raw0,
                        raw1,
                        raw2,
                        raw3,
                        raw4,
                        raw5,
                        raw6,
                        raw7,
                        base_tw,
                        offset_m0_arr[$beta],
                        offset_m1_arr[$beta],
                        offset_m2_arr[$beta],
                        tw_l0,
                        tw_l1,
                        tw_l2,
                        tw_l3,
                        tw_m0,
                        tw_m1,
                        tw_n,
                    );
                    store_dif_packed_three_layer_outputs_ext(
                        $block, k_logical, m0, m1, m2, out,
                    );
                }};
            }

            compute_block!(block1, 4);
            compute_block!(block2, 2);
            compute_block!(block3, 6);
            compute_block!(block4, 1);
            compute_block!(block5, 5);
            compute_block!(block6, 3);
            compute_block!(block7, 7);
        }
    }
}

// ---------------------------------------------------------------------------
// Zero-padded prefix3 (non-fused) path -- iteration 1B.
//
// When `raw_len_logical * 8 == fft.size()` (code rate 1/8 = prefix_layers=3),
// 7/8 of the FFT input is zero. The top 3 DIF layers degenerate into a chirp
// multiplication: `buf[q*M + j] = a_j * omega^(beta_q * j)` where
// `beta_q = bitrev3(q)`. We materialise these 8 chirp blocks in one pass and
// hand off to `fft_in_place_from_layer_ext` starting at layer 3.
//
// On-the-fly twiddle path only (no `chirp_table` precomputed reads). The
// chirp_table fast path is iteration 1C.
// ---------------------------------------------------------------------------

/// Chirp expansion for prefix_layers=3. Splits `buf` into 8 equal blocks of
/// length `raw.len()` (in PEF blocks), fills each block with `src * chirp(beta)`
/// where `beta` is the bit-reversed pair index. After this call, layers 0..3
/// of the DIF are materialised and the buffer is ready for layer-3 onwards.
#[inline]
fn fft_into_zero_padded_prefix3_pair_major_ext<T: ExtFftElem, const SRC_IS_MONT: bool>(
    fft: &MamaBearFFT,
    raw: &[T],
    buf: &mut [T],
) {
    let n_packed = buf.len();
    let n = n_packed * 8;
    let raw_packed_len = raw.len();
    let raw_len_logical = raw_packed_len * 8;
    debug_assert_eq!(raw_len_logical * 8, n, "prefix_layers must equal 3");
    debug_assert_eq!(raw_packed_len * 8, n_packed);

    let elements_mont = fft.elements_mont();

    // Split buf into 8 equal-length packed blocks, each of length raw_packed_len.
    let (block0, tail) = buf.split_at_mut(raw_packed_len);
    let (block1, tail) = tail.split_at_mut(raw_packed_len);
    let (block2, tail) = tail.split_at_mut(raw_packed_len);
    let (block3, tail) = tail.split_at_mut(raw_packed_len);
    let (block4, tail) = tail.split_at_mut(raw_packed_len);
    let (block5, tail) = tail.split_at_mut(raw_packed_len);
    let (block6, block7) = tail.split_at_mut(raw_packed_len);

    // Iterate over each input PEF block (covering 8 logical scalar positions
    // per iteration). For each chunk, build the 7 chirp variants via on-the-fly
    // twiddle vectors (steps 1, 2, 4 -> betas 1, 2, 4) chained for betas 3, 5, 6, 7.
    //
    // Range: src after to_montgomery in [0, P) per component. tw{1,2,4} are
    // base twiddles in [0, 1.5P]. mont_mul output in [0, 1.5P) (both inputs
    // < 2P). Chained mont_muls also in [0, 1.5P) (input < 2P). After
    // con_sub_xp(1) each stored block lane is in [0, P).
    for k_packed in 0..raw_packed_len {
        let k_logical = k_packed * 8;
        let src = maybe_to_mont::<T, SRC_IS_MONT>(raw[k_packed]);

        let tw1 = load_twiddle_vector_pbf(elements_mont, k_logical, 1);
        let tw2 = load_twiddle_vector_pbf(elements_mont, k_logical, 2);
        let tw4 = load_twiddle_vector_pbf(elements_mont, k_logical, 4);

        let block1_raw = src.mul_pbf(tw4); // beta = 4
        let block2_raw = src.mul_pbf(tw2); // beta = 2
        let block4_raw = src.mul_pbf(tw1); // beta = 1
        let block3_raw = block2_raw.mul_pbf(tw4); // beta = 6 = 2 + 4
        let block5_raw = block4_raw.mul_pbf(tw4); // beta = 5 = 1 + 4
        let block6_raw = block4_raw.mul_pbf(tw2); // beta = 3 = 1 + 2
        let block7_raw = block6_raw.mul_pbf(tw4); // beta = 7 = 1 + 2 + 4

        // Block ordering follows base path: bit-reversed pair index of
        // {0..7} = {0, 4, 2, 6, 1, 5, 3, 7}. Block index i stores the
        // chirp variant for bitrev3(i).
        block0[k_packed] = src; // beta = 0 (identity)
        block1[k_packed] = block1_raw.con_sub_xp(1);
        block2[k_packed] = block2_raw.con_sub_xp(1);
        block3[k_packed] = block3_raw.con_sub_xp(1);
        block4[k_packed] = block4_raw.con_sub_xp(1);
        block5[k_packed] = block5_raw.con_sub_xp(1);
        block6[k_packed] = block6_raw.con_sub_xp(1);
        block7[k_packed] = block7_raw.con_sub_xp(1);
    }
}

/// `to_montgomery` over a packed PEF buffer in place. Lanes already at 0 (zero
/// padding) are preserved (`to_montgomery(0) = 0`).
#[inline]
fn ext_buf_to_montgomery_in_place<T: ExtFftElem>(buf: &mut [T]) {
    for v in buf.iter_mut() {
        *v = v.to_montgomery();
    }
}

/// Compile-time-conditional `to_montgomery`: when `SRC_IS_MONT == true`,
/// returns the value unchanged (skip `to_mont`). When `false`, applies
/// `to_montgomery()`. The branch is elided at monomorphisation, so calling
/// the chirp / dense3 builder with `SRC_IS_MONT == true` produces strictly
/// fewer mont_muls than the normal path (one mont_mul per logical lane saved).
#[inline(always)]
fn maybe_to_mont<T: ExtFftElem, const SRC_IS_MONT: bool>(v: T) -> T {
    if SRC_IS_MONT {
        v
    } else {
        v.to_montgomery()
    }
}

/// Generic public entry: run an FFT on a PEF buffer of arbitrary input length.
///
/// `raw` is the input in either normal or Montgomery form (controlled by the
/// `SRC_IS_MONT` const generic). When `SRC_IS_MONT == false`, the FFT does
/// `to_montgomery()` on each loaded element as the first step; when `true`,
/// the load is the identity. The const-generic guarantees the unused branch
/// is compile-time eliminated, so calling with `SRC_IS_MONT == true`
/// produces strictly-cheaper code than the normal path (one `to_montgomery`
/// per logical lane saved).
///
/// Output is Montgomery-form, pair-major blocked, with each component
/// canonicalised to `[0, P)`. `buf.len() * 8 == fft.size()`.
fn fft_into_packed_ext<T: ExtFftElem, const SRC_IS_MONT: bool>(
    fft: &MamaBearFFT,
    raw: &[T],
    buf: &mut [T],
) {
    let n_packed = buf.len();
    let n = n_packed * 8;
    assert_eq!(n, fft.size(), "buf len * 8 != fft.size()");
    assert!(
        raw.len() <= n_packed,
        "raw len exceeds domain size in packed units"
    );

    let raw_len_logical = raw.len() * 8;
    let log_n = fft.log_order as usize;

    // Iteration 1B/1C fast path: zero-padded with prefix_layers == 3 (code rate
    // 1/8). This is the lookup main case: input is N/8 of the domain, top 3 DIF
    // layers degenerate to chirp multiplication.
    if let Some(prefix_layers) = fft.zero_padding_prefix_layers_for(raw_len_logical) {
        if prefix_layers == 3 {
            assert!(
                n >= 64,
                "ext FFT prefix3 fast path requires fft.size() >= 64"
            );
            let remaining = log_n - prefix_layers;
            if remaining >= 6 && raw_len_logical >= 64 {
                // Iteration 1C: fused chirp + dense 3-layer in one register pass,
                // then continue from layer 6.
                fft_into_zero_padded_prefix3_fused_dense3_pair_major_ext::<T, SRC_IS_MONT>(
                    fft, raw, buf,
                );
                fft_in_place_from_layer_ext(buf, fft, prefix_layers + 3);
                return;
            }
            // Iteration 1B: non-fused chirp expansion + continue from layer 3.
            fft_into_zero_padded_prefix3_pair_major_ext::<T, SRC_IS_MONT>(fft, raw, buf);
            fft_in_place_from_layer_ext(buf, fft, prefix_layers);
            return;
        }
        // Other prefix_layers values (1, 2, 4, ...) -> A fallback below.
    }

    // A fallback: materialise full domain (raw + zeros), then run
    // `fft_in_place_ext`. Triggered for raw_len == fft.size() (no zero
    // padding) and for prefix_layers != 3 with non-power-of-two raw lengths
    // (which `zero_padding_prefix_layers_for` already rejects).
    let raw_packed_len = raw.len();
    buf[..raw_packed_len].copy_from_slice(raw);
    for v in buf[raw_packed_len..].iter_mut() {
        *v = T::ZERO_NORMAL;
    }
    if !SRC_IS_MONT {
        ext_buf_to_montgomery_in_place(&mut buf[..raw_packed_len]);
    }
    fft_in_place_ext(buf, fft);
}

// ---------------------------------------------------------------------------
// Public API (PEF3 monomorphic entries).
//
// Two flavours:
//   - `fft_into_packed_pef3` -- input in normal (non-Mont) form. The FFT
//     does `to_montgomery()` per loaded element internally as the first step.
//   - `fft_into_packed_pef3_mont` -- input ALREADY in Montgomery form. The
//     FFT skips the per-element `to_montgomery()` step (compile-time elided
//     via `SRC_IS_MONT == true`). Use this entry when feeding output of
//     `batch_invert_pef3_in_place` (or any other Mont-form producer)
//     directly into commit, saving one `to_mont` sweep per polynomial.
// ---------------------------------------------------------------------------

/// Forward FFT for `PackedMamaBearAVX512Ext3`, **normal-form** input.
pub fn fft_into_packed_pef3(
    fft: &MamaBearFFT,
    raw: &[PackedMamaBearAVX512Ext3],
    buf: &mut [PackedMamaBearAVX512Ext3],
) {
    fft_into_packed_ext::<PackedMamaBearAVX512Ext3, false>(fft, raw, buf);
}

/// Forward FFT for `PackedMamaBearAVX512Ext3`, **Montgomery-form** input.
/// Skips the internal `to_montgomery()` step. Use this when feeding the
/// output of `batch_invert_pef3_in_place` (or any Mont-form polynomial)
/// directly into commit, avoiding a redundant `from_mont -> to_mont` round-trip.
pub fn fft_into_packed_pef3_mont(
    fft: &MamaBearFFT,
    raw_mont: &[PackedMamaBearAVX512Ext3],
    buf: &mut [PackedMamaBearAVX512Ext3],
) {
    fft_into_packed_ext::<PackedMamaBearAVX512Ext3, true>(fft, raw_mont, buf);
}

// ---------------------------------------------------------------------------
// Helpers used in tests only.
// ---------------------------------------------------------------------------

/// Reinterpret `&[PackedMamaBearAVX512Ext3]` as its c0 / c1 / c2 components.
/// Each PEF3 block has c0/c1/c2 laid out contiguously in memory (`#[repr(C)]`);
/// viewing it lane-wise requires per-block strided extraction. We don't need a
/// packed-view helper for runtime use -- only the tests need this for the
/// componentwise correspondence check.
#[cfg(test)]
mod componentwise {
    use super::*;

    pub fn pef3_extract_components(
        src: &[PackedMamaBearAVX512Ext3],
    ) -> (
        Vec<MamaBearScalar>,
        Vec<MamaBearScalar>,
        Vec<MamaBearScalar>,
    ) {
        let mut c0 = Vec::with_capacity(src.len() * 8);
        let mut c1 = Vec::with_capacity(src.len() * 8);
        let mut c2 = Vec::with_capacity(src.len() * 8);
        for block in src {
            let arr0 = unsafe { block.c0.array };
            let arr1 = unsafe { block.c1.array };
            let arr2 = unsafe { block.c2.array };
            for i in 0..8 {
                c0.push(MamaBearScalar(arr0[i]));
                c1.push(MamaBearScalar(arr1[i]));
                c2.push(MamaBearScalar(arr2[i]));
            }
        }
        (c0, c1, c2)
    }
}

// Suppress unused-import warning when tests are off.
#[allow(dead_code)]
fn _ptr_keep_alive() {
    let _ = ptr::null::<u8>();
}

// ---------------------------------------------------------------------------
// Tests.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::componentwise::pef3_extract_components;
    use super::*;
    use rand::rngs::SmallRng;
    use rand::{RngCore, SeedableRng};

    fn make_rng(seed: u64) -> SmallRng {
        SmallRng::seed_from_u64(seed)
    }

    /// Run the closure on a fresh thread with a 16 MB stack. The fused dense3
    /// kernel returns 8-tuples of PEF (1536 bytes for Ext3) which inline
    /// cleanly in release but balloon stack frames at
    /// `-Copt-level=0` (debug). The default test-thread stack is 2 MB and
    /// overflows under that path.
    fn run_with_large_stack<F: FnOnce() + Send + 'static>(f: F) {
        std::thread::Builder::new()
            .stack_size(16 * 1024 * 1024)
            .spawn(f)
            .expect("spawn large-stack test thread")
            .join()
            .expect("test thread panicked");
    }

    fn random_pef3(n_packed: usize, rng: &mut impl RngCore) -> Vec<PackedMamaBearAVX512Ext3> {
        let mut out = Vec::with_capacity(n_packed);
        for _ in 0..n_packed {
            let mut a0 = [0u64; 8];
            let mut a1 = [0u64; 8];
            let mut a2 = [0u64; 8];
            for i in 0..8 {
                a0[i] = rng.next_u64() % crate::field::mamabear::P;
                a1[i] = rng.next_u64() % crate::field::mamabear::P;
                a2[i] = rng.next_u64() % crate::field::mamabear::P;
            }
            out.push(PackedMamaBearAVX512Ext3::new(
                PackedMamaBearAVX512::from_array(a0),
                PackedMamaBearAVX512::from_array(a1),
                PackedMamaBearAVX512::from_array(a2),
            ));
        }
        out
    }

    #[test]
    fn fft_into_packed_pef3_matches_componentwise_pbf_zero_padded_prefix3_non_fused() {
        run_with_large_stack(|| {
        let mut rng = make_rng(0xC0DE_FACE);

        for log_order in [6usize, 7, 8] {
            let fft = MamaBearFFT::new(log_order as u32);
            let n = 1usize << log_order;
            let n_packed = n / 8;
            let raw_packed_len = n_packed / 8;
            assert!(raw_packed_len >= 1);

            let raw_pef = random_pef3(raw_packed_len, &mut rng);
            let (raw_c0, raw_c1, raw_c2) = pef3_extract_components(&raw_pef);

            let mut ref_c0 = vec![MamaBearScalar(0); n];
            let mut ref_c1 = vec![MamaBearScalar(0); n];
            let mut ref_c2 = vec![MamaBearScalar(0); n];
            fft.fft_into(&raw_c0, &mut ref_c0);
            fft.fft_into(&raw_c1, &mut ref_c1);
            fft.fft_into(&raw_c2, &mut ref_c2);

            let mut buf_pef = vec![PackedMamaBearAVX512Ext3::ZERO_NORMAL; n_packed];
            fft_into_packed_pef3(&fft, &raw_pef, &mut buf_pef);
            let (got_c0, got_c1, got_c2) = pef3_extract_components(&buf_pef);

            assert_eq!(
                got_c0, ref_c0,
                "PEF3 ext FFT (zero-padded prefix3) c0 mismatch at log_order={log_order}"
            );
            assert_eq!(
                got_c1, ref_c1,
                "PEF3 ext FFT (zero-padded prefix3) c1 mismatch at log_order={log_order}"
            );
            assert_eq!(
                got_c2, ref_c2,
                "PEF3 ext FFT (zero-padded prefix3) c2 mismatch at log_order={log_order}"
            );
        }
        });
    }

    /// Iteration 1C fused dense3 path. Triggered when prefix_layers == 3,
    /// remaining_layers >= 6, raw_len_logical >= 64. We test both
    /// chirp_table-populated and chirp_table-empty (on-the-fly chirp) variants
    /// to cover both branches of the fused kernel.
    #[test]
    fn fft_into_packed_pef3_fused_dense3_matches_componentwise_chirp_table() {
        run_with_large_stack(|| {
        let mut rng = make_rng(0xA11C_E000);
        for log_order in [9usize, 10, 12] {
            let mut fft = MamaBearFFT::new(log_order as u32);
            fft.precompute_chirp_prefix3();
            let n = 1usize << log_order;
            let n_packed = n / 8;
            let raw_packed_len = n_packed / 8;

            let raw_pef = random_pef3(raw_packed_len, &mut rng);
            let (raw_c0, raw_c1, raw_c2) = pef3_extract_components(&raw_pef);

            let mut ref_c0 = vec![MamaBearScalar(0); n];
            let mut ref_c1 = vec![MamaBearScalar(0); n];
            let mut ref_c2 = vec![MamaBearScalar(0); n];
            fft.fft_into(&raw_c0, &mut ref_c0);
            fft.fft_into(&raw_c1, &mut ref_c1);
            fft.fft_into(&raw_c2, &mut ref_c2);

            let mut buf_pef = vec![PackedMamaBearAVX512Ext3::ZERO_NORMAL; n_packed];
            fft_into_packed_pef3(&fft, &raw_pef, &mut buf_pef);
            let (got_c0, got_c1, got_c2) = pef3_extract_components(&buf_pef);

            assert_eq!(
                got_c0, ref_c0,
                "PEF3 fused dense3 (chirp_table) c0 mismatch at log_order={log_order}"
            );
            assert_eq!(
                got_c1, ref_c1,
                "PEF3 fused dense3 (chirp_table) c1 mismatch at log_order={log_order}"
            );
            assert_eq!(
                got_c2, ref_c2,
                "PEF3 fused dense3 (chirp_table) c2 mismatch at log_order={log_order}"
            );
        }
        });
    }

    #[test]
    fn fft_into_packed_pef3_fused_dense3_matches_componentwise_on_the_fly() {
        run_with_large_stack(|| {
        let mut rng = make_rng(0xBABE_FACE);
        for log_order in [9usize, 10, 12] {
            let fft = MamaBearFFT::new(log_order as u32);
            assert!(fft.chirp_prefix3().is_none());
            let n = 1usize << log_order;
            let n_packed = n / 8;
            let raw_packed_len = n_packed / 8;

            let raw_pef = random_pef3(raw_packed_len, &mut rng);
            let (raw_c0, raw_c1, raw_c2) = pef3_extract_components(&raw_pef);

            let mut ref_c0 = vec![MamaBearScalar(0); n];
            let mut ref_c1 = vec![MamaBearScalar(0); n];
            let mut ref_c2 = vec![MamaBearScalar(0); n];
            fft.fft_into(&raw_c0, &mut ref_c0);
            fft.fft_into(&raw_c1, &mut ref_c1);
            fft.fft_into(&raw_c2, &mut ref_c2);

            let mut buf_pef = vec![PackedMamaBearAVX512Ext3::ZERO_NORMAL; n_packed];
            fft_into_packed_pef3(&fft, &raw_pef, &mut buf_pef);
            let (got_c0, got_c1, got_c2) = pef3_extract_components(&buf_pef);

            assert_eq!(
                got_c0, ref_c0,
                "PEF3 fused dense3 (on-the-fly) c0 mismatch at log_order={log_order}"
            );
            assert_eq!(
                got_c1, ref_c1,
                "PEF3 fused dense3 (on-the-fly) c1 mismatch at log_order={log_order}"
            );
            assert_eq!(
                got_c2, ref_c2,
                "PEF3 fused dense3 (on-the-fly) c2 mismatch at log_order={log_order}"
            );
        }
        });
    }

    #[test]
    fn fft_into_packed_pef3_matches_componentwise_pbf_full_domain() {
        let mut rng = make_rng(0xBEEF_0FF1);

        // Small-size sweep: exercises the scalar tail (log_order 3-5) and
        // the packed paths (>= 6).
        for log_order in [3usize, 4, 5, 6, 7, 8, 10] {
            let fft = MamaBearFFT::new(log_order as u32);
            let n = 1usize << log_order;
            let n_packed = n / 8;

            let raw_pef = random_pef3(n_packed, &mut rng);
            let (raw_c0, raw_c1, raw_c2) = pef3_extract_components(&raw_pef);

            let mut ref_c0 = vec![MamaBearScalar(0); n];
            let mut ref_c1 = vec![MamaBearScalar(0); n];
            let mut ref_c2 = vec![MamaBearScalar(0); n];
            fft.fft_into(&raw_c0, &mut ref_c0);
            fft.fft_into(&raw_c1, &mut ref_c1);
            fft.fft_into(&raw_c2, &mut ref_c2);

            let mut buf_pef = vec![PackedMamaBearAVX512Ext3::ZERO_NORMAL; n_packed];
            fft_into_packed_pef3(&fft, &raw_pef, &mut buf_pef);
            let (got_c0, got_c1, got_c2) = pef3_extract_components(&buf_pef);

            assert_eq!(
                got_c0, ref_c0,
                "PEF3 ext FFT c0 mismatch at log_order={log_order}"
            );
            assert_eq!(
                got_c1, ref_c1,
                "PEF3 ext FFT c1 mismatch at log_order={log_order}"
            );
            assert_eq!(
                got_c2, ref_c2,
                "PEF3 ext FFT c2 mismatch at log_order={log_order}"
            );
        }
    }

    /// Compare base (PBF) / ext3 (PEF3) FFT cost on the same domain size.
    /// `#[ignore]` so it runs only on demand:
    ///
    /// ```bash
    /// RUSTFLAGS="-C target-cpu=native" cargo test -p arithmetic --release \
    ///     fft_mamabear_ext::tests::bench_base_vs_ext_fft -- --nocapture --ignored
    /// ```
    #[test]
    #[ignore]
    fn bench_base_vs_ext_fft() {
        run_with_large_stack(|| {
        use std::time::Instant;
        let mut rng = make_rng(0xBEEF);
        // Pick log_order that triggers the fused dense3 path (>= 9). nv=14 is
        // the standard FFT-bench size; raw_len = N/8 = 2^11 = 2048.
        for log_order in [12usize, 14, 16, 18] {
            let mut fft = MamaBearFFT::new(log_order as u32);
            fft.precompute_chirp_prefix3();
            let n = 1usize << log_order;
            let n_packed = n / 8;
            let raw_packed_len = n_packed / 8;

            // Warmup + iter counts
            let warmup = 3usize;
            let iters = if log_order <= 14 { 50usize } else { 20usize };

            // --- BASE PBF: raw of N/8 logical scalars ---
            let raw_pbf: Vec<MamaBearScalar> = (0..n / 8)
                .map(|_| MamaBearScalar(rng.next_u64() % crate::field::mamabear::P))
                .collect();
            let mut buf_pbf = vec![MamaBearScalar(0); n];
            for _ in 0..warmup {
                fft.fft_into(&raw_pbf, &mut buf_pbf);
            }
            let t0 = Instant::now();
            for _ in 0..iters {
                fft.fft_into(&raw_pbf, &mut buf_pbf);
            }
            let t_pbf = t0.elapsed().as_micros() as f64 / iters as f64;

            // --- PEF3 ---
            let raw_pef3 = random_pef3(raw_packed_len, &mut rng);
            let mut buf_pef3 = vec![PackedMamaBearAVX512Ext3::ZERO_NORMAL; n_packed];
            for _ in 0..warmup {
                fft_into_packed_pef3(&fft, &raw_pef3, &mut buf_pef3);
            }
            let t0 = Instant::now();
            for _ in 0..iters {
                fft_into_packed_pef3(&fft, &raw_pef3, &mut buf_pef3);
            }
            let t_pef3 = t0.elapsed().as_micros() as f64 / iters as f64;

            eprintln!(
                "log_order={log_order} (n={n}): base_pbf={:.1}us pef3={:.1}us ({:.2}x)",
                t_pbf,
                t_pef3,
                t_pef3 / t_pbf,
            );
        }
        });
    }
}
