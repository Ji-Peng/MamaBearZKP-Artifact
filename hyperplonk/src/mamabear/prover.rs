use std::marker::PhantomData;

use arithmetic::{
    field::{
        mamabear::{
            LazyReduction, MamaBearScalar, MamaBearScalarExt3,
            PackedExtensionField, PackedExtensionPairStride, PackedMamaBearAVX512,
            PackedMamaBearAVX512Ext3,
        },
        Field,
    },
    poly::MultiLinearPoly,
};
use poly_commit::{
    deepfold::MerkleRoot,
    deepfold_mamabear::{
        DeepFoldExtField, DeepFoldMamaBearParam, DeepFoldMamaBearProver,
        DeepFoldMamaBearVerifier,
    },
    CommitmentSerde,
};
use util::fiat_shamir::{Proof, Transcript};

use crate::{
    circuit::Circuit,
    prodcheck_mamabear_perwire::ProdEqCheckMamaBearPerWire,
    sumcheck_mamabear::{MontgomeryOps, SumcheckExtField, SumcheckMamaBear},
};

type SBF = MamaBearScalar;
type PBF = PackedMamaBearAVX512;
type SEF3 = MamaBearScalarExt3;
type PEF3 = PackedMamaBearAVX512Ext3;

const ZERO_CHECK_EXT3_ELL0: usize = 2;
const ZERO_CHECK_EXT3_PAR_ELL0: usize = 2;
const NUM_WIRES: usize = 3;
const NUM_FIXED: usize = 4;
const NUM_ALL_TABLES: usize = 7;
const ID_SHIFT_1: u64 = 1 << 29;
const ID_SHIFT_2: u64 = 1 << 30;

/// log2(SIMD lanes) — determines the Theorem-2 variable permutation size.
pub(crate) const SIGMA_W: usize = 3;

/// Convert a point from SIMD-round order (output by a normal-layout SIMD
/// sumcheck) to natural variable order (point[k] is the value plugged into
/// x_k when calling MLE eval).
///
/// With 2^w SIMD lanes (w = `SIGMA_W` = 3 here) and normal memory layout
/// `evals[p][lambda] = f(bin(2^w * p + lambda))`, the SIMD sumcheck in
/// this crate folds pairs of adjacent packed blocks first (eliminating
/// bits of the vector index p, i.e. x_w, x_{w+1}, ..., x_{nv-1}) and then
/// folds lanes of the final packed block (eliminating x_0, ..., x_{w-1}).
/// Hence the elimination permutation is
///
/// ```text
/// sigma: k -> SIMD-round index that eliminates x_k
/// sigma = [x_w, x_{w+1}, ..., x_{nv-1}, x_0, ..., x_{w-1}]
/// ```
///
/// so `simd_point[i]` is the value substituted for x_{sigma(i)}. To get
/// `natural_point[k] = x_k`'s value we invert sigma:
///
/// ```text
/// natural_point[k] = simd_point[sigma^-1(k)]
/// ```
///
/// which is a right-rotation of `simd_point` by `w` positions.
pub(crate) fn simd_to_natural_point<T: Copy>(simd_point: &[T]) -> Vec<T> {
    let nv = simd_point.len();
    if nv <= SIGMA_W {
        // Too few variables for the SIMD permutation to take effect. Just
        // return as-is; the caller should handle small cases carefully.
        return simd_point.to_vec();
    }
    let mut result = Vec::with_capacity(nv);
    for k in 0..nv {
        let src = if k >= SIGMA_W { k - SIGMA_W } else { k + nv - SIGMA_W };
        result.push(simd_point[src]);
    }
    result
}

pub trait MamaBearExtConfig:
    DeepFoldExtField + MontgomeryOps + Copy + std::fmt::Debug + 'static
{
    type Packed: SumcheckExtField<Scalar = Self> + PackedExtensionPairStride<ScalarExt = Self>;

    fn packed_from_base(value: PBF) -> Self::Packed;

    fn prove_zero_check(
        evals: [Vec<PBF>; 4],
        point: &[Self],
        transcript: &mut Transcript,
    ) -> (Vec<Self>, [Self; 5]);

    /// Profiled variant that fills `timings` with sub-stage breakdown.
    fn prove_zero_check_profiled(
        evals: [Vec<PBF>; 4],
        point: &[Self],
        transcript: &mut Transcript,
        timings: &mut crate::sumcheck_mamabear::ZeroCheckTimings,
    ) -> (Vec<Self>, [Self; 5]);

    /// Parallel variant: uses rayon-parallel fold kernels for the packed
    /// rounds and transition fold. Falls back to serial below the nv
    /// threshold.
    fn prove_zero_check_par(
        evals: [Vec<PBF>; 4],
        point: &[Self],
        transcript: &mut Transcript,
    ) -> (Vec<Self>, [Self; 5]);

    /// Profiled parallel variant.
    fn prove_zero_check_par_profiled(
        evals: [Vec<PBF>; 4],
        point: &[Self],
        transcript: &mut Transcript,
        timings: &mut crate::sumcheck_mamabear::ZeroCheckTimings,
    ) -> (Vec<Self>, [Self; 5]);

    /// Zip-interleave two packed extension elements at the lane level.
    /// `permute2(a.component, b.component, indices)` is applied per component.
    fn zip_packed(a: Self::Packed, b: Self::Packed, indices: [u64; 8]) -> Self::Packed;

    /// Vectorized multilinear evaluation: base-field evals (packed) -> extension-field result.
    ///
    /// Takes `evals_pbf` from `AlignedPoly::as_pbf()` (zero-cost, normal-order layout)
    /// and `point` in Montgomery form. Returns the scalar extension-field result.
    ///
    /// Uses vector-order fold: with normal-order layout, lane-wise fold on consecutive
    /// PBF blocks eliminates variables x3..x_{nv-1} first, then the scalar tail
    /// eliminates x0..x2. The mathematical result is identical to scalar eval_multilinear
    /// because multilinear evaluation is order-independent.
    fn eval_multilinear_base_packed(evals_pbf: &[PBF], point: &[Self]) -> Self;
}


impl MamaBearExtConfig for SEF3 {
    type Packed = PEF3;

    fn packed_from_base(value: PBF) -> Self::Packed {
        value.into()
    }

    fn prove_zero_check(
        evals: [Vec<PBF>; 4],
        point: &[Self],
        transcript: &mut Transcript,
    ) -> (Vec<Self>, [Self; 5]) {
        SumcheckMamaBear::prove_add_mul_ell0_ext3(evals, point, ZERO_CHECK_EXT3_ELL0, transcript)
    }

    fn prove_zero_check_profiled(
        evals: [Vec<PBF>; 4],
        point: &[Self],
        transcript: &mut Transcript,
        timings: &mut crate::sumcheck_mamabear::ZeroCheckTimings,
    ) -> (Vec<Self>, [Self; 5]) {
        SumcheckMamaBear::prove_add_mul_ell0_ext3_profiled(
            evals,
            point,
            ZERO_CHECK_EXT3_ELL0,
            transcript,
            timings,
        )
    }

    fn prove_zero_check_par(
        evals: [Vec<PBF>; 4],
        point: &[Self],
        transcript: &mut Transcript,
    ) -> (Vec<Self>, [Self; 5]) {
        SumcheckMamaBear::prove_add_mul_ell0_ext3_par(
            evals,
            point,
            ZERO_CHECK_EXT3_PAR_ELL0,
            transcript,
        )
    }

    fn prove_zero_check_par_profiled(
        evals: [Vec<PBF>; 4],
        point: &[Self],
        transcript: &mut Transcript,
        timings: &mut crate::sumcheck_mamabear::ZeroCheckTimings,
    ) -> (Vec<Self>, [Self; 5]) {
        SumcheckMamaBear::prove_add_mul_ell0_ext3_par_profiled(
            evals,
            point,
            ZERO_CHECK_EXT3_PAR_ELL0,
            transcript,
            timings,
        )
    }

    #[inline(always)]
    fn zip_packed(a: PEF3, b: PEF3, indices: [u64; 8]) -> PEF3 {
        PEF3 {
            c0: a.c0.permute2(b.c0, indices),
            c1: a.c1.permute2(b.c1, indices),
            c2: a.c2.permute2(b.c2, indices),
        }
    }

    fn eval_multilinear_base_packed(evals_pbf: &[PBF], point: &[Self]) -> Self {
        eval_multilinear_base_packed_ext3(evals_pbf, point)
    }
}


/// Vectorized eval_multilinear for Ext3 using vector-order fold (no permute).
fn eval_multilinear_base_packed_ext3(evals_pbf: &[PBF], point: &[SEF3]) -> SEF3 {
    let nv = point.len();
    let num_pbf = evals_pbf.len();
    debug_assert_eq!(num_pbf, 1 << (nv - 3));

    if num_pbf < 2 {
        let sbf_slice =
            unsafe { std::slice::from_raw_parts(evals_pbf.as_ptr() as *const SBF, num_pbf * 8) };
        return MultiLinearPoly::eval_multilinear(sbf_slice, point);
    }

    const W: usize = 3;

    // --- Round 0: base -> ext3 (fold x_3 with point[3]) ---
    let r0 = point[W];
    let r0_packed = PEF3::new(PBF::from(r0.c0.0), PBF::from(r0.c1.0), PBF::from(r0.c2.0));
    let new_len = num_pbf / 2;
    // Uninit alloc: the round-0 loop below writes scratch[q] for every
    // q in 0..new_len, so we skip the IsZero slow path on this PEF3 newtype.
    let mut scratch: Vec<PEF3> = Vec::with_capacity(new_len);
    unsafe { scratch.set_len(new_len); }

    for q in 0..new_len {
        let v0 = evals_pbf[2 * q];
        let v1 = evals_pbf[2 * q + 1];

        let diff = v1.lazy_add_xp(2).lazy_sub(v0).con_sub_xp(2); // [0, 2P)
        let product = r0_packed.mul_base_elem(diff); // comp < 1.5P
        let mut result = product.add_base_elem(v0); // c0 < 3P, c1/c2 < 1.5P
        result.c0 = result.c0.con_sub_xp(1); // c0 in [0, 2P)
        scratch[q] = result;
    }

    // --- Rounds 1..nv-W-1: ext -> ext (fold x4..x_{nv-1}) ---
    for round in 1..(nv - W) {
        let r = point[W + round];
        let r_packed = PEF3::new(PBF::from(r.c0.0), PBF::from(r.c1.0), PBF::from(r.c2.0));
        let half = scratch.len() / 2;

        for q in 0..half {
            let v0 = scratch[2 * q];
            let v1 = scratch[2 * q + 1];

            let diff = PEF3::new(
                v1.c0.lazy_add_xp(2).lazy_sub(v0.c0).con_sub_xp(2),
                v1.c1.lazy_add_xp(2).lazy_sub(v0.c1).con_sub_xp(2),
                v1.c2.lazy_add_xp(2).lazy_sub(v0.c2).con_sub_xp(2),
            );

            let product = r_packed * diff;
            scratch[q] = product.lazy_add(v0).reduce_fast();
        }
        scratch.truncate(half);
    }

    // --- Scalar tail: fold x0, x1, x2 ---
    debug_assert_eq!(scratch.len(), 1);
    let mut tail = [SEF3::default(); 8];
    scratch[0].unpack_into_slice(&mut tail);

    for t in tail.iter_mut() {
        *t = t.reduce_mod_p();
    }

    let mut len = 8;
    for i in 0..W {
        let r = point[i];
        let half = len / 2;
        for j in 0..half {
            let v0 = tail[2 * j];
            let v1 = tail[2 * j + 1];
            tail[j] = (v0 + (v1 - v0) * r).reduce_mod_p();
        }
        len = half;
    }

    tail[0]
}

pub struct ProverKeyMamaBear<F: MamaBearExtConfig> {
    pub selector: AlignedPoly,
    pub identical: [AlignedPoly; NUM_WIRES],
    pub commitments: DeepFoldMamaBearProver<F>,
    pub permutation: [AlignedPoly; NUM_WIRES],
}

pub struct VerifierKeyMamaBear<F: MamaBearExtConfig> {
    pub commitment: DeepFoldMamaBearVerifier<F>,
    pub _data: PhantomData<F>,
}

pub struct ProverMamaBear<F: MamaBearExtConfig> {
    pub prover_key: ProverKeyMamaBear<F>,
}

fn to_mont_vec(values: &[SBF]) -> Vec<SBF> {
    values.iter().copied().map(SBF::to_montgomery).collect()
}

pub fn pack_base_evals(values: &[SBF]) -> Vec<PBF> {
    debug_assert_eq!(values.len() % 8, 0);
    (0..(values.len() / 8))
        .map(|idx| PBF::load_scalar_slice(&values[(idx << 3)..(idx << 3) + 8]))
        .collect()
}

/// 64-byte aligned polynomial storage for zero-cost SBF<->PBF reinterpretation.
///
/// Backed by `Vec<PBF>` whose allocator guarantees 64-byte alignment (from
/// `__m512i`). This enables `slice::from_raw_parts` casts in both directions:
/// - `as_sbf()`: PBF ptr -> SBF ptr (relaxing alignment, always safe)
/// - `as_pbf()`: direct `&[PBF]` access (already aligned)
#[derive(Clone)]
pub struct AlignedPoly(Vec<PBF>);

impl AlignedPoly {
    /// Create from SBF data (copies into 64-byte aligned storage).
    /// Requires `data.len() % 8 == 0`.
    pub fn from_sbf(data: &[SBF]) -> Self {
        Self(pack_base_evals(data))
    }

    /// Zero-cost view as `&[SBF]`.
    #[inline(always)]
    pub fn as_sbf(&self) -> &[SBF] {
        unsafe { std::slice::from_raw_parts(self.0.as_ptr() as *const SBF, self.0.len() * 8) }
    }

    /// Zero-cost view as `&[PBF]`.
    #[inline(always)]
    pub fn as_pbf(&self) -> &[PBF] {
        &self.0
    }

    /// Zero-cost mutable view as `&mut [PBF]`.
    #[inline(always)]
    pub fn as_pbf_mut(&mut self) -> &mut [PBF] {
        &mut self.0
    }

    /// Convert all elements to Montgomery form in-place using packed SIMD `to_montgomery()`.
    pub fn to_montgomery_in_place(&mut self) {
        for v in self.0.iter_mut() {
            *v = v.to_montgomery();
        }
    }

    #[inline(always)]
    pub fn pbf_len(&self) -> usize {
        self.0.len()
    }

    #[inline(always)]
    pub fn sbf_len(&self) -> usize {
        self.0.len() * 8
    }
}

fn build_identical_tables(nv: usize) -> [Vec<SBF>; NUM_WIRES] {
    [SBF::zero(), SBF::from(ID_SHIFT_1), SBF::from(ID_SHIFT_2)].map(|offset| {
        MultiLinearPoly::new_identical(nv, offset)
            .evals
            .into_iter()
            .map(SBF::to_montgomery)
            .collect()
    })
}

pub fn setup_mamabear<F: MamaBearExtConfig>(
    circuit: &Circuit<F>,
    pp: &DeepFoldMamaBearParam,
) -> (ProverKeyMamaBear<F>, VerifierKeyMamaBear<F>) {
    let commitments = DeepFoldMamaBearProver::<F>::new(
        pp,
        &[
            &circuit.selector,
            &circuit.permutation[0],
            &circuit.permutation[1],
            &circuit.permutation[2],
        ],
    );
    let commitment = commitments.commit();

    let selector = AlignedPoly::from_sbf(&to_mont_vec(&circuit.selector));
    let permutation = circuit
        .permutation
        .clone()
        .map(|poly| AlignedPoly::from_sbf(&to_mont_vec(&poly)));
    let identical =
        build_identical_tables(pp.variable_num).map(|poly| AlignedPoly::from_sbf(&poly));

    (
        ProverKeyMamaBear {
            selector,
            identical,
            commitments: commitments.clone(),
            permutation,
        },
        VerifierKeyMamaBear {
            commitment: DeepFoldMamaBearVerifier::new(pp, commitment, NUM_FIXED),
            _data: PhantomData,
        },
    )
}

pub fn build_productcheck_inputs<F: MamaBearExtConfig>(
    witness: &[AlignedPoly; NUM_WIRES],
    identical: &[AlignedPoly; NUM_WIRES],
    permutation: &[AlignedPoly; NUM_WIRES],
    r0_mont: F,
    r1_mont: F,
) -> [[Vec<F::Packed>; NUM_WIRES]; 2] {
    let r0 = F::Packed::from_scalar(r0_mont);
    let r1 = F::Packed::from_scalar(r1_mont);
    std::array::from_fn(|tree| {
        std::array::from_fn(|wire| {
            let rhs_pbf = if tree == 0 {
                identical[wire].as_pbf()
            } else {
                permutation[wire].as_pbf()
            };
            let witness_pbf = witness[wire].as_pbf();
            let mut out = Vec::with_capacity(witness_pbf.len());
            for idx in 0..witness_pbf.len() {
                let value = r1
                    .mul_base_elem(rhs_pbf[idx])
                    .lazy_add(r0)
                    .reduce_fast()
                    .add_base_elem(witness_pbf[idx])
                    .reduce_fast();
                out.push(value);
            }
            out
        })
    })
}

/// SIMD-accelerated pre-combine: fuse the 7 base-field witness/fixed tables
/// into the two extension-field trees that the subsequent sumcheck folds.
///
///   tree0 = selector + rho*w0 + rho^2*w1 + rho^3*w2
///   tree1 = perm0 + rho*perm1 + rho^2*perm2 + rho^3*w0 + rho^4*w1 + rho^5*w2
///
/// Each Vec<SBF> is reinterpreted as a PBF slice (zero-copy: both share the
/// R=2^52 Montgomery representation). Returns `Vec<F::Packed>` in normal order
/// (no AoS unpack) — each packed element holds 8 consecutive extension scalars.
pub(crate) fn precombine_rho_trees_packed<F: MamaBearExtConfig>(
    evals: [&AlignedPoly; NUM_ALL_TABLES],
    rho_mont: F,
    rho2: F,
    rho3: F,
    rho4: F,
    rho5: F,
) -> [Vec<F::Packed>; 2] {
    let blocks = evals[0].pbf_len();

    let rho_p = <F::Packed as SumcheckExtField>::from_scalar(rho_mont);
    let rho2_p = <F::Packed as SumcheckExtField>::from_scalar(rho2);
    let rho3_p = <F::Packed as SumcheckExtField>::from_scalar(rho3);
    let rho4_p = <F::Packed as SumcheckExtField>::from_scalar(rho4);
    let rho5_p = <F::Packed as SumcheckExtField>::from_scalar(rho5);

    // Direct PBF access — AlignedPoly guarantees 64-byte alignment.
    let p0 = evals[0].as_pbf();
    let p1 = evals[1].as_pbf();
    let p2 = evals[2].as_pbf();
    let p3 = evals[3].as_pbf();
    let p4 = evals[4].as_pbf();
    let p5 = evals[5].as_pbf();
    let p6 = evals[6].as_pbf();

    let mut tree0: Vec<F::Packed> = Vec::with_capacity(blocks);
    let mut tree1: Vec<F::Packed> = Vec::with_capacity(blocks);

    for blk in 0..blocks {
        // tree0 = b0 + rho*b4 + rho^2*b5 + rho^3*b6
        // Each `mul_base_elem` gives PEF components in [0, 1.5P). Summing
        // four of them (plus b0 injected via `packed_from_base`) stays well
        // inside Mul's [0, 4P) input range; a final `.reduce()` brings each
        // lane back to [0, P).
        let t0 = F::packed_from_base(p0[blk])
            .lazy_add(rho_p.mul_base_elem(p4[blk]))
            .lazy_add(rho2_p.mul_base_elem(p5[blk]))
            .lazy_add(rho3_p.mul_base_elem(p6[blk]))
            .reduce();
        // tree1 = b1 + rho*b2 + rho^2*b3 + rho^3*b4 + rho^4*b5 + rho^5*b6
        let t1 = F::packed_from_base(p1[blk])
            .lazy_add(rho_p.mul_base_elem(p2[blk]))
            .lazy_add(rho2_p.mul_base_elem(p3[blk]))
            .lazy_add(rho3_p.mul_base_elem(p4[blk]))
            .lazy_add(rho4_p.mul_base_elem(p5[blk]))
            .lazy_add(rho5_p.mul_base_elem(p6[blk]))
            .reduce();

        tree0.push(t0);
        tree1.push(t1);
    }

    [tree0, tree1]
}

/// Build eq table via tensor expansion, vectorized with SIMD.
///
/// first min(3, nv) iterations in scalar (up to 8 values).
/// pack into Vec<F::Packed>, continue with packed muls for the
///          remaining nv-3 iterations — processes 8 lanes per mul, ~8x faster.
///          After each packed mul, the (1-bit) and bit results are interleaved
///          so the output maintains sequential scalar order.
///
/// Output: Vec<F::Packed> in normal (sequential) order, each element holds 8
///         consecutive eq(point, i) values in Montgomery form, reduced to [0, P).
pub(crate) fn build_eq_table_packed<F: MamaBearExtConfig>(point: &[F], one: F) -> Vec<F::Packed> {
    let nv = point.len();
    let scalar_iters = nv.min(3);

    // scalar tensor expansion for first `scalar_iters` bits.
    let mut eq_scalar = vec![one];
    for i in 0..scalar_iters {
        let bit = point[nv - 1 - i];
        let one_minus_bit = (one.lazy_add_xp(2).lazy_sub(bit).con_sub_xp(2)).reduce();
        eq_scalar = eq_scalar
            .iter()
            .flat_map(|&prod| [(prod * one_minus_bit).reduce(), (prod * bit).reduce()])
            .collect();
    }

    if nv <= 3 {
        return eq_scalar
            .chunks(8)
            .map(|chunk| {
                if chunk.len() == 8 {
                    <F::Packed as PackedExtensionField>::pack_slice_exact(chunk)
                } else {
                    <F::Packed as PackedExtensionField>::pack_partial(chunk)
                }
            })
            .collect();
    }

    // pack 8 scalars → 1 packed element, then continue with packed muls.
    // After each iteration, we interleave the (1-bit) and bit results so that
    // the output remains in sequential scalar order (matching tree table layout).
    //
    // For each packed element p = {v0..v7}:
    //   a = p * (1-bit) = {v0*(1-b), v1*(1-b), ..., v7*(1-b)}
    //   b = p * bit     = {v0*b,     v1*b,     ..., v7*b}
    //
    // Sequential scalar order requires:
    //   out[2j]   = {v0*(1-b), v0*b, v1*(1-b), v1*b, v2*(1-b), v2*b, v3*(1-b), v3*b}
    //   out[2j+1] = {v4*(1-b), v4*b, v5*(1-b), v5*b, v6*(1-b), v6*b, v7*(1-b), v7*b}
    //
    // This is a zip/interleave of a and b at the lane level, achieved via
    // permutex2var on each base-field component.
    debug_assert_eq!(eq_scalar.len(), 8);
    let mut eq_packed = vec![<F::Packed as PackedExtensionField>::pack_slice_exact(
        &eq_scalar,
    )];

    // Indices for zip-low (first 4 pairs) and zip-high (last 4 pairs).
    const ZIP_LO: [u64; 8] = [0, 8, 1, 9, 2, 10, 3, 11];
    const ZIP_HI: [u64; 8] = [4, 12, 5, 13, 6, 14, 7, 15];

    for i in scalar_iters..nv {
        let bit = point[nv - 1 - i];
        let bit_p = <F::Packed as SumcheckExtField>::from_scalar(bit);
        let one_minus_bit = (one.lazy_add_xp(2).lazy_sub(bit).con_sub_xp(2)).reduce();
        let one_minus_bit_p = <F::Packed as SumcheckExtField>::from_scalar(one_minus_bit);

        let old_len = eq_packed.len();
        let mut new_eq = Vec::with_capacity(old_len * 2);
        for j in 0..old_len {
            let prod = eq_packed[j]; // [0, P)
            let a = (prod * one_minus_bit_p).reduce(); // [0, P)
            let b = (prod * bit_p).reduce(); // [0, P)
                                             // Interleave a and b to maintain sequential order.
            new_eq.push(F::zip_packed(a, b, ZIP_LO));
            new_eq.push(F::zip_packed(a, b, ZIP_HI));
        }
        eq_packed = new_eq;
    }

    eq_packed
}

pub fn prove_final_reduce_sumcheck<F: MamaBearExtConfig>(
    evals: [&AlignedPoly; NUM_ALL_TABLES],
    sumcheck_point: &[F],
    prod_point: &[F],
    transcript: &mut Transcript,
) -> Vec<F> {
    let num_vars = sumcheck_point.len();
    assert_eq!(num_vars, prod_point.len());

    // ── Draw rho and materialize rho^1..rho^5. ──
    let rho_raw: F = transcript.challenge_f();
    let rho_mont = rho_raw.to_mont();
    let rho2 = (rho_mont * rho_mont).reduce();
    let rho3 = (rho2 * rho_mont).reduce();
    let rho4 = (rho3 * rho_mont).reduce();
    let rho5 = (rho4 * rho_mont).reduce();

    // ── SIMD pre-combine: 7 base tables -> 2 ext trees as Vec<F::Packed>
    //    in normal order. No AoS unpack — stays packed throughout. ──
    let [mut tree0, mut tree1] =
        precombine_rho_trees_packed::<F>(evals, rho_mont, rho2, rho3, rho4, rho5);
    // (`evals` is an array of refs now — no clones.)

    // ── Build eq tables via tensor expansion, vectorized with SIMD. ──
    // first 3 iterations in scalar (produces 8 values).
    // pack into Vec<F::Packed>, continue remaining iterations
    //          using packed muls — 8x fewer operations than scalar.
    //
    // Point conventions:
    // - `sumcheck_point`: challenges from `prove_zero_check` — each challenge
    //   was pushed via `transcript.challenge_f::<SEF>().to_montgomery()`, so
    //   values are ALREADY in Montgomery form. Do NOT re-apply `to_mont`.
    // - `prod_point`: challenges from `ProdEqCheckMamaBearPerWire::prove` —
    //   also pushed in Montgomery form (same convention).
    //
    // Both points arrive in SIMD-round order (output of normal-layout SIMD
    // sumchecks). `build_eq_table_packed` produces a table whose index bit k
    // corresponds to natural variable x_k, so it expects `point[k]` to be the
    // value plugged into x_k. Permute SIMD-order → natural order so the eq
    // table represents `eq(natural_reduced_pt, x)`, which makes the initial-
    // sum semantics match the claim-based Horner combination on the verifier
    // side.
    let point0_mont = simd_to_natural_point(sumcheck_point);
    let point1_mont = simd_to_natural_point(prod_point);
    let one = F::one().to_mont();
    let mut eq0 = build_eq_table_packed::<F>(&point0_mont, one);
    let mut eq1 = build_eq_table_packed::<F>(&point1_mont, one);

    // ── Main sumcheck loop with fused fold + eval. ──
    //
    // Data layout: each table is Vec<F::Packed> in normal order.
    // Normal-order folding: table[2g] (left) and table[2g+1] (right) form a pair.
    // No pack_strided_pair_slice shuffles needed.
    //
    // Three- structure:
    //   Round 0: eval-only (no prior fold needed).
    //   Rounds 1..packed_rounds-1: fused fold(prev_alpha) + eval in ONE pass.
    //     Reads 4 consecutive packed elements per group (e0,e1,e2,e3),
    //     folds (e0,e1)→v0, (e2,e3)→v1, stores folded, evals from (v0,v1).
    //   Scalar tail: last ≤3 rounds when table fits in one packed element.
    let mut challenges = Vec::with_capacity(num_vars);
    // Packed rounds need ≥ 2 packed elements. Table starts at 2^(nv-3) packed.
    // After each packed round, table halves. Last packed round leaves 1 element.
    let packed_rounds = if num_vars >= 4 { num_vars - 3 } else { 0 };

    // ── Round 0: eval-only from initial packed tables. ──
    if packed_rounds > 0 {
        let half_packed = tree0.len() >> 1;
        let mut acc_p: [[F::Packed; 3]; 2] = [[F::Packed::zero(); 3], [F::Packed::zero(); 3]];

        for g in 0..half_packed {
            let t0_l = tree0[2 * g]; // [0, P) from precombine reduce()
            let t0_r = tree0[2 * g + 1];
            let t1_l = tree1[2 * g];
            let t1_r = tree1[2 * g + 1];
            let e0_l = eq0[2 * g];
            let e0_r = eq0[2 * g + 1];
            let e1_l = eq1[2 * g];
            let e1_r = eq1[2 * g + 1];

            // Extrapolate to x=2: x2 = r + (r - l).
            // diff ∈ [0, 2P), x2 ∈ [0, 2P) — safe for mul input < 4P.
            let t0_diff = t0_r.lazy_add_xp(2).lazy_sub(t0_l).con_sub_xp(2);
            let t0_x2 = t0_r.lazy_add(t0_diff).con_sub_xp(2);
            let t1_diff = t1_r.lazy_add_xp(2).lazy_sub(t1_l).con_sub_xp(2);
            let t1_x2 = t1_r.lazy_add(t1_diff).con_sub_xp(2);
            let e0_diff = e0_r.lazy_add_xp(2).lazy_sub(e0_l).con_sub_xp(2);
            let e0_x2 = e0_r.lazy_add(e0_diff).con_sub_xp(2);
            let e1_diff = e1_r.lazy_add_xp(2).lazy_sub(e1_l).con_sub_xp(2);
            let e1_x2 = e1_r.lazy_add(e1_diff).con_sub_xp(2);

            acc_p[0][0] = acc_p[0][0].lazy_add(t0_l * e0_l);
            acc_p[0][1] = acc_p[0][1].lazy_add(t0_r * e0_r);
            acc_p[0][2] = acc_p[0][2].lazy_add(t0_x2 * e0_x2);
            acc_p[1][0] = acc_p[1][0].lazy_add(t1_l * e1_l);
            acc_p[1][1] = acc_p[1][1].lazy_add(t1_r * e1_r);
            acc_p[1][2] = acc_p[1][2].lazy_add(t1_x2 * e1_x2);

            if (g & 7) == 7 {
                for tree in 0..2 {
                    for h in 0..3 {
                        acc_p[tree][h] = acc_p[tree][h].reduce_fast();
                    }
                }
            }
        }

        let mut acc = [[F::zero(); 3]; 2];
        for tree in 0..2 {
            for h in 0..3 {
                acc[tree][h] = <F::Packed as SumcheckExtField>::sum_lanes_to_mont(
                    acc_p[tree][h].reduce_fast(),
                );
            }
        }
        for tree in 0..2 {
            for value in acc[tree] {
                transcript.append_f(value.reduce().from_mont());
            }
        }
        let alpha: F = transcript.challenge_f();
        challenges.push(alpha.to_mont());
    }

    // ── Rounds 1..packed_rounds-1: fused fold(prev_alpha) + eval in ONE pass. ──
    // Reads groups of 4 consecutive packed elements (e0,e1,e2,e3).
    // Folds: v0 = fold(e0,e1,alpha_prev), v1 = fold(e2,e3,alpha_prev).
    // Stores folded (v0,v1) and computes round poly from (v0,v1) simultaneously.
    // This halves the table AND produces the round polynomial in a single scan.
    for _round in 1..packed_rounds {
        let alpha_prev = *challenges.last().unwrap();
        let alpha_p = <F::Packed as SumcheckExtField>::from_scalar(alpha_prev);
        // Before fold: table has 2^(nv-3-round+1) packed elements.
        // After fold: half that. Groups of 4 → 2 folded values.
        let current_len = tree0.len(); // must be multiple of 2
        let next_len = current_len >> 1;
        let packed_groups = next_len >> 1; // number of (v0,v1) output pairs

        let mut acc_p: [[F::Packed; 3]; 2] = [[F::Packed::zero(); 3], [F::Packed::zero(); 3]];

        for g in 0..packed_groups {
            let src = g << 2;
            let dst = g << 1;

            // ── Fold 4 elements → 2 for each table, then eval from (v0,v1). ──
            // Macro-like inline for each (tree, eq) pair.
            macro_rules! fold_and_eval {
                ($table:expr, $eq:expr, $tree_idx:expr) => {{
                    let te0 = $table[src]; // [0, 2P) from prior fold
                    let te1 = $table[src + 1];
                    let te2 = $table[src + 2];
                    let te3 = $table[src + 3];
                    let diff0 = te1.lazy_add_xp(2).lazy_sub(te0); // [0, 4P)
                    let diff1 = te3.lazy_add_xp(2).lazy_sub(te2); // [0, 4P)
                    let tv0 = (alpha_p * diff0).lazy_add(te0).con_sub_xp(2); // [0, 2P)
                    let tv1 = (alpha_p * diff1).lazy_add(te2).con_sub_xp(2); // [0, 2P)
                    $table[dst] = tv0;
                    $table[dst + 1] = tv1;

                    let qe0 = $eq[src];
                    let qe1 = $eq[src + 1];
                    let qe2 = $eq[src + 2];
                    let qe3 = $eq[src + 3];
                    let ed0 = qe1.lazy_add_xp(2).lazy_sub(qe0);
                    let ed1 = qe3.lazy_add_xp(2).lazy_sub(qe2);
                    let ev0 = (alpha_p * ed0).lazy_add(qe0).con_sub_xp(2); // [0, 2P)
                    let ev1 = (alpha_p * ed1).lazy_add(qe2).con_sub_xp(2); // [0, 2P)
                    $eq[dst] = ev0;
                    $eq[dst + 1] = ev1;

                    // Eval round poly from folded (tv0,tv1) × (ev0,ev1).
                    let t_diff = tv1.lazy_add_xp(2).lazy_sub(tv0).con_sub_xp(2); // [0, 2P)
                    let t_x2 = tv1.lazy_add(t_diff).con_sub_xp(2); // [0, 2P)
                    let e_diff = ev1.lazy_add_xp(2).lazy_sub(ev0).con_sub_xp(2);
                    let e_x2 = ev1.lazy_add(e_diff).con_sub_xp(2);

                    acc_p[$tree_idx][0] = acc_p[$tree_idx][0].lazy_add(tv0 * ev0);
                    acc_p[$tree_idx][1] = acc_p[$tree_idx][1].lazy_add(tv1 * ev1);
                    acc_p[$tree_idx][2] = acc_p[$tree_idx][2].lazy_add(t_x2 * e_x2);
                }};
            }

            fold_and_eval!(tree0, eq0, 0);
            fold_and_eval!(tree1, eq1, 1);

            if (g & 7) == 7 {
                for tree in 0..2 {
                    for h in 0..3 {
                        acc_p[tree][h] = acc_p[tree][h].reduce_fast();
                    }
                }
            }
        }

        // Handle residual pair if next_len is odd (one leftover pair after groups of 4).
        if (next_len & 1) == 1 {
            let residual_src = packed_groups << 2;
            let residual_dst = packed_groups << 1;
            // Fold the last 2 elements → 1.
            macro_rules! fold_residual {
                ($table:expr, $eq:expr, $tree_idx:expr) => {{
                    let te0 = $table[residual_src];
                    let te1 = $table[residual_src + 1];
                    let diff = te1.lazy_add_xp(2).lazy_sub(te0);
                    let tv = (alpha_p * diff).lazy_add(te0).con_sub_xp(2);
                    $table[residual_dst] = tv;

                    let qe0 = $eq[residual_src];
                    let qe1 = $eq[residual_src + 1];
                    let ed = qe1.lazy_add_xp(2).lazy_sub(qe0);
                    let ev = (alpha_p * ed).lazy_add(qe0).con_sub_xp(2);
                    $eq[residual_dst] = ev;
                    // Single element — contributes to x=0 only (no pair for x=1,2).
                    // This residual will be paired in the next round.
                }};
            }
            fold_residual!(tree0, eq0, 0);
            fold_residual!(tree1, eq1, 1);
        }

        tree0.truncate(next_len);
        tree1.truncate(next_len);
        eq0.truncate(next_len);
        eq1.truncate(next_len);

        // Sum lanes and send round polynomial.
        let mut acc = [[F::zero(); 3]; 2];
        for tree in 0..2 {
            for h in 0..3 {
                acc[tree][h] = <F::Packed as SumcheckExtField>::sum_lanes_to_mont(
                    acc_p[tree][h].reduce_fast(),
                );
            }
        }

        // Add residual single-packed-element contribution if next_len was odd.
        // The residual is a lone packed element — it only contributes to x=0.
        // Actually for the round polynomial, the residual packed element forms
        // only half a pair; it will be properly handled in the next round.
        // For the current round, packed_groups covers all complete (v0,v1) pairs.
        // If next_len is odd, the lone element at index next_len-1 means the
        // un-folded table still had that element — but we already folded it.
        // Wait — we need to account for the folded residual in the round poly.
        // Since it has no partner, it contributes as a "left-only" entry:
        // p(0) += tv * ev, p(1) is not affected (no right), p(2) extrapolation too.
        // Actually this shouldn't happen for our use case (nv >= 4, power-of-2 sizes).

        for tree in 0..2 {
            for value in acc[tree] {
                transcript.append_f(value.reduce().from_mont());
            }
        }
        let alpha: F = transcript.challenge_f();
        challenges.push(alpha.to_mont());
    }

    // ── Final fold before scalar tail: apply last packed-round alpha. ──
    if packed_rounds > 0 && tree0.len() >= 2 {
        let alpha_last = *challenges.last().unwrap();
        let alpha_p = <F::Packed as SumcheckExtField>::from_scalar(alpha_last);
        let half = tree0.len() >> 1;
        for g in 0..half {
            let fold = |left: F::Packed, right: F::Packed| -> F::Packed {
                let diff = right.lazy_add_xp(2).lazy_sub(left); // [0, 4P)
                (alpha_p * diff).lazy_add(left).con_sub_xp(2) // [0, 2P)
            };
            tree0[g] = fold(tree0[2 * g], tree0[2 * g + 1]);
            tree1[g] = fold(tree1[2 * g], tree1[2 * g + 1]);
            eq0[g] = fold(eq0[2 * g], eq0[2 * g + 1]);
            eq1[g] = fold(eq1[2 * g], eq1[2 * g + 1]);
        }
        tree0.truncate(half);
        tree1.truncate(half);
        eq0.truncate(half);
        eq1.truncate(half);
    }

    // ── Scalar tail: unpack remaining packed elements, finish last rounds. ──
    let mut s_tree0 = unpack_packed_vec::<F>(&tree0);
    let mut s_tree1 = unpack_packed_vec::<F>(&tree1);
    let mut s_eq0 = unpack_packed_vec::<F>(&eq0);
    let mut s_eq1 = unpack_packed_vec::<F>(&eq1);

    for _round in packed_rounds..num_vars {
        let half = s_tree0.len() >> 1;
        let mut acc = [[F::zero(); 3]; 2];

        for i in 0..half {
            let li = i << 1;
            let ri = li + 1;

            let t0_l = s_tree0[li];
            let t0_r = s_tree0[ri];
            let t0_diff = t0_r.lazy_add_xp(2).lazy_sub(t0_l).con_sub_xp(2);
            let t0_x2 = t0_r.lazy_add(t0_diff).con_sub_xp(2);
            let t1_l = s_tree1[li];
            let t1_r = s_tree1[ri];
            let t1_diff = t1_r.lazy_add_xp(2).lazy_sub(t1_l).con_sub_xp(2);
            let t1_x2 = t1_r.lazy_add(t1_diff).con_sub_xp(2);
            let e0_l = s_eq0[li];
            let e0_r = s_eq0[ri];
            let e0_diff = e0_r.lazy_add_xp(2).lazy_sub(e0_l).con_sub_xp(2);
            let e0_x2 = e0_r.lazy_add(e0_diff).con_sub_xp(2);
            let e1_l = s_eq1[li];
            let e1_r = s_eq1[ri];
            let e1_diff = e1_r.lazy_add_xp(2).lazy_sub(e1_l).con_sub_xp(2);
            let e1_x2 = e1_r.lazy_add(e1_diff).con_sub_xp(2);

            acc[0][0] = acc[0][0].lazy_add((t0_l * e0_l).reduce()).reduce();
            acc[0][1] = acc[0][1].lazy_add((t0_r * e0_r).reduce()).reduce();
            acc[0][2] = acc[0][2].lazy_add((t0_x2 * e0_x2).reduce()).reduce();
            acc[1][0] = acc[1][0].lazy_add((t1_l * e1_l).reduce()).reduce();
            acc[1][1] = acc[1][1].lazy_add((t1_r * e1_r).reduce()).reduce();
            acc[1][2] = acc[1][2].lazy_add((t1_x2 * e1_x2).reduce()).reduce();
        }

        for tree in 0..2 {
            for value in acc[tree] {
                transcript.append_f(value.reduce().from_mont());
            }
        }
        let alpha: F = transcript.challenge_f();
        let alpha_mont = alpha.to_mont();
        challenges.push(alpha_mont);

        // Scalar fold in-place.
        for j in 0..half {
            let l = s_tree0[j << 1];
            let r = s_tree0[(j << 1) + 1];
            let diff = r.lazy_add_xp(2).lazy_sub(l).con_sub_xp(2);
            s_tree0[j] = (alpha_mont * diff).lazy_add(l).reduce();
        }
        s_tree0.truncate(half);
        for j in 0..half {
            let l = s_tree1[j << 1];
            let r = s_tree1[(j << 1) + 1];
            let diff = r.lazy_add_xp(2).lazy_sub(l).con_sub_xp(2);
            s_tree1[j] = (alpha_mont * diff).lazy_add(l).reduce();
        }
        s_tree1.truncate(half);
        for j in 0..half {
            let l = s_eq0[j << 1];
            let r = s_eq0[(j << 1) + 1];
            let diff = r.lazy_add_xp(2).lazy_sub(l).con_sub_xp(2);
            s_eq0[j] = (alpha_mont * diff).lazy_add(l).reduce();
        }
        s_eq0.truncate(half);
        for j in 0..half {
            let l = s_eq1[j << 1];
            let r = s_eq1[(j << 1) + 1];
            let diff = r.lazy_add_xp(2).lazy_sub(l).con_sub_xp(2);
            s_eq1[j] = (alpha_mont * diff).lazy_add(l).reduce();
        }
        s_eq1.truncate(half);
    }

    challenges
}

/// Unpack a Vec<F::Packed> (normal order) into a Vec<F> of scalar extension values.
#[inline]
pub(crate) fn unpack_packed_vec<F: MamaBearExtConfig>(packed: &[F::Packed]) -> Vec<F> {
    let mut out = Vec::with_capacity(packed.len() * 8);
    for &p in packed {
        out.extend_from_slice(&<F::Packed as PackedExtensionField>::unpack_to_array(p));
    }
    out
}

impl<F: MamaBearExtConfig> ProverMamaBear<F> {
    pub fn prove(
        &self,
        pp: &DeepFoldMamaBearParam,
        nv: usize,
        mut witness: [AlignedPoly; NUM_WIRES],
    ) -> Proof {
        let mut transcript = Transcript::new();
        // Commit uses raw (non-mont) SBF data via zero-cost as_sbf() view
        let witness_pc = DeepFoldMamaBearProver::<F>::new(
            pp,
            &[
                witness[0].as_sbf(),
                witness[1].as_sbf(),
                witness[2].as_sbf(),
            ],
        );

        let commit = witness_pc.commit();
        let mut buffer = vec![0u8; MerkleRoot::size(nv, NUM_WIRES)];
        commit.serialize_into(&mut buffer);
        transcript.append_u8_slice(&buffer, buffer.len());

        // Convert to Montgomery domain in-place using packed SIMD
        for poly in witness.iter_mut() {
            poly.to_montgomery_in_place();
        }

        // prove_zero_check folds in-place — clone PBF data for owned copies
        let sumcheck_r = (0..nv)
            .map(|_| transcript.challenge_f::<F>())
            .collect::<Vec<_>>();
        let (sumcheck_point, zero_evals) = F::prove_zero_check(
            [
                self.prover_key.selector.as_pbf().to_vec(),
                witness[0].as_pbf().to_vec(),
                witness[1].as_pbf().to_vec(),
                witness[2].as_pbf().to_vec(),
            ],
            &sumcheck_r,
            &mut transcript,
        );
        for value in zero_evals.into_iter().take(4) {
            transcript.append_f(value);
        }

        let prod_r0: F = transcript.challenge_f();
        let prod_r1: F = transcript.challenge_f();
        // build_productcheck_inputs is read-only — zero-cost PBF views via AlignedPoly
        let prod_point = ProdEqCheckMamaBearPerWire::prove::<F::Packed>(
            build_productcheck_inputs::<F>(
                &witness,
                &self.prover_key.identical,
                &self.prover_key.permutation,
                prod_r0.to_mont(),
                prod_r1.to_mont(),
            ),
            &mut transcript,
        );

        // Mid-stage openings: the per-wire ProductCheck sumcheck output
        // `prod_point` is in SIMD-round order — under normal memory layout,
        // the SIMD sumcheck eliminates x_3, x_4, ..., x_{nv-1} in the packed
        // rounds and x_0, x_1, x_2 in the scalar tail (see
        // `simd_to_natural_point` above). We need witness_g / perm_g
        // evaluated at the NATURAL reduced point so the verifier's ProductCheck reconstruction check
        // (which computes id_g / sigma_g at the natural point) is
        // semantically consistent.
        let prod_point_natural = simd_to_natural_point(&prod_point[..nv]);
        for poly in witness.iter() {
            transcript.append_f(
                F::eval_multilinear_base_packed(poly.as_pbf(), &prod_point_natural).from_mont(),
            );
        }
        for poly in self.prover_key.permutation.iter() {
            transcript.append_f(
                F::eval_multilinear_base_packed(poly.as_pbf(), &prod_point_natural).from_mont(),
            );
        }

        // point_mont is already in Montgomery domain.
        // prove_final_reduce_sumcheck takes array-of-refs — no clones.
        let evals: [&AlignedPoly; NUM_ALL_TABLES] = [
            &self.prover_key.selector,
            &self.prover_key.permutation[0],
            &self.prover_key.permutation[1],
            &self.prover_key.permutation[2],
            &witness[0],
            &witness[1],
            &witness[2],
        ];
        let point_mont = prove_final_reduce_sumcheck::<F>(
            evals,
            &sumcheck_point,
            &prod_point[..nv],
            &mut transcript,
        );

        // Final openings + PCS open: `point_mont` is the final-reduce sumcheck
        // output in SIMD-round order. Permute to natural order so that
        // `eval_multilinear_base_packed` (which interprets point[k] as x_k's
        // value) returns the MLE at the NATURAL reduced point of the
        // sumcheck. The PCS must be called with the same natural point so its
        // cryptographic check aligns with these evaluations.
        let point_mont_natural = simd_to_natural_point(&point_mont);

        transcript.append_f(
            F::eval_multilinear_base_packed(self.prover_key.selector.as_pbf(), &point_mont_natural)
                .from_mont(),
        );
        for poly in self.prover_key.permutation.iter() {
            transcript.append_f(
                F::eval_multilinear_base_packed(poly.as_pbf(), &point_mont_natural).from_mont(),
            );
        }
        for poly in witness.iter() {
            transcript.append_f(
                F::eval_multilinear_base_packed(poly.as_pbf(), &point_mont_natural).from_mont(),
            );
        }

        DeepFoldMamaBearProver::open(
            pp,
            &[&self.prover_key.commitments, &witness_pc],
            point_mont_natural,
            &mut transcript,
        );

        transcript.proof
    }
}
