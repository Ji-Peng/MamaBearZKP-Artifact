///! Parallel prover orchestrator.
///!
///! Uses parallel versions of the hot paths:
///! - `new_par`: parallel DeepFold commit (FFT + leaf hash)
///! - `ProdEqCheckMamaBearPerWirePar::prove`: parallel ProductCheck

use arithmetic::field::mamabear::*;
use arithmetic::field::Field;
use poly_commit::{
    deepfold::MerkleRoot,
    deepfold_mamabear::{DeepFoldMamaBearParam, DeepFoldMamaBearProver},
    deepfold_mamabear_par,
    CommitmentSerde,
};
use rayon::prelude::*;
use util::fiat_shamir::Proof;
use util::fiat_shamir::Transcript;

use crate::prover_mamabear::{
    build_productcheck_inputs, precombine_rho_trees_packed, prove_final_reduce_sumcheck,
    unpack_packed_vec, AlignedPoly, MamaBearExtConfig, ProverMamaBear,
};
use crate::prodcheck_mamabear_perwire_par::ProdEqCheckMamaBearPerWirePar;
use crate::sumcheck_mamabear::SumcheckExtField;

// final_rsc parallelization thresholds and chunk sizes.
// `pub(crate)` so the final-reduce par twins (B.1 5+6 tree in
// `prover_full_par.rs`, custom 8+6+4 tree in `prover_custom_par.rs`) reuse the
// SAME thresholds / chunk boundaries as the 2-tree template here, keeping their
// fallback + deterministic-chunk behaviour bit-for-bit consistent.
// Threshold retuning (was 1<<10): the final-reduce sumcheck grew a tree
// (b1 2->3, custom 3->4). The `final_reduce_b1_crossover` sweep (i7-11700K,
// 8c, 3-tree b1) shows par is slower than serial below 8192 PBF blocks (0.84x
// at 1024, 0.96x at 2048, 0.85x at 4096) and wins from 8192 (nv=16, 1.16x;
// nv=18 1.99x; nv=20 2.08x). We set 1<<13 = 8192 so the par final-reduce is
// never slower than serial for direct callers. This shared constant also gates
// the lighter 2-tree add/mul template and the heavier 4-tree custom path;
// raising it from 1024 is strictly safer for all three. Production is
// unaffected: the orchestrators only run the par final-reduce at
// nv >= CUSTOM_FULL_PAR_MIN_NV (18) where blocks >= 32768 >> 8192.
pub(crate) const PAR_FINAL_RSC_MIN_BLOCKS: usize = 1 << 13;
pub(crate) const PAR_FINAL_RSC_PRECOMBINE_CHUNK: usize = 4096;
const PAR_FINAL_RSC_EQ_MIN_OLDLEN: usize = 256;
const PAR_FINAL_RSC_EQ_CHUNK: usize = 1024; // in terms of old_len src elements
pub(crate) const PAR_FINAL_RSC_ROUND_GROUP_CHUNK: usize = 1024; // groups (each = 2 dst / 4 src)

/// Minimum num_vars below which mid_clm / final_clm batch helpers fall back to
/// serial — each eval_multilinear_base_packed is under a millisecond and
/// rayon::scope spawn overhead of 6-7 tasks (~40-70 us total) eats the gain.
const PAR_CLAIMS_MIN_NV: usize = 12;

/// Batched parallel multilinear evaluation: one rayon task per polynomial.
///
/// All polys are evaluated independently at `point`; output order matches
/// input order so the caller can append results to a transcript
/// deterministically. Used by `mid_clm` (N=6) and `final_clm` (N=7).
///
/// Below `PAR_CLAIMS_MIN_NV`, falls back to a serial loop to avoid
/// `rayon::scope` spawn overhead on work that's already under ~1 ms per eval.
pub fn eval_multilinear_base_packed_batch_par<F: MamaBearExtConfig>(
    polys: &[&[PBF]],
    point: &[F],
) -> Vec<F>
where
    F: Send + Sync,
    F::Packed: Send + Sync,
{
    if point.len() <= PAR_CLAIMS_MIN_NV {
        return polys
            .iter()
            .map(|p| F::eval_multilinear_base_packed(p, point))
            .collect();
    }
    polys
        .par_iter()
        .map(|p| F::eval_multilinear_base_packed(p, point))
        .collect()
}

const NUM_WIRES: usize = 3;
const NUM_ALL_TABLES: usize = 7;

type PBF = PackedMamaBearAVX512;

/// Parallel counterpart to `build_productcheck_inputs`.
///
/// Builds the 6 per-wire ProductCheck input vectors `[[Vec<F::Packed>; 3]; 2]`
/// with one rayon task per `(tree, wire)` pair (6 tasks total). Each task is
/// an embarrassingly-parallel compute loop of `5` field ops per packed element
/// over `2^(nv-3)` packed elements, with no cross-task dependencies.
///
/// At nv=23 this substage is ~172 ms serial; 6-way parallel cuts it to ~30 ms
/// and unblocks the remaining `prod_chk` speedup ceiling.
///
/// Output buffers use the `Vec::with_capacity + ptr::write + set_len` fast-path
/// idiom: allocating `u64` obtains zero-page-backed memory, and
/// Initialization") to avoid the `IsZero` generic-fallback slow path that would
/// otherwise single-thread a ~128 MB memset per task.
pub fn build_productcheck_inputs_par<F>(
    witness: &[AlignedPoly; NUM_WIRES],
    identical: &[AlignedPoly; NUM_WIRES],
    permutation: &[AlignedPoly; NUM_WIRES],
    r0_mont: F,
    r1_mont: F,
) -> [[Vec<F::Packed>; NUM_WIRES]; 2]
where
    F: MamaBearExtConfig + Send + Sync,
    F::Packed: Send + Sync,
{
    let n = witness[0].as_pbf().len();

    // Fork/join overhead dominates for tiny inputs — fall back to serial.
    if n < 1024 {
        return build_productcheck_inputs::<F>(
            witness,
            identical,
            permutation,
            r0_mont,
            r1_mont,
        );
    }

    let r0 = F::Packed::from_scalar(r0_mont);
    let r1 = F::Packed::from_scalar(r1_mont);

    let w0 = witness[0].as_pbf();
    let w1 = witness[1].as_pbf();
    let w2 = witness[2].as_pbf();
    let i0 = identical[0].as_pbf();
    let i1 = identical[1].as_pbf();
    let i2 = identical[2].as_pbf();
    let p0 = permutation[0].as_pbf();
    let p1 = permutation[1].as_pbf();
    let p2 = permutation[2].as_pbf();

    let mut slots: [Option<Vec<F::Packed>>; 6] = [None, None, None, None, None, None];
    {
        let [s0, s1, s2, s3, s4, s5] = &mut slots;
        rayon::scope(|sc| {
            sc.spawn(|_| *s0 = Some(build_one_productcheck_input::<F>(r0, r1, w0, i0)));
            sc.spawn(|_| *s1 = Some(build_one_productcheck_input::<F>(r0, r1, w1, i1)));
            sc.spawn(|_| *s2 = Some(build_one_productcheck_input::<F>(r0, r1, w2, i2)));
            sc.spawn(|_| *s3 = Some(build_one_productcheck_input::<F>(r0, r1, w0, p0)));
            sc.spawn(|_| *s4 = Some(build_one_productcheck_input::<F>(r0, r1, w1, p1)));
            sc.spawn(|_| *s5 = Some(build_one_productcheck_input::<F>(r0, r1, w2, p2)));
        });
    }

    let [s0, s1, s2, s3, s4, s5] = slots;
    [
        [s0.unwrap(), s1.unwrap(), s2.unwrap()],
        [s3.unwrap(), s4.unwrap(), s5.unwrap()],
    ]
}

#[inline]
fn build_one_productcheck_input<F: MamaBearExtConfig>(
    r0: F::Packed,
    r1: F::Packed,
    witness_pbf: &[PBF],
    rhs_pbf: &[PBF],
) -> Vec<F::Packed> {
    let n = witness_pbf.len();
    debug_assert_eq!(rhs_pbf.len(), n);
    let mut out: Vec<F::Packed> = Vec::with_capacity(n);
    let ptr = out.as_mut_ptr();
    for idx in 0..n {
        let value = r1
            .mul_base_elem(rhs_pbf[idx])
            .lazy_add(r0)
            .reduce_fast()
            .add_base_elem(witness_pbf[idx])
            .reduce_fast();
        unsafe { std::ptr::write(ptr.add(idx), value); }
    }
    unsafe { out.set_len(n); }
    out
}

impl<F: MamaBearExtConfig> ProverMamaBear<F>
where
    F::Packed: Send + Sync,
    F: Send + Sync,
{
    /// Parallel prove — same transcript output as serial `prove`.
    ///
    /// Uses parallel commit (wpc_new) and parallel ProductCheck.
    /// ZeroCheck, final_rsc, and open remain serial for now.
    pub fn prove_par(
        &self,
        pp: &DeepFoldMamaBearParam,
        nv: usize,
        mut witness: [AlignedPoly; NUM_WIRES],
    ) -> Proof {
        let mut transcript = Transcript::new();

        // Parallel commit
        let witness_pc = deepfold_mamabear_par::new_par::<F>(
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

        // Convert to Montgomery domain
        for poly in witness.iter_mut() {
            poly.to_montgomery_in_place();
        }

        // ZeroCheck (serial — only 3-5% of total)
        let sumcheck_r = (0..nv)
            .map(|_| transcript.challenge_f::<F>())
            .collect::<Vec<_>>();
        let (sumcheck_point, zero_evals) = F::prove_zero_check_par(
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

        // Parallel ProductCheck
        let prod_r0: F = transcript.challenge_f();
        let prod_r1: F = transcript.challenge_f();
        let prod_point = ProdEqCheckMamaBearPerWirePar::prove::<F::Packed>(
            build_productcheck_inputs_par::<F>(
                &witness,
                &self.prover_key.identical,
                &self.prover_key.permutation,
                prod_r0.to_mont(),
                prod_r1.to_mont(),
            ),
            &mut transcript,
        );

        // Mid claims — batched parallel evaluation over 6 polynomials.
        // See serial prove() for rationale: mid-stage openings must use the
        // per-wire ProductCheck's NATURAL reduced point (= simd_to_natural_point
        // of prod_point) to align with the verifier's ProductCheck reconstruction check.
        let prod_point_natural = crate::prover_mamabear::simd_to_natural_point(&prod_point[..nv]);
        let mid_polys: [&[_]; 6] = [
            witness[0].as_pbf(),
            witness[1].as_pbf(),
            witness[2].as_pbf(),
            self.prover_key.permutation[0].as_pbf(),
            self.prover_key.permutation[1].as_pbf(),
            self.prover_key.permutation[2].as_pbf(),
        ];
        let mid_evals = eval_multilinear_base_packed_batch_par::<F>(
            &mid_polys,
            &prod_point_natural,
        );
        for v in &mid_evals {
            transcript.append_f(v.from_mont());
        }

        // Final reduce sumcheck — parallel variant. Array-of-refs; no clones.
        let evals: [&AlignedPoly; NUM_ALL_TABLES] = [
            &self.prover_key.selector,
            &self.prover_key.permutation[0],
            &self.prover_key.permutation[1],
            &self.prover_key.permutation[2],
            &witness[0],
            &witness[1],
            &witness[2],
        ];
        let point_mont = prove_final_reduce_sumcheck_par::<F>(
            evals,
            &sumcheck_point,
            &prod_point[..nv],
            &mut transcript,
        );

        // Final claims + PCS open: permute to natural order so eval_multilinear_
        // base_packed and PCS see the NATURAL reduced point of the final-reduce
        // sumcheck.
        let point_mont_natural = crate::prover_mamabear::simd_to_natural_point(&point_mont);
        let final_polys: [&[_]; 7] = [
            self.prover_key.selector.as_pbf(),
            self.prover_key.permutation[0].as_pbf(),
            self.prover_key.permutation[1].as_pbf(),
            self.prover_key.permutation[2].as_pbf(),
            witness[0].as_pbf(),
            witness[1].as_pbf(),
            witness[2].as_pbf(),
        ];
        let final_evals = eval_multilinear_base_packed_batch_par::<F>(
            &final_polys,
            &point_mont_natural,
        );
        for v in &final_evals {
            transcript.append_f(v.from_mont());
        }

        // PCS open with parallel combine_polys and combine_subs
        DeepFoldMamaBearProver::open_par(
            pp,
            &[&self.prover_key.commitments, &witness_pc],
            point_mont_natural,
            &mut transcript,
        );

        transcript.proof
    }
}

// =============================================================================
// final_rsc (prove_final_reduce_sumcheck_par)
// =============================================================================

/// Parallel counterpart to `precombine_rho_trees_packed`.
///
/// The serial body is a pure map over `blocks` independent packed elements —
/// no dependencies, uniform cost per block. We dispatch to `par_chunks_mut`
/// when blocks >= PAR_FINAL_RSC_MIN_BLOCKS, falling back to serial for small
/// inputs (fork/join overhead dominates).
///
/// Output buffers use `with_capacity + set_len` to avoid the `IsZero` slow path
/// on the newtype `F::Packed`.
pub fn precombine_rho_trees_packed_par<F>(
    evals: [&AlignedPoly; NUM_ALL_TABLES],
    rho_mont: F,
    rho2: F,
    rho3: F,
    rho4: F,
    rho5: F,
) -> [Vec<F::Packed>; 2]
where
    F: MamaBearExtConfig + Send + Sync,
    F::Packed: Send + Sync,
{
    let blocks = evals[0].pbf_len();

    if blocks < PAR_FINAL_RSC_MIN_BLOCKS {
        return precombine_rho_trees_packed::<F>(evals, rho_mont, rho2, rho3, rho4, rho5);
    }

    let rho_p = <F::Packed as SumcheckExtField>::from_scalar(rho_mont);
    let rho2_p = <F::Packed as SumcheckExtField>::from_scalar(rho2);
    let rho3_p = <F::Packed as SumcheckExtField>::from_scalar(rho3);
    let rho4_p = <F::Packed as SumcheckExtField>::from_scalar(rho4);
    let rho5_p = <F::Packed as SumcheckExtField>::from_scalar(rho5);

    let p0 = evals[0].as_pbf();
    let p1 = evals[1].as_pbf();
    let p2 = evals[2].as_pbf();
    let p3 = evals[3].as_pbf();
    let p4 = evals[4].as_pbf();
    let p5 = evals[5].as_pbf();
    let p6 = evals[6].as_pbf();

    let mut tree0: Vec<F::Packed> = Vec::with_capacity(blocks);
    let mut tree1: Vec<F::Packed> = Vec::with_capacity(blocks);
    unsafe {
        tree0.set_len(blocks);
        tree1.set_len(blocks);
    }

    tree0
        .par_chunks_mut(PAR_FINAL_RSC_PRECOMBINE_CHUNK)
        .zip(tree1.par_chunks_mut(PAR_FINAL_RSC_PRECOMBINE_CHUNK))
        .enumerate()
        .for_each(|(chunk_idx, (t0_out, t1_out))| {
            let base = chunk_idx * PAR_FINAL_RSC_PRECOMBINE_CHUNK;
            let n = t0_out.len();
            for local in 0..n {
                let blk = base + local;
                // tree0 = b0 + rho*b4 + rho^2*b5 + rho^3*b6
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
                t0_out[local] = t0;
                t1_out[local] = t1;
            }
        });

    [tree0, tree1]
}

/// Parallel counterpart to `build_eq_table_packed`.
///
/// (first 3 scalar iterations) stays serial; parallelizes
/// the inner `j` loop per outer iteration via `par_chunks_mut` on the
/// destination buffer. Falls back to serial when `old_len` is below the
/// threshold.
pub(crate) fn build_eq_table_packed_par<F>(point: &[F], one: F) -> Vec<F::Packed>
where
    F: MamaBearExtConfig + Send + Sync,
    F::Packed: Send + Sync,
{
    let nv = point.len();
    let scalar_iters = nv.min(3);

    // scalar tensor expansion (up to 8 values).
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

    debug_assert_eq!(eq_scalar.len(), 8);
    let mut eq_packed = vec![<F::Packed as PackedExtensionField>::pack_slice_exact(&eq_scalar)];

    const ZIP_LO: [u64; 8] = [0, 8, 1, 9, 2, 10, 3, 11];
    const ZIP_HI: [u64; 8] = [4, 12, 5, 13, 6, 14, 7, 15];

    for i in scalar_iters..nv {
        let bit = point[nv - 1 - i];
        let bit_p = <F::Packed as SumcheckExtField>::from_scalar(bit);
        let one_minus_bit = (one.lazy_add_xp(2).lazy_sub(bit).con_sub_xp(2)).reduce();
        let one_minus_bit_p = <F::Packed as SumcheckExtField>::from_scalar(one_minus_bit);

        let old_len = eq_packed.len();
        let mut new_eq: Vec<F::Packed> = Vec::with_capacity(old_len * 2);
        unsafe { new_eq.set_len(old_len * 2); }

        if old_len >= PAR_FINAL_RSC_EQ_MIN_OLDLEN {
            // Parallel: each dst chunk of size `2 * src_chunk` corresponds to
            // `src_chunk` consecutive src elements. Writes are disjoint, reads
            // come from the immutable `eq_packed` captured by reference.
            let src_chunk = PAR_FINAL_RSC_EQ_CHUNK;
            let dst_chunk = src_chunk * 2;
            let src_view: &[F::Packed] = &eq_packed;
            new_eq
                .par_chunks_mut(dst_chunk)
                .enumerate()
                .for_each(|(chunk_idx, out)| {
                    let src_base = chunk_idx * src_chunk;
                    let pair_count = out.len() / 2;
                    for local in 0..pair_count {
                        let prod = src_view[src_base + local];
                        let a = (prod * one_minus_bit_p).reduce();
                        let b = (prod * bit_p).reduce();
                        out[local * 2] = F::zip_packed(a, b, ZIP_LO);
                        out[local * 2 + 1] = F::zip_packed(a, b, ZIP_HI);
                    }
                });
        } else {
            for j in 0..old_len {
                let prod = eq_packed[j];
                let a = (prod * one_minus_bit_p).reduce();
                let b = (prod * bit_p).reduce();
                new_eq[j * 2] = F::zip_packed(a, b, ZIP_LO);
                new_eq[j * 2 + 1] = F::zip_packed(a, b, ZIP_HI);
            }
        }

        eq_packed = new_eq;
    }

    eq_packed
}

/// Raw-pointer Send wrapper for rayon closures that need to write into
/// disjoint slices of a shared buffer. Callers must guarantee disjointness.
#[derive(Copy, Clone)]
pub(crate) struct SendPtrMut<T>(pub(crate) *mut T);
unsafe impl<T> Send for SendPtrMut<T> {}
unsafe impl<T> Sync for SendPtrMut<T> {}

impl<T> SendPtrMut<T> {
    #[inline(always)]
    pub(crate) fn ptr(self) -> *mut T {
        self.0
    }
}

/// Parallel Round-0 eval-only accumulation for final_rsc.
///
/// Reads-only over 4 input tables, returns the 2-tree, 3-hat accumulator.
#[inline]
pub(crate) fn round0_eval_par<F>(
    tree0: &[F::Packed],
    tree1: &[F::Packed],
    eq0: &[F::Packed],
    eq1: &[F::Packed],
) -> [[F::Packed; 3]; 2]
where
    F: MamaBearExtConfig + Send + Sync,
    F::Packed: Send + Sync,
{
    let half = tree0.len() >> 1;
    let chunk = PAR_FINAL_RSC_ROUND_GROUP_CHUNK;
    let n_chunks = (half + chunk - 1) / chunk;

    (0..n_chunks)
        .into_par_iter()
        .map(|chunk_idx| {
            let g_start = chunk_idx * chunk;
            let g_end = ((chunk_idx + 1) * chunk).min(half);
            let mut acc_p: [[F::Packed; 3]; 2] = [
                [F::Packed::zero(); 3],
                [F::Packed::zero(); 3],
            ];
            for g in g_start..g_end {
                let t0_l = tree0[2 * g];
                let t0_r = tree0[2 * g + 1];
                let t1_l = tree1[2 * g];
                let t1_r = tree1[2 * g + 1];
                let e0_l = eq0[2 * g];
                let e0_r = eq0[2 * g + 1];
                let e1_l = eq1[2 * g];
                let e1_r = eq1[2 * g + 1];

                // Extrapolate to x=2: x2 = r + (r - l).
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

                let g_local = g - g_start;
                if (g_local & 7) == 7 {
                    for tree in 0..2 {
                        for h in 0..3 {
                            acc_p[tree][h] = acc_p[tree][h].reduce_fast();
                        }
                    }
                }
            }
            // Final reduce_fast so the partial accumulator is safe to combine.
            for tree in 0..2 {
                for h in 0..3 {
                    acc_p[tree][h] = acc_p[tree][h].reduce_fast();
                }
            }
            acc_p
        })
        .reduce(
            || [[F::Packed::zero(); 3], [F::Packed::zero(); 3]],
            |mut a, b| {
                for tree in 0..2 {
                    for h in 0..3 {
                        a[tree][h] = a[tree][h].lazy_add(b[tree][h]).reduce_fast();
                    }
                }
                a
            },
        )
}

/// Parallel fused fold + eval for one packed round (rounds >= 1).
///
/// Reads the 4 src tables (length `current_len`), writes the 4 dst tables
/// (length `next_len = current_len / 2`) using ping-pong buffers from the
/// caller — src and dst are disjoint Vecs, so per-chunk mutations don't
/// alias reads.
///
/// Each group processes 4 src elements -> 2 dst elements: `src = 4g`,
/// `dst = 2g`. Chunks are slices of `PAR_FINAL_RSC_ROUND_GROUP_CHUNK` groups,
/// each thread handling disjoint src/dst ranges.
#[inline]
pub(crate) fn fold_eval_round_par<F>(
    alpha_p: F::Packed,
    tree0_src: &[F::Packed],
    tree0_dst: &mut [F::Packed],
    tree1_src: &[F::Packed],
    tree1_dst: &mut [F::Packed],
    eq0_src: &[F::Packed],
    eq0_dst: &mut [F::Packed],
    eq1_src: &[F::Packed],
    eq1_dst: &mut [F::Packed],
) -> [[F::Packed; 3]; 2]
where
    F: MamaBearExtConfig + Send + Sync,
    F::Packed: Send + Sync,
{
    let next_len = tree0_dst.len();
    debug_assert_eq!(tree1_dst.len(), next_len);
    debug_assert_eq!(eq0_dst.len(), next_len);
    debug_assert_eq!(eq1_dst.len(), next_len);
    debug_assert!(next_len * 2 <= tree0_src.len());
    let packed_groups = next_len >> 1;
    if packed_groups == 0 {
        return [[F::Packed::zero(); 3], [F::Packed::zero(); 3]];
    }

    let t0_ptr = SendPtrMut(tree0_dst.as_mut_ptr());
    let t1_ptr = SendPtrMut(tree1_dst.as_mut_ptr());
    let e0_ptr = SendPtrMut(eq0_dst.as_mut_ptr());
    let e1_ptr = SendPtrMut(eq1_dst.as_mut_ptr());

    let chunk = PAR_FINAL_RSC_ROUND_GROUP_CHUNK;
    let n_chunks = (packed_groups + chunk - 1) / chunk;

    (0..n_chunks)
        .into_par_iter()
        .map(move |chunk_idx| {
            let g_start = chunk_idx * chunk;
            let g_end = ((chunk_idx + 1) * chunk).min(packed_groups);
            // Call `ptr(self)` so the closure captures the whole `SendPtrMut`
            // struct (which impls Send+Sync) rather than field-projecting the
            // bare `*mut T` (which does not).
            let t0_p = t0_ptr.ptr();
            let t1_p = t1_ptr.ptr();
            let e0_p = e0_ptr.ptr();
            let e1_p = e1_ptr.ptr();
            let mut acc_p: [[F::Packed; 3]; 2] = [
                [F::Packed::zero(); 3],
                [F::Packed::zero(); 3],
            ];

            for g in g_start..g_end {
                let src = g << 2;
                let dst = g << 1;

                // tree0 fold -> (tv0, tv1)
                let te0 = tree0_src[src];
                let te1 = tree0_src[src + 1];
                let te2 = tree0_src[src + 2];
                let te3 = tree0_src[src + 3];
                let diff0 = te1.lazy_add_xp(2).lazy_sub(te0);
                let diff1 = te3.lazy_add_xp(2).lazy_sub(te2);
                let tv0_0 = (alpha_p * diff0).lazy_add(te0).con_sub_xp(2);
                let tv1_0 = (alpha_p * diff1).lazy_add(te2).con_sub_xp(2);
                unsafe {
                    std::ptr::write(t0_p.add(dst), tv0_0);
                    std::ptr::write(t0_p.add(dst + 1), tv1_0);
                }

                // tree1 fold -> (tv0, tv1)
                let te0 = tree1_src[src];
                let te1 = tree1_src[src + 1];
                let te2 = tree1_src[src + 2];
                let te3 = tree1_src[src + 3];
                let diff0 = te1.lazy_add_xp(2).lazy_sub(te0);
                let diff1 = te3.lazy_add_xp(2).lazy_sub(te2);
                let tv0_1 = (alpha_p * diff0).lazy_add(te0).con_sub_xp(2);
                let tv1_1 = (alpha_p * diff1).lazy_add(te2).con_sub_xp(2);
                unsafe {
                    std::ptr::write(t1_p.add(dst), tv0_1);
                    std::ptr::write(t1_p.add(dst + 1), tv1_1);
                }

                // eq0 fold -> (ev0, ev1)
                let qe0 = eq0_src[src];
                let qe1 = eq0_src[src + 1];
                let qe2 = eq0_src[src + 2];
                let qe3 = eq0_src[src + 3];
                let ed0 = qe1.lazy_add_xp(2).lazy_sub(qe0);
                let ed1 = qe3.lazy_add_xp(2).lazy_sub(qe2);
                let ev0_0 = (alpha_p * ed0).lazy_add(qe0).con_sub_xp(2);
                let ev1_0 = (alpha_p * ed1).lazy_add(qe2).con_sub_xp(2);
                unsafe {
                    std::ptr::write(e0_p.add(dst), ev0_0);
                    std::ptr::write(e0_p.add(dst + 1), ev1_0);
                }

                // eq1 fold -> (ev0, ev1)
                let qe0 = eq1_src[src];
                let qe1 = eq1_src[src + 1];
                let qe2 = eq1_src[src + 2];
                let qe3 = eq1_src[src + 3];
                let ed0 = qe1.lazy_add_xp(2).lazy_sub(qe0);
                let ed1 = qe3.lazy_add_xp(2).lazy_sub(qe2);
                let ev0_1 = (alpha_p * ed0).lazy_add(qe0).con_sub_xp(2);
                let ev1_1 = (alpha_p * ed1).lazy_add(qe2).con_sub_xp(2);
                unsafe {
                    std::ptr::write(e1_p.add(dst), ev0_1);
                    std::ptr::write(e1_p.add(dst + 1), ev1_1);
                }

                // Eval round poly from the folded (tv0,tv1), (ev0,ev1).
                let t_diff0 = tv1_0.lazy_add_xp(2).lazy_sub(tv0_0).con_sub_xp(2);
                let t_x2_0 = tv1_0.lazy_add(t_diff0).con_sub_xp(2);
                let e_diff0 = ev1_0.lazy_add_xp(2).lazy_sub(ev0_0).con_sub_xp(2);
                let e_x2_0 = ev1_0.lazy_add(e_diff0).con_sub_xp(2);
                acc_p[0][0] = acc_p[0][0].lazy_add(tv0_0 * ev0_0);
                acc_p[0][1] = acc_p[0][1].lazy_add(tv1_0 * ev1_0);
                acc_p[0][2] = acc_p[0][2].lazy_add(t_x2_0 * e_x2_0);

                let t_diff1 = tv1_1.lazy_add_xp(2).lazy_sub(tv0_1).con_sub_xp(2);
                let t_x2_1 = tv1_1.lazy_add(t_diff1).con_sub_xp(2);
                let e_diff1 = ev1_1.lazy_add_xp(2).lazy_sub(ev0_1).con_sub_xp(2);
                let e_x2_1 = ev1_1.lazy_add(e_diff1).con_sub_xp(2);
                acc_p[1][0] = acc_p[1][0].lazy_add(tv0_1 * ev0_1);
                acc_p[1][1] = acc_p[1][1].lazy_add(tv1_1 * ev1_1);
                acc_p[1][2] = acc_p[1][2].lazy_add(t_x2_1 * e_x2_1);

                let g_local = g - g_start;
                if (g_local & 7) == 7 {
                    for tree in 0..2 {
                        for h in 0..3 {
                            acc_p[tree][h] = acc_p[tree][h].reduce_fast();
                        }
                    }
                }
            }
            for tree in 0..2 {
                for h in 0..3 {
                    acc_p[tree][h] = acc_p[tree][h].reduce_fast();
                }
            }
            acc_p
        })
        .reduce(
            || [[F::Packed::zero(); 3], [F::Packed::zero(); 3]],
            |mut a, b| {
                for tree in 0..2 {
                    for h in 0..3 {
                        a[tree][h] = a[tree][h].lazy_add(b[tree][h]).reduce_fast();
                    }
                }
                a
            },
        )
}

/// Parallel counterpart to `prove_final_reduce_sumcheck` in prover_mamabear.rs.
///
/// Parallelization layout:
///  - `precombine_rho_trees_packed_par`: par_chunks_mut over the 7-poly combine.
///  - `build_eq_table_packed_par` x 2 via `rayon::join` (inner parallel too).
///  - Round 0: par fold-reduce over `half_packed` read-only groups.
///  - Rounds 1..packed_rounds-1: ping-pong buffers, parallel fused fold+eval.
///  - Final pre-tail fold + scalar tail: serial (tables are tiny here).
///
/// Produces a bit-identical transcript to the serial version.
pub fn prove_final_reduce_sumcheck_par<F>(
    evals: [&AlignedPoly; NUM_ALL_TABLES],
    sumcheck_point: &[F],
    prod_point: &[F],
    transcript: &mut Transcript,
) -> Vec<F>
where
    F: MamaBearExtConfig + Send + Sync,
    F::Packed: Send + Sync,
{
    let num_vars = sumcheck_point.len();
    assert_eq!(num_vars, prod_point.len());

    // Small-input fallback: below threshold, fork/join cost > parallel benefit.
    if evals[0].pbf_len() < PAR_FINAL_RSC_MIN_BLOCKS {
        return prove_final_reduce_sumcheck::<F>(
            evals,
            sumcheck_point,
            prod_point,
            transcript,
        );
    }

    // Draw rho and materialize rho^1..rho^5.
    let rho_raw: F = transcript.challenge_f();
    let rho_mont = rho_raw.to_mont();
    let rho2 = (rho_mont * rho_mont).reduce();
    let rho3 = (rho2 * rho_mont).reduce();
    let rho4 = (rho3 * rho_mont).reduce();
    let rho5 = (rho4 * rho_mont).reduce();

    // ---- Parallel precombine (7 base polys -> 2 ext trees). ----
    let [mut tree0, mut tree1] =
        precombine_rho_trees_packed_par::<F>(evals, rho_mont, rho2, rho3, rho4, rho5);

    // ---- Parallel eq-table build (eq0 from sumcheck_point, eq1 from prod_point). ----
    // Both sumcheck_point and prod_point arrive in Montgomery form (SIMD-round
    // order). Permute to natural order (no extra to_mont — already Mont).
    // See prove_final_reduce_sumcheck in the serial module for the full rationale.
    let point0_mont = crate::prover_mamabear::simd_to_natural_point(sumcheck_point);
    let point1_mont = crate::prover_mamabear::simd_to_natural_point(prod_point);
    let one = F::one().to_mont();
    let (mut eq0, mut eq1) = rayon::join(
        || build_eq_table_packed_par::<F>(&point0_mont, one),
        || build_eq_table_packed_par::<F>(&point1_mont, one),
    );

    // ---- Ping-pong scratch buffers (peak = tree0.len() / 2). ----
    // Uninit allocation: every element is written by the parallel round loop
    // before it is read, so we skip the IsZero slow path.
    let peak = tree0.len() / 2;
    let mut tree0_b: Vec<F::Packed> = Vec::with_capacity(peak);
    let mut tree1_b: Vec<F::Packed> = Vec::with_capacity(peak);
    let mut eq0_b: Vec<F::Packed> = Vec::with_capacity(peak);
    let mut eq1_b: Vec<F::Packed> = Vec::with_capacity(peak);
    unsafe {
        tree0_b.set_len(peak);
        tree1_b.set_len(peak);
        eq0_b.set_len(peak);
        eq1_b.set_len(peak);
    }

    let mut challenges: Vec<F> = Vec::with_capacity(num_vars);
    let packed_rounds = if num_vars >= 4 { num_vars - 3 } else { 0 };

    // ---- Round 0: eval-only from initial packed tables. ----
    if packed_rounds > 0 {
        let acc_p = round0_eval_par::<F>(&tree0, &tree1, &eq0, &eq1);

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

    // ---- Rounds 1..packed_rounds-1: parallel fused fold + eval via ping-pong. ----
    // `cur_is_a == true`: src is tree*/eq* (the "a" side), dst is tree*_b/eq*_b.
    // Flip after each round.
    let mut cur_is_a = true;
    let mut current_len = tree0.len();

    for _round in 1..packed_rounds {
        let alpha_prev = *challenges.last().unwrap();
        let alpha_p = <F::Packed as SumcheckExtField>::from_scalar(alpha_prev);
        let next_len = current_len >> 1;

        let acc_p = if cur_is_a {
            let (tree0_src, tree0_dst) = (&tree0[..current_len], &mut tree0_b[..next_len]);
            let (tree1_src, tree1_dst) = (&tree1[..current_len], &mut tree1_b[..next_len]);
            let (eq0_src, eq0_dst) = (&eq0[..current_len], &mut eq0_b[..next_len]);
            let (eq1_src, eq1_dst) = (&eq1[..current_len], &mut eq1_b[..next_len]);
            fold_eval_round_par::<F>(
                alpha_p,
                tree0_src, tree0_dst,
                tree1_src, tree1_dst,
                eq0_src, eq0_dst,
                eq1_src, eq1_dst,
            )
        } else {
            let (tree0_src, tree0_dst) = (&tree0_b[..current_len], &mut tree0[..next_len]);
            let (tree1_src, tree1_dst) = (&tree1_b[..current_len], &mut tree1[..next_len]);
            let (eq0_src, eq0_dst) = (&eq0_b[..current_len], &mut eq0[..next_len]);
            let (eq1_src, eq1_dst) = (&eq1_b[..current_len], &mut eq1[..next_len]);
            fold_eval_round_par::<F>(
                alpha_p,
                tree0_src, tree0_dst,
                tree1_src, tree1_dst,
                eq0_src, eq0_dst,
                eq1_src, eq1_dst,
            )
        };

        // Sum lanes and send round polynomial.
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

        cur_is_a = !cur_is_a;
        current_len = next_len;
    }

    // ---- Final fold before scalar tail: in-place, serial (tiny). ----
    // Produces a single packed-length-1 set of tables prior to unpack.
    //
    // If `cur_is_a == true`, the valid data is in tree0/tree1/eq0/eq1; otherwise
    // it's in tree0_b/tree1_b/eq0_b/eq1_b. We call the final fold on whichever
    // is current and consolidate into the a-side so the scalar tail can read
    // from tree0/tree1/eq0/eq1 uniformly.
    if packed_rounds > 0 && current_len >= 2 {
        let alpha_last = *challenges.last().unwrap();
        let alpha_p = <F::Packed as SumcheckExtField>::from_scalar(alpha_last);
        let fold = |left: F::Packed, right: F::Packed| -> F::Packed {
            let diff = right.lazy_add_xp(2).lazy_sub(left);
            (alpha_p * diff).lazy_add(left).con_sub_xp(2)
        };
        let half = current_len >> 1;
        if cur_is_a {
            for g in 0..half {
                tree0[g] = fold(tree0[2 * g], tree0[2 * g + 1]);
                tree1[g] = fold(tree1[2 * g], tree1[2 * g + 1]);
                eq0[g] = fold(eq0[2 * g], eq0[2 * g + 1]);
                eq1[g] = fold(eq1[2 * g], eq1[2 * g + 1]);
            }
            tree0.truncate(half);
            tree1.truncate(half);
            eq0.truncate(half);
            eq1.truncate(half);
        } else {
            // Fold from *_b (current) back into the a-side tables (which happen
            // to be at the right capacity), then re-wire a-side as the current
            // buffer going into the scalar tail.
            for g in 0..half {
                tree0[g] = fold(tree0_b[2 * g], tree0_b[2 * g + 1]);
                tree1[g] = fold(tree1_b[2 * g], tree1_b[2 * g + 1]);
                eq0[g] = fold(eq0_b[2 * g], eq0_b[2 * g + 1]);
                eq1[g] = fold(eq1_b[2 * g], eq1_b[2 * g + 1]);
            }
            tree0.truncate(half);
            tree1.truncate(half);
            eq0.truncate(half);
            eq1.truncate(half);
        }
    } else if packed_rounds > 0 && !cur_is_a {
        // No final fold needed (current_len < 2 after the loop) but the valid
        // data is in *_b. Copy the single element into the a-side so the tail
        // can unpack uniformly.
        tree0[..current_len].copy_from_slice(&tree0_b[..current_len]);
        tree1[..current_len].copy_from_slice(&tree1_b[..current_len]);
        eq0[..current_len].copy_from_slice(&eq0_b[..current_len]);
        eq1[..current_len].copy_from_slice(&eq1_b[..current_len]);
        tree0.truncate(current_len);
        tree1.truncate(current_len);
        eq0.truncate(current_len);
        eq1.truncate(current_len);
    }

    // ---- Scalar tail (last 3 rounds): serial, small. ----
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
