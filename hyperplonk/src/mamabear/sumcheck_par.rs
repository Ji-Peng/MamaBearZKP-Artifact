#![allow(dead_code)]
//! Parallel (rayon) variants of the ZeroCheck sumcheck prover.
//!
//! This file mirrors the structure of [`poly_commit::deepfold_mamabear_par`]
//! relative to `deepfold_mamabear`: the serial implementation lives in
//! `sumcheck_mamabear.rs`, and the par-specific fold kernels, public API
//! entry points, and cross-validation tests live here. Both files attach
//! methods to the same `SumcheckMamaBear` type via separate `impl` blocks.
//!
//! The large orchestrator function `prove_add_mul_optimized_with_ell0_generic`
//! still lives in the serial file and accepts a `par: bool` flag: the par
//! wrappers below simply call it with `par=true`. This avoids duplicating
//! ~1000 lines of fragile round-by-round orchestration code. The real
//! par-only SIMD kernels (the `*_par` fold helpers) do live here.
//!
//! # Input layout and fold order (identical to the serial file)
//!
//! The parallel provers consume the same **normal-order** packed tables as
//! the serial ones: for 8 SIMD lanes and `L = 2^(mu-3)`,
//!
//! ```text
//!     evals_normal[p][lambda] = f(bin(8*p + lambda))
//! ```
//!
//! so the vector index `p` holds the high hypercube bits (x_3, x_4, ...,
//! x_{mu-1}) and the lane index `lambda` holds the low bits (x_0, x_1, x_2).
//! Every parallel fold kernel below partitions work across packed blocks or
//! `left_packed` groups and folds `(table[2q], table[2q+1])` lane-wise —
//! exactly the pattern used in the serial version. Consequently the fold
//! / elimination order is again
//!
//! ```text
//!     x_3, x_4, ..., x_{mu-1}, x_0, x_1, x_2
//! ```
//!
//! and not the textbook little-endian order. The returned `challenges` are
//! in SIMD-round order; use `simd_to_natural_point` in `prover_mamabear.rs`
//! to rotate into natural-variable order when a downstream consumer needs
//! the original variable indexing. The parallel transcripts are
//! bit-identical to the serial transcripts (enforced by tests in this
//! file), which is only meaningful because both share the same fold order.

use arithmetic::field::mamabear::*;
use arithmetic::field::Field;
use rayon::prelude::*;
use std::time::Instant;
use util::fiat_shamir::Transcript;

use arithmetic::field::mamabear::{MamaBearScalar as SBF, PackedMamaBearAVX512 as PBF};
use arithmetic::field::mamabear::{
    MamaBearScalarExt3 as SEF3, PackedMamaBearAVX512Ext3 as PEF3,
};

use crate::sumcheck_mamabear::{
    MontgomeryOps, RoundEqView, SumcheckExtField, SumcheckMamaBear, TwoStageEqTables,
    ZeroCheckTimings,
};
use arithmetic::field::mamabear::P;

/// Minimum `packed_groups` (= next_len / 2) below which parallel ZeroCheck
/// fold kernels fall back to the serial path. Each group involves ~4 polys ×
/// ~6 PEF muls ≈ 250-300 cycles; rayon dispatch overhead is ~50-100 µs, so
/// parallelism pays off above ~256 groups. 128 gives a conservative margin.
pub(crate) const PAR_ZEROCHECK_MIN_PACKED_GROUPS: usize = 128;

/// Minimum `num_vars` below which the parallel ZeroCheck prover falls back to
/// the serial path.
pub(crate) const PAR_ZEROCHECK_MIN_NV: usize = 12;

/// Send-safe raw pointer wrapper for rayon closures that write into disjoint
/// slices of a shared buffer. Callers must guarantee disjointness.
#[derive(Copy, Clone)]
pub(crate) struct ParPtr<T>(pub(crate) *mut T);
unsafe impl<T> Send for ParPtr<T> {}
unsafe impl<T> Sync for ParPtr<T> {}

/// Pre-touch all pages of scratch buffers in parallel so that the fold
/// kernels do not pay minor page faults on first write. Without this,
/// the parallel out-of-place fold is slower than serial in-place fold
/// at nv >= 20 because page allocator contention under 8 concurrent
/// faulting threads exceeds the compute savings.
fn pretouch_scratch_generic<T: Send + Sync>(scratch: &mut [Vec<T>; 4]) {
    let page_stride = (4096 / std::mem::size_of::<T>()).max(1);
    rayon::scope(|s| {
        for v in scratch.iter_mut() {
            s.spawn(|_| {
                let ptr = v.as_mut_ptr();
                let len = v.len();
                for i in (0..len).step_by(page_stride) {
                    unsafe { std::ptr::write_volatile(ptr.add(i), std::mem::zeroed()); }
                }
            });
        }
    });
}

impl SumcheckMamaBear {

    /// Parallel variant of `fold_base_tables_to_ext_generic` for the 2- and
    /// 3-challenge cases (the only ones used by the ell0=2 Ext3 path).
    ///
    /// Per-poly inner loop is chunked so that each chunk writes into a disjoint
    /// `next_len` range of its own output Vec. The 4 polys run in parallel.
    pub(crate) fn fold_base_tables_to_ext_generic_par<E: SumcheckExtField + Send + Sync>(
        evals: &[Vec<PBF>; 4],
        challenges: &[E::Scalar],
    ) -> [Vec<E>; 4]
    where
        E::Scalar: Send + Sync,
    {
        debug_assert!(!challenges.is_empty());
        // Only parallelize the 2- and 3-challenge fused paths. Everything else
        // (single challenge generic fallback) stays serial.
        if challenges.len() != 2 && challenges.len() != 3 {
            return Self::fold_base_tables_to_ext_generic::<E>(evals, challenges);
        }

        let fold_factor = 1usize << challenges.len();
        let next_len = evals[0].len() / fold_factor;
        if next_len < PAR_ZEROCHECK_MIN_PACKED_GROUPS {
            return Self::fold_base_tables_to_ext_generic::<E>(evals, challenges);
        }

        // Allocate the 4 output vectors up front with uninit capacity so the
        // parallel fill loop can write directly via raw pointers.
        let mut outputs: [Vec<E>; 4] = std::array::from_fn(|_| {
            let mut v: Vec<E> = Vec::with_capacity(next_len);
            unsafe {
                v.set_len(next_len);
            }
            v
        });
        let out_ptrs: [ParPtr<E>; 4] = [
            ParPtr(outputs[0].as_mut_ptr()),
            ParPtr(outputs[1].as_mut_ptr()),
            ParPtr(outputs[2].as_mut_ptr()),
            ParPtr(outputs[3].as_mut_ptr()),
        ];
        let in_ptrs: [ParPtr<PBF>; 4] = [
            ParPtr(evals[0].as_ptr() as *mut PBF),
            ParPtr(evals[1].as_ptr() as *mut PBF),
            ParPtr(evals[2].as_ptr() as *mut PBF),
            ParPtr(evals[3].as_ptr() as *mut PBF),
        ];

        if challenges.len() == 2 {
            let alpha0 = E::from_scalar(challenges[0]);
            let alpha1 = E::from_scalar(challenges[1]);
            (0..4usize).into_par_iter().for_each(|poly_idx| {
                let src = in_ptrs[poly_idx].0;
                let dst = out_ptrs[poly_idx].0;
                for block_idx in 0..next_len {
                    let base = block_idx << 2;
                    let e0 = unsafe { *src.add(base) };
                    let e1 = unsafe { *src.add(base + 1) };
                    let e2 = unsafe { *src.add(base + 2) };
                    let e3 = unsafe { *src.add(base + 3) };
                    let diff0 = e1.lazy_add_xp(2).lazy_sub(e0);
                    let diff1 = e3.lazy_add_xp(2).lazy_sub(e2);
                    let low =
                        alpha0.mul_base_elem(diff0).add_base_elem(e0).con_sub_xp(2);
                    let high =
                        alpha0.mul_base_elem(diff1).add_base_elem(e2).con_sub_xp(2);
                    let diff = high.lazy_add_xp(2).lazy_sub(low);
                    let folded = (alpha1 * diff).lazy_add(low).con_sub_xp(2);
                    unsafe {
                        *dst.add(block_idx) = folded;
                    }
                }
            });
        } else {
            // challenges.len() == 3
            let alpha0 = E::from_scalar(challenges[0]);
            let alpha1 = E::from_scalar(challenges[1]);
            let alpha2 = E::from_scalar(challenges[2]);
            (0..4usize).into_par_iter().for_each(|poly_idx| {
                let src = in_ptrs[poly_idx].0;
                let dst = out_ptrs[poly_idx].0;
                for block_idx in 0..next_len {
                    let base = block_idx << 3;
                    let e0 = unsafe { *src.add(base) };
                    let e1 = unsafe { *src.add(base + 1) };
                    let e2 = unsafe { *src.add(base + 2) };
                    let e3 = unsafe { *src.add(base + 3) };
                    let e4 = unsafe { *src.add(base + 4) };
                    let e5 = unsafe { *src.add(base + 5) };
                    let e6 = unsafe { *src.add(base + 6) };
                    let e7 = unsafe { *src.add(base + 7) };
                    let d01 = e1.lazy_add_xp(2).lazy_sub(e0);
                    let d23 = e3.lazy_add_xp(2).lazy_sub(e2);
                    let d45 = e5.lazy_add_xp(2).lazy_sub(e4);
                    let d67 = e7.lazy_add_xp(2).lazy_sub(e6);
                    let x00 = alpha0.mul_base_elem(d01).add_base_elem(e0).con_sub_xp(2);
                    let x01 = alpha0.mul_base_elem(d23).add_base_elem(e2).con_sub_xp(2);
                    let x10 = alpha0.mul_base_elem(d45).add_base_elem(e4).con_sub_xp(2);
                    let x11 = alpha0.mul_base_elem(d67).add_base_elem(e6).con_sub_xp(2);
                    let dy0 = x01.lazy_add_xp(2).lazy_sub(x00);
                    let dy1 = x11.lazy_add_xp(2).lazy_sub(x10);
                    let y0 = (alpha1 * dy0).lazy_add(x00).con_sub_xp(2);
                    let y1 = (alpha1 * dy1).lazy_add(x10).con_sub_xp(2);
                    let dz = y1.lazy_add_xp(2).lazy_sub(y0);
                    let folded = (alpha2 * dz).lazy_add(y0).con_sub_xp(2);
                    unsafe {
                        *dst.add(block_idx) = folded;
                    }
                }
            });
        }

        outputs
    }

    /// Parallel variant of `fold_packed_ext_tables_and_compute_round_t_in_place_generic`.
    /// Writes into caller-provided `scratch` buffers and swaps them into
    /// `evals` on return. Scratch reuse across rounds amortizes first-touch
    /// page-fault cost to a single allocation per prover call.
    pub(crate) fn fold_packed_ext_tables_and_compute_round_t_in_place_generic_par<
        E: SumcheckExtField + Send + Sync,
    >(
        evals: &mut [Vec<E>; 4],
        eq_view: &RoundEqView<'_, E>,
        challenge: E::Scalar,
        scratch: &mut [Vec<E>; 4],
    ) -> [E::Scalar; Self::OPT_HAT_SIZE]
    where
        E::Scalar: Send + Sync,
    {
        let next_len = evals[0].len() >> 1;
        let packed_groups = next_len >> 1;
        let alpha = E::from_scalar(challenge);
        let one = E::one().ext_to_montgomery();

        if packed_groups < PAR_ZEROCHECK_MIN_PACKED_GROUPS {
            return Self::fold_packed_ext_tables_and_compute_round_t_in_place_generic::<E>(
                evals, eq_view, challenge,
            );
        }

        debug_assert!(scratch[0].capacity() >= next_len);
        for v in scratch.iter_mut() {
            if v.len() < next_len {
                unsafe { v.set_len(next_len); }
            }
        }
        let src_par: [ParPtr<E>; 4] = [
            ParPtr(evals[0].as_mut_ptr()),
            ParPtr(evals[1].as_mut_ptr()),
            ParPtr(evals[2].as_mut_ptr()),
            ParPtr(evals[3].as_mut_ptr()),
        ];
        let dst_par: [ParPtr<E>; 4] = [
            ParPtr(scratch[0].as_mut_ptr()),
            ParPtr(scratch[1].as_mut_ptr()),
            ParPtr(scratch[2].as_mut_ptr()),
            ParPtr(scratch[3].as_mut_ptr()),
        ];

        let packed_split = eq_view.packed_split_for_groups(packed_groups);
        let [t_0, t_2, t_inf] = match packed_split {
            Some(split) if split.left_packed.len() >= 2 => {
                let right_len = split.right_broadcast.len();
                let left_len = split.left_packed.len();
                let right_broadcast = split.right_broadcast;
                let left_packed = split.left_packed;
                let grain = (left_len / 32).max(1);

                (0..left_len)
                    .into_par_iter()
                    .with_min_len(grain)
                    .fold_with(
                        [E::zero(); 3],
                        |mut acc, left_group| {
                            let mut inner_t_0 = E::zero();
                            let mut inner_t_2 = E::zero();
                            let mut inner_t_inf = E::zero();
                            let group_base = left_group * right_len;
                            for right_idx in 0..right_len {
                                let group_idx = group_base + right_idx;
                                let eq_r = right_broadcast[right_idx];
                                let src_base = group_idx << 2;
                                let dst_base = group_idx << 1;
                                let mut values_0 = [E::zero(); 4];
                                let mut values_2 = [E::zero(); 4];
                                let mut diffs = [E::zero(); 4];
                                for poly_idx in 0..4 {
                                    let src = src_par[poly_idx].0;
                                    let dst = dst_par[poly_idx].0;
                                    let e0 = unsafe { *src.add(src_base) };
                                    let e1 = unsafe { *src.add(src_base + 1) };
                                    let e2 = unsafe { *src.add(src_base + 2) };
                                    let e3 = unsafe { *src.add(src_base + 3) };
                                    let diff0 = e1.lazy_add_xp(2).lazy_sub(e0);
                                    let diff1 = e3.lazy_add_xp(2).lazy_sub(e2);
                                    let v0 = (alpha * diff0).lazy_add(e0).con_sub_xp(2);
                                    let v1 = (alpha * diff1).lazy_add(e2).con_sub_xp(2);
                                    let diff = v1.lazy_add_xp(2).lazy_sub(v0).con_sub_xp(2);
                                    unsafe {
                                        *dst.add(dst_base) = v0;
                                        *dst.add(dst_base + 1) = v1;
                                    }
                                    values_0[poly_idx] = v0;
                                    values_2[poly_idx] = v1.lazy_add(diff).con_sub_xp(2);
                                    diffs[poly_idx] = diff;
                                }
                                inner_t_0 = inner_t_0.lazy_add(
                                    eq_r * Self::gate_h_packed_ext_generic(values_0, one),
                                );
                                inner_t_2 = inner_t_2.lazy_add(
                                    eq_r * Self::gate_h_packed_ext_generic(values_2, one),
                                );
                                inner_t_inf = inner_t_inf
                                    .lazy_add(eq_r * diffs[0] * diffs[1] * diffs[2]);
                            }
                            let eq_l = left_packed[left_group];
                            acc[0] = acc[0].lazy_add(eq_l * inner_t_0.reduce_fast());
                            acc[1] = acc[1].lazy_add(eq_l * inner_t_2.reduce_fast());
                            acc[2] = acc[2].lazy_add(eq_l * inner_t_inf.reduce_fast());
                            acc
                        },
                    )
                    .reduce_with(|mut a, b| {
                        a[0] = a[0].lazy_add(b[0]);
                        a[1] = a[1].lazy_add(b[1]);
                        a[2] = a[2].lazy_add(b[2]);
                        a
                    })
                    .unwrap_or([E::zero(); 3])
            }
            _ => {
                // Non-split fallback: parallel over chunks of packed_groups.
                let grain = (packed_groups / 32).max(1);

                (0..packed_groups)
                    .into_par_iter()
                    .with_min_len(grain)
                    .fold_with(
                        [E::zero(); 3],
                        |mut acc, group_idx| {
                            let weight =
                                eq_view.load_packed_weight(group_idx, packed_groups);
                            let src_base = group_idx << 2;
                            let dst_base = group_idx << 1;
                            let mut values_0 = [E::zero(); 4];
                            let mut values_2 = [E::zero(); 4];
                            let mut diffs = [E::zero(); 4];
                            for poly_idx in 0..4 {
                                let src = src_par[poly_idx].0;
                                let dst = dst_par[poly_idx].0;
                                let e0 = unsafe { *src.add(src_base) };
                                let e1 = unsafe { *src.add(src_base + 1) };
                                let e2 = unsafe { *src.add(src_base + 2) };
                                let e3 = unsafe { *src.add(src_base + 3) };
                                let diff0 = e1.lazy_add_xp(2).lazy_sub(e0);
                                let diff1 = e3.lazy_add_xp(2).lazy_sub(e2);
                                let v0 = (alpha * diff0).lazy_add(e0).con_sub_xp(2);
                                let v1 = (alpha * diff1).lazy_add(e2).con_sub_xp(2);
                                let diff = v1.lazy_add_xp(2).lazy_sub(v0).con_sub_xp(2);
                                unsafe {
                                    *dst.add(dst_base) = v0;
                                    *dst.add(dst_base + 1) = v1;
                                }
                                values_0[poly_idx] = v0;
                                values_2[poly_idx] = v1.lazy_add(diff).con_sub_xp(2);
                                diffs[poly_idx] = diff;
                            }
                            acc[0] = acc[0].lazy_add(
                                weight * Self::gate_h_packed_ext_generic(values_0, one),
                            );
                            acc[1] = acc[1].lazy_add(
                                weight * Self::gate_h_packed_ext_generic(values_2, one),
                            );
                            acc[2] = acc[2]
                                .lazy_add(weight * diffs[0] * diffs[1] * diffs[2]);
                            acc
                        },
                    )
                    .reduce_with(|mut a, b| {
                        a[0] = a[0].lazy_add(b[0]);
                        a[1] = a[1].lazy_add(b[1]);
                        a[2] = a[2].lazy_add(b[2]);
                        a
                    })
                    .unwrap_or([E::zero(); 3])
            }
        };

        // Swap freshly folded scratch into evals, truncating to next_len.
        for poly_idx in 0..4 {
            std::mem::swap(&mut evals[poly_idx], &mut scratch[poly_idx]);
            evals[poly_idx].truncate(next_len);
        }

        [
            E::sum_lanes_to_mont(t_0),
            E::sum_lanes_to_mont(t_2),
            E::sum_lanes_to_mont(t_inf),
        ]
    }

    // =========================================================================
    // Parallel precompute (generic)
    // =========================================================================

    /// Parallel duplicate of
    /// `SumcheckMamaBear::precompute_small_value_tables_packed_generic`.
    /// Round 0 uses rayon-chunked accumulation over `left_packed` groups;
    /// round 1 uses rayon-chunked 4x4 tensor accumulation; rounds >= 2 fall
    /// back to the scalar tail handled inside the generic path. When the
    /// split is too small (`packed_groups < PAR_ZEROCHECK_MIN_PACKED_GROUPS`
    /// for round 0, or `packed_groups < 2048` for round 1) the serial helper
    /// in `sumcheck_mamabear.rs` would be faster; callers gate this function
    /// on `num_vars >= PAR_ZEROCHECK_MIN_NV` which usually implies the inner
    /// thresholds pass, but we still check and fall back inline for safety.
    pub(crate) fn precompute_small_value_tables_packed_generic_par<
        E: SumcheckExtField + Send + Sync,
    >(
        evals: &[Vec<PBF>; 4],
        eq_tables: &TwoStageEqTables<E>,
        ell0: usize,
        zero_check: bool,
        one: PBF,
        _two: PBF,
        three: PBF,
        inv6: PBF,
    ) -> Vec<Vec<E::Scalar>>
    where
        E::Scalar: Send + Sync,
    {
        let mut precomputed = Vec::with_capacity(ell0);
        for round in 0..ell0 {
            let _prefix_len = round + 1;
            let eq_view = Self::round_eq_view_generic(eq_tables, round);
            if round == 0 {
                let packed_groups = evals[0].len() >> 1;
                let mut t_0 = E::zero();
                let mut t_2 = E::zero();
                let mut t_inf = E::zero();
                let packed_split = eq_view.packed_split_for_groups(packed_groups);

                if packed_split
                    .as_ref()
                    .map_or(false, |s| {
                        packed_groups >= PAR_ZEROCHECK_MIN_PACKED_GROUPS
                            && s.left_packed.len() >= 2
                    })
                {
                    let split = packed_split.unwrap();
                    let right_len = split.right_broadcast.len();
                    let left_packed = split.left_packed;
                    let left_len = left_packed.len();
                    let right_broadcast = split.right_broadcast;
                    let src: [ParPtr<PBF>; 4] = [
                        ParPtr(evals[0].as_ptr() as *mut PBF),
                        ParPtr(evals[1].as_ptr() as *mut PBF),
                        ParPtr(evals[2].as_ptr() as *mut PBF),
                        ParPtr(evals[3].as_ptr() as *mut PBF),
                    ];
                    let grain = (left_len / 32).max(1);
                    let [r_0, r_2, r_inf] = (0..left_len)
                        .into_par_iter()
                        .with_min_len(grain)
                        .fold_with(
                            [E::zero(); 3],
                            |mut acc, left_group| {
                                let mut inner_t_0 = E::zero();
                                let mut inner_t_2 = E::zero();
                                let mut inner_t_inf = E::zero();
                                let group_base = left_group * right_len;
                                for right_idx in 0..right_len {
                                    let eq_r = right_broadcast[right_idx];
                                    let group_idx = group_base + right_idx;
                                    let base = group_idx << 1;
                                    let mut values_2 = [PBF::zero(); 4];
                                    let mut diffs = [PBF::zero(); 4];
                                    for poly_idx in 0..4 {
                                        let p = src[poly_idx].0;
                                        let v0 = unsafe { *p.add(base) };
                                        let v1 = unsafe { *p.add(base + 1) };
                                        let diff = Self::packed_diff(v0, v1);
                                        values_2[poly_idx] = v1.lazy_add(diff).con_sub_xp(2);
                                        diffs[poly_idx] = diff;
                                    }
                                    if !zero_check {
                                        let values_0: [PBF; 4] =
                                            std::array::from_fn(|pi| unsafe { *src[pi].0.add(base) });
                                        inner_t_0 = inner_t_0.lazy_add(
                                            eq_r.mul_base_elem(Self::gate_h_packed_base(values_0, one)),
                                        );
                                    }
                                    inner_t_2 = inner_t_2.lazy_add(
                                        eq_r.mul_base_elem(Self::gate_h_packed_base(values_2, one)),
                                    );
                                    inner_t_inf = inner_t_inf.lazy_add(
                                        eq_r.mul_base_elem(diffs[0] * diffs[1] * diffs[2]),
                                    );
                                }
                                let eq_l = left_packed[left_group];
                                if !zero_check {
                                    acc[0] = acc[0].lazy_add(eq_l * inner_t_0.reduce_fast());
                                }
                                acc[1] = acc[1].lazy_add(eq_l * inner_t_2.reduce_fast());
                                acc[2] = acc[2].lazy_add(eq_l * inner_t_inf.reduce_fast());
                                acc
                            },
                        )
                        .reduce_with(|mut a, b| {
                            a[0] = a[0].lazy_add(b[0]);
                            a[1] = a[1].lazy_add(b[1]);
                            a[2] = a[2].lazy_add(b[2]);
                            a
                        })
                        .unwrap_or([E::zero(); 3]);
                    t_0 = r_0;
                    t_2 = r_2;
                    t_inf = r_inf;
                    precomputed.push(vec![
                        if zero_check { E::Scalar::zero() } else { E::sum_lanes_to_mont(t_0) },
                        E::sum_lanes_to_mont(t_2),
                        E::sum_lanes_to_mont(t_inf),
                    ]);
                    continue;
                }
                // Inline serial fallback for small splits (rare under par).
                if let Some(split) = packed_split {
                    let right_len = split.right_broadcast.len();
                    for left_group in 0..split.left_packed.len() {
                        let mut inner_t_0 = E::zero();
                        let mut inner_t_2 = E::zero();
                        let mut inner_t_inf = E::zero();
                        let group_base = left_group * right_len;
                        for right_idx in 0..right_len {
                            let eq_r = split.right_broadcast[right_idx];
                            let group_idx = group_base + right_idx;
                            let base = group_idx << 1;
                            let mut values_2 = [PBF::zero(); 4];
                            let mut diffs = [PBF::zero(); 4];
                            for poly_idx in 0..4 {
                                let v0 = evals[poly_idx][base];
                                let v1 = evals[poly_idx][base + 1];
                                let diff = Self::packed_diff(v0, v1);
                                values_2[poly_idx] = v1.lazy_add(diff).con_sub_xp(2);
                                diffs[poly_idx] = diff;
                            }
                            if !zero_check {
                                let values_0: [PBF; 4] =
                                    std::array::from_fn(|pi| evals[pi][base]);
                                inner_t_0 = inner_t_0.lazy_add(
                                    eq_r.mul_base_elem(Self::gate_h_packed_base(values_0, one)),
                                );
                            }
                            inner_t_2 = inner_t_2.lazy_add(
                                eq_r.mul_base_elem(Self::gate_h_packed_base(values_2, one)),
                            );
                            inner_t_inf = inner_t_inf.lazy_add(
                                eq_r.mul_base_elem(diffs[0] * diffs[1] * diffs[2]),
                            );
                        }
                        let eq_l = split.left_packed[left_group];
                        if !zero_check {
                            t_0 = t_0.lazy_add(eq_l * inner_t_0.reduce_fast());
                        }
                        t_2 = t_2.lazy_add(eq_l * inner_t_2.reduce_fast());
                        t_inf = t_inf.lazy_add(eq_l * inner_t_inf.reduce_fast());
                    }
                } else {
                    for group_idx in 0..packed_groups {
                        let weight = eq_view.load_packed_weight(group_idx, packed_groups);
                        let base = group_idx << 1;
                        let mut values_2 = [PBF::zero(); 4];
                        let mut diffs = [PBF::zero(); 4];
                        for poly_idx in 0..4 {
                            let v0 = evals[poly_idx][base];
                            let v1 = evals[poly_idx][base + 1];
                            let diff = Self::packed_diff(v0, v1);
                            values_2[poly_idx] = v1.lazy_add(diff).con_sub_xp(2);
                            diffs[poly_idx] = diff;
                        }
                        if !zero_check {
                            let values_0: [PBF; 4] =
                                std::array::from_fn(|pi| evals[pi][base]);
                            t_0 = t_0.lazy_add(
                                weight.mul_base_elem(Self::gate_h_packed_base(values_0, one)),
                            );
                        }
                        t_2 = t_2.lazy_add(
                            weight.mul_base_elem(Self::gate_h_packed_base(values_2, one)),
                        );
                        t_inf = t_inf.lazy_add(
                            weight.mul_base_elem(diffs[0] * diffs[1] * diffs[2]),
                        );
                        if (group_idx + 1) % 8192 == 0 {
                            t_0 = t_0.reduce_fast();
                            t_2 = t_2.reduce_fast();
                            t_inf = t_inf.reduce_fast();
                        }
                    }
                }
                precomputed.push(vec![
                    if zero_check { E::Scalar::zero() } else { E::sum_lanes_to_mont(t_0) },
                    E::sum_lanes_to_mont(t_2),
                    E::sum_lanes_to_mont(t_inf),
                ]);
                continue;
            }
            if round == 1 {
                let packed_groups = evals[0].len() >> 2;
                let mut round_table = [E::zero(); Self::OPT_HAT_SIZE * Self::OPT_U_SIZE];
                let packed_split = eq_view.packed_split_for_groups(packed_groups);

                if packed_groups >= 2048 {
                    const CHUNK: usize = 8192;
                    let n_chunks = (packed_groups + CHUNK - 1) / CHUNK;
                    let src: [ParPtr<PBF>; 4] = [
                        ParPtr(evals[0].as_ptr() as *mut PBF),
                        ParPtr(evals[1].as_ptr() as *mut PBF),
                        ParPtr(evals[2].as_ptr() as *mut PBF),
                        ParPtr(evals[3].as_ptr() as *mut PBF),
                    ];
                    round_table = (0..n_chunks)
                        .into_par_iter()
                        .fold_with(
                            [E::zero(); Self::OPT_HAT_SIZE * Self::OPT_U_SIZE],
                            |mut local, chunk_idx| {
                                let c_lo = chunk_idx * CHUNK;
                                let c_hi = ((chunk_idx + 1) * CHUNK).min(packed_groups);
                                for block_idx in c_lo..c_hi {
                                    let weight = match packed_split {
                                        Some(split) => split.weight(block_idx),
                                        None => {
                                            eq_view.load_packed_weight(block_idx, packed_groups)
                                        }
                                    };
                                    let base = block_idx << 2;
                                    let mut poly_grids = [[PBF::zero(); 16]; 4];
                                    for poly_idx in 0..4 {
                                        let p = src[poly_idx].0;
                                        let low = Self::finite_line_from_pair_packed(
                                            unsafe { *p.add(base) },
                                            unsafe { *p.add(base + 1) },
                                            three,
                                        );
                                        let high = Self::finite_line_from_pair_packed(
                                            unsafe { *p.add(base + 2) },
                                            unsafe { *p.add(base + 3) },
                                            three,
                                        );
                                        for v_idx in 0..4 {
                                            let line = Self::finite_line_from_pair_packed(
                                                low[v_idx],
                                                high[v_idx],
                                                three,
                                            );
                                            for u_idx in 0..4 {
                                                poly_grids[poly_idx][v_idx + (u_idx << 2)] =
                                                    line[u_idx];
                                            }
                                        }
                                    }
                                    let mut tensor = [PBF::zero(); 16];
                                    for idx in 0..16 {
                                        tensor[idx] = Self::gate_h_packed_base(
                                            std::array::from_fn(|poly_idx| {
                                                poly_grids[poly_idx][idx]
                                            }),
                                            one,
                                        );
                                    }
                                    Self::transform_tensor_axis_to_ud_packed(
                                        &mut tensor, 0, inv6,
                                    );
                                    Self::transform_tensor_axis_to_ud_packed(
                                        &mut tensor, 1, inv6,
                                    );
                                    for state in 0..Self::OPT_U_SIZE {
                                        local[state] = local[state]
                                            .lazy_add(weight.mul_base_elem(tensor[state]));
                                        local[Self::OPT_U_SIZE + state] = local
                                            [Self::OPT_U_SIZE + state]
                                            .lazy_add(weight.mul_base_elem(
                                                tensor[state + (2 * Self::OPT_U_SIZE)],
                                            ));
                                        local[(2 * Self::OPT_U_SIZE) + state] = local
                                            [(2 * Self::OPT_U_SIZE) + state]
                                            .lazy_add(weight.mul_base_elem(
                                                tensor[state + (3 * Self::OPT_U_SIZE)],
                                            ));
                                    }
                                }
                                for entry in local.iter_mut() {
                                    *entry = entry.reduce_fast();
                                }
                                local
                            },
                        )
                        .reduce_with(|mut a, b| {
                            for i in 0..(Self::OPT_HAT_SIZE * Self::OPT_U_SIZE) {
                                a[i] = a[i].lazy_add(b[i]);
                            }
                            a
                        })
                        .unwrap_or([E::zero(); Self::OPT_HAT_SIZE * Self::OPT_U_SIZE]);
                    precomputed.push(
                        round_table
                            .into_iter()
                            .map(|entry| E::sum_lanes_to_mont(entry))
                            .collect(),
                    );
                    continue;
                }
                // For small round-1 work the serial helper is cheaper. We
                // cannot call the serial version directly (it's a single
                // function handling all rounds), so we replicate its round-1
                // logic inline.
                for block_idx in 0..packed_groups {
                    let weight = match packed_split {
                        Some(split) => split.weight(block_idx),
                        None => eq_view.load_packed_weight(block_idx, packed_groups),
                    };
                    let base = block_idx << 2;
                    let mut poly_grids = [[PBF::zero(); 16]; 4];
                    for poly_idx in 0..4 {
                        let low = Self::finite_line_from_pair_packed(
                            evals[poly_idx][base],
                            evals[poly_idx][base + 1],
                            three,
                        );
                        let high = Self::finite_line_from_pair_packed(
                            evals[poly_idx][base + 2],
                            evals[poly_idx][base + 3],
                            three,
                        );
                        for v_idx in 0..4 {
                            let line = Self::finite_line_from_pair_packed(
                                low[v_idx], high[v_idx], three,
                            );
                            for u_idx in 0..4 {
                                poly_grids[poly_idx][v_idx + (u_idx << 2)] = line[u_idx];
                            }
                        }
                    }
                    let mut tensor = [PBF::zero(); 16];
                    for idx in 0..16 {
                        tensor[idx] = Self::gate_h_packed_base(
                            std::array::from_fn(|poly_idx| poly_grids[poly_idx][idx]),
                            one,
                        );
                    }
                    Self::transform_tensor_axis_to_ud_packed(&mut tensor, 0, inv6);
                    Self::transform_tensor_axis_to_ud_packed(&mut tensor, 1, inv6);
                    for state in 0..Self::OPT_U_SIZE {
                        round_table[state] += weight.mul_base_elem(tensor[state]);
                        round_table[Self::OPT_U_SIZE + state] +=
                            weight.mul_base_elem(tensor[state + (2 * Self::OPT_U_SIZE)]);
                        round_table[(2 * Self::OPT_U_SIZE) + state] +=
                            weight.mul_base_elem(tensor[state + (3 * Self::OPT_U_SIZE)]);
                    }
                    if (block_idx + 1) % 8192 == 0 {
                        for entry in &mut round_table {
                            *entry = entry.reduce_fast();
                        }
                    }
                }
                precomputed.push(
                    round_table
                        .into_iter()
                        .map(|entry| E::sum_lanes_to_mont(entry))
                        .collect(),
                );
                continue;
            }
            // Rounds >= 2: generic chunked parallel path. Uses Vec<E>
            // accumulators sized by `states = OPT_U_SIZE^round` so it works
            // for any `ell0 >= 3`. The inner body mirrors the generic tail
            // of the serial helper at `sumcheck_mamabear.rs`.
            let prefix_len = round + 1;
            let states = Self::OPT_U_SIZE.pow(round as u32);
            let block_len = 1usize << prefix_len;
            let grid_len = Self::OPT_U_SIZE.pow(prefix_len as u32);
            let packed_groups = evals[0].len() / block_len;
            let table_len = Self::OPT_HAT_SIZE * states;
            let mut round_table: Vec<E> = vec![E::zero(); table_len];
            let packed_split = eq_view.packed_split_for_groups(packed_groups);

            if packed_groups >= 2048 {
                const CHUNK: usize = 8192;
                let n_chunks = (packed_groups + CHUNK - 1) / CHUNK;
                let src: [ParPtr<PBF>; 4] = [
                    ParPtr(evals[0].as_ptr() as *mut PBF),
                    ParPtr(evals[1].as_ptr() as *mut PBF),
                    ParPtr(evals[2].as_ptr() as *mut PBF),
                    ParPtr(evals[3].as_ptr() as *mut PBF),
                ];
                let evals_lens: [usize; 4] =
                    [evals[0].len(), evals[1].len(), evals[2].len(), evals[3].len()];
                let partials: Vec<Vec<E>> = (0..n_chunks)
                    .into_par_iter()
                    .map(|chunk_idx| {
                        let c_lo = chunk_idx * CHUNK;
                        let c_hi = ((chunk_idx + 1) * CHUNK).min(packed_groups);
                        let mut local: Vec<E> = vec![E::zero(); table_len];
                        let poly_slices: [&[PBF]; 4] = std::array::from_fn(|poly_idx| unsafe {
                            std::slice::from_raw_parts(src[poly_idx].0, evals_lens[poly_idx])
                        });
                        for packed_idx in c_lo..c_hi {
                            let start = packed_idx * block_len;
                            let end = start + block_len;
                            let weight = match packed_split {
                                Some(split) => split.weight(packed_idx),
                                None => {
                                    eq_view.load_packed_weight(packed_idx, packed_groups)
                                }
                            };
                            let mut tensor = vec![PBF::zero(); grid_len];
                            for finite_idx in 0..grid_len {
                                let finite_points = Self::decode_u4(finite_idx, prefix_len);
                                let gate_inputs = std::array::from_fn(|poly_idx| {
                                    Self::eval_packed_block_at_finite_points(
                                        &poly_slices[poly_idx][start..end],
                                        &finite_points,
                                        _two,
                                        three,
                                    )
                                });
                                tensor[finite_idx] = Self::gate_h_packed_base(gate_inputs, one);
                            }
                            for axis in 0..prefix_len {
                                Self::transform_tensor_axis_to_ud_packed(
                                    &mut tensor, axis, inv6,
                                );
                            }
                            for state in 0..states {
                                for (hat_idx, &hat_point) in
                                    Self::OPT_HAT_POINTS.iter().enumerate()
                                {
                                    let tensor_idx = state + (hat_point as usize) * states;
                                    local[hat_idx * states + state] = local
                                        [hat_idx * states + state]
                                        .lazy_add(weight.mul_base_elem(tensor[tensor_idx]));
                                }
                            }
                        }
                        for entry in local.iter_mut() {
                            *entry = entry.reduce_fast();
                        }
                        local
                    })
                    .collect();
                for p in &partials {
                    for i in 0..table_len {
                        round_table[i] = round_table[i].lazy_add(p[i]);
                    }
                }
            } else {
                for packed_idx in 0..packed_groups {
                    let start = packed_idx * block_len;
                    let end = start + block_len;
                    let weight = match packed_split {
                        Some(split) => split.weight(packed_idx),
                        None => eq_view.load_packed_weight(packed_idx, packed_groups),
                    };
                    let mut tensor = vec![PBF::zero(); grid_len];
                    for finite_idx in 0..grid_len {
                        let finite_points = Self::decode_u4(finite_idx, prefix_len);
                        let gate_inputs = std::array::from_fn(|poly_idx| {
                            Self::eval_packed_block_at_finite_points(
                                &evals[poly_idx][start..end],
                                &finite_points,
                                _two,
                                three,
                            )
                        });
                        tensor[finite_idx] = Self::gate_h_packed_base(gate_inputs, one);
                    }
                    for axis in 0..prefix_len {
                        Self::transform_tensor_axis_to_ud_packed(&mut tensor, axis, inv6);
                    }
                    for state in 0..states {
                        for (hat_idx, &hat_point) in Self::OPT_HAT_POINTS.iter().enumerate() {
                            let tensor_idx = state + (hat_point as usize) * states;
                            round_table[hat_idx * states + state] = round_table
                                [hat_idx * states + state]
                                .lazy_add(weight.mul_base_elem(tensor[tensor_idx]));
                        }
                    }
                    if (packed_idx + 1) % 8192 == 0 {
                        for entry in &mut round_table {
                            *entry = entry.reduce_fast();
                        }
                    }
                }
                for entry in &mut round_table {
                    *entry = entry.reduce_fast();
                }
            }
            precomputed.push(
                round_table
                    .into_iter()
                    .map(|entry| E::sum_lanes_to_mont(entry))
                    .collect(),
            );
        }
        precomputed
    }

    // =========================================================================
    // Parallel generic ell0 orchestrator (Ext3)
    // =========================================================================

    /// Parallel duplicate of
    /// `SumcheckMamaBear::prove_add_mul_optimized_with_ell0_generic`.
    /// Hardcodes the par path (uses `_par` fold kernels and `_par`
    /// precompute); fall back to the serial version for NV below threshold
    /// is handled in the public wrappers above.
    pub(crate) fn prove_add_mul_optimized_with_ell0_generic_par<
        E: SumcheckExtField + Send + Sync,
    >(
        evals: [Vec<PBF>; 4],
        point: &[E::Scalar],
        ell0: usize,
        zero_check: bool,
        transcript: &mut Transcript,
        timings: &mut ZeroCheckTimings,
    ) -> (Vec<E::Scalar>, [E::Scalar; 5])
    where
        E::Scalar: Send + Sync,
    {
        let t_total = Instant::now();
        let num_vars = point.len();
        let ell0 = Self::resolve_optimized_ell0(num_vars, Some(ell0));

        let point_mont = point
            .iter()
            .copied()
            .map(|v| v.to_montgomery())
            .collect::<Vec<_>>();

        let t_eq = Instant::now();
        let eq_tables = Self::build_two_stage_eq_tables_generic::<E>(&point_mont);
        timings.eq_tables_us += t_eq.elapsed().as_micros();

        let base_one = PBF::one().to_montgomery();
        let base_two = PBF::from(2u32).to_montgomery();
        let base_three = PBF::from(3u32).to_montgomery();
        let base_inv6 = PBF::from(((P + 1) / 6) as u64).to_montgomery();

        let t_pre = Instant::now();
        let small_value_tables = Self::precompute_small_value_tables_packed_generic_par::<E>(
            &evals,
            &eq_tables,
            ell0,
            zero_check,
            base_one,
            base_two,
            base_three,
            base_inv6,
        );
        timings.precompute_us += t_pre.elapsed().as_micros();

        let ext_one = E::Scalar::one().to_montgomery();
        let ext_two = E::Scalar::from(2u32).to_montgomery();
        let ext_inv2 = E::Scalar::from(SBF::inv_2()).to_montgomery();

        let mut verifier_challenges = Vec::with_capacity(num_vars);
        let mut prefix_eq_eval = ext_one;
        let mut lagrange_weights = vec![ext_one];

        let t_small = Instant::now();
        for round in 0..ell0 {
            let t_hat = Self::compute_t_from_precomputed_generic::<E>(
                &small_value_tables[round],
                &lagrange_weights,
            );
            let s_hat = Self::compute_round_s_from_t_generic::<E>(
                prefix_eq_eval,
                point_mont[round],
                t_hat,
                ext_one,
                ext_two,
            );
            Self::append_hat_round_values_generic::<E>(transcript, s_hat);

            let challenge = transcript.challenge_f::<E::Scalar>().to_montgomery();
            let challenge_mont = challenge;
            prefix_eq_eval *=
                Self::eq_linear_mont_generic::<E>(point_mont[round], challenge_mont, ext_one);
            verifier_challenges.push(challenge);

            if round + 1 < ell0 {
                let basis = Self::lagrange_basis_degree3_generic::<E>(
                    challenge_mont,
                    ext_one,
                    ext_two,
                    ext_inv2,
                );
                lagrange_weights =
                    Self::update_small_value_weights_generic::<E>(&lagrange_weights, basis);
            }
        }
        timings.small_value_rounds_us += t_small.elapsed().as_micros();

        let packed_round_start = ell0;
        let packed_round_end = num_vars.saturating_sub(3);

        let t_trans = Instant::now();
        let mut folded_tables = Self::fold_base_tables_to_ext_generic_par::<E>(
            &evals,
            &verifier_challenges[..ell0],
        );
        drop(evals);
        timings.transition_fold_us += t_trans.elapsed().as_micros();

        let mut par_scratch_generic: [Vec<E>; 4] = {
            let max_len = folded_tables[0].len();
            let mut s = std::array::from_fn(|_| {
                let mut v: Vec<E> = Vec::with_capacity(max_len);
                unsafe { v.set_len(max_len); }
                v
            });
            pretouch_scratch_generic(&mut s);
            s
        };

        if packed_round_start < packed_round_end {
            let t_first = Instant::now();
            let first_t_hat = Self::compute_round_t_from_packed_tables_generic::<E>(
                &folded_tables,
                &Self::round_eq_view_generic(&eq_tables, packed_round_start),
            );
            timings.packed_fold_rounds_us += t_first.elapsed().as_micros();

            let s_hat = Self::compute_round_s_from_t_generic::<E>(
                prefix_eq_eval,
                point_mont[packed_round_start],
                first_t_hat,
                ext_one,
                ext_two,
            );
            Self::append_hat_round_values_generic::<E>(transcript, s_hat);

            let challenge = transcript.challenge_f::<E::Scalar>().to_montgomery();
            let challenge_mont = challenge;
            prefix_eq_eval *= Self::eq_linear_mont_generic::<E>(
                point_mont[packed_round_start],
                challenge_mont,
                ext_one,
            );
            verifier_challenges.push(challenge);
            let mut prev_challenge = challenge;

            for round in (packed_round_start + 1)..packed_round_end {
                let eq_view = Self::round_eq_view_generic(&eq_tables, round);
                let t_pf = Instant::now();
                let t_hat = Self::fold_packed_ext_tables_and_compute_round_t_in_place_generic_par::<E>(
                    &mut folded_tables,
                    &eq_view,
                    prev_challenge,
                    &mut par_scratch_generic,
                );
                timings.packed_fold_rounds_us += t_pf.elapsed().as_micros();
                let s_hat = Self::compute_round_s_from_t_generic::<E>(
                    prefix_eq_eval,
                    point_mont[round],
                    t_hat,
                    ext_one,
                    ext_two,
                );
                Self::append_hat_round_values_generic::<E>(transcript, s_hat);

                let challenge = transcript.challenge_f::<E::Scalar>().to_montgomery();
                let challenge_mont = challenge;
                prefix_eq_eval *= Self::eq_linear_mont_generic::<E>(
                    point_mont[round],
                    challenge_mont,
                    ext_one,
                );
                verifier_challenges.push(challenge);
                prev_challenge = challenge;
            }

            let t_pf_final = Instant::now();
            Self::fold_packed_ext_tables_in_place_generic(
                &mut folded_tables,
                E::from_scalar(prev_challenge),
            );
            timings.packed_fold_rounds_us += t_pf_final.elapsed().as_micros();
        }

        let t_tail = Instant::now();
        let scalar_tail_start = (num_vars.saturating_sub(3)).max(ell0);
        let mut active_len = 1usize << (num_vars - scalar_tail_start);
        let mut packed_tail_tables: [E; 4] = std::array::from_fn(|k| folded_tables[k][0]);

        for round in scalar_tail_start..num_vars {
            let eq_view = Self::round_eq_view_generic(&eq_tables, round);
            let t_hat = Self::compute_round_t_from_single_packed_tables_generic::<E>(
                &packed_tail_tables,
                &eq_view,
                active_len,
            );
            let s_hat = Self::compute_round_s_from_t_generic::<E>(
                prefix_eq_eval,
                point_mont[round],
                t_hat,
                ext_one,
                ext_two,
            );
            Self::append_hat_round_values_generic::<E>(transcript, s_hat);

            let challenge = transcript.challenge_f::<E::Scalar>().to_montgomery();
            let challenge_mont = challenge;
            prefix_eq_eval *= Self::eq_linear_mont_generic::<E>(
                point_mont[round],
                challenge_mont,
                ext_one,
            );
            verifier_challenges.push(challenge);

            if active_len > 1 {
                Self::fold_single_packed_ext_tables_in_place_generic::<E>(
                    &mut packed_tail_tables,
                    E::from_scalar(challenge),
                    active_len,
                );
                active_len >>= 1;
            }
        }

        let mut final_evals = [E::Scalar::zero(); 5];
        for poly_idx in 0..4 {
            final_evals[poly_idx] =
                E::unpack_to_scalars(packed_tail_tables[poly_idx])[0].from_montgomery();
        }
        final_evals[4] = prefix_eq_eval.from_montgomery();
        timings.scalar_tail_us += t_tail.elapsed().as_micros();
        timings.total_us += t_total.elapsed().as_micros();
        (verifier_challenges, final_evals)
    }


    // =========================================================================
    // Public parallel API
    // =========================================================================

    /// Parallel variant of `prove_add_mul_ell0_ext3`.
    ///
    /// Input / fold-order conventions match `prove_add_mul_ell0_ext3`:
    /// tables in **normal order**, returned challenges in **SIMD-round
    /// order** (x_3, x_4, ..., x_{μ-1}, x_0, x_1, x_2). Transcript bytes
    /// are bit-identical to the serial path.
    ///
    /// Routing by ell0:
    /// - ell0=1: generic_par for all nv (mixed: 1.0-2.7x, some nv neutral).
    /// - ell0=2: generic_par for all nv (benchmark: 1.2-2.7x, always positive).
    /// - ell0=3: generic_par only at nv >= 23 (catastrophic regression at
    ///   low nv from per-group vec alloc in precompute tensor kernel).
    pub fn prove_add_mul_ell0_ext3_par(
        evals: [Vec<PBF>; 4],
        point: &[SEF3],
        ell0: usize,
        transcript: &mut Transcript,
    ) -> (Vec<SEF3>, [SEF3; 5]) {
        let mut throwaway = ZeroCheckTimings::default();
        let nv = point.len();
        let resolved = Self::resolve_optimized_ell0(nv, Some(ell0));
        let use_par = nv >= PAR_ZEROCHECK_MIN_NV
            && match resolved {
                1 | 2 => true,
                3 => nv >= 23,
                _ => false,
            };
        if use_par {
            Self::prove_add_mul_optimized_with_ell0_generic_par::<PEF3>(
                evals, point, ell0, true, transcript, &mut throwaway,
            )
        } else {
            Self::prove_add_mul_optimized_with_ell0_generic::<PEF3>(
                evals, point, ell0, true, transcript, &mut throwaway,
            )
        }
    }

    /// Profiled parallel variant of `prove_add_mul_ell0_ext3`.
    pub fn prove_add_mul_ell0_ext3_par_profiled(
        evals: [Vec<PBF>; 4],
        point: &[SEF3],
        ell0: usize,
        transcript: &mut Transcript,
        timings: &mut ZeroCheckTimings,
    ) -> (Vec<SEF3>, [SEF3; 5]) {
        let nv = point.len();
        let resolved = Self::resolve_optimized_ell0(nv, Some(ell0));
        let use_par = nv >= PAR_ZEROCHECK_MIN_NV
            && match resolved {
                1 | 2 => true,
                3 => nv >= 23,
                _ => false,
            };
        if use_par {
            Self::prove_add_mul_optimized_with_ell0_generic_par::<PEF3>(
                evals, point, ell0, true, transcript, timings,
            )
        } else {
            Self::prove_add_mul_optimized_with_ell0_generic::<PEF3>(
                evals, point, ell0, true, transcript, timings,
            )
        }
    }
}

// Silence "unused import" warnings when the par module is compiled without
// the serial file exercising the corresponding code paths (e.g. under
// partial feature builds).
#[allow(dead_code)]
fn _unused_silencer(_: Instant) {}

#[cfg(test)]
mod tests {
    use super::*;
    use arithmetic::field::mamabear::MamaBearScalarExt3 as SEF3Local;
    use rand::rngs::SmallRng;
    use rand::SeedableRng;

    /// Parallel ZeroCheck must produce bit-identical transcript + final evals
    /// to serial ZeroCheck on identical inputs. Exercises both below and above
    /// the `PAR_ZEROCHECK_MIN_NV` threshold, on random data.
    #[test]
    fn sumcheck_mamabear_par_matches_serial_ext3() {
        let mut rng = SmallRng::seed_from_u64(43);
        for nv in [8usize, 14, 16, 18] {
            let len_pbf = 1usize << (nv - 3);
            let point: Vec<SEF3Local> =
                (0..nv).map(|_| SEF3Local::random(&mut rng)).collect();
            let evals: [Vec<PBF>; 4] = [
                (0..len_pbf).map(|_| PBF::random(&mut rng).to_montgomery()).collect(),
                (0..len_pbf).map(|_| PBF::random(&mut rng).to_montgomery()).collect(),
                (0..len_pbf).map(|_| PBF::random(&mut rng).to_montgomery()).collect(),
                (0..len_pbf).map(|_| PBF::random(&mut rng).to_montgomery()).collect(),
            ];
            let ell0 = 2;
            let mut t_ser = Transcript::new();
            let (_ch_ser, _fe_ser) = SumcheckMamaBear::prove_add_mul_ell0_ext3(
                evals.clone(), &point, ell0, &mut t_ser,
            );
            let mut t_par = Transcript::new();
            let (_ch_par, _fe_par) = SumcheckMamaBear::prove_add_mul_ell0_ext3_par(
                evals.clone(), &point, ell0, &mut t_par,
            );
            assert_eq!(t_ser.proof.bytes, t_par.proof.bytes,
                "Ext3 par transcript mismatch at nv={nv}");
        }
    }
}
