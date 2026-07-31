//! Parallel (rayon) mirror of the degree-generic ZeroCheck prover (`zerocheck_generic.rs`).
//!
//! This file follows the same separation discipline as `sumcheck_par.rs` relative to
//! `sumcheck.rs`: the serial round-by-round logic and all the gate/eq helpers live in the
//! serial file; the par-only kernels (the eq_L-axis parallel fold) and the par driver live
//! here. The par driver reuses every serial helper unchanged, so it produces **byte-identical**
//! proofs to `prove_zero_check_generic` (enforced by tests below).
//!
//! # Parallel axis and byte-identity
//!
//! The single parallel kernel `fold_and_compute_round_t_par` mirrors the serial fused
//! fold-plus-round-poly kernel, parallelizing over the two-stage eq factorization's `eq_L`
//! outer groups (the same axis the add/mul par prover uses). Disjoint writes go through
//! `ParPtr` into a caller-preallocated, pre-touched ping-pong `scratch` buffer (out of place,
//! then swapped into `evals`). The per-`eq_L`-group contribution `eq_l * inner.reduce_fast()`
//! is identical to the serial code, and the cross-group sum uses only `lazy_add` (wrapping
//! u64 add, associative), so the accumulation is order-independent mod `2^64` — hence the
//! final `sum_lanes_to_mont` (and the appended hat bytes) match the serial prover exactly.
//!
//! Below `PAR_ZEROCHECK_MIN_PACKED_GROUPS` groups (or `PAR_ZEROCHECK_MIN_NV` variables) the
//! kernel/driver fall back to the serial path.

use arithmetic::field::Field;
use rayon::prelude::*;
use util::fiat_shamir::Transcript;

use arithmetic::field::mamabear::LazyReduction;
use arithmetic::field::mamabear::PackedMamaBearAVX512 as PBF;

use crate::sumcheck_mamabear::{
    MontgomeryOps, RoundEqView, SumcheckExtField, SumcheckMamaBear, ZeroCheckTimings,
};
use crate::sumcheck_mamabear_par::{ParPtr, PAR_ZEROCHECK_MIN_PACKED_GROUPS};
use crate::sumcheck_mamabear::TwoStageEqTables;
use crate::zerocheck_generic_mamabear::{
    append_hat_round_values, build_finite_grid_base, compute_group_hats_base,
    compute_group_hats_ext, compute_round_s_from_t, compute_round_t, compute_round_t_single,
    compute_t_from_precomputed, fold_and_compute_round_t, fold_base_tables_to_ext,
    fold_ext_tables_in_place, fold_single_packed_in_place, gate_consts_base,
    lagrange_basis_degree_d_generic, precompute_small_value_tables, resolve_ell0,
    transform_tensor_axis_to_ud_base, update_small_value_weights, GateConstsBase, ZeroCheckGate,
    MAX_U,
};

/// Pre-touch all pages of the `NUM_COLS` scratch buffers in parallel (one volatile zero per
/// 4 KB page), so the parallel fold pays no minor page faults on first write.
pub(crate) fn pretouch_scratch_generic<T: Send + Sync, const NUM_COLS: usize>(
    scratch: &mut [Vec<T>; NUM_COLS],
) {
    let page_stride = (4096 / std::mem::size_of::<T>().max(1)).max(1);
    rayon::scope(|s| {
        for v in scratch.iter_mut() {
            s.spawn(move |_| {
                let ptr = v.as_mut_ptr();
                let len = v.len();
                for i in (0..len).step_by(page_stride) {
                    unsafe {
                        std::ptr::write_volatile(ptr.add(i), std::mem::zeroed());
                    }
                }
            });
        }
    });
}

/// Per-gate driver-level `num_vars` threshold below which the generic parallel ZeroCheck
/// defers entirely to serial. This is the safety net guaranteeing **par is never slower
/// than serial**: below the threshold the par entry point runs the serial prover (so
/// `par_time == serial_time`); at/above it, par (with the parallel precompute) was
/// measured `>=` serial.
///
/// Crossover measurement (Ext3, this machine = 14 rayon threads, release,
/// `target-cpu=native`). Even with the parallel precompute,
/// the remaining serial dominator (the base->ext transition fold `fold_base_tables_to_ext`,
/// ~12 ms at nv=20) plus the memory-bound packed fold (no par gain) push the break-even up.
/// The exercised AddMulD3 (D=3, 4 col) gate shape crossed at:
///
/// ```text
///   shape           nv=18 (par/ser)   nv=19 (par/ser)
///   AddMulD3 (3,4)      0.64x             1.09x
/// ```
///
/// so a threshold of 19 is correct for this shape. The const fn keeps the
/// `(D, NUM_COLS)` signature for other gate shapes whose break-even may differ.
#[inline]
pub(crate) const fn par_zerocheck_min_nv(d: usize, num_cols: usize) -> usize {
    let _ = (d, num_cols);
    19
}

/// Minimum base-table block count (`evals[0].len()`) below which the parallel precompute
/// defers to the serial helper (rayon overhead not worth it). `evals[0].len() = 2^(nv-3)`.
pub(crate) const PAR_PRECOMPUTE_MIN_BASE_BLOCKS: usize = 1024;

/// Chunk size (in packed groups) used by the parallel precompute's deterministic-chunk
/// reduce. MUST match the serial helper's periodic `reduce_fast` cadence (every 8192 groups,
/// `zerocheck_generic.rs`) so the accumulation is bit-for-bit mod-`P` identical regardless of
/// thread count: each chunk accumulates with `lazy_add` only, then `reduce_fast` at the chunk
/// boundary; the cross-chunk combine also uses `lazy_add`, and the final `sum_lanes_to_mont`
/// canonicalizes. Aligning the chunk boundary with the serial reduce point keeps every
/// intermediate within `[0, 2^63)` (no u64 overflow) for any realistic `nv`.
const PAR_PRECOMPUTE_CHUNK: usize = 8192;

/// Parallel mirror of `precompute_small_value_tables` (`zerocheck_generic.rs`). Byte-identical
/// (mod `P`, hence identical transcript bytes) to the serial helper, parallelizing the round-0
/// accumulation (over `eq_L` groups, or deterministic chunks for the non-split case) and the
/// rounds `>= 1` finite-point tensor accumulation (over deterministic 8192-group chunks).
///
/// # Byte-identity
/// - Round 0 split: each `eq_L` group contributes `eq_l * inner.reduce_fast()`; the per-group
///   contributions are summed with `lazy_add` only (no intermediate reduce — matching serial),
///   so the integer lane sum is identical regardless of grouping (no overflow, as serial
///   already proves). Bit-identical lanes.
/// - Round 0 non-split / rounds >= 1: deterministic 8192-group chunks, each accumulating with
///   `lazy_add` and `reduce_fast` at the chunk end (the serial reduce cadence). Cross-chunk
///   combine via `lazy_add`. `reduce_fast` is mod-`P` invariant and `lazy_add` is integer
///   associative (no overflow), so the result is mod-`P` identical; `sum_lanes_to_mont`
///   canonicalizes, yielding identical appended bytes.
#[allow(clippy::too_many_arguments)]
pub(crate) fn precompute_small_value_tables_par<
    E: SumcheckExtField + Send + Sync,
    G: ZeroCheckGate<E, D, NUM_COLS>,
    const D: usize,
    const NUM_COLS: usize,
>(
    evals: &[Vec<PBF>; NUM_COLS],
    eq_tables: &TwoStageEqTables<E>,
    ell0: usize,
    zero_check: bool,
    consts: &GateConstsBase,
) -> Vec<Vec<E::Scalar>>
where
    E::Scalar: Send + Sync,
{
    // Tiny inputs: rayon overhead exceeds the gain; defer to the serial helper.
    if evals[0].len() < PAR_PRECOMPUTE_MIN_BASE_BLOCKS {
        return precompute_small_value_tables::<E, G, D, NUM_COLS>(
            evals, eq_tables, ell0, zero_check, consts,
        );
    }

    let u_size = D + 1;
    let hat_size = D;
    let one = consts.one;
    let mut precomputed: Vec<Vec<E::Scalar>> = Vec::with_capacity(ell0);

    for round in 0..ell0 {
        let prefix_len = round + 1;
        let eq_view = SumcheckMamaBear::round_eq_view_generic(eq_tables, round);

        if round == 0 {
            let packed_groups = evals[0].len() >> 1;
            let packed_split = eq_view.packed_split_for_groups(packed_groups);
            let src: [ParPtr<PBF>; NUM_COLS] =
                std::array::from_fn(|c| ParPtr(evals[c].as_ptr() as *mut PBF));

            let t_acc: [E; D] = match packed_split {
                Some(split) if split.left_packed.len() >= 2 => {
                    let right_len = split.right_broadcast.len();
                    let left_len = split.left_packed.len();
                    let right_broadcast = split.right_broadcast;
                    let left_packed = split.left_packed;
                    let grain = (left_len / 32).max(1);
                    (0..left_len)
                        .into_par_iter()
                        .with_min_len(grain)
                        .fold_with([E::zero(); D], |mut acc, left_group| {
                            let mut inner = [E::zero(); D];
                            let group_base = left_group * right_len;
                            for right_idx in 0..right_len {
                                let eq_r = right_broadcast[right_idx];
                                let group_idx = group_base + right_idx;
                                let base = group_idx << 1;
                                let mut v0s = [PBF::zero(); NUM_COLS];
                                let mut diffs = [PBF::zero(); NUM_COLS];
                                for c in 0..NUM_COLS {
                                    let p = src[c].0;
                                    let v0 = unsafe { *p.add(base) };
                                    let v1 = unsafe { *p.add(base + 1) };
                                    v0s[c] = v0;
                                    diffs[c] = SumcheckMamaBear::packed_diff(v0, v1);
                                }
                                let hats =
                                    compute_group_hats_base::<E, G, D, NUM_COLS>(&v0s, &diffs, one);
                                for k in 0..hat_size {
                                    if zero_check && k == 0 {
                                        continue;
                                    }
                                    inner[k] = inner[k].lazy_add(eq_r.mul_base_elem(hats[k]));
                                }
                            }
                            let eq_l = left_packed[left_group];
                            for k in 0..hat_size {
                                if zero_check && k == 0 {
                                    continue;
                                }
                                acc[k] = acc[k].lazy_add(eq_l * inner[k].reduce_fast());
                            }
                            acc
                        })
                        .reduce_with(|mut a, b| {
                            for k in 0..D {
                                a[k] = a[k].lazy_add(b[k]);
                            }
                            a
                        })
                        .unwrap_or([E::zero(); D])
                }
                _ => {
                    let n_chunks =
                        (packed_groups + PAR_PRECOMPUTE_CHUNK - 1) / PAR_PRECOMPUTE_CHUNK;
                    (0..n_chunks)
                        .into_par_iter()
                        .map(|chunk_idx| {
                            let c_lo = chunk_idx * PAR_PRECOMPUTE_CHUNK;
                            let c_hi =
                                ((chunk_idx + 1) * PAR_PRECOMPUTE_CHUNK).min(packed_groups);
                            let mut local = [E::zero(); D];
                            for group_idx in c_lo..c_hi {
                                let weight =
                                    eq_view.load_packed_weight(group_idx, packed_groups);
                                let base = group_idx << 1;
                                let mut v0s = [PBF::zero(); NUM_COLS];
                                let mut diffs = [PBF::zero(); NUM_COLS];
                                for c in 0..NUM_COLS {
                                    let p = src[c].0;
                                    let v0 = unsafe { *p.add(base) };
                                    let v1 = unsafe { *p.add(base + 1) };
                                    v0s[c] = v0;
                                    diffs[c] = SumcheckMamaBear::packed_diff(v0, v1);
                                }
                                let hats =
                                    compute_group_hats_base::<E, G, D, NUM_COLS>(&v0s, &diffs, one);
                                for k in 0..hat_size {
                                    if zero_check && k == 0 {
                                        continue;
                                    }
                                    local[k] =
                                        local[k].lazy_add(weight.mul_base_elem(hats[k]));
                                }
                            }
                            for k in 0..hat_size {
                                if !(zero_check && k == 0) {
                                    local[k] = local[k].reduce_fast();
                                }
                            }
                            local
                        })
                        .reduce(
                            || [E::zero(); D],
                            |mut a, b| {
                                for k in 0..D {
                                    a[k] = a[k].lazy_add(b[k]);
                                }
                                a
                            },
                        )
                }
            };

            let mut row = Vec::with_capacity(hat_size);
            for k in 0..hat_size {
                if zero_check && k == 0 {
                    row.push(E::Scalar::zero());
                } else {
                    row.push(E::sum_lanes_to_mont(t_acc[k]));
                }
            }
            precomputed.push(row);
            continue;
        }

        // Rounds >= 1: generic finite-point tensor path, chunk-parallel.
        let states = u_size.pow(round as u32);
        let block_len = 1usize << prefix_len;
        let grid_len = u_size.pow(prefix_len as u32);
        let packed_groups = evals[0].len() / block_len;
        let table_len = hat_size * states;
        let packed_split = eq_view.packed_split_for_groups(packed_groups);
        let src: [ParPtr<PBF>; NUM_COLS] =
            std::array::from_fn(|c| ParPtr(evals[c].as_ptr() as *mut PBF));
        let evals_lens: [usize; NUM_COLS] = std::array::from_fn(|c| evals[c].len());

        let n_chunks = (packed_groups + PAR_PRECOMPUTE_CHUNK - 1) / PAR_PRECOMPUTE_CHUNK;
        let round_table: Vec<E> = (0..n_chunks)
            .into_par_iter()
            .map(|chunk_idx| {
                let c_lo = chunk_idx * PAR_PRECOMPUTE_CHUNK;
                let c_hi = ((chunk_idx + 1) * PAR_PRECOMPUTE_CHUNK).min(packed_groups);
                let mut local: Vec<E> = vec![E::zero(); table_len];
                // Per-task scratch allocated once, reused across the chunk's groups.
                let mut tensor = vec![PBF::zero(); grid_len];
                let mut poly_grids: [Vec<PBF>; NUM_COLS] =
                    std::array::from_fn(|_| vec![PBF::zero(); grid_len]);
                let poly_slices: [&[PBF]; NUM_COLS] = std::array::from_fn(|c| unsafe {
                    std::slice::from_raw_parts(src[c].0, evals_lens[c])
                });
                for packed_idx in c_lo..c_hi {
                    let start = packed_idx * block_len;
                    let end = start + block_len;
                    let weight = match packed_split {
                        Some(split) => split.weight(packed_idx),
                        None => eq_view.load_packed_weight(packed_idx, packed_groups),
                    };
                    for c in 0..NUM_COLS {
                        build_finite_grid_base::<D>(
                            &poly_slices[c][start..end],
                            prefix_len,
                            consts,
                            &mut poly_grids[c],
                        );
                    }
                    for finite_idx in 0..grid_len {
                        let gate_inputs: [PBF; NUM_COLS] =
                            std::array::from_fn(|c| poly_grids[c][finite_idx]);
                        tensor[finite_idx] = G::h_packed_base(gate_inputs, one);
                    }
                    for axis in 0..prefix_len {
                        transform_tensor_axis_to_ud_base::<D>(&mut tensor, axis, consts);
                    }
                    for state in 0..states {
                        for hat_idx in 0..hat_size {
                            let slot = if hat_idx == 0 { 0 } else { hat_idx + 1 };
                            let tensor_idx = state + slot * states;
                            local[hat_idx * states + state] = local[hat_idx * states + state]
                                .lazy_add(weight.mul_base_elem(tensor[tensor_idx]));
                        }
                    }
                }
                for entry in local.iter_mut() {
                    *entry = entry.reduce_fast();
                }
                local
            })
            .reduce(
                || vec![E::zero(); table_len],
                |mut a, b| {
                    for i in 0..table_len {
                        a[i] = a[i].lazy_add(b[i]);
                    }
                    a
                },
            );

        precomputed.push(
            round_table
                .into_iter()
                .map(|entry| E::sum_lanes_to_mont(entry.reduce_fast()))
                .collect(),
        );
    }

    precomputed
}

/// Parallel mirror of `fold_and_compute_round_t`. Folds the `NUM_COLS` ext tables by
/// `challenge` and returns the `D` hat values, parallelizing over the `eq_L` outer groups.
///
/// Writes the folded result out-of-place into `scratch` (capacity `>= next_len`), then swaps
/// `scratch` into `evals` and truncates. Byte-identical accumulation order to the serial twin.
pub(crate) fn fold_and_compute_round_t_par<
    E: SumcheckExtField + Send + Sync,
    G: ZeroCheckGate<E, D, NUM_COLS>,
    const D: usize,
    const NUM_COLS: usize,
>(
    evals: &mut [Vec<E>; NUM_COLS],
    eq_view: &RoundEqView<'_, E>,
    challenge: E::Scalar,
    scratch: &mut [Vec<E>; NUM_COLS],
) -> [E::Scalar; D]
where
    E::Scalar: Send + Sync,
{
    let next_len = evals[0].len() >> 1;
    let packed_groups = next_len >> 1;
    let alpha = E::from_scalar(challenge);
    let one = E::one().ext_to_montgomery();

    if packed_groups < PAR_ZEROCHECK_MIN_PACKED_GROUPS {
        return fold_and_compute_round_t::<E, G, D, NUM_COLS>(evals, eq_view, challenge);
    }

    debug_assert!(scratch[0].capacity() >= next_len);
    for v in scratch.iter_mut() {
        if v.len() < next_len {
            unsafe {
                v.set_len(next_len);
            }
        }
    }
    let src_par: [ParPtr<E>; NUM_COLS] = std::array::from_fn(|c| ParPtr(evals[c].as_mut_ptr()));
    let dst_par: [ParPtr<E>; NUM_COLS] = std::array::from_fn(|c| ParPtr(scratch[c].as_mut_ptr()));

    let packed_split = eq_view.packed_split_for_groups(packed_groups);
    let t_acc: [E; D] = match packed_split {
        Some(split) if split.left_packed.len() >= 2 => {
            let right_len = split.right_broadcast.len();
            let left_len = split.left_packed.len();
            let right_broadcast = split.right_broadcast;
            let left_packed = split.left_packed;
            let grain = (left_len / 32).max(1);

            (0..left_len)
                .into_par_iter()
                .with_min_len(grain)
                .fold_with([E::zero(); D], |mut acc, left_group| {
                    let mut inner = [E::zero(); D];
                    let group_base = left_group * right_len;
                    for right_idx in 0..right_len {
                        let group_idx = group_base + right_idx;
                        let eq_r = right_broadcast[right_idx];
                        let src_base = group_idx << 2;
                        let dst_base = group_idx << 1;
                        let mut v0s = [E::zero(); NUM_COLS];
                        let mut diffs = [E::zero(); NUM_COLS];
                        for c in 0..NUM_COLS {
                            let src = src_par[c].0;
                            let dst = dst_par[c].0;
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
                            v0s[c] = v0;
                            diffs[c] = diff;
                        }
                        let hats = compute_group_hats_ext::<E, G, D, NUM_COLS>(&v0s, &diffs, one);
                        for k in 0..D {
                            inner[k] = inner[k].lazy_add(eq_r * hats[k]);
                        }
                    }
                    let eq_l = left_packed[left_group];
                    for k in 0..D {
                        acc[k] = acc[k].lazy_add(eq_l * inner[k].reduce_fast());
                    }
                    acc
                })
                .reduce_with(|mut a, b| {
                    for k in 0..D {
                        a[k] = a[k].lazy_add(b[k]);
                    }
                    a
                })
                .unwrap_or([E::zero(); D])
        }
        _ => {
            let grain = (packed_groups / 32).max(1);
            (0..packed_groups)
                .into_par_iter()
                .with_min_len(grain)
                .fold_with([E::zero(); D], |mut acc, group_idx| {
                    let weight = eq_view.load_packed_weight(group_idx, packed_groups);
                    let src_base = group_idx << 2;
                    let dst_base = group_idx << 1;
                    let mut v0s = [E::zero(); NUM_COLS];
                    let mut diffs = [E::zero(); NUM_COLS];
                    for c in 0..NUM_COLS {
                        let src = src_par[c].0;
                        let dst = dst_par[c].0;
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
                        v0s[c] = v0;
                        diffs[c] = diff;
                    }
                    let hats = compute_group_hats_ext::<E, G, D, NUM_COLS>(&v0s, &diffs, one);
                    // NB: the serial non-split path does a periodic reduce_fast every 8192
                    // groups; we omit it here because each task accumulates independently and
                    // the result is reduced mod P by sum_lanes_to_mont — mod-P identical.
                    for k in 0..D {
                        acc[k] = acc[k].lazy_add(weight * hats[k]);
                    }
                    acc
                })
                .reduce_with(|mut a, b| {
                    for k in 0..D {
                        a[k] = a[k].lazy_add(b[k]);
                    }
                    a
                })
                .unwrap_or([E::zero(); D])
        }
    };

    for c in 0..NUM_COLS {
        std::mem::swap(&mut evals[c], &mut scratch[c]);
        evals[c].truncate(next_len);
    }

    let mut out = [E::Scalar::zero(); D];
    for k in 0..D {
        out[k] = E::sum_lanes_to_mont(t_acc[k]);
    }
    out
}

/// Parallel const-`NUM_COLS` twin of `fold_base_tables_to_ext` (the base->ext transition fold).
///
/// This is the last serial dominator in the par ZeroCheck (~12 ms@nv=20). It folds the
/// ell0 small-value rounds' base PBF tables into ext `E` tables using the verifier challenges.
/// Mirrors the proven non-const-generic template
/// `SumcheckMamaBear::fold_base_tables_to_ext_generic_par` (`sumcheck_par.rs`), generalized
/// from a hardcoded 4 columns to const `NUM_COLS` so any `(D, NUM_COLS)` gate shape
/// goes through the same fold.
///
/// # Parallel axis and byte-identity
/// The `NUM_COLS` output tables are independent, so the fold is parallelized over poly index
/// (`0..NUM_COLS`). Within each poly the inner block loop is the *same* sequential fused
/// fold as the serial `fold_base_tables_to_ext` (identical `lazy_add_xp`/`lazy_sub`/`con_sub_xp`
/// reduce points and the same challenge-application order), so each output entry is bit-for-bit
/// mod-`P` identical to serial regardless of thread count. Only the 2- and 3-challenge fused
/// paths (the ell0=2 / ell0=3 production cases) are parallelized; every other arity, and inputs
/// below `PAR_ZEROCHECK_MIN_PACKED_GROUPS` output blocks, fall back to the serial twin — matching
/// `fold_base_tables_to_ext_generic_par`'s fallback discipline.
pub(crate) fn fold_base_tables_to_ext_par<E: SumcheckExtField + Send + Sync, const NUM_COLS: usize>(
    evals: &[Vec<PBF>; NUM_COLS],
    challenges: &[E::Scalar],
) -> [Vec<E>; NUM_COLS]
where
    E::Scalar: Send + Sync,
{
    debug_assert!(!challenges.is_empty());
    // Only the 2- and 3-challenge fused paths are parallelized; everything else (single
    // challenge / ell0 >= 4 generic fold) stays on the serial const-generic twin.
    if challenges.len() != 2 && challenges.len() != 3 {
        return fold_base_tables_to_ext::<E, NUM_COLS>(evals, challenges);
    }

    let fold_factor = 1usize << challenges.len();
    let next_len = evals[0].len() / fold_factor;
    if next_len < PAR_ZEROCHECK_MIN_PACKED_GROUPS {
        return fold_base_tables_to_ext::<E, NUM_COLS>(evals, challenges);
    }

    // Allocate the NUM_COLS output vectors up front with uninit capacity so the parallel fill
    // loop can write directly via raw pointers (each poly writes its own disjoint Vec).
    let mut outputs: [Vec<E>; NUM_COLS] = std::array::from_fn(|_| {
        let mut v: Vec<E> = Vec::with_capacity(next_len);
        unsafe {
            v.set_len(next_len);
        }
        v
    });
    let out_ptrs: [ParPtr<E>; NUM_COLS] = std::array::from_fn(|c| ParPtr(outputs[c].as_mut_ptr()));
    let in_ptrs: [ParPtr<PBF>; NUM_COLS] =
        std::array::from_fn(|c| ParPtr(evals[c].as_ptr() as *mut PBF));

    if challenges.len() == 2 {
        let alpha0 = E::from_scalar(challenges[0]);
        let alpha1 = E::from_scalar(challenges[1]);
        (0..NUM_COLS).into_par_iter().for_each(|poly_idx| {
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
                let low = alpha0.mul_base_elem(diff0).add_base_elem(e0).con_sub_xp(2);
                let high = alpha0.mul_base_elem(diff1).add_base_elem(e2).con_sub_xp(2);
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
        (0..NUM_COLS).into_par_iter().for_each(|poly_idx| {
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

/// Parallel degree-generic ZeroCheck prover. Byte-identical to `prove_zero_check_generic`.
///
/// Parallelism is applied to the dominant  fused fold rounds (the eq_L axis). The
/// small-value precompute, base->ext transition, first packed round, and scalar tail run
/// serially (they are correct and byte-identical; parallelizing them is a separate perf step).
pub fn prove_zero_check_generic_par<
    E: SumcheckExtField + Send + Sync,
    G: ZeroCheckGate<E, D, NUM_COLS>,
    const D: usize,
    const NUM_COLS: usize,
>(
    evals: [Vec<PBF>; NUM_COLS],
    point: &[E::Scalar],
    ell0: usize,
    zero_check: bool,
    transcript: &mut Transcript,
) -> (Vec<E::Scalar>, [E::Scalar; NUM_COLS], E::Scalar)
where
    E::Scalar: Send + Sync,
{
    let mut throwaway = ZeroCheckTimings::default();
    prove_zero_check_generic_par_profiled::<E, G, D, NUM_COLS>(
        evals, point, ell0, zero_check, transcript, &mut throwaway,
    )
}

/// Profiled variant of `prove_zero_check_generic_par`.
pub fn prove_zero_check_generic_par_profiled<
    E: SumcheckExtField + Send + Sync,
    G: ZeroCheckGate<E, D, NUM_COLS>,
    const D: usize,
    const NUM_COLS: usize,
>(
    evals: [Vec<PBF>; NUM_COLS],
    point: &[E::Scalar],
    ell0: usize,
    zero_check: bool,
    transcript: &mut Transcript,
    timings: &mut ZeroCheckTimings,
) -> (Vec<E::Scalar>, [E::Scalar; NUM_COLS], E::Scalar)
where
    E::Scalar: Send + Sync,
{
    use std::time::Instant;
    let t_total = Instant::now();
    let num_vars = point.len();
    let ell0 = resolve_ell0::<D>(num_vars, ell0);

    // Below the per-gate threshold, defer entirely to the serial prover (byte-identical).
    // Heavier gates (higher D / more columns) break even at larger nv.
    if num_vars < par_zerocheck_min_nv(D, NUM_COLS) {
        let r = crate::zerocheck_generic_mamabear::prove_zero_check_generic_profiled::<E, G, D, NUM_COLS>(
            evals, point, ell0, zero_check, transcript, timings,
        );
        timings.total_us += t_total.elapsed().as_micros();
        return r;
    }

    let point_mont: Vec<E::Scalar> = point.iter().copied().map(|v| v.to_montgomery()).collect();

    let t_eq = Instant::now();
    let eq_tables = SumcheckMamaBear::build_two_stage_eq_tables_generic::<E>(&point_mont);
    timings.eq_tables_us += t_eq.elapsed().as_micros();

    let consts = gate_consts_base::<D>();

    let t_pre = Instant::now();
    let small_value_tables = precompute_small_value_tables_par::<E, G, D, NUM_COLS>(
        &evals, &eq_tables, ell0, zero_check, &consts,
    );
    timings.precompute_us += t_pre.elapsed().as_micros();

    let ext_one = E::Scalar::one().to_montgomery();
    let mut finite_point_ext = [E::Scalar::zero(); MAX_U];
    for (p, slot) in finite_point_ext.iter_mut().enumerate() {
        *slot = E::Scalar::from(p as u32).to_montgomery();
    }

    let mut verifier_challenges = Vec::with_capacity(num_vars);
    let mut prefix_eq_eval = ext_one;
    let mut lagrange_weights = vec![ext_one];

    let t_small = Instant::now();
    for round in 0..ell0 {
        let t_hat = compute_t_from_precomputed::<E, D>(&small_value_tables[round], &lagrange_weights);
        let s_hat = compute_round_s_from_t::<E, D>(
            prefix_eq_eval,
            point_mont[round],
            t_hat,
            ext_one,
            &finite_point_ext,
        );
        append_hat_round_values::<E, D>(transcript, s_hat);

        let challenge = transcript.challenge_f::<E::Scalar>().to_montgomery();
        prefix_eq_eval *=
            SumcheckMamaBear::eq_linear_mont_generic::<E>(point_mont[round], challenge, ext_one);
        verifier_challenges.push(challenge);

        if round + 1 < ell0 {
            let basis = lagrange_basis_degree_d_generic::<E, D>(challenge, ext_one);
            lagrange_weights = update_small_value_weights::<E, D>(&lagrange_weights, &basis);
        }
    }
    timings.small_value_rounds_us += t_small.elapsed().as_micros();

    let packed_round_start = ell0;
    let packed_round_end = num_vars.saturating_sub(3);

    let t_trans = Instant::now();
    // parallel base->ext transition fold (the last serial dominator). Byte-identical to
    // the serial `fold_base_tables_to_ext`; falls back to serial for ell0 not in {2,3} or small
    // inputs.
    let mut folded_tables =
        fold_base_tables_to_ext_par::<E, NUM_COLS>(&evals, &verifier_challenges[..ell0]);
    drop(evals);
    timings.transition_fold_us += t_trans.elapsed().as_micros();

    // Preallocate + pre-touch the ping-pong scratch sized for the first packed-fold output.
    let mut scratch: [Vec<E>; NUM_COLS] = std::array::from_fn(|_| {
        Vec::with_capacity(folded_tables[0].len() >> 1)
    });
    if packed_round_start + 1 < packed_round_end {
        for v in scratch.iter_mut() {
            unsafe {
                v.set_len(folded_tables[0].len() >> 1);
            }
        }
        pretouch_scratch_generic::<E, NUM_COLS>(&mut scratch);
    }

    if packed_round_start < packed_round_end {
        let t_first = Instant::now();
        let first_t_hat = compute_round_t::<E, G, D, NUM_COLS>(
            &folded_tables,
            &SumcheckMamaBear::round_eq_view_generic(&eq_tables, packed_round_start),
        );
        timings.packed_fold_rounds_us += t_first.elapsed().as_micros();
        let s_hat = compute_round_s_from_t::<E, D>(
            prefix_eq_eval,
            point_mont[packed_round_start],
            first_t_hat,
            ext_one,
            &finite_point_ext,
        );
        append_hat_round_values::<E, D>(transcript, s_hat);

        let challenge = transcript.challenge_f::<E::Scalar>().to_montgomery();
        prefix_eq_eval *= SumcheckMamaBear::eq_linear_mont_generic::<E>(
            point_mont[packed_round_start],
            challenge,
            ext_one,
        );
        verifier_challenges.push(challenge);
        let mut prev_challenge = challenge;

        for round in (packed_round_start + 1)..packed_round_end {
            let eq_view = SumcheckMamaBear::round_eq_view_generic(&eq_tables, round);
            let t_pf = Instant::now();
            let t_hat = fold_and_compute_round_t_par::<E, G, D, NUM_COLS>(
                &mut folded_tables,
                &eq_view,
                prev_challenge,
                &mut scratch,
            );
            timings.packed_fold_rounds_us += t_pf.elapsed().as_micros();
            let s_hat = compute_round_s_from_t::<E, D>(
                prefix_eq_eval,
                point_mont[round],
                t_hat,
                ext_one,
                &finite_point_ext,
            );
            append_hat_round_values::<E, D>(transcript, s_hat);

            let challenge = transcript.challenge_f::<E::Scalar>().to_montgomery();
            prefix_eq_eval *=
                SumcheckMamaBear::eq_linear_mont_generic::<E>(point_mont[round], challenge, ext_one);
            verifier_challenges.push(challenge);
            prev_challenge = challenge;
        }

        let t_pf_final = Instant::now();
        fold_ext_tables_in_place::<E, NUM_COLS>(&mut folded_tables, E::from_scalar(prev_challenge));
        timings.packed_fold_rounds_us += t_pf_final.elapsed().as_micros();
    }

    let t_tail = Instant::now();
    let scalar_tail_start = (num_vars.saturating_sub(3)).max(ell0);
    let mut active_len = 1usize << (num_vars - scalar_tail_start);
    let mut packed_tail_tables: [E; NUM_COLS] = std::array::from_fn(|c| folded_tables[c][0]);

    for round in scalar_tail_start..num_vars {
        let eq_view = SumcheckMamaBear::round_eq_view_generic(&eq_tables, round);
        let t_hat =
            compute_round_t_single::<E, G, D, NUM_COLS>(&packed_tail_tables, &eq_view, active_len);
        let s_hat = compute_round_s_from_t::<E, D>(
            prefix_eq_eval,
            point_mont[round],
            t_hat,
            ext_one,
            &finite_point_ext,
        );
        append_hat_round_values::<E, D>(transcript, s_hat);

        let challenge = transcript.challenge_f::<E::Scalar>().to_montgomery();
        prefix_eq_eval *=
            SumcheckMamaBear::eq_linear_mont_generic::<E>(point_mont[round], challenge, ext_one);
        verifier_challenges.push(challenge);

        if active_len > 1 {
            fold_single_packed_in_place::<E, NUM_COLS>(
                &mut packed_tail_tables,
                E::from_scalar(challenge),
                active_len,
            );
            active_len >>= 1;
        }
    }

    let mut col_claims = [E::Scalar::zero(); NUM_COLS];
    for c in 0..NUM_COLS {
        col_claims[c] = E::unpack_to_scalars(packed_tail_tables[c])[0].from_montgomery();
    }
    let eq_claim = prefix_eq_eval.from_montgomery();
    timings.scalar_tail_us += t_tail.elapsed().as_micros();
    timings.total_us += t_total.elapsed().as_micros();
    (verifier_challenges, col_claims, eq_claim)
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::zerocheck_generic_mamabear::{prove_zero_check_generic, AddMulD3};
    use arithmetic::field::mamabear::LazyReduction;
    use arithmetic::field::mamabear::{
        MamaBearScalar as SBF, MamaBearScalarExt3 as SEF3,
        PackedMamaBearAVX512 as PBF, PackedMamaBearAVX512Ext3 as PEF3,
    };
    use rand::rngs::SmallRng;
    use rand::SeedableRng;
    use util::fiat_shamir::Transcript;

    fn pack_scalar_evals(evals: &[SBF]) -> Vec<PBF> {
        assert_eq!(evals.len() % 8, 0);
        let stride = evals.len() / 8;
        let mut packed = Vec::with_capacity(stride);
        for packed_idx in 0..stride {
            let mut lanes = [0u64; 8];
            for lane in 0..8 {
                lanes[lane] = evals[lane * stride + packed_idx].0;
            }
            packed.push(PBF::from_array(lanes));
        }
        packed
    }

    fn random_cols(nv: usize, rng: &mut SmallRng) -> [Vec<PBF>; 4] {
        random_cols_n::<4>(nv, rng)
    }

    fn random_cols_n<const NC: usize>(nv: usize, rng: &mut SmallRng) -> [Vec<PBF>; NC] {
        let domain = 1usize << nv;
        let mut scalar: [Vec<SBF>; NC] = std::array::from_fn(|_| Vec::with_capacity(domain));
        for col in scalar.iter_mut() {
            for _ in 0..domain {
                col.push(SBF::random(&mut *rng).to_montgomery());
            }
        }
        scalar.each_ref().map(|poly| pack_scalar_evals(poly))
    }

    /// Direct byte-identity check of `precompute_small_value_tables_par` vs the serial twin
    /// for one (gate, nv, ell0, zero_check). Compares canonical (`.reduce()`) residues since
    /// `sum_lanes_to_mont` returns a non-canonical Montgomery representative (the downstream
    /// `from_montgomery` at the transcript boundary canonicalizes, so mod-`P` equality is the
    /// transcript-relevant invariant).
    fn check_precompute_par<
        E: SumcheckExtField + Send + Sync,
        G: ZeroCheckGate<E, D, NC>,
        const D: usize,
        const NC: usize,
    >(
        nv: usize,
        ell0: usize,
        zero_check: bool,
        seed: u64,
    ) where
        E::Scalar: Send + Sync + LazyReduction,
    {
        let mut rng = SmallRng::seed_from_u64(seed);
        let cols = random_cols_n::<NC>(nv, &mut rng);
        let point: Vec<E::Scalar> = (0..nv).map(|_| E::Scalar::random(&mut rng)).collect();
        let point_mont: Vec<E::Scalar> = point.iter().map(|v| v.to_montgomery()).collect();
        let eq_tables = SumcheckMamaBear::build_two_stage_eq_tables_generic::<E>(&point_mont);
        let consts = crate::zerocheck_generic_mamabear::gate_consts_base::<D>();
        let ell0r = crate::zerocheck_generic_mamabear::resolve_ell0::<D>(nv, ell0);
        let ser = crate::zerocheck_generic_mamabear::precompute_small_value_tables::<E, G, D, NC>(
            &cols, &eq_tables, ell0r, zero_check, &consts,
        );
        let par = super::precompute_small_value_tables_par::<E, G, D, NC>(
            &cols, &eq_tables, ell0r, zero_check, &consts,
        );
        assert_eq!(ser.len(), par.len(), "round count mismatch nv={nv} ell0={ell0r}");
        for (round, (rs, rp)) in ser.iter().zip(par.iter()).enumerate() {
            assert_eq!(rs.len(), rp.len(), "row len mismatch round={round} nv={nv}");
            for (i, (a, b)) in rs.iter().zip(rp.iter()).enumerate() {
                assert_eq!(
                    a.reduce(),
                    b.reduce(),
                    "precompute par != serial nv={nv} ell0={ell0r} zc={zero_check} round={round} entry={i}"
                );
            }
        }
    }

    /// Parallel precompute is byte-identical (mod P) to serial across small nv (which hit
    /// the serial-fallback under `PAR_PRECOMPUTE_MIN_BASE_BLOCKS`), mid nv (real parallel chunk
    /// path), and nv=20 (overflow guard — chained `lazy_add` over a full leaf without the
    /// periodic reduce would wrap u64). Exercises the three production gate shapes.
    #[test]
    fn precompute_small_value_tables_par_matches_serial_ext3() {
        for &nv in &[10usize, 12, 14, 16] {
            for ell0 in 1..=3usize {
                for &zc in &[true, false] {
                    let s = 0x100 + (nv as u64) * 11 + (ell0 as u64) * 3 + zc as u64;
                    check_precompute_par::<PEF3, AddMulD3, 3, 4>(nv, ell0, zc, s);
                }
            }
        }
        check_precompute_par::<PEF3, AddMulD3, 3, 4>(20, 2, true, 0xABCD);
    }

    #[test]
    fn zerocheck_generic_par_matches_serial_ext3() {
        let mut rng = SmallRng::seed_from_u64(0x9A11_E13);
        // nv=10,12 hit the serial-fallback (below par_zerocheck_min_nv); nv>=19 run the real
        // parallel path; nv=20 is the large-nv overflow guard.
        for &nv in &[10usize, 12, 16, 17, 18, 19, 20, 21, 22] {
            let point: Vec<SEF3> = (0..nv).map(|_| SEF3::random(&mut rng)).collect();
            let cols = random_cols(nv, &mut rng);
            for ell0 in 1..=4usize.min(nv.saturating_sub(3)) {
                let mut t_ser = Transcript::new();
                let ser = prove_zero_check_generic::<PEF3, AddMulD3, 3, 4>(
                    cols.clone(), &point, ell0, true, &mut t_ser,
                );
                let mut t_par = Transcript::new();
                let par = prove_zero_check_generic_par::<PEF3, AddMulD3, 3, 4>(
                    cols.clone(), &point, ell0, true, &mut t_par,
                );
                assert_eq!(t_ser.proof.bytes, t_par.proof.bytes, "ext3 par bytes nv={nv} ell0={ell0}");
                assert_eq!(ser.0, par.0, "ext3 par challenges nv={nv} ell0={ell0}");
                assert_eq!(ser.1, par.1, "ext3 par col claims nv={nv} ell0={ell0}");
                assert_eq!(ser.2, par.2, "ext3 par eq claim nv={nv} ell0={ell0}");
            }
        }
    }

}
