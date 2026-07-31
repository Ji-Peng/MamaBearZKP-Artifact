///! Parallel Per-wire Batched ProductCheck.
///!
///! Creates parallel versions of the hot paths in prodcheck_mamabear_perwire.rs:
///! - 6-way parallel tree build
///! - parallel compute_round_t (left_idx parallelism)
///! - 6-way parallel fold_all_wires
///!
///! All functions produce bit-identical results to their serial counterparts.

#[allow(unused_imports)]
use arithmetic::field::mamabear::*;
use arithmetic::field::Field;
use rayon::prelude::*;
use util::fiat_shamir::Transcript;

use crate::prodcheck_mamabear_perwire::ProdCheckTimings;
use crate::sumcheck_mamabear::{MontgomeryOps, SumcheckExtField, SumcheckMamaBear};

const HAT_SIZE: usize = SumcheckMamaBear::OPT_HAT_SIZE; // 3: {0, 2, ∞}
const NUM_WIRES: usize = 3;
const NUM_TREES: usize = 2;

// =============================================================================
// Reuse helpers from serial version
// =============================================================================

fn build_packed_product_tree<E: SumcheckExtField>(base: Vec<E>, nv: usize) -> Vec<Vec<E>> {
    debug_assert_eq!(base.len(), 1 << (nv - 3));
    let num_levels = nv - 2;
    let mut tree: Vec<Vec<E>> = Vec::with_capacity(num_levels);
    tree.push(base);
    for j in 1..num_levels {
        let prev = &tree[j - 1];
        let len = prev.len() >> 1;
        let mut level = Vec::with_capacity(len);
        for k in 0..len {
            level.push(prev[k << 1] * prev[(k << 1) + 1]);
        }
        tree.push(level);
    }
    tree
}

fn reduce_packed_to_scalar_products<E: SumcheckExtField>(top: E) -> (E::Scalar, E::Scalar) {
    let l = E::unpack_to_scalars(top);
    let a = (l[0] * l[1]) * (l[2] * l[3]);
    let b = (l[4] * l[5]) * (l[6] * l[7]);
    (a, b)
}

#[inline(always)]
fn eval_batched_gate<E: SumcheckExtField>(
    wires: &[[&[E]; NUM_TREES]; NUM_WIRES],
    base: usize,
    eq_r_a: E,
    eq_r_b: E,
    eq_r_c: E,
) -> [[E; HAT_SIZE]; NUM_TREES] {
    let mut result = [[E::zero(); HAT_SIZE]; NUM_TREES];
    for t in 0..NUM_TREES {
        let e0_a = wires[0][t][base];
        let o0_a = wires[0][t][base + 1];
        let e1_a = wires[0][t][base + 2];
        let o1_a = wires[0][t][base + 3];
        let de_a = e1_a.lazy_add_xp(2).lazy_sub(e0_a).con_sub_xp(2);
        let do_a = o1_a.lazy_add_xp(2).lazy_sub(o0_a).con_sub_xp(2);
        let we0_a = eq_r_a * e0_a;
        let wd_a = eq_r_a * de_a;

        let e0_b = wires[1][t][base];
        let o0_b = wires[1][t][base + 1];
        let e1_b = wires[1][t][base + 2];
        let o1_b = wires[1][t][base + 3];
        let de_b = e1_b.lazy_add_xp(2).lazy_sub(e0_b).con_sub_xp(2);
        let do_b = o1_b.lazy_add_xp(2).lazy_sub(o0_b).con_sub_xp(2);
        let we0_b = eq_r_b * e0_b;
        let wd_b = eq_r_b * de_b;

        let e0_c = wires[2][t][base];
        let o0_c = wires[2][t][base + 1];
        let e1_c = wires[2][t][base + 2];
        let o1_c = wires[2][t][base + 3];
        let de_c = e1_c.lazy_add_xp(2).lazy_sub(e0_c).con_sub_xp(2);
        let do_c = o1_c.lazy_add_xp(2).lazy_sub(o0_c).con_sub_xp(2);
        let we0_c = eq_r_c * e0_c;
        let wd_c = eq_r_c * de_c;

        result[t][0] = (we0_a * o0_a).lazy_add(we0_b * o0_b).lazy_add(we0_c * o0_c);
        let we2_a = we0_a.lazy_add(wd_a).lazy_add(wd_a).con_sub_xp(2);
        let o2_a = o0_a.lazy_add(do_a).lazy_add(do_a).con_sub_xp(2);
        let we2_b = we0_b.lazy_add(wd_b).lazy_add(wd_b).con_sub_xp(2);
        let o2_b = o0_b.lazy_add(do_b).lazy_add(do_b).con_sub_xp(2);
        let we2_c = we0_c.lazy_add(wd_c).lazy_add(wd_c).con_sub_xp(2);
        let o2_c = o0_c.lazy_add(do_c).lazy_add(do_c).con_sub_xp(2);
        result[t][1] = (we2_a * o2_a).lazy_add(we2_b * o2_b).lazy_add(we2_c * o2_c);
        result[t][2] = (wd_a * do_a).lazy_add(wd_b * do_b).lazy_add(wd_c * do_c);
    }
    result
}

#[inline(always)]
fn compute_round_s_from_t<E: SumcheckExtField>(
    prefix_eq: E::Scalar,
    w_i: E::Scalar,
    t_hat: [[E::Scalar; HAT_SIZE]; NUM_TREES],
    one: E::Scalar,
    two: E::Scalar,
) -> [[E::Scalar; HAT_SIZE]; NUM_TREES] {
    let eq_0 = one - w_i;
    let eq_2 = SumcheckMamaBear::eq_linear_mont_generic::<E>(w_i, two, one);
    let eq_inf = w_i + w_i - one;
    let c = [prefix_eq * eq_0, prefix_eq * eq_2, prefix_eq * eq_inf];
    std::array::from_fn(|t| [c[0] * t_hat[t][0], c[1] * t_hat[t][1], c[2] * t_hat[t][2]])
}

#[inline(always)]
fn append_hat_round_values<E: SumcheckExtField>(
    transcript: &mut Transcript,
    s_hat: [[E::Scalar; HAT_SIZE]; NUM_TREES],
) {
    for tree in 0..NUM_TREES {
        for h in 0..HAT_SIZE {
            transcript.append_f(s_hat[tree][h].from_montgomery());
        }
    }
}

fn fold_one_interleaved<E: SumcheckExtField>(arr: &mut Vec<E>, alpha: E) {
    let new_len = arr.len() >> 1;
    for g in 0..(new_len >> 1) {
        let src = g << 2;
        let dst = g << 1;
        let e0 = arr[src];
        let o0 = arr[src + 1];
        let e1 = arr[src + 2];
        let o1 = arr[src + 3];
        arr[dst] = (alpha * (e1.lazy_add_xp(2).lazy_sub(e0)))
            .lazy_add(e0)
            .con_sub_xp(2);
        arr[dst + 1] = (alpha * (o1.lazy_add_xp(2).lazy_sub(o0)))
            .lazy_add(o0)
            .con_sub_xp(2);
    }
    arr.truncate(new_len);
}

// =============================================================================
// Parallel compute_round_t
// =============================================================================

/// Parallel version of compute_round_t_perwire (byte-identical to the serial
/// path). Parallelizes over `left_group` with per-`left_group` weighted results
/// that are summed back in ascending order, exactly matching the serial
/// accumulation. See `compute_round_t_perwire` for the eq-weighting convention:
/// eq_R broadcast (immediate-next block bits) and eq_L packed per-lane (later
/// block bits AND the data lanes), so the data lanes are eq-weighted per-lane
/// and the reduced claim telescopes in natural variable order.
fn compute_round_t_perwire_par<E: SumcheckExtField + Send + Sync>(
    wires: &[[&[E]; NUM_TREES]; NUM_WIRES],
    eq_view: &crate::sumcheck_mamabear::RoundEqView<'_, E>,
    lambda: E,
    lambda2: E,
) -> [[E::Scalar; HAT_SIZE]; NUM_TREES]
where
    E::Scalar: Send + Sync,
{
    let fold_pairs = wires[0][0].len() >> 2;

    // Two-stage factored path (eq_R broadcast, eq_L packed per-lane).
    if let Some(split) = eq_view.packed_split_for_groups(fold_pairs) {
        let right_len = split.right_broadcast.len();
        let left_count = split.left_packed.len();

        // Precompute lambda * eq_R and lambda^2 * eq_R (eq_R is broadcast).
        let lambda_eq_r: Vec<E> = split.right_broadcast.iter().map(|&e| lambda * e).collect();
        let lambda2_eq_r: Vec<E> = split.right_broadcast.iter().map(|&e| lambda2 * e).collect();

        let weight_left = |left_group: usize| -> [[E; HAT_SIZE]; NUM_TREES] {
            let mut inner_t = [[E::zero(); HAT_SIZE]; NUM_TREES];
            let group_base = left_group * right_len;
            for right_idx in 0..right_len {
                let g = group_base + right_idx;
                let base = g << 2;
                let gate = eval_batched_gate::<E>(
                    wires,
                    base,
                    split.right_broadcast[right_idx],
                    lambda_eq_r[right_idx],
                    lambda2_eq_r[right_idx],
                );
                for tree in 0..NUM_TREES {
                    for h in 0..HAT_SIZE {
                        inner_t[tree][h] = inner_t[tree][h].lazy_add(gate[tree][h]);
                    }
                }
            }
            let eq_l_packed = split.left_packed[left_group]; // per-lane [0, 2P)
            let mut weighted = [[E::zero(); HAT_SIZE]; NUM_TREES];
            for tree in 0..NUM_TREES {
                for h in 0..HAT_SIZE {
                    weighted[tree][h] = eq_l_packed * inner_t[tree][h].reduce_fast();
                }
            }
            weighted
        };

        // Compute per-left_group weighted contributions, then sum in ascending
        // order (byte-identical to the serial lazy_add chain).
        let partial_results: Vec<[[E; HAT_SIZE]; NUM_TREES]> = if left_count >= 16 {
            (0..left_count).into_par_iter().map(weight_left).collect()
        } else {
            (0..left_count).map(weight_left).collect()
        };

        let mut t = [[E::zero(); HAT_SIZE]; NUM_TREES];
        for partial in &partial_results {
            for tree in 0..NUM_TREES {
                for h in 0..HAT_SIZE {
                    t[tree][h] = t[tree][h].lazy_add(partial[tree][h]);
                }
            }
        }

        let mut result = [[E::Scalar::zero(); HAT_SIZE]; NUM_TREES];
        for tree in 0..NUM_TREES {
            for h in 0..HAT_SIZE {
                result[tree][h] = E::sum_lanes_to_mont(t[tree][h].reduce_fast());
            }
        }
        return result;
    }

    // Flat path (same as serial): per-lane eq weight via load_packed_weight.
    let mut t = [[E::zero(); HAT_SIZE]; NUM_TREES];
    for g in 0..fold_pairs {
        let weight = eq_view.load_packed_weight(g, fold_pairs);
        let base = g << 2;
        for tree in 0..NUM_TREES {
            let mut gate_t = [E::zero(); HAT_SIZE];
            for wire in 0..NUM_WIRES {
                let e0 = wires[wire][tree][base];
                let o0 = wires[wire][tree][base + 1];
                let e1 = wires[wire][tree][base + 2];
                let o1 = wires[wire][tree][base + 3];
                let de = e1.lazy_add_xp(2).lazy_sub(e0).con_sub_xp(2);
                let d_o = o1.lazy_add_xp(2).lazy_sub(o0).con_sub_xp(2);
                let lambda_w = if wire == 0 {
                    E::from_scalar(E::Scalar::one().to_montgomery())
                } else if wire == 1 { lambda } else { lambda2 };
                let g0 = lambda_w * (e0 * o0);
                let e2 = e0.lazy_add(de).lazy_add(de).con_sub_xp(2);
                let o2 = o0.lazy_add(d_o).lazy_add(d_o).con_sub_xp(2);
                let g2 = lambda_w * (e2 * o2);
                let gi = lambda_w * (de * d_o);
                gate_t[0] = gate_t[0].lazy_add(g0);
                gate_t[1] = gate_t[1].lazy_add(g2);
                gate_t[2] = gate_t[2].lazy_add(gi);
            }
            for h in 0..HAT_SIZE {
                t[tree][h] = t[tree][h].lazy_add(weight * gate_t[h]);
            }
        }
    }
    let mut result = [[E::Scalar::zero(); HAT_SIZE]; NUM_TREES];
    for tree in 0..NUM_TREES {
        for h in 0..HAT_SIZE {
            result[tree][h] = E::sum_lanes_to_mont(t[tree][h].reduce_fast());
        }
    }
    result
}

// =============================================================================
// Parallel fold
// =============================================================================

fn fold_all_wires_par<E: SumcheckExtField + Send + Sync>(
    wire_data: &mut [[Vec<E>; NUM_TREES]; NUM_WIRES],
    challenge: E::Scalar,
) where E::Scalar: Send + Sync {
    let alpha = E::from_scalar(challenge);
    // Split the array to get independent mutable references
    let [ref mut w0, ref mut w1, ref mut w2] = wire_data;
    let [ref mut w0t0, ref mut w0t1] = w0;
    let [ref mut w1t0, ref mut w1t1] = w1;
    let [ref mut w2t0, ref mut w2t1] = w2;
    rayon::scope(|s| {
        s.spawn(|_| fold_one_interleaved::<E>(w0t0, alpha));
        s.spawn(|_| fold_one_interleaved::<E>(w0t1, alpha));
        s.spawn(|_| fold_one_interleaved::<E>(w1t0, alpha));
        s.spawn(|_| fold_one_interleaved::<E>(w1t1, alpha));
        s.spawn(|_| fold_one_interleaved::<E>(w2t0, alpha));
        s.spawn(|_| fold_one_interleaved::<E>(w2t1, alpha));
    });
}

// =============================================================================
// Parallel prove
// =============================================================================

pub struct ProdEqCheckMamaBearPerWirePar;

impl ProdEqCheckMamaBearPerWirePar {
    /// Parallel ProductCheck prover — bit-identical to serial `prove`.
    pub fn prove<E: SumcheckExtField + Send + Sync>(
        evals: [[Vec<E>; NUM_WIRES]; NUM_TREES],
        transcript: &mut Transcript,
    ) -> Vec<E::Scalar>
    where
        E::Scalar: Send + Sync,
    {
        let packed_len = evals[0][0].len();
        assert!(packed_len.is_power_of_two() && packed_len >= 2);
        let nv = (packed_len << 3).ilog2() as usize;
        assert!(nv >= 4);

        let one = E::Scalar::one().to_montgomery();
        let two = E::Scalar::from(2u32).to_montgomery();

        // : Build 6 product trees in parallel
        let [[e00, e01, e02], [e10, e11, e12]] = evals;
        let (left, right) = rayon::join(
            || {
                let (pair01, t02) = rayon::join(
                    || rayon::join(
                        || build_packed_product_tree::<E>(e00, nv),
                        || build_packed_product_tree::<E>(e01, nv),
                    ),
                    || build_packed_product_tree::<E>(e02, nv),
                );
                (pair01.0, pair01.1, t02)
            },
            || {
                let (pair01, t12) = rayon::join(
                    || rayon::join(
                        || build_packed_product_tree::<E>(e10, nv),
                        || build_packed_product_tree::<E>(e11, nv),
                    ),
                    || build_packed_product_tree::<E>(e12, nv),
                );
                (pair01.0, pair01.1, t12)
            },
        );
        let (t00, t01, t02) = left;
        let (t10, t11, t12) = right;

        let mut trees: [[Vec<Vec<E>>; NUM_WIRES]; NUM_TREES] = [
            [t00, t01, t02],
            [t10, t11, t12],
        ];

        //  cont: top extraction (serial — tiny)
        let mut tops = [[E::zero(); NUM_WIRES]; NUM_TREES];
        for t in 0..NUM_TREES {
            for g in 0..NUM_WIRES {
                tops[t][g] = *trees[t][g].last().unwrap().last().unwrap();
            }
        }
        for t in 0..NUM_TREES {
            for g in 0..NUM_WIRES {
                let (a, b) = reduce_packed_to_scalar_products::<E>(tops[t][g]);
                transcript.append_f(a.from_montgomery());
                transcript.append_f(b.from_montgomery());
            }
        }

        let mut scalar_prods: [[Vec<Vec<E::Scalar>>; NUM_WIRES]; NUM_TREES] =
            std::array::from_fn(|_| std::array::from_fn(|_| vec![]));
        for t in 0..NUM_TREES {
            for g in 0..NUM_WIRES {
                let lanes = E::unpack_to_scalars(tops[t][g]);
                let l0 = lanes.to_vec();
                let l1: Vec<E::Scalar> = (0..4).map(|k| lanes[2 * k] * lanes[2 * k + 1]).collect();
                let l2 = vec![l1[0] * l1[1], l1[2] * l1[3]];
                scalar_prods[t][g] = vec![l0, l1, l2];
            }
        }

        let mut fp = [E::Scalar::zero(); NUM_TREES];
        for t in 0..NUM_TREES {
            fp[t] = one;
            for g in 0..NUM_WIRES {
                fp[t] = fp[t] * scalar_prods[t][g][2][0] * scalar_prods[t][g][2][1];
            }
            transcript.append_f(fp[t].from_montgomery());
        }

        // : Challenges
        let mut point: Vec<E::Scalar> = vec![transcript.challenge_f::<E::Scalar>().to_montgomery()];
        let lambda_scalar = transcript.challenge_f::<E::Scalar>().to_montgomery();
        let lambda2_scalar = lambda_scalar * lambda_scalar;
        let lambda_packed = E::from_scalar(lambda_scalar);
        let lambda2_packed = E::from_scalar(lambda2_scalar);

        // : Iterative sumcheck
        for level in (0..nv - 1).rev() {
            if level >= nv - 3 {
                // Scalar level — use serial sumcheck_scalar (tiny data)
                let sl_idx = level - (nv - 3);
                let eq_tables = SumcheckMamaBear::build_two_stage_eq_tables_generic::<E>(&point);
                let (challenges, final_evals) =
                    crate::prodcheck_mamabear_perwire::ProdEqCheckMamaBearPerWire::sumcheck_scalar_pub::<E>(
                        &scalar_prods, sl_idx, &eq_tables, &point,
                        lambda_scalar, lambda2_scalar, one, two, transcript,
                    );
                for t in 0..NUM_TREES {
                    for val in &final_evals[t] {
                        transcript.append_f(val.from_montgomery());
                    }
                }
                let r = transcript.challenge_f::<E::Scalar>().to_montgomery();
                let mut new_point = vec![r];
                new_point.extend(challenges);
                point = new_point;
            } else {
                // Packed level — use parallel round_t and fold
                let eq_tables = SumcheckMamaBear::build_two_stage_eq_tables_generic::<E>(&point);
                let mut wire_data: [[Vec<E>; NUM_TREES]; NUM_WIRES] =
                    std::array::from_fn(|_| std::array::from_fn(|_| vec![]));
                for g in 0..NUM_WIRES {
                    for t in 0..NUM_TREES {
                        wire_data[g][t] = std::mem::take(&mut trees[t][g][level]);
                    }
                }

                let (challenges, final_evals) = sumcheck_packed_par::<E>(
                    &mut wire_data, &eq_tables, &point,
                    lambda_packed, lambda2_packed, one, two, transcript,
                );

                for t in 0..NUM_TREES {
                    for val in &final_evals[t] {
                        transcript.append_f(val.from_montgomery());
                    }
                }
                let r = transcript.challenge_f::<E::Scalar>().to_montgomery();
                let mut new_point = vec![r];
                new_point.extend(challenges);
                point = new_point;
            }
        }

        point
    }

    /// Profiled version of the parallel ProductCheck prover.
    pub fn prove_profiled<E: SumcheckExtField + Send + Sync>(
        evals: [[Vec<E>; NUM_WIRES]; NUM_TREES],
        transcript: &mut Transcript,
        timings: &mut ProdCheckTimings,
    ) -> Vec<E::Scalar>
    where
        E::Scalar: Send + Sync,
    {
        use std::time::Instant;

        let packed_len = evals[0][0].len();
        assert!(packed_len.is_power_of_two() && packed_len >= 2);
        let nv = (packed_len << 3).ilog2() as usize;
        assert!(nv >= 4);

        let one = E::Scalar::one().to_montgomery();
        let two = E::Scalar::from(2u32).to_montgomery();

        // : Build 6 product trees in parallel
        let t0 = Instant::now();
        let [[e00, e01, e02], [e10, e11, e12]] = evals;
        let (left, right) = rayon::join(
            || {
                let (pair01, t02) = rayon::join(
                    || rayon::join(
                        || build_packed_product_tree::<E>(e00, nv),
                        || build_packed_product_tree::<E>(e01, nv),
                    ),
                    || build_packed_product_tree::<E>(e02, nv),
                );
                (pair01.0, pair01.1, t02)
            },
            || {
                let (pair01, t12) = rayon::join(
                    || rayon::join(
                        || build_packed_product_tree::<E>(e10, nv),
                        || build_packed_product_tree::<E>(e11, nv),
                    ),
                    || build_packed_product_tree::<E>(e12, nv),
                );
                (pair01.0, pair01.1, t12)
            },
        );
        let (t00, t01, t02) = left;
        let (t10, t11, t12) = right;
        let mut trees: [[Vec<Vec<E>>; NUM_WIRES]; NUM_TREES] = [
            [t00, t01, t02],
            [t10, t11, t12],
        ];
        timings.tree_build_us += t0.elapsed().as_micros();

        //  cont: top extraction
        let t0 = Instant::now();
        let mut tops = [[E::zero(); NUM_WIRES]; NUM_TREES];
        for t in 0..NUM_TREES {
            for g in 0..NUM_WIRES {
                tops[t][g] = *trees[t][g].last().unwrap().last().unwrap();
            }
        }
        for t in 0..NUM_TREES {
            for g in 0..NUM_WIRES {
                let (a, b) = reduce_packed_to_scalar_products::<E>(tops[t][g]);
                transcript.append_f(a.from_montgomery());
                transcript.append_f(b.from_montgomery());
            }
        }
        let mut scalar_prods: [[Vec<Vec<E::Scalar>>; NUM_WIRES]; NUM_TREES] =
            std::array::from_fn(|_| std::array::from_fn(|_| vec![]));
        for t in 0..NUM_TREES {
            for g in 0..NUM_WIRES {
                let lanes = E::unpack_to_scalars(tops[t][g]);
                let l0 = lanes.to_vec();
                let l1: Vec<E::Scalar> = (0..4).map(|k| lanes[2 * k] * lanes[2 * k + 1]).collect();
                let l2 = vec![l1[0] * l1[1], l1[2] * l1[3]];
                scalar_prods[t][g] = vec![l0, l1, l2];
            }
        }
        let mut fp = [E::Scalar::zero(); NUM_TREES];
        for t in 0..NUM_TREES {
            fp[t] = one;
            for g in 0..NUM_WIRES {
                fp[t] = fp[t] * scalar_prods[t][g][2][0] * scalar_prods[t][g][2][1];
            }
            transcript.append_f(fp[t].from_montgomery());
        }
        timings.top_extract_us += t0.elapsed().as_micros();

        // : Challenges
        let mut point: Vec<E::Scalar> = vec![transcript.challenge_f::<E::Scalar>().to_montgomery()];
        let lambda_scalar = transcript.challenge_f::<E::Scalar>().to_montgomery();
        let lambda2_scalar = lambda_scalar * lambda_scalar;
        let lambda_packed = E::from_scalar(lambda_scalar);
        let lambda2_packed = E::from_scalar(lambda2_scalar);

        // : Iterative sumcheck
        for level in (0..nv - 1).rev() {
            if level >= nv - 3 {
                let t0 = Instant::now();
                let sl_idx = level - (nv - 3);
                let eq_tables = SumcheckMamaBear::build_two_stage_eq_tables_generic::<E>(&point);
                let (challenges, final_evals) =
                    crate::prodcheck_mamabear_perwire::ProdEqCheckMamaBearPerWire::sumcheck_scalar_pub::<E>(
                        &scalar_prods, sl_idx, &eq_tables, &point,
                        lambda_scalar, lambda2_scalar, one, two, transcript,
                    );
                for t in 0..NUM_TREES {
                    for val in &final_evals[t] {
                        transcript.append_f(val.from_montgomery());
                    }
                }
                let r = transcript.challenge_f::<E::Scalar>().to_montgomery();
                let mut new_point = vec![r];
                new_point.extend(challenges);
                point = new_point;
                timings.scalar_tail_us += t0.elapsed().as_micros();
            } else {
                // Packed level — see serial prove_profiled for the substage
                // breakdown rationale.
                let t_eq = Instant::now();
                let eq_tables = SumcheckMamaBear::build_two_stage_eq_tables_generic::<E>(&point);
                timings.eq_tables_us += t_eq.elapsed().as_micros();

                let mut wire_data: [[Vec<E>; NUM_TREES]; NUM_WIRES] =
                    std::array::from_fn(|_| std::array::from_fn(|_| vec![]));
                for g in 0..NUM_WIRES {
                    for t in 0..NUM_TREES {
                        wire_data[g][t] = std::mem::take(&mut trees[t][g][level]);
                    }
                }

                // Inline the parallel sumcheck with timing
                let interleaved_pairs = wire_data[0][0].len() >> 1;
                let num_vars = interleaved_pairs.ilog2() as usize + 3;
                let packed_round_end = if interleaved_pairs > 1 {
                    interleaved_pairs.ilog2() as usize
                } else { 0 };
                let mut challenges_inner = Vec::with_capacity(num_vars);
                let mut prefix_eq = one;

                if packed_round_end > 0 {
                    // Round 0: eval only
                    let t_rt = Instant::now();
                    let eq_view = SumcheckMamaBear::round_eq_view_generic(&eq_tables, 0);
                    let wires_ref: [[&[E]; NUM_TREES]; NUM_WIRES] =
                        std::array::from_fn(|g| std::array::from_fn(|t| wire_data[g][t].as_slice()));
                    let t_hat = compute_round_t_perwire_par::<E>(
                        &wires_ref, &eq_view, lambda_packed, lambda2_packed,
                    );
                    timings.round_t_us += t_rt.elapsed().as_micros();

                    let s_hat = compute_round_s_from_t::<E>(prefix_eq, point[0], t_hat, one, two);
                    append_hat_round_values::<E>(transcript, s_hat);
                    let ch = transcript.challenge_f::<E::Scalar>().to_montgomery();
                    prefix_eq *= SumcheckMamaBear::eq_linear_mont_generic::<E>(point[0], ch, one);
                    challenges_inner.push(ch);

                    for round in 1..packed_round_end {
                        let t_fold = Instant::now();
                        fold_all_wires_par::<E>(&mut wire_data, challenges_inner[round - 1]);
                        timings.fold_us += t_fold.elapsed().as_micros();

                        let t_rt = Instant::now();
                        let eq_view = SumcheckMamaBear::round_eq_view_generic(&eq_tables, round);
                        let wires_ref: [[&[E]; NUM_TREES]; NUM_WIRES] =
                            std::array::from_fn(|g| std::array::from_fn(|t| wire_data[g][t].as_slice()));
                        let t_hat = compute_round_t_perwire_par::<E>(
                            &wires_ref, &eq_view, lambda_packed, lambda2_packed,
                        );
                        timings.round_t_us += t_rt.elapsed().as_micros();

                        let s_hat = compute_round_s_from_t::<E>(prefix_eq, point[round], t_hat, one, two);
                        append_hat_round_values::<E>(transcript, s_hat);
                        let ch = transcript.challenge_f::<E::Scalar>().to_montgomery();
                        prefix_eq *= SumcheckMamaBear::eq_linear_mont_generic::<E>(point[round], ch, one);
                        challenges_inner.push(ch);
                    }

                    let t_fold = Instant::now();
                    fold_all_wires_par::<E>(&mut wire_data, *challenges_inner.last().unwrap());
                    timings.fold_us += t_fold.elapsed().as_micros();
                }

                //  B: within-register fold (3 rounds) — serial tail
                // of every packed level.
                let t_inreg = Instant::now();
                debug_assert!(wire_data[0][0].len() == 2);
                let mut pe: [[E; NUM_WIRES]; NUM_TREES] =
                    std::array::from_fn(|t| std::array::from_fn(|g| wire_data[g][t][0]));
                let mut po: [[E; NUM_WIRES]; NUM_TREES] =
                    std::array::from_fn(|t| std::array::from_fn(|g| wire_data[g][t][1]));
                let mut active_lanes = 8usize;
                let phase_b_start = packed_round_end;
                for round in phase_b_start..num_vars {
                    let eq_view = SumcheckMamaBear::round_eq_view_generic(&eq_tables, round);
                    let pair_count = active_lanes >> 1;
                    let mut sw = [E::Scalar::zero(); 8];
                    for i in 0..pair_count { sw[i] = eq_view.scalar_weight(i); }
                    let pw = E::pack_scalars(&sw[..pair_count]);
                    let mut t_hat = [[E::Scalar::zero(); HAT_SIZE]; NUM_TREES];
                    for t in 0..NUM_TREES {
                        let mut gate_0 = E::zero();
                        let mut gate_2 = E::zero();
                        let mut gate_inf = E::zero();
                        for g in 0..NUM_WIRES {
                            let (v0_e, _, d_e) = SumcheckMamaBear::build_single_packed_pair_views_generic::<E>(pe[t][g], pair_count);
                            let (v0_o, _, d_o) = SumcheckMamaBear::build_single_packed_pair_views_generic::<E>(po[t][g], pair_count);
                            let lw = if g == 0 { E::from_scalar(one) } else if g == 1 { lambda_packed } else { lambda2_packed };
                            gate_0 = gate_0.lazy_add(lw * (v0_e * v0_o));
                            let e2 = v0_e.lazy_add(d_e).lazy_add(d_e).con_sub_xp(2);
                            let o2 = v0_o.lazy_add(d_o).lazy_add(d_o).con_sub_xp(2);
                            gate_2 = gate_2.lazy_add(lw * (e2 * o2));
                            gate_inf = gate_inf.lazy_add(lw * (d_e * d_o));
                        }
                        t_hat[t][0] = E::sum_lanes_to_mont((pw * gate_0).reduce_fast());
                        t_hat[t][1] = E::sum_lanes_to_mont((pw * gate_2).reduce_fast());
                        t_hat[t][2] = E::sum_lanes_to_mont((pw * gate_inf).reduce_fast());
                    }
                    let s_hat = compute_round_s_from_t::<E>(prefix_eq, point[round], t_hat, one, two);
                    append_hat_round_values::<E>(transcript, s_hat);
                    let ch = transcript.challenge_f::<E::Scalar>().to_montgomery();
                    prefix_eq *= SumcheckMamaBear::eq_linear_mont_generic::<E>(point[round], ch, one);
                    challenges_inner.push(ch);
                    let alpha = E::from_scalar(ch);
                    for t in 0..NUM_TREES {
                        for g in 0..NUM_WIRES {
                            for val in [&mut pe[t][g], &mut po[t][g]] {
                                let (v0, _, diff) = SumcheckMamaBear::build_single_packed_pair_views_generic::<E>(*val, pair_count);
                                *val = (alpha * diff).con_sub_xp(2).lazy_add(v0).con_sub_xp(2);
                            }
                        }
                    }
                    active_lanes >>= 1;
                }
                let mut final_evals = [[E::Scalar::zero(); 2 * NUM_WIRES]; NUM_TREES];
                for t in 0..NUM_TREES {
                    for g in 0..NUM_WIRES {
                        final_evals[t][2 * g] = E::unpack_to_scalars(pe[t][g])[0];
                        final_evals[t][2 * g + 1] = E::unpack_to_scalars(po[t][g])[0];
                    }
                }
                timings.inreg_fold_us += t_inreg.elapsed().as_micros();

                for t in 0..NUM_TREES {
                    for val in &final_evals[t] {
                        transcript.append_f(val.from_montgomery());
                    }
                }
                let r = transcript.challenge_f::<E::Scalar>().to_montgomery();
                let mut new_point = vec![r];
                new_point.extend(challenges_inner);
                point = new_point;
            }
        }

        point
    }
}

/// Parallel packed sumcheck with parallel round_t and fold.
fn sumcheck_packed_par<E: SumcheckExtField + Send + Sync>(
    wire_data: &mut [[Vec<E>; NUM_TREES]; NUM_WIRES],
    eq_tables: &crate::sumcheck_mamabear::TwoStageEqTables<E>,
    point: &[E::Scalar],
    lambda: E,
    lambda2: E,
    one: E::Scalar,
    two: E::Scalar,
    transcript: &mut Transcript,
) -> (Vec<E::Scalar>, [[E::Scalar; 2 * NUM_WIRES]; NUM_TREES])
where
    E::Scalar: Send + Sync,
{
    let interleaved_pairs = wire_data[0][0].len() >> 1;
    let num_vars = interleaved_pairs.ilog2() as usize + 3;
    let packed_round_end = if interleaved_pairs > 1 {
        interleaved_pairs.ilog2() as usize
    } else {
        0
    };

    let mut challenges = Vec::with_capacity(num_vars);
    let mut prefix_eq = one;

    if packed_round_end > 0 {
        // Round 0: eval only (no fold)
        let eq_view = SumcheckMamaBear::round_eq_view_generic(eq_tables, 0);
        let wires_ref: [[&[E]; NUM_TREES]; NUM_WIRES] =
            std::array::from_fn(|g| std::array::from_fn(|t| wire_data[g][t].as_slice()));
        let t_hat = compute_round_t_perwire_par::<E>(&wires_ref, &eq_view, lambda, lambda2);
        let s_hat = compute_round_s_from_t::<E>(prefix_eq, point[0], t_hat, one, two);
        append_hat_round_values::<E>(transcript, s_hat);
        let ch = transcript.challenge_f::<E::Scalar>().to_montgomery();
        prefix_eq *= SumcheckMamaBear::eq_linear_mont_generic::<E>(point[0], ch, one);
        challenges.push(ch);

        // Subsequent rounds: parallel fold then parallel round_t
        for round in 1..packed_round_end {
            fold_all_wires_par::<E>(wire_data, challenges[round - 1]);

            let eq_view = SumcheckMamaBear::round_eq_view_generic(eq_tables, round);
            let wires_ref: [[&[E]; NUM_TREES]; NUM_WIRES] =
                std::array::from_fn(|g| std::array::from_fn(|t| wire_data[g][t].as_slice()));
            let t_hat = compute_round_t_perwire_par::<E>(&wires_ref, &eq_view, lambda, lambda2);
            let s_hat = compute_round_s_from_t::<E>(prefix_eq, point[round], t_hat, one, two);
            append_hat_round_values::<E>(transcript, s_hat);
            let ch = transcript.challenge_f::<E::Scalar>().to_montgomery();
            prefix_eq *= SumcheckMamaBear::eq_linear_mont_generic::<E>(point[round], ch, one);
            challenges.push(ch);
        }

        // Final fold
        fold_all_wires_par::<E>(wire_data, *challenges.last().unwrap());
    }

    //  B: within-register fold (3 rounds) — serial (tiny data)
    debug_assert!(wire_data[0][0].len() == 2);
    let mut pe: [[E; NUM_WIRES]; NUM_TREES] =
        std::array::from_fn(|t| std::array::from_fn(|g| wire_data[g][t][0]));
    let mut po: [[E; NUM_WIRES]; NUM_TREES] =
        std::array::from_fn(|t| std::array::from_fn(|g| wire_data[g][t][1]));
    let mut active_lanes = 8usize;
    let phase_b_start = packed_round_end;
    for round in phase_b_start..num_vars {
        let eq_view = SumcheckMamaBear::round_eq_view_generic(eq_tables, round);
        let pair_count = active_lanes >> 1;
        let mut sw = [E::Scalar::zero(); 8];
        for i in 0..pair_count { sw[i] = eq_view.scalar_weight(i); }
        let pw = E::pack_scalars(&sw[..pair_count]);
        let mut t_hat = [[E::Scalar::zero(); HAT_SIZE]; NUM_TREES];
        for t in 0..NUM_TREES {
            let mut gate_0 = E::zero();
            let mut gate_2 = E::zero();
            let mut gate_inf = E::zero();
            for g in 0..NUM_WIRES {
                let (v0_e, _, d_e) = SumcheckMamaBear::build_single_packed_pair_views_generic::<E>(pe[t][g], pair_count);
                let (v0_o, _, d_o) = SumcheckMamaBear::build_single_packed_pair_views_generic::<E>(po[t][g], pair_count);
                let lw = if g == 0 { E::from_scalar(one) } else if g == 1 { lambda } else { lambda2 };
                gate_0 = gate_0.lazy_add(lw * (v0_e * v0_o));
                let e2 = v0_e.lazy_add(d_e).lazy_add(d_e).con_sub_xp(2);
                let o2 = v0_o.lazy_add(d_o).lazy_add(d_o).con_sub_xp(2);
                gate_2 = gate_2.lazy_add(lw * (e2 * o2));
                gate_inf = gate_inf.lazy_add(lw * (d_e * d_o));
            }
            t_hat[t][0] = E::sum_lanes_to_mont((pw * gate_0).reduce_fast());
            t_hat[t][1] = E::sum_lanes_to_mont((pw * gate_2).reduce_fast());
            t_hat[t][2] = E::sum_lanes_to_mont((pw * gate_inf).reduce_fast());
        }
        let s_hat = compute_round_s_from_t::<E>(prefix_eq, point[round], t_hat, one, two);
        append_hat_round_values::<E>(transcript, s_hat);
        let ch = transcript.challenge_f::<E::Scalar>().to_montgomery();
        prefix_eq *= SumcheckMamaBear::eq_linear_mont_generic::<E>(point[round], ch, one);
        challenges.push(ch);
        let alpha = E::from_scalar(ch);
        for t in 0..NUM_TREES {
            for g in 0..NUM_WIRES {
                for val in [&mut pe[t][g], &mut po[t][g]] {
                    let (v0, _, diff) = SumcheckMamaBear::build_single_packed_pair_views_generic::<E>(*val, pair_count);
                    *val = (alpha * diff).con_sub_xp(2).lazy_add(v0).con_sub_xp(2);
                }
            }
        }
        active_lanes >>= 1;
    }

    let mut final_evals = [[E::Scalar::zero(); 2 * NUM_WIRES]; NUM_TREES];
    for t in 0..NUM_TREES {
        for g in 0..NUM_WIRES {
            final_evals[t][2 * g] = E::unpack_to_scalars(pe[t][g])[0];
            final_evals[t][2 * g + 1] = E::unpack_to_scalars(po[t][g])[0];
        }
    }
    (challenges, final_evals)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prodcheck_mamabear_perwire::ProdEqCheckMamaBearPerWire;
    use arithmetic::field::mamabear::PackedMamaBearAVX512Ext3;
    use arithmetic::field::Field;
    use rand::rngs::SmallRng;
    use rand::SeedableRng;

    type PEF3Ty = PackedMamaBearAVX512Ext3;

    fn pack_input<E: SumcheckExtField>(scalars: &[E::Scalar]) -> Vec<E> {
        let packed_len = scalars.len() / 8;
        (0..packed_len)
            .map(|i| {
                let base = i << 3;
                let s: [E::Scalar; 8] = std::array::from_fn(|k| scalars[base + k].to_montgomery());
                E::pack_scalars(&s)
            })
            .collect()
    }

    /// Verify that parallel prove produces bit-identical transcript to serial prove.
    fn run_par_vs_serial<E: SumcheckExtField + Send + Sync>(seed: u64, nv: usize)
    where E::Scalar: Send + Sync
    {
        let n = 1 << nv;
        let mut rng = SmallRng::seed_from_u64(seed);
        let evals_serial: [[Vec<E::Scalar>; NUM_WIRES]; NUM_TREES] = std::array::from_fn(|_| {
            std::array::from_fn(|_| (0..n).map(|_| E::Scalar::random(&mut rng)).collect())
        });
        let evals_par: [[Vec<E::Scalar>; NUM_WIRES]; NUM_TREES] = evals_serial.clone();

        let packed_serial: [[Vec<E>; NUM_WIRES]; NUM_TREES] =
            std::array::from_fn(|t| std::array::from_fn(|g| pack_input::<E>(&evals_serial[t][g])));
        let packed_par: [[Vec<E>; NUM_WIRES]; NUM_TREES] =
            std::array::from_fn(|t| std::array::from_fn(|g| pack_input::<E>(&evals_par[t][g])));

        let mut t_serial = Transcript::new();
        let r_serial = ProdEqCheckMamaBearPerWire::prove::<E>(packed_serial, &mut t_serial);

        let mut t_par = Transcript::new();
        let r_par = ProdEqCheckMamaBearPerWirePar::prove::<E>(packed_par, &mut t_par);

        assert_eq!(t_serial.proof.bytes, t_par.proof.bytes, "Transcript mismatch at nv={}", nv);
        assert_eq!(r_serial, r_par, "Point mismatch at nv={}", nv);
    }

    #[test]
    fn par_ext3_nv8() { run_par_vs_serial::<PEF3Ty>(77, 8); }
    #[test]
    fn par_ext3_nv12() { run_par_vs_serial::<PEF3Ty>(999, 12); }
}
