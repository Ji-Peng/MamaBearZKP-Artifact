//! Per-wire Batched ProductCheck: eliminates 25% padding overhead.
//!
//! Instead of padding 3 wires to 4·2^nv, builds 3 independent product trees
//! (one per wire) and uses λ-batching to combine their sumcheck claims.
//!
//! Gate: `output^(t) = Σ_g λ^g · f_{g,even}^(t) · f_{g,odd}^(t) · eq(point, x)`
//! where g ∈ {a, b, c} (3 wires), t ∈ {0, 1} (2 trees).
//!
//! Uses packed-interleaved layout per wire (same as prodcheck_mamabear_packed).
//! Two-stage eq with pre-packed eq_R + broadcast eq_L.

use arithmetic::field::mamabear::*;
use arithmetic::field::Field;
use util::fiat_shamir::Transcript;

use crate::sumcheck_mamabear::{MontgomeryOps, SumcheckExtField, SumcheckMamaBear};

const HAT_SIZE: usize = SumcheckMamaBear::OPT_HAT_SIZE; // 3: {0, 2, ∞}
const NUM_WIRES: usize = 3; // a, b, c
const NUM_TREES: usize = 2; // evals1, evals2

pub struct ProdEqCheckMamaBearPerWire;

// =============================================================================
// Reuse product tree from packed version
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

// =============================================================================
// λ-batched gate: 3 wires combined with eq_R fusing
// =============================================================================

/// Evaluate λ-batched gate × eq_R at 3 hat points.
///
/// For each tree t:
///   output(X) = Σ_g λ^g · even_g(X) · odd_g(X)
///
/// Fuses eq_R into even inputs: `we_g = (λ^g · eq_R) × even_g`.
/// Then: `output_t_h = Σ_g we_g_h × odd_g_h`.
///
/// `eq_r_a/b/c` = pre-computed `[eq_R, λ·eq_R, λ²·eq_R]` (hoisted from caller).
/// 6 weighted-even + 9 gate products = 15 PEF muls per tree, 30 total.
#[inline(always)]
fn eval_batched_gate<E: SumcheckExtField>(
    wires: &[[&[E]; NUM_TREES]; NUM_WIRES],
    base: usize,
    eq_r_a: E, // eq_R (wire a weight)
    eq_r_b: E, // λ · eq_R (wire b weight)
    eq_r_c: E, // λ² · eq_R (wire c weight)
) -> [[E; HAT_SIZE]; NUM_TREES] {
    let mut result = [[E::zero(); HAT_SIZE]; NUM_TREES];

    for t in 0..NUM_TREES {
        // Wire a (λ^0 = 1)
        let e0_a = wires[0][t][base];
        let o0_a = wires[0][t][base + 1];
        let e1_a = wires[0][t][base + 2];
        let o1_a = wires[0][t][base + 3];
        let de_a = e1_a.lazy_add_xp(2).lazy_sub(e0_a).con_sub_xp(2);
        let do_a = o1_a.lazy_add_xp(2).lazy_sub(o0_a).con_sub_xp(2);
        let we0_a = eq_r_a * e0_a; // [0, 1.5P)
        let wd_a = eq_r_a * de_a;

        // Wire b (λ^1)
        let e0_b = wires[1][t][base];
        let o0_b = wires[1][t][base + 1];
        let e1_b = wires[1][t][base + 2];
        let o1_b = wires[1][t][base + 3];
        let de_b = e1_b.lazy_add_xp(2).lazy_sub(e0_b).con_sub_xp(2);
        let do_b = o1_b.lazy_add_xp(2).lazy_sub(o0_b).con_sub_xp(2);
        let we0_b = eq_r_b * e0_b;
        let wd_b = eq_r_b * de_b;

        // Wire c (λ^2)
        let e0_c = wires[2][t][base];
        let o0_c = wires[2][t][base + 1];
        let e1_c = wires[2][t][base + 2];
        let o1_c = wires[2][t][base + 3];
        let de_c = e1_c.lazy_add_xp(2).lazy_sub(e0_c).con_sub_xp(2);
        let do_c = o1_c.lazy_add_xp(2).lazy_sub(o0_c).con_sub_xp(2);
        let we0_c = eq_r_c * e0_c;
        let wd_c = eq_r_c * de_c;

        // Hat X=0: Σ_g we0_g × o0_g
        result[t][0] = (we0_a * o0_a).lazy_add(we0_b * o0_b).lazy_add(we0_c * o0_c);

        // Hat X=2: Σ_g (we0_g + 2·wd_g) × (o0_g + 2·do_g)
        let we2_a = we0_a.lazy_add(wd_a).lazy_add(wd_a).con_sub_xp(2);
        let o2_a = o0_a.lazy_add(do_a).lazy_add(do_a).con_sub_xp(2);
        let we2_b = we0_b.lazy_add(wd_b).lazy_add(wd_b).con_sub_xp(2);
        let o2_b = o0_b.lazy_add(do_b).lazy_add(do_b).con_sub_xp(2);
        let we2_c = we0_c.lazy_add(wd_c).lazy_add(wd_c).con_sub_xp(2);
        let o2_c = o0_c.lazy_add(do_c).lazy_add(do_c).con_sub_xp(2);
        result[t][1] = (we2_a * o2_a).lazy_add(we2_b * o2_b).lazy_add(we2_c * o2_c);

        // Hat X=∞: Σ_g wd_g × do_g
        result[t][2] = (wd_a * do_a).lazy_add(wd_b * do_b).lazy_add(wd_c * do_c);
    }

    result
}

// =============================================================================
// Round polynomial computation with two-stage eq
// =============================================================================

/// Compute the three hat values of this round's eq-divided gate polynomial for
/// all wires and both trees, over the packed (interleaved even/odd) blocks.
///
/// The eq weight `eq(point[round+1:], future)` must be associated with the
/// variables in the SAME order they are folded: the remaining BLOCK-position
/// bits (folded first, in the packed rounds) map to the LOW tail positions, and
/// the 8 within-register LANE bits (folded LAST, in the scalar/-B tail)
/// map to the HIGH, SIMD-strided tail positions. This is exactly the layout
/// that `RoundEqView::{load_packed_weight, packed_split_for_groups}` encode
/// (lane `k` of packed group `g` -> scalar index `g + k * fold_pairs`), i.e.
/// the same convention the add/mul main sumcheck uses. Weighting the data lanes
/// per-lane (rather than broadcasting eq across them) is what makes the reduced
/// claim telescope in natural variable order, so the verifier's per-level oracle
/// check `y_reduced == leaf_gate * eq(point, challenges)` holds.
fn compute_round_t_perwire<E: SumcheckExtField>(
    wires: &[[&[E]; NUM_TREES]; NUM_WIRES],
    eq_view: &crate::sumcheck_mamabear::RoundEqView<'_, E>,
    lambda: E,
    lambda2: E,
) -> [[E::Scalar; HAT_SIZE]; NUM_TREES] {
    let fold_pairs = wires[0][0].len() >> 2;

    // Two-stage factored path: eq(point, x) = eq_L(x_L) * eq_R(x_R), with eq_R
    // broadcast across all 8 lanes (the immediate-next block bits) and eq_L
    // packed per-lane via `pack_eq_l` (the later block bits AND the data lanes).
    // Identical factorization to the add/mul main sumcheck.
    let mut t = [[E::zero(); HAT_SIZE]; NUM_TREES];
    if let Some(split) = eq_view.packed_split_for_groups(fold_pairs) {
        let right_len = split.right_broadcast.len();
        // Hoist lambda * eq_R and lambda^2 * eq_R (eq_R is broadcast per right idx).
        let lambda_eq_r: Vec<E> = split.right_broadcast.iter().map(|&e| lambda * e).collect();
        let lambda2_eq_r: Vec<E> = split.right_broadcast.iter().map(|&e| lambda2 * e).collect();

        for left_group in 0..split.left_packed.len() {
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
            for tree in 0..NUM_TREES {
                for h in 0..HAT_SIZE {
                    t[tree][h] = t[tree][h].lazy_add(eq_l_packed * inner_t[tree][h].reduce_fast());
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

    // Flat path: gather the per-lane eq weight for each block group directly
    // (lane `k` <- scalar_weight(g + k * fold_pairs)), matching the main
    // sumcheck's `load_packed_weight` convention.
    for g in 0..fold_pairs {
        let weight = eq_view.load_packed_weight(g, fold_pairs);
        let base = g << 2;

        // Use weight directly (not fused into gate) for fallback
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
                } else if wire == 1 {
                    lambda
                } else {
                    lambda2
                };

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
// s_hat from t_hat
// =============================================================================

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
    values: [[E::Scalar; HAT_SIZE]; NUM_TREES],
) {
    for t in 0..NUM_TREES {
        for h in 0..HAT_SIZE {
            transcript.append_f(values[t][h].from_montgomery());
        }
    }
}

// =============================================================================
// Interleaved fold (in-place, for all 6 wire×tree arrays)
// =============================================================================

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

fn fold_all_wires<E: SumcheckExtField>(
    wire_data: &mut [[Vec<E>; NUM_TREES]; NUM_WIRES],
    challenge: E::Scalar,
) {
    let alpha = E::from_scalar(challenge);
    for g in 0..NUM_WIRES {
        for t in 0..NUM_TREES {
            fold_one_interleaved::<E>(&mut wire_data[g][t], alpha);
        }
    }
}

// =============================================================================
// Main prover
// =============================================================================

/// Timing breakdown for profiled ProductCheck prove.
///
/// Notes on what each field captures:
/// - `build_inputs_us`: `build_productcheck_inputs` — populated by the profile
///   binary (external to `prove_profiled`), not by the prover itself.
/// - `tree_build_us`: 6-way parallel `build_packed_product_tree` calls.
/// - `top_extract_us`: final top-level extraction + scalar product chain +
///   the transcript `append_f` calls for the top products (small).
/// - `eq_tables_us`: `build_two_stage_eq_tables_generic` per packed level +
///   per-round `prepack_eq_r` precompute.
/// - `round_t_us`: `compute_round_t_perwire(_par)` across all packed rounds.
/// - `fold_us`: `fold_all_wires(_par)` across all packed rounds.
/// - `inreg_fold_us`: the "within-register fold" (3 scalar-ish rounds
///   per packed level that operate on 2 packed blocks only). This is the
///   serial tail at the bottom of every packed level.
/// - `scalar_tail_us`: the final pure-scalar levels (`level >= nv - 3`) that
///   run `sumcheck_scalar`.
/// - `round_misc_us`: remaining per-round serial overhead inside packed
///   levels: `compute_round_s_from_t`, `append_hat_round_values`,
///   `challenge_f`, `eq_linear_mont_generic`, the `mem::take` of tree levels
///   into wire_data, and the final transcript appends / next-point update
///   at the end of each level. None of these is a single large line, but
///   together they account for the residual "other" inside `prod_chk`.
#[derive(Clone, Debug, Default)]
pub struct ProdCheckTimings {
    pub build_inputs_us: u128,
    pub tree_build_us: u128,
    pub top_extract_us: u128,
    pub eq_tables_us: u128,
    pub round_t_us: u128,
    pub fold_us: u128,
    pub inreg_fold_us: u128,
    pub scalar_tail_us: u128,
    pub round_misc_us: u128,
}

impl ProdEqCheckMamaBearPerWire {
    /// Per-wire Batched ProductCheck prover.
    ///
    /// # Input
    ///
    /// `evals`: `[[Vec<E>; NUM_WIRES]; NUM_TREES]` — 6 pre-packed vectors:
    /// - `evals[t][g]`: tree t, wire g, each `2^(nv-3)` packed elements in Montgomery form.
    ///
    /// # Output
    ///
    /// Random evaluation point in normal form.
    pub fn prove<E: SumcheckExtField>(
        evals: [[Vec<E>; NUM_WIRES]; NUM_TREES],
        transcript: &mut Transcript,
    ) -> Vec<E::Scalar> {
        let packed_len = evals[0][0].len();
        assert!(packed_len.is_power_of_two() && packed_len >= 2);
        let nv = (packed_len << 3).ilog2() as usize;
        assert!(nv >= 4);

        let one = E::Scalar::one().to_montgomery();
        let two = E::Scalar::from(2u32).to_montgomery();

        // =====================================================================
        // Build 6 product trees (3 wires × 2 trees)
        // =====================================================================
        let mut trees: [[Vec<Vec<E>>; NUM_WIRES]; NUM_TREES] = std::array::from_fn(|_t| {
            std::array::from_fn(|_g| {
                // Move data out: no copy
                // Can't move from array in from_fn, use a workaround below
                vec![]
            })
        });
        // Build trees by moving ownership
        let [[e00, e01, e02], [e10, e11, e12]] = evals;
        trees[0][0] = build_packed_product_tree::<E>(e00, nv);
        trees[0][1] = build_packed_product_tree::<E>(e01, nv);
        trees[0][2] = build_packed_product_tree::<E>(e02, nv);
        trees[1][0] = build_packed_product_tree::<E>(e10, nv);
        trees[1][1] = build_packed_product_tree::<E>(e11, nv);
        trees[1][2] = build_packed_product_tree::<E>(e12, nv);

        // =====================================================================
        // cont: extract top products, send to transcript
        // =====================================================================
        let mut tops = [[E::zero(); NUM_WIRES]; NUM_TREES];
        for t in 0..NUM_TREES {
            for g in 0..NUM_WIRES {
                tops[t][g] = *trees[t][g].last().unwrap().last().unwrap();
            }
        }

        // Reduce 8 lanes to 2 half-products per wire×tree, send to transcript
        for t in 0..NUM_TREES {
            for g in 0..NUM_WIRES {
                let (a, b) = reduce_packed_to_scalar_products::<E>(tops[t][g]);
                transcript.append_f(a.from_montgomery());
                transcript.append_f(b.from_montgomery());
            }
        }

        // =====================================================================
        // cont: scalar top levels (8→4→2 within register)
        // =====================================================================
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

        // Final product equality check (verifier checks this)
        let mut fp = [E::Scalar::zero(); NUM_TREES];
        for t in 0..NUM_TREES {
            fp[t] = one;
            for g in 0..NUM_WIRES {
                fp[t] = fp[t] * scalar_prods[t][g][2][0] * scalar_prods[t][g][2][1];
            }
            transcript.append_f(fp[t].from_montgomery());
        }

        // =====================================================================
        // Challenges
        // =====================================================================
        let mut point: Vec<E::Scalar> = vec![transcript.challenge_f::<E::Scalar>().to_montgomery()];
        let lambda_scalar = transcript.challenge_f::<E::Scalar>().to_montgomery();
        let lambda2_scalar = lambda_scalar * lambda_scalar;
        let lambda_packed = E::from_scalar(lambda_scalar);
        let lambda2_packed = E::from_scalar(lambda2_scalar);

        // =====================================================================
        // cont: Iterative sumcheck from top to bottom
        // =====================================================================
        for level in (0..nv - 1).rev() {
            if level >= nv - 3 {
                // Scalar level
                let sl_idx = level - (nv - 3);
                let eq_tables = SumcheckMamaBear::build_two_stage_eq_tables_generic::<E>(&point);

                // Run scalar sumcheck for this level
                let (challenges, final_evals) = Self::sumcheck_scalar::<E>(
                    &scalar_prods,
                    sl_idx,
                    &eq_tables,
                    &point,
                    lambda_scalar,
                    lambda2_scalar,
                    one,
                    two,
                    transcript,
                );

                // Send 12 final evals (3 wires × 2 (even/odd) × 2 trees)
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
                // Packed level
                let eq_tables = SumcheckMamaBear::build_two_stage_eq_tables_generic::<E>(&point);

                // Move wire data out of trees (no clone)
                let mut wire_data: [[Vec<E>; NUM_TREES]; NUM_WIRES] =
                    std::array::from_fn(|_| std::array::from_fn(|_| vec![]));
                for g in 0..NUM_WIRES {
                    for t in 0..NUM_TREES {
                        wire_data[g][t] = std::mem::take(&mut trees[t][g][level]);
                    }
                }

                let (challenges, final_evals) = Self::sumcheck_packed::<E>(
                    &mut wire_data,
                    &eq_tables,
                    &point,
                    lambda_packed,
                    lambda2_packed,
                    one,
                    two,
                    transcript,
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

        // Return point in Montgomery domain directly — caller expects mont values
        point
    }

    /// Scalar sumcheck for small levels.
    fn sumcheck_scalar<E: SumcheckExtField>(
        scalar_prods: &[[Vec<Vec<E::Scalar>>; NUM_WIRES]; NUM_TREES],
        sl_idx: usize,
        eq_tables: &crate::sumcheck_mamabear::TwoStageEqTables<E>,
        point: &[E::Scalar],
        lambda: E::Scalar,
        lambda2: E::Scalar,
        one: E::Scalar,
        two: E::Scalar,
        transcript: &mut Transcript,
    ) -> (Vec<E::Scalar>, [[E::Scalar; 2 * NUM_WIRES]; NUM_TREES]) {
        // Extract even/odd for each wire×tree
        let mut even: [[Vec<E::Scalar>; NUM_WIRES]; NUM_TREES] =
            std::array::from_fn(|_| std::array::from_fn(|_| vec![]));
        let mut odd: [[Vec<E::Scalar>; NUM_WIRES]; NUM_TREES] =
            std::array::from_fn(|_| std::array::from_fn(|_| vec![]));

        for t in 0..NUM_TREES {
            for g in 0..NUM_WIRES {
                let values = &scalar_prods[t][g][sl_idx];
                let half = values.len() / 2;
                even[t][g] = (0..half).map(|j| values[2 * j]).collect();
                odd[t][g] = (0..half).map(|j| values[2 * j + 1]).collect();
            }
        }

        let half = even[0][0].len();
        let num_vars = half.ilog2() as usize;
        let mut challenges = Vec::with_capacity(num_vars);
        let mut prefix_eq = one;

        // Pack into single packed elements per wire×tree
        let mut pe: [[E; NUM_WIRES]; NUM_TREES] =
            std::array::from_fn(|t| std::array::from_fn(|g| E::pack_scalars(&even[t][g])));
        let mut po: [[E; NUM_WIRES]; NUM_TREES] =
            std::array::from_fn(|t| std::array::from_fn(|g| E::pack_scalars(&odd[t][g])));
        let mut active = half >> 1;

        let lambda_weights = [one, lambda, lambda2];

        for round in 0..num_vars {
            let eq_view = SumcheckMamaBear::round_eq_view_generic(eq_tables, round);
            let mut sw = [E::Scalar::zero(); 8];
            for i in 0..active {
                sw[i] = eq_view.scalar_weight(i);
            }
            let pw = E::pack_scalars(&sw[..active]);

            let mut t_hat = [[E::Scalar::zero(); HAT_SIZE]; NUM_TREES];
            for t in 0..NUM_TREES {
                let mut gate_0 = E::zero();
                let mut gate_2 = E::zero();
                let mut gate_inf = E::zero();
                for g in 0..NUM_WIRES {
                    let (v0_e, _, d_e) = SumcheckMamaBear::build_single_packed_pair_views_generic::<
                        E,
                    >(pe[t][g], active);
                    let (v0_o, _, d_o) = SumcheckMamaBear::build_single_packed_pair_views_generic::<
                        E,
                    >(po[t][g], active);
                    let lw = E::from_scalar(lambda_weights[g]);
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

            // Fold with challenge — must run for active >= 1 (including last round)
            let alpha = E::from_scalar(ch);
            for t in 0..NUM_TREES {
                for g in 0..NUM_WIRES {
                    for val in [&mut pe[t][g], &mut po[t][g]] {
                        let (v0, _, diff) =
                            SumcheckMamaBear::build_single_packed_pair_views_generic::<E>(
                                *val, active,
                            );
                        *val = (alpha * diff).con_sub_xp(2).lazy_add(v0).con_sub_xp(2);
                    }
                }
            }
            active >>= 1;
        }

        // Extract final evals: 6 values per tree (even_a, odd_a, even_b, odd_b, even_c, odd_c)
        let mut final_evals = [[E::Scalar::zero(); 2 * NUM_WIRES]; NUM_TREES];
        for t in 0..NUM_TREES {
            for g in 0..NUM_WIRES {
                final_evals[t][2 * g] = E::unpack_to_scalars(pe[t][g])[0];
                final_evals[t][2 * g + 1] = E::unpack_to_scalars(po[t][g])[0];
            }
        }
        (challenges, final_evals)
    }

    /// Public wrapper for sumcheck_scalar, accessible from parallel module.
    pub(crate) fn sumcheck_scalar_pub<E: SumcheckExtField>(
        scalar_prods: &[[Vec<Vec<E::Scalar>>; NUM_WIRES]; NUM_TREES],
        sl_idx: usize,
        eq_tables: &crate::sumcheck_mamabear::TwoStageEqTables<E>,
        point: &[E::Scalar],
        lambda: E::Scalar,
        lambda2: E::Scalar,
        one: E::Scalar,
        two: E::Scalar,
        transcript: &mut Transcript,
    ) -> (Vec<E::Scalar>, [[E::Scalar; 2 * NUM_WIRES]; NUM_TREES]) {
        Self::sumcheck_scalar::<E>(scalar_prods, sl_idx, eq_tables, point, lambda, lambda2, one, two, transcript)
    }

    /// Packed sumcheck for large levels.
    fn sumcheck_packed<E: SumcheckExtField>(
        wire_data: &mut [[Vec<E>; NUM_TREES]; NUM_WIRES],
        eq_tables: &crate::sumcheck_mamabear::TwoStageEqTables<E>,
        point: &[E::Scalar],
        lambda: E,
        lambda2: E,
        one: E::Scalar,
        two: E::Scalar,
        transcript: &mut Transcript,
    ) -> (Vec<E::Scalar>, [[E::Scalar; 2 * NUM_WIRES]; NUM_TREES]) {
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
            // Round 0: eval only (no previous challenge to fold with)
            let eq_view = SumcheckMamaBear::round_eq_view_generic(eq_tables, 0);
            let wires_ref: [[&[E]; NUM_TREES]; NUM_WIRES] =
                std::array::from_fn(|g| std::array::from_fn(|t| wire_data[g][t].as_slice()));
            let t_hat = compute_round_t_perwire::<E>(&wires_ref, &eq_view, lambda, lambda2);
            let s_hat = compute_round_s_from_t::<E>(prefix_eq, point[0], t_hat, one, two);
            // Debug: verify Newton evaluation of round polynomial
            append_hat_round_values::<E>(transcript, s_hat);
            let ch = transcript.challenge_f::<E::Scalar>().to_montgomery();
            prefix_eq *= SumcheckMamaBear::eq_linear_mont_generic::<E>(point[0], ch, one);
            challenges.push(ch);

            // Subsequent rounds: fold then eval
            for round in 1..packed_round_end {
                fold_all_wires::<E>(wire_data, challenges[round - 1]);

                let eq_view = SumcheckMamaBear::round_eq_view_generic(eq_tables, round);
                let wires_ref: [[&[E]; NUM_TREES]; NUM_WIRES] =
                    std::array::from_fn(|g| std::array::from_fn(|t| wire_data[g][t].as_slice()));
                let t_hat = compute_round_t_perwire::<E>(&wires_ref, &eq_view, lambda, lambda2);
                let s_hat = compute_round_s_from_t::<E>(prefix_eq, point[round], t_hat, one, two);
                append_hat_round_values::<E>(transcript, s_hat);
                let ch = transcript.challenge_f::<E::Scalar>().to_montgomery();
                prefix_eq *= SumcheckMamaBear::eq_linear_mont_generic::<E>(point[round], ch, one);
                challenges.push(ch);
            }

            // Final fold
            fold_all_wires::<E>(wire_data, *challenges.last().unwrap());
        }

        // : within-register fold (3 rounds)
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
            for i in 0..pair_count {
                sw[i] = eq_view.scalar_weight(i);
            }
            let pw = E::pack_scalars(&sw[..pair_count]);

            let mut t_hat = [[E::Scalar::zero(); HAT_SIZE]; NUM_TREES];
            for t in 0..NUM_TREES {
                let mut gate_0 = E::zero();
                let mut gate_2 = E::zero();
                let mut gate_inf = E::zero();
                for g in 0..NUM_WIRES {
                    let (v0_e, _, d_e) = SumcheckMamaBear::build_single_packed_pair_views_generic::<
                        E,
                    >(pe[t][g], pair_count);
                    let (v0_o, _, d_o) = SumcheckMamaBear::build_single_packed_pair_views_generic::<
                        E,
                    >(po[t][g], pair_count);
                    let lw = if g == 0 {
                        E::from_scalar(one)
                    } else if g == 1 {
                        lambda
                    } else {
                        lambda2
                    };
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
                        let (v0, _, diff) =
                            SumcheckMamaBear::build_single_packed_pair_views_generic::<E>(
                                *val, pair_count,
                            );
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

    /// Profiled variant of `prove` — same computation, with per- timing.
    pub fn prove_profiled<E: SumcheckExtField>(
        evals: [[Vec<E>; NUM_WIRES]; NUM_TREES],
        transcript: &mut Transcript,
        timings: &mut ProdCheckTimings,
    ) -> Vec<E::Scalar> {
        use std::time::Instant;

        let packed_len = evals[0][0].len();
        assert!(packed_len.is_power_of_two() && packed_len >= 2);
        let nv = (packed_len << 3).ilog2() as usize;
        assert!(nv >= 4);

        let one = E::Scalar::one().to_montgomery();
        let two = E::Scalar::from(2u32).to_montgomery();

        // Build 6 product trees
        let t0 = Instant::now();
        let mut trees: [[Vec<Vec<E>>; NUM_WIRES]; NUM_TREES] = std::array::from_fn(|_t| {
            std::array::from_fn(|_g| vec![])
        });
        let [[e00, e01, e02], [e10, e11, e12]] = evals;
        trees[0][0] = build_packed_product_tree::<E>(e00, nv);
        trees[0][1] = build_packed_product_tree::<E>(e01, nv);
        trees[0][2] = build_packed_product_tree::<E>(e02, nv);
        trees[1][0] = build_packed_product_tree::<E>(e10, nv);
        trees[1][1] = build_packed_product_tree::<E>(e11, nv);
        trees[1][2] = build_packed_product_tree::<E>(e12, nv);
        timings.tree_build_us += t0.elapsed().as_micros();

        // cont: top extraction + scalar top levels
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

        // Challenges
        let mut point: Vec<E::Scalar> = vec![transcript.challenge_f::<E::Scalar>().to_montgomery()];
        let lambda_scalar = transcript.challenge_f::<E::Scalar>().to_montgomery();
        let lambda2_scalar = lambda_scalar * lambda_scalar;
        let lambda_packed = E::from_scalar(lambda_scalar);
        let lambda2_packed = E::from_scalar(lambda2_scalar);

        // Iterative sumcheck
        for level in (0..nv - 1).rev() {
            if level >= nv - 3 {
                let t0 = Instant::now();
                let sl_idx = level - (nv - 3);
                let eq_tables = SumcheckMamaBear::build_two_stage_eq_tables_generic::<E>(&point);
                let (challenges, final_evals) = Self::sumcheck_scalar::<E>(
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
                // Packed level. Account for three previously-invisible costs:
                //   (a) eq_tables_us: building the two-stage eq tables and the
                //       per-round `prepack_eq_r` precompute
                //   (b) round_misc_us: per-round serial overhead (compute_s,
                //       append_hat, challenge_f, eq_linear_mont, mem::take).
                //       Computed by the caller as the residual.
                //   (c) inreg_fold_us: 's 3-round within-register fold,
                //       which was previously lumped into scalar_tail_us.
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

                // Time the packed sumcheck, decomposing round_t and fold
                let interleaved_pairs = wire_data[0][0].len() >> 1;
                let num_vars = interleaved_pairs.ilog2() as usize + 3;
                let packed_round_end = if interleaved_pairs > 1 {
                    interleaved_pairs.ilog2() as usize
                } else {
                    0
                };

                let mut challenges_inner = Vec::with_capacity(num_vars);
                let mut prefix_eq = one;

                if packed_round_end > 0 {
                    // Round 0: eval only
                    let t_rt = Instant::now();
                    let eq_view = SumcheckMamaBear::round_eq_view_generic(&eq_tables, 0);
                    let wires_ref: [[&[E]; NUM_TREES]; NUM_WIRES] =
                        std::array::from_fn(|g| std::array::from_fn(|t| wire_data[g][t].as_slice()));
                    let t_hat = compute_round_t_perwire::<E>(
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
                        fold_all_wires::<E>(&mut wire_data, challenges_inner[round - 1]);
                        timings.fold_us += t_fold.elapsed().as_micros();

                        let t_rt = Instant::now();
                        let eq_view = SumcheckMamaBear::round_eq_view_generic(&eq_tables, round);
                        let wires_ref: [[&[E]; NUM_TREES]; NUM_WIRES] =
                            std::array::from_fn(|g| std::array::from_fn(|t| wire_data[g][t].as_slice()));
                        let t_hat = compute_round_t_perwire::<E>(
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
                    fold_all_wires::<E>(&mut wire_data, *challenges_inner.last().unwrap());
                    timings.fold_us += t_fold.elapsed().as_micros();
                }

                // : within-register fold (3 rounds). These operate on
                // 2 packed blocks only and are the serial tail of every
                // packed level, distinct from the pure scalar levels at the
                // very end.
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
                    let mut t_hat_inner = [[E::Scalar::zero(); HAT_SIZE]; NUM_TREES];
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
                        t_hat_inner[t][0] = E::sum_lanes_to_mont((pw * gate_0).reduce_fast());
                        t_hat_inner[t][1] = E::sum_lanes_to_mont((pw * gate_2).reduce_fast());
                        t_hat_inner[t][2] = E::sum_lanes_to_mont((pw * gate_inf).reduce_fast());
                    }
                    let s_hat = compute_round_s_from_t::<E>(prefix_eq, point[round], t_hat_inner, one, two);
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

        // round_misc_us is computed by the caller as the residual:
        //   `prove_profiled_total - (tree_build + top_extract + eq_tables
        //    + round_t + fold + inreg_fold + scalar_tail)`.
        // It rolls up all the small per-round serial overhead (compute_s,
        // append_hat, challenge_f, eq_linear_mont, mem::take, per-level
        // final transcript append + challenge draw) into a single visible
        // bucket.

        point
    }

    /// Per-wire Batched ProductCheck verifier.
    ///
    /// All arithmetic uses lazy ops to maintain correct Montgomery ranges.
    /// Proof values (normal form) are converted to Montgomery on read.
    /// Returns `(point, y)` in normal form.
    pub fn verify<E: SumcheckExtField>(
        nv: usize,
        transcript: &mut Transcript,
        proof: &mut util::fiat_shamir::Proof,
    ) -> (Vec<E::Scalar>, [E::Scalar; NUM_TREES], E::Scalar) {
        use arithmetic::field::mamabear::P;

        assert!(nv >= 4);

        // .reduce() canonicalizes c1=P→0 in extension field constants.
        // Without this, the P bias in c1 causes cumulative errors in
        // chained quadratic (degree-2) extension multiplications (Lagrange basis sum ≠ 1).
        let one = E::Scalar::one().to_montgomery().reduce();
        let two = E::Scalar::from(2u32).to_montgomery().reduce();
        let three = E::Scalar::from(3u32).to_montgomery().reduce();
        let six = E::Scalar::from(6u32).to_montgomery().reduce();
        let inv2 = E::Scalar::from(MamaBearScalar((P + 1) / 2)).to_montgomery().reduce();
        let inv6 = E::Scalar::from(MamaBearScalar((P + 1) / 6)).to_montgomery().reduce();
        let neg_inv2 = (-inv2).reduce();
        let neg_inv6 = (-inv6).reduce();

        // Helper: read scalar from proof, append to transcript, return Montgomery form
        macro_rules! read_mont {
            ($proof:expr, $transcript:expr) => {{
                let v: E::Scalar = $proof.get_next_and_step();
                $transcript.append_f(v);
                v.to_montgomery()
            }};
        }

        // =====================================================================
        // Read top-level half-products and full products
        // =====================================================================
        let mut tau = [[[E::Scalar::zero(); 2]; NUM_WIRES]; NUM_TREES];
        for t in 0..NUM_TREES {
            for g in 0..NUM_WIRES {
                for k in 0..2 {
                    tau[t][g][k] = read_mont!(proof, transcript);
                }
            }
        }
        let mut fp = [E::Scalar::zero(); NUM_TREES];
        for t in 0..NUM_TREES {
            fp[t] = read_mont!(proof, transcript);
        }

        // Bind each top full-product `fp[t]` to the committed half-products `tau`.
        //
        // `fp[t]` is the claimed product of all top-level half-products of tree t
        // (Π_g tau[t][g][0]·tau[t][g][1]). The prover sends `fp` and `tau`
        // independently, so without this check `fp` is unconstrained and the later
        // `fp[0] == fp[1]` equality is vacuous — a malicious prover could pick any
        // `fp[0] == fp[1]` decoupled from the `tau` that seed the sumcheck's initial
        // claim. Recompute the product from `tau` and compare in canonical normal
        // form, which neutralizes the `mont_mul(0,0)=P` bias. `tau` is in Montgomery
        // [0,2P); reducing each to [0,P) keeps every `mont_mul` of two <2P inputs in
        // [0,1.5P), so the running product never leaves [0,2P) — no overflow.
        for t in 0..NUM_TREES {
            let mut prod = one; // [0, P)
            for g in 0..NUM_WIRES {
                let a = tau[t][g][0].reduce(); // [0, P)
                let b = tau[t][g][1].reduce(); // [0, P)
                prod = prod * a * b; // stays in [0, 1.5P)
            }
            assert_eq!(
                prod.reduce().from_montgomery(),
                fp[t].reduce().from_montgomery(),
                "fp/tau binding failed: fp[{t}] != Π_g tau[{t}][g][0]·tau[{t}][g][1]"
            );
        }

        assert_eq!(
            fp[0].from_montgomery(),
            fp[1].from_montgomery(),
            "Product equality check failed"
        );

        // =====================================================================
        // Challenges
        // =====================================================================
        let r_0 = transcript.challenge_f::<E::Scalar>().to_montgomery();
        let lambda = transcript.challenge_f::<E::Scalar>().to_montgomery();
        let lambda2 = lambda * lambda; // [0, 1.5P)

        // Initial claimed sums: y[t] = Σ_g λ^g · lerp(tau[t][g][0], tau[t][g][1], r_0)
        // All tau values from read_mont are in [0, 2P) (to_montgomery output).
        let lambda_w = [one, lambda.reduce(), lambda2.reduce()]; // all in [0, P)
        let mut y = [E::Scalar::zero(); NUM_TREES];
        for t in 0..NUM_TREES {
            let mut sum = E::Scalar::zero(); // [0, P) initially
            for g in 0..NUM_WIRES {
                let t0 = tau[t][g][0].reduce(); // [0, P)
                let t1 = tau[t][g][1].reduce(); // [0, P)
                // diff = t1 - t0 ∈ [0, 2P) via lazy sub
                let diff = t1.lazy_add_xp(2).lazy_sub(t0).con_sub_xp(2);
                // interp = t0 + diff * r_0: mont_mul(diff∈[0,2P), r_0∈[0,2P)) → [0,1.5P)
                let interp = (diff * r_0).lazy_add(t0).reduce(); // [0, P)
                // λ^g * interp: inputs in [0, P) → product in [0, 1.5P) → reduce → [0, P)
                sum = sum.lazy_add(lambda_w[g] * interp).reduce(); // [0, P)
            }
            y[t] = sum; // [0, P)
        }

        let mut point = vec![r_0]; // Montgomery

        // =====================================================================
        // Iterative sumcheck verification (top → bottom)
        // =====================================================================
        for _level_idx in (0..nv - 1).rev() {
            let num_vars = point.len();
            let mut challenges = Vec::with_capacity(num_vars);

            for _round_idx in 0..num_vars {
                // Read 6 hat values (3 per tree), convert to Montgomery
                let mut s = [[E::Scalar::zero(); 3]; NUM_TREES];
                for t in 0..NUM_TREES {
                    for h in 0..3 {
                        s[t][h] = read_mont!(proof, transcript);
                    }
                }

                let ch = transcript.challenge_f::<E::Scalar>().to_montgomery();
                challenges.push(ch);

                for t in 0..NUM_TREES {
                    // Reduce inputs for clean range tracking
                    let s0 = s[t][0].reduce(); // s(0) ∈ [0, P)
                    let s1 = y[t].lazy_add_xp(2).lazy_sub(s0).con_sub_xp(2).reduce(); // s(1) = y - s(0) ∈ [0, P)
                    let s2 = s[t][1].reduce(); // s(2) ∈ [0, P)
                    let a3 = s[t][2].reduce(); // s(∞) = leading coeff ∈ [0, P)

                    // s(X) is degree 3. s_hat[2] = a₃ (leading coefficient).
                    // Δ³s = 6·a₃, so s(3) = 6·a₃ + 3·s(2) - 3·s(1) + s(0).
                    let s3 = (six * a3) // 6·a₃ ∈ [0, 1.5P)
                        .lazy_add(three * s2) // + 3·s(2) ∈ [0, 1.5P) → [0, 3P)
                        .lazy_add_xp(4) // + 4P → [4P, 7P)
                        .lazy_sub(three * s1) // - [0, 1.5P) → [2.5P, 7P)
                        .lazy_add(s0) // + [0, P) → [2.5P, 8P)
                        .reduce(); // [0, P)

                    // Lagrange interpolation at X = ch over {0, 1, 2, 3}
                    let r = ch.reduce(); // [0, P)
                    let r_m1 = r.lazy_add_xp(2).lazy_sub(one).con_sub_xp(2); // [0, 2P)
                    let r_m2 = r.lazy_add_xp(2).lazy_sub(two).con_sub_xp(2); // [0, 2P)
                    let r_m3 = r.lazy_add_xp(2).lazy_sub(three).con_sub_xp(2); // [0, 2P)

                    // L_0 = (r-1)(r-2)(r-3) / (-6)
                    let l0 = (r_m1 * r_m2 * r_m3 * neg_inv6).reduce();
                    // L_1 = r(r-2)(r-3) / 2
                    let l1 = (r * r_m2 * r_m3 * inv2).reduce();
                    // L_2 = r(r-1)(r-3) / (-2)
                    let l2 = (r * r_m1 * r_m3 * neg_inv2).reduce();
                    // L_3 = r(r-1)(r-2) / 6
                    let l3 = (r * r_m1 * r_m2 * inv6).reduce();

                    // y[t] = Σ l_i · s_i, all in [0, P) before mul → products in [0, 1.5P)
                    y[t] = (l0 * s0)
                        .lazy_add(l1 * s1)
                        .lazy_add(l2 * s2)
                        .lazy_add(l3 * s3)
                        .reduce(); // [0, P)
                }
            }

            // Read 12 final evals (3 wires × 2 values × 2 trees) in Montgomery
            let mut v = [[E::Scalar::zero(); 2 * NUM_WIRES]; NUM_TREES];
            for t in 0..NUM_TREES {
                for j in 0..2 * NUM_WIRES {
                    v[t][j] = read_mont!(proof, transcript);
                }
            }

            // =================================================================
            // Per-level oracle check (the reference cross-level binding).
            //
            // The sumcheck of this level reduces the incoming claim `y_init` down
            // to the running claim `y[t]` (the cascade above). Because the prover
            // now folds the per-level sumcheck in NATURAL variable order — the
            // within-register LANE bits and the BLOCK bits are eq-weighted in the
            // exact order they are folded (see `compute_round_t_perwire`) — the
            // round-0 consistency `s(0)+s(1) == y_init` holds for honest proofs,
            // and the reduced claim telescopes to
            //
            //     y[t] == leaf_gate[t] · eq(point, challenges),
            //
            // where leaf_gate[t] = Σ_g λ^g · v_even_g · v_odd_g and
            // eq(point, challenges) = Π_r eq_linear(point[r], challenges[r]).
            //
            // This single equation binds the incoming claim (which was formed from
            // the PARENT level's leaf evals) to THIS level's leaf evals `v`. Chained
            // across all levels, it forces every intermediate product-tree node to
            // be consistent with its children — closing the cross-level hole where a
            // malicious prover could pick free intermediate leaves whose product is
            // unequal while still proving `fp[0] == fp[1]`. It also subsumes the
            // per-round leaf binding (a tampered `v` fails this check).
            //
            // `s(1)` is derived (not sent) via `s(0)+s(1)=y`, so the round-0
            // consistency is enforced by construction; the oracle check below binds
            // the whole reduced chain to the leaves. All operands are reduced to
            // canonical form before comparison to neutralize the `mont_mul(0,0)=P`
            // bias from the prover's inactive packed lanes.
            {
                // full_prefix = Π_r eq_linear(point[r], challenges[r]) ∈ [0, P)
                let mut full_prefix = one;
                for r in 0..num_vars {
                    let e = SumcheckMamaBear::eq_linear_mont_generic::<E>(
                        point[r],
                        challenges[r],
                        one,
                    );
                    full_prefix = (full_prefix * e).reduce(); // [0, P)
                }
                for t in 0..NUM_TREES {
                    // leaf_gate = Σ_g λ^g · v_even_g · v_odd_g ∈ [0, P)
                    let mut leaf_gate = E::Scalar::zero();
                    for g in 0..NUM_WIRES {
                        let ve = v[t][2 * g].reduce(); // [0, P)
                        let vo = v[t][2 * g + 1].reduce(); // [0, P)
                        let prod = (ve * vo).reduce(); // [0, P)
                        leaf_gate = leaf_gate.lazy_add(lambda_w[g] * prod).reduce(); // [0, P)
                    }
                    let expected = (leaf_gate * full_prefix).reduce(); // [0, P)
                    assert_eq!(
                        expected.from_montgomery(),
                        y[t].reduce().from_montgomery(),
                        "prodcheck oracle check failed: level num_vars={num_vars} tree {t}"
                    );
                }
            }

            // Update y for next level: y[t] = Σ_g λ^g · (v_even_g + (v_odd_g - v_even_g) · r)
            let r = transcript.challenge_f::<E::Scalar>().to_montgomery();
            for t in 0..NUM_TREES {
                let mut new_y = E::Scalar::zero();
                for g in 0..NUM_WIRES {
                    let ve = v[t][2 * g].reduce(); // [0, P)
                    let vo = v[t][2 * g + 1].reduce(); // [0, P)
                    let diff = vo.lazy_add_xp(2).lazy_sub(ve).con_sub_xp(2); // [0, 2P)
                    let interp = (diff * r).lazy_add(ve).reduce(); // [0, P)
                    new_y = new_y.lazy_add(lambda_w[g] * interp).reduce(); // [0, P)
                }
                y[t] = new_y;
            }

            let mut new_point = vec![r];
            new_point.extend(challenges);
            point = new_point;
        }

        // Convert results to normal form
        let point_normal = point
            .iter()
            .map(|x| x.reduce().from_montgomery())
            .collect();
        let y_normal = [
            y[0].reduce().from_montgomery(),
            y[1].reduce().from_montgomery(),
        ];
        (point_normal, y_normal, lambda.reduce().from_montgomery())
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use arithmetic::field::mamabear::PackedMamaBearAVX512Ext3;
    use arithmetic::field::Field;
    use rand::rngs::SmallRng;
    use rand::SeedableRng;
    use util::fiat_shamir::Transcript;

    type PEF3 = PackedMamaBearAVX512Ext3;

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

    fn run_determinism_test<E: SumcheckExtField>(seed: u64, nv: usize) {
        let n = 1 << nv;
        let mut rng = SmallRng::seed_from_u64(seed);
        // Generate 6 random vectors (3 wires × 2 trees)
        let evals: [[Vec<E::Scalar>; NUM_WIRES]; NUM_TREES] = std::array::from_fn(|_| {
            std::array::from_fn(|_| (0..n).map(|_| E::Scalar::random(&mut rng)).collect())
        });
        let packed1: [[Vec<E>; NUM_WIRES]; NUM_TREES] =
            std::array::from_fn(|t| std::array::from_fn(|g| pack_input::<E>(&evals[t][g])));
        let packed2: [[Vec<E>; NUM_WIRES]; NUM_TREES] =
            std::array::from_fn(|t| std::array::from_fn(|g| pack_input::<E>(&evals[t][g])));

        let mut t1 = Transcript::new();
        let r1 = ProdEqCheckMamaBearPerWire::prove::<E>(packed1, &mut t1);
        let mut t2 = Transcript::new();
        let r2 = ProdEqCheckMamaBearPerWire::prove::<E>(packed2, &mut t2);
        assert_eq!(t1.proof.bytes, t2.proof.bytes, "Transcript mismatch");
        assert_eq!(r1, r2, "Point mismatch");
    }

    #[test]
    fn perwire_ext3_nv4() {
        run_determinism_test::<PEF3>(77, 4);
    }
    #[test]
    fn perwire_ext3_nv8() {
        run_determinism_test::<PEF3>(999, 8);
    }
    #[test]
    fn perwire_ext3_nv12() {
        run_determinism_test::<PEF3>(2024, 12);
    }

    // =========================================================================
    // Prove + Verify tests
    // =========================================================================

    fn run_prove_verify_test<E: SumcheckExtField>(seed: u64, nv: usize) {
        let n = 1 << nv;
        let mut rng = SmallRng::seed_from_u64(seed);

        // Generate tree 0 data randomly, tree 1 = reversed (same product per wire)
        let evals_t0: [Vec<E::Scalar>; NUM_WIRES] =
            std::array::from_fn(|_| (0..n).map(|_| E::Scalar::random(&mut rng)).collect());
        let evals_t1: [Vec<E::Scalar>; NUM_WIRES] = std::array::from_fn(|g| {
            evals_t0[g].iter().copied().rev().collect()
        });

        let packed: [[Vec<E>; NUM_WIRES]; NUM_TREES] = [
            std::array::from_fn(|g| pack_input::<E>(&evals_t0[g])),
            std::array::from_fn(|g| pack_input::<E>(&evals_t1[g])),
        ];

        // Prove
        let mut transcript = Transcript::new();
        let point = ProdEqCheckMamaBearPerWire::prove::<E>(packed, &mut transcript);

        // Verify
        let mut proof = transcript.proof;
        let mut v_transcript = Transcript::new();
        let (v_point, _v_y, _lambda) =
            ProdEqCheckMamaBearPerWire::verify::<E>(nv, &mut v_transcript, &mut proof);

        // prove returns Montgomery domain; verify returns normal domain — convert for comparison
        let point_normal: Vec<_> = point.into_iter().map(|x| x.from_montgomery()).collect();
        assert_eq!(point_normal, v_point, "Point mismatch between prove and verify");

        // The reduced point recovers the input MLE via `simd_to_natural_point`
        // (right-rotate by the SIMD lane-log w=3), the SAME convention the add/mul
        // main sumcheck uses. Confirm prod_y[t] == Σ_g λ^g · MLE(input_{t,g}, nat).
        use arithmetic::poly::MultiLinearPoly;
        let nat: Vec<E::Scalar> = crate::prover_mamabear::simd_to_natural_point(&v_point)
            .iter()
            .map(|x| x.to_montgomery())
            .collect();
        let lam = _lambda.to_montgomery();
        let lw = [E::Scalar::one().to_montgomery(), lam, lam * lam];
        for t in 0..NUM_TREES {
            let evals_t = if t == 0 { &evals_t0 } else { &evals_t1 };
            let mut acc = E::Scalar::zero();
            for g in 0..NUM_WIRES {
                let ev_m: Vec<E::Scalar> = evals_t[g].iter().map(|x| x.to_montgomery()).collect();
                acc = acc + lw[g] * MultiLinearPoly::eval_multilinear_ext(&ev_m, &nat);
            }
            assert_eq!(
                acc.reduce().from_montgomery(),
                _v_y[t],
                "prod_y[{t}] must equal Σ_g λ^g · MLE(input, simd_to_natural(point))"
            );
        }
    }

    #[test]
    fn perwire_verify_ext3_nv4() {
        run_prove_verify_test::<PEF3>(42, 4);
    }
    #[test]
    fn perwire_verify_ext3_nv8() {
        run_prove_verify_test::<PEF3>(123, 8);
    }
    #[test]
    fn perwire_verify_ext3_nv12() {
        run_prove_verify_test::<PEF3>(2024, 12);
    }
    #[test]
    fn perwire_verify_ext3_nv16() {
        run_prove_verify_test::<PEF3>(7, 16);
    }
    #[test]
    fn perwire_verify_ext3_nv20() {
        run_prove_verify_test::<PEF3>(7, 20);
    }

    // =========================================================================
    // Soundness PoCs (must REJECT). Release-active (plain assert! in verify).
    // =========================================================================

    /// Prove an honest instance and return the flat proof.
    fn honest_proof<E: SumcheckExtField>(seed: u64, nv: usize) -> util::fiat_shamir::Proof {
        let n = 1 << nv;
        let mut rng = SmallRng::seed_from_u64(seed);
        let evals_t0: [Vec<E::Scalar>; NUM_WIRES] =
            std::array::from_fn(|_| (0..n).map(|_| E::Scalar::random(&mut rng)).collect());
        let evals_t1: [Vec<E::Scalar>; NUM_WIRES] =
            std::array::from_fn(|g| evals_t0[g].iter().copied().rev().collect());
        let packed: [[Vec<E>; NUM_WIRES]; NUM_TREES] = [
            std::array::from_fn(|g| pack_input::<E>(&evals_t0[g])),
            std::array::from_fn(|g| pack_input::<E>(&evals_t1[g])),
        ];
        let mut transcript = Transcript::new();
        let _ = ProdEqCheckMamaBearPerWire::prove::<E>(packed, &mut transcript);
        transcript.proof
    }

    /// Forged top full-products: bumping BOTH `fp[0]` and `fp[1]` by the same
    /// amount keeps `fp[0] == fp[1]` (equal products serialize identically) but
    /// breaks `fp == Π tau`, so the fp/tau binding must reject.
    fn run_forged_fp_poc<E: SumcheckExtField>(seed: u64, nv: usize) {
        let mut proof = honest_proof::<E>(seed, nv);
        let sz = <E::Scalar as Field>::SIZE;
        // Layout: 12 tau scalars (NUM_TREES*NUM_WIRES*2), then fp[0], fp[1].
        let fp0 = 12 * sz;
        let fp1 = 13 * sz;
        assert_eq!(
            proof.bytes[fp0..fp0 + sz],
            proof.bytes[fp1..fp1 + sz],
            "test precondition: fp[0] and fp[1] bytes must be equal"
        );
        proof.bytes[fp0] = proof.bytes[fp0].wrapping_add(1);
        proof.bytes[fp1] = proof.bytes[fp1].wrapping_add(1);
        let mut vt = Transcript::new();
        // Must panic on the fp/tau binding assert.
        let _ = ProdEqCheckMamaBearPerWire::verify::<E>(nv, &mut vt, &mut proof);
    }

    /// Tamper one leaf eval of the deepest level (the last 12 ext scalars of the
    /// proof). Must trip the per-level oracle check for that level.
    fn run_tampered_leaf_poc<E: SumcheckExtField>(seed: u64, nv: usize) {
        let mut proof = honest_proof::<E>(seed, nv);
        let sz = <E::Scalar as Field>::SIZE;
        let total = proof.bytes.len();
        // The final 12 (= 2*NUM_WIRES*NUM_TREES) scalars are the deepest level's
        // leaf evals; tamper the first byte of the first one.
        let off = total - 12 * sz;
        proof.bytes[off] = proof.bytes[off].wrapping_add(1);
        let mut vt = Transcript::new();
        // Must panic on the oracle check.
        let _ = ProdEqCheckMamaBearPerWire::verify::<E>(nv, &mut vt, &mut proof);
    }

    /// Decoupled intermediate product: tamper an INTERMEDIATE, NON-LAST round
    /// polynomial hat of an inner sumcheck level. This is the load-bearing test
    /// for the cross-level binding: the earlier verify only bound each level's
    /// LEAVES to its LAST round polynomial, leaving the middle round polynomials
    /// (and hence the intermediate product-tree nodes) unbound to the incoming
    /// claim. A malicious prover could therefore decouple an intermediate node
    /// from its children while keeping the top `fp` forged-equal and the bottom
    /// leaves committed. The restored per-level oracle check (which telescopes
    /// the FULL cascade `y_reduced == leaf_gate · eq(point, challenges)`) rejects
    /// any such decoupling. Here we tamper the round-0 hat of the `num_vars == 2`
    /// level (its round 1 is the last round, so round 0 is a genuine middle round
    /// that the last-round-only leaf binding would miss).
    fn run_decoupled_intermediate_poc<E: SumcheckExtField>(seed: u64, nv: usize) {
        assert!(nv >= 3, "need at least a num_vars==2 sumcheck level");
        let mut proof = honest_proof::<E>(seed, nv);
        let sz = <E::Scalar as Field>::SIZE;
        // Proof layout: 12 tau + 2 fp (= 14 scalars), then per level from
        // num_vars=1 upward: [6 hats/round * num_vars rounds] + [12 leaf evals].
        // The num_vars==1 level contributes 6 + 12 = 18 scalars, so the
        // num_vars==2 level's round-0 first hat starts at scalar index 14+18=32.
        let off = 32 * sz;
        proof.bytes[off] = proof.bytes[off].wrapping_add(1);
        let mut vt = Transcript::new();
        // Must panic: the oracle check at the num_vars==2 level (or an earlier
        // Fiat-Shamir-divergent check) rejects the decoupled round polynomial.
        let _ = ProdEqCheckMamaBearPerWire::verify::<E>(nv, &mut vt, &mut proof);
    }

    #[test]
    #[should_panic]
    fn perwire_soundness_forged_fp_ext3() {
        run_forged_fp_poc::<PEF3>(99, 8);
    }
    #[test]
    #[should_panic]
    fn perwire_soundness_tampered_leaf_ext3() {
        run_tampered_leaf_poc::<PEF3>(99, 8);
    }
    #[test]
    #[should_panic]
    fn perwire_soundness_decoupled_intermediate_ext3() {
        run_decoupled_intermediate_poc::<PEF3>(99, 8);
    }
}
