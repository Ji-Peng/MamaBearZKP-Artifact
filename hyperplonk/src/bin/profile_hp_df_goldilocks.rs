use std::{cmp::Reverse, fs, path::PathBuf, time::Instant};

use arithmetic::{
    field::{
        goldilocks64::{Goldilocks64, Goldilocks64Ext},
        Field,
    },
    poly::MultiLinearPoly,
    mul_group::Radix2Group,
};
use hyperplonk::{
    circuit::Circuit,
    prodcheck::ProdEqCheck,
    prover::Prover,
    sumcheck::Sumcheck,
    verifier::Verifier,
};
use poly_commit::{deepfold::{DeepFoldParam, DeepFoldProver, DeepFoldVerifier}, CommitmentSerde, PolyCommitProver};
use rand::rngs::SmallRng;
use rand::SeedableRng;

type DeepFoldGoldilocksProver = Prover<Goldilocks64Ext, DeepFoldProver<Goldilocks64Ext>>;
type DeepFoldGoldilocksVerifier = Verifier<Goldilocks64Ext, DeepFoldVerifier<Goldilocks64Ext>>;

#[derive(Clone, Debug, Default)]
struct ProverProfile {
    domain_size: usize,
    base_field_bytes: usize,
    ext_field_bytes: usize,
    witness_input_bytes: usize,
    bookkeeping_bytes: usize,
    eq_r_bytes: usize,
    witness_flatten_bytes: usize,
    identical_bytes: usize,
    permutation_bytes: usize,
    product_eval_bytes: usize,
    final_sumcheck_input_bytes: usize,
    witness_pc_new_us: u128,
    witness_commit_us: u128,
    commit_serialize_us: u128,
    bookkeeping_build_us: u128,
    eq_r_build_us: u128,
    zerocheck_us: u128,
    witness_flatten_build_us: u128,
    identical_build_us: u128,
    permutation_flatten_build_us: u128,
    product_eval_build_us: u128,
    productcheck_us: u128,
    product_point_eval_append_us: u128,
    final_sumcheck_input_build_us: u128,
    final_sumcheck_us: u128,
    final_point_eval_append_us: u128,
    pcs_open_us: u128,
    total_us: u128,
    proof_bytes: usize,
}

impl ProverProfile {
    fn stage_rows(&self) -> [(&'static str, u128); 16] {
        [
            ("witness_pc_new", self.witness_pc_new_us),
            ("witness_commit", self.witness_commit_us),
            ("commit_serialize", self.commit_serialize_us),
            ("bookkeeping_build", self.bookkeeping_build_us),
            ("eq_r_build", self.eq_r_build_us),
            ("zerocheck", self.zerocheck_us),
            ("witness_flatten_build", self.witness_flatten_build_us),
            ("identical_build", self.identical_build_us),
            ("permutation_flatten_build", self.permutation_flatten_build_us),
            ("product_eval_build", self.product_eval_build_us),
            ("productcheck", self.productcheck_us),
            ("product_point_eval_append", self.product_point_eval_append_us),
            ("final_sumcheck_input_build", self.final_sumcheck_input_build_us),
            ("final_sumcheck", self.final_sumcheck_us),
            ("final_point_eval_append", self.final_point_eval_append_us),
            ("pcs_open", self.pcs_open_us),
        ]
    }

    fn construction_rows(&self) -> [(&'static str, usize); 6] {
        [
            ("witness_input", self.witness_input_bytes),
            ("bookkeeping", self.bookkeeping_bytes),
            ("eq_r", self.eq_r_bytes),
            ("witness_flatten", self.witness_flatten_bytes),
            ("identical_plus_permutation", self.identical_bytes + self.permutation_bytes),
            (
                "product_evals_plus_final_sumcheck_inputs",
                self.product_eval_bytes + self.final_sumcheck_input_bytes,
            ),
        ]
    }
}

fn measure<T>(f: impl FnOnce() -> T) -> (T, u128) {
    let start = Instant::now();
    let value = f();
    (value, start.elapsed().as_micros())
}

fn prove_with_profile(
    prover: &DeepFoldGoldilocksProver,
    pp: &DeepFoldParam<Goldilocks64Ext>,
    nv: usize,
    witness: [Vec<Goldilocks64>; 3],
) -> (util::fiat_shamir::Proof, ProverProfile) {
    let total_start = Instant::now();
    let domain_size = 1usize << nv;
    let mut profile = ProverProfile {
        domain_size,
        base_field_bytes: Goldilocks64::SIZE,
        ext_field_bytes: Goldilocks64Ext::SIZE,
        witness_input_bytes: 3 * domain_size * Goldilocks64::SIZE,
        bookkeeping_bytes: 3 * domain_size * Goldilocks64Ext::SIZE,
        eq_r_bytes: domain_size * Goldilocks64Ext::SIZE,
        witness_flatten_bytes: 4 * domain_size * Goldilocks64Ext::SIZE,
        identical_bytes: 4 * domain_size * Goldilocks64::SIZE,
        permutation_bytes: 4 * domain_size * Goldilocks64::SIZE,
        product_eval_bytes: 8 * domain_size * Goldilocks64Ext::SIZE,
        final_sumcheck_input_bytes: 4 * domain_size * Goldilocks64Ext::SIZE,
        ..Default::default()
    };
    let mut transcript = util::fiat_shamir::Transcript::new();

    let (witness_pc, elapsed) = measure(|| DeepFoldProver::new(pp, &witness));
    profile.witness_pc_new_us = elapsed;
    let (commit, elapsed) = measure(|| witness_pc.commit());
    profile.witness_commit_us = elapsed;
    let mut buffer = vec![0u8; <DeepFoldProver<Goldilocks64Ext> as PolyCommitProver<Goldilocks64Ext>>::Commitment::size(nv, 3)];
    let (_, elapsed) = measure(|| {
        commit.serialize_into(&mut buffer);
        transcript.append_u8_slice(
            &buffer,
            <DeepFoldProver<Goldilocks64Ext> as PolyCommitProver<Goldilocks64Ext>>::Commitment::size(nv, 3),
        );
    });
    profile.commit_serialize_us = elapsed;

    let (bookkeeping, elapsed) = measure(|| {
        witness
            .clone()
            .map(|x| x.into_iter().map(Goldilocks64Ext::from).collect::<Vec<_>>())
    });
    profile.bookkeeping_build_us = elapsed;

    let r = (0..nv)
        .map(|_| transcript.challenge_f::<Goldilocks64Ext>())
        .collect::<Vec<_>>();
    let (eq_r, elapsed) = measure(|| MultiLinearPoly::new_eq(&r));
    profile.eq_r_build_us = elapsed;

    let ((sumcheck_point, v), elapsed) = measure(|| {
        Sumcheck::prove(
            [
                prover
                    .prover_key
                    .selector
                    .evals
                    .iter()
                    .map(|x| Goldilocks64Ext::from(*x))
                    .collect(),
                bookkeeping[0].clone(),
                bookkeeping[1].clone(),
                bookkeeping[2].clone(),
                eq_r.evals.clone(),
            ],
            4,
            &mut transcript,
            |v: [Goldilocks64Ext; 5]| {
                [v[4]
                    * ((Goldilocks64Ext::one() - v[0]) * (v[1] + v[2])
                        + v[0] * v[1] * v[2]
                        + v[3])]
            },
        )
    });
    profile.zerocheck_us = elapsed;

    for value in v.iter().take(4) {
        transcript.append_f(*value);
    }

    let (witness_flatten, elapsed) = measure(|| {
        bookkeeping[0]
            .clone()
            .into_iter()
            .chain(bookkeeping[1].clone())
            .chain(bookkeeping[2].clone())
            .chain((0..domain_size).map(|_| Goldilocks64Ext::zero()))
            .collect::<Vec<_>>()
    });
    profile.witness_flatten_build_us = elapsed;
    let (identical, elapsed) = measure(|| {
        MultiLinearPoly::new_identical(nv, Goldilocks64::zero())
            .evals
            .into_iter()
            .chain(MultiLinearPoly::new_identical(nv, Goldilocks64::from(1u32 << 29)).evals)
            .chain(MultiLinearPoly::new_identical(nv, Goldilocks64::from(1u32 << 30)).evals)
            .chain((0..domain_size).map(|_| Goldilocks64::zero()))
            .collect::<Vec<_>>()
    });
    profile.identical_build_us = elapsed;
    let (permutation, elapsed) = measure(|| {
        prover.prover_key.permutation[0]
            .clone()
            .evals
            .into_iter()
            .chain(prover.prover_key.permutation[1].clone().evals)
            .chain(prover.prover_key.permutation[2].clone().evals)
            .chain((0..domain_size).map(|_| Goldilocks64::zero()))
            .collect::<Vec<_>>()
    });
    profile.permutation_flatten_build_us = elapsed;

    let r = [0; 2].map(|_| transcript.challenge_f::<Goldilocks64Ext>());
    let ((evals1, evals2), elapsed) = measure(|| {
        (
            witness_flatten
                .iter()
                .zip(identical.iter())
                .map(|(&x, &y)| r[0] + x + r[1].mul_base_elem(y))
                .collect::<Vec<_>>(),
            witness_flatten
                .iter()
                .zip(permutation.iter())
                .map(|(&x, &y)| r[0] + x + r[1].mul_base_elem(y))
                .collect::<Vec<_>>(),
        )
    });
    profile.product_eval_build_us = elapsed;
    let (prod_point, elapsed) = measure(|| ProdEqCheck::prove([evals1, evals2], &mut transcript));
    profile.productcheck_us = elapsed;

    let (_, elapsed) = measure(|| {
        for i in 0..3 {
            transcript.append_f(MultiLinearPoly::eval_multilinear(&witness[i], &prod_point[..nv]));
        }
        for i in 0..3 {
            transcript.append_f(MultiLinearPoly::eval_multilinear(
                &prover.prover_key.permutation[i].evals,
                &prod_point[..nv],
            ));
        }
    });
    profile.product_point_eval_append_us = elapsed;

    let r: Goldilocks64Ext = transcript.challenge_f();
    let r2 = r * r;
    let r3 = r2 * r;
    let r4 = r3 * r;
    let r5 = r4 * r;
    let ((poly0, poly1, eq_sum, eq_prod), elapsed) = measure(|| {
        (
            prover
                .prover_key
                .selector
                .evals
                .iter()
                .zip(witness[0].iter())
                .zip(witness[1].iter())
                .zip(witness[2].iter())
                .map(|(((&x1, &x2), &x3), &x4)| {
                    Goldilocks64Ext::from(x1)
                        + r.mul_base_elem(x2)
                        + r2.mul_base_elem(x3)
                        + r3.mul_base_elem(x4)
                })
                .collect::<Vec<_>>(),
            prover
                .prover_key
                .permutation[0]
                .evals
                .iter()
                .zip(prover.prover_key.permutation[1].evals.iter())
                .zip(prover.prover_key.permutation[2].evals.iter())
                .zip(witness[0].iter())
                .zip(witness[1].iter())
                .zip(witness[2].iter())
                .map(|(((((&x1, &x2), &x3), &x4), &x5), &x6)| {
                    Goldilocks64Ext::from(x1)
                        + r.mul_base_elem(x2)
                        + r2.mul_base_elem(x3)
                        + r3.mul_base_elem(x4)
                        + r4.mul_base_elem(x5)
                        + r5.mul_base_elem(x6)
                })
                .collect::<Vec<_>>(),
            MultiLinearPoly::new_eq(&sumcheck_point).evals,
            MultiLinearPoly::new_eq(&prod_point[..nv].to_vec()).evals,
        )
    });
    profile.final_sumcheck_input_build_us = elapsed;
    let ((point, _), elapsed) = measure(|| {
        Sumcheck::prove(
            [poly0, poly1, eq_sum, eq_prod],
            2,
            &mut transcript,
            |v: [Goldilocks64Ext; 4]| [v[0] * v[2], v[1] * v[3]],
        )
    });
    profile.final_sumcheck_us = elapsed;

    let (_, elapsed) = measure(|| {
        transcript.append_f(MultiLinearPoly::eval_multilinear(
            &prover.prover_key.selector.evals,
            &point,
        ));
        for i in 0..3 {
            transcript.append_f(MultiLinearPoly::eval_multilinear(
                &prover.prover_key.permutation[i].evals,
                &point,
            ));
        }
        for i in 0..3 {
            transcript.append_f(MultiLinearPoly::eval_multilinear(&witness[i], &point));
        }
    });
    profile.final_point_eval_append_us = elapsed;

    let (_, elapsed) = measure(|| {
        DeepFoldProver::open(
            pp,
            vec![&prover.prover_key.commitments, &witness_pc],
            point,
            &mut transcript,
        );
    });
    profile.pcs_open_us = elapsed;
    profile.total_us = total_start.elapsed().as_micros();
    profile.proof_bytes = transcript.proof.bytes.len();

    (transcript.proof, profile)
}

struct ProfileCase {
    prover: DeepFoldGoldilocksProver,
    verifier: DeepFoldGoldilocksVerifier,
    pp: DeepFoldParam<Goldilocks64Ext>,
    witness: [Vec<Goldilocks64>; 3],
}

fn build_mult_subgroups(nv: u32) -> Vec<Radix2Group<Goldilocks64>> {
    let mut mult_subgroups = vec![Radix2Group::<Goldilocks64>::new(nv + 2)];
    for i in 1..nv as usize {
        mult_subgroups.push(mult_subgroups[i - 1].exp(2));
    }
    mult_subgroups
}

fn build_case(nv: usize) -> ProfileCase {
    let mut rng = SmallRng::seed_from_u64(1);
    let num_gates = 1u32 << nv;
    let circuit = Circuit::<Goldilocks64Ext> {
        permutation: [
            (0..num_gates).map(|x| x.into()).collect(),
            (0..num_gates).map(|x| (x + (1 << 29)).into()).collect(),
            (0..num_gates).map(|x| (x + (1 << 30)).into()).collect(),
        ],
        selector: (0..num_gates).map(|x| (x & 1).into()).collect(),
    };
    let pp = DeepFoldParam::<Goldilocks64Ext> {
        mult_subgroups: build_mult_subgroups(nv as u32),
        variable_num: nv,
        query_num: 45,
    };
    let (pk, vk) = circuit.setup::<DeepFoldProver<_>, DeepFoldVerifier<_>>(&pp, &pp);
    let prover = Prover { prover_key: pk };
    let verifier = Verifier { verifier_key: vk };
    let a = (0..num_gates)
        .map(|_| Goldilocks64::random(&mut rng))
        .collect::<Vec<_>>();
    let b = (0..num_gates)
        .map(|_| Goldilocks64::random(&mut rng))
        .collect::<Vec<_>>();
    let c = (0..num_gates)
        .map(|i| {
            let i = i as usize;
            let s = circuit.selector[i];
            -((Goldilocks64::one() - s) * (a[i] + b[i]) + s * a[i] * b[i])
        })
        .collect::<Vec<_>>();

    ProfileCase {
        prover,
        verifier,
        pp,
        witness: [a, b, c],
    }
}

fn repetitions_for_nv(nv: usize) -> usize {
    if nv <= 20 {
        3
    } else if nv <= 22 {
        2
    } else {
        1
    }
}

fn average_profiles(profiles: &[ProverProfile]) -> ProverProfile {
    let len = profiles.len() as u128;
    let sum_u128 = |f: fn(&ProverProfile) -> u128| profiles.iter().map(f).sum::<u128>() / len;
    let sum_usize = |f: fn(&ProverProfile) -> usize| profiles.iter().map(f).sum::<usize>() / profiles.len();
    ProverProfile {
        domain_size: profiles[0].domain_size,
        base_field_bytes: profiles[0].base_field_bytes,
        ext_field_bytes: profiles[0].ext_field_bytes,
        witness_input_bytes: profiles[0].witness_input_bytes,
        bookkeeping_bytes: profiles[0].bookkeeping_bytes,
        eq_r_bytes: profiles[0].eq_r_bytes,
        witness_flatten_bytes: profiles[0].witness_flatten_bytes,
        identical_bytes: profiles[0].identical_bytes,
        permutation_bytes: profiles[0].permutation_bytes,
        product_eval_bytes: profiles[0].product_eval_bytes,
        final_sumcheck_input_bytes: profiles[0].final_sumcheck_input_bytes,
        witness_pc_new_us: sum_u128(|p| p.witness_pc_new_us),
        witness_commit_us: sum_u128(|p| p.witness_commit_us),
        commit_serialize_us: sum_u128(|p| p.commit_serialize_us),
        bookkeeping_build_us: sum_u128(|p| p.bookkeeping_build_us),
        eq_r_build_us: sum_u128(|p| p.eq_r_build_us),
        zerocheck_us: sum_u128(|p| p.zerocheck_us),
        witness_flatten_build_us: sum_u128(|p| p.witness_flatten_build_us),
        identical_build_us: sum_u128(|p| p.identical_build_us),
        permutation_flatten_build_us: sum_u128(|p| p.permutation_flatten_build_us),
        product_eval_build_us: sum_u128(|p| p.product_eval_build_us),
        productcheck_us: sum_u128(|p| p.productcheck_us),
        product_point_eval_append_us: sum_u128(|p| p.product_point_eval_append_us),
        final_sumcheck_input_build_us: sum_u128(|p| p.final_sumcheck_input_build_us),
        final_sumcheck_us: sum_u128(|p| p.final_sumcheck_us),
        final_point_eval_append_us: sum_u128(|p| p.final_point_eval_append_us),
        pcs_open_us: sum_u128(|p| p.pcs_open_us),
        total_us: sum_u128(|p| p.total_us),
        proof_bytes: sum_usize(|p| p.proof_bytes),
    }
}

fn format_mib(bytes: usize) -> String {
    format!("{:.2}", bytes as f64 / (1024.0 * 1024.0))
}

fn format_ms(us: u128) -> String {
    format!("{:.3}", us as f64 / 1000.0)
}

fn stage_share(profile: &ProverProfile, stage_us: u128) -> String {
    if profile.total_us == 0 {
        return "0.00".to_string();
    }
    format!("{:.2}", stage_us as f64 * 100.0 / profile.total_us as f64)
}

fn top_stages(profile: &ProverProfile) -> Vec<(&'static str, u128)> {
    let mut rows = profile.stage_rows().to_vec();
    rows.sort_by_key(|(_, us)| Reverse(*us));
    rows.into_iter().take(3).collect()
}

/// One row (per nv) of a stage/share table pair. `total_us` is the
/// denominator used for the share computation; `items` holds raw us values
/// in the canonical order (parallel to the `names` vector used to build
/// the layout).
#[derive(Clone, Debug)]
struct TableRow {
    nv: usize,
    total_us: u128,
    items: Vec<u128>,
}

/// Sort + collapse layout shared by a (stage ms, share %) table pair.
/// `visible_idx` lists the canonical item indices in descending max-nv share
/// order; `collapsed_idx` lists items whose share at the largest nv falls
/// below the 1% threshold. Both printers iterate `visible_idx` so the two
/// tables always agree on column order.
#[derive(Clone, Debug)]
struct TableLayout {
    visible_idx: Vec<usize>,
    collapsed_idx: Vec<usize>,
    names: Vec<&'static str>,
}

impl TableLayout {
    fn has_other(&self) -> bool {
        !self.collapsed_idx.is_empty()
    }

    fn collapsed_sum(&self, row: &TableRow) -> u128 {
        self.collapsed_idx.iter().map(|&i| row.items[i]).sum()
    }
}

fn build_layout(
    names: Vec<&'static str>,
    rows: &[TableRow],
    threshold_pct: f64,
) -> TableLayout {
    assert!(!rows.is_empty());
    let ref_row = rows.iter().max_by_key(|r| r.nv).unwrap();
    let denom = ref_row.total_us.max(1) as f64;
    let mut ordered: Vec<(usize, f64)> = (0..names.len())
        .map(|i| (i, ref_row.items[i] as f64 / denom * 100.0))
        .collect();
    ordered.sort_by(|a, b| {
        b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut visible_idx = Vec::new();
    let mut collapsed_idx = Vec::new();
    for (i, pct) in ordered {
        if pct >= threshold_pct {
            visible_idx.push(i);
        } else {
            collapsed_idx.push(i);
        }
    }
    TableLayout {
        visible_idx,
        collapsed_idx,
        names,
    }
}

fn append_ms_table(out: &mut String, layout: &TableLayout, rows: &[TableRow]) {
    let mut header = String::from("| nv | total |");
    let mut sep = String::from("| --- | --- |");
    for &i in &layout.visible_idx {
        header.push_str(&format!(" {} |", layout.names[i]));
        sep.push_str(" --- |");
    }
    if layout.has_other() {
        header.push_str(" other |");
        sep.push_str(" --- |");
    }
    out.push_str(&header);
    out.push('\n');
    out.push_str(&sep);
    out.push('\n');
    for r in rows {
        let mut line = format!("| {} | {} |", r.nv, format_ms(r.total_us));
        for &i in &layout.visible_idx {
            line.push_str(&format!(" {} |", format_ms(r.items[i])));
        }
        if layout.has_other() {
            line.push_str(&format!(" {} |", format_ms(layout.collapsed_sum(r))));
        }
        out.push_str(&line);
        out.push('\n');
    }
}

fn append_share_table(out: &mut String, layout: &TableLayout, rows: &[TableRow]) {
    let mut header = String::from("| nv |");
    let mut sep = String::from("| --- |");
    for &i in &layout.visible_idx {
        header.push_str(&format!(" {} |", layout.names[i]));
        sep.push_str(" --- |");
    }
    if layout.has_other() {
        header.push_str(" other |");
        sep.push_str(" --- |");
    }
    out.push_str(&header);
    out.push('\n');
    out.push_str(&sep);
    out.push('\n');
    for r in rows {
        let t = r.total_us.max(1) as f64;
        let mut line = format!("| {} |", r.nv);
        for &i in &layout.visible_idx {
            let pct = r.items[i] as f64 / t * 100.0;
            line.push_str(&format!(" {:.2} |", pct));
        }
        if layout.has_other() {
            let pct = layout.collapsed_sum(r) as f64 / t * 100.0;
            line.push_str(&format!(" {:.2} |", pct));
        }
        out.push_str(&line);
        out.push('\n');
    }
}

fn append_layout_legend(out: &mut String, layout: &TableLayout) {
    if layout.has_other() {
        let names: Vec<&'static str> = layout
            .collapsed_idx
            .iter()
            .map(|&i| layout.names[i])
            .collect();
        out.push_str(&format!("other includes: {}\n\n", names.join(", ")));
    } else {
        out.push('\n');
    }
}

fn build_stage_model(results: &[(usize, usize, ProverProfile)]) -> (Vec<&'static str>, Vec<TableRow>) {
    let rows: Vec<TableRow> = results
        .iter()
        .map(|(nv, _reps, profile)| {
            let stage_rows = profile.stage_rows();
            TableRow {
                nv: *nv,
                total_us: profile.total_us,
                items: stage_rows.iter().map(|(_, us)| *us).collect(),
            }
        })
        .collect();
    let names: Vec<&'static str> = results[0]
        .2
        .stage_rows()
        .iter()
        .map(|(name, _)| *name)
        .collect();
    (names, rows)
}

fn build_report(results: &[(usize, usize, ProverProfile)]) -> String {
    let mut out = String::new();
    out.push_str("# HyperPlonk DeepFold Goldilocks Profile\n\n");
    out.push_str("## Method\n\n");
    out.push_str("- Scope: baseline Goldilocks prover only. Circuit setup and verifier are excluded from stage timings.\n\n");
    out.push_str("- Command: `RUSTFLAGS=\"-C target-cpu=native\" cargo run -p hyperplonk --release --bin profile_hp_df_goldilocks`\n\n");
    out.push_str("- PCS: DeepFold Goldilocks with `query_num = 45`.\n\n");
    out.push_str("- Repetitions: NV 18..20 -> 3, NV 21..22 -> 2, NV 23..24 -> 1.\n\n");
    out.push_str("## Summary\n\n");
    out.push_str("| NV | Repetitions | Total prove ms | Proof bytes | Top stage 1 | Share % | Top stage 2 | Share % | Top stage 3 | Share % |\n");
    out.push_str("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n");
    for (nv, reps, profile) in results {
        let top = top_stages(profile);
        out.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |\n",
            nv,
            reps,
            format_ms(profile.total_us),
            profile.proof_bytes,
            top[0].0,
            stage_share(profile, top[0].1),
            top[1].0,
            stage_share(profile, top[1].1),
            top[2].0,
            stage_share(profile, top[2].1),
        ));
    }
    out.push_str("\n## Main Findings\n\n");
    out.push_str("- `witness_pc_new` and `pcs_open` are the dominant stages across all tested NV. They encapsulate DeepFold FFT/interpolation, Merkle commitments, folding, and query openings, so the PCS layer is the first optimization target in the full baseline system.\n\n");
    out.push_str("- The non-PCS bottlenecks are the large dense vector constructions: `product_eval_build`, `final_sumcheck_input_build`, and `witness_flatten_build`. These are pure allocation/copy/combine work and scale linearly with domain size.\n\n");
    out.push_str("- `zerocheck` and `productcheck` are visible but usually smaller than PCS work plus the dense preprocessing around them, which means improving only the algebraic kernels will not fully fix prover throughput unless the data movement is also reduced.\n\n");

    out.push_str("## Stage Breakdown\n\n");
    let (names, rows) = build_stage_model(results);
    let layout = build_layout(names, &rows, 1.0);
    out.push_str("### Stage (ms)\n\n");
    append_ms_table(&mut out, &layout, &rows);
    out.push('\n');
    out.push_str("### Stage share (%)\n\n");
    append_share_table(&mut out, &layout, &rows);
    out.push('\n');
    append_layout_legend(&mut out, &layout);

    for (nv, reps, profile) in results {
        out.push_str(&format!("## NV {}\n\n", nv));
        out.push_str(&format!("Averaged over {} prover runs.\n\n", reps));
        out.push_str("### Construction and Copy Footprint\n\n");
        out.push_str("| Item | Approx bytes | Approx MiB |\n");
        out.push_str("| --- | --- | --- |\n");
        for (item, bytes) in profile.construction_rows() {
            out.push_str(&format!(
                "| {} | {} | {} |\n",
                item,
                bytes,
                format_mib(bytes),
            ));
        }
        out.push_str("\n");
    }
    out
}

fn main() {
    let nv_min: usize = std::env::var("BENCH_NV_MIN")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(18);
    let nv_max: usize = std::env::var("BENCH_NV_MAX")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(24);
    let output_path = std::env::var("PROFILE_OUTPUT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("hp_df_goldilocks_profile.md"));

    let mut results = Vec::new();
    for nv in nv_min..=nv_max {
        let case = build_case(nv);
        let reps = repetitions_for_nv(nv);
        let mut profiles = Vec::with_capacity(reps);
        let mut last_proof = None;
        for _ in 0..reps {
            let (proof, profile) = prove_with_profile(&case.prover, &case.pp, nv, case.witness.clone());
            last_proof = Some(proof);
            profiles.push(profile);
        }
        let proof = last_proof.expect("proof must exist");
        assert!(case.verifier.verify(&case.pp, nv, proof));
        results.push((nv, reps, average_profiles(&profiles)));
    }

    let report = build_report(&results);
    fs::write(&output_path, report).expect("write profile report");
    println!("wrote {}", output_path.display());
}