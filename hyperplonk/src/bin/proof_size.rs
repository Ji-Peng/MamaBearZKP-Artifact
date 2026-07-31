// x86_64-only profiling/size utility (AVX-512 MamaBear types). On other
// architectures this is a stub `fn main()` so `cargo bench` / `cargo test` --
// which compile every bin in the package -- still succeed; the real body
// below is x86_64-gated and unchanged.
#[cfg(not(target_arch = "x86_64"))]
fn main() {}
#[cfg(target_arch = "x86_64")]
fn main() {
    x86_impl::main();
}
#[cfg(target_arch = "x86_64")]
mod x86_impl {
//! Proof size measurement binary.
//!
//! Records `proof.bytes.len()` for the `prove` / `prove_par` phase of every
//! `hp_df_*` variant and appends one KB-formatted line per
//! measurement to the txt path in `BENCH_OUTPUT_FILE` (if set).
//!
//! Runs in its own process per NV so results from one variant/NV cannot
//! interfere with another. Unlike `peak_memory`, no custom allocator is
//! installed — proof size does not depend on allocation behavior.
//!
//! Usage:
//!   cargo run --release -p hyperplonk --bin proof_size -- <variant>
//!
//! Env vars honored:
//!   BENCH_NV_MIN / BENCH_NV_MAX  (same semantics as the time benches)
//!   BENCH_SPLIT                  (MamaBear variants)
//!   BENCH_SECURITY=conj96|prov97|prov128   (DEFAULT prov128)
//!                                For Goldilocks/BabyBear: selects the query
//!                                  count only (conj96 = 36/32, prov128 = 101).
//!                                MamaBear only supports prov128 (Ext3, 88
//!                                  queries + 16 grinding bits) -- conj96/
//!                                  prov97 used to route to MamaBear Ext2,
//!                                  which has been removed (insufficient
//!                                  soundness); requesting them for a
//!                                  MamaBear variant now panics.
//!                                prov128 is the default so that an unset
//!                                  variable yields the paper regime; conj96
//!                                  understates every number.
//!   RAYON_NUM_THREADS            (parallel variants)
//!   BENCH_OUTPUT_FILE            if set, append one line per measurement
//!                                (key + size + KB) to this txt path. Stdout
//!                                always mirrors the same line.

use std::fs::OpenOptions;
use std::io::Write as _;

use arithmetic::field::{
    babybear::{BabyBearExt4, BabyBearField},
    goldilocks64::{Goldilocks64, Goldilocks64Ext},
    mamabear::{MamaBearScalar, MamaBearScalarExt3, P},
    Field,
};
use arithmetic::mul_group::Radix2Group;
use hyperplonk::{
    aes_circuit::{
        build_aes128_circuit, build_aes128_circuit_babybear, build_aes128_circuit_goldilocks,
    },
    circuit::Circuit,
    prover::Prover,
    prover_mamabear::{setup_mamabear, AlignedPoly, MamaBearExtConfig, ProverMamaBear},
    sha256_circuit::{
        build_sha256_circuit, build_sha256_circuit_babybear, build_sha256_circuit_goldilocks,
    },
};
use poly_commit::deepfold::{DeepFoldParam, DeepFoldProver, DeepFoldVerifier};
use poly_commit::deepfold_mamabear::{DeepFoldMamaBearParam, DEFAULT_SPLIT_LEVEL};
use rand::rngs::SmallRng;
use rand::SeedableRng;
use util::params::{
    babybear, goldilocks,
    gates::{AES128_GATES_PER_CALL, SHA256_GATES_PER_BLOCK},
    mamabear::{self, GRINDING_BITS_EXT3_PROV_QUERY128},
    CODE_RATE_LOG, QUERY_NUM_PROV_QUERY97,
};

#[derive(Clone, Copy, PartialEq, Eq)]
enum SecurityLevel {
    Conj96,
    Prov97,
    Prov128,
}

impl SecurityLevel {
    fn label(self) -> &'static str {
        match self {
            SecurityLevel::Conj96 => "conj96",
            SecurityLevel::Prov97 => "prov97",
            SecurityLevel::Prov128 => "prov128",
        }
    }
    fn query_num_goldilocks(self) -> usize {
        match self {
            SecurityLevel::Conj96 => goldilocks::QUERY_NUM_CONJ96,
            SecurityLevel::Prov97 => QUERY_NUM_PROV_QUERY97,
            SecurityLevel::Prov128 => goldilocks::QUERY_NUM_PROV_QUERY128,
        }
    }
    fn query_num_babybear(self) -> usize {
        match self {
            SecurityLevel::Conj96 => babybear::QUERY_NUM_CONJ96,
            SecurityLevel::Prov97 => QUERY_NUM_PROV_QUERY97,
            SecurityLevel::Prov128 => babybear::QUERY_NUM_PROV_QUERY128,
        }
    }
    fn query_num_mamabear(self) -> usize {
        match self {
            SecurityLevel::Conj96 => mamabear::QUERY_NUM_CONJ96,
            SecurityLevel::Prov97 => QUERY_NUM_PROV_QUERY97,
            SecurityLevel::Prov128 => mamabear::QUERY_NUM_PROV_QUERY128,
        }
    }
}

fn select_security() -> SecurityLevel {
    match std::env::var("BENCH_SECURITY").ok().as_deref() {
        Some("conj96") => SecurityLevel::Conj96,
        Some("prov97") => SecurityLevel::Prov97,
        Some("prov128") => SecurityLevel::Prov128,
        // prov128 is the DEFAULT (2026-07 refactor). An UNSET variable must yield
        // the paper regime, PROV_QUERY128 (88 queries + 16 grinding at rate 1/8);
        // it used to fall through to conj96 (32 queries, no grinding), which
        // silently understated every number by measuring a weaker instance.
        None => SecurityLevel::Prov128,
        // An unrecognised value is a typo, not a request for a default. Failing
        // loudly here is the whole point: `BENCH_SECURITY=prov` silently mapping
        // to some regime is how a mis-measured run reaches a results file.
        Some(other) => panic!(
            "BENCH_SECURITY={other:?} is not one of conj96|prov97|prov128 \
             (unset means prov128)"
        ),
    }
}

/// Smallest nv such that `1 << nv >= gates` — required because the circuit
/// builder pads to exactly 2^nv gates and panics if the un-padded circuit
/// already exceeds the target.
fn min_nv_for(gates: usize) -> usize {
    assert!(gates > 0);
    (usize::BITS - (gates - 1).leading_zeros()) as usize
}

// ---------------------------------------------------------------------------
// Output: append one KB-formatted line per measurement to BENCH_OUTPUT_FILE
// (txt). Stdout always mirrors the same line.
// ---------------------------------------------------------------------------

struct Recorder {
    out: Option<std::fs::File>,
}

impl Recorder {
    fn open() -> Self {
        let out = std::env::var("BENCH_OUTPUT_FILE").ok().map(|p| {
            OpenOptions::new()
                .create(true)
                .append(true)
                .open(&p)
                .unwrap_or_else(|e| panic!("BENCH_OUTPUT_FILE={p:?} open failed: {e}"))
        });
        Self { out }
    }

    fn put(&mut self, key: String, proof_bytes: u64) {
        let kb = (proof_bytes as f64) / 1024.0;
        let line = format!("{key:<60} {kb:>10.2} KiB");
        println!("{line}");
        if let Some(f) = self.out.as_mut() {
            let _ = writeln!(f, "{line}");
            let _ = f.flush();
        }
    }
}

/// Map the CLI variant string to the short, human-readable label used in the
/// txt output. Falls back to the raw variant name so newly added variants
/// still appear instead of silently mangling.
fn display_name(variant: &str) -> &'static str {
    match variant {
        "hp_df_goldilocks" => "HP DF Goldilocks",
        "hp_df_goldilocks_sha256" => "HP DF Goldilocks SHA256",
        "hp_df_goldilocks_aes128" => "HP DF Goldilocks AES128",
        "hp_df_babybear" => "HP DF Babybear",
        "hp_df_babybear_sha256" => "HP DF Babybear SHA256",
        "hp_df_babybear_aes128" => "HP DF Babybear AES128",
        "hp_df_mamabear" => "HP DF Mamabear",
        "hp_df_mamabear_par" => "HP DF Mamabear par",
        "hp_df_mamabear_sha256" => "HP DF Mamabear SHA256",
        "hp_df_mamabear_sha256_par" => "HP DF Mamabear SHA256 par",
        "hp_df_mamabear_aes128" => "HP DF Mamabear AES128",
        "hp_df_mamabear_aes128_par" => "HP DF Mamabear AES128 par",
        other => Box::leak(other.to_string().into_boxed_str()),
    }
}

fn make_key(
    variant: &str,
    label: Option<&str>,
    split: Option<usize>,
    nv: usize,
    phase: &str,
) -> String {
    let name = display_name(variant);
    let label_part = label.map(|l| format!(" {l}")).unwrap_or_default();
    let split_part = split.map(|s| format!(" split={s}")).unwrap_or_default();
    format!("{name}{label_part}{split_part} nv={nv} {phase}")
}

// ---------------------------------------------------------------------------
// Helpers shared by generic (babybear / goldilocks) variants
// ---------------------------------------------------------------------------

fn build_mult_subgroups_babybear(nv: usize) -> Vec<Radix2Group<BabyBearField>> {
    let log_n = (nv + CODE_RATE_LOG) as u32;
    let mut g = vec![Radix2Group::<BabyBearField>::new(log_n)];
    for i in 1..nv {
        g.push(g[i - 1].exp(2));
    }
    g
}

fn build_mult_subgroups_goldilocks(nv: usize) -> Vec<Radix2Group<Goldilocks64>> {
    let log_n = (nv + CODE_RATE_LOG) as u32;
    let mut g = vec![Radix2Group::<Goldilocks64>::new(log_n)];
    for i in 1..nv {
        g.push(g[i - 1].exp(2));
    }
    g
}

// ---------------------------------------------------------------------------
// Per-variant run functions — one per `hp_df_*` bench file.
// Each runs witness_setup / pp_setup / prove and records only the proof size.
// ---------------------------------------------------------------------------

fn run_goldilocks(rec: &mut Recorder, variant: &str, nv: usize, sec: SecurityLevel) {
    let sec_label = sec.label();
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
    let a: Vec<_> = (0..num_gates)
        .map(|_| Goldilocks64::random(&mut rng))
        .collect();
    let b: Vec<_> = (0..num_gates)
        .map(|_| Goldilocks64::random(&mut rng))
        .collect();
    let c: Vec<_> = (0..num_gates)
        .map(|i| {
            let i = i as usize;
            let s = circuit.selector[i];
            -((Goldilocks64::one() - s) * (a[i] + b[i]) + s * a[i] * b[i])
        })
        .collect();
    let witness = [a, b, c];

    let pp = DeepFoldParam::<Goldilocks64Ext> {
        mult_subgroups: build_mult_subgroups_goldilocks(nv),
        variable_num: nv,
        query_num: sec.query_num_goldilocks(),
    };
    let (pk, _vk) = circuit.setup::<DeepFoldProver<_>, DeepFoldVerifier<_>>(&pp, &pp);
    let prover = Prover::<Goldilocks64Ext, DeepFoldProver<Goldilocks64Ext>> { prover_key: pk };
    drop(circuit);

    let proof = prover.prove(&pp, nv, witness);
    rec.put(
        make_key(variant, Some(sec_label), None, nv, "prove"),
        proof.bytes.len() as u64,
    );
}

fn run_goldilocks_sha256(rec: &mut Recorder, variant: &str, nv: usize, sec: SecurityLevel) {
    let sec_label = sec.label();
    let total_gates = 1usize << nv;
    let num_blocks = (total_gates / SHA256_GATES_PER_BLOCK).max(1);
    let (circuit, witness): (Circuit<Goldilocks64Ext>, _) =
        build_sha256_circuit_goldilocks(num_blocks, nv);

    let pp = DeepFoldParam::<Goldilocks64Ext> {
        mult_subgroups: build_mult_subgroups_goldilocks(nv),
        variable_num: nv,
        query_num: sec.query_num_goldilocks(),
    };
    let (pk, _vk) = circuit.setup::<DeepFoldProver<_>, DeepFoldVerifier<_>>(&pp, &pp);
    let prover = Prover::<Goldilocks64Ext, DeepFoldProver<Goldilocks64Ext>> { prover_key: pk };
    drop(circuit);

    let proof = prover.prove(&pp, nv, witness);
    rec.put(
        make_key(variant, Some(sec_label), None, nv, "prove"),
        proof.bytes.len() as u64,
    );
}

fn run_goldilocks_aes128(rec: &mut Recorder, variant: &str, nv: usize, sec: SecurityLevel) {
    let sec_label = sec.label();
    let total_gates = 1usize << nv;
    let num_calls = (total_gates / AES128_GATES_PER_CALL).max(1);
    let (circuit, witness): (Circuit<Goldilocks64Ext>, _) =
        build_aes128_circuit_goldilocks(num_calls, nv);

    let pp = DeepFoldParam::<Goldilocks64Ext> {
        mult_subgroups: build_mult_subgroups_goldilocks(nv),
        variable_num: nv,
        query_num: sec.query_num_goldilocks(),
    };
    let (pk, _vk) = circuit.setup::<DeepFoldProver<_>, DeepFoldVerifier<_>>(&pp, &pp);
    let prover = Prover::<Goldilocks64Ext, DeepFoldProver<Goldilocks64Ext>> { prover_key: pk };
    drop(circuit);

    let proof = prover.prove(&pp, nv, witness);
    rec.put(
        make_key(variant, Some(sec_label), None, nv, "prove"),
        proof.bytes.len() as u64,
    );
}

fn run_babybear(rec: &mut Recorder, variant: &str, nv: usize, sec: SecurityLevel) {
    let sec_label = sec.label();
    let mut rng = SmallRng::seed_from_u64(1);
    let num_gates = 1u32 << nv;
    let circuit = Circuit::<BabyBearExt4> {
        permutation: [
            (0..num_gates).map(|x| x.into()).collect(),
            (0..num_gates).map(|x| (x + (1 << 29)).into()).collect(),
            (0..num_gates).map(|x| (x + (1 << 30)).into()).collect(),
        ],
        selector: (0..num_gates).map(|x| (x & 1).into()).collect(),
    };
    let a: Vec<_> = (0..num_gates)
        .map(|_| BabyBearField::random(&mut rng))
        .collect();
    let b: Vec<_> = (0..num_gates)
        .map(|_| BabyBearField::random(&mut rng))
        .collect();
    let c: Vec<_> = (0..num_gates)
        .map(|i| {
            let i = i as usize;
            let s = circuit.selector[i];
            -((BabyBearField::one() - s) * (a[i] + b[i]) + s * a[i] * b[i])
        })
        .collect();
    let witness = [a, b, c];

    let pp = DeepFoldParam::<BabyBearExt4> {
        mult_subgroups: build_mult_subgroups_babybear(nv),
        variable_num: nv,
        query_num: sec.query_num_babybear(),
    };
    let (pk, _vk) = circuit.setup::<DeepFoldProver<_>, DeepFoldVerifier<_>>(&pp, &pp);
    let prover = Prover::<BabyBearExt4, DeepFoldProver<BabyBearExt4>> { prover_key: pk };
    drop(circuit);

    let proof = prover.prove(&pp, nv, witness);
    rec.put(
        make_key(variant, Some(sec_label), None, nv, "prove"),
        proof.bytes.len() as u64,
    );
}

fn run_babybear_sha256(rec: &mut Recorder, variant: &str, nv: usize, sec: SecurityLevel) {
    let sec_label = sec.label();
    let total_gates = 1usize << nv;
    let num_blocks = (total_gates / SHA256_GATES_PER_BLOCK).max(1);
    let (circuit, witness): (Circuit<BabyBearExt4>, _) =
        build_sha256_circuit_babybear(num_blocks, nv);

    let pp = DeepFoldParam::<BabyBearExt4> {
        mult_subgroups: build_mult_subgroups_babybear(nv),
        variable_num: nv,
        query_num: sec.query_num_babybear(),
    };
    let (pk, _vk) = circuit.setup::<DeepFoldProver<_>, DeepFoldVerifier<_>>(&pp, &pp);
    let prover = Prover::<BabyBearExt4, DeepFoldProver<BabyBearExt4>> { prover_key: pk };
    drop(circuit);

    let proof = prover.prove(&pp, nv, witness);
    rec.put(
        make_key(variant, Some(sec_label), None, nv, "prove"),
        proof.bytes.len() as u64,
    );
}

fn run_babybear_aes128(rec: &mut Recorder, variant: &str, nv: usize, sec: SecurityLevel) {
    let sec_label = sec.label();
    let total_gates = 1usize << nv;
    let num_calls = (total_gates / AES128_GATES_PER_CALL).max(1);
    let (circuit, witness): (Circuit<BabyBearExt4>, _) =
        build_aes128_circuit_babybear(num_calls, nv);

    let pp = DeepFoldParam::<BabyBearExt4> {
        mult_subgroups: build_mult_subgroups_babybear(nv),
        variable_num: nv,
        query_num: sec.query_num_babybear(),
    };
    let (pk, _vk) = circuit.setup::<DeepFoldProver<_>, DeepFoldVerifier<_>>(&pp, &pp);
    let prover = Prover::<BabyBearExt4, DeepFoldProver<BabyBearExt4>> { prover_key: pk };
    drop(circuit);

    let proof = prover.prove(&pp, nv, witness);
    rec.put(
        make_key(variant, Some(sec_label), None, nv, "prove"),
        proof.bytes.len() as u64,
    );
}

// ---------- MamaBear variants ----------

fn mamabear_witness_synthetic<F: MamaBearExtConfig>(
    nv: usize,
) -> (Circuit<F>, [AlignedPoly; 3]) {
    let mut rng = SmallRng::seed_from_u64(1);
    let num_gates = 1usize << nv;
    let circuit = Circuit::<F> {
        permutation: [
            (0..num_gates)
                .map(|x| MamaBearScalar::from(x as u64))
                .collect(),
            (0..num_gates)
                .map(|x| MamaBearScalar::from((x + (1 << 29)) as u64))
                .collect(),
            (0..num_gates)
                .map(|x| MamaBearScalar::from((x + (1 << 30)) as u64))
                .collect(),
        ],
        selector: (0..num_gates)
            .map(|x| MamaBearScalar::from((x & 1) as u64))
            .collect(),
    };
    let a: Vec<_> = (0..num_gates)
        .map(|_| MamaBearScalar::random(&mut rng))
        .collect();
    let b: Vec<_> = (0..num_gates)
        .map(|_| MamaBearScalar::random(&mut rng))
        .collect();
    // `MamaBearScalar::Mul` is `mont_mul` (x·y/R), not a raw product — use
    // u128 modular arithmetic to build a witness that actually satisfies
    // (1-s)(a+b) + s·a·b + c ≡ 0 (mod P). See hyperplonk/src/lib.rs test.
    let raw_mul = |x: u64, y: u64| ((x as u128 * y as u128) % (P as u128)) as u64;
    let c: Vec<_> = (0..num_gates)
        .map(|i| {
            let s = circuit.selector[i].0;
            let a_i = a[i].0;
            let b_i = b[i].0;
            let a_plus_b = (a_i + b_i) % P;
            let one_minus_s = (P + 1 - s) % P;
            let term1 = raw_mul(one_minus_s, a_plus_b);
            let s_a_b = raw_mul(raw_mul(s, a_i), b_i);
            let expr = (term1 + s_a_b) % P;
            let neg = if expr == 0 { 0 } else { P - expr };
            MamaBearScalar(neg)
        })
        .collect();
    let witness = [
        AlignedPoly::from_sbf(&a),
        AlignedPoly::from_sbf(&b),
        AlignedPoly::from_sbf(&c),
    ];
    (circuit, witness)
}

fn run_mamabear_variant<F: MamaBearExtConfig>(
    rec: &mut Recorder,
    variant: &str,
    label: &str,
    nv: usize,
    split: usize,
    parallel: bool,
    query_num: usize,
    grinding_bits: u32,
    witness_builder: impl FnOnce() -> (Circuit<F>, [AlignedPoly; 3]),
) where
    F::Packed: Send + Sync,
    F: Send + Sync,
{
    let (circuit, witness) = witness_builder();

    let mut pp = DeepFoldMamaBearParam::new(nv, CODE_RATE_LOG, query_num, split);
    pp.grinding_bits = grinding_bits;
    let (pk, _vk) = setup_mamabear::<F>(&circuit, &pp);
    let prover = ProverMamaBear::<F> { prover_key: pk };
    drop(circuit);

    let phase = if parallel { "prove_par" } else { "prove" };
    let proof = if parallel {
        prover.prove_par(&pp, nv, witness)
    } else {
        prover.prove(&pp, nv, witness)
    };
    rec.put(
        make_key(variant, Some(label), Some(split), nv, phase),
        proof.bytes.len() as u64,
    );
}

fn run_mamabear_synthetic(
    rec: &mut Recorder,
    variant: &str,
    nv: usize,
    split: usize,
    parallel: bool,
    sec: SecurityLevel,
) {
    let sec_label = sec.label();
    let query_num = sec.query_num_mamabear();
    match sec {
        SecurityLevel::Conj96 | SecurityLevel::Prov97 => panic!(
            "BENCH_SECURITY={sec_label} is not supported for MamaBear: conj96/prov97 used \
             to route to MamaBear Ext2, which has been removed (insufficient soundness). \
             MamaBear only supports prov128 (Ext3)."
        ),
        SecurityLevel::Prov128 => run_mamabear_variant::<MamaBearScalarExt3>(
            rec,
            variant,
            sec_label,
            nv,
            split,
            parallel,
            query_num,
            GRINDING_BITS_EXT3_PROV_QUERY128,
            || mamabear_witness_synthetic::<MamaBearScalarExt3>(nv),
        ),
    }
}

fn run_mamabear_sha256_any(
    rec: &mut Recorder,
    variant: &str,
    nv: usize,
    split: usize,
    parallel: bool,
    sec: SecurityLevel,
) {
    let sec_label = sec.label();
    let query_num = sec.query_num_mamabear();
    let total_gates = 1usize << nv;
    let num_blocks = (total_gates / SHA256_GATES_PER_BLOCK).max(1);
    match sec {
        SecurityLevel::Conj96 | SecurityLevel::Prov97 => panic!(
            "BENCH_SECURITY={sec_label} is not supported for MamaBear: conj96/prov97 used \
             to route to MamaBear Ext2, which has been removed (insufficient soundness). \
             MamaBear only supports prov128 (Ext3)."
        ),
        SecurityLevel::Prov128 => run_mamabear_variant::<MamaBearScalarExt3>(
            rec,
            variant,
            sec_label,
            nv,
            split,
            parallel,
            query_num,
            GRINDING_BITS_EXT3_PROV_QUERY128,
            || {
                let (circuit, raw) = build_sha256_circuit::<MamaBearScalarExt3>(num_blocks, nv);
                let witness = [
                    AlignedPoly::from_sbf(&raw[0]),
                    AlignedPoly::from_sbf(&raw[1]),
                    AlignedPoly::from_sbf(&raw[2]),
                ];
                (circuit, witness)
            },
        ),
    }
}

fn run_mamabear_aes128_any(
    rec: &mut Recorder,
    variant: &str,
    nv: usize,
    split: usize,
    parallel: bool,
    sec: SecurityLevel,
) {
    let sec_label = sec.label();
    let query_num = sec.query_num_mamabear();
    let total_gates = 1usize << nv;
    let num_calls = (total_gates / AES128_GATES_PER_CALL).max(1);
    match sec {
        SecurityLevel::Conj96 | SecurityLevel::Prov97 => panic!(
            "BENCH_SECURITY={sec_label} is not supported for MamaBear: conj96/prov97 used \
             to route to MamaBear Ext2, which has been removed (insufficient soundness). \
             MamaBear only supports prov128 (Ext3)."
        ),
        SecurityLevel::Prov128 => run_mamabear_variant::<MamaBearScalarExt3>(
            rec,
            variant,
            sec_label,
            nv,
            split,
            parallel,
            query_num,
            GRINDING_BITS_EXT3_PROV_QUERY128,
            || {
                let (circuit, raw) = build_aes128_circuit::<MamaBearScalarExt3>(num_calls, nv);
                let witness = [
                    AlignedPoly::from_sbf(&raw[0]),
                    AlignedPoly::from_sbf(&raw[1]),
                    AlignedPoly::from_sbf(&raw[2]),
                ];
                (circuit, witness)
            },
        ),
    }
}

// ---------------------------------------------------------------------------
// main: CLI dispatch
// ---------------------------------------------------------------------------

pub fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: proof_size <variant>");
        eprintln!("variants: hp_df_{{goldilocks,babybear,mamabear,mamabear_par,");
        eprintln!(
            "          mamabear_sha256,mamabear_sha256_par,mamabear_aes128,mamabear_aes128_par,"
        );
        eprintln!(
            "          goldilocks_sha256,goldilocks_aes128,babybear_sha256,babybear_aes128}}"
        );
        std::process::exit(2);
    }
    let variant = args[1].as_str();

    let nv_min: usize = std::env::var("BENCH_NV_MIN")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(18);
    let nv_max: usize = std::env::var("BENCH_NV_MAX")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(18);
    let split: usize = std::env::var("BENCH_SPLIT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_SPLIT_LEVEL);
    let sec = select_security();

    let mut rec = Recorder::open();

    let sha_min = min_nv_for(SHA256_GATES_PER_BLOCK);
    let aes_min = min_nv_for(AES128_GATES_PER_CALL);
    let skip_min = match variant {
        v if v.ends_with("_sha256") || v.ends_with("_sha256_par") => Some(sha_min),
        v if v.ends_with("_aes128") || v.ends_with("_aes128_par") => Some(aes_min),
        _ => None,
    };

    for nv in nv_min..=nv_max {
        if let Some(min) = skip_min {
            if nv < min {
                eprintln!(
                    "[{variant}] skip NV={nv}: circuit needs >= 2^{min} gates (1 unit)"
                );
                continue;
            }
        }
        println!("[{variant}] NV={nv}");
        match variant {
            "hp_df_goldilocks" => run_goldilocks(&mut rec, variant, nv, sec),
            "hp_df_goldilocks_sha256" => {
                run_goldilocks_sha256(&mut rec, variant, nv, sec)
            }
            "hp_df_goldilocks_aes128" => {
                run_goldilocks_aes128(&mut rec, variant, nv, sec)
            }
            "hp_df_babybear" => run_babybear(&mut rec, variant, nv, sec),
            "hp_df_babybear_sha256" => {
                run_babybear_sha256(&mut rec, variant, nv, sec)
            }
            "hp_df_babybear_aes128" => {
                run_babybear_aes128(&mut rec, variant, nv, sec)
            }
            "hp_df_mamabear" => {
                run_mamabear_synthetic(&mut rec, variant, nv, split, false, sec)
            }
            "hp_df_mamabear_par" => {
                run_mamabear_synthetic(&mut rec, variant, nv, split, true, sec)
            }
            "hp_df_mamabear_sha256" => {
                run_mamabear_sha256_any(&mut rec, variant, nv, split, false, sec)
            }
            "hp_df_mamabear_sha256_par" => {
                run_mamabear_sha256_any(&mut rec, variant, nv, split, true, sec)
            }
            "hp_df_mamabear_aes128" => {
                run_mamabear_aes128_any(&mut rec, variant, nv, split, false, sec)
            }
            "hp_df_mamabear_aes128_par" => {
                run_mamabear_aes128_any(&mut rec, variant, nv, split, true, sec)
            }
            other => {
                eprintln!("unknown variant: {other}");
                std::process::exit(2);
            }
        }
    }
}
}
