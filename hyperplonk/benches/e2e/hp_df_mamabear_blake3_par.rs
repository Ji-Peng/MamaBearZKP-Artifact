//! hp_df_mamabear_blake3_par: custom warmup+median harness (replaces criterion).
//!
//! HyperPlonk + DeepFold (parallel prover) on a BLAKE3 permutation circuit.
//! The permutation unit matches plonky3's `p3-blake3-air::Blake3Air` (7 rounds,
//! no feed-forward). At NV=X this bench proves
//! `floor(2^X / BLAKE3_GATES_PER_PERMUTATION)` permutations, comparable to
//! plonky3's `--log-trace-length = log2(num_perms)` row.
//!
//! Env vars:
//!   BENCH_NV_MIN / BENCH_NV_MAX  inclusive NV range (default 20..=20)
//!   BENCH_SPLIT=N      override split level (default DEFAULT_SPLIT_LEVEL)
//!   BENCH_SECURITY=conj96|prov97|prov128  FRI security level (DEFAULT prov128)
//!   RAYON_NUM_THREADS  controls rayon worker count (caller sets this)
//!   BENCH_WARMUP=N    warmup iterations (default 1)
//!   BENCH_SAMPLES=N   measured iterations (default 5, must be >= 1)
//!   BENCH_OUTPUT_FILE=PATH  if set, append one line per cell to PATH

use std::fs::OpenOptions;
use std::hint::black_box;
use std::io::Write as _;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use arithmetic::field::mamabear::MamaBearScalarExt3;
use hyperplonk::{
    blake3_circuit::build_blake3_circuit,
    prover_mamabear::{setup_mamabear, AlignedPoly, MamaBearExtConfig, ProverMamaBear},
    verifier_mamabear::VerifierMamaBear,
};
use poly_commit::deepfold_mamabear::{DeepFoldMamaBearParam, DEFAULT_SPLIT_LEVEL};
use util::fiat_shamir::Proof;
use util::params::{
    gates::BLAKE3_GATES_PER_PERMUTATION,
    mamabear::{GRINDING_BITS_EXT3_PROV_QUERY128, QUERY_NUM_CONJ96, QUERY_NUM_PROV_QUERY128},
    CODE_RATE_LOG, QUERY_NUM_PROV_QUERY97,
};

fn min_nv_for(gates: usize) -> usize {
    assert!(gates > 0);
    (usize::BITS - (gates - 1).leading_zeros()) as usize
}

// ---------------------------------------------------------------------------
// Shared harness scaffold
// ---------------------------------------------------------------------------

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

static MEASURE_CFG: OnceLock<(usize, usize)> = OnceLock::new();
static OUTPUT_FILE: OnceLock<Option<Mutex<std::fs::File>>> = OnceLock::new();

fn init_output_file() {
    let file = std::env::var("BENCH_OUTPUT_FILE").ok().map(|p| {
        let f = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&p)
            .unwrap_or_else(|e| panic!("BENCH_OUTPUT_FILE={p:?} open failed: {e}"));
        Mutex::new(f)
    });
    OUTPUT_FILE
        .set(file)
        .ok()
        .expect("OUTPUT_FILE already initialized");
}

fn measure_cfg() -> (usize, usize) {
    *MEASURE_CFG.get().expect("MEASURE_CFG not initialized")
}

fn record(line: &str) {
    println!("{line}");
    if let Some(Some(m)) = OUTPUT_FILE.get() {
        let mut f = m.lock().unwrap();
        let _ = writeln!(f, "{line}");
        let _ = f.flush();
    }
}

fn measure<F: FnMut() -> Duration>(label: &str, mut one_sample: F) {
    let (warmup, samples) = measure_cfg();
    for _ in 0..warmup {
        let _ = one_sample();
    }
    let mut times: Vec<Duration> = (0..samples).map(|_| one_sample()).collect();
    times.sort();
    let median_ms = times[samples / 2].as_secs_f64() * 1000.0;
    record(&format!("{label:<60} {median_ms:>10.3} ms"));
}

// ---------------------------------------------------------------------------
// SecurityLevel
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
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
    fn query_num(self) -> usize {
        match self {
            SecurityLevel::Conj96 => QUERY_NUM_CONJ96,
            SecurityLevel::Prov97 => QUERY_NUM_PROV_QUERY97,
            SecurityLevel::Prov128 => QUERY_NUM_PROV_QUERY128,
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

// ---------------------------------------------------------------------------
// Bench case
// ---------------------------------------------------------------------------

struct BenchCase<F: MamaBearExtConfig>
where
    F::Packed: Send + Sync,
    F: Send + Sync,
{
    nv: usize,
    num_perms: usize,
    pp: DeepFoldMamaBearParam,
    prover: ProverMamaBear<F>,
    verifier: VerifierMamaBear<F>,
    verify_ok: bool,
    witness: [AlignedPoly; 3],
    proof: Proof,
}

fn build_case<F: MamaBearExtConfig>(
    nv: usize,
    split: usize,
    sec: SecurityLevel,
    grinding_bits: u32,
) -> BenchCase<F>
where
    F::Packed: Send + Sync,
    F: Send + Sync,
{
    let total_gates = 1usize << nv;
    let num_perms = (total_gates / BLAKE3_GATES_PER_PERMUTATION).max(1);

    let (circuit, witness_raw) = build_blake3_circuit::<F>(num_perms, nv);
    let mut pp = DeepFoldMamaBearParam::new(nv, CODE_RATE_LOG, sec.query_num(), split);
    pp.grinding_bits = grinding_bits;
    let (pk, vk) = setup_mamabear::<F>(&circuit, &pp);
    let prover = ProverMamaBear { prover_key: pk };
    let verifier = VerifierMamaBear { verifier_key: vk };

    let witness = [
        AlignedPoly::from_sbf(&witness_raw[0]),
        AlignedPoly::from_sbf(&witness_raw[1]),
        AlignedPoly::from_sbf(&witness_raw[2]),
    ];
    let proof = prover.prove_par(&pp, nv, witness.clone());
    let verify_ok = verifier.verify_par(&pp, nv, proof.clone());

    BenchCase {
        nv,
        num_perms,
        pp,
        prover,
        verifier,
        verify_ok,
        witness,
        proof,
    }
}

fn bench_case<F: MamaBearExtConfig>(label: &str, sec_label: &str, case: &BenchCase<F>)
where
    F::Packed: Send + Sync,
    F: Send + Sync,
{
    measure(
        &format!(
            "HP DF MamaBear BLAKE3 Par {label} split={} {sec_label} prove_par NV={} perms={}",
            case.pp.split_level, case.nv, case.num_perms
        ),
        || {
            let witness = case.witness.clone();
            let start = Instant::now();
            black_box(case.prover.prove_par(&case.pp, case.nv, witness));
            start.elapsed()
        },
    );

    if case.verify_ok {
        measure(
            &format!(
                "HP DF MamaBear BLAKE3 Par {label} split={} {sec_label} verify_par NV={} perms={}",
                case.pp.split_level, case.nv, case.num_perms
            ),
            || {
                let proof = case.proof.clone();
                let start = Instant::now();
                black_box(case.verifier.verify_par(&case.pp, case.nv, proof));
                start.elapsed()
            },
        );
    }
}

fn main() {
    let nv_min = env_usize("BENCH_NV_MIN", 20);
    let nv_max = env_usize("BENCH_NV_MAX", 20);
    let split = env_usize("BENCH_SPLIT", DEFAULT_SPLIT_LEVEL);
    let sec = select_security();
    let sec_label = sec.label();
    let min_nv = min_nv_for(BLAKE3_GATES_PER_PERMUTATION);

    let warmup = env_usize("BENCH_WARMUP", 1);
    let samples = env_usize("BENCH_SAMPLES", 5);
    assert!(samples >= 1, "BENCH_SAMPLES must be >= 1 (got {samples})");
    MEASURE_CFG
        .set((warmup, samples))
        .expect("MEASURE_CFG set more than once");
    init_output_file();

    for nv in nv_min..=nv_max {
        if nv < min_nv {
            eprintln!(
                "[hp_df_mamabear_blake3_par] skip NV={nv}: \
                 circuit needs >= 2^{min_nv} gates (1 BLAKE3 permutation)"
            );
            continue;
        }
        match sec {
            SecurityLevel::Conj96 | SecurityLevel::Prov97 => panic!(
                "BENCH_SECURITY={sec_label} is not supported for MamaBear: conj96/prov97 used \
                 to route to MamaBear Ext2, which has been removed (insufficient soundness). \
                 MamaBear only supports prov128 (Ext3)."
            ),
            SecurityLevel::Prov128 => {
                let case =
                    build_case::<MamaBearScalarExt3>(nv, split, sec, GRINDING_BITS_EXT3_PROV_QUERY128);
                bench_case("Ext3", sec_label, &case);
            }
        }
    }
}
