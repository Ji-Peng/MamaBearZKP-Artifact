//! hp_df_goldilocks_blake3: custom warmup+median harness (replaces criterion).
//!
//! HyperPlonk + DeepFold on Goldilocks with a BLAKE3 permutation circuit.
//! The permutation unit matches plonky3's `p3-blake3-air::Blake3Air` (7 rounds,
//! no feed-forward). At NV=X this bench proves
//! `floor(2^X / BLAKE3_GATES_PER_PERMUTATION)` permutations, comparable to
//! plonky3's `--log-trace-length = log2(num_perms)` row.
//!
//! Env vars:
//!   BENCH_NV_MIN / BENCH_NV_MAX  inclusive NV range (default 20..=20)
//!   BENCH_SECURITY=conj96|prov97|prov128  FRI security level (DEFAULT prov128)
//!   BENCH_WARMUP=N    warmup iterations (default 1)
//!   BENCH_SAMPLES=N   measured iterations (default 5, must be >= 1)
//!   BENCH_OUTPUT_FILE=PATH  if set, append one line per cell to PATH

use std::fs::OpenOptions;
use std::hint::black_box;
use std::io::Write as _;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use arithmetic::{
    field::goldilocks64::{Goldilocks64, Goldilocks64Ext},
    mul_group::Radix2Group,
};
use hyperplonk::{
    blake3_circuit::build_blake3_circuit_goldilocks, circuit::Circuit, prover::Prover,
    verifier::Verifier,
};
use poly_commit::deepfold::{DeepFoldParam, DeepFoldProver, DeepFoldVerifier};
use util::fiat_shamir::Proof;

type DeepFoldGoldilocksProver = Prover<Goldilocks64Ext, DeepFoldProver<Goldilocks64Ext>>;
type DeepFoldGoldilocksVerifier = Verifier<Goldilocks64Ext, DeepFoldVerifier<Goldilocks64Ext>>;

use util::params::{
    gates::BLAKE3_GATES_PER_PERMUTATION,
    goldilocks::{QUERY_NUM_CONJ96, QUERY_NUM_PROV_QUERY128},
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

struct BenchCase {
    nv: usize,
    num_perms: usize,
    pp: DeepFoldParam<Goldilocks64Ext>,
    prover: DeepFoldGoldilocksProver,
    verifier: DeepFoldGoldilocksVerifier,
    witness: [Vec<Goldilocks64>; 3],
    proof: Proof,
}

fn build_mult_subgroups(nv: usize) -> Vec<Radix2Group<Goldilocks64>> {
    let log_n = (nv + CODE_RATE_LOG) as u32;
    let mut mult_subgroups = vec![Radix2Group::<Goldilocks64>::new(log_n)];
    for i in 1..nv {
        mult_subgroups.push(mult_subgroups[i - 1].exp(2));
    }
    mult_subgroups
}

fn build_case(nv: usize, query_num: usize) -> BenchCase {
    let total_gates = 1usize << nv;
    let num_perms = (total_gates / BLAKE3_GATES_PER_PERMUTATION).max(1);

    let (circuit, witness): (Circuit<Goldilocks64Ext>, _) =
        build_blake3_circuit_goldilocks(num_perms, nv);

    let pp = DeepFoldParam::<Goldilocks64Ext> {
        mult_subgroups: build_mult_subgroups(nv),
        variable_num: nv,
        query_num,
    };
    let (pk, vk) = circuit.setup::<DeepFoldProver<_>, DeepFoldVerifier<_>>(&pp, &pp);
    let prover = Prover { prover_key: pk };
    let verifier = Verifier { verifier_key: vk };

    let proof = prover.prove(&pp, nv, witness.clone());
    assert!(verifier.verify(&pp, nv, proof.clone()));

    BenchCase {
        nv,
        num_perms,
        pp,
        prover,
        verifier,
        witness,
        proof,
    }
}

fn bench_case(case: &BenchCase, sec_label: &str) {
    measure(
        &format!(
            "HP DF Goldilocks BLAKE3 {sec_label} prove NV={} perms={}",
            case.nv, case.num_perms
        ),
        || {
            let witness = case.witness.clone();
            let start = Instant::now();
            black_box(case.prover.prove(&case.pp, case.nv, witness));
            start.elapsed()
        },
    );

    measure(
        &format!(
            "HP DF Goldilocks BLAKE3 {sec_label} verify NV={} perms={}",
            case.nv, case.num_perms
        ),
        || {
            let proof = case.proof.clone();
            let start = Instant::now();
            black_box(case.verifier.verify(&case.pp, case.nv, proof));
            start.elapsed()
        },
    );
}

fn main() {
    let nv_min = env_usize("BENCH_NV_MIN", 20);
    let nv_max = env_usize("BENCH_NV_MAX", 20);
    let sec = select_security();
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
                "[hp_df_goldilocks_blake3] skip NV={nv}: \
                 circuit needs >= 2^{min_nv} gates (1 BLAKE3 permutation)"
            );
            continue;
        }
        let case = build_case(nv, sec.query_num());
        bench_case(&case, sec.label());
    }
}
