//! deepfold_babybear: custom warmup+median harness (replaces criterion).
//!
//! Env vars:
//!   BENCH_NV_MIN / BENCH_NV_MAX  inclusive NV range (default 18..=18)
//!   BENCH_SECURITY=conj96|prov97|prov128  FRI security level (DEFAULT prov128)
//!   BENCH_WARMUP=N    warmup iterations (default 1)
//!   BENCH_SAMPLES=N   measured iterations (default 5, must be >= 1)
//!   BENCH_OUTPUT_FILE=PATH  if set, append one line per cell to PATH

use std::fs::OpenOptions;
use std::io::Write as _;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use arithmetic::{
    field::{
        babybear::{BabyBearExt4, BabyBearField},
        Field,
    },
    mul_group::Radix2Group,
};
use poly_commit::{
    deepfold::{DeepFoldParam, DeepFoldProver, MerkleRoot},
    CommitmentSerde, PolyCommitProver,
};
use rand::rngs::SmallRng;
use rand::SeedableRng;
use util::fiat_shamir::Transcript;
use util::params::{
    babybear::{QUERY_NUM_CONJ96, QUERY_NUM_PROV_QUERY128},
    CODE_RATE_LOG, NUM_WITNESS, QUERY_NUM_PROV_QUERY97,
};

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

struct Case {
    nv: usize,
    pp: DeepFoldParam<BabyBearExt4>,
    witness: Vec<Vec<BabyBearField>>,
    prover_key_commitments: DeepFoldProver<BabyBearExt4>,
    witness_pc: DeepFoldProver<BabyBearExt4>,
    point: Vec<BabyBearExt4>,
    commit_buf_size: usize,
}

fn build_case(nv: usize, query_num: usize) -> Case {
    let mut rng = SmallRng::seed_from_u64(42);

    let len = 1usize << nv;
    fn rand_poly(rng: &mut SmallRng, len: usize) -> Vec<BabyBearField> {
        (0..len)
            .map(|_| BabyBearField::random(&mut *rng))
            .collect()
    }

    let selector = rand_poly(&mut rng, len);
    let permutation: Vec<Vec<BabyBearField>> =
        (0..3).map(|_| rand_poly(&mut rng, len)).collect();
    let witness: Vec<Vec<BabyBearField>> =
        (0..NUM_WITNESS).map(|_| rand_poly(&mut rng, len)).collect();
    let point: Vec<BabyBearExt4> = (0..nv).map(|_| BabyBearExt4::random(&mut rng)).collect();

    let mut mult_subgroups = vec![Radix2Group::<BabyBearField>::new(
        (nv + CODE_RATE_LOG) as u32,
    )];
    for i in 1..nv {
        mult_subgroups.push(mult_subgroups[i - 1].exp(2));
    }
    let pp = DeepFoldParam::<BabyBearExt4> {
        mult_subgroups,
        variable_num: nv,
        query_num,
    };

    let fixed: Vec<Vec<BabyBearField>> = std::iter::once(selector.clone())
        .chain(permutation.iter().cloned())
        .collect();
    let prover_key_commitments = DeepFoldProver::<BabyBearExt4>::new(&pp, &fixed);
    let witness_pc = DeepFoldProver::<BabyBearExt4>::new(&pp, &witness);

    Case {
        nv,
        pp,
        witness,
        prover_key_commitments,
        witness_pc,
        point,
        commit_buf_size: MerkleRoot::size(nv, NUM_WITNESS),
    }
}

fn bench_commit(case: &Case, sec_label: &str) {
    measure(
        &format!("BabyBear {sec_label} Commit NV={}", case.nv),
        || {
            let witness = case.witness.clone();
            let start = Instant::now();
            let prover = DeepFoldProver::<BabyBearExt4>::new(&case.pp, &witness);
            let commit = prover.commit();
            let mut buffer = vec![0u8; case.commit_buf_size];
            commit.serialize_into(&mut buffer);
            let mut transcript = Transcript::new();
            transcript.append_u8_slice(&buffer, buffer.len());
            start.elapsed()
        },
    );
}

fn bench_open(case: &Case, sec_label: &str) {
    measure(
        &format!("BabyBear {sec_label} Open NV={}", case.nv),
        || {
            let mut transcript = Transcript::new();
            let start = Instant::now();
            DeepFoldProver::<BabyBearExt4>::open(
                &case.pp,
                vec![&case.prover_key_commitments, &case.witness_pc],
                case.point.clone(),
                &mut transcript,
            );
            start.elapsed()
        },
    );
}

fn main() {
    let nv_min = env_usize("BENCH_NV_MIN", 18);
    let nv_max = env_usize("BENCH_NV_MAX", 18);
    let sec = select_security();

    let warmup = env_usize("BENCH_WARMUP", 1);
    let samples = env_usize("BENCH_SAMPLES", 5);
    assert!(samples >= 1, "BENCH_SAMPLES must be >= 1 (got {samples})");
    MEASURE_CFG
        .set((warmup, samples))
        .expect("MEASURE_CFG set more than once");
    init_output_file();

    for nv in nv_min..=nv_max {
        let case = build_case(nv, sec.query_num());
        bench_commit(&case, sec.label());
        bench_open(&case, sec.label());
    }
}
