//! zerocheck_goldilocks: custom warmup+median harness (replaces criterion).
//!
//! Env vars:
//!   BENCH_NV_MIN / BENCH_NV_MAX  inclusive NV range (default 18..=18)
//!   BENCH_WARMUP=N   warmup iterations (default 2)
//!   BENCH_SAMPLES=N  measured iterations (default 5, must be >= 1)
//!   BENCH_OUTPUT_FILE=PATH  if set, append one line per cell to PATH

use std::fs::OpenOptions;
use std::io::Write as _;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use arithmetic::field::goldilocks64::{Goldilocks64, Goldilocks64Ext};
use arithmetic::field::Field;
use arithmetic::poly::MultiLinearPoly;
use hyperplonk::sumcheck::Sumcheck;
use rand::rngs::SmallRng;
use rand::SeedableRng;
use util::fiat_shamir::Transcript;

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

fn random_evals<S: Field>(rng: &mut SmallRng, n: usize) -> Vec<S> {
    (0..n).map(|_| S::random(&mut *rng)).collect()
}

fn bench_zerocheck(nv: usize) {
    let mut rng = SmallRng::seed_from_u64(42);
    let domain_size = 1 << nv;

    let selector_base: Vec<Goldilocks64> = random_evals(&mut rng, domain_size);
    let bookkeeping: [Vec<Goldilocks64Ext>; 3] =
        std::array::from_fn(|_| random_evals(&mut rng, domain_size));
    let r: Vec<Goldilocks64Ext> = (0..nv)
        .map(|_| Goldilocks64Ext::random(&mut rng))
        .collect();

    measure(&format!("Zerocheck Goldilocks64Ext NV={nv}"), || {
        let selector_base = selector_base.clone();
        let bookkeeping = bookkeeping.clone();
        let r = r.clone();
        let mut transcript = Transcript::new();
        let start = Instant::now();
        let eq_r = MultiLinearPoly::new_eq(&r);
        Sumcheck::prove(
            [
                selector_base
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
        );
        start.elapsed()
    });
}

fn main() {
    let nv_min = env_usize("BENCH_NV_MIN", 18);
    let nv_max = env_usize("BENCH_NV_MAX", 18);
    let warmup = env_usize("BENCH_WARMUP", 1);
    let samples = env_usize("BENCH_SAMPLES", 5);
    assert!(samples >= 1, "BENCH_SAMPLES must be >= 1 (got {samples})");
    MEASURE_CFG
        .set((warmup, samples))
        .expect("MEASURE_CFG set more than once");
    init_output_file();

    for nv in nv_min..=nv_max {
        bench_zerocheck(nv);
    }
}
