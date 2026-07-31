//! zerocheck_mamabear: custom warmup+median harness (replaces criterion).
//!
//! Env vars (criterion-compat):
//!   BENCH_NV_MIN / BENCH_NV_MAX            inclusive NV range (default 18..=18)
//!   BENCH_LEGACY_EXT3_ONLY
//!   BENCH_OPT_EXT3_ONLY
//!   BENCH_OPT_EXT3_PAR_ONLY
//!     If any *_ONLY flag is set, only enabled kinds run; otherwise all run.
//!   ELL0=1|2|3                             restrict optimized benches to one ell0 (default all)
//!
//! Harness knobs:
//!   BENCH_WARMUP=N    warmup iterations  (default 2)
//!   BENCH_SAMPLES=N   measured iters     (default 5, must be >= 1)
//!                     median = sorted[N/2] (upper-middle on even N)
//!   BENCH_OUTPUT_FILE=PATH  if set, append one line per measured cell to PATH
//!                           (in addition to stdout); reproduce.sh truncates
//!                           the file once before each per-NV invocation loop.

use std::fs::OpenOptions;
use std::io::Write as _;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use arithmetic::field::mamabear::{
    MamaBearScalar, MamaBearScalarExt3, PackedMamaBearAVX512Ext3,
};
use arithmetic::field::Field;
use arithmetic::poly::MultiLinearPoly;
use hyperplonk::prover_mamabear::AlignedPoly;
use hyperplonk::sumcheck_mamabear::SumcheckMamaBear;
use hyperplonk::zerocheck_generic_mamabear::{prove_zero_check_generic, AddMulD3};
use rand::rngs::SmallRng;
use rand::SeedableRng;
use util::fiat_shamir::Transcript;

// ---------------------------------------------------------------------------
// Shared harness scaffold, inlined so each benchmark has an independent entry point.
// ---------------------------------------------------------------------------

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn env_flag(key: &str) -> bool {
    std::env::var(key).is_ok()
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
// Filters
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
enum Kind {
    LegacyExt3,
    OptExt3,
    OptExt3Par,
}

struct Filters {
    nv_min: usize,
    nv_max: usize,
    kinds: [bool; 3],
    ell0s: [bool; 3],
    warmup: usize,
    samples: usize,
}

impl Filters {
    fn from_env() -> Self {
        let l3 = env_flag("BENCH_LEGACY_EXT3_ONLY");
        let o3 = env_flag("BENCH_OPT_EXT3_ONLY");
        let o3p = env_flag("BENCH_OPT_EXT3_PAR_ONLY");
        let any_only = l3 || o3 || o3p;
        let kinds = if any_only {
            [l3, o3, o3p]
        } else {
            [true; 3]
        };

        let ell0s = match std::env::var("ELL0").ok() {
            Some(s) => {
                let v: usize = s.parse().expect("ELL0 must be 1, 2, or 3");
                assert!((1..=3).contains(&v), "ELL0 must be 1, 2, or 3 (got {v})");
                let mut arr = [false; 3];
                arr[v - 1] = true;
                arr
            }
            None => [true; 3],
        };

        let samples = env_usize("BENCH_SAMPLES", 5);
        assert!(samples >= 1, "BENCH_SAMPLES must be >= 1 (got {samples})");

        Filters {
            nv_min: env_usize("BENCH_NV_MIN", 18),
            nv_max: env_usize("BENCH_NV_MAX", 18),
            kinds,
            ell0s,
            warmup: env_usize("BENCH_WARMUP", 1),
            samples,
        }
    }

    fn kind(&self, k: Kind) -> bool {
        self.kinds[k as usize]
    }

    fn ell0(&self, v: usize) -> bool {
        self.ell0s[v - 1]
    }
}

// ---------------------------------------------------------------------------
// Input builders
// ---------------------------------------------------------------------------

fn pack_input<E: hyperplonk::sumcheck_mamabear::SumcheckExtField>(scalars: &[E::Scalar]) -> Vec<E> {
    use hyperplonk::sumcheck_mamabear::MontgomeryOps;
    let packed_len = scalars.len() / 8;
    (0..packed_len)
        .map(|i| {
            let base = i << 3;
            let s: [E::Scalar; 8] = std::array::from_fn(|k| scalars[base + k].to_montgomery());
            E::pack_scalars(&s)
        })
        .collect()
}

fn random_point_ext3(rng: &mut SmallRng, num_vars: usize) -> Vec<MamaBearScalarExt3> {
    (0..num_vars)
        .map(|_| MamaBearScalarExt3::random(&mut *rng))
        .collect()
}

fn build_selector(rng: &mut SmallRng, len: usize) -> AlignedPoly {
    let data: Vec<MamaBearScalar> = (0..len).map(|_| MamaBearScalar::random(&mut *rng)).collect();
    let mont: Vec<MamaBearScalar> = data.iter().map(|x| x.to_montgomery()).collect();
    AlignedPoly::from_sbf(&mont)
}

fn build_witness(rng: &mut SmallRng, len: usize) -> [AlignedPoly; 3] {
    std::array::from_fn(|_| {
        let data: Vec<MamaBearScalar> =
            (0..len).map(|_| MamaBearScalar::random(&mut *rng)).collect();
        let mut poly = AlignedPoly::from_sbf(&data);
        poly.to_montgomery_in_place();
        poly
    })
}

// ---------------------------------------------------------------------------
// Per-kind dispatch
// ---------------------------------------------------------------------------

fn bench_legacy_ext3(nv: usize) {
    let mut rng = SmallRng::seed_from_u64(42);
    let len = 1 << nv;
    let selector = build_selector(&mut rng, len);
    let witness = build_witness(&mut rng, len);
    let r = random_point_ext3(&mut rng, nv);

    measure(&format!("Legacy Ext3 NV={nv}"), || {
        let eq_r = MultiLinearPoly::new_eq(&r);
        let eq_packed = pack_input::<PackedMamaBearAVX512Ext3>(&eq_r.evals);
        let mut transcript = Transcript::new();
        let start = Instant::now();
        SumcheckMamaBear::prove_add_mul_ext3(
            [
                selector.as_pbf().to_vec(),
                witness[0].as_pbf().to_vec(),
                witness[1].as_pbf().to_vec(),
                witness[2].as_pbf().to_vec(),
            ],
            eq_packed,
            &mut transcript,
        );
        start.elapsed()
    });
}

fn bench_opt_ext3(nv: usize, ell0: usize) {
    let mut rng = SmallRng::seed_from_u64(42);
    let len = 1 << nv;
    let selector = build_selector(&mut rng, len);
    let witness = build_witness(&mut rng, len);
    let point = random_point_ext3(&mut rng, nv);

    measure(&format!("Optimized Ext3 ell0={ell0} NV={nv}"), || {
        let mut transcript = Transcript::new();
        let start = Instant::now();
        SumcheckMamaBear::prove_add_mul_ell0_ext3(
            [
                selector.as_pbf().to_vec(),
                witness[0].as_pbf().to_vec(),
                witness[1].as_pbf().to_vec(),
                witness[2].as_pbf().to_vec(),
            ],
            &point,
            ell0,
            &mut transcript,
        );
        start.elapsed()
    });
}

fn bench_opt_ext3_par(nv: usize, ell0: usize) {
    let mut rng = SmallRng::seed_from_u64(42);
    let len = 1 << nv;
    let selector = build_selector(&mut rng, len);
    let witness = build_witness(&mut rng, len);
    let point = random_point_ext3(&mut rng, nv);

    measure(&format!("Optimized Ext3 Par ell0={ell0} NV={nv}"), || {
        let mut transcript = Transcript::new();
        let start = Instant::now();
        SumcheckMamaBear::prove_add_mul_ell0_ext3_par(
            [
                selector.as_pbf().to_vec(),
                witness[0].as_pbf().to_vec(),
                witness[1].as_pbf().to_vec(),
                witness[2].as_pbf().to_vec(),
            ],
            &point,
            ell0,
            &mut transcript,
        );
        start.elapsed()
    });
}

fn bench_generic_ext3(nv: usize, ell0: usize) {
    let mut rng = SmallRng::seed_from_u64(42);
    let len = 1 << nv;
    let selector = build_selector(&mut rng, len);
    let witness = build_witness(&mut rng, len);
    let point = random_point_ext3(&mut rng, nv);

    measure(&format!("Generic D3 Ext3 ell0={ell0} NV={nv}"), || {
        let mut transcript = Transcript::new();
        let start = Instant::now();
        prove_zero_check_generic::<PackedMamaBearAVX512Ext3, AddMulD3, 3, 4>(
            [
                selector.as_pbf().to_vec(),
                witness[0].as_pbf().to_vec(),
                witness[1].as_pbf().to_vec(),
                witness[2].as_pbf().to_vec(),
            ],
            &point,
            ell0,
            true,
            &mut transcript,
        );
        start.elapsed()
    });
}

// ---------------------------------------------------------------------------
// Driver
// ---------------------------------------------------------------------------

fn main() {
    let f = Filters::from_env();
    MEASURE_CFG
        .set((f.warmup, f.samples))
        .expect("MEASURE_CFG set more than once");
    init_output_file();

    for nv in f.nv_min..=f.nv_max {
        if f.kind(Kind::LegacyExt3) {
            bench_legacy_ext3(nv);
        }
        for ell0 in 1..=3 {
            if !f.ell0(ell0) {
                continue;
            }
            if f.kind(Kind::OptExt3) {
                bench_opt_ext3(nv, ell0);
            }
            if env_flag("BENCH_GENERIC") {
                bench_generic_ext3(nv, ell0);
            }
            if f.kind(Kind::OptExt3Par) {
                bench_opt_ext3_par(nv, ell0);
            }
        }
    }
}
