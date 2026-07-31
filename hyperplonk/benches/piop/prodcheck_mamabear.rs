//! prodcheck_mamabear: custom warmup+median harness (replaces criterion).
//!
//! Env vars:
//!   BENCH_NV_MIN / BENCH_NV_MAX  inclusive NV range (default 18..=24)
//!   BENCH_PERWIRE_EXT3_ONLY
//!   BENCH_PERWIRE_EXT3_PAR_ONLY
//!     If any *_ONLY flag is set, only enabled kinds run; otherwise all run.
//!   BENCH_WARMUP=N   warmup iterations (default 2)
//!   BENCH_SAMPLES=N  measured iterations (default 5, must be >= 1)
//!   BENCH_OUTPUT_FILE=PATH  if set, append one line per cell to PATH
//!
//! Mirrors the criterion file (`prodcheck_mamabear_benches`): only the four
//! per-wire variants run by default. The scalar / packed comparison baselines
//! are intentionally omitted (they were commented out in the criterion driver).

#![allow(dead_code)]

use std::fs::OpenOptions;
use std::io::Write as _;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use arithmetic::field::mamabear::{
    MamaBearScalar, MamaBearScalarExt3, PackedMamaBearAVX512Ext3,
};
use arithmetic::field::Field;
use arithmetic::poly::MultiLinearPoly;
use hyperplonk::prodcheck_mamabear_perwire::ProdEqCheckMamaBearPerWire;
use hyperplonk::prodcheck_mamabear_perwire_par::ProdEqCheckMamaBearPerWirePar;
use hyperplonk::prover_mamabear::{build_productcheck_inputs, AlignedPoly};
use poly_commit::deepfold_mamabear::DeepFoldExtField;
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
    PerWireExt3,
    PerWireParExt3,
}

struct Filters {
    nv_min: usize,
    nv_max: usize,
    kinds: [bool; 2],
    warmup: usize,
    samples: usize,
}

impl Filters {
    fn from_env() -> Self {
        let e3 = env_flag("BENCH_PERWIRE_EXT3_ONLY");
        let e3p = env_flag("BENCH_PERWIRE_EXT3_PAR_ONLY");
        let any_only = e3 || e3p;
        let kinds = if any_only {
            [e3, e3p]
        } else {
            [true; 2]
        };
        let samples = env_usize("BENCH_SAMPLES", 5);
        assert!(samples >= 1, "BENCH_SAMPLES must be >= 1 (got {samples})");
        Filters {
            nv_min: env_usize("BENCH_NV_MIN", 18),
            nv_max: env_usize("BENCH_NV_MAX", 24),
            kinds,
            warmup: env_usize("BENCH_WARMUP", 1),
            samples,
        }
    }

    fn kind(&self, k: Kind) -> bool {
        self.kinds[k as usize]
    }
}

// ---------------------------------------------------------------------------
// Input builders
// ---------------------------------------------------------------------------

fn build_perwire_inputs(
    rng: &mut SmallRng,
    nv: usize,
) -> ([AlignedPoly; 3], [AlignedPoly; 3], [AlignedPoly; 3]) {
    let len = 1 << nv;
    let witness: [AlignedPoly; 3] = std::array::from_fn(|_| {
        let data: Vec<MamaBearScalar> =
            (0..len).map(|_| MamaBearScalar::random(&mut *rng)).collect();
        let mut poly = AlignedPoly::from_sbf(&data);
        poly.to_montgomery_in_place();
        poly
    });
    let identical: [AlignedPoly; 3] = [0u64, 1u64 << 29, 1u64 << 30].map(|offset| {
        let evals: Vec<MamaBearScalar> =
            MultiLinearPoly::new_identical(nv, MamaBearScalar::from(offset))
                .evals
                .into_iter()
                .map(MamaBearScalar::to_montgomery)
                .collect();
        AlignedPoly::from_sbf(&evals)
    });
    let permutation: [AlignedPoly; 3] = std::array::from_fn(|_| {
        let data: Vec<MamaBearScalar> =
            (0..len).map(|_| MamaBearScalar::random(&mut *rng)).collect();
        let mont: Vec<MamaBearScalar> = data.iter().map(|x| x.to_montgomery()).collect();
        AlignedPoly::from_sbf(&mont)
    });
    (witness, identical, permutation)
}

// ---------------------------------------------------------------------------
// Per-kind dispatch
// ---------------------------------------------------------------------------

fn bench_perwire_ext3(nv: usize) {
    let mut rng = SmallRng::seed_from_u64(42);
    let (witness, identical, permutation) = build_perwire_inputs(&mut rng, nv);

    measure(&format!("ProdEqCheck PerWire Ext3 NV={nv}"), || {
        let mut t = Transcript::new();
        let start = Instant::now();
        let prod_r0: MamaBearScalarExt3 = t.challenge_f();
        let prod_r1: MamaBearScalarExt3 = t.challenge_f();
        let inputs = build_productcheck_inputs::<MamaBearScalarExt3>(
            &witness,
            &identical,
            &permutation,
            prod_r0.to_mont(),
            prod_r1.to_mont(),
        );
        ProdEqCheckMamaBearPerWire::prove::<PackedMamaBearAVX512Ext3>(inputs, &mut t);
        start.elapsed()
    });
}

fn bench_perwire_par_ext3(nv: usize) {
    let mut rng = SmallRng::seed_from_u64(42);
    let (witness, identical, permutation) = build_perwire_inputs(&mut rng, nv);

    measure(&format!("ProdEqCheck PerWirePar Ext3 NV={nv}"), || {
        let mut t = Transcript::new();
        let start = Instant::now();
        let prod_r0: MamaBearScalarExt3 = t.challenge_f();
        let prod_r1: MamaBearScalarExt3 = t.challenge_f();
        let inputs = build_productcheck_inputs::<MamaBearScalarExt3>(
            &witness,
            &identical,
            &permutation,
            prod_r0.to_mont(),
            prod_r1.to_mont(),
        );
        ProdEqCheckMamaBearPerWirePar::prove::<PackedMamaBearAVX512Ext3>(inputs, &mut t);
        start.elapsed()
    });
}

fn main() {
    let f = Filters::from_env();
    MEASURE_CFG
        .set((f.warmup, f.samples))
        .expect("MEASURE_CFG set more than once");
    init_output_file();

    for nv in f.nv_min..=f.nv_max {
        if f.kind(Kind::PerWireExt3) {
            bench_perwire_ext3(nv);
        }
        if f.kind(Kind::PerWireParExt3) {
            bench_perwire_par_ext3(nv);
        }
    }
}
