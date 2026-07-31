//! hyperplonk_basefold_goldilocks: custom warmup+median harness (replaces criterion).
//!
//! Random-circuit HyperPlonk + BaseFold (Goldilocks).
//!
//! Env vars:
//!   BENCH_NV_MIN / BENCH_NV_MAX  inclusive NV range (default 18..=18)
//!   BENCH_WARMUP=N    warmup iterations (default 1)
//!   BENCH_SAMPLES=N   measured iterations (default 5, must be >= 1)
//!   BENCH_OUTPUT_FILE=PATH  if set, append one line per cell to PATH

use std::fs::OpenOptions;
use std::hint::black_box;
use std::io::Write as _;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use arithmetic::{
    field::{
        goldilocks64::{Goldilocks64, Goldilocks64Ext},
        Field,
    },
    mul_group::Radix2Group,
};
use hyperplonk::{circuit::Circuit, prover::Prover, verifier::Verifier};
use poly_commit::basefold::{BaseFoldParam, BaseFoldVerifier, BasefoldProver};
use rand::rngs::SmallRng;
use rand::SeedableRng;
use util::fiat_shamir::Proof;

type BasefoldGoldilocksProver = Prover<Goldilocks64Ext, BasefoldProver<Goldilocks64Ext>>;
type BasefoldGoldilocksVerifier = Verifier<Goldilocks64Ext, BaseFoldVerifier<Goldilocks64Ext>>;

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

struct BenchCase {
    nv: usize,
    pp: BaseFoldParam<Goldilocks64Ext>,
    prover: BasefoldGoldilocksProver,
    verifier: BasefoldGoldilocksVerifier,
    witness: [Vec<Goldilocks64>; 3],
    proof: Proof,
}

fn build_mult_subgroups(nv: u32) -> Vec<Radix2Group<Goldilocks64>> {
    let mut mult_subgroups = vec![Radix2Group::<Goldilocks64>::new(nv + 2)];
    for i in 1..nv as usize {
        mult_subgroups.push(mult_subgroups[i - 1].exp(2));
    }
    mult_subgroups
}

fn build_case(nv: usize) -> BenchCase {
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
    let pp = BaseFoldParam::<Goldilocks64Ext> {
        mult_subgroups: build_mult_subgroups(nv as u32),
        variable_num: nv,
        query_num: 120,
    };
    let (pk, vk) = circuit.setup::<BasefoldProver<_>, BaseFoldVerifier<_>>(&pp, &pp);
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
    let witness = [a, b, c];
    let proof = prover.prove(&pp, nv, witness.clone());
    assert!(verifier.verify(&pp, nv, proof.clone()));
    BenchCase {
        nv,
        pp,
        prover,
        verifier,
        witness,
        proof,
    }
}

fn bench_case(case: &BenchCase) {
    measure(&format!("HP BF Goldilocks prove NV={}", case.nv), || {
        let witness = case.witness.clone();
        let start = Instant::now();
        black_box(case.prover.prove(&case.pp, case.nv, witness));
        start.elapsed()
    });

    measure(&format!("HP BF Goldilocks verify NV={}", case.nv), || {
        let proof = case.proof.clone();
        let start = Instant::now();
        black_box(case.verifier.verify(&case.pp, case.nv, proof));
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
        let case = build_case(nv);
        bench_case(&case);
    }
}
