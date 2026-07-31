//! deepfold_mamabear_par: custom warmup+median harness (replaces criterion).
//!
//! Mirrors `deepfold_mamabear.rs` but uses parallel commit (`new_par`) and
//! parallel open (`open_par`).
//!
//! Env vars:
//!   BENCH_NV_MIN / BENCH_NV_MAX  inclusive NV range (default 18..=18)
//!   BENCH_SPLIT=N      override split level (default DEFAULT_SPLIT_LEVEL)
//!   BENCH_SECURITY=conj96|prov97|prov128  FRI security level (DEFAULT prov128)
//!   RAYON_NUM_THREADS  controls rayon worker count (caller sets this)
//!   BENCH_WARMUP=N     warmup iterations (default 2)
//!   BENCH_SAMPLES=N    measured iterations (default 5, must be >= 1)
//!   BENCH_OUTPUT_FILE=PATH  if set, append one line per cell to PATH

use std::fs::OpenOptions;
use std::io::Write as _;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use arithmetic::field::mamabear::{MamaBearScalar, MamaBearScalarExt3, P};
use poly_commit::{
    deepfold::MerkleRoot,
    deepfold_mamabear::{
        DeepFoldExtField, DeepFoldMamaBearParam, DeepFoldMamaBearProver, DEFAULT_SPLIT_LEVEL,
    },
    deepfold_mamabear_par::new_par,
    CommitmentSerde,
};
use rand::rngs::SmallRng;
use rand::{RngCore, SeedableRng};
use util::fiat_shamir::Transcript;
use util::params::{
    mamabear::{GRINDING_BITS_EXT3_PROV_QUERY128, QUERY_NUM_CONJ96, QUERY_NUM_PROV_QUERY128},
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

// ---------------------------------------------------------------------------
// Random helpers
// ---------------------------------------------------------------------------

fn rand_poly(rng: &mut SmallRng, len: usize) -> Vec<MamaBearScalar> {
    (0..len).map(|_| MamaBearScalar(rng.next_u64() % P)).collect()
}

fn rand_ext3(rng: &mut SmallRng) -> MamaBearScalarExt3 {
    MamaBearScalarExt3 {
        c0: MamaBearScalar(rng.next_u64() % P),
        c1: MamaBearScalar(rng.next_u64() % P),
        c2: MamaBearScalar(rng.next_u64() % P),
    }
}

// ---------------------------------------------------------------------------
// Par case
// ---------------------------------------------------------------------------

struct CasePar<F: DeepFoldExtField> {
    nv: usize,
    split: usize,
    pp: DeepFoldMamaBearParam,
    witness: Vec<Vec<MamaBearScalar>>,
    prover_key_commitments: DeepFoldMamaBearProver<F>,
    witness_pc: DeepFoldMamaBearProver<F>,
    point_mont: Vec<F>,
    commit_buf_size: usize,
}

fn build_case_par<F, FRand>(
    nv: usize,
    split: usize,
    sec: SecurityLevel,
    grinding_bits: u32,
    mut rand_pt_raw: FRand,
) -> CasePar<F>
where
    F: DeepFoldExtField,
    FRand: FnMut(&mut SmallRng) -> F,
{
    let mut rng = SmallRng::seed_from_u64(42);
    let len = 1usize << nv;

    let selector = rand_poly(&mut rng, len);
    let permutation: Vec<Vec<MamaBearScalar>> = (0..3).map(|_| rand_poly(&mut rng, len)).collect();
    let witness: Vec<Vec<MamaBearScalar>> =
        (0..NUM_WITNESS).map(|_| rand_poly(&mut rng, len)).collect();
    let point_raw: Vec<F> = (0..nv).map(|_| rand_pt_raw(&mut rng)).collect();
    let point_mont: Vec<F> = point_raw.iter().map(|x| x.to_mont()).collect();

    let mut pp = DeepFoldMamaBearParam::new(nv, CODE_RATE_LOG, sec.query_num(), split);
    pp.grinding_bits = grinding_bits;

    let fixed_refs: Vec<&[MamaBearScalar]> = std::iter::once(selector.as_slice())
        .chain(permutation.iter().map(|p| p.as_slice()))
        .collect();
    let prover_key_commitments = new_par::<F>(&pp, &fixed_refs);

    let witness_refs: Vec<&[MamaBearScalar]> = witness.iter().map(|w| w.as_slice()).collect();
    let witness_pc = new_par::<F>(&pp, &witness_refs);

    CasePar {
        nv,
        split,
        pp,
        witness,
        prover_key_commitments,
        witness_pc,
        point_mont,
        commit_buf_size: MerkleRoot::size(nv, NUM_WITNESS),
    }
}

fn bench_commit_par<F: DeepFoldExtField>(
    case: &CasePar<F>,
    ext_label: &str,
    sec_label: &str,
) {
    measure(
        &format!(
            "{ext_label} {sec_label} Commit Par NV={} split={}",
            case.nv, case.split
        ),
        || {
            let witness = case.witness.clone();
            let start = Instant::now();
            let witness_refs: Vec<&[MamaBearScalar]> =
                witness.iter().map(|w| w.as_slice()).collect();
            let prover = new_par::<F>(&case.pp, &witness_refs);
            let commit = prover.commit();
            let mut buffer = vec![0u8; case.commit_buf_size];
            commit.serialize_into(&mut buffer);
            let mut transcript = Transcript::new();
            transcript.append_u8_slice(&buffer, buffer.len());
            start.elapsed()
        },
    );
}

fn bench_open_par<F: DeepFoldExtField>(
    case: &CasePar<F>,
    ext_label: &str,
    sec_label: &str,
) {
    measure(
        &format!(
            "{ext_label} {sec_label} Open Par NV={} split={}",
            case.nv, case.split
        ),
        || {
            let mut transcript = Transcript::new();
            let start = Instant::now();
            DeepFoldMamaBearProver::<F>::open_par(
                &case.pp,
                &[&case.prover_key_commitments, &case.witness_pc],
                case.point_mont.clone(),
                &mut transcript,
            );
            start.elapsed()
        },
    );
}

// ---------------------------------------------------------------------------
// Driver
// ---------------------------------------------------------------------------

fn main() {
    let nv_min = env_usize("BENCH_NV_MIN", 18);
    let nv_max = env_usize("BENCH_NV_MAX", 18);
    let split = env_usize("BENCH_SPLIT", DEFAULT_SPLIT_LEVEL);
    let sec = select_security();
    let sec_label = sec.label();

    let warmup = env_usize("BENCH_WARMUP", 1);
    let samples = env_usize("BENCH_SAMPLES", 5);
    assert!(samples >= 1, "BENCH_SAMPLES must be >= 1 (got {samples})");
    MEASURE_CFG
        .set((warmup, samples))
        .expect("MEASURE_CFG set more than once");
    init_output_file();

    for nv in nv_min..=nv_max {
        match sec {
            SecurityLevel::Conj96 | SecurityLevel::Prov97 => panic!(
                "BENCH_SECURITY={sec_label} is not supported for MamaBear: conj96/prov97 used \
                 to route to MamaBear Ext2, which has been removed (insufficient soundness). \
                 MamaBear only supports prov128 (Ext3)."
            ),
            SecurityLevel::Prov128 => {
                let case = build_case_par::<MamaBearScalarExt3, _>(
                    nv,
                    split,
                    sec,
                    GRINDING_BITS_EXT3_PROV_QUERY128,
                    |rng| rand_ext3(rng),
                );
                bench_commit_par(&case, "Ext3", sec_label);
                bench_open_par(&case, "Ext3", sec_label);
                drop(case);
            }
        }
    }
}
