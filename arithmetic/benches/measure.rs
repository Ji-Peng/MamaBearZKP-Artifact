//! Shared custom-harness scaffold for arithmetic field microbenches.
//!
//! Replaces criterion with a deterministic warmup + median-of-samples timer,
//! matching the style used in `hyperplonk/benches/hp_df_babybear.rs`.
//!
//! Env vars:
//!   BENCH_WARMUP=N          warmup iterations (default 10)
//!   BENCH_SAMPLES=N         measured iterations (default 100, must be >= 1)
//!   BENCH_OUTPUT_FILE=PATH  if set, append one aligned line per cell to PATH
//!
//! Output format (column-aligned so all field benches collected into one file
//! stay visually comparable):
//!
//!     {field:<22} {op:<12} {count:>12}  {median_ms:>10.3} ms

use std::fs::OpenOptions;
use std::io::Write as _;
use std::sync::{Mutex, OnceLock};
use std::time::Duration;

static MEASURE_CFG: OnceLock<(usize, usize)> = OnceLock::new();
static OUTPUT_FILE: OnceLock<Option<Mutex<std::fs::File>>> = OnceLock::new();

pub fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

/// Initialize the harness from environment variables. Must be called once
/// from `fn main()` before any `bench_op` call.
pub fn init_from_env() {
    let warmup = env_usize("BENCH_WARMUP", 1);
    let samples = env_usize("BENCH_SAMPLES", 10);
    assert!(samples >= 1, "BENCH_SAMPLES must be >= 1 (got {samples})");
    MEASURE_CFG
        .set((warmup, samples))
        .expect("MEASURE_CFG set more than once");

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

fn record(line: &str) {
    println!("{line}");
    if let Some(Some(m)) = OUTPUT_FILE.get() {
        let mut f = m.lock().unwrap();
        let _ = writeln!(f, "{line}");
        let _ = f.flush();
    }
}

/// Run a single bench cell: warmup iterations, then collect `samples` timings
/// and report the median in milliseconds on a single aligned line.
///
/// The `one_sample` closure must return the elapsed time for one unit of work
/// (e.g., one outer iter batch). The caller is responsible for constructing
/// inputs inside the closure if those should not be timed; only the `start..
/// elapsed` span within the closure counts toward the measurement.
///
/// `count` is the total logical operation count credited to one sample
/// (typically `REPS * 10 * LANES`); it is emitted verbatim so that results
/// across types are unified to the same denominator.
pub fn bench_op<F: FnMut() -> Duration>(field: &str, op: &str, count: usize, mut one_sample: F) {
    let (warmup, samples) = *MEASURE_CFG
        .get()
        .expect("measure::init_from_env() must be called before bench_op");
    for _ in 0..warmup {
        let _ = one_sample();
    }
    let mut times: Vec<Duration> = (0..samples).map(|_| one_sample()).collect();
    times.sort();
    let median_ms = times[samples / 2].as_secs_f64() * 1000.0;
    record(&format!(
        "{field:<22} {op:<12} {count:>12}  {median_ms:>10.3} ms"
    ));
}
