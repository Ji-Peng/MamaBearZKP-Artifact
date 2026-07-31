//! grinding_bench: custom warmup+median harness (replaces criterion).
//!
//! Measures `util::grinding`'s BLAKE3 PoW search cost at 16/20/24/28/32 leading
//! zero bits, for both serial and rayon-parallel paths.
//!
//! Env vars:
//!   GRINDING_BITS=16,20,24      comma-separated subset of {16,20,24,28,32}
//!                               (default 16,20,24)
//!   GRINDING_MODE=serial|parallel|both   default both
//!   BENCH_WARMUP=N    warmup iterations (default 2; forced to 1 for bits>=28)
//!   BENCH_SAMPLES=N   measured iterations (default 5; forced to 3 for bits>=28)
//!   BENCH_OUTPUT_FILE=PATH   if set, append one line per cell to PATH
//!
//! 32-bit serial is ~1 minute per sample on a single core; high-bit (>=28) runs
//! force a smaller sample count (3 + 1 warmup) regardless of the env to cap
//! wall time.

use std::fs::OpenOptions;
use std::hint::black_box;
use std::io::Write as _;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use util::grinding::{grind_blake3_par, grind_blake3_serial};

// ---------------------------------------------------------------------------
// Shared harness scaffold
// ---------------------------------------------------------------------------

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

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

fn record(line: &str) {
    println!("{line}");
    if let Some(Some(m)) = OUTPUT_FILE.get() {
        let mut f = m.lock().unwrap();
        let _ = writeln!(f, "{line}");
        let _ = f.flush();
    }
}

fn measure_with<F: FnMut() -> Duration>(label: &str, warmup: usize, samples: usize, mut one_sample: F) {
    for _ in 0..warmup {
        let _ = one_sample();
    }
    let mut times: Vec<Duration> = (0..samples).map(|_| one_sample()).collect();
    times.sort();
    let median_ms = times[samples / 2].as_secs_f64() * 1000.0;
    record(&format!("{label:<60} {median_ms:>10.3} ms"));
}

// ---------------------------------------------------------------------------

fn seed_for_bits(bits: u32) -> [u8; 32] {
    let mut s = [0u8; 32];
    s[0..4].copy_from_slice(&bits.to_le_bytes());
    for i in 4..32 {
        s[i] = 0xA5 ^ (i as u8);
    }
    s
}

fn parse_bits_list() -> Vec<u32> {
    let raw = std::env::var("GRINDING_BITS").unwrap_or_else(|_| "16,20,24".to_string());
    raw.split(',')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(|s| {
            let v: u32 = s.parse().unwrap_or_else(|_| {
                panic!("GRINDING_BITS: cannot parse {s:?} as u32 (need 16|20|24|28|32)")
            });
            assert!(
                matches!(v, 16 | 20 | 24 | 28 | 32),
                "GRINDING_BITS: {v} not in {{16,20,24,28,32}}"
            );
            v
        })
        .collect()
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    Serial,
    Parallel,
    Both,
}

fn parse_mode() -> Mode {
    match std::env::var("GRINDING_MODE").ok().as_deref() {
        Some("serial") => Mode::Serial,
        Some("parallel") => Mode::Parallel,
        Some("both") | None => Mode::Both,
        Some(other) => panic!("GRINDING_MODE: {other:?} not in {{serial,parallel,both}}"),
    }
}

fn bench_serial(bits: u32, warmup: usize, samples: usize) {
    measure_with(
        &format!("grinding serial bits={bits}"),
        warmup,
        samples,
        || {
            let seed = seed_for_bits(bits);
            let start = Instant::now();
            let w = grind_blake3_serial(black_box(seed), black_box(bits));
            let elapsed = start.elapsed();
            black_box(w);
            elapsed
        },
    );
}

fn bench_parallel(bits: u32, warmup: usize, samples: usize) {
    measure_with(
        &format!("grinding parallel bits={bits}"),
        warmup,
        samples,
        || {
            let seed = seed_for_bits(bits);
            let start = Instant::now();
            let w = grind_blake3_par(black_box(seed), black_box(bits));
            let elapsed = start.elapsed();
            black_box(w);
            elapsed
        },
    );
}

fn main() {
    let bits_list = parse_bits_list();
    let mode = parse_mode();
    let warmup_default = env_usize("BENCH_WARMUP", 1);
    let samples_default = env_usize("BENCH_SAMPLES", 5);
    assert!(samples_default >= 1, "BENCH_SAMPLES must be >= 1");

    init_output_file();

    for &bits in &bits_list {
        // Cap the sample budget on the slow (>=28-bit) variants regardless of
        // env. 32-bit serial is ~60s per sample; even 3+1 runs is ~4 minutes.
        let (warmup, samples) = if bits >= 28 {
            (1, 3)
        } else {
            (warmup_default, samples_default)
        };
        if matches!(mode, Mode::Serial | Mode::Both) {
            bench_serial(bits, warmup, samples);
        }
        if matches!(mode, Mode::Parallel | Mode::Both) {
            bench_parallel(bits, warmup, samples);
        }
    }
}
