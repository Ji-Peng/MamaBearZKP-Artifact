//! Merkle-leaf hash throughput calibration probe.
//!
//! Times the exact leaf-hashing kernel the DeepFold commit stage uses
//! (`blake3_batch::hash_leaves_batch_flat`), which dispatches to the 16-way
//! AVX-512 / 8-way AVX-2 batch hasher on x86_64 and the 4-way NEON batch
//! hasher on aarch64. Reporting the same kernel on both machines yields the
//! cross-machine hash-throughput ratio used to calibrate the streaming/hash
//! stages of the end-to-end expectation model (the commit stage is
//! hash+FFT dominated, so its cross-machine cost tracks this ratio, not the
//! field-multiply ratio).
//!
//! Output is bytes/s and leaves/s per leaf length, so the two machines are
//! compared at matched leaf length. No arch-specific code lives here — the
//! kernel selects its SIMD arm internally — so the bench cross-compiles and
//! runs unchanged on x86_64 and aarch64.
//!
//! Env vars:
//!   HASH_CALIB_LEAF_LENS   comma-separated leaf byte lengths (default
//!                          "16,24,32,64,128"; 24 = one Ext3 field element
//!                          triple, the DeepFold fat-leaf granule)
//!   HASH_CALIB_COUNT       number of leaves hashed per timed pass
//!                          (default 1<<20)
//!   BENCH_WARMUP           warmup passes (default 1)
//!   BENCH_SAMPLES          measured passes; the median is reported (default 5)
//!   BENCH_OUTPUT_FILE      if set, append one line per leaf length to PATH

use std::fs::OpenOptions;
use std::hint::black_box;
use std::io::Write as _;
use std::time::{Duration, Instant};

use util::blake3_batch::hash_leaves_batch_flat;

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key).ok().and_then(|s| s.parse().ok()).unwrap_or(default)
}

fn median(mut v: Vec<Duration>) -> Duration {
    v.sort_unstable();
    v[v.len() / 2]
}

fn main() {
    let leaf_lens: Vec<usize> = std::env::var("HASH_CALIB_LEAF_LENS")
        .unwrap_or_else(|_| "16,24,32,64,128".to_string())
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    let count = env_usize("HASH_CALIB_COUNT", 1 << 20);
    let warmup = env_usize("BENCH_WARMUP", 1);
    let samples = env_usize("BENCH_SAMPLES", 5).max(1);

    let out_path = std::env::var("BENCH_OUTPUT_FILE").ok();

    #[cfg(target_arch = "x86_64")]
    let arch = "x86_64";
    #[cfg(target_arch = "aarch64")]
    let arch = "aarch64";
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    let arch = "other";

    println!("# hash_calib: production Merkle-leaf batch hash (blake3_batch), arch={arch}");
    println!("# leaf_len  count  median_ms  bytes_per_s  leaves_per_s");

    for &leaf_len in &leaf_lens {
        // Deterministic non-zero leaf bytes; content is irrelevant to hash cost.
        let data: Vec<u8> = (0..count * leaf_len).map(|i| (i as u8) | 1).collect();
        let mut out = vec![[0u8; 32]; count];

        for _ in 0..warmup {
            hash_leaves_batch_flat(&data, count, leaf_len, &mut out);
            black_box(&out);
        }

        let mut times = Vec::with_capacity(samples);
        for _ in 0..samples {
            let t0 = Instant::now();
            hash_leaves_batch_flat(black_box(&data), count, leaf_len, &mut out);
            times.push(t0.elapsed());
            black_box(&out);
        }
        let med = median(times);
        let secs = med.as_secs_f64();
        let bytes = (count * leaf_len) as f64;
        let bytes_per_s = bytes / secs;
        let leaves_per_s = count as f64 / secs;

        println!(
            "{:>8}  {:>8}  {:>9.3}  {:>14.3e}  {:>14.3e}",
            leaf_len,
            count,
            secs * 1e3,
            bytes_per_s,
            leaves_per_s
        );

        if let Some(p) = &out_path {
            if let Ok(mut f) = OpenOptions::new().create(true).append(true).open(p) {
                let _ = writeln!(
                    f,
                    "hash_calib arch={arch} leaf_len={leaf_len} count={count} median_ms={:.3} bytes_per_s={:.3e} leaves_per_s={:.3e}",
                    secs * 1e3,
                    bytes_per_s,
                    leaves_per_s
                );
            }
        }
    }
}
