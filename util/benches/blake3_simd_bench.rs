///! BLAKE3 SIMD microbenchmark: scalar vs AVX-2 (8-way) vs AVX-512 (16-way)
///!
///! Tests hashing 16-byte inputs (typical Merkle leaf size in DeepFold: 2 MamaBear Ext2 = 16B).
///!
///! Usage:
///!   RUSTFLAGS="-C target-cpu=native" cargo bench -p util --bench blake3_simd_bench

use criterion::{criterion_group, criterion_main, Criterion, black_box};
use std::time::Duration;

// ── BLAKE3 constants ───────────────────────────────────────────────────────
const IV: [u32; 8] = [
    0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
    0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19,
];
const CHUNK_START: u8 = 1 << 0;
const CHUNK_END: u8 = 1 << 1;
const PARENT: u8 = 1 << 2;
const OUT_LEN: usize = 32;
const BLOCK_LEN: usize = 64;

unsafe extern "C" {
    fn blake3_hash_many_avx512(
        inputs: *const *const u8,
        num_inputs: usize,
        blocks: usize,
        key: *const u32,
        counter: u64,
        increment_counter: bool,
        flags: u8,
        flags_start: u8,
        flags_end: u8,
        out: *mut u8,
    );
    fn blake3_hash_many_avx2(
        inputs: *const *const u8,
        num_inputs: usize,
        blocks: usize,
        key: *const u32,
        counter: u64,
        increment_counter: bool,
        flags: u8,
        flags_start: u8,
        flags_end: u8,
        out: *mut u8,
    );
}

// ── Benchmark: scalar blake3::hash on 16-byte inputs ───────────────────────
fn bench_scalar_16b(c: &mut Criterion) {
    // 16 independent 16-byte inputs (to match the total work of 16-way AVX-512)
    let inputs: Vec<[u8; 16]> = (0..16u64)
        .map(|i| {
            let mut buf = [0u8; 16];
            buf[..8].copy_from_slice(&i.to_le_bytes());
            buf
        })
        .collect();

    c.bench_function("scalar blake3::hash x16 (16B each)", |b| {
        b.iter(|| {
            let mut out = [[0u8; 32]; 16];
            for i in 0..16 {
                out[i] = *blake3::hash(black_box(&inputs[i])).as_bytes();
            }
            black_box(out);
        });
    });

    // Also benchmark single hash for per-call latency
    c.bench_function("scalar blake3::hash x1 (16B)", |b| {
        b.iter(|| {
            black_box(*blake3::hash(black_box(&inputs[0])).as_bytes());
        });
    });
}

// ── Benchmark: AVX-2 8-way batch on 16-byte inputs (padded to 64B) ────────
fn bench_avx2_8way_16b(c: &mut Criterion) {
    // 8 inputs, each 16 bytes padded to 64 bytes
    let padded: Vec<[u8; BLOCK_LEN]> = (0..8u64)
        .map(|i| {
            let mut buf = [0u8; BLOCK_LEN];
            buf[..8].copy_from_slice(&i.to_le_bytes());
            buf
        })
        .collect();

    c.bench_function("AVX2 hash_many x8 (16B padded to 64B)", |b| {
        b.iter(|| {
            let ptrs: Vec<*const u8> = padded.iter().map(|p| p.as_ptr()).collect();
            let mut out = [0u8; 8 * OUT_LEN];
            unsafe {
                blake3_hash_many_avx2(
                    black_box(ptrs.as_ptr()),
                    8,
                    1, // 64 / 64 = 1 block
                    IV.as_ptr(),
                    0,
                    false,
                    0,
                    CHUNK_START | CHUNK_END,
                    0,
                    out.as_mut_ptr(),
                );
            }
            black_box(out);
        });
    });

    // Also bench 16 inputs via AVX-2 (two batches of 8) for fair comparison
    let padded16: Vec<[u8; BLOCK_LEN]> = (0..16u64)
        .map(|i| {
            let mut buf = [0u8; BLOCK_LEN];
            buf[..8].copy_from_slice(&i.to_le_bytes());
            buf
        })
        .collect();

    c.bench_function("AVX2 hash_many x16 (16B padded to 64B, 2 batches)", |b| {
        b.iter(|| {
            let ptrs: Vec<*const u8> = padded16.iter().map(|p| p.as_ptr()).collect();
            let mut out = [0u8; 16 * OUT_LEN];
            unsafe {
                // AVX-2 hash_many handles > 8 inputs by cascading: 8+8
                blake3_hash_many_avx2(
                    black_box(ptrs.as_ptr()),
                    16,
                    1,
                    IV.as_ptr(),
                    0,
                    false,
                    0,
                    CHUNK_START | CHUNK_END,
                    0,
                    out.as_mut_ptr(),
                );
            }
            black_box(out);
        });
    });
}

// ── Benchmark: AVX-512 16-way batch on 16-byte inputs (padded to 64B) ─────
fn bench_avx512_16way_16b(c: &mut Criterion) {
    // 16 inputs, each 16 bytes padded to 64 bytes
    let padded: Vec<[u8; BLOCK_LEN]> = (0..16u64)
        .map(|i| {
            let mut buf = [0u8; BLOCK_LEN];
            buf[..8].copy_from_slice(&i.to_le_bytes());
            buf
        })
        .collect();

    c.bench_function("AVX512 hash_many x16 (16B padded to 64B)", |b| {
        b.iter(|| {
            let ptrs: Vec<*const u8> = padded.iter().map(|p| p.as_ptr()).collect();
            let mut out = [0u8; 16 * OUT_LEN];
            unsafe {
                blake3_hash_many_avx512(
                    black_box(ptrs.as_ptr()),
                    16,
                    1,
                    IV.as_ptr(),
                    0,
                    false,
                    0,
                    CHUNK_START | CHUNK_END,
                    0,
                    out.as_mut_ptr(),
                );
            }
            black_box(out);
        });
    });

    // 8-way via AVX-512 (uses internal blake3_hash8_avx512)
    let padded8: Vec<[u8; BLOCK_LEN]> = (0..8u64)
        .map(|i| {
            let mut buf = [0u8; BLOCK_LEN];
            buf[..8].copy_from_slice(&i.to_le_bytes());
            buf
        })
        .collect();

    c.bench_function("AVX512 hash_many x8 (16B padded to 64B)", |b| {
        b.iter(|| {
            let ptrs: Vec<*const u8> = padded8.iter().map(|p| p.as_ptr()).collect();
            let mut out = [0u8; 8 * OUT_LEN];
            unsafe {
                blake3_hash_many_avx512(
                    black_box(ptrs.as_ptr()),
                    8,
                    1,
                    IV.as_ptr(),
                    0,
                    false,
                    0,
                    CHUNK_START | CHUNK_END,
                    0,
                    out.as_mut_ptr(),
                );
            }
            black_box(out);
        });
    });
}

// ── Benchmark: parent hashing (64B input = two 32B child hashes) ───────────
fn bench_parent_hashing(c: &mut Criterion) {
    // Scalar: hash 16 parents one by one
    let parents: Vec<[u8; 64]> = (0..16u64)
        .map(|i| {
            let mut buf = [0u8; 64];
            buf[..8].copy_from_slice(&i.to_le_bytes());
            buf[32..40].copy_from_slice(&(i + 100).to_le_bytes());
            buf
        })
        .collect();

    c.bench_function("scalar blake3::hash x16 (64B parent)", |b| {
        b.iter(|| {
            let mut out = [[0u8; 32]; 16];
            for i in 0..16 {
                out[i] = *blake3::hash(black_box(&parents[i])).as_bytes();
            }
            black_box(out);
        });
    });

    c.bench_function("AVX2 hash_many x16 (64B parent, PARENT flag)", |b| {
        b.iter(|| {
            let ptrs: Vec<*const u8> = parents.iter().map(|p| p.as_ptr()).collect();
            let mut out = [0u8; 16 * OUT_LEN];
            unsafe {
                blake3_hash_many_avx2(
                    black_box(ptrs.as_ptr()),
                    16,
                    1,
                    IV.as_ptr(),
                    0,
                    false,
                    PARENT,
                    0,
                    0,
                    out.as_mut_ptr(),
                );
            }
            black_box(out);
        });
    });

    c.bench_function("AVX512 hash_many x16 (64B parent, PARENT flag)", |b| {
        b.iter(|| {
            let ptrs: Vec<*const u8> = parents.iter().map(|p| p.as_ptr()).collect();
            let mut out = [0u8; 16 * OUT_LEN];
            unsafe {
                blake3_hash_many_avx512(
                    black_box(ptrs.as_ptr()),
                    16,
                    1,
                    IV.as_ptr(),
                    0,
                    false,
                    PARENT,
                    0,
                    0,
                    out.as_mut_ptr(),
                );
            }
            black_box(out);
        });
    });
}

// ── Large batch: simulate Merkle tree level ────────────────────────────────
fn bench_merkle_level(c: &mut Criterion) {
    // Simulate hashing 2^14 = 16384 parent nodes (typical for NV=18 Merkle tree)
    let n = 16384usize;
    let parents: Vec<[u8; 64]> = (0..n as u64)
        .map(|i| {
            let mut buf = [0u8; 64];
            buf[..8].copy_from_slice(&i.to_le_bytes());
            buf
        })
        .collect();

    c.bench_function(&format!("scalar blake3::hash x{n} (64B parent)"), |b| {
        b.iter(|| {
            let mut out = vec![[0u8; 32]; n];
            for i in 0..n {
                out[i] = *blake3::hash(black_box(&parents[i])).as_bytes();
            }
            black_box(&out);
        });
    });

    c.bench_function(&format!("AVX512 hash_many x{n} (64B parent)"), |b| {
        b.iter(|| {
            let ptrs: Vec<*const u8> = parents.iter().map(|p| p.as_ptr()).collect();
            let mut out = vec![0u8; n * OUT_LEN];
            unsafe {
                blake3_hash_many_avx512(
                    black_box(ptrs.as_ptr()),
                    n,
                    1,
                    IV.as_ptr(),
                    0,
                    false,
                    PARENT,
                    0,
                    0,
                    out.as_mut_ptr(),
                );
            }
            black_box(&out);
        });
    });
}

criterion_group! {
    name = blake3_simd;
    config =
        Criterion::default()
        .warm_up_time(Duration::from_secs(3))
        .measurement_time(Duration::from_secs(5))
        .sample_size(100);
    targets = bench_scalar_16b, bench_avx2_8way_16b, bench_avx512_16way_16b,
              bench_parent_hashing, bench_merkle_level
}
criterion_main!(blake3_simd);
