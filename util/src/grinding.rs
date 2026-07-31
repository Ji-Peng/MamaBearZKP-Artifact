//! FRI grinding (proof-of-work).
//!
//! Prover searches for a `u64` witness `w` such that
//! `BLAKE3(seed || w_le)` has `bits` low-order zero bits in the first
//! little-endian u32 word of the output (Plonky3 convention).
//!
//! Hashing uses the same AVX-512 / AVX-2 FFI as `blake3_batch.rs`. 16 candidates
//! per batch on AVX-512, 8 per batch on AVX-2. Each 64-byte block per candidate
//! is `seed (32B) || nonce_le (8B) || pad (24B)`.
//!
//! Two search entry points:
//! - `grind_blake3_serial`: single-threaded, for serial provers.
//! - `grind_blake3_par`: rayon-parallel with `AtomicU64` early-exit, for parallel provers.
//!
//! Verifier side (`check_blake3`) is one BLAKE3 call; threading model irrelevant.

use std::sync::atomic::{AtomicU64, Ordering};

// Shared BLAKE3 constants (kept local to avoid cross-module private exposure).
const IV: [u32; 8] = [
    0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
    0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19,
];
const CHUNK_START: u8 = 1 << 0;
const CHUNK_END: u8 = 1 << 1;
const OUT_LEN: usize = 32;
const BLOCK_LEN: usize = 64;

// AVX-512 handles up to 16 candidates per call; AVX-2 up to 8.
const BATCH_AVX512: usize = 16;
const BATCH_AVX2: usize = 8;

// FFI: same C symbols as `blake3_batch.rs`. x86_64 links AVX-512/AVX-2; aarch64
// links the NEON batch hasher plus blake3's portable single-block compression
// (exported by the crate's NEON C build). See `blake3_batch.rs` for details.
#[cfg(target_arch = "x86_64")]
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

#[cfg(target_arch = "aarch64")]
unsafe extern "C" {
    fn blake3_hash_many_neon(
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
    fn blake3_compress_in_place_portable(
        cv: *mut u32,
        block: *const u8,
        block_len: u8,
        counter: u64,
        flags: u8,
    );
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum SimdLevel {
    #[cfg(target_arch = "x86_64")]
    Avx512,
    #[cfg(target_arch = "x86_64")]
    Avx2,
    #[cfg(target_arch = "aarch64")]
    Neon,
    #[cfg(target_arch = "aarch64")]
    Portable,
}

fn detect_simd() -> SimdLevel {
    use std::sync::OnceLock;
    static LEVEL: OnceLock<SimdLevel> = OnceLock::new();
    *LEVEL.get_or_init(|| {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512vl") {
                return SimdLevel::Avx512;
            }
            SimdLevel::Avx2
        }
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                return SimdLevel::Neon;
            }
            SimdLevel::Portable
        }
    })
}

fn batch_size() -> usize {
    match detect_simd() {
        #[cfg(target_arch = "x86_64")]
        SimdLevel::Avx512 => BATCH_AVX512,
        #[cfg(target_arch = "x86_64")]
        SimdLevel::Avx2 => BATCH_AVX2,
        // NEON cascades 4→1 internally; the portable fallback loops per input.
        // Batch width is correctness-neutral (all cascade internally); reuse the
        // 8-wide batch so the fixed-size scratch arrays below still fit.
        #[cfg(target_arch = "aarch64")]
        SimdLevel::Neon => BATCH_AVX2,
        #[cfg(target_arch = "aarch64")]
        SimdLevel::Portable => BATCH_AVX2,
    }
}

/// Mask for the PoW check. `bits == 32` returns `u32::MAX`; smaller values
/// return the standard `(1 << bits) - 1`. Avoids the `1u32 << 32` UB.
#[inline(always)]
fn pow_mask(bits: u32) -> u32 {
    debug_assert!(bits <= 32, "grinding bits must be <= 32");
    if bits >= 32 { u32::MAX } else { (1u32 << bits) - 1 }
}

/// Format 64-byte BLAKE3 input blocks for `n` consecutive nonces starting at
/// `nonce_base`. Each block = seed (32B) || nonce_le (8B) || zero pad (24B).
#[inline]
fn fill_batch(seed: &[u8; 32], nonce_base: u64, n: usize, blocks: &mut [[u8; BLOCK_LEN]]) {
    debug_assert!(n <= blocks.len());
    for i in 0..n {
        let blk = &mut blocks[i];
        blk[..32].copy_from_slice(seed);
        blk[32..40].copy_from_slice(&(nonce_base + i as u64).to_le_bytes());
        // 40..64 stays zero (initialised on first use; caller must pre-zero).
    }
}

/// Hash a batch of up to `batch_size()` candidates. Outputs 32 bytes per input.
#[inline]
fn hash_batch(
    blocks: &[[u8; BLOCK_LEN]],
    n: usize,
    out: &mut [[u8; OUT_LEN]],
) {
    debug_assert!(n <= blocks.len() && n <= out.len());
    if n == 0 {
        return;
    }
    let ptrs: [*const u8; BATCH_AVX512] = {
        let mut arr = [std::ptr::null::<u8>(); BATCH_AVX512];
        for i in 0..n {
            arr[i] = blocks[i].as_ptr();
        }
        arr
    };
    unsafe {
        match detect_simd() {
            #[cfg(target_arch = "x86_64")]
            SimdLevel::Avx512 => blake3_hash_many_avx512(
                ptrs.as_ptr(),
                n,
                1, // blocks = 64 / 64
                IV.as_ptr(),
                0,
                false,
                0,
                CHUNK_START | CHUNK_END,
                0,
                out.as_mut_ptr() as *mut u8,
            ),
            #[cfg(target_arch = "x86_64")]
            SimdLevel::Avx2 => blake3_hash_many_avx2(
                ptrs.as_ptr(),
                n,
                1,
                IV.as_ptr(),
                0,
                false,
                0,
                CHUNK_START | CHUNK_END,
                0,
                out.as_mut_ptr() as *mut u8,
            ),
            #[cfg(target_arch = "aarch64")]
            SimdLevel::Neon => blake3_hash_many_neon(
                ptrs.as_ptr(),
                n,
                1,
                IV.as_ptr(),
                0,
                false,
                0,
                CHUNK_START | CHUNK_END,
                0,
                out.as_mut_ptr() as *mut u8,
            ),
            #[cfg(target_arch = "aarch64")]
            SimdLevel::Portable => hash_batch_portable(&ptrs, n, out),
        }
    }
}

/// Portable single-block leaf hasher (aarch64 fallback for `hash_batch`).
///
/// Each candidate is one full 64-byte block hashed with CHUNK_START|CHUNK_END,
/// counter 0 — byte-identical to the NEON / AVX paths. Uses blake3's portable
/// compression exported by the crate's NEON C build.
#[cfg(target_arch = "aarch64")]
unsafe fn hash_batch_portable(ptrs: &[*const u8], n: usize, out: &mut [[u8; OUT_LEN]]) {
    for i in 0..n {
        let mut cv: [u32; 8] = IV;
        blake3_compress_in_place_portable(
            cv.as_mut_ptr(),
            ptrs[i],
            BLOCK_LEN as u8,
            0,
            CHUNK_START | CHUNK_END,
        );
        for (w, word) in cv.iter().enumerate() {
            out[i][w * 4..w * 4 + 4].copy_from_slice(&word.to_le_bytes());
        }
    }
}

/// Scan a batch output for the first nonce whose PoW check passes.
/// Returns `Some((nonce, _))` for the first valid nonce, or `None`.
#[inline]
fn first_valid_in_batch(
    nonce_base: u64,
    n: usize,
    out: &[[u8; OUT_LEN]],
    mask: u32,
) -> Option<u64> {
    for i in 0..n {
        let word = u32::from_le_bytes(out[i][0..4].try_into().unwrap());
        if (word & mask) == 0 {
            return Some(nonce_base + i as u64);
        }
    }
    None
}

/// Verifier-side PoW check. Returns `true` iff `BLAKE3(seed || witness_le)` has
/// `bits` low-order zero bits in its first LE u32 word. `bits == 0` always accepts.
pub fn check_blake3(seed: [u8; 32], witness: u64, bits: u32) -> bool {
    if bits == 0 {
        return true;
    }
    let mask = pow_mask(bits);
    let mut blk = [[0u8; BLOCK_LEN]; BATCH_AVX512];
    let mut out = [[0u8; OUT_LEN]; BATCH_AVX512];
    blk[0][..32].copy_from_slice(&seed);
    blk[0][32..40].copy_from_slice(&witness.to_le_bytes());
    hash_batch(&blk, 1, &mut out);
    let word = u32::from_le_bytes(out[0][0..4].try_into().unwrap());
    (word & mask) == 0
}

/// Serial PoW search. Loops over nonces in SIMD-sized batches on the calling
/// thread. Returns the first valid witness.
pub fn grind_blake3_serial(seed: [u8; 32], bits: u32) -> u64 {
    if bits == 0 {
        return 0;
    }
    let mask = pow_mask(bits);
    let bs = batch_size();
    let mut blk = [[0u8; BLOCK_LEN]; BATCH_AVX512];
    let mut out = [[0u8; OUT_LEN]; BATCH_AVX512];

    let mut nonce: u64 = 0;
    loop {
        fill_batch(&seed, nonce, bs, &mut blk);
        hash_batch(&blk, bs, &mut out);
        if let Some(w) = first_valid_in_batch(nonce, bs, &out, mask) {
            return w;
        }
        nonce += bs as u64;
    }
}

/// Parallel PoW search via rayon. Each worker claims contiguous nonce ranges via
/// an atomic stride counter and exits early once any worker publishes a hit.
pub fn grind_blake3_par(seed: [u8; 32], bits: u32) -> u64 {
    if bits == 0 {
        return 0;
    }
    let mask = pow_mask(bits);
    let bs = batch_size() as u64;

    // Chunk the nonce space: each rayon task consumes `CHUNK_BATCHES` consecutive
    // batches before checking the global sentinel. Larger → less atomic traffic;
    // smaller → faster early exit. 64 batches (= 1024 hashes on AVX-512) is a
    // reasonable mid-point.
    const CHUNK_BATCHES: u64 = 64;
    let chunk_nonces = bs * CHUNK_BATCHES;

    let found = AtomicU64::new(u64::MAX);
    let next_chunk = AtomicU64::new(0);

    rayon::broadcast(|_ctx| {
        let mut blk = [[0u8; BLOCK_LEN]; BATCH_AVX512];
        let mut out = [[0u8; OUT_LEN]; BATCH_AVX512];
        loop {
            // Bail out if another worker already found a witness.
            if found.load(Ordering::Relaxed) != u64::MAX {
                return;
            }
            let chunk_idx = next_chunk.fetch_add(1, Ordering::Relaxed);
            let nonce_start = chunk_idx * chunk_nonces;

            for b in 0..CHUNK_BATCHES {
                let nonce = nonce_start + b * bs;
                fill_batch(&seed, nonce, bs as usize, &mut blk);
                hash_batch(&blk, bs as usize, &mut out);
                if let Some(w) = first_valid_in_batch(nonce, bs as usize, &out, mask) {
                    // Publish the smallest witness seen: only overwrite if our
                    // candidate is smaller (keeps result deterministic across runs
                    // with the same seed/bits when multiple workers race).
                    loop {
                        let cur = found.load(Ordering::Relaxed);
                        if cur <= w {
                            break;
                        }
                        if found
                            .compare_exchange(cur, w, Ordering::Relaxed, Ordering::Relaxed)
                            .is_ok()
                        {
                            break;
                        }
                    }
                    return;
                }
            }
        }
    });

    let w = found.load(Ordering::Relaxed);
    debug_assert_ne!(w, u64::MAX, "parallel grind returned sentinel");
    w
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pow_mask_boundary() {
        assert_eq!(pow_mask(1), 1);
        assert_eq!(pow_mask(8), 0xFF);
        assert_eq!(pow_mask(16), 0xFFFF);
        assert_eq!(pow_mask(31), 0x7FFF_FFFF);
        assert_eq!(pow_mask(32), u32::MAX);
    }

    #[test]
    fn check_zero_bits_accepts_anything() {
        let seed = [7u8; 32];
        assert!(check_blake3(seed, 0, 0));
        assert!(check_blake3(seed, 12345, 0));
    }

    #[test]
    fn serial_roundtrip_8_bits() {
        let seed = [0xAAu8; 32];
        let w = grind_blake3_serial(seed, 8);
        assert!(check_blake3(seed, w, 8), "serial witness failed check at 8 bits");
    }

    #[test]
    fn serial_roundtrip_12_bits() {
        let seed = [0x5Au8; 32];
        let w = grind_blake3_serial(seed, 12);
        assert!(check_blake3(seed, w, 12));
        // Verifier rejects the prior nonce (almost certainly invalid).
        if w > 0 {
            assert!(!check_blake3(seed, w - 1, 12));
        }
    }

    #[test]
    fn par_roundtrip_12_bits() {
        let seed = [0x33u8; 32];
        let w = grind_blake3_par(seed, 12);
        assert!(check_blake3(seed, w, 12));
    }

    /// Byte-identity of the parallel grind against the serial grind at the
    /// production 16-bit grinding count. Both entry points return the globally
    /// SMALLEST valid nonce: the serial search scans nonces in increasing order
    /// and returns the first hit, and the parallel search resolves worker races
    /// by keeping the minimum witness via a compare-exchange loop. So the two
    /// results are equal by construction, independent of thread count or the
    /// batch SIMD width. The PoW definition is byte-identical across every
    /// hashing backend, so this equality is a property to assert, not to hope
    /// for. aarch64-gated: it exercises the NEON batch hasher and pins the ARM
    /// port of the parallel grind.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn grind_par_matches_serial_witness() {
        for seed_byte in [0x00u8, 0x9E] {
            let seed = [seed_byte; 32];
            let serial = grind_blake3_serial(seed, 16);
            let parallel = grind_blake3_par(seed, 16);
            assert_eq!(
                serial, parallel,
                "par/serial witness mismatch at 16 bits (seed byte {seed_byte:#04x})"
            );
            assert!(
                check_blake3(seed, parallel, 16),
                "grind witness failed the 16-bit check (seed byte {seed_byte:#04x})"
            );
        }
    }

    #[test]
    fn check_rejects_wrong_witness() {
        let seed = [0x42u8; 32];
        let w = grind_blake3_serial(seed, 10);
        assert!(check_blake3(seed, w, 10));
        assert!(!check_blake3(seed, w.wrapping_add(1), 10) || !check_blake3(seed, w.wrapping_add(2), 10),
                "at least one neighbour should fail the 10-bit check");
    }

    #[test]
    fn different_seeds_produce_different_witnesses() {
        let s1 = [0x11u8; 32];
        let s2 = [0x22u8; 32];
        let w1 = grind_blake3_serial(s1, 8);
        let w2 = grind_blake3_serial(s2, 8);
        // Witnesses are unrelated; at 8 bits they very likely differ.
        assert_ne!(w1, w2, "distinct seeds produced identical witnesses (extremely unlikely at 8 bits)");
    }
}
