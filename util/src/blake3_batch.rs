//! Batch BLAKE3 hashing via AVX-512 (16-way) / AVX-2 (8-way) FFI on x86_64,
//! and NEON (4-way) FFI with a portable fallback on aarch64.
//!
//! Calls the already-linked `blake3_hash_many_avx512` / `blake3_hash_many_avx2`
//! (x86_64) or `blake3_hash_many_neon` (aarch64) C FFI symbols from the blake3
//! crate. No fork needed — blake3's build.rs compiles the C/assembly code, and
//! the symbols have external linkage. On little-endian aarch64 the blake3 build
//! compiles `blake3_neon.c` by default (no cargo feature required).
//!
//! BLAKE3 output bytes are machine-independent: every arch arm applies the same
//! tree-node flag semantics (leaf = CHUNK_START|CHUNK_END, parent = PARENT,
//! root = PARENT|ROOT, counter = 0, no counter increment), so all arms produce
//! byte-identical hashes. Only throughput differs across arms.
//!
//! # Hash modes
//! - **Leaf hashing**: `flags=0, flags_start=CHUNK_START|CHUNK_END, flags_end=0`
//!   All leaf data padded to a multiple of BLOCK_LEN (64 bytes).
//! - **Parent hashing**: `flags=PARENT, flags_start=0, flags_end=0`
//!   Input = left_hash(32B) || right_hash(32B) = 64 bytes, blocks=1.
//! - **Root**: Single parent with `PARENT|ROOT` flags.

use std::sync::OnceLock;

// ── BLAKE3 spec constants ──────────────────────────────────────────────────
const IV: [u32; 8] = [
    0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
    0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19,
];
const CHUNK_START: u8 = 1 << 0; // 0x01
const CHUNK_END: u8 = 1 << 1;   // 0x02
const PARENT: u8 = 1 << 2;      // 0x04
const ROOT: u8 = 1 << 3;        // 0x08
pub const OUT_LEN: usize = 32;
const BLOCK_LEN: usize = 64;

// ── FFI declarations (x86_64) ──────────────────────────────────────────────
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

// ── FFI declarations (aarch64) ─────────────────────────────────────────────
#[cfg(target_arch = "aarch64")]
unsafe extern "C" {
    // 4-way NEON batch hasher from the blake3 crate's C build (`blake3_neon.c`,
    // compiled by default for little-endian aarch64). Same ABI as the x86
    // externs above; it takes `num_inputs` and cascades internally (4-way then
    // 1-way), so the 16-input batch calling shape is unchanged.
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
    // Portable single-block compression, exported (`#[no_mangle]`) by the blake3
    // crate whenever its NEON C intrinsics are built — i.e. on aarch64. Used by
    // the portable fallback arm so it stays byte-identical to every SIMD path.
    fn blake3_compress_in_place_portable(
        cv: *mut u32,
        block: *const u8,
        block_len: u8,
        counter: u64,
        flags: u8,
    );
}

// ── SIMD level detection ───────────────────────────────────────────────────
// Variants are arch-gated: x86_64 selects AVX-512/AVX-2, aarch64 selects NEON
// with a portable fallback. This file is inherently arch-specific FFI, so it
// compiles only for x86_64 and aarch64 (the two supported targets).
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
            // NEON (AdvSIMD) is architecturally mandatory on ARMv8-A, but probe
            // at runtime anyway so we degrade to the byte-identical portable
            // fallback on any environment that under-reports the feature.
            if std::arch::is_aarch64_feature_detected!("neon") {
                return SimdLevel::Neon;
            }
            SimdLevel::Portable
        }
    })
}

/// Portable single-input BLAKE3 tree-node hasher (aarch64 fallback).
///
/// Correctness-equal to (but slower than) the SIMD arms: it walks each input
/// one 64-byte block at a time through the blake3 crate's portable compression,
/// applying CHUNK_START on the first block and CHUNK_END on the last, with the
/// caller's `flags` (PARENT / PARENT|ROOT for node hashing) on every block and
/// counter fixed at 0 (all call sites pass `increment_counter == false`). This
/// mirrors the blake3 C `hash_one` path exactly, so its output bytes match the
/// NEON / AVX arms bit-for-bit.
#[cfg(target_arch = "aarch64")]
unsafe fn hash_many_portable(
    ptrs: &[*const u8],
    blocks: usize,
    flags: u8,
    flags_start: u8,
    flags_end: u8,
    out: *mut u8,
) {
    for (i, &input) in ptrs.iter().enumerate() {
        let mut cv: [u32; 8] = IV; // chunk/parent state starts at the IV key
        let mut block_flags = flags | flags_start;
        for b in 0..blocks {
            if b + 1 == blocks {
                block_flags |= flags_end;
            }
            // Every block here is a full 64-byte block (leaves are padded to a
            // multiple of BLOCK_LEN; parents are exactly one block).
            blake3_compress_in_place_portable(
                cv.as_mut_ptr(),
                input.add(b * BLOCK_LEN),
                BLOCK_LEN as u8,
                0,
                block_flags,
            );
            block_flags = flags; // CHUNK_START only applies to the first block
        }
        // Serialize the 8 CV words little-endian into 32 output bytes.
        let out_i = out.add(i * OUT_LEN);
        for (w, word) in cv.iter().enumerate() {
            let bytes = word.to_le_bytes();
            core::ptr::copy_nonoverlapping(bytes.as_ptr(), out_i.add(w * 4), 4);
        }
    }
}

// ── Core dispatch: call hash_many with appropriate SIMD level ──────────────
/// Raw batch hash dispatch. All inputs must be exactly `blocks * 64` bytes.
/// `ptrs` is a slice of pointers to input data.
/// `out` must be at least `ptrs.len() * 32` bytes.
unsafe fn hash_many_raw(
    ptrs: &[*const u8],
    blocks: usize,
    flags: u8,
    flags_start: u8,
    flags_end: u8,
    out: *mut u8,
) {
    let n = ptrs.len();
    if n == 0 {
        return;
    }
    match detect_simd() {
        #[cfg(target_arch = "x86_64")]
        SimdLevel::Avx512 => {
            // AVX-512 hash_many internally cascades: 16→8→4→1
            blake3_hash_many_avx512(
                ptrs.as_ptr(),
                n,
                blocks,
                IV.as_ptr(),
                0,     // counter
                false, // no increment
                flags,
                flags_start,
                flags_end,
                out,
            );
        }
        #[cfg(target_arch = "x86_64")]
        SimdLevel::Avx2 => {
            // AVX-2 hash_many cascades: 8→(sse41 fallback)
            blake3_hash_many_avx2(
                ptrs.as_ptr(),
                n,
                blocks,
                IV.as_ptr(),
                0,
                false,
                flags,
                flags_start,
                flags_end,
                out,
            );
        }
        #[cfg(target_arch = "aarch64")]
        SimdLevel::Neon => {
            // NEON hash_many cascades: 4→1 (see blake3_hash_many_neon).
            blake3_hash_many_neon(
                ptrs.as_ptr(),
                n,
                blocks,
                IV.as_ptr(),
                0,
                false,
                flags,
                flags_start,
                flags_end,
                out,
            );
        }
        #[cfg(target_arch = "aarch64")]
        SimdLevel::Portable => {
            hash_many_portable(ptrs, blocks, flags, flags_start, flags_end, out);
        }
    }
}

// ── Public API ─────────────────────────────────────────────────────────────

/// Batch hash leaf data. All leaves must have the same length.
/// Pads each leaf to a multiple of 64 bytes internally.
/// Uses CHUNK_START|CHUNK_END flags (message-mode chaining values).
///
/// # Arguments
/// * `leaves` - Slice of leaf byte slices, all the same length
/// * `out` - Output buffer, must have length >= leaves.len()
pub fn hash_leaves_batch(leaves: &[Vec<u8>], out: &mut [[u8; OUT_LEN]]) {
    let n = leaves.len();
    assert!(out.len() >= n);
    if n == 0 {
        return;
    }

    let leaf_len = leaves[0].len();
    debug_assert!(leaves.iter().all(|l| l.len() == leaf_len));

    // Pad to multiple of BLOCK_LEN
    let padded_len = (leaf_len + BLOCK_LEN - 1) / BLOCK_LEN * BLOCK_LEN;
    let blocks = padded_len / BLOCK_LEN;

    if padded_len == leaf_len {
        // No padding needed — point directly into leaf data
        let ptrs: Vec<*const u8> = leaves.iter().map(|l| l.as_ptr()).collect();
        unsafe {
            if blocks == 1 {
                // Single-block: CHUNK_START|CHUNK_END on the one block
                hash_many_raw(
                    &ptrs,
                    1,
                    0,
                    CHUNK_START | CHUNK_END,
                    0,
                    out.as_mut_ptr() as *mut u8,
                );
            } else {
                // Multi-block: CHUNK_START on first, CHUNK_END on last
                hash_many_raw(
                    &ptrs,
                    blocks,
                    0,
                    CHUNK_START,
                    CHUNK_END,
                    out.as_mut_ptr() as *mut u8,
                );
            }
        }
    } else {
        // Need padding — allocate padded buffer
        let mut padded = vec![0u8; n * padded_len];
        for (i, leaf) in leaves.iter().enumerate() {
            padded[i * padded_len..i * padded_len + leaf_len].copy_from_slice(leaf);
            // Rest is already zero-filled
        }
        let ptrs: Vec<*const u8> = (0..n).map(|i| padded[i * padded_len..].as_ptr()).collect();

        // For multi-block: flags_start on first block, flags_end on last block
        // For single-block (blocks==1): effective = flags | flags_start | flags_end
        unsafe {
            hash_many_raw(
                &ptrs,
                blocks,
                0,           // flags (applied to all blocks)
                CHUNK_START, // flags_start (first block)
                CHUNK_END,   // flags_end (last block)
                out.as_mut_ptr() as *mut u8,
            );
        }
    }
}

/// Batch hash leaf data from a flat contiguous buffer.
///
/// Same semantics as `hash_leaves_batch`, but takes a single `&[u8]` buffer
/// where leaf `i` starts at offset `i * leaf_len`. Avoids the `Vec<Vec<u8>>`
/// indirection and its associated heap allocations.
pub fn hash_leaves_batch_flat(
    data: &[u8],
    leaf_count: usize,
    leaf_len: usize,
    out: &mut [[u8; OUT_LEN]],
) {
    assert!(out.len() >= leaf_count);
    assert_eq!(data.len(), leaf_count * leaf_len);
    if leaf_count == 0 {
        return;
    }

    let padded_len = (leaf_len + BLOCK_LEN - 1) / BLOCK_LEN * BLOCK_LEN;
    let blocks = padded_len / BLOCK_LEN;

    if padded_len == leaf_len {
        // No padding needed — stride directly into flat buffer
        let ptrs: Vec<*const u8> = (0..leaf_count)
            .map(|i| unsafe { data.as_ptr().add(i * leaf_len) })
            .collect();
        unsafe {
            if blocks == 1 {
                hash_many_raw(
                    &ptrs, 1, 0, CHUNK_START | CHUNK_END, 0,
                    out.as_mut_ptr() as *mut u8,
                );
            } else {
                hash_many_raw(
                    &ptrs, blocks, 0, CHUNK_START, CHUNK_END,
                    out.as_mut_ptr() as *mut u8,
                );
            }
        }
    } else {
        // Need padding — copy each leaf into padded buffer
        let mut padded = vec![0u8; leaf_count * padded_len];
        for i in 0..leaf_count {
            padded[i * padded_len..i * padded_len + leaf_len]
                .copy_from_slice(&data[i * leaf_len..i * leaf_len + leaf_len]);
        }
        let ptrs: Vec<*const u8> = (0..leaf_count)
            .map(|i| padded[i * padded_len..].as_ptr())
            .collect();
        unsafe {
            hash_many_raw(
                &ptrs, blocks, 0, CHUNK_START, CHUNK_END,
                out.as_mut_ptr() as *mut u8,
            );
        }
    }
}

/// Batch hash parent nodes. Zero-copy: reads directly from the tree buffer.
///
/// Each parent input is 64 bytes = tree[2i](32B) || tree[2i+1](32B).
/// Since `Vec<[u8; 32]>` is contiguous, `&tree[2i]` points to 64 contiguous bytes.
///
/// Uses PARENT flag, counter=0, no increment.
///
/// # Arguments
/// * `tree` - The flat tree buffer (tree[0] unused, tree[1]=root, tree[n..2n]=leaves)
/// * `parent_start` - First parent index (inclusive)
/// * `parent_end` - Last parent index (exclusive)
///
/// Computes tree[i] = parent_hash(tree[2i], tree[2i+1]) for i in parent_start..parent_end.
pub fn hash_parents_level(tree: &mut [[u8; OUT_LEN]], parent_start: usize, parent_end: usize) {
    let num_parents = parent_end - parent_start;
    if num_parents == 0 {
        return;
    }

    // Build pointer array: each points to tree[2*i] which is 64 contiguous bytes
    let ptrs: Vec<*const u8> = (parent_start..parent_end)
        .map(|i| tree[2 * i].as_ptr())
        .collect();

    // Temporary output buffer (can't write into tree while reading from it)
    let mut out_buf = vec![[0u8; OUT_LEN]; num_parents];

    unsafe {
        hash_many_raw(
            &ptrs,
            1, // blocks = 64/64 = 1
            PARENT,
            0, // flags_start (not used for parent mode)
            0, // flags_end
            out_buf.as_mut_ptr() as *mut u8,
        );
    }

    // Write results back to tree
    for (idx, i) in (parent_start..parent_end).enumerate() {
        tree[i] = out_buf[idx];
    }
}

/// Hash the root node: parent_hash(left, right) with PARENT|ROOT flags.
pub fn hash_root(left: &[u8; OUT_LEN], right: &[u8; OUT_LEN]) -> [u8; OUT_LEN] {
    let mut input = [0u8; BLOCK_LEN];
    input[..OUT_LEN].copy_from_slice(left);
    input[OUT_LEN..].copy_from_slice(right);

    let mut out = [0u8; OUT_LEN];
    let ptr: *const u8 = input.as_ptr();

    unsafe {
        hash_many_raw(
            &[ptr],
            1,
            PARENT | ROOT,
            0,
            0,
            out.as_mut_ptr(),
        );
    }
    out
}

/// Batch parent hash over arbitrary (left, right) pairs.
///
/// `pair_bufs` is a slice of n x 64-byte buffers, each formatted
/// `[left_hash_32B | right_hash_32B]`. Writes `n` x 32-byte outputs into
/// `out` in the same order. Uses the SIMD (AVX-512 16-way / AVX-2 8-way)
/// `hash_many_raw` internally, so scales roughly with input count until
/// ~1 lane per SIMD register is saturated.
///
/// Unlike `hash_parents_level`, pairs do not need to be laid out as a
/// contiguous tree slice — the verifier's Merkle-path reconstruction
/// collects (left, right) pairs in arbitrary node-index order.
///
/// Use for non-root parent nodes. For the root (index 1) call `hash_root`
/// separately or pass a length-1 slice through this with a different flag
/// set (not supported; use `hash_root`).
pub fn hash_parents_batch(pair_bufs: &[[u8; BLOCK_LEN]], out: &mut [[u8; OUT_LEN]]) {
    assert_eq!(
        pair_bufs.len(),
        out.len(),
        "hash_parents_batch: pair_bufs and out must have equal length"
    );
    let n = pair_bufs.len();
    if n == 0 {
        return;
    }
    let ptrs: Vec<*const u8> = pair_bufs.iter().map(|p| p.as_ptr()).collect();
    unsafe {
        hash_many_raw(
            &ptrs,
            1, // blocks = 64 / 64 = 1
            PARENT,
            0,
            0,
            out.as_mut_ptr() as *mut u8,
        );
    }
}

/// Hash a single parent node (for verifier's sequential path).
/// Same semantics as batch parent hashing: PARENT flag, counter=0.
pub fn hash_parent_single(left: &[u8; OUT_LEN], right: &[u8; OUT_LEN]) -> [u8; OUT_LEN] {
    let mut input = [0u8; BLOCK_LEN];
    input[..OUT_LEN].copy_from_slice(left);
    input[OUT_LEN..].copy_from_slice(right);

    let mut out = [0u8; OUT_LEN];
    let ptr: *const u8 = input.as_ptr();

    unsafe {
        hash_many_raw(
            &[ptr],
            1,
            PARENT,
            0,
            0,
            out.as_mut_ptr(),
        );
    }
    out
}

/// Hash a single leaf (for verifier's sequential path).
/// Matches batch leaf hashing semantics: pads to 64 bytes, CHUNK_START|CHUNK_END.
pub fn hash_leaf_single(data: &[u8]) -> [u8; OUT_LEN] {
    let padded_len = (data.len() + BLOCK_LEN - 1) / BLOCK_LEN * BLOCK_LEN;
    let padded_len = padded_len.max(BLOCK_LEN); // at least one block
    let blocks = padded_len / BLOCK_LEN;

    let mut padded = vec![0u8; padded_len];
    padded[..data.len()].copy_from_slice(data);

    let mut out = [0u8; OUT_LEN];
    let ptr: *const u8 = padded.as_ptr();

    unsafe {
        hash_many_raw(
            &[ptr],
            blocks,
            0,
            CHUNK_START,
            CHUNK_END,
            out.as_mut_ptr(),
        );
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hash_parent_matches_batch() {
        // Single parent hash should match batch of 1
        let left = [0xAAu8; 32];
        let right = [0xBBu8; 32];

        let single = hash_parent_single(&left, &right);

        let mut tree = vec![[0u8; 32]; 4];
        tree[2] = left;
        tree[3] = right;
        hash_parents_level(&mut tree, 1, 2);

        assert_eq!(single, tree[1]);
    }

    #[test]
    fn test_hash_root_differs_from_parent() {
        // Root hash uses PARENT|ROOT flags, parent uses just PARENT
        let left = [0xAAu8; 32];
        let right = [0xBBu8; 32];

        let root = hash_root(&left, &right);
        let parent = hash_parent_single(&left, &right);

        // Different flags → different outputs
        assert_ne!(root, parent);
    }

    #[test]
    fn test_hash_leaves_batch_consistency() {
        // Batch of N leaves should match N individual leaf hashes
        let leaves: Vec<Vec<u8>> = (0..16u64)
            .map(|i| i.to_le_bytes().to_vec())
            .collect();

        let mut batch_out = vec![[0u8; 32]; 16];
        hash_leaves_batch(&leaves, &mut batch_out);

        for (i, leaf) in leaves.iter().enumerate() {
            let single = hash_leaf_single(leaf);
            assert_eq!(batch_out[i], single, "mismatch at leaf {}", i);
        }
    }

    #[test]
    fn test_hash_leaves_batch_64byte_inputs() {
        // 64-byte inputs (no padding needed)
        let leaves: Vec<Vec<u8>> = (0..8u64)
            .map(|i| {
                let mut v = vec![0u8; 64];
                v[..8].copy_from_slice(&i.to_le_bytes());
                v
            })
            .collect();

        let mut batch_out = vec![[0u8; 32]; 8];
        hash_leaves_batch(&leaves, &mut batch_out);

        for (i, leaf) in leaves.iter().enumerate() {
            let single = hash_leaf_single(leaf);
            assert_eq!(batch_out[i], single, "mismatch at 64-byte leaf {}", i);
        }
    }

    #[test]
    fn test_batch_parent_16way() {
        // Test with exactly 16 parents (full AVX-512 width)
        let mut tree = vec![[0u8; 32]; 64]; // 32 leaves → 16 parents
        for (i, node) in tree[32..64].iter_mut().enumerate() {
            node[0] = i as u8;
        }

        // Batch parents
        hash_parents_level(&mut tree, 16, 32);

        // Verify each parent individually
        for i in 16..32 {
            let expected = hash_parent_single(&tree[2 * i], &tree[2 * i + 1]);
            // tree[i] was already written by hash_parents_level, but we need
            // to check against fresh computation. Since we already wrote tree[i],
            // and the children at tree[2i],tree[2i+1] are still the originals
            // (only indices 16..32 were modified, children are at 32..64),
            // this comparison is valid.
            assert_eq!(tree[i], expected, "parent {} mismatch", i);
        }
    }

    #[test]
    fn test_full_tree_build() {
        // Build a complete tree and verify root
        let n = 8usize;
        let mut tree = vec![[0u8; 32]; 2 * n];

        // Set leaf hashes
        for i in 0..n {
            tree[n + i][0] = i as u8;
        }

        // Build parents level by level
        let mut level_size = n; // number of nodes at current level
        while level_size > 2 {
            let parent_start = level_size / 2;
            let parent_end = level_size;
            hash_parents_level(&mut tree, parent_start, parent_end);
            level_size /= 2;
        }

        // Root
        tree[1] = hash_root(&tree[2], &tree[3]);

        // Verify: recompute the tree manually
        let mut expected_tree = tree.clone();
        for i in (1..n).rev() {
            if i == 1 {
                expected_tree[1] = hash_root(&expected_tree[2], &expected_tree[3]);
            } else {
                expected_tree[i] = hash_parent_single(&expected_tree[2 * i], &expected_tree[2 * i + 1]);
            }
        }
        assert_eq!(tree[1], expected_tree[1]);
    }
}
