///! Custom Merkle tree for MamaBear DeepFold — avoids rs_merkle overhead.
///!
///! Key optimizations:
///! - Flat in-place tree layout (single Vec<[u8; 32]>)
///! - Batch BLAKE3 hashing via AVX-512 (16-way) / AVX-2 (8-way) FFI
///! - Zero-copy parent hashing: pointer array into contiguous tree buffer
///! - Bottom-up tree construction
///!
///! Hash modes (different from blake3::hash — uses BLAKE3 internal tree semantics):
///! - Leaves: CHUNK_START|CHUNK_END flags, padded to 64B blocks
///! - Parents: PARENT flag, 64B input (left_hash || right_hash)
///! - Root: PARENT|ROOT flags
///!
///! Tree layout: layer 0 = leaves, layer 1 = parents, ..., root at index 1.
///! Index 0 is unused. For n leaves, tree[n..2n] = leaf hashes, tree[n/2..n] = layer 1, etc.
///! Parent of tree[i] = tree[i/2], children of tree[i] = tree[2i], tree[2i+1].

use crate::blake3_batch;
use rayon::prelude::*;

pub const HASH_SIZE: usize = 32;
type Hash = [u8; HASH_SIZE];

/// Build internal nodes of the tree from leaf hashes using batch parent hashing.
/// tree[n..2n] must already contain leaf hashes.
/// Fills tree[1..n] with parent hashes (tree[1] = root).
fn build_parents(tree: &mut Vec<Hash>, n: usize) {
    // Build level by level, bottom-up.
    // Level with parents at indices [level_start..level_end):
    //   level_end = n (first parent level), n/2, n/4, ..., 2
    //   level_start = level_end / 2
    // Children of tree[i] are tree[2i] and tree[2i+1].

    let mut level_end = n; // one past the last parent index at this level
    while level_end > 2 {
        let level_start = level_end / 2;
        // Batch hash parents at indices [level_start, level_end)
        // Their children are at [2*level_start, 2*level_end) which are already computed
        blake3_batch::hash_parents_level(tree, level_start, level_end);
        level_end = level_start;
    }

    // Root node (index 1) with PARENT|ROOT flags
    tree[1] = blake3_batch::hash_root(&tree[2], &tree[3]);
}

/// Parallel variant of `build_parents`. Within each level the parent
/// computations are independent, so we split the parent range into chunks
/// and dispatch `blake3_batch::hash_parents_level` across rayon workers.
/// Levels shrink 2x per iteration, so small levels fall through to serial.
fn build_parents_par(tree: &mut Vec<Hash>, n: usize) {
    // Threshold below which rayon scheduling overhead outweighs the batch work.
    const PARALLEL_THRESHOLD: usize = 4096;
    // Per-worker chunk inside a level. Matches the 16-way blake3 batch width
    // comfortably and keeps each chunk's working set in L2.
    const CHUNK: usize = 1024;

    let tree_ptr = TreePtr(tree.as_mut_ptr());
    let tree_len = tree.len();

    let mut level_end = n;
    while level_end > 2 {
        let level_start = level_end / 2;
        let num_parents = level_end - level_start;

        if num_parents >= PARALLEL_THRESHOLD {
            // Partition [level_start..level_end) into CHUNK-sized sub-ranges.
            // Each sub-range writes to disjoint tree[i] slots and reads from
            // disjoint tree[2i..2i+2] slots in the already-finished child
            // level, so the parallel dispatch is race-free.
            let num_chunks = (num_parents + CHUNK - 1) / CHUNK;
            (0..num_chunks).into_par_iter().for_each(move |ci| {
                // Rebind the `TreePtr` wrapper inside the closure so 2021
                // disjoint-captures doesn't lift out the raw pointer field
                // (which is not `Sync`).
                let tp = tree_ptr;
                let sub_start = level_start + ci * CHUNK;
                let sub_end = (sub_start + CHUNK).min(level_end);
                // Safety: each worker takes a `&mut [Hash]` view of the full
                // tree but only writes tree[sub_start..sub_end] and only reads
                // tree[2*sub_start..2*sub_end]. Disjoint sub-ranges have
                // disjoint parent writes and disjoint child reads.
                unsafe {
                    let view = std::slice::from_raw_parts_mut(tp.0, tree_len);
                    blake3_batch::hash_parents_level(view, sub_start, sub_end);
                }
            });
        } else {
            blake3_batch::hash_parents_level(tree, level_start, level_end);
        }

        level_end = level_start;
    }

    // Root node (index 1) with PARENT|ROOT flags
    tree[1] = blake3_batch::hash_root(&tree[2], &tree[3]);
}

/// Send/Sync wrapper around a raw `*mut Hash` so rayon workers can
/// re-derive disjoint `&mut [Hash]` views into a shared tree buffer.
#[derive(Clone, Copy)]
struct TreePtr(*mut Hash);
unsafe impl Send for TreePtr {}
unsafe impl Sync for TreePtr {}

/// Custom Merkle tree prover with flat layout.
#[derive(Clone)]
pub struct MerkleTreeProverMB {
    /// tree[1..2n]: tree[n..2n] = leaf hashes, tree[1] = root.
    /// tree[0] is unused (sentinel).
    tree: Vec<Hash>,
    leave_num: usize,
}

impl MerkleTreeProverMB {
    /// Build Merkle tree from raw leaf byte slices.
    /// Each element of leaf_values is the byte representation of one leaf.
    ///
    /// Uses batch BLAKE3 hashing:
    /// - Leaves: 16-way AVX-512 batch with CHUNK_START|CHUNK_END flags
    /// - Parents: 16-way AVX-512 batch with PARENT flag, level by level
    /// - Root: PARENT|ROOT flags
    pub fn new(leaf_values: Vec<Vec<u8>>) -> Self {
        let n = leaf_values.len();
        assert!(n.is_power_of_two(), "leaf count must be power of 2");

        let mut tree = vec![[0u8; HASH_SIZE]; 2 * n];

        // Batch hash all leaves: tree[n..2n]
        blake3_batch::hash_leaves_batch(&leaf_values, &mut tree[n..2 * n]);

        // Build internal nodes bottom-up using batch parent hashing
        build_parents(&mut tree, n);

        MerkleTreeProverMB {
            tree,
            leave_num: n,
        }
    }

    /// Build Merkle tree from a flat contiguous leaf buffer.
    ///
    /// Leaf `i` occupies `data[i*leaf_len .. (i+1)*leaf_len]`.
    /// Avoids the `Vec<Vec<u8>>` indirection and its 2M+ heap allocations.
    pub fn from_flat_leaves(data: &[u8], leaf_count: usize, leaf_len: usize) -> Self {
        assert!(leaf_count.is_power_of_two(), "leaf count must be power of 2");
        assert_eq!(data.len(), leaf_count * leaf_len);

        let mut tree = vec![[0u8; HASH_SIZE]; 2 * leaf_count];

        blake3_batch::hash_leaves_batch_flat(data, leaf_count, leaf_len, &mut tree[leaf_count..2 * leaf_count]);

        build_parents(&mut tree, leaf_count);

        MerkleTreeProverMB {
            tree,
            leave_num: leaf_count,
        }
    }

    /// Build from pre-hashed leaf data (avoids double-hashing when caller already has hashes).
    pub fn from_leaf_hashes(leaf_hashes: &[Hash]) -> Self {
        let n = leaf_hashes.len();
        assert!(n.is_power_of_two());

        let mut tree = vec![[0u8; HASH_SIZE]; 2 * n];
        tree[n..2 * n].copy_from_slice(leaf_hashes);

        // Build internal nodes bottom-up using batch parent hashing
        build_parents(&mut tree, n);

        MerkleTreeProverMB {
            tree,
            leave_num: n,
        }
    }

    /// Parallel variant of `from_leaf_hashes`. Uses rayon to parallelize
    /// the per-level parent hashing via `build_parents_par`. Produces a
    /// byte-identical tree to the serial constructor.
    pub fn from_leaf_hashes_par(leaf_hashes: &[Hash]) -> Self {
        let n = leaf_hashes.len();
        assert!(n.is_power_of_two());

        let mut tree = vec![[0u8; HASH_SIZE]; 2 * n];
        tree[n..2 * n].copy_from_slice(leaf_hashes);

        build_parents_par(&mut tree, n);

        MerkleTreeProverMB {
            tree,
            leave_num: n,
        }
    }

    pub fn leave_num(&self) -> usize {
        self.leave_num
    }

    pub fn commit(&self) -> Hash {
        self.tree[1]
    }

    /// Generate Merkle proof for given leaf indices.
    /// Returns the sibling hashes needed for verification, concatenated.
    pub fn open(&self, leaf_indices: &[usize]) -> Vec<u8> {
        let n = self.leave_num;
        let log_n = n.ilog2() as usize;

        // Collect all node indices we need in the proof
        let mut current_layer: Vec<usize> = leaf_indices.iter().map(|&i| n + i).collect();
        let mut proof = Vec::new();

        for _ in 0..log_n {
            // For each node, its sibling is needed unless the sibling is also in our set
            let mut siblings: Vec<usize> = current_layer.iter().map(|&idx| idx ^ 1).collect();
            siblings.sort();
            siblings.dedup();

            // Remove siblings that are already in current_layer
            let current_set: std::collections::HashSet<usize> =
                current_layer.iter().copied().collect();
            for &sib in &siblings {
                if !current_set.contains(&sib) {
                    proof.extend_from_slice(&self.tree[sib]);
                }
            }

            // Move to parent layer
            current_layer = current_layer.iter().map(|&idx| idx >> 1).collect();
            current_layer.sort();
            current_layer.dedup();
        }

        proof
    }
}

/// Custom Merkle tree verifier.
#[derive(Debug, Clone)]
pub struct MerkleTreeVerifierMB {
    pub merkle_root: Hash,
    pub leave_number: usize,
}

impl MerkleTreeVerifierMB {
    pub fn new(leave_number: usize, merkle_root: Hash) -> Self {
        Self {
            leave_number,
            merkle_root,
        }
    }

    /// Compute the expected proof length in bytes for given indices.
    pub fn proof_length(&self, indices: &[usize]) -> usize {
        let n = self.leave_number;
        let log_n = n.ilog2() as usize;
        let mut current_layer: Vec<usize> = indices.iter().map(|&i| n + i).collect();
        let mut count = 0;

        for _ in 0..log_n {
            let siblings: Vec<usize> = current_layer.iter().map(|&idx| idx ^ 1).collect();
            let current_set: std::collections::HashSet<usize> =
                current_layer.iter().copied().collect();
            for &sib in &siblings {
                if !current_set.contains(&sib) {
                    count += 1;
                }
            }
            current_layer = current_layer.iter().map(|&idx| idx >> 1).collect();
            current_layer.sort();
            current_layer.dedup();
        }

        count * HASH_SIZE
    }

    /// Verify a Merkle proof against leaf hashes and indices.
    ///
    /// The layer state is kept as a sorted `Vec<(index, Hash)>` (index
    /// ascending, deduplicated) rather than a `HashMap<index, Hash>`. All
    /// intra-layer lookups are binary searches, which avoids hashing overhead
    /// and allocator churn on the ~100-sized layers that PROV128 verification
    /// produces.
    ///
    /// Parent hashing at each layer uses the SIMD-batched
    /// `blake3_batch::hash_parents_batch` (16-way AVX-512 / 8-way AVX-2)
    /// instead of a per-pair `hash_parent_single`. The only exception is the
    /// very last hash (tree root), which needs the `PARENT|ROOT` flag and so
    /// goes through `blake3_batch::hash_root`.
    pub fn verify(
        &self,
        proof_bytes: &[u8],
        indices: &[usize],
        leaf_data: &[Vec<u8>],
    ) -> bool {
        let n = self.leave_number;
        let log_n = n.ilog2() as usize;

        // Initial layer: leaf hashes at indices `n + i`, sorted + deduped.
        let mut current: Vec<(usize, Hash)> = indices
            .iter()
            .enumerate()
            .map(|(i, &idx)| (n + idx, blake3_batch::hash_leaf_single(&leaf_data[i])))
            .collect();
        current.sort_by_key(|&(k, _)| k);
        current.dedup_by_key(|p| p.0);

        let mut proof_offset = 0;

        for layer in 0..log_n {
            // Figure out which siblings at this layer need to come from the
            // proof. A sibling is missing iff its index (= node_idx ^ 1) is
            // not already present in `current`.
            let mut missing_sibs: Vec<usize> = Vec::with_capacity(current.len());
            for &(idx, _) in &current {
                let sib = idx ^ 1;
                if current.binary_search_by_key(&sib, |p| p.0).is_err() {
                    missing_sibs.push(sib);
                }
            }
            missing_sibs.sort();
            missing_sibs.dedup();

            // Consume proof bytes for each missing sibling in ascending index
            // order. This must match the prover's serialization order in
            // `open()` exactly, which also emits proof entries in ascending
            // layer/index order.
            if proof_offset + missing_sibs.len() * HASH_SIZE > proof_bytes.len() {
                return false;
            }
            let mut sib_hashes: Vec<(usize, Hash)> = Vec::with_capacity(missing_sibs.len());
            for (i, &sib_idx) in missing_sibs.iter().enumerate() {
                let mut h = [0u8; HASH_SIZE];
                h.copy_from_slice(
                    &proof_bytes[proof_offset + i * HASH_SIZE..proof_offset + (i + 1) * HASH_SIZE],
                );
                sib_hashes.push((sib_idx, h));
            }
            proof_offset += missing_sibs.len() * HASH_SIZE;

            // Merge `current` (sorted) and `sib_hashes` (sorted) into
            // `full_layer` (sorted). Because a node and its sibling always
            // form adjacent even/odd pairs, the merged list has consecutive
            // (2k, 2k+1) pairs — exactly what the batched parent hash wants.
            let mut full_layer: Vec<(usize, Hash)> =
                Vec::with_capacity(current.len() + sib_hashes.len());
            let (mut ci, mut si) = (0usize, 0usize);
            while ci < current.len() && si < sib_hashes.len() {
                if current[ci].0 <= sib_hashes[si].0 {
                    full_layer.push(current[ci]);
                    ci += 1;
                } else {
                    full_layer.push(sib_hashes[si]);
                    si += 1;
                }
            }
            while ci < current.len() {
                full_layer.push(current[ci]);
                ci += 1;
            }
            while si < sib_hashes.len() {
                full_layer.push(sib_hashes[si]);
                si += 1;
            }
            if full_layer.len() % 2 != 0 {
                return false; // malformed proof — cannot pair up
            }

            let num_parents = full_layer.len() / 2;
            let is_top = layer + 1 == log_n;

            let mut next_layer: Vec<(usize, Hash)> = Vec::with_capacity(num_parents);

            if is_top {
                // Root needs PARENT|ROOT flags; `hash_parents_batch` only
                // produces PARENT-flagged outputs. Top layer has exactly one
                // parent (the root), so do it sequentially.
                for k in 0..num_parents {
                    let (li, lh) = full_layer[2 * k];
                    let (ri, rh) = full_layer[2 * k + 1];
                    if ri != li + 1 {
                        return false;
                    }
                    let parent_idx = li >> 1;
                    let parent_hash = blake3_batch::hash_root(&lh, &rh);
                    next_layer.push((parent_idx, parent_hash));
                }
            } else {
                // Non-root: pack (left, right) into contiguous 64-byte pair
                // buffers, batch-hash all pairs in one SIMD call.
                const BLOCK_LEN: usize = 64;
                let mut pair_bufs: Vec<[u8; BLOCK_LEN]> =
                    vec![[0u8; BLOCK_LEN]; num_parents];
                for k in 0..num_parents {
                    let (li, lh) = full_layer[2 * k];
                    let (ri, _rh) = full_layer[2 * k + 1];
                    if ri != li + 1 {
                        return false;
                    }
                    pair_bufs[k][..HASH_SIZE].copy_from_slice(&lh);
                    pair_bufs[k][HASH_SIZE..].copy_from_slice(&full_layer[2 * k + 1].1);
                }
                let mut out_hashes: Vec<Hash> = vec![[0u8; HASH_SIZE]; num_parents];
                blake3_batch::hash_parents_batch(&pair_bufs, &mut out_hashes);
                for k in 0..num_parents {
                    let parent_idx = full_layer[2 * k].0 >> 1;
                    next_layer.push((parent_idx, out_hashes[k]));
                }
            }

            // `full_layer` was sorted, so parent indices are non-decreasing.
            // They may coincide when two different children produce the same
            // parent only in malformed inputs; dedup_by_key is a safeguard.
            next_layer.dedup_by_key(|p| p.0);
            current = next_layer;
        }

        current.len() == 1 && current[0].0 == 1 && current[0].1 == self.merkle_root
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_merkle_tree_commit_verify() {
        let leaves: Vec<Vec<u8>> = (0..8u64).map(|i| i.to_le_bytes().to_vec()).collect();

        let prover = MerkleTreeProverMB::new(leaves.clone());
        let root = prover.commit();

        let indices = vec![2, 3, 5];
        let proof = prover.open(&indices);

        let verifier = MerkleTreeVerifierMB::new(8, root);
        let queried_leaves: Vec<Vec<u8>> = indices.iter().map(|&i| leaves[i].clone()).collect();
        assert!(verifier.verify(&proof, &indices, &queried_leaves));

        // Verify proof length matches
        assert_eq!(proof.len(), verifier.proof_length(&indices));
    }

    #[test]
    fn test_merkle_tree_single_leaf() {
        let leaves: Vec<Vec<u8>> = (0..16u64).map(|i| i.to_le_bytes().to_vec()).collect();

        let prover = MerkleTreeProverMB::new(leaves.clone());
        let root = prover.commit();

        for i in 0..16 {
            let proof = prover.open(&[i]);
            let verifier = MerkleTreeVerifierMB::new(16, root);
            assert!(verifier.verify(&proof, &[i], &[leaves[i].clone()]));
        }
    }

    #[test]
    fn test_from_leaf_hashes_matches_new() {
        let leaves: Vec<Vec<u8>> = (0..32u64).map(|i| i.to_le_bytes().to_vec()).collect();

        // Build tree via new()
        let prover1 = MerkleTreeProverMB::new(leaves.clone());

        // Build tree via from_leaf_hashes() with pre-hashed leaves
        let leaf_hashes: Vec<Hash> = leaves
            .iter()
            .map(|l| blake3_batch::hash_leaf_single(l))
            .collect();
        let prover2 = MerkleTreeProverMB::from_leaf_hashes(&leaf_hashes);

        assert_eq!(prover1.commit(), prover2.commit());
    }

    #[test]
    fn test_from_leaf_hashes_par_matches_serial() {
        // Exercises both below-threshold (serial fallback) and above-threshold
        // (parallel dispatch) levels across multiple power-of-two sizes.
        for log_n in [10usize, 14, 16, 18] {
            let n = 1usize << log_n;
            let leaf_hashes: Vec<Hash> = (0..n as u64)
                .map(|i| {
                    let mut h = [0u8; HASH_SIZE];
                    h[..8].copy_from_slice(&i.to_le_bytes());
                    h[8..16].copy_from_slice(&i.wrapping_mul(0x9E3779B97F4A7C15).to_le_bytes());
                    h
                })
                .collect();

            let serial = MerkleTreeProverMB::from_leaf_hashes(&leaf_hashes);
            let parallel = MerkleTreeProverMB::from_leaf_hashes_par(&leaf_hashes);

            assert_eq!(serial.leave_num(), parallel.leave_num());
            assert_eq!(serial.commit(), parallel.commit(), "root mismatch at log_n={}", log_n);
            assert_eq!(serial.tree, parallel.tree, "tree buffer mismatch at log_n={}", log_n);
        }
    }

    #[test]
    fn test_merkle_tree_large() {
        // Test with 1024 leaves to exercise full AVX-512 batching (16-way)
        let leaves: Vec<Vec<u8>> = (0..1024u64).map(|i| i.to_le_bytes().to_vec()).collect();

        let prover = MerkleTreeProverMB::new(leaves.clone());
        let root = prover.commit();

        // Verify a few random indices
        let indices = vec![0, 7, 100, 511, 1023];
        let proof = prover.open(&indices);

        let verifier = MerkleTreeVerifierMB::new(1024, root);
        let queried_leaves: Vec<Vec<u8>> = indices.iter().map(|&i| leaves[i].clone()).collect();
        assert!(verifier.verify(&proof, &indices, &queried_leaves));
        assert_eq!(proof.len(), verifier.proof_length(&indices));
    }
}
