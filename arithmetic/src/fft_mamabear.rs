use std::ptr;
///! MamaBear FFT module — Montgomery-correct Radix-2 Cooley-Tukey FFT.
///!
///! All operations work in Montgomery form (R = 2^52).
///! Twiddle factors are precomputed and stored in Montgomery form.
///! Input/output arrays contain Montgomery-form MamaBearScalar values.
///!
///! # Usage
///! ```ignore
///! let fft = MamaBearFFT::new(log_n);
///! let mut data: Vec<MamaBearScalar> = coeffs.iter().map(|x| x.to_montgomery()).collect();
///! fft.fft_in_place(&mut data);
///! // data now contains evaluations in Montgomery form
///! ```
use std::sync::Arc;

use crate::field::mamabear::{LazyReduction, MamaBearScalar, PackedMamaBearAVX512, P};
use crate::field::Field;

/// Montgomery form of 1: R mod P = 2^52 mod P.
const ONE_MONT: u64 = (1u64 << 52) % P;
const TWIDDLE_LAZY_REDUCE_BOUND: u64 = P + (P >> 1);

/// MamaBear FFT with precomputed twiddle factors in Montgomery form.
#[derive(Debug, Clone)]
pub struct MamaBearFFT {
    pub log_order: u32,
    /// Primitive 2^log_order-th root of unity in Montgomery form.
    #[allow(dead_code)]
    omega_mont: MamaBearScalar,
    /// Precomputed group elements: omega^0, omega^1, ..., omega^{2^log_order - 1}
    /// all in Montgomery form.
    elements_mont: Arc<Vec<MamaBearScalar>>,
    /// Packed twiddles for the final three pair-major tail layers.
    tail_twiddle_last3: PackedMamaBearAVX512,
    tail_twiddle_last2: PackedMamaBearAVX512,
    tail_twiddle_last1: PackedMamaBearAVX512,
    /// Precomputed packed chirp twiddle tables for zero-padded FFT with prefix_layers=3.
    /// Flat Vec of length 7 * (N/64), indexed as [beta_idx * (N/64) + chunk].
    /// Entry at [beta_idx * packed_count + c] holds
    ///   [omega^{(beta_idx+1)*(8c)}, ..., omega^{(beta_idx+1)*(8c+7)}]
    /// for beta_idx 0..6 (mapping to beta 1..7).
    /// Only populated via `precompute_chirp_prefix3()`.
    chirp_prefix3: Option<Vec<PackedMamaBearAVX512>>,
}

impl MamaBearFFT {
    /// Create a new FFT instance for domain size 2^log_order.
    /// Precomputes all twiddle factors in Montgomery form.
    pub fn new(log_order: u32) -> Self {
        assert!(
            log_order <= <MamaBearScalar as FftField>::LOG_ORDER,
            "log_order {} exceeds max 2-adic order {}",
            log_order,
            <MamaBearScalar as FftField>::LOG_ORDER
        );

        // Compute omega = ROOT_OF_UNITY^(2^(LOG_ORDER - log_order)) in raw form
        let raw_omega = <MamaBearScalar as FftField>::ROOT_OF_UNITY
            .exp(1usize << (<MamaBearScalar as FftField>::LOG_ORDER - log_order));

        // Convert to Montgomery form
        let omega_mont = raw_omega.to_montgomery();

        // Precompute elements: omega^0, omega^1, ..., omega^{n-1} in Montgomery form
        let n = 1usize << log_order;
        let mut elements = Vec::with_capacity(n);
        let mut current = MamaBearScalar(ONE_MONT); // 1 in Montgomery form
        for _ in 0..n {
            elements.push(current);
            // mont_mul(a_mont, b_mont) = (a*b)_mont — correct in Montgomery domain
            current = MamaBearScalar(MamaBearScalar::mont_mul(current.0, omega_mont.0));
        }

        debug_assert!(
            elements
                .iter()
                .all(|value| value.0 <= TWIDDLE_LAZY_REDUCE_BOUND),
            "FFT twiddles must stay within [0, 1.5P] for forward lazy-reduce invariants",
        );

        let w1 = elements[n >> 3];
        let w2 = elements[n >> 2];
        let w3 = elements[(n >> 3) * 3];

        let tail_twiddle_last3 = PackedMamaBearAVX512::from_array([
            ONE_MONT, w1.0, w2.0, w3.0, ONE_MONT, w1.0, w2.0, w3.0,
        ]);
        let tail_twiddle_last2 = PackedMamaBearAVX512::from_array([
            ONE_MONT, w2.0, ONE_MONT, w2.0, ONE_MONT, w2.0, ONE_MONT, w2.0,
        ]);
        let tail_twiddle_last1 = PackedMamaBearAVX512::broadcast(ONE_MONT);

        MamaBearFFT {
            log_order,
            omega_mont,
            elements_mont: Arc::new(elements),
            tail_twiddle_last3,
            tail_twiddle_last2,
            tail_twiddle_last1,
            chirp_prefix3: None,
        }
    }

    /// Domain size = 2^log_order.
    pub fn size(&self) -> usize {
        1 << self.log_order
    }

    // -----------------------------------------------------------------------
    // Read-only accessors for external FFT kernels (e.g. extension-field FFT).
    // These expose the precomputed twiddle / chirp tables and the zero-padding
    // dispatch decision without re-allocating or re-computing them. Pure
    // forwarding -- no behavioral change relative to the private fields.
    // -----------------------------------------------------------------------

    /// Precomputed twiddle table `omega^0 .. omega^{n-1}` in Montgomery form.
    #[inline]
    pub fn elements_mont(&self) -> &[MamaBearScalar] {
        &self.elements_mont
    }

    /// Packed twiddle for the third-from-last pair-major tail layer.
    #[inline]
    pub fn tail_twiddle_last3(&self) -> PackedMamaBearAVX512 {
        self.tail_twiddle_last3
    }

    /// Packed twiddle for the second-from-last pair-major tail layer.
    #[inline]
    pub fn tail_twiddle_last2(&self) -> PackedMamaBearAVX512 {
        self.tail_twiddle_last2
    }

    /// Packed twiddle for the last pair-major tail layer.
    #[inline]
    pub fn tail_twiddle_last1(&self) -> PackedMamaBearAVX512 {
        self.tail_twiddle_last1
    }

    /// Optional precomputed chirp table for zero-padded FFT with prefix_layers=3.
    /// `None` when the table exceeds the L2 budget; callers must fall back to
    /// on-the-fly chirp computation in that case (mirroring the base path).
    #[inline]
    pub fn chirp_prefix3(&self) -> Option<&[PackedMamaBearAVX512]> {
        self.chirp_prefix3.as_deref()
    }

    /// Zero-padding dispatch helper: returns `Some(prefix_layers)` if the input
    /// length triggers the zero-padded fast path, `None` otherwise. Mirrors the
    /// internal predicate used by `fft_into`.
    #[inline]
    pub fn zero_padding_prefix_layers_for(&self, raw_len: usize) -> Option<usize> {
        self.zero_padding_prefix_layers(raw_len)
    }

    /// Precompute packed chirp twiddle tables for zero-padded FFT with prefix_layers=3
    /// (code_rate_log=3, input is N/8 of the domain).
    ///
    /// For each beta in 1..=7, stores raw_len/8 packed vectors where packed vector
    /// at chunk index c holds [omega^{beta*(8c)}, omega^{beta*(8c+1)}, ..., omega^{beta*(8c+7)}].
    /// Max element index: 7*(raw_len-1) = 7*(N/8-1) < N — no wraparound.
    ///
    /// The table is only allocated when it fits within the L2 cache budget (4 MB).
    /// For larger domains the on-the-fly fallback is used, which has better cache
    /// locality by reusing the already-resident elements_mont table.
    pub fn precompute_chirp_prefix3(&mut self) {
        assert!(
            self.log_order >= 6,
            "chirp_prefix3 requires log_order >= 6, got {}",
            self.log_order
        );
        let n = self.size();
        let raw_len = n / 8; // N/8 for prefix_layers=3
        let packed_count = raw_len / 8; // N/64

        // 7 betas * packed_count vectors * 64 bytes each
        const MAX_CHIRP_TABLE_BYTES: usize = 4 * 1024 * 1024; // 4 MB
        let table_bytes = 7 * packed_count * 64;
        if table_bytes > MAX_CHIRP_TABLE_BYTES {
            return; // Too large for cache — fallback will be used
        }

        let mut table = Vec::with_capacity(7 * packed_count);
        for beta in 1usize..=7 {
            for chunk in 0..packed_count {
                let mut arr = [0u64; 8];
                for i in 0..8 {
                    arr[i] = self.elements_mont[beta * (chunk * 8 + i)].0;
                }
                table.push(PackedMamaBearAVX512::from_array(arr));
            }
        }
        self.chirp_prefix3 = Some(table);
    }

    /// Get omega^index in Montgomery form.
    pub fn element_at(&self, index: usize) -> MamaBearScalar {
        self.elements_mont[index]
    }

    /// Get omega^{-index} in Montgomery form.
    pub fn element_inv_at(&self, index: usize) -> MamaBearScalar {
        if index == 0 {
            MamaBearScalar(ONE_MONT)
        } else {
            self.elements_mont[self.size() - index]
        }
    }

    #[inline(always)]
    fn reverse_index(index: usize, log_len: usize) -> usize {
        if log_len == 0 {
            return 0;
        }
        index.reverse_bits() >> (usize::BITS as usize - log_len)
    }

    /// Pair slot order matches the historical bit-reversed pair order:
    /// pair `j` corresponds to `omega^{bit_reverse(j)}` on the half-sized suffix.
    #[inline(always)]
    pub fn bit_reversed_pair_element_inv_at(&self, pair_index: usize) -> MamaBearScalar {
        debug_assert!(pair_index < (self.size() >> 1));
        self.element_inv_at(Self::reverse_index(pair_index, self.log_order as usize - 1))
    }

    #[inline(always)]
    pub fn pair_slots_per_block(&self) -> usize {
        Self::pair_slots_per_block_for_pair_count(self.size() >> 1)
    }

    #[inline(always)]
    pub fn pair_slots_per_block_for_pair_count(pair_count: usize) -> usize {
        pair_count.min(8)
    }

    /// Storage layout for FFT output is pair-major in local blocks:
    /// `[x0..xk, nx0..nxk]`, where `k = pair_slots_per_block - 1`.
    #[inline(always)]
    pub fn pair_storage_positions(&self, pair_index: usize) -> (usize, usize) {
        Self::pair_storage_positions_for_pair_count(pair_index, self.size() >> 1)
    }

    #[inline(always)]
    pub fn pair_storage_positions_for_pair_count(
        pair_index: usize,
        pair_count: usize,
    ) -> (usize, usize) {
        let pairs_per_block = Self::pair_slots_per_block_for_pair_count(pair_count);
        let block = pair_index / pairs_per_block;
        let lane = pair_index % pairs_per_block;
        let x_pos = block * (pairs_per_block * 2) + lane;
        let nx_pos = x_pos + pairs_per_block;
        (x_pos, nx_pos)
    }

    /// Create the "squared" subgroup: omega^2 generates a group of half the size.
    pub fn exp(&self, index: usize) -> MamaBearFFT {
        assert_eq!(index & (index - 1), 0, "index must be a power of 2");
        MamaBearFFT::new(self.log_order - index.ilog2())
    }

    /// In-place bit-reversal permutation without an auxiliary `Vec<usize>`.
    ///
    /// This avoids re-allocating and scanning a large permutation table for every
    /// `fft_in_place_natural` call, which is critical in DeepFold where the same
    /// FFT instance is reused for many sub-FFTs.
    #[inline]
    fn bit_reverse_in_place(coeff: &mut [MamaBearScalar]) {
        let n = coeff.len();
        if n <= 2 {
            return;
        }

        let mut j = 0usize;
        for i in 1..(n - 1) {
            let mut bit = n >> 1;
            while j & bit != 0 {
                j ^= bit;
                bit >>= 1;
            }
            j ^= bit;
            if i < j {
                coeff.swap(i, j);
            }
        }
    }

    #[inline]
    fn raw_to_montgomery_in_place(coeff: &mut [MamaBearScalar]) {
        let packed_count = coeff.len() / 8;
        for chunk in 0..packed_count {
            let off = chunk * 8;
            let packed = Self::load_packed(&coeff[off..]);
            Self::store_packed(&mut coeff[off..], packed.to_montgomery());
        }

        for value in coeff[packed_count * 8..].iter_mut() {
            *value = value.to_montgomery();
        }
    }

    #[inline]
    fn scale_montgomery_slice_in_place(coeff: &mut [MamaBearScalar], scalar_mont: MamaBearScalar) {
        let scalar_packed = PackedMamaBearAVX512::from(scalar_mont.0);
        let packed_count = coeff.len() / 8;
        for chunk in 0..packed_count {
            let off = chunk * 8;
            let values = Self::load_packed(&coeff[off..]);
            let scaled = (values * scalar_packed).con_sub_xp(1);
            Self::store_packed(&mut coeff[off..], scaled);
        }

        for value in coeff[packed_count * 8..].iter_mut() {
            *value = MamaBearScalar(MamaBearScalar::mont_mul(value.0, scalar_mont.0)).con_sub_xp(1);
        }
    }

    #[inline(always)]
    fn zero_padding_prefix_layers(&self, raw_len: usize) -> Option<usize> {
        let n = self.size();
        if raw_len == 0 || raw_len >= n || !raw_len.is_power_of_two() {
            return None;
        }
        Some(self.log_order as usize - raw_len.ilog2() as usize)
    }

    #[inline]
    fn mul_montgomery_slice_by_geometric_progression(
        src: &[MamaBearScalar],
        dst: &mut [MamaBearScalar],
        ratio_mont: MamaBearScalar,
    ) {
        debug_assert_eq!(src.len(), dst.len());

        if ratio_mont.0 == ONE_MONT {
            dst.copy_from_slice(src);
            return;
        }

        let packed_count = src.len() / 8;
        let mut powers = [0u64; 8];
        powers[0] = ONE_MONT;
        for idx in 1..8 {
            powers[idx] = MamaBearScalar::mont_mul(powers[idx - 1], ratio_mont.0);
        }

        let step8 = MamaBearScalar(MamaBearScalar::mont_mul(powers[7], ratio_mont.0));
        let step8_packed = PackedMamaBearAVX512::from(step8);
        let mut packed_tw = PackedMamaBearAVX512::from_array(powers);

        for chunk in 0..packed_count {
            let off = chunk * 8;
            let values = Self::load_packed(&src[off..]);
            let scaled = (values * packed_tw).con_sub_xp(1);
            Self::store_packed(&mut dst[off..], scaled);
            packed_tw = (packed_tw * step8_packed).con_sub_xp(1);
        }

        let mut scalar_tw = MamaBearScalar(packed_tw.to_array()[0]);
        for idx in (packed_count * 8)..src.len() {
            dst[idx] =
                MamaBearScalar(MamaBearScalar::mont_mul(src[idx].0, scalar_tw.0)).con_sub_xp(1);
            scalar_tw =
                MamaBearScalar(MamaBearScalar::mont_mul(scalar_tw.0, ratio_mont.0)).con_sub_xp(1);
        }
    }

    #[inline]
    fn materialize_zero_padded_prefix_state(
        &self,
        raw_coeffs: &[MamaBearScalar],
        buf: &mut [MamaBearScalar],
        prefix_layers: usize,
    ) {
        let n = self.size();
        let block_len = raw_coeffs.len();
        let block_count = 1usize << prefix_layers;

        debug_assert!(prefix_layers > 0);
        debug_assert_eq!(block_len * block_count, n);

        let (first_block, rest) = buf[..n].split_at_mut(block_len);
        first_block.copy_from_slice(raw_coeffs);
        Self::raw_to_montgomery_in_place(first_block);

        for block_idx in 1..block_count {
            let beta = Self::reverse_index(block_idx, prefix_layers);
            let ratio = self.elements_mont[beta];
            let dst_start = (block_idx - 1) * block_len;
            let dst = &mut rest[dst_start..dst_start + block_len];
            Self::mul_montgomery_slice_by_geometric_progression(first_block, dst, ratio);
        }
    }

    #[inline]
    fn fft_into_zero_padded_prefix3_pair_major(
        &self,
        raw_coeffs: &[MamaBearScalar],
        buf: &mut [MamaBearScalar],
    ) {
        let n = self.size();
        let raw_len = raw_coeffs.len();

        debug_assert_eq!(raw_len << 3, n);

        let (block0, tail) = buf[..n].split_at_mut(raw_len);
        let (block1, tail) = tail.split_at_mut(raw_len);
        let (block2, tail) = tail.split_at_mut(raw_len);
        let (block3, tail) = tail.split_at_mut(raw_len);
        let (block4, tail) = tail.split_at_mut(raw_len);
        let (block5, tail) = tail.split_at_mut(raw_len);
        let (block6, block7) = tail.split_at_mut(raw_len);

        let packed_count = raw_len / 8;

        if let Some(ref chirp_table) = self.chirp_prefix3 {
            // Precomputed chirp tables: direct packed loads, all 7 muls independent.
            // beta_idx: beta 1->0, 2->1, 3->2, 4->3, 5->4, 6->5, 7->6
            for chunk in 0..packed_count {
                let off = chunk * 8;
                let src = Self::load_packed(&raw_coeffs[off..]).to_montgomery();

                // block0: identity (beta=0)
                Self::store_packed(&mut block0[off..], src);

                // block1: beta=4, beta_idx=3
                let b1 = (src * chirp_table[3 * packed_count + chunk]).con_sub_xp(1);
                Self::store_packed(&mut block1[off..], b1);
                // block2: beta=2, beta_idx=1
                let b2 = (src * chirp_table[1 * packed_count + chunk]).con_sub_xp(1);
                Self::store_packed(&mut block2[off..], b2);
                // block3: beta=6, beta_idx=5
                let b3 = (src * chirp_table[5 * packed_count + chunk]).con_sub_xp(1);
                Self::store_packed(&mut block3[off..], b3);
                // block4: beta=1, beta_idx=0
                let b4 = (src * chirp_table[0 * packed_count + chunk]).con_sub_xp(1);
                Self::store_packed(&mut block4[off..], b4);
                // block5: beta=5, beta_idx=4
                let b5 = (src * chirp_table[4 * packed_count + chunk]).con_sub_xp(1);
                Self::store_packed(&mut block5[off..], b5);
                // block6: beta=3, beta_idx=2
                let b6 = (src * chirp_table[2 * packed_count + chunk]).con_sub_xp(1);
                Self::store_packed(&mut block6[off..], b6);
                // block7: beta=7, beta_idx=6
                let b7 = (src * chirp_table[6 * packed_count + chunk]).con_sub_xp(1);
                Self::store_packed(&mut block7[off..], b7);
            }
        } else {
            // Fallback: compute chirp twiddles on-the-fly
            for chunk in 0..packed_count {
                let off = chunk * 8;
                let src = Self::load_packed(&raw_coeffs[off..]).to_montgomery();

                let tw1 = self.load_twiddle_vector(off, 1);
                let tw2 = self.load_twiddle_vector(off, 2);
                let tw4 = self.load_twiddle_vector(off, 4);

                let block1_raw = src * tw4; // beta = 4
                let block2_raw = src * tw2; // beta = 2
                let block4_raw = src * tw1; // beta = 1
                let block3_raw = block2_raw * tw4; // beta = 6
                let block5_raw = block4_raw * tw4; // beta = 5
                let block6_raw = block4_raw * tw2; // beta = 3
                let block7_raw = block6_raw * tw4; // beta = 7

                Self::store_packed(&mut block0[off..], src);
                Self::store_packed(&mut block1[off..], block1_raw.con_sub_xp(1));
                Self::store_packed(&mut block2[off..], block2_raw.con_sub_xp(1));
                Self::store_packed(&mut block3[off..], block3_raw.con_sub_xp(1));
                Self::store_packed(&mut block4[off..], block4_raw.con_sub_xp(1));
                Self::store_packed(&mut block5[off..], block5_raw.con_sub_xp(1));
                Self::store_packed(&mut block6[off..], block6_raw.con_sub_xp(1));
                Self::store_packed(&mut block7[off..], block7_raw.con_sub_xp(1));
            }
        }

        // Scalar tail for elements not fitting a full packed chunk
        for idx in (packed_count * 8)..raw_len {
            let src = raw_coeffs[idx].to_montgomery();

            let tw1 = self.elements_mont[idx];
            let tw2 = self.elements_mont[idx * 2];
            let tw4 = self.elements_mont[idx * 4];

            let block1_raw = MamaBearScalar(MamaBearScalar::mont_mul(src.0, tw4.0));
            let block2_raw = MamaBearScalar(MamaBearScalar::mont_mul(src.0, tw2.0));
            let block4_raw = MamaBearScalar(MamaBearScalar::mont_mul(src.0, tw1.0));
            let block3_raw = MamaBearScalar(MamaBearScalar::mont_mul(block2_raw.0, tw4.0));
            let block5_raw = MamaBearScalar(MamaBearScalar::mont_mul(block4_raw.0, tw4.0));
            let block6_raw = MamaBearScalar(MamaBearScalar::mont_mul(block4_raw.0, tw2.0));
            let block7_raw = MamaBearScalar(MamaBearScalar::mont_mul(block6_raw.0, tw4.0));

            block0[idx] = src;
            block1[idx] = block1_raw.con_sub_xp(1);
            block2[idx] = block2_raw.con_sub_xp(1);
            block3[idx] = block3_raw.con_sub_xp(1);
            block4[idx] = block4_raw.con_sub_xp(1);
            block5[idx] = block5_raw.con_sub_xp(1);
            block6[idx] = block6_raw.con_sub_xp(1);
            block7[idx] = block7_raw.con_sub_xp(1);
        }
    }

    #[inline(always)]
    fn dif_packed_three_layer_from_registers(
        p0: PackedMamaBearAVX512,
        p1: PackedMamaBearAVX512,
        p2: PackedMamaBearAVX512,
        p3: PackedMamaBearAVX512,
        p4: PackedMamaBearAVX512,
        p5: PackedMamaBearAVX512,
        p6: PackedMamaBearAVX512,
        p7: PackedMamaBearAVX512,
        tw_l0: PackedMamaBearAVX512,
        tw_l1: PackedMamaBearAVX512,
        tw_l2: PackedMamaBearAVX512,
        tw_l3: PackedMamaBearAVX512,
        tw_m0: PackedMamaBearAVX512,
        tw_m1: PackedMamaBearAVX512,
        tw_n: PackedMamaBearAVX512,
    ) -> (
        PackedMamaBearAVX512,
        PackedMamaBearAVX512,
        PackedMamaBearAVX512,
        PackedMamaBearAVX512,
        PackedMamaBearAVX512,
        PackedMamaBearAVX512,
        PackedMamaBearAVX512,
        PackedMamaBearAVX512,
    ) {
        let (a0, a4) = Self::dif_packed_butterfly_pair(p0, p4, tw_l0);
        let (a1, a5) = Self::dif_packed_butterfly_pair(p1, p5, tw_l1);
        let (a2, a6) = Self::dif_packed_butterfly_pair(p2, p6, tw_l2);
        let (a3, a7) = Self::dif_packed_butterfly_pair(p3, p7, tw_l3);

        let (b0, b2) = Self::dif_packed_butterfly_pair(a0, a2, tw_m0);
        let (b1, b3) = Self::dif_packed_butterfly_pair(a1, a3, tw_m1);
        let (b4, b6) = Self::dif_packed_butterfly_pair(a4, a6, tw_m0);
        let (b5, b7) = Self::dif_packed_butterfly_pair(a5, a7, tw_m1);

        let (r0, r1) = Self::dif_packed_butterfly_pair(b0, b1, tw_n);
        let (r2, r3) = Self::dif_packed_butterfly_pair(b2, b3, tw_n);
        let (r4, r5) = Self::dif_packed_butterfly_pair(b4, b5, tw_n);
        let (r6, r7) = Self::dif_packed_butterfly_pair(b6, b7, tw_n);

        (r0, r1, r2, r3, r4, r5, r6, r7)
    }

    #[inline(always)]
    fn store_dif_packed_three_layer_outputs(
        coeff: &mut [MamaBearScalar],
        k: usize,
        m0: usize,
        m1: usize,
        m2: usize,
        outputs: (
            PackedMamaBearAVX512,
            PackedMamaBearAVX512,
            PackedMamaBearAVX512,
            PackedMamaBearAVX512,
            PackedMamaBearAVX512,
            PackedMamaBearAVX512,
            PackedMamaBearAVX512,
            PackedMamaBearAVX512,
        ),
    ) {
        let (r0, r1, r2, r3, r4, r5, r6, r7) = outputs;
        Self::store_packed(&mut coeff[k..], r0);
        Self::store_packed(&mut coeff[k + m2..], r1);
        Self::store_packed(&mut coeff[k + m1..], r2);
        Self::store_packed(&mut coeff[k + m1 + m2..], r3);
        Self::store_packed(&mut coeff[k + m0..], r4);
        Self::store_packed(&mut coeff[k + m0 + m2..], r5);
        Self::store_packed(&mut coeff[k + m0 + m1..], r6);
        Self::store_packed(&mut coeff[k + m0 + m1 + m2..], r7);
    }

    #[inline(always)]
    fn prefix3_dense3_block_from_base_twiddle(
        raw0: PackedMamaBearAVX512,
        raw1: PackedMamaBearAVX512,
        raw2: PackedMamaBearAVX512,
        raw3: PackedMamaBearAVX512,
        raw4: PackedMamaBearAVX512,
        raw5: PackedMamaBearAVX512,
        raw6: PackedMamaBearAVX512,
        raw7: PackedMamaBearAVX512,
        base_tw: PackedMamaBearAVX512,
        offset_m0: PackedMamaBearAVX512,
        offset_m1: PackedMamaBearAVX512,
        offset_m2: PackedMamaBearAVX512,
        tw_l0: PackedMamaBearAVX512,
        tw_l1: PackedMamaBearAVX512,
        tw_l2: PackedMamaBearAVX512,
        tw_l3: PackedMamaBearAVX512,
        tw_m0: PackedMamaBearAVX512,
        tw_m1: PackedMamaBearAVX512,
        tw_n: PackedMamaBearAVX512,
    ) -> (
        PackedMamaBearAVX512,
        PackedMamaBearAVX512,
        PackedMamaBearAVX512,
        PackedMamaBearAVX512,
        PackedMamaBearAVX512,
        PackedMamaBearAVX512,
        PackedMamaBearAVX512,
        PackedMamaBearAVX512,
    ) {
        let tw_seg1 = (base_tw * offset_m2).con_sub_xp(1);
        let tw_seg2 = (base_tw * offset_m1).con_sub_xp(1);
        let tw_seg3 = (tw_seg2 * offset_m2).con_sub_xp(1);
        let tw_seg4 = (base_tw * offset_m0).con_sub_xp(1);
        let tw_seg5 = (tw_seg4 * offset_m2).con_sub_xp(1);
        let tw_seg6 = (tw_seg4 * offset_m1).con_sub_xp(1);
        let tw_seg7 = (tw_seg6 * offset_m2).con_sub_xp(1);

        let p0 = (raw0 * base_tw).con_sub_xp(1);
        let p1 = (raw1 * tw_seg1).con_sub_xp(1);
        let p2 = (raw2 * tw_seg2).con_sub_xp(1);
        let p3 = (raw3 * tw_seg3).con_sub_xp(1);
        let p4 = (raw4 * tw_seg4).con_sub_xp(1);
        let p5 = (raw5 * tw_seg5).con_sub_xp(1);
        let p6 = (raw6 * tw_seg6).con_sub_xp(1);
        let p7 = (raw7 * tw_seg7).con_sub_xp(1);

        Self::dif_packed_three_layer_from_registers(
            p0, p1, p2, p3, p4, p5, p6, p7, tw_l0, tw_l1, tw_l2, tw_l3, tw_m0, tw_m1, tw_n,
        )
    }

    /// Chirp-table variant: all 8 per-segment chirp twiddles are loaded from the
    /// precomputed table (sequential packed loads), eliminating the strided gather
    /// and the 7 dependent segment-offset multiplications.
    ///
    /// Range: chirp values in [0, 1.5P) (from elements_mont), raw values after
    /// to_montgomery in [0, P). mont_mul output in [0, 1.5P), con_sub_xp(1) -> [0, P).
    #[inline(always)]
    fn prefix3_dense3_block_from_chirp_table(
        raw0: PackedMamaBearAVX512,
        raw1: PackedMamaBearAVX512,
        raw2: PackedMamaBearAVX512,
        raw3: PackedMamaBearAVX512,
        raw4: PackedMamaBearAVX512,
        raw5: PackedMamaBearAVX512,
        raw6: PackedMamaBearAVX512,
        raw7: PackedMamaBearAVX512,
        chirp0: PackedMamaBearAVX512, // omega^{beta * (k..k+7)}
        chirp1: PackedMamaBearAVX512, // omega^{beta * (k+m2..k+m2+7)}
        chirp2: PackedMamaBearAVX512, // omega^{beta * (k+m1..k+m1+7)}
        chirp3: PackedMamaBearAVX512, // omega^{beta * (k+m1+m2..)}
        chirp4: PackedMamaBearAVX512, // omega^{beta * (k+m0..)}
        chirp5: PackedMamaBearAVX512, // omega^{beta * (k+m0+m2..)}
        chirp6: PackedMamaBearAVX512, // omega^{beta * (k+m0+m1..)}
        chirp7: PackedMamaBearAVX512, // omega^{beta * (k+m0+m1+m2..)}
        tw_l0: PackedMamaBearAVX512,
        tw_l1: PackedMamaBearAVX512,
        tw_l2: PackedMamaBearAVX512,
        tw_l3: PackedMamaBearAVX512,
        tw_m0: PackedMamaBearAVX512,
        tw_m1: PackedMamaBearAVX512,
        tw_n: PackedMamaBearAVX512,
    ) -> (
        PackedMamaBearAVX512,
        PackedMamaBearAVX512,
        PackedMamaBearAVX512,
        PackedMamaBearAVX512,
        PackedMamaBearAVX512,
        PackedMamaBearAVX512,
        PackedMamaBearAVX512,
        PackedMamaBearAVX512,
    ) {
        // All 8 multiplications are independent — maximum ILP.
        let p0 = (raw0 * chirp0).con_sub_xp(1); // [0, 1.5P) -> [0, P)
        let p1 = (raw1 * chirp1).con_sub_xp(1);
        let p2 = (raw2 * chirp2).con_sub_xp(1);
        let p3 = (raw3 * chirp3).con_sub_xp(1);
        let p4 = (raw4 * chirp4).con_sub_xp(1);
        let p5 = (raw5 * chirp5).con_sub_xp(1);
        let p6 = (raw6 * chirp6).con_sub_xp(1);
        let p7 = (raw7 * chirp7).con_sub_xp(1);

        Self::dif_packed_three_layer_from_registers(
            p0, p1, p2, p3, p4, p5, p6, p7, tw_l0, tw_l1, tw_l2, tw_l3, tw_m0, tw_m1, tw_n,
        )
    }

    #[inline]
    fn fft_into_zero_padded_prefix3_fused_dense3_pair_major(
        &self,
        raw_coeffs: &[MamaBearScalar],
        buf: &mut [MamaBearScalar],
    ) {
        let n = self.size();
        let raw_len = raw_coeffs.len();
        let m0 = raw_len / 2;
        let m1 = raw_len / 4;
        let m2 = raw_len / 8;
        let twiddle_step0 = 1usize << 3;
        let twiddle_step1 = 1usize << 4;
        let twiddle_step2 = 1usize << 5;

        debug_assert_eq!(raw_len << 3, n);
        debug_assert!(raw_len >= 64);

        let (block0, tail) = buf[..n].split_at_mut(raw_len);
        let (block1, tail) = tail.split_at_mut(raw_len);
        let (block2, tail) = tail.split_at_mut(raw_len);
        let (block3, tail) = tail.split_at_mut(raw_len);
        let (block4, tail) = tail.split_at_mut(raw_len);
        let (block5, tail) = tail.split_at_mut(raw_len);
        let (block6, block7) = tail.split_at_mut(raw_len);

        if let Some(ref chirp_table) = self.chirp_prefix3 {
            // Precomputed chirp tables available — use direct table loads.
            // N/64
            let packed_count = raw_len / 8;
            // Chunk offsets for the 8 segment positions within each chirp table.
            let c_off = [
                0,
                m2 / 8,
                m1 / 8,
                (m1 + m2) / 8,
                m0 / 8,
                (m0 + m2) / 8,
                (m0 + m1) / 8,
                (m0 + m1 + m2) / 8,
            ];
            // beta_idx for blocks 1..7: beta values [4,2,6,1,5,3,7] -> beta_idx [3,1,5,0,4,2,6]
            let bi = [3usize, 1, 5, 0, 4, 2, 6];
            let blocks: [&mut [MamaBearScalar]; 7] =
                [block1, block2, block3, block4, block5, block6, block7];

            for k in (0..m2).step_by(8) {
                let raw0 = Self::load_packed(&raw_coeffs[k..]).to_montgomery();
                let raw1 = Self::load_packed(&raw_coeffs[k + m2..]).to_montgomery();
                let raw2 = Self::load_packed(&raw_coeffs[k + m1..]).to_montgomery();
                let raw3 = Self::load_packed(&raw_coeffs[k + m1 + m2..]).to_montgomery();
                let raw4 = Self::load_packed(&raw_coeffs[k + m0..]).to_montgomery();
                let raw5 = Self::load_packed(&raw_coeffs[k + m0 + m2..]).to_montgomery();
                let raw6 = Self::load_packed(&raw_coeffs[k + m0 + m1..]).to_montgomery();
                let raw7 = Self::load_packed(&raw_coeffs[k + m0 + m1 + m2..]).to_montgomery();

                let tw_l0 = self.load_twiddle_vector(k, twiddle_step0);
                let tw_l1 = self.load_twiddle_vector(k + m2, twiddle_step0);
                let tw_l2 = self.load_twiddle_vector(k + m1, twiddle_step0);
                let tw_l3 = self.load_twiddle_vector(k + m1 + m2, twiddle_step0);
                let tw_m0 = self.load_twiddle_vector(k, twiddle_step1);
                let tw_m1 = self.load_twiddle_vector(k + m2, twiddle_step1);
                let tw_n = self.load_twiddle_vector(k, twiddle_step2);

                // Block 0: beta=0, identity (no chirp)
                let block0_out = Self::dif_packed_three_layer_from_registers(
                    raw0, raw1, raw2, raw3, raw4, raw5, raw6, raw7, tw_l0, tw_l1, tw_l2, tw_l3,
                    tw_m0, tw_m1, tw_n,
                );
                Self::store_dif_packed_three_layer_outputs(block0, k, m0, m1, m2, block0_out);

                // Blocks 1..7: load chirp twiddles from precomputed table
                let chunk = k / 8;
                for (blk_idx, &beta_idx) in bi.iter().enumerate() {
                    let base = beta_idx * packed_count;
                    let out = Self::prefix3_dense3_block_from_chirp_table(
                        raw0,
                        raw1,
                        raw2,
                        raw3,
                        raw4,
                        raw5,
                        raw6,
                        raw7,
                        chirp_table[base + chunk + c_off[0]],
                        chirp_table[base + chunk + c_off[1]],
                        chirp_table[base + chunk + c_off[2]],
                        chirp_table[base + chunk + c_off[3]],
                        chirp_table[base + chunk + c_off[4]],
                        chirp_table[base + chunk + c_off[5]],
                        chirp_table[base + chunk + c_off[6]],
                        chirp_table[base + chunk + c_off[7]],
                        tw_l0,
                        tw_l1,
                        tw_l2,
                        tw_l3,
                        tw_m0,
                        tw_m1,
                        tw_n,
                    );
                    Self::store_dif_packed_three_layer_outputs(blocks[blk_idx], k, m0, m1, m2, out);
                }
            }
        } else {
            // Fallback: compute chirp twiddles on-the-fly
            let offset_m0: [PackedMamaBearAVX512; 8] = std::array::from_fn(|beta| {
                PackedMamaBearAVX512::broadcast(self.elements_mont[beta * m0].0)
            });
            let offset_m1: [PackedMamaBearAVX512; 8] = std::array::from_fn(|beta| {
                PackedMamaBearAVX512::broadcast(self.elements_mont[beta * m1].0)
            });
            let offset_m2: [PackedMamaBearAVX512; 8] = std::array::from_fn(|beta| {
                PackedMamaBearAVX512::broadcast(self.elements_mont[beta * m2].0)
            });

            for k in (0..m2).step_by(8) {
                let raw0 = Self::load_packed(&raw_coeffs[k..]).to_montgomery();
                let raw1 = Self::load_packed(&raw_coeffs[k + m2..]).to_montgomery();
                let raw2 = Self::load_packed(&raw_coeffs[k + m1..]).to_montgomery();
                let raw3 = Self::load_packed(&raw_coeffs[k + m1 + m2..]).to_montgomery();
                let raw4 = Self::load_packed(&raw_coeffs[k + m0..]).to_montgomery();
                let raw5 = Self::load_packed(&raw_coeffs[k + m0 + m2..]).to_montgomery();
                let raw6 = Self::load_packed(&raw_coeffs[k + m0 + m1..]).to_montgomery();
                let raw7 = Self::load_packed(&raw_coeffs[k + m0 + m1 + m2..]).to_montgomery();

                let tw_l0 = self.load_twiddle_vector(k, twiddle_step0);
                let tw_l1 = self.load_twiddle_vector(k + m2, twiddle_step0);
                let tw_l2 = self.load_twiddle_vector(k + m1, twiddle_step0);
                let tw_l3 = self.load_twiddle_vector(k + m1 + m2, twiddle_step0);
                let tw_m0 = self.load_twiddle_vector(k, twiddle_step1);
                let tw_m1 = self.load_twiddle_vector(k + m2, twiddle_step1);
                let tw_n = self.load_twiddle_vector(k, twiddle_step2);

                let block0_out = Self::dif_packed_three_layer_from_registers(
                    raw0, raw1, raw2, raw3, raw4, raw5, raw6, raw7, tw_l0, tw_l1, tw_l2, tw_l3,
                    tw_m0, tw_m1, tw_n,
                );
                Self::store_dif_packed_three_layer_outputs(block0, k, m0, m1, m2, block0_out);

                let block1_out = Self::prefix3_dense3_block_from_base_twiddle(
                    raw0,
                    raw1,
                    raw2,
                    raw3,
                    raw4,
                    raw5,
                    raw6,
                    raw7,
                    self.load_twiddle_vector(k, 4),
                    offset_m0[4],
                    offset_m1[4],
                    offset_m2[4],
                    tw_l0,
                    tw_l1,
                    tw_l2,
                    tw_l3,
                    tw_m0,
                    tw_m1,
                    tw_n,
                );
                Self::store_dif_packed_three_layer_outputs(block1, k, m0, m1, m2, block1_out);

                let block2_out = Self::prefix3_dense3_block_from_base_twiddle(
                    raw0,
                    raw1,
                    raw2,
                    raw3,
                    raw4,
                    raw5,
                    raw6,
                    raw7,
                    self.load_twiddle_vector(k, 2),
                    offset_m0[2],
                    offset_m1[2],
                    offset_m2[2],
                    tw_l0,
                    tw_l1,
                    tw_l2,
                    tw_l3,
                    tw_m0,
                    tw_m1,
                    tw_n,
                );
                Self::store_dif_packed_three_layer_outputs(block2, k, m0, m1, m2, block2_out);

                let block3_out = Self::prefix3_dense3_block_from_base_twiddle(
                    raw0,
                    raw1,
                    raw2,
                    raw3,
                    raw4,
                    raw5,
                    raw6,
                    raw7,
                    self.load_twiddle_vector(k, 6),
                    offset_m0[6],
                    offset_m1[6],
                    offset_m2[6],
                    tw_l0,
                    tw_l1,
                    tw_l2,
                    tw_l3,
                    tw_m0,
                    tw_m1,
                    tw_n,
                );
                Self::store_dif_packed_three_layer_outputs(block3, k, m0, m1, m2, block3_out);

                let block4_out = Self::prefix3_dense3_block_from_base_twiddle(
                    raw0,
                    raw1,
                    raw2,
                    raw3,
                    raw4,
                    raw5,
                    raw6,
                    raw7,
                    self.load_twiddle_vector(k, 1),
                    offset_m0[1],
                    offset_m1[1],
                    offset_m2[1],
                    tw_l0,
                    tw_l1,
                    tw_l2,
                    tw_l3,
                    tw_m0,
                    tw_m1,
                    tw_n,
                );
                Self::store_dif_packed_three_layer_outputs(block4, k, m0, m1, m2, block4_out);

                let block5_out = Self::prefix3_dense3_block_from_base_twiddle(
                    raw0,
                    raw1,
                    raw2,
                    raw3,
                    raw4,
                    raw5,
                    raw6,
                    raw7,
                    self.load_twiddle_vector(k, 5),
                    offset_m0[5],
                    offset_m1[5],
                    offset_m2[5],
                    tw_l0,
                    tw_l1,
                    tw_l2,
                    tw_l3,
                    tw_m0,
                    tw_m1,
                    tw_n,
                );
                Self::store_dif_packed_three_layer_outputs(block5, k, m0, m1, m2, block5_out);

                let block6_out = Self::prefix3_dense3_block_from_base_twiddle(
                    raw0,
                    raw1,
                    raw2,
                    raw3,
                    raw4,
                    raw5,
                    raw6,
                    raw7,
                    self.load_twiddle_vector(k, 3),
                    offset_m0[3],
                    offset_m1[3],
                    offset_m2[3],
                    tw_l0,
                    tw_l1,
                    tw_l2,
                    tw_l3,
                    tw_m0,
                    tw_m1,
                    tw_n,
                );
                Self::store_dif_packed_three_layer_outputs(block6, k, m0, m1, m2, block6_out);

                let block7_out = Self::prefix3_dense3_block_from_base_twiddle(
                    raw0,
                    raw1,
                    raw2,
                    raw3,
                    raw4,
                    raw5,
                    raw6,
                    raw7,
                    self.load_twiddle_vector(k, 7),
                    offset_m0[7],
                    offset_m1[7],
                    offset_m2[7],
                    tw_l0,
                    tw_l1,
                    tw_l2,
                    tw_l3,
                    tw_m0,
                    tw_m1,
                    tw_n,
                );
                Self::store_dif_packed_three_layer_outputs(block7, k, m0, m1, m2, block7_out);
            }
        }
    }

    #[inline]
    fn fft_in_place_from_layer(&self, coeff: &mut [MamaBearScalar], start_layer: usize) {
        let n = coeff.len();
        assert_eq!(n, self.size(), "coeff length must equal domain size");
        let log_n = self.log_order as usize;

        if start_layer >= log_n {
            for value in coeff.iter_mut() {
                *value = value.reduce();
            }
            return;
        }

        if n < 64 {
            for layer in start_layer..log_n {
                let m = n >> (layer + 1);
                let twiddle_step = 1usize << layer;

                if m >= 8 {
                    self.dif_packed_one_layer_no_shuffle(coeff, m, twiddle_step);
                } else {
                    self.dif_scalar_butterfly_layer(coeff, m, twiddle_step);
                }
            }
            Self::reorder_pair_adjacent_to_pair_major_blocks_and_reduce_in_place(coeff);
            return;
        }

        let mut layer = start_layer;
        let mut remaining = log_n - start_layer;

        while remaining > 3 {
            let m0 = n >> (layer + 1);
            let twiddle_step0 = 1usize << layer;

            if remaining >= 6 {
                self.dif_packed_three_layer_no_shuffle(coeff, m0, twiddle_step0);
                layer += 3;
                remaining -= 3;
            } else if remaining == 5 {
                self.dif_packed_two_layer_no_shuffle(coeff, m0, twiddle_step0);
                layer += 2;
                remaining -= 2;
            } else {
                debug_assert_eq!(remaining, 4);
                self.dif_packed_one_layer_no_shuffle(coeff, m0, twiddle_step0);
                layer += 1;
                remaining -= 1;
            }
        }

        if remaining == 3 {
            self.dif_packed_tail_three_layer_pairmajor(coeff);
            return;
        }

        while remaining > 0 {
            let m = n >> (layer + 1);
            let twiddle_step = 1usize << layer;

            if m >= 8 {
                self.dif_packed_one_layer_no_shuffle(coeff, m, twiddle_step);
            } else {
                self.dif_scalar_butterfly_layer(coeff, m, twiddle_step);
            }

            layer += 1;
            remaining -= 1;
        }

        Self::reorder_pair_adjacent_to_pair_major_blocks_and_reduce_in_place(coeff);
    }

    #[inline]
    fn fft_into_zero_padded_pow2_pair_major(
        &self,
        raw_coeffs: &[MamaBearScalar],
        buf: &mut [MamaBearScalar],
        prefix_layers: usize,
    ) {
        debug_assert!(prefix_layers > 0);

        if prefix_layers == 3 {
            let remaining_layers = self.log_order as usize - prefix_layers;
            if remaining_layers >= 6 && raw_coeffs.len() >= 64 {
                self.fft_into_zero_padded_prefix3_fused_dense3_pair_major(raw_coeffs, buf);
                self.fft_in_place_from_layer(&mut buf[..self.size()], prefix_layers + 3);
                return;
            }

            self.fft_into_zero_padded_prefix3_pair_major(raw_coeffs, buf);
            self.fft_in_place_from_layer(&mut buf[..self.size()], prefix_layers);
            return;
        }

        self.materialize_zero_padded_prefix_state(raw_coeffs, buf, prefix_layers);
        self.fft_in_place_from_layer(&mut buf[..self.size()], prefix_layers);
    }

    /// In-place Radix-2 DIF (Gentleman-Sande) FFT on Montgomery-form values.
    ///
    /// Input: `coeff[i]` in Montgomery form, natural order, all values in [0, 2P).
    /// Output: evaluations in canonical Montgomery form, pair-major blocked order,
    /// all values in [0, P).
    ///
    /// DIF processes top-down (largest stride first), which eliminates the
    /// bit-reversal permutation and provides better cache access patterns for
    /// the most expensive early layers.
    ///
    /// DIF butterfly: u' = u + v, v' = (u - v) * w
    pub fn fft_in_place(&self, coeff: &mut [MamaBearScalar]) {
        let n = coeff.len();
        assert_eq!(n, self.size(), "coeff length must equal domain size");
        let log_n = self.log_order as usize;

        if n <= 1 {
            for value in coeff.iter_mut() {
                *value = value.reduce();
            }
            return;
        }

        if log_n < 6 {
            self.fft_in_place_small(coeff);
            Self::reorder_pair_adjacent_to_pair_major_blocks_and_reduce_in_place(coeff);
            return;
        }

        let mut layer = 0usize;
        let mut remaining = log_n;

        while remaining > 3 {
            let m0 = n >> (layer + 1);
            let twiddle_step0 = 1usize << layer;

            if remaining >= 6 {
                self.dif_packed_three_layer_no_shuffle(coeff, m0, twiddle_step0);
                layer += 3;
                remaining -= 3;
            } else if remaining == 5 {
                self.dif_packed_two_layer_no_shuffle(coeff, m0, twiddle_step0);
                layer += 2;
                remaining -= 2;
            } else {
                debug_assert_eq!(remaining, 4);
                self.dif_packed_one_layer_no_shuffle(coeff, m0, twiddle_step0);
                layer += 1;
                remaining -= 1;
            }
        }

        debug_assert_eq!(remaining, 3);
        self.dif_packed_tail_three_layer_pairmajor(coeff);
    }

    #[inline]
    fn fft_in_place_small(&self, coeff: &mut [MamaBearScalar]) {
        let n = coeff.len();
        let log_n = self.log_order as usize;

        for layer in 0..log_n {
            let m = n >> (layer + 1);
            let twiddle_step = 1usize << layer;

            if m >= 8 {
                self.dif_packed_one_layer_no_shuffle(coeff, m, twiddle_step);
            } else {
                self.dif_scalar_butterfly_layer(coeff, m, twiddle_step);
            }
        }
    }

    /// In-place Radix-2 DIT (Cooley-Tukey) FFT on Montgomery-form values.
    /// Kept for backward compatibility (natural-order output).
    ///
    /// Input: `coeff[i]` in Montgomery form, all values in [0, 2P).
    /// Output: `coeff[i]` = evaluations in Montgomery form, **natural order**, all in [0, 2P).
    pub fn fft_in_place_natural(&self, coeff: &mut [MamaBearScalar]) {
        let n = coeff.len();
        assert_eq!(n, self.size(), "coeff length must equal domain size");
        let log_n = self.log_order as usize;

        // Bit-reversal permutation
        Self::bit_reverse_in_place(coeff);

        // DIT butterfly layers (bottom-up)
        for log_m in 0..log_n {
            let m = 1usize << log_m;
            let twiddle_step = n >> (log_m + 1);

            if m >= 8 {
                self.packed_butterfly_layer(coeff, m, twiddle_step);
            } else {
                self.scalar_butterfly_layer(coeff, m, twiddle_step);
            }
        }
    }

    /// DIF scalar butterfly layer for stride < 8.
    /// DIF butterfly: u' = u + v, v' = (u - v) * w
    ///
    /// Range analysis (inputs in [0, 2P), twiddles in [0, 1.5P]):
    ///   u' = u + v                           → [0, 4P) then con_sub_xp(2) → [0, 2P)
    ///   diff = u + 2P - v                    → [0, 4P)
    ///   v' = mont_mul(w, diff)               → [0, 1.75P)
    #[inline]
    fn dif_scalar_butterfly_layer(
        &self,
        coeff: &mut [MamaBearScalar],
        m: usize,
        twiddle_step: usize,
    ) {
        let n = coeff.len();
        for j in (0..n).step_by(m * 2) {
            for k in 0..m {
                let w = self.elements_mont[k * twiddle_step]; // [0, 2P)
                let u = coeff[j + k]; // [0, 2P)
                let v = coeff[j + k + m]; // [0, 2P)
                coeff[j + k] = u.lazy_add(v).con_sub_xp(2); // [0, 2P)
                let diff = u.lazy_add_xp(2).lazy_sub(v); // [0, 4P)
                coeff[j + k + m] = MamaBearScalar(MamaBearScalar::mont_mul(w.0, diff.0));
                // [0, 1.75P)
            }
        }
    }

    /// DIF packed butterfly layer for stride >= 8.
    /// DIF butterfly: u' = u + v, v' = (u - v) * w
    ///
    /// Range analysis: same as scalar — inputs [0, 2P), outputs [0, 2P).
    #[inline]
    fn dif_packed_one_layer_no_shuffle(
        &self,
        coeff: &mut [MamaBearScalar],
        m: usize,
        twiddle_step: usize,
    ) {
        let n = coeff.len();
        for j in (0..n).step_by(m * 2) {
            for k in (0..m).step_by(8) {
                let mut w_arr = [0u64; 8];
                for i in 0..8 {
                    w_arr[i] = self.elements_mont[(k + i) * twiddle_step].0;
                }
                let w_packed = PackedMamaBearAVX512::from_array(w_arr);

                let u = Self::load_packed(&coeff[j + k..]);
                let v = Self::load_packed(&coeff[j + k + m..]);

                // DIF: u' = u + v, v' = (u - v) * w
                let out_plus = u.lazy_add(v).con_sub_xp(2); // [0, 2P)
                let diff = u.lazy_add_xp(2).lazy_sub(v); // [0, 4P)
                let out_minus = w_packed * diff; // [0, 1.75P)

                Self::store_packed(&mut coeff[j + k..], out_plus);
                Self::store_packed(&mut coeff[j + k + m..], out_minus);
            }
        }
    }

    /// Register-fused two-layer DIF pass: layers L and L+1 with zero intermediate memory traffic.
    ///
    /// For each group of 4 packed vectors at positions (k, k+m1, k+m0, k+m0+m1):
    ///   1. Load 4 vectors: a, b, c, d                                     (4 loads)
    ///   2. Layer L: butterfly(a,c)->u_ac,v_ac, butterfly(b,d)->u_bd,v_bd  (in registers)
    ///   3. Layer L+1: butterfly(u_ac,u_bd)->r0,r1, butterfly(v_ac,v_bd)->r2,r3 (in registers)
    ///   4. Store 4 results: r0, r1, r2, r3                               (4 stores)
    ///
    /// vs cache-level merge: 8 loads + 8 stores -> 4 loads + 4 stores (halved).
    /// 3 twiddle loads per iteration (vs 4 in cache-level: w0[k] reused across both L+1 halves).
    #[inline]
    fn dif_packed_two_layer_no_shuffle(
        &self,
        coeff: &mut [MamaBearScalar],
        m0: usize,
        twiddle_step0: usize,
    ) {
        let n = coeff.len();
        let m1 = m0 / 2;
        let twiddle_step1 = twiddle_step0 * 2;

        for j in (0..n).step_by(m0 * 2) {
            for k in (0..m1).step_by(8) {
                // Load 4 packed vectors from the 4 quarter-positions
                let a = Self::load_packed(&coeff[j + k..]); // 1st quarter
                let b = Self::load_packed(&coeff[j + k + m1..]); // 2nd quarter
                let c = Self::load_packed(&coeff[j + k + m0..]); // 3rd quarter
                let d = Self::load_packed(&coeff[j + k + m0 + m1..]); // 4th quarter

                // 3 twiddle vectors (w1_k shared by both L+1 butterflies)
                let w0_k = self.load_twiddle_vector(k, twiddle_step0);
                let w0_km1 = self.load_twiddle_vector(k + m1, twiddle_step0);
                let w1_k = self.load_twiddle_vector(k, twiddle_step1);

                // Layer L: two butterflies at stride m0 — results stay in registers
                let (u_ac, v_ac) = Self::dif_packed_butterfly_pair(a, c, w0_k);
                let (u_bd, v_bd) = Self::dif_packed_butterfly_pair(b, d, w0_km1);

                // Layer L+1: two butterflies at stride m1 — directly from registers
                let (r0, r1) = Self::dif_packed_butterfly_pair(u_ac, u_bd, w1_k);
                let (r2, r3) = Self::dif_packed_butterfly_pair(v_ac, v_bd, w1_k);

                // Store 4 results
                Self::store_packed(&mut coeff[j + k..], r0);
                Self::store_packed(&mut coeff[j + k + m1..], r1);
                Self::store_packed(&mut coeff[j + k + m0..], r2);
                Self::store_packed(&mut coeff[j + k + m0 + m1..], r3);
            }
        }
    }

    /// Register-fused three-layer DIF pass: layers L, L+1, L+2 in one sweep.
    ///
    /// 8 positions at m2-intervals: (k, k+m2, k+m1, k+m1+m2, k+m0, k+m0+m2, k+m0+m1, k+m0+m1+m2)
    /// where m0 = stride_L, m1 = m0/2, m2 = m0/4.
    ///
    /// Per iteration: 8 loads, 12 butterflies (all in registers), 8 stores.
    /// vs unfused: 24 loads + 24 stores. vs two-layer fused: ~12 loads + 12 stores.
    /// Twiddle loads: 7 per iteration (4 for L, 2 for L+1, 1 for L+2).
    /// Peak register usage: ~16 (8 data + 4 twiddles + 4 temps). Fits AVX-512's 32 easily.
    #[inline]
    fn dif_packed_three_layer_no_shuffle(
        &self,
        coeff: &mut [MamaBearScalar],
        m0: usize,
        twiddle_step0: usize,
    ) {
        let n = coeff.len();
        let m1 = m0 / 2;
        let m2 = m0 / 4;
        let twiddle_step1 = twiddle_step0 * 2;
        let twiddle_step2 = twiddle_step0 * 4;

        for j in (0..n).step_by(m0 * 2) {
            for k in (0..m2).step_by(8) {
                // Load 8 packed vectors at m2-spaced positions
                let p0 = Self::load_packed(&coeff[j + k..]);
                let p1 = Self::load_packed(&coeff[j + k + m2..]);
                let p2 = Self::load_packed(&coeff[j + k + m1..]);
                let p3 = Self::load_packed(&coeff[j + k + m1 + m2..]);
                let p4 = Self::load_packed(&coeff[j + k + m0..]);
                let p5 = Self::load_packed(&coeff[j + k + m0 + m2..]);
                let p6 = Self::load_packed(&coeff[j + k + m0 + m1..]);
                let p7 = Self::load_packed(&coeff[j + k + m0 + m1 + m2..]);

                // Layer L (stride m0): 4 butterflies — twiddles at k, k+m2, k+m1, k+m1+m2
                let tw_l0 = self.load_twiddle_vector(k, twiddle_step0);
                let tw_l1 = self.load_twiddle_vector(k + m2, twiddle_step0);
                let tw_l2 = self.load_twiddle_vector(k + m1, twiddle_step0);
                let tw_l3 = self.load_twiddle_vector(k + m1 + m2, twiddle_step0);

                // Layer L+1 (stride m1): 4 butterflies — twiddles at k, k+m2
                let tw_m0 = self.load_twiddle_vector(k, twiddle_step1);
                let tw_m1 = self.load_twiddle_vector(k + m2, twiddle_step1);

                // Layer L+2 (stride m2): 4 butterflies — single twiddle at k
                let tw_n = self.load_twiddle_vector(k, twiddle_step2);

                let (r0, r1, r2, r3, r4, r5, r6, r7) = Self::dif_packed_three_layer_from_registers(
                    p0, p1, p2, p3, p4, p5, p6, p7, tw_l0, tw_l1, tw_l2, tw_l3, tw_m0, tw_m1, tw_n,
                );

                // Store 8 results
                Self::store_packed(&mut coeff[j + k..], r0);
                Self::store_packed(&mut coeff[j + k + m2..], r1);
                Self::store_packed(&mut coeff[j + k + m1..], r2);
                Self::store_packed(&mut coeff[j + k + m1 + m2..], r3);
                Self::store_packed(&mut coeff[j + k + m0..], r4);
                Self::store_packed(&mut coeff[j + k + m0 + m2..], r5);
                Self::store_packed(&mut coeff[j + k + m0 + m1..], r6);
                Self::store_packed(&mut coeff[j + k + m0 + m1 + m2..], r7);
            }
        }
    }

    #[inline(always)]
    fn load_twiddle_vector(&self, start: usize, twiddle_step: usize) -> PackedMamaBearAVX512 {
        let mut w_arr = [0u64; 8];
        for i in 0..8 {
            w_arr[i] = self.elements_mont[(start + i) * twiddle_step].0;
        }
        PackedMamaBearAVX512::from_array(w_arr)
    }

    #[inline(always)]
    fn dif_packed_butterfly_pair(
        u: PackedMamaBearAVX512,
        v: PackedMamaBearAVX512,
        twiddle: PackedMamaBearAVX512,
    ) -> (PackedMamaBearAVX512, PackedMamaBearAVX512) {
        let out_plus = u.lazy_add(v).con_sub_xp(2); // [0, 2P)
        let diff = u.lazy_add_xp(2).lazy_sub(v); // [0, 4P)
        let out_minus = twiddle * diff; // [0, 1.75P)
        (out_plus, out_minus)
    }

    #[inline(always)]
    fn dif_packed_butterfly_pair_final_reduce(
        u: PackedMamaBearAVX512,
        v: PackedMamaBearAVX512,
        twiddle: PackedMamaBearAVX512,
    ) -> (PackedMamaBearAVX512, PackedMamaBearAVX512) {
        let (out_plus, out_minus) = Self::dif_packed_butterfly_pair(u, v, twiddle);
        (out_plus.reduce(), out_minus.reduce())
    }

    #[inline(always)]
    fn dif_packed_tail_stage(
        left: PackedMamaBearAVX512,
        right: PackedMamaBearAVX512,
        shuffle_lo: [u64; 8],
        shuffle_hi: [u64; 8],
        twiddle: PackedMamaBearAVX512,
    ) -> (PackedMamaBearAVX512, PackedMamaBearAVX512) {
        let u = left.permute2(right, shuffle_lo);
        let v = left.permute2(right, shuffle_hi);
        Self::dif_packed_butterfly_pair(u, v, twiddle)
    }

    #[inline(always)]
    fn dif_packed_tail_stage_final_reduce(
        left: PackedMamaBearAVX512,
        right: PackedMamaBearAVX512,
        shuffle_lo: [u64; 8],
        shuffle_hi: [u64; 8],
        twiddle: PackedMamaBearAVX512,
    ) -> (PackedMamaBearAVX512, PackedMamaBearAVX512) {
        let u = left.permute2(right, shuffle_lo);
        let v = left.permute2(right, shuffle_hi);
        Self::dif_packed_butterfly_pair_final_reduce(u, v, twiddle)
    }

    /// Fuse the final 3 DIF layers into a pair-major blocked layout.
    ///
    /// Each adjacent register pair carries two independent 8-point subproblems.
    /// The output of every 16-value block is `[x0..x7, nx0..nx7]` for 8 pair slots.
    #[inline]
    fn dif_packed_tail_three_layer_pairmajor(&self, coeff: &mut [MamaBearScalar]) {
        debug_assert_eq!(coeff.len() & 63, 0);

        const SHUF4_LO: [u64; 8] = [0, 1, 2, 3, 8, 9, 10, 11];
        const SHUF4_HI: [u64; 8] = [4, 5, 6, 7, 12, 13, 14, 15];
        const SHUF2_LO: [u64; 8] = [0, 1, 8, 9, 4, 5, 12, 13];
        const SHUF2_HI: [u64; 8] = [2, 3, 10, 11, 6, 7, 14, 15];
        const SHUF1_LO: [u64; 8] = [0, 8, 2, 10, 4, 12, 6, 14];
        const SHUF1_HI: [u64; 8] = [1, 9, 3, 11, 5, 13, 7, 15];

        let mut groups = coeff.chunks_exact_mut(64);
        for group in &mut groups {
            let mut p0 = Self::load_packed(&group[0..]);
            let mut p1 = Self::load_packed(&group[8..]);
            let mut p2 = Self::load_packed(&group[16..]);
            let mut p3 = Self::load_packed(&group[24..]);
            let mut p4 = Self::load_packed(&group[32..]);
            let mut p5 = Self::load_packed(&group[40..]);
            let mut p6 = Self::load_packed(&group[48..]);
            let mut p7 = Self::load_packed(&group[56..]);

            (p0, p1) =
                Self::dif_packed_tail_stage(p0, p1, SHUF4_LO, SHUF4_HI, self.tail_twiddle_last3);
            (p2, p3) =
                Self::dif_packed_tail_stage(p2, p3, SHUF4_LO, SHUF4_HI, self.tail_twiddle_last3);
            (p4, p5) =
                Self::dif_packed_tail_stage(p4, p5, SHUF4_LO, SHUF4_HI, self.tail_twiddle_last3);
            (p6, p7) =
                Self::dif_packed_tail_stage(p6, p7, SHUF4_LO, SHUF4_HI, self.tail_twiddle_last3);

            (p0, p1) =
                Self::dif_packed_tail_stage(p0, p1, SHUF2_LO, SHUF2_HI, self.tail_twiddle_last2);
            (p2, p3) =
                Self::dif_packed_tail_stage(p2, p3, SHUF2_LO, SHUF2_HI, self.tail_twiddle_last2);
            (p4, p5) =
                Self::dif_packed_tail_stage(p4, p5, SHUF2_LO, SHUF2_HI, self.tail_twiddle_last2);
            (p6, p7) =
                Self::dif_packed_tail_stage(p6, p7, SHUF2_LO, SHUF2_HI, self.tail_twiddle_last2);

            (p0, p1) = Self::dif_packed_tail_stage_final_reduce(
                p0,
                p1,
                SHUF1_LO,
                SHUF1_HI,
                self.tail_twiddle_last1,
            );
            (p2, p3) = Self::dif_packed_tail_stage_final_reduce(
                p2,
                p3,
                SHUF1_LO,
                SHUF1_HI,
                self.tail_twiddle_last1,
            );
            (p4, p5) = Self::dif_packed_tail_stage_final_reduce(
                p4,
                p5,
                SHUF1_LO,
                SHUF1_HI,
                self.tail_twiddle_last1,
            );
            (p6, p7) = Self::dif_packed_tail_stage_final_reduce(
                p6,
                p7,
                SHUF1_LO,
                SHUF1_HI,
                self.tail_twiddle_last1,
            );

            Self::store_packed(&mut group[0..], p0);
            Self::store_packed(&mut group[8..], p1);
            Self::store_packed(&mut group[16..], p2);
            Self::store_packed(&mut group[24..], p3);
            Self::store_packed(&mut group[32..], p4);
            Self::store_packed(&mut group[40..], p5);
            Self::store_packed(&mut group[48..], p6);
            Self::store_packed(&mut group[56..], p7);
        }

        debug_assert!(groups.into_remainder().is_empty());
    }

    /// DIT scalar butterfly layer for stride < 8 (used by fft_in_place_natural / ifft).
    ///
    /// Range analysis per butterfly (inputs in [0, 2P)):
    ///   t = mont_mul(w, c[j+k+m])          → [0, 1.5P)   (both inputs < 2P)
    ///   c[j+k+m] = c[j+k] + 2P - t         → [0, 4P) then con_sub_xp(2) → [0, 2P)
    ///   c[j+k]   = c[j+k] + t               → [0, 3.5P) then con_sub_xp(2) → [0, 2P)
    #[inline]
    fn scalar_butterfly_layer(&self, coeff: &mut [MamaBearScalar], m: usize, twiddle_step: usize) {
        let n = coeff.len();
        for j in (0..n).step_by(m * 2) {
            for k in 0..m {
                let w = self.elements_mont[k * twiddle_step]; // [0, 2P)
                let u = coeff[j + k]; // [0, 2P)
                let v = coeff[j + k + m]; // [0, 2P)
                let t = MamaBearScalar(MamaBearScalar::mont_mul(w.0, v.0)); // [0, 1.5P)
                coeff[j + k] = u.lazy_add(t).con_sub_xp(2); // [0, 3.5P) → [0, 2P)
                coeff[j + k + m] = u.lazy_add_xp(2).lazy_sub(t).con_sub_xp(2); // [0, 4P) → [0, 2P)
            }
        }
    }

    /// Packed butterfly layer for stride >= 8.
    /// Processes 8 consecutive butterflies sharing the same twiddle factor as one PBF operation.
    ///
    /// Range analysis: same as scalar — inputs [0, 2P), outputs [0, 2P).
    #[inline]
    fn packed_butterfly_layer(&self, coeff: &mut [MamaBearScalar], m: usize, twiddle_step: usize) {
        let n = coeff.len();
        for j in (0..n).step_by(m * 2) {
            for k in (0..m).step_by(8) {
                // Load 8 different twiddle factors for positions k..k+8
                let mut w_arr = [0u64; 8];
                for i in 0..8 {
                    w_arr[i] = self.elements_mont[(k + i) * twiddle_step].0;
                }
                let w_packed = PackedMamaBearAVX512::from_array(w_arr);

                // Load 8 consecutive scalars into packed register (unaligned-safe)
                let u = Self::load_packed(&coeff[j + k..]);
                let v = Self::load_packed(&coeff[j + k + m..]);

                // Butterfly: t = w * v, out+ = u + t, out- = u - t
                let t = w_packed * v; // mont_mul, [0, 1.5P) since inputs < 2P
                let out_plus = u.lazy_add(t).con_sub_xp(2); // [0, 2P)
                let out_minus = u.lazy_add_xp(2).lazy_sub(t).con_sub_xp(2); // [0, 2P)

                Self::store_packed(&mut coeff[j + k..], out_plus);
                Self::store_packed(&mut coeff[j + k + m..], out_minus);
            }
        }
    }

    /// Load 8 consecutive MamaBearScalar values into a PackedMamaBearAVX512.
    #[inline(always)]
    fn load_packed(slice: &[MamaBearScalar]) -> PackedMamaBearAVX512 {
        debug_assert!(slice.len() >= 8);
        unsafe { ptr::read_unaligned(slice.as_ptr() as *const PackedMamaBearAVX512) }
    }

    /// Store a PackedMamaBearAVX512 back into 8 consecutive MamaBearScalar values.
    #[inline(always)]
    fn store_packed(slice: &mut [MamaBearScalar], val: PackedMamaBearAVX512) {
        debug_assert!(slice.len() >= 8);
        unsafe {
            ptr::write_unaligned(slice.as_mut_ptr() as *mut PackedMamaBearAVX512, val);
        }
    }

    #[inline]
    fn reorder_pair_adjacent_to_pair_major_blocks_and_reduce_in_place(
        coeff: &mut [MamaBearScalar],
    ) {
        let pair_count = coeff.len() >> 1;
        if pair_count == 0 {
            for value in coeff.iter_mut() {
                *value = value.reduce();
            }
            return;
        }

        let pairs_per_block = Self::pair_slots_per_block_for_pair_count(pair_count);
        let block_len = pairs_per_block * 2;
        let mut tmp = [MamaBearScalar(0); 16];

        for block in coeff.chunks_exact_mut(block_len) {
            tmp[..block_len].copy_from_slice(block);
            for lane in 0..pairs_per_block {
                block[lane] = tmp[2 * lane].reduce();
                block[pairs_per_block + lane] = tmp[2 * lane + 1].reduce();
            }
        }
    }

    #[inline]
    fn reorder_pair_major_blocks_to_adjacent_in_place(coeff: &mut [MamaBearScalar]) {
        let pair_count = coeff.len() >> 1;
        if pair_count == 0 {
            return;
        }

        let pairs_per_block = Self::pair_slots_per_block_for_pair_count(pair_count);
        let block_len = pairs_per_block * 2;
        let mut tmp = [MamaBearScalar(0); 16];

        for block in coeff.chunks_exact_mut(block_len) {
            tmp[..block_len].copy_from_slice(block);
            for lane in 0..pairs_per_block {
                block[2 * lane] = tmp[lane];
                block[2 * lane + 1] = tmp[pairs_per_block + lane];
            }
        }
    }

    /// Forward FFT: coefficients -> evaluations in pair-major blocked order.
    ///
    /// Accepts raw-form coefficients, returns Montgomery-form evaluations.
    /// Pair slot order matches the historical bit-reversed pair order, but each
    /// local block is stored as `[x0..xk, nx0..nxk]` instead of
    /// `[x0, nx0, x1, nx1, ...]`.
    pub fn fft(&self, raw_coeffs: &[MamaBearScalar]) -> Vec<MamaBearScalar> {
        let n = self.size();
        let mut buf = vec![MamaBearScalar(0); n];
        self.fft_into(raw_coeffs, &mut buf);
        buf
    }

    /// Forward FFT into a caller-provided buffer (avoids per-call allocation).
    ///
    /// `buf` must have length >= self.size(). On return, `buf[0..n]` contains
    /// canonical Montgomery-form evaluations in pair-major blocked order.
    pub fn fft_into(&self, raw_coeffs: &[MamaBearScalar], buf: &mut [MamaBearScalar]) {
        let n = self.size();
        assert!(raw_coeffs.len() <= n, "coeff length exceeds domain size");
        assert!(buf.len() >= n, "buffer too small");

        if let Some(prefix_layers) = self.zero_padding_prefix_layers(raw_coeffs.len()) {
            self.fft_into_zero_padded_pow2_pair_major(raw_coeffs, buf, prefix_layers);
            return;
        }

        buf[..raw_coeffs.len()].copy_from_slice(raw_coeffs);
        Self::raw_to_montgomery_in_place(&mut buf[..raw_coeffs.len()]);
        // Zero-pad the rest (MamaBearScalar(0) — same as fft())
        for v in buf[raw_coeffs.len()..n].iter_mut() {
            v.0 = 0;
        }

        self.fft_in_place(&mut buf[..n]);
    }

    /// Forward FFT: coefficients → evaluations in **natural order**.
    ///
    /// Accepts raw-form coefficients, returns Montgomery-form evaluations.
    /// Uses DIT with bit-reversal — for use cases that need natural-order output.
    pub fn fft_natural(&self, raw_coeffs: &[MamaBearScalar]) -> Vec<MamaBearScalar> {
        let n = self.size();
        assert!(raw_coeffs.len() <= n, "coeff length exceeds domain size");

        let mut data = vec![MamaBearScalar(ONE_MONT).lazy_sub(MamaBearScalar(ONE_MONT)); n];
        data[..raw_coeffs.len()].copy_from_slice(raw_coeffs);
        Self::raw_to_montgomery_in_place(&mut data[..raw_coeffs.len()]);

        self.fft_in_place_natural(&mut data);
        data
    }

    /// Inverse FFT: evaluations (pair-major blocked order) -> coefficients (natural order).
    ///
    /// Input: pair-major blocked Montgomery-form evaluations (as produced by `fft`/`fft_in_place`).
    /// Output: natural-order Montgomery-form coefficients.
    ///
    /// Uses a local layout inverse to recover the historical pair-adjacent
    /// bit-reversed order, then runs the existing DIT inverse.
    pub fn ifft_in_place(&self, evals: &mut [MamaBearScalar]) {
        let n = evals.len();
        assert_eq!(n, self.size());
        let log_n = self.log_order as usize;

        Self::reorder_pair_major_blocks_to_adjacent_in_place(evals);

        // Use inverse omega = omega^{n-1} (in Montgomery form)
        let inv_omega_mont = self.elements_mont[n - 1];

        // Build inverse twiddle table
        let mut inv_elements = Vec::with_capacity(n);
        let mut current = MamaBearScalar(ONE_MONT);
        for _ in 0..n {
            inv_elements.push(current);
            current = MamaBearScalar(MamaBearScalar::mont_mul(current.0, inv_omega_mont.0));
        }

        // After the local inverse layout pass, the buffer is back in the
        // historical pair-adjacent bit-reversed order expected by the DIT inverse.
        for log_m in 0..log_n {
            let m = 1usize << log_m;
            let twiddle_step = n >> (log_m + 1);
            for j in (0..n).step_by(m * 2) {
                for k in 0..m {
                    let w = inv_elements[k * twiddle_step];
                    let u = evals[j + k];
                    let v = evals[j + k + m];
                    let t = MamaBearScalar(MamaBearScalar::mont_mul(w.0, v.0));
                    evals[j + k] = u.lazy_add(t).con_sub_xp(2);
                    evals[j + k + m] = u.lazy_add_xp(2).lazy_sub(t).con_sub_xp(2);
                }
            }
        }

        // Scale by 1/n in Montgomery form
        let n_inv_raw = MamaBearScalar(n as u64).inv().unwrap();
        let n_inv_mont = n_inv_raw.to_montgomery();
        Self::scale_montgomery_slice_in_place(evals, n_inv_mont);
    }

    // ── Bailey's 4-Step FFT ─────────────────────────────────────────────

    /// Threshold: use Bailey when log_n > this value.
    /// 2^13 = 8K elements = 64 KB ≈ L1D cache.
    const BAILEY_LOG_THRESHOLD: u32 = 13;

    /// Forward FFT using Bailey's 4-Step algorithm for large domains.
    ///
    /// For small domains (log_n ≤ 13), falls back to standard DIT.
    /// For large domains, uses column-first decomposition:
    ///   1. Transpose → column DFTs as rows → twiddle → transpose → row DFTs → digit-reversal
    ///
    /// Input: raw-form coefficients. Output: Montgomery-form evaluations in natural order.
    pub fn fft_natural_bailey(&self, raw_coeffs: &[MamaBearScalar]) -> Vec<MamaBearScalar> {
        let n = self.size();
        assert!(raw_coeffs.len() <= n, "coeff length exceeds domain size");

        // Convert to Montgomery and pad
        let mut data: Vec<MamaBearScalar> = raw_coeffs.iter().map(|x| x.to_montgomery()).collect();
        data.resize(
            n,
            MamaBearScalar(ONE_MONT).lazy_sub(MamaBearScalar(ONE_MONT)),
        );

        if self.log_order <= Self::BAILEY_LOG_THRESHOLD {
            // Small domain: use standard DIT
            self.fft_in_place_natural(&mut data);
            return data;
        }

        // Split N = N1 × N2. View data as N1 rows × N2 columns (row-major).
        // Column-first: do column DFTs (size N1) first, then row DFTs (size N2).
        // For L1 friendliness, we want both N1 and N2 to fit in L1 cache (48KB → ≤ 2^12 elements).
        let log_n1 = self.log_order / 2; // column DFT size
        let log_n2 = self.log_order - log_n1; // row DFT size
        let n1 = 1usize << log_n1; // number of rows
        let n2 = 1usize << log_n2; // row length (columns)

        let fft_n1 = MamaBearFFT::new(log_n1);
        let fft_n2 = MamaBearFFT::new(log_n2);

        // Step 1: Transpose N1×N2 → N2×N1 (columns become rows)
        let mut tmp = vec![MamaBearScalar(0); n];
        Self::block_transpose(&data, n1, n2, &mut tmp);

        // Step 2: Column DFTs — N2 independent DFT_N1 (each row of transposed matrix)
        for j in 0..n2 {
            let row = &mut tmp[j * n1..(j + 1) * n1];
            fft_n1.fft_in_place_natural(row);
        }

        // Step 3: Twiddle — element (j, k1) in the N2×N1 matrix *= ω_N^{j·k1}
        // This applies the "cross" twiddle factor between column and row DFTs.
        self.bailey_twiddle_multiply(&mut tmp, n2, n1);

        // Step 4: Transpose N2×N1 → N1×N2
        Self::block_transpose(&tmp, n2, n1, &mut data);

        // Step 5: Row DFTs — N1 independent DFT_N2 (each row of the N1×N2 matrix)
        for i in 0..n1 {
            let row = &mut data[i * n2..(i + 1) * n2];
            fft_n2.fft_in_place_natural(row);
        }

        // Step 6: Digit-reversal — output[k1+k2*N1] = data[k1*N2+k2]
        // The column-first decomposition produces X[k1+k2*N1] at position k1*N2+k2.
        // To get natural order X[k] at position k, we need this permutation.
        Self::digit_reversal(&data, n1, n2, &mut tmp);

        tmp
    }

    /// Forward FFT using Bailey's 4-Step algorithm, returning pair-major blocked output.
    ///
    /// This remains a semantic path: natural-order Bailey, then a final bit-reversal,
    /// then the same local pair-major block transform used by the main DIF path.
    /// The final reorder also canonicalizes each Montgomery value to [0, P).
    pub fn fft_bailey_pair_major(&self, raw_coeffs: &[MamaBearScalar]) -> Vec<MamaBearScalar> {
        if self.zero_padding_prefix_layers(raw_coeffs.len()).is_some() {
            return self.fft(raw_coeffs);
        }

        let mut data = self.fft_natural_bailey(raw_coeffs);
        Self::bit_reverse_in_place(&mut data);
        Self::reorder_pair_adjacent_to_pair_major_blocks_and_reduce_in_place(&mut data);
        data
    }

    /// Digit-reversal permutation: output[k1 + k2*n1] = input[k1*n2 + k2].
    /// For n1 == n2, this is equivalent to a matrix transpose.
    /// Uses 8×8 blocks for cache locality.
    fn digit_reversal(src: &[MamaBearScalar], n1: usize, n2: usize, dst: &mut [MamaBearScalar]) {
        // src is n1×n2 row-major: src[k1*n2+k2]
        // dst[k1+k2*n1] = src[k1*n2+k2]
        // This is the same as: for each (k1,k2), dst[k2*n1+k1] = src[k1*n2+k2]
        // Which is a matrix transpose: src[n1×n2] → dst[n2×n1]
        Self::block_transpose(src, n1, n2, dst);
    }

    /// Twiddle multiplication for Bailey's 4-Step: data[i*n2+j] *= ω_N^{i·j}.
    /// Uses online recurrence with PBF SIMD (8 elements per iteration).
    fn bailey_twiddle_multiply(&self, data: &mut [MamaBearScalar], n1: usize, n2: usize) {
        let n = n1 * n2;
        // Row 0: twiddle = ω^{0·j} = 1 for all j → skip
        for i in 1..n1 {
            // ω^i from precomputed table
            let w_i = self.elements_mont[i % n];

            // Build 8 seeds: [ω^{0}, ω^{i}, ω^{2i}, ..., ω^{7i}]
            let mut seeds = [0u64; 8];
            seeds[0] = ONE_MONT;
            seeds[1] = w_i.0;
            for k in 2..8 {
                seeds[k] = MamaBearScalar::mont_mul(seeds[k - 1], w_i.0);
            }
            // step8 = ω^{8i}
            let step8 = MamaBearScalar::mont_mul(seeds[7], w_i.0);
            let step8_packed = PackedMamaBearAVX512::from(step8);

            let mut packed_tw = PackedMamaBearAVX512::from_array(seeds);
            let row_start = i * n2;

            // Packed twiddle multiply: 8 elements per iteration
            let packed_count = n2 / 8;
            for chunk in 0..packed_count {
                let off = row_start + chunk * 8;
                let vals = Self::load_packed(&data[off..]);
                // mont_mul([0,2P) × [0,2P)) → [0,1.5P), con_sub_xp(1) → [0,2P)
                let result = (vals * packed_tw).con_sub_xp(1);
                Self::store_packed(&mut data[off..], result);
                // Advance: packed_tw *= broadcast(step8)
                // mont_mul([0,2P) × [0,2P)) → [0,1.5P), con_sub_xp(1) → [0,2P)
                packed_tw = (packed_tw * step8_packed).con_sub_xp(1);
            }

            // Scalar tail (n2 is always a power of 2 and ≥ 8 when Bailey is used)
            for j in (packed_count * 8)..n2 {
                let tw_idx = ((i as u128 * j as u128) % n as u128) as usize;
                let tw = self.elements_mont[tw_idx];
                let off = row_start + j;
                data[off] =
                    MamaBearScalar(MamaBearScalar::mont_mul(data[off].0, tw.0)).con_sub_xp(1);
            }
        }
    }

    // ── Bailey's 4-Step FFT V2: Column-First + Fused Twiddle ─────────

    /// Optimized Bailey's 4-Step FFT with fused twiddle in column DFT last layer.
    ///
    /// Saves 1 full data pass vs V1 by fusing the twiddle multiply into the column
    /// DFT's last butterfly layer (no separate `bailey_twiddle_multiply` pass).
    /// Input: raw-form coefficients. Output: Montgomery-form evaluations in natural order.
    pub fn fft_natural_bailey_v2(&self, raw_coeffs: &[MamaBearScalar]) -> Vec<MamaBearScalar> {
        let n = self.size();
        assert!(raw_coeffs.len() <= n, "coeff length exceeds domain size");

        let mut data: Vec<MamaBearScalar> = raw_coeffs.iter().map(|x| x.to_montgomery()).collect();
        data.resize(
            n,
            MamaBearScalar(ONE_MONT).lazy_sub(MamaBearScalar(ONE_MONT)),
        );

        if self.log_order <= Self::BAILEY_LOG_THRESHOLD {
            self.fft_in_place_natural(&mut data);
            return data;
        }

        let mut tmp = vec![MamaBearScalar(0); n];
        self.fft_bailey_v2_inplace(&mut data, &mut tmp);
        tmp
    }

    /// In-place optimized Bailey's 4-Step FFT with fused twiddle.
    ///
    /// Input: `data` in Montgomery form, length = self.size(), values in [0, 2P).
    /// Output: `tmp` contains evaluations in Montgomery form, natural order, [0, 2P).
    /// `tmp` is a scratch buffer, must have at least `data.len()` elements.
    ///
    /// Algorithm (column-first, fused twiddle):
    ///   1. Transpose N1×N2 → N2×N1
    ///   2. N2 column DFTs of size N1 with fused twiddle ω_N^{j·k1} in last layer
    ///   3. Transpose N2×N1 → N1×N2
    ///   4. N1 row DFTs of size N2
    ///   5. Digit-reversal (transpose N1×N2 → N2×N1)
    pub fn fft_bailey_v2_inplace(&self, data: &mut [MamaBearScalar], tmp: &mut [MamaBearScalar]) {
        let n = self.size();
        assert_eq!(data.len(), n);
        assert!(tmp.len() >= n);

        let log_n1 = self.log_order / 2; // column DFT size
        let log_n2 = self.log_order - log_n1; // row DFT size
        let n1 = 1usize << log_n1;
        let n2 = 1usize << log_n2;

        let fft_n1 = MamaBearFFT::new(log_n1);
        let fft_n2 = MamaBearFFT::new(log_n2);

        // Step 1: Transpose N1×N2 → N2×N1 (columns become rows)
        Self::block_transpose(data, n1, n2, tmp);

        // Step 2: N2 column DFTs of size N1, with fused twiddle on last layer.
        // Row j of the N2×N1 matrix: DFT of size N1.
        // Global twiddle for row j: tw[k1] = ω_N^{j·k1}, tw_base = ω_N^j.
        // Row 0: twiddle is identity, use standard DIT.
        fft_n1.fft_in_place_natural(&mut tmp[0..n1]);
        for j in 1..n2 {
            let row = &mut tmp[j * n1..(j + 1) * n1];
            let tw_base = self.elements_mont[j]; // ω_N^j
            fft_n1.fft_dit_all_but_last_then_fused_twiddle(row, tw_base);
        }

        // Step 3: Transpose N2×N1 → N1×N2
        Self::block_transpose(tmp, n2, n1, data);

        // Step 4: N1 row DFTs of size N2
        for i in 0..n1 {
            let row = &mut data[i * n2..(i + 1) * n2];
            fft_n2.fft_in_place_natural(row);
        }

        // Step 5: Digit-reversal (N1×N2 → N2×N1)
        Self::digit_reversal(data, n1, n2, tmp);
    }

    /// DIT FFT: all layers except the last, then fused last layer with global twiddle.
    ///
    /// `tw_base` = ω_N^{row_index}, the global twiddle base for this row.
    /// After DIT completes, element at position k1 is also multiplied by tw_base^{k1}.
    fn fft_dit_all_but_last_then_fused_twiddle(
        &self,
        coeff: &mut [MamaBearScalar],
        tw_base: MamaBearScalar,
    ) {
        let n = coeff.len();
        assert_eq!(n, self.size());
        let log_n = self.log_order as usize;

        // Bit-reversal permutation (same as fft_in_place_natural)
        Self::bit_reverse_in_place(coeff);

        if log_n == 0 {
            return;
        }

        // DIT layers 0..log_n-2 (all but last)
        for log_m in 0..(log_n - 1) {
            let m = 1usize << log_m;
            let twiddle_step = n >> (log_m + 1);
            if m >= 8 {
                self.packed_butterfly_layer(coeff, m, twiddle_step);
            } else {
                self.scalar_butterfly_layer(coeff, m, twiddle_step);
            }
        }

        // Last DIT layer (log_m = log_n - 1) with fused global twiddle
        let m = 1usize << (log_n - 1); // = n/2
        let twiddle_step = 1;
        // tw_half = tw_base^{n/2}: twiddle offset for the v-half positions
        let tw_half = Self::mont_pow(tw_base, n / 2);

        if m >= 8 {
            self.fused_last_dit_layer_packed(coeff, m, twiddle_step, tw_base, tw_half);
        } else {
            self.fused_last_dit_layer_scalar(coeff, m, twiddle_step, tw_base, tw_half);
        }
    }

    /// Compute base^exp in Montgomery form via square-and-multiply.
    fn mont_pow(base: MamaBearScalar, exp: usize) -> MamaBearScalar {
        if exp == 0 {
            return MamaBearScalar(ONE_MONT);
        }
        let mut result = MamaBearScalar(ONE_MONT);
        let mut cur = base;
        let mut e = exp;
        while e > 0 {
            if e & 1 == 1 {
                result = MamaBearScalar(MamaBearScalar::mont_mul(result.0, cur.0));
            }
            cur = MamaBearScalar(MamaBearScalar::mont_mul(cur.0, cur.0));
            e >>= 1;
        }
        result
    }

    /// Fused last DIT layer + global twiddle (packed version, m >= 8).
    ///
    /// Standard DIT butterfly at positions (k, k+m):
    ///   t = w * v, u' = u + t, v' = u - t
    /// Fused: also multiply u' by tw[k] and v' by tw[k+m], where
    ///   tw[k] = tw_base^k (global twiddle for this row)
    ///   tw[k+m] = tw[k] * tw_half
    ///
    /// Range: inputs [0, 2P), outputs [0, 2P).
    #[inline]
    fn fused_last_dit_layer_packed(
        &self,
        coeff: &mut [MamaBearScalar],
        m: usize,            // = n/2
        twiddle_step: usize, // = 1 for last layer
        tw_base: MamaBearScalar,
        tw_half: MamaBearScalar,
    ) {
        let tw_half_packed = PackedMamaBearAVX512::from(tw_half.0);

        // Build twiddle recurrence seeds: [tw_base^0, tw_base^1, ..., tw_base^7]
        let mut seeds = [0u64; 8];
        seeds[0] = ONE_MONT;
        seeds[1] = tw_base.0;
        for k in 2..8 {
            seeds[k] = MamaBearScalar::mont_mul(seeds[k - 1], tw_base.0);
        }
        let step8_val = MamaBearScalar::mont_mul(seeds[7], tw_base.0);
        let step8_packed = PackedMamaBearAVX512::from(step8_val);
        let mut packed_tw = PackedMamaBearAVX512::from_array(seeds);

        // Last DIT layer: only one butterfly group (j=0), m pairs.
        debug_assert_eq!(coeff.len(), m * 2);
        for k in (0..m).step_by(8) {
            // Load butterfly twiddle w[k..k+7]
            let mut w_arr = [0u64; 8];
            for i in 0..8 {
                w_arr[i] = self.elements_mont[(k + i) * twiddle_step].0;
            }
            let w_packed = PackedMamaBearAVX512::from_array(w_arr);

            let u = Self::load_packed(&coeff[k..]);
            let v = Self::load_packed(&coeff[k + m..]);

            // Butterfly: t = w * v
            let t = w_packed * v; // [0, 1.5P)
            let u_raw = u.lazy_add(t).con_sub_xp(2); // [0, 2P)
            let v_raw = u.lazy_add_xp(2).lazy_sub(t).con_sub_xp(2); // [0, 2P)

            // Global twiddle: tw_u for positions k..k+7, tw_v = tw_u * tw_half
            let tw_v = (packed_tw * tw_half_packed).con_sub_xp(1); // [0, 2P)

            let u_out = (u_raw * packed_tw).con_sub_xp(1); // [0, 1.5P) → [0, 2P)
            let v_out = (v_raw * tw_v).con_sub_xp(1); // [0, 1.5P) → [0, 2P)

            Self::store_packed(&mut coeff[k..], u_out);
            Self::store_packed(&mut coeff[k + m..], v_out);

            // Advance twiddle recurrence
            packed_tw = (packed_tw * step8_packed).con_sub_xp(1);
        }
    }

    /// Fused last DIT layer + global twiddle (scalar version, m < 8).
    #[inline]
    fn fused_last_dit_layer_scalar(
        &self,
        coeff: &mut [MamaBearScalar],
        m: usize,
        twiddle_step: usize,
        tw_base: MamaBearScalar,
        tw_half: MamaBearScalar,
    ) {
        debug_assert_eq!(coeff.len(), m * 2);
        let mut tw_k = MamaBearScalar(ONE_MONT);
        for k in 0..m {
            let w = self.elements_mont[k * twiddle_step];
            let u = coeff[k];
            let v = coeff[k + m];
            let t = MamaBearScalar(MamaBearScalar::mont_mul(w.0, v.0)); // [0, 1.5P)
            let u_raw = u.lazy_add(t).con_sub_xp(2); // [0, 2P)
            let v_raw = u.lazy_add_xp(2).lazy_sub(t).con_sub_xp(2); // [0, 2P)

            // Global twiddle
            let tw_v_k = MamaBearScalar(MamaBearScalar::mont_mul(tw_k.0, tw_half.0)).con_sub_xp(1); // [0, 2P)
            coeff[k] = MamaBearScalar(MamaBearScalar::mont_mul(u_raw.0, tw_k.0)).con_sub_xp(1); // [0, 2P)
            coeff[k + m] =
                MamaBearScalar(MamaBearScalar::mont_mul(v_raw.0, tw_v_k.0)).con_sub_xp(1); // [0, 2P)

            // Advance tw_k
            tw_k = MamaBearScalar(MamaBearScalar::mont_mul(tw_k.0, tw_base.0)).con_sub_xp(1);
            // [0, 2P)
        }
    }

    /// Block matrix transpose: src[n1×n2] → dst[n2×n1], using 8×8 blocks for cache locality.
    fn block_transpose(src: &[MamaBearScalar], n1: usize, n2: usize, dst: &mut [MamaBearScalar]) {
        const BLK: usize = 8;
        // Process full 8×8 blocks
        let n1_full = n1 / BLK * BLK;
        let n2_full = n2 / BLK * BLK;

        for bi in (0..n1_full).step_by(BLK) {
            for bj in (0..n2_full).step_by(BLK) {
                // Transpose 8×8 block
                for di in 0..BLK {
                    for dj in 0..BLK {
                        dst[(bj + dj) * n1 + bi + di] = src[(bi + di) * n2 + bj + dj];
                    }
                }
            }
            // Right edge (columns n2_full..n2)
            if n2_full < n2 {
                let cols_rem = n2 - n2_full;
                for di in 0..BLK {
                    for dj in 0..cols_rem {
                        dst[(n2_full + dj) * n1 + bi + di] = src[(bi + di) * n2 + n2_full + dj];
                    }
                }
            }
        }
        // Bottom edge (rows n1_full..n1)
        if n1_full < n1 {
            let rows_rem = n1 - n1_full;
            for bj in (0..n2).step_by(BLK) {
                let cols = BLK.min(n2 - bj);
                for di in 0..rows_rem {
                    for dj in 0..cols {
                        dst[(bj + dj) * n1 + n1_full + di] = src[(n1_full + di) * n2 + bj + dj];
                    }
                }
            }
        }
    }
}

use crate::field::FftField;

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::SmallRng;
    use rand::{RngCore, SeedableRng};

    #[test]
    fn test_fft_roundtrip() {
        let log_n = 5;
        let fft = MamaBearFFT::new(log_n);
        let n = fft.size();
        let mut rng = SmallRng::seed_from_u64(42);

        // Generate random coefficients in [0, P)
        let raw_coeffs: Vec<MamaBearScalar> =
            (0..n).map(|_| MamaBearScalar(rng.next_u64() % P)).collect();

        // FFT then IFFT
        let mut evals = fft.fft(&raw_coeffs);
        fft.ifft_in_place(&mut evals);

        // Convert back to raw and compare
        for (i, val) in evals.iter().enumerate() {
            let recovered = val.from_montgomery();
            assert_eq!(
                recovered.0, raw_coeffs[i].0,
                "Mismatch at index {}: got {}, expected {}",
                i, recovered.0, raw_coeffs[i].0
            );
        }
    }

    #[test]
    fn test_fft_roundtrip_with_padding() {
        let log_n = 5;
        let fft = MamaBearFFT::new(log_n);
        let n = fft.size();
        let mut rng = SmallRng::seed_from_u64(123);

        // Coefficients shorter than domain
        let raw_coeffs: Vec<MamaBearScalar> = (0..n / 2)
            .map(|_| MamaBearScalar(rng.next_u64() % P))
            .collect();

        let mut evals = fft.fft(&raw_coeffs);
        fft.ifft_in_place(&mut evals);

        // First half should match, second half should be zero
        for (i, val) in evals.iter().enumerate() {
            let recovered = val.from_montgomery();
            let expected = if i < raw_coeffs.len() {
                raw_coeffs[i].0
            } else {
                0
            };
            assert_eq!(
                recovered.0, expected,
                "Mismatch at index {}: got {}, expected {}",
                i, recovered.0, expected
            );
        }
    }

    #[test]
    fn test_fft_zero_padded_pow2_matches_explicit_padding() {
        let mut rng = SmallRng::seed_from_u64(321);

        for log_n in 4..=12 {
            let fft = MamaBearFFT::new(log_n);
            let n = fft.size();

            for pad_bits in 1..=(log_n as usize - 1).min(4) {
                let raw_len = n >> pad_bits;
                let raw_coeffs: Vec<MamaBearScalar> = (0..raw_len)
                    .map(|_| MamaBearScalar(rng.next_u64() % P))
                    .collect();

                let fast = fft.fft(&raw_coeffs);

                let mut explicit = vec![MamaBearScalar(0); n];
                explicit[..raw_len].copy_from_slice(&raw_coeffs);
                let baseline = fft.fft(&explicit);

                assert_eq!(
                    fast, baseline,
                    "log_n={log_n}, raw_len={raw_len}: zero-padding fast path mismatch"
                );
            }
        }
    }

    #[test]
    fn test_fft_polynomial_eval() {
        // Verify DIF FFT computes polynomial evaluations correctly under the
        // pair-major blocked output layout.
        // f(x) = 3 + 5x + 7x^2 + 11x^3
        let log_n = 3; // n = 8
        let fft = MamaBearFFT::new(log_n);
        let coeffs = vec![
            MamaBearScalar(3),
            MamaBearScalar(5),
            MamaBearScalar(7),
            MamaBearScalar(11),
        ];

        let evals = fft.fft(&coeffs);

        for pair_idx in 0..(fft.size() >> 1) {
            let natural_idx = MamaBearFFT::reverse_index(pair_idx, log_n as usize - 1);
            let x = fft.element_at(natural_idx);
            let nx = fft.element_at(natural_idx + (fft.size() >> 1));
            let c0 = MamaBearScalar(3).to_montgomery();
            let c1 = MamaBearScalar(5).to_montgomery();
            let c2 = MamaBearScalar(7).to_montgomery();
            let c3 = MamaBearScalar(11).to_montgomery();

            let mut direct_x = c3;
            direct_x = MamaBearScalar(MamaBearScalar::mont_mul(direct_x.0, x.0))
                .lazy_add(c2)
                .con_sub_xp(2);
            direct_x = MamaBearScalar(MamaBearScalar::mont_mul(direct_x.0, x.0))
                .lazy_add(c1)
                .con_sub_xp(2);
            direct_x = MamaBearScalar(MamaBearScalar::mont_mul(direct_x.0, x.0))
                .lazy_add(c0)
                .con_sub_xp(2);

            let mut direct_nx = c3;
            direct_nx = MamaBearScalar(MamaBearScalar::mont_mul(direct_nx.0, nx.0))
                .lazy_add(c2)
                .con_sub_xp(2);
            direct_nx = MamaBearScalar(MamaBearScalar::mont_mul(direct_nx.0, nx.0))
                .lazy_add(c1)
                .con_sub_xp(2);
            direct_nx = MamaBearScalar(MamaBearScalar::mont_mul(direct_nx.0, nx.0))
                .lazy_add(c0)
                .con_sub_xp(2);

            let (x_pos, nx_pos) = fft.pair_storage_positions(pair_idx);
            assert_eq!(
                evals[x_pos].reduce().0,
                direct_x.reduce().0,
                "DIF FFT x mismatch at pair slot {}",
                pair_idx,
            );
            assert_eq!(
                evals[nx_pos].reduce().0,
                direct_nx.reduce().0,
                "DIF FFT -x mismatch at pair slot {}",
                pair_idx,
            );
        }
    }

    #[test]
    fn test_fft_natural_polynomial_eval() {
        // Verify DIT (natural order) FFT for backward compatibility.
        let log_n = 3;
        let fft = MamaBearFFT::new(log_n);
        let coeffs = vec![
            MamaBearScalar(3),
            MamaBearScalar(5),
            MamaBearScalar(7),
            MamaBearScalar(11),
        ];

        let evals = fft.fft_natural(&coeffs);

        for i in 0..8 {
            let x = fft.element_at(i);
            let c0 = MamaBearScalar(3).to_montgomery();
            let c1 = MamaBearScalar(5).to_montgomery();
            let c2 = MamaBearScalar(7).to_montgomery();
            let c3 = MamaBearScalar(11).to_montgomery();
            let mut result = c3;
            result = MamaBearScalar(MamaBearScalar::mont_mul(result.0, x.0))
                .lazy_add(c2)
                .con_sub_xp(2);
            result = MamaBearScalar(MamaBearScalar::mont_mul(result.0, x.0))
                .lazy_add(c1)
                .con_sub_xp(2);
            result = MamaBearScalar(MamaBearScalar::mont_mul(result.0, x.0))
                .lazy_add(c0)
                .con_sub_xp(2);

            assert_eq!(
                evals[i].reduce().0,
                result.reduce().0,
                "Natural FFT eval mismatch at index {}",
                i
            );
        }
    }

    #[test]
    fn test_fft_various_sizes() {
        let mut rng = SmallRng::seed_from_u64(999);
        // Test for sizes 2^3 through 2^14
        for log_n in 3..=14 {
            let fft = MamaBearFFT::new(log_n);
            let n = fft.size();
            let raw_coeffs: Vec<MamaBearScalar> =
                (0..n).map(|_| MamaBearScalar(rng.next_u64() % P)).collect();

            let mut evals = fft.fft(&raw_coeffs);
            fft.ifft_in_place(&mut evals);

            for (i, val) in evals.iter().enumerate() {
                let recovered = val.from_montgomery();
                assert_eq!(
                    recovered.0, raw_coeffs[i].0,
                    "Size 2^{}: mismatch at index {}",
                    log_n, i
                );
            }
        }
    }

    #[test]
    fn test_twiddle_elements() {
        let fft = MamaBearFFT::new(5);
        let n = fft.size();

        // omega^n should be 1 in Montgomery form
        let omega_n = MamaBearScalar(MamaBearScalar::mont_mul(
            fft.elements_mont[n - 1].0,
            fft.omega_mont.0,
        ));
        let one_mont = MamaBearScalar(ONE_MONT);
        assert_eq!(omega_n.reduce().0, one_mont.reduce().0, "omega^n != 1");

        // element_inv_at(i) * element_at(i) = 1
        for i in 0..n {
            let prod = MamaBearScalar(MamaBearScalar::mont_mul(
                fft.element_at(i).0,
                fft.element_inv_at(i).0,
            ));
            assert_eq!(
                prod.reduce().0,
                one_mont.reduce().0,
                "element({})*element_inv({}) != 1",
                i,
                i
            );
        }
    }

    #[test]
    fn test_bailey_fft_matches_standard() {
        let mut rng = SmallRng::seed_from_u64(42);
        // Test: Bailey output should match standard fft_natural for various sizes.
        // log_n = 10..18 covers both below and above the Bailey threshold (13).
        for log_n in 10..=18 {
            let fft = MamaBearFFT::new(log_n);
            let n = fft.size();
            let raw_coeffs: Vec<MamaBearScalar> =
                (0..n).map(|_| MamaBearScalar(rng.next_u64() % P)).collect();

            let standard = fft.fft_natural(&raw_coeffs);
            let bailey = fft.fft_natural_bailey(&raw_coeffs);

            for i in 0..n {
                assert_eq!(
                    standard[i].reduce().0,
                    bailey[i].reduce().0,
                    "log_n={}: mismatch at index {}",
                    log_n,
                    i
                );
            }
        }
    }

    #[test]
    fn test_bailey_fft_with_padding() {
        let mut rng = SmallRng::seed_from_u64(123);
        // Coefficients shorter than domain
        for log_n in 14..=16 {
            let fft = MamaBearFFT::new(log_n);
            let n = fft.size();
            let raw_coeffs: Vec<MamaBearScalar> = (0..n / 4)
                .map(|_| MamaBearScalar(rng.next_u64() % P))
                .collect();

            let standard = fft.fft_natural(&raw_coeffs);
            let bailey = fft.fft_natural_bailey(&raw_coeffs);

            for i in 0..n {
                assert_eq!(
                    standard[i].reduce().0,
                    bailey[i].reduce().0,
                    "log_n={} (padded): mismatch at index {}",
                    log_n,
                    i
                );
            }
        }
    }

    #[test]
    fn test_bailey_pair_major_zero_padded_matches_explicit_padding() {
        let mut rng = SmallRng::seed_from_u64(20260329);

        for log_n in 14..=16 {
            let fft = MamaBearFFT::new(log_n);
            let n = fft.size();
            let raw_len = n >> 3;
            let raw_coeffs: Vec<MamaBearScalar> = (0..raw_len)
                .map(|_| MamaBearScalar(rng.next_u64() % P))
                .collect();

            let fast = fft.fft_bailey_pair_major(&raw_coeffs);

            let mut explicit = vec![MamaBearScalar(0); n];
            explicit[..raw_len].copy_from_slice(&raw_coeffs);
            let baseline = fft.fft(&explicit);

            assert_eq!(
                fast, baseline,
                "Bailey pair-major padded mismatch for log_n={log_n}"
            );
        }
    }

    #[test]
    fn test_bailey_v2_matches_standard() {
        let mut rng = SmallRng::seed_from_u64(42);
        // Test: Bailey V2 (row-first + fused twiddle) should match standard fft_natural.
        for log_n in 10..=18 {
            let fft = MamaBearFFT::new(log_n);
            let n = fft.size();
            let raw_coeffs: Vec<MamaBearScalar> =
                (0..n).map(|_| MamaBearScalar(rng.next_u64() % P)).collect();

            let standard = fft.fft_natural(&raw_coeffs);
            let bailey_v2 = fft.fft_natural_bailey_v2(&raw_coeffs);

            for i in 0..n {
                assert_eq!(
                    standard[i].reduce().0,
                    bailey_v2[i].reduce().0,
                    "Bailey V2 log_n={}: mismatch at index {}",
                    log_n,
                    i
                );
            }
        }
    }

    #[test]
    fn test_bailey_v2_matches_v1() {
        let mut rng = SmallRng::seed_from_u64(99);
        // Bailey V2 should match Bailey V1 for sizes above threshold.
        for log_n in 14..=16 {
            let fft = MamaBearFFT::new(log_n);
            let n = fft.size();
            let raw_coeffs: Vec<MamaBearScalar> =
                (0..n).map(|_| MamaBearScalar(rng.next_u64() % P)).collect();

            let v1 = fft.fft_natural_bailey(&raw_coeffs);
            let v2 = fft.fft_natural_bailey_v2(&raw_coeffs);

            for i in 0..n {
                assert_eq!(
                    v1[i].reduce().0,
                    v2[i].reduce().0,
                    "Bailey V1 vs V2 log_n={}: mismatch at index {}",
                    log_n,
                    i
                );
            }
        }
    }
}
