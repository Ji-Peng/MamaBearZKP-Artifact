use std::mem::size_of;

use arithmetic::field::Field;
use sha2::{Digest, Sha256};

const HASH_SIZE: usize = 32;

#[derive(Debug, Clone, Default)]
pub struct Proof {
    idx: usize,
    pub bytes: Vec<u8>,
}

impl Proof {
    #[inline(always)]
    pub fn append_u8_slice(&mut self, buffer: &[u8], size: usize) {
        self.bytes.extend_from_slice(&buffer[..size]);
    }

    #[inline(always)]
    fn step(&mut self, size: usize) {
        self.idx += size;
    }

    #[inline(always)]
    pub fn get_next_and_step<F: Field>(&mut self) -> F {
        let ret = F::deserialize_from(&self.bytes[self.idx..(self.idx + F::SIZE)]);
        self.step(F::SIZE);
        ret
    }

    pub fn get_next_hash(&mut self) -> [u8; HASH_SIZE] {
        let ret = self.bytes[self.idx..(self.idx + HASH_SIZE)]
            .try_into()
            .unwrap();
        self.step(HASH_SIZE);
        ret
    }

    pub fn get_next_slice(&mut self, len: usize) -> Vec<u8> {
        let ret = self.bytes[self.idx..(self.idx + len)].to_vec();
        self.step(len);
        ret
    }

    pub fn get_next_u64(&mut self) -> u64 {
        let bytes: [u8; 8] = self.bytes[self.idx..(self.idx + 8)].try_into().unwrap();
        self.step(8);
        u64::from_le_bytes(bytes)
    }
}

#[derive(Debug, Clone, Default)]
pub struct SHA256hasher;

impl SHA256hasher {
    pub fn hash(&self, output: &mut [u8], input: &[u8], input_len: usize) {
        let hashed = Sha256::digest(&input[..input_len]);
        output.copy_from_slice(&hashed[..]);
    }
    pub fn hash_inplace(&self, buffer: &mut [u8], input_len: usize) {
        let hashed = Sha256::digest(&buffer[..input_len]);
        buffer.copy_from_slice(&hashed[..]);
    }
    /// `digest <- SHA256(digest || input)`. Feeding the previous digest back in is
    /// what makes the transcript a chain: without it a challenge would depend only
    /// on `input` and carry no binding to anything absorbed earlier.
    pub fn hash_chained(&self, digest: &mut [u8], input: &[u8]) {
        let mut hasher = Sha256::new();
        hasher.update(&*digest);
        hasher.update(input);
        digest.copy_from_slice(&hasher.finalize()[..]);
    }
}

#[derive(Clone)]
pub struct Transcript {
    pub hasher: SHA256hasher,
    hash_start_idx: usize,
    digest: [u8; Self::DIGEST_SIZE],
    pub proof: Proof,
}

impl Default for Transcript {
    fn default() -> Self {
        Self::new()
    }
}

impl Transcript {
    pub const DIGEST_SIZE: usize = 32;

    /// `digest_i = SHA256(digest_{i-1} || bytes absorbed since the last challenge)`.
    ///
    /// The previous digest MUST stay an input here: it is the only thing making a
    /// challenge depend on the whole transcript prefix rather than just the bytes
    /// since the previous draw. Absorbing nothing degenerates to `digest <- H(digest)`,
    /// which is how repeated draws (wide `challenge_f`, `challenge_usizes`) advance.
    fn hash_to_digest(&mut self) {
        let hash_end_idx = self.proof.bytes.len();
        self.hasher
            .hash_chained(&mut self.digest, &self.proof.bytes[self.hash_start_idx..]);
        self.hash_start_idx = hash_end_idx;
    }

    #[inline]
    pub fn new() -> Self {
        Transcript {
            hasher: SHA256hasher,
            hash_start_idx: 0,
            digest: [0u8; Self::DIGEST_SIZE],
            proof: Proof::default(),
        }
    }

    pub fn append_f<F: Field>(&mut self, f: F) {
        let cur_size = self.proof.bytes.len();
        self.proof.bytes.resize(cur_size + F::SIZE, 0);
        f.serialize_into(&mut self.proof.bytes[cur_size..]);
    }

    pub fn append_u8_slice(&mut self, buffer: &[u8], size: usize) {
        self.proof.append_u8_slice(buffer, size);
    }

    pub fn challenge_f<F: Field>(&mut self) -> F {
        let needed = F::uniform_bytes_needed();
        if needed <= Self::DIGEST_SIZE {
            // Historical fast path: exactly one `hash_to_digest()` producing 32 bytes,
            // then a wide draw over the first `needed` bytes. When `needed == 32` this
            // is byte-for-byte identical to the pre-change `from_uniform_bytes(&digest)`
            // flow (the default `from_uniform_bytes_wide` copies all 32 bytes and
            // delegates to `from_uniform_bytes`).
            self.hash_to_digest();
            F::from_uniform_bytes_wide(&self.digest[..needed])
        } else {
            // Wide draw: gather `ceil(needed / 32)` fresh digests (32 bytes each) by
            // repeated `hash_to_digest()`, concatenate, then draw from the buffer.
            let mut buf = Vec::with_capacity(needed);
            while buf.len() < needed {
                self.hash_to_digest();
                buf.extend_from_slice(&self.digest);
            }
            buf.truncate(needed);
            F::from_uniform_bytes_wide(&buf)
        }
    }

    pub fn challenge_usizes(&mut self, num: usize) -> Vec<usize> {
        (0..num)
            .map(|_| {
                self.hash_to_digest();
                usize::from_be_bytes(self.digest[0..size_of::<usize>()].try_into().unwrap())
            })
            .collect()
    }

    // FRI grinding (proof-of-work). The current transcript digest is used as
    // the BLAKE3 seed for the PoW search. The witness is appended to the
    // transcript as 8 LE bytes so subsequent challenges stay in sync with the
    // verifier. `bits == 0` is a no-op: nothing is hashed and nothing is
    // appended, preserving transcripts for PCS params that leave grinding off.

    fn grinding_seed(&mut self) -> [u8; Self::DIGEST_SIZE] {
        self.hash_to_digest();
        self.digest
    }

    /// Prover-side serial grinding. Returns the found witness.
    pub fn grind(&mut self, bits: u32) -> u64 {
        if bits == 0 {
            return 0;
        }
        let seed = self.grinding_seed();
        let witness = crate::grinding::grind_blake3_serial(seed, bits);
        self.proof
            .append_u8_slice(&witness.to_le_bytes(), 8);
        witness
    }

    /// Prover-side parallel grinding. Returns the found witness.
    pub fn grind_par(&mut self, bits: u32) -> u64 {
        if bits == 0 {
            return 0;
        }
        let seed = self.grinding_seed();
        let witness = crate::grinding::grind_blake3_par(seed, bits);
        self.proof
            .append_u8_slice(&witness.to_le_bytes(), 8);
        witness
    }

    /// Verifier-side grinding check. Reads 8 bytes of witness from the
    /// external proof stream, mirrors them into the transcript (so subsequent
    /// challenges match the prover's), and returns whether the PoW is valid.
    pub fn verify_grind(&mut self, proof: &mut Proof, bits: u32) -> bool {
        if bits == 0 {
            return true;
        }
        let seed = self.grinding_seed();
        let witness = proof.get_next_u64();
        let ok = crate::grinding::check_blake3(seed, witness, bits);
        self.proof
            .append_u8_slice(&witness.to_le_bytes(), 8);
        ok
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arithmetic::field::mamabear::MamaBearScalar;

    /// For MamaBear (`uniform_bytes_needed() == 32`), `challenge_f` must do EXACTLY one
    /// `hash_to_digest()` then `from_uniform_bytes(&digest)`. We reproduce that flow
    /// independently, including the digest chaining: the transcript starts at a
    /// zero digest, so the first draw is `SHA256(0^32 || data)`.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn challenge_f_mamabear_matches_single_digest_flow() {
        use arithmetic::field::mamabear::MamaBearScalar;
        let data: &[u8] = b"mamabear-fs-pin-2026-plan02-D2-8";
        let mut t = Transcript::new();
        t.append_u8_slice(data, data.len());
        let got: MamaBearScalar = t.challenge_f();

        let mut digest = [0u8; 32];
        let mut hasher = Sha256::new();
        hasher.update(digest);
        hasher.update(data);
        digest.copy_from_slice(&hasher.finalize()[..]);
        let want = MamaBearScalar::from_uniform_bytes(&digest);
        assert_eq!(got, want, "MamaBear challenge_f drifted from single-digest flow");

        // Regression pin: the serialized challenge bytes stay constant across runs.
        let mut got_bytes = [0u8; MamaBearScalar::SIZE];
        got.serialize_into(&mut got_bytes);
        const PINNED: [u8; 7] = [236, 128, 130, 240, 240, 106, 1];
        assert_eq!(got_bytes, PINNED, "MamaBear challenge_f value pin changed");
    }

    /// The transcript is a CHAIN: a challenge must depend on everything absorbed before
    /// it, not just on the bytes since the previous draw. Absorbing different history and
    /// then the SAME suffix must give a different follow-up challenge.
    ///
    /// This is the direct regression guard for the unchained-digest defect, where
    /// `hash_to_digest` hashed only `proof.bytes[hash_start_idx..]` and OVERWROTE the
    /// digest, so `c2` below collided across differing prefixes.
    #[test]
    fn challenge_f_binds_the_whole_transcript_prefix() {
        fn run(prefix: &[u8]) -> (MamaBearScalar, MamaBearScalar) {
            let mut t = Transcript::new();
            t.append_u8_slice(prefix, prefix.len());
            let c1: MamaBearScalar = t.challenge_f();
            t.append_u8_slice(b"ZZZZ", 4);
            let c2: MamaBearScalar = t.challenge_f();
            (c1, c2)
        }

        let (c1a, c2a) = run(b"AAAA");
        let (c1b, c2b) = run(b"BBBB");

        assert_ne!(c1a.0, c1b.0, "sanity: differing prefixes must give different c1");
        assert_ne!(
            c2a.0, c2b.0,
            "UNCHAINED transcript: c2 ignored the prefix and bound only the suffix"
        );
    }
}
