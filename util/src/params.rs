//! Centralized FRI / circuit constants shared across HyperPlonk + DeepFold
//! benches and profile binaries.
//!
//! Grouped by scope:
//! - Field-agnostic values at the module root.
//! - Per-circuit gate counts under [`gates`].
//! - Field-specific FRI parameters under [`mamabear`], [`babybear`],
//!   [`goldilocks`]. Security-level suffixes:
//!   - `CONJ96` — conjectured 96-bit soundness (this one IS a total).
//!   - `PROV_QUERY97` / `PROV_QUERY128` — provable soundness on the QUERY path
//!     only, i.e. `S_query`, NOT the total provable figure. See the ledger
//!     below: the total is `S_prov = min(C, S_query)`, and for MamaBear Ext3 it
//!     is the commit term C that BINDS, at **107.6 / 106.5** for nv = 19 / 20.
//!     These were named `PROV97` / `PROV128` until 2026-07-17; that name read
//!     as "provable 128-bit" and misled at least one reader (see HISTORY
//!     below), hence the explicit `_QUERY`.
//!
//!     SUPERSEDED: this line previously read "C BINDS at ~81-83", which is the
//!     figure from the older `O(n^2)` bad-challenge count and contradicted the
//!     ledger table further down in this same file. The commit term is now
//!     taken from the linear-in-`n` proximity gap inside the Johnson radius,
//!     which is worth `log2(n) + 2.61` bits; `81-83` must not be quoted. An
//!     external reviewer read the stale line and correctly flagged it against
//!     the table, which is what prompted this correction — a header summarizing
//!     a table is exactly the place a superseded number survives longest.
//!   - `GRINDING_BITS_EXT3_PROV_QUERY128` (MamaBear only) — PoW grinding bits
//!     (zeta_q) added to the query path.
//!
//! # Which decoding radius each suffix assumes (rate 1/8, `CODE_RATE_LOG = 3`)
//!
//! Read it before touching
//! any number here. Do NOT re-derive this ledger from first principles — the
//! idealized textbook radii give answers that are wrong by tens of bits.
//!
//! Write `R = CODE_RATE_LOG = 3` (so rate = 2^-R = 1/8) and `s` = query count.
//!
//! ## Conjectured (ethSTARK), the `CONJ*` suffix
//!
//!   S_conj = min(zeta_q + R*s, log2|F_ext|)          <- note the FIELD CAP
//!
//! then MINUS the Fenzi-Sanso list-size penalty (ePrint 2025/2197), which is
//! ~2-4 bits for MamaBear^3 and ~10-12 bits for BabyBear^4.
//!
//! ## Provable (Johnson-radius analysis), the `PROV*` suffix
//!
//! The provable ledger has TWO independent terms and the total is their min:
//!
//!   Q(R,s,m) = s * (R/2 - log2(1 + 1/(2m)))          = s * 1.2776  at R=3, m=3
//!   C        = log2|F_ext| - log2(n) - log2(log2 n)
//!              - log2(c_PG)                          <- commit term, n = FRI domain
//!   S_prov   = min(zeta_c + C, zeta_q + Q)
//!
//! We ship (zeta_c, zeta_q) = (0, 16), so S_prov = min(C, 16 + Q).
//!
//! QUERY term, from Ben-Sasson-Carmon-Ishai-Riabzev-Saraf 2020 Thm 8.3 with
//! Haboeck 2022 Thm 1: the analysis needs a slack parameter m >= 3, and the
//! Johnson radius it actually gives is `1 - sqrt(rate)*(1 + 1/(2m))`, NOT the
//! idealized `1 - sqrt(rate)`. You cannot take m -> infinity to recover 1.5
//! bits/query, because C pays for large m too (see below).
//!
//! COMMIT term, from Ben-Sasson-Carmon-Haboeck-Kopparty-Saraf, STOC 2026,
//! Thm 1.5: strictly inside the Johnson radius the number of bad folding
//! challenges is `O(n)`, not `O(n^2)`, with proximity loss 0 and an explicit
//! constant
//!
//!   c_PG = (2*(m + 1/2)^5 + 3*(m + 1/2)*delta*rho) / (3*rho^(3/2))
//!        = 7928.7  (i.e. 12.95 bits)  at rho = 1/8, m = 3
//!
//! where `eta = sqrt(rho)/(2m)` and `delta = 1 - sqrt(rho) - eta`. This
//! SUPERSEDES the `O(n^2)` count previously used here (BCIKS FOCS 2020 +
//! Haboeck 2022), whose commit term was
//!
//!   C_old = log2|F_ext| - 2*log2(n) - log2(log2 n)
//!           - 7*log2(m + 1/2) + log2(3) - 3R/2
//!
//! The switch costs nothing — no protocol, code or proof-byte change, only the
//! cited theorem — and is worth `log2(n) + 2.61` bits, about +24.6 at nv=19.
//! It also halves the slope: C now falls ~1 bit per +1 nv, not ~2. Under the
//! old count m was penalised as `(m+1/2)^7`; it is now `(m+1/2)^5`, but m = 3
//! remains optimal because the commit term still binds for MamaBear.
//!
//! CAVEAT, recorded here so the constant is auditable from the source alone:
//! the STOC version states Thm 1.5 for the LINE case. This PCS commits a random
//! linear combination of several polynomials, so it strictly needs the
//! affine-space (batched) version, whose constant is to be confirmed against the
//! full version. The 7928.7 above is the line-case constant.
//!
//! ## What our constants actually evaluate to
//!
//! MamaBear Ext3 (log2|F_ext| = 147), the configuration these constants ship for:
//!
//! | constant        |  s |            S_query |   C (nv=19/20) |        S_prov |
//! |-----------------|---:|-------------------:|---------------:|--------------:|
//! | `CONJ96`        | 32 | 0 + 32*1.2776=40.9 | 107.6 / 106.5  |          40.9 |
//! | `PROV_QUERY128` | 88 | 16 + 112.4 = 128.4 | 107.6 / 106.5  | 107.6 / 106.5 |
//!
//! CRITICAL: **`PROV_QUERY128` names the QUERY-PATH target `S_query`, not
//! `S_prov`** — which is exactly what the name now says. C is structural: no
//! number of queries and no amount of grinding lifts it, so the repo sets a
//! target on S_query and reports C beside it rather than folding them into one
//! number. For MamaBear the honest full provable figure is
//! `S_prov = min(C, S_query) = 107.6` at nv=19, and it is COMMIT-bound.
//!
//! ## The ledger assumes UNIFORM challenges; the sampler is not exactly uniform
//!
//! Every bound above is stated for challenges drawn uniformly from the
//! extension field. The implemented sampler does not achieve that exactly, and
//! the gap is small but it is not zero, so it belongs in the ledger rather than
//! only in the field code.
//!
//! `MamaBearScalar::from_uniform_bytes` reduces one little-endian `u64` mod
//! `P`. Writing `2^64 = qP + r`, residues below `r` get `q+1` preimages and
//! the rest get `q`. Because `P = 2^49 - 2^34 + 1` is Solinas we have
//! `r = 2^64 mod P = 2^34 - 2^15 - 1`, so `r/P = 2^-15.00` — large for a
//! 49-bit prime, and forced by the same sparsity that makes reduction cheap.
//!
//! The measure to apply is the POINTWISE probability ratio, not the total
//! variation distance:
//!
//! ```text
//!   max ratio = 1 + (P-r)/2^64 = 1 + 2^-15.00      <- use this for soundness
//!   min ratio = 1 -  r   /2^64 = 1 - 2^-30.00
//!   TV        = r(P-r)/(2^64 P) = 2^-30.000        <- use this ONLY for
//!                                                     distinguishing claims
//! ```
//!
//! Reading TV additively would say a `2^-100` cheating probability becomes
//! `2^-30`. That bound is valid and badly loose: the bias is diffuse, so the
//! multiplicative bound applies instead, and for `k` independent components
//! `Pr_biased[B] <= (1 + 2^-15)^k Pr_uniform[B]` for every event `B`. The bound
//! is tight (an adversary whose bad set sits inside `[0, r)` attains it).
//!
//! WHICH TERMS OF THIS LEDGER MOVE, and it is not all of them:
//!
//! - `S_query` — does NOT move. Query positions come from `challenge_usizes`
//!   and are reduced `% (fat_domain >> 1)` where `fat_domain = 1 << log_order`;
//!   a power-of-two modulus is an exact truncation of uniform bytes. Grinding
//!   is a proof of work over the digest, also not a field draw. So the
//!   `zeta_q + s * 1.2776` term is exactly as stated.
//! - `C` — moves. The FRI folding challenges, the batching challenge and the
//!   out-of-domain `alpha` are all field draws, so the commit-side bound picks
//!   up the multiplicative factor.
//!
//! Measured by instrumenting the transcript over a full prove of the add/mul
//! SNARK: `nv = 20` draws 315 Ext3 challenges, and each Ext3 draw consumes
//! THREE base components (bytes 0..8, 8..16, 16..24 of one digest), so
//! `k = 945` and the factor is `(1 + 2^-15)^945 = 1.0293`, i.e. **0.042 bit**.
//! The count does not depend on the query count — `query_num` feeds only
//! `challenge_usizes`, never a `challenge_f` loop — so it applies unchanged at
//! the shipped 88 queries.
//!
//! Charging that nv = 20 budget against the nv = 19 figure OVER-states the loss
//! (nv = 19 draws fewer challenges), and even so `S_prov = C = 107.6` becomes
//! 107.558, which still prints as 107.6. No number in this file and no number
//! under `results/` moves. Reaching a full bit of loss would take about `2^15`
//! components in one transcript, 35x more than the protocol draws.
//!
//! Two caveats worth carrying rather than rediscovering:
//!
//! 1. The unit is the COMPONENT, not the draw. Counting the 315 Ext3 draws
//!    instead of the 945 components under-states the exponent threefold.
//! 2. The 0.042 bit is a SOUNDNESS figure and depends on this system being a
//!    succinct argument with no zero-knowledge claim. A distinguishing claim
//!    needs the additive `k * TV = 945 * 2^-30 = 2^-20.1` instead, which would
//!    cap statistical indistinguishability far below any `2^-100` target.
//!    Widen the draw before adding blinding, never after.
//!
//! The fix is cheap and is documented at the call site: the function already
//! receives 32 uniform bytes and consumes 8, so 10 bytes per component still
//! fits three components in one digest — no extra hashing — and takes the ratio
//! to `1 + 2^-31`. It is deliberately not applied here, because changing a
//! challenge value changes the Fiat-Shamir chain, the query indices, the
//! Merkle-path deduplication and therefore the recorded proof sizes, in
//! exchange for a fraction of a bit that does not survive rounding.
//!
//! The binding side is NOT universal — check it per backend and per nv rather
//! than assuming. A backend with a LARGER extension field can be QUERY-bound at
//! the same nv where MamaBear is COMMIT-bound (more queries WOULD lift it), and
//! only return to commit-bound at larger nv. The `bench_config` binary computes
//! and labels this per nv for every result-file header.
//!
//! Conjectured side for `PROV_QUERY128`: min(16 + 3*88, 147) = 147 (field cap
//! binds, NOT 280), minus Fenzi-Sanso -> ~143-145.
//!
//! HISTORY (2026-07-17): this table previously claimed "Johnson = 1.5
//! bits/query, so 88*1.5 = 132 >= 128". Both halves were wrong — 1.5 is the
//! m -> infinity idealization (real: 1.2776 at m=3), and it ignored the commit
//! term C entirely. That error was made precisely because the old name
//! `PROV128` was read as "provable 128-bit". The constant 88 is fine; the
//! justification was not, and the name has been changed to say what it means.
//!
//! # Which radius our IMPLEMENTED protocol supports
//!
//! The Johnson-radius ledger above applies to the protocol as implemented: the
//! PCS performs the full DeepFold out-of-domain binding — it draws an OOD
//! challenge `alpha`, appends `c = f^(0)(alpha)`, and draws a fresh per-round
//! `alpha_i` — which is the machinery the DeepFold binding and per-round
//! codeword-uniqueness arguments invoke. (An earlier revision of this note said
//! the implementation had NEITHER the alpha machinery NOR an interleaved
//! sumcheck, and inferred that even the Johnson row was unjustified; the alpha
//! machinery has since landed, so that inference no longer holds.)
//!
//! What remains genuinely open is only the CAPACITY row: `CONJ96 = 32` is a
//! capacity-regime count, and reaching capacity rests on the list-decoding
//! conjecture rather than on any theorem cited above. That is why every number
//! this repo reports uses `PROV_QUERY128`, not `CONJ96`.

/// FRI code rate log2. Same for all field backends.
pub const CODE_RATE_LOG: usize = 3;

/// Number of witness polynomials committed in DeepFold benches.
pub const NUM_WITNESS: usize = 3;

/// FRI queries for the 97-bit provable-security level. Currently identical
/// across MamaBear / BabyBear / Goldilocks; kept at module root to avoid
/// triple duplication. Move into per-field submodules if a field diverges.
pub const QUERY_NUM_PROV_QUERY97: usize = 76;

/// Per-circuit gate counts used by size-scaling benches and binaries.
pub mod gates {
    pub const SHA256_GATES_PER_BLOCK: usize = 302_000;
    pub const BLAKE3_GATES_PER_PERMUTATION: usize = 160_000;
    pub const AES128_GATES_PER_CALL: usize = 88_000;
    pub const KECCAKF_GATES_PER_PERMUTATION: usize = 584_000;

    // Poseidon2 -- two S-box variants x three fields. The uniform x^11
    // S-box is field-invariant; the native S-box varies per field
    // (x^5 on MamaBear, x^7 on BabyBear / Goldilocks).
    pub const POSEIDON2_X11_GATES_PER_PERMUTATION: usize = 5_321;
    pub const POSEIDON2_NATIVE_GATES_PER_PERMUTATION_MAMABEAR:   usize = 4_757;
    pub const POSEIDON2_NATIVE_GATES_PER_PERMUTATION_BABYBEAR:   usize = 5_039;
    pub const POSEIDON2_NATIVE_GATES_PER_PERMUTATION_GOLDILOCKS: usize = 5_039;
}

/// FRI parameters for the MamaBear (P = 2^49 - 2^34 + 1) backend.
pub mod mamabear {
    pub const QUERY_NUM_CONJ96: usize = 32;
    pub const QUERY_NUM_PROV_QUERY128: usize = 88;
    pub const GRINDING_BITS_EXT3_PROV_QUERY128: u32 = 16;
}

/// FRI parameters for the BabyBear backend.
pub mod babybear {
    pub const QUERY_NUM_CONJ96: usize = 36;
    pub const QUERY_NUM_PROV_QUERY128: usize = 101;
}

/// FRI parameters for the Goldilocks backend.
pub mod goldilocks {
    pub const QUERY_NUM_CONJ96: usize = 32;
    pub const QUERY_NUM_PROV_QUERY128: usize = 101;
}
