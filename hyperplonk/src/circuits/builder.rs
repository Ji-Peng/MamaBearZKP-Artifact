//! Lightweight circuit builder for constructing HyperPlonk circuits from
//! add/mul gates with copy constraints (wire routing).
//!
//! The builder produces `Circuit<F>` + `[Vec<MamaBearScalar>; 3]` witness
//! matching the format consumed by `setup_mamabear` and `ProverMamaBear::prove`.
//!
//! # Gate model
//!
//! The single gate type is: `(1-s)(a+b) + s*a*b + c = 0`
//! - s=0 (add gate): c = -(a+b) in the field
//! - s=1 (mul gate): c = -(a*b) in the field
//!
//! All arithmetic is actual field arithmetic over MamaBear (P = 2^49 - 2^34 + 1).

use arithmetic::field::{CanonicalBaseScalar, Field};
// MamaBear is x86_64-only (AVX-512IFMA); the MamaBear-typed `build` path and its
// witness construction are gated below, so this import is x86-only too. The
// field-agnostic builder body (u128 mod-p gadgets) compiles on every arch.
#[cfg(target_arch = "x86_64")]
use arithmetic::field::mamabear::MamaBearScalar;
use crate::circuit::Circuit;

const ID_SHIFT_1: u64 = 1 << 29;
const ID_SHIFT_2: u64 = 1 << 30;

/// MamaBear prime: P = 2^49 - 2^34 + 1.
pub const MAMABEAR_P: u64 = 562932773552129;

/// Goldilocks prime: P = 2^64 - 2^32 + 1.
pub const GOLDILOCKS_P: u64 = 0xFFFF_FFFF_0000_0001;

/// BabyBear prime: P = 2^31 - 2^27 + 1.
pub const BABYBEAR_P: u64 = 2013265921;

// ---------------------------------------------------------------------------
// WireId: identifies a specific wire position in the circuit
// ---------------------------------------------------------------------------

/// Identifies a wire position: (wire_index, gate_index).
/// wire = 0 → a-wire, 1 → b-wire, 2 → c-wire (output).
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct WireId {
    pub wire: usize,
    pub gate: usize,
}

impl WireId {
    #[inline]
    pub fn a(gate: usize) -> Self { Self { wire: 0, gate } }
    #[inline]
    pub fn b(gate: usize) -> Self { Self { wire: 1, gate } }
    #[inline]
    pub fn c(gate: usize) -> Self { Self { wire: 2, gate } }
}

// ---------------------------------------------------------------------------
// Field arithmetic helpers (parameterized by prime P)
// ---------------------------------------------------------------------------

#[inline]
fn field_add(p: u64, a: u64, b: u64) -> u64 {
    // Use u128 to handle near-u64 primes (e.g., Goldilocks).
    let sum = a as u128 + b as u128;
    let p128 = p as u128;
    if sum >= p128 { (sum - p128) as u64 } else { sum as u64 }
}

#[inline]
fn field_neg(p: u64, a: u64) -> u64 {
    if a == 0 { 0 } else { p - a }
}

#[inline]
fn field_mul(p: u64, a: u64, b: u64) -> u64 {
    ((a as u128 * b as u128) % p as u128) as u64
}

// ---------------------------------------------------------------------------
// CircuitBuilder
// ---------------------------------------------------------------------------

/// Accumulates gates and copy constraints, then emits `Circuit<F>` + witness.
///
/// Wire values are stored in 3 separate Vecs (matching HyperPlonk's 3-wire
/// ProductCheck layout). Permutation cycles use circular linked lists stored
/// per-wire-column.
pub struct CircuitBuilder {
    /// Prime of the target field. Determines field arithmetic semantics.
    p: u64,

    num_gates: usize,
    selectors: Vec<u8>,

    /// Witness values (normal form, in [0, P)).
    wire_vals: [Vec<u64>; 3],

    /// Permutation cycles: `perm_next[w][i] = (next_wire, next_gate)`.
    /// Initially each position points to itself (identity = no copy constraint).
    perm_next: [Vec<(usize, usize)>; 3],

    /// Constant pool: maps field value -> WireId (to reuse constant wires).
    const_pool: std::collections::HashMap<u64, WireId>,
}

impl CircuitBuilder {
    /// Create a builder with the given prime. Use the appropriate helper
    /// (`new_mamabear`, `new_goldilocks`, `new_babybear`) for standard fields.
    pub fn new_with_prime(p: u64) -> Self {
        let mut builder = Self {
            p,
            num_gates: 0,
            selectors: Vec::new(),
            wire_vals: [Vec::new(), Vec::new(), Vec::new()],
            perm_next: [Vec::new(), Vec::new(), Vec::new()],
            const_pool: std::collections::HashMap::new(),
        };
        // Pre-allocate constant 0 and 1 via dummy add gates.
        // Gate 0: a=0, b=0 -> c=0. Provides constant 0.
        let zero_c = builder.raw_add_gate(0, 0, 0);
        builder.const_pool.insert(0, zero_c);
        // Gate 1: a=1, b=0 -> c = -(1+0) = P-1. Provides constant 1 at wire-a.
        let _ = builder.raw_add_gate(0, 1, 0);
        builder.const_pool.insert(1, WireId::a(1));
        builder
    }

    /// Default constructor: MamaBear prime. Preserved for backward compatibility.
    pub fn new() -> Self {
        Self::new_with_prime(MAMABEAR_P)
    }

    /// Builder for the MamaBear field (P = 2^49 - 2^34 + 1).
    pub fn new_mamabear() -> Self {
        Self::new_with_prime(MAMABEAR_P)
    }

    /// Builder for the Goldilocks field (P = 2^64 - 2^32 + 1).
    pub fn new_goldilocks() -> Self {
        Self::new_with_prime(GOLDILOCKS_P)
    }

    /// Builder for the BabyBear field (P = 2^31 - 2^27 + 1).
    pub fn new_babybear() -> Self {
        Self::new_with_prime(BABYBEAR_P)
    }

    /// Access the prime used by this builder.
    #[inline]
    pub fn prime(&self) -> u64 {
        self.p
    }

    /// Total number of gates currently allocated.
    #[inline]
    pub fn num_gates(&self) -> usize {
        self.num_gates
    }

    // -----------------------------------------------------------------------
    // Low-level gate allocation
    // -----------------------------------------------------------------------

    /// Allocate a gate with given selector, wire values. Returns WireId of c-wire.
    /// Does NOT set up any copy constraints.
    fn raw_add_gate(&mut self, sel: u8, a_val: u64, b_val: u64) -> WireId {
        debug_assert!(a_val < self.p && b_val < self.p);
        let c_val = if sel == 0 {
            field_neg(self.p, field_add(self.p, a_val, b_val))
        } else {
            field_neg(self.p, field_mul(self.p, a_val, b_val))
        };
        let idx = self.num_gates;
        self.selectors.push(sel);
        self.wire_vals[0].push(a_val);
        self.wire_vals[1].push(b_val);
        self.wire_vals[2].push(c_val);
        for w in 0..3 {
            self.perm_next[w].push((w, idx)); // self-loop (identity)
        }
        self.num_gates += 1;
        WireId::c(idx)
    }

    // -----------------------------------------------------------------------
    // Public gate constructors
    // -----------------------------------------------------------------------

    /// Add gate: c = -(a_val + b_val).
    /// `a_src` and `b_src` are existing WireIds whose values are used as inputs.
    /// Returns WireId of the c-wire (holding -(a+b)).
    pub fn add_gate(&mut self, a_src: WireId, b_src: WireId) -> WireId {
        let a_val = self.get_val(a_src);
        let b_val = self.get_val(b_src);
        let c_id = self.raw_add_gate(0, a_val, b_val);
        let gate = c_id.gate;
        self.connect(a_src, WireId::a(gate));
        self.connect(b_src, WireId::b(gate));
        c_id
    }

    /// Mul gate: c = -(a_val * b_val).
    /// Returns WireId of the c-wire (holding -(a*b)).
    pub fn mul_gate(&mut self, a_src: WireId, b_src: WireId) -> WireId {
        let a_val = self.get_val(a_src);
        let b_val = self.get_val(b_src);
        let c_id = self.raw_add_gate(1, a_val, b_val);
        let gate = c_id.gate;
        self.connect(a_src, WireId::a(gate));
        self.connect(b_src, WireId::b(gate));
        c_id
    }

    /// Allocate a fresh input wire with the given value.
    /// Creates a dummy add gate: a=val, b=0, c=-(val).
    /// Returns WireId of the a-wire (holding `val`).
    pub fn alloc_input(&mut self, val: u64) -> WireId {
        debug_assert!(val < self.p);
        let zero = self.constant(0);
        let zero_val = 0u64;
        let _ = self.raw_add_gate(0, val, zero_val);
        let gate = self.num_gates - 1;
        self.connect(zero, WireId::b(gate));
        WireId::a(gate)
    }

    /// Get (or create) a wire holding the given constant value.
    pub fn constant(&mut self, val: u64) -> WireId {
        debug_assert!(val < self.p);
        if let Some(&id) = self.const_pool.get(&val) {
            return id;
        }
        // Create a new gate that produces the constant.
        // add gate: a=val, b=0 -> c=-(val). We return wire-a = val.
        let zero_id = *self.const_pool.get(&0).unwrap();
        let _ = self.raw_add_gate(0, val, 0);
        let gate = self.num_gates - 1;
        self.connect(zero_id, WireId::b(gate));
        let id = WireId::a(gate);
        self.const_pool.insert(val, id);
        id
    }

    // -----------------------------------------------------------------------
    // Wire value access
    // -----------------------------------------------------------------------

    /// Read the field value at a wire position.
    #[inline]
    pub fn get_val(&self, id: WireId) -> u64 {
        self.wire_vals[id.wire][id.gate]
    }

    // -----------------------------------------------------------------------
    // Copy constraints (permutation cycles)
    // -----------------------------------------------------------------------

    /// Merge the copy-constraint cycles of two wire positions.
    /// After this call, the prover will enforce that both positions carry equal values.
    ///
    /// Panics if the two positions have different witness values (sanity check).
    pub fn connect(&mut self, w1: WireId, w2: WireId) {
        debug_assert_eq!(
            self.get_val(w1), self.get_val(w2),
            "connect: wire values must match ({} vs {})",
            self.get_val(w1), self.get_val(w2)
        );
        if w1 == w2 {
            return;
        }
        // Splice two circular linked lists by swapping next-pointers.
        let next1 = self.perm_next[w1.wire][w1.gate];
        let next2 = self.perm_next[w2.wire][w2.gate];
        self.perm_next[w1.wire][w1.gate] = next2;
        self.perm_next[w2.wire][w2.gate] = next1;
    }

    // -----------------------------------------------------------------------
    // Padding and finalization
    // -----------------------------------------------------------------------

    /// Pad with dummy add gates (s=0, a=b=c=0) until num_gates is a power of 2.
    pub fn pad_to_power_of_two(&mut self) {
        let target = self.num_gates.next_power_of_two();
        let zero = *self.const_pool.get(&0).unwrap();
        while self.num_gates < target {
            let _ = self.raw_add_gate(0, 0, 0);
            let gate = self.num_gates - 1;
            // Connect a and b wires to the zero constant pool.
            self.connect(zero, WireId::a(gate));
            self.connect(zero, WireId::b(gate));
            self.connect(zero, WireId::c(gate));
        }
    }

    /// Pad to exactly 2^target_nv gates.
    pub fn pad_to_nv(&mut self, target_nv: usize) {
        let target = 1usize << target_nv;
        assert!(
            self.num_gates <= target,
            "Circuit has {} gates but target is 2^{} = {}",
            self.num_gates, target_nv, target
        );
        let zero = *self.const_pool.get(&0).unwrap();
        while self.num_gates < target {
            let _ = self.raw_add_gate(0, 0, 0);
            let gate = self.num_gates - 1;
            self.connect(zero, WireId::a(gate));
            self.connect(zero, WireId::b(gate));
            self.connect(zero, WireId::c(gate));
        }
    }

    /// Convert perm_next position to HyperPlonk tag.
    fn wire_tag(wire: usize, gate: usize) -> u64 {
        match wire {
            0 => gate as u64,
            1 => gate as u64 + ID_SHIFT_1,
            2 => gate as u64 + ID_SHIFT_2,
            _ => unreachable!(),
        }
    }

    /// Build the final `Circuit<F>` and witness `[Vec<MamaBearScalar>; 3]`.
    /// Requires this builder to use the MamaBear prime.
    ///
    /// MamaBear-specific (x86_64-only): direct `MamaBearScalar(v)` construction.
    /// For a field-generic build over any canonical base scalar use
    /// [`build_with`](Self::build_with). This method is kept byte-identical.
    #[cfg(target_arch = "x86_64")]
    pub fn build<F: Field<BaseField = MamaBearScalar>>(
        &self,
    ) -> (Circuit<F>, [Vec<MamaBearScalar>; 3]) {
        assert_eq!(self.p, MAMABEAR_P, "build() requires MamaBear prime");
        assert!(
            self.num_gates.is_power_of_two(),
            "num_gates must be power of 2 (call pad_to_power_of_two first)"
        );
        let n = self.num_gates;

        // Build permutation tables.
        let permutation: [Vec<MamaBearScalar>; 3] = std::array::from_fn(|w| {
            (0..n)
                .map(|i| {
                    let (nw, ng) = self.perm_next[w][i];
                    MamaBearScalar::from(Self::wire_tag(nw, ng))
                })
                .collect()
        });

        // Build selector.
        let selector: Vec<MamaBearScalar> = self
            .selectors
            .iter()
            .map(|&s| MamaBearScalar::from(s as u64))
            .collect();

        // Build witness.
        let witness: [Vec<MamaBearScalar>; 3] = std::array::from_fn(|w| {
            self.wire_vals[w]
                .iter()
                .map(|&v| MamaBearScalar(v))
                .collect()
        });

        let circuit = Circuit { permutation, selector };
        (circuit, witness)
    }

    /// Safely convert a u64 value to a field element via 32-bit decomposition.
    ///
    /// Uses only `B::from(u32)` (which every field implements correctly as an
    /// in-range embedding) and field arithmetic. This avoids the buggy
    /// `From<u64>` implementation in `Goldilocks64` which does not perform
    /// modular reduction for values >= 2^63.
    ///
    /// Given u64 `v`, decompose as `v = hi * 2^32 + lo` where both halves fit
    /// in u32, then compute `hi * 2^32 + lo` using field operations. This is
    /// correct for any field element whose canonical representation fits in u64.
    fn u64_to_field<B: Field + From<u32>>(v: u64) -> B {
        let lo = (v & 0xFFFFFFFF) as u32;
        let hi = (v >> 32) as u32;
        if hi == 0 {
            B::from(lo)
        } else {
            // 2^32 via repeated squaring of 2^16 (no overflow concerns).
            let two_16: B = B::from(1u32 << 16);
            let two_32 = two_16 * two_16;
            B::from(hi) * two_32 + B::from(lo)
        }
    }

    /// Generic build: produces `Circuit<F>` and `[Vec<B>; 3]` where `B` is the
    /// base field type.
    ///
    /// Use [`CircuitBuilder::build`] for MamaBear (which needs the non-canonical
    /// `MamaBearScalar(v)` constructor). This generic version goes through
    /// [`u64_to_field`] to safely handle the full u64 range.
    pub fn build_generic<F, B>(&self) -> (Circuit<F>, [Vec<B>; 3])
    where
        F: Field<BaseField = B>,
        B: Field + From<u32>,
    {
        assert!(
            self.num_gates.is_power_of_two(),
            "num_gates must be power of 2 (call pad_to_power_of_two first)"
        );
        let n = self.num_gates;

        let permutation: [Vec<B>; 3] = std::array::from_fn(|w| {
            (0..n)
                .map(|i| {
                    let (nw, ng) = self.perm_next[w][i];
                    Self::u64_to_field::<B>(Self::wire_tag(nw, ng))
                })
                .collect()
        });

        let selector: Vec<B> = self
            .selectors
            .iter()
            .map(|&s| Self::u64_to_field::<B>(s as u64))
            .collect();

        let witness: [Vec<B>; 3] = std::array::from_fn(|w| {
            self.wire_vals[w]
                .iter()
                .map(|&v| Self::u64_to_field::<B>(v))
                .collect()
        });

        let circuit = Circuit { permutation, selector };
        (circuit, witness)
    }

    /// Field-generic build over any [`CanonicalBaseScalar`] base `B`.
    ///
    /// Mirrors [`build`](Self::build) exactly, but constructs every emitted
    /// value (permutation wire tags, selectors, witness) via
    /// `B::from_canonical_u64` instead of the MamaBear-specific `MamaBearScalar(v)`.
    /// The builder's values are already canonical `[0, PRIME)` (all arithmetic is
    /// u128 mod-p), so this is an exact, direct construction — no `From<u32>`
    /// decomposition (unlike [`build_generic`](Self::build_generic)). Asserts the
    /// builder prime matches `B::PRIME`. The MamaBear path stays on `build`.
    pub fn build_with<F, B>(&self) -> (Circuit<F>, [Vec<B>; 3])
    where
        F: Field<BaseField = B>,
        B: CanonicalBaseScalar,
    {
        assert_eq!(self.p, B::PRIME, "build_with() prime mismatch with B::PRIME");
        assert!(
            self.num_gates.is_power_of_two(),
            "num_gates must be power of 2 (call pad_to_power_of_two first)"
        );
        let n = self.num_gates;

        let permutation: [Vec<B>; 3] = std::array::from_fn(|w| {
            (0..n)
                .map(|i| {
                    let (nw, ng) = self.perm_next[w][i];
                    B::from_canonical_u64(Self::wire_tag(nw, ng))
                })
                .collect()
        });

        let selector: Vec<B> = self
            .selectors
            .iter()
            .map(|&s| B::from_canonical_u64(s as u64))
            .collect();

        let witness: [Vec<B>; 3] = std::array::from_fn(|w| {
            self.wire_vals[w]
                .iter()
                .map(|&v| B::from_canonical_u64(v))
                .collect()
        });

        let circuit = Circuit { permutation, selector };
        (circuit, witness)
    }

    // -----------------------------------------------------------------------
    // Bit-level gadgets
    // -----------------------------------------------------------------------

    /// Boolean-constrain a wire: assert val ∈ {0, 1} via b*(1-b)=0.
    /// Two gates: mul(b, b) → -(b^2); add(b^2, -(b)) → b^2-b (must be 0).
    /// Panics (in debug) if the value is not 0 or 1.
    pub fn bool_constrain(&mut self, b: WireId) {
        let bv = self.get_val(b);
        debug_assert!(bv == 0 || bv == 1, "bool_constrain: value {} is not boolean", bv);
        // Gate 1: mul(b, b) → c1 = -(b*b) = -(b^2)
        let c1 = self.mul_gate(b, b);
        // c1 value: field_neg(field_mul(bv, bv)) = -(bv^2).
        // For boolean b: -(0^2) = 0 or -(1^2) = P-1.

        // Gate 2: add(b^2, neg_b) → c2 = -(b^2 + (-b)) = -(b^2 - b) = b - b^2.
        // For boolean: b - b^2 = 0.
        // We need a wire holding b^2. c1 holds -(b^2). Negate it.
        let b_sq = self.negate(c1); // b^2
        let neg_b = self.negate(b);  // -b = P-b for b∈{0,1}
        let c2 = self.add_gate(b_sq, neg_b);
        // c2 = -(b^2 + (-b)) = -(b^2 - b) = b - b^2.
        // For boolean b: 0. Connect to zero constant.
        let zero = self.constant(0);
        // c2 should equal 0 (since b is boolean). We verify this in debug.
        debug_assert_eq!(self.get_val(c2), 0, "bool_constrain: b*(1-b) != 0");
        self.connect(c2, zero);
    }

    /// Negate a wire value: returns WireId holding field_neg(val).
    /// Uses add gate: negate(x) → add(0, x) gives c = -(0+x) = -x.
    /// Wait, that gives -x, but we might want +x or -x depending on context.
    /// This returns a WireId whose value is field_neg(get_val(src)).
    pub fn negate(&mut self, src: WireId) -> WireId {
        let _val = self.get_val(src);
        // Create a gate where one input is src and the other is 0,
        // then the c-wire is -(val + 0) = -val. That's what we want.
        let zero = self.constant(0);
        self.add_gate(src, zero)
        // add_gate(src, zero) → c = -(val + 0) = -val = field_neg(val). ✓
    }

    /// XOR two boolean wires: xor(a, b) = a + b - 2*a*b.
    /// Returns WireId with value (a_val ^ b_val) as field element 0 or 1.
    pub fn xor_bit(&mut self, a: WireId, b: WireId) -> WireId {
        let av = self.get_val(a);
        let bv = self.get_val(b);
        debug_assert!(av <= 1 && bv <= 1);

        // ab = a * b (via mul gate, gives -(a*b))
        let neg_ab = self.mul_gate(a, b);
        // -(a*b) value: if a=1,b=1 → P-1; else 0.

        // We need a + b - 2*a*b.
        // Strategy: add(a, b) → -(a+b). Then add(-(a+b), 2ab) → (a+b)-2ab = xor.
        // But we have neg_ab = -(a*b). We need 2*a*b.
        // add(neg_ab, neg_ab) → -(2*neg_ab) = -(-2ab) = 2ab.
        let two_ab = self.add_gate(neg_ab, neg_ab);
        // two_ab value: field_neg(field_add(neg_ab_val, neg_ab_val)) = 2*a*b.

        // add(a, b) → neg_sum = -(a+b)
        let neg_sum = self.add_gate(a, b);

        // add(two_ab, neg_sum) → -(2ab + (-(a+b))) = (a+b) - 2ab = xor
        let xor_val = self.add_gate(two_ab, neg_sum);
        debug_assert_eq!(self.get_val(xor_val), av ^ bv);
        xor_val
    }

    /// AND two boolean wires: and(a, b) = a * b.
    /// Returns WireId with value (a_val & b_val) as field element 0 or 1.
    pub fn and_bit(&mut self, a: WireId, b: WireId) -> WireId {
        let av = self.get_val(a);
        let bv = self.get_val(b);
        debug_assert!(av <= 1 && bv <= 1);

        // mul(a, b) → -(a*b)
        let neg_ab = self.mul_gate(a, b);
        // Negate to get a*b
        let ab = self.negate(neg_ab);
        debug_assert_eq!(self.get_val(ab), av & bv);
        ab
    }

    /// NOT a boolean wire: not(a) = 1 - a.
    /// Fast 1-gate implementation: `add_gate(a, const(P-1))` returns a wire
    /// with value `-(a + (P-1)) = 1 - a mod P`. For a in {0, 1}, the result
    /// is in {1, 0} = NOT(a). Cheaper than the old xor-with-one path (4 gates).
    pub fn not_bit(&mut self, a: WireId) -> WireId {
        let av = self.get_val(a);
        debug_assert!(av <= 1);
        let neg_one = self.constant(self.p - 1);
        let not_a = self.add_gate(a, neg_one);
        debug_assert_eq!(self.get_val(not_a), 1 - av);
        not_a
    }

    /// XNOR two boolean wires: xnor(a, b) = 1 - (a XOR b).
    /// Uses `xor_bit` followed by the fast `not_bit` = 4 + 1 = 5 gates
    /// (vs. 4 + 4 = 8 in the naive xor+not combination).
    pub fn xnor_bit(&mut self, a: WireId, b: WireId) -> WireId {
        let xor = self.xor_bit(a, b);
        self.not_bit(xor)
    }

    // -----------------------------------------------------------------------
    // Sign-tracked XOR helpers (used by Boyar-Peralta S-box)
    // -----------------------------------------------------------------------
    //
    // These helpers use the identity `XOR(x, y) = (x - y)^2` (over booleans)
    // to compute XOR in fewer gates than the traditional `x + y - 2xy` form,
    // *when one or both inputs are already held as their field-negation*
    // (which happens naturally when chaining `mul_gate` outputs representing
    // -(x*y) without materializing the positive +xy form).
    //
    // Semantics: a "NEG wire" has value `-v mod P` where `v` is the logical
    // boolean; a "POS wire" has value `v` directly. Both xor_* helpers
    // below return a wire with value `-(x XOR y)` = -XOR = NEG-form xor.
    // To get the positive XOR, call `negate` on the result (1 extra gate).

    /// XOR when the two inputs have OPPOSITE sign conventions (one POS, one
    /// NEG). Cost: **2 gates**. Returns NEG wire holding -(x XOR y).
    ///
    /// Works because `add_gate(+a, -b) = -(a + (-b)) = b - a`, and
    /// `(b - a)^2 = XOR(a, b)` for booleans a, b. The final `mul_gate`
    /// squares and negates in one step.
    pub fn xor_mixed_sign_to_neg(&mut self, w1: WireId, w2: WireId) -> WireId {
        let diff = self.add_gate(w1, w2);
        self.mul_gate(diff, diff)
    }

    /// XOR when the two inputs have the SAME sign convention (both POS or
    /// both NEG). Cost: **3 gates**. Returns NEG wire holding -(x XOR y).
    ///
    /// One extra negate to flip one input's sign and create the diff; then
    /// the same (diff, diff) squaring step as `xor_mixed_sign_to_neg`.
    pub fn xor_same_sign_to_neg(&mut self, w1: WireId, w2: WireId) -> WireId {
        let neg_w2 = self.negate(w2);
        let diff = self.add_gate(w1, neg_w2);
        self.mul_gate(diff, diff)
    }

    /// Decompose a u32 value into 32 boolean WireIds (LSB first).
    /// Each bit is boolean-constrained.
    pub fn decompose_u32(&mut self, val: u32) -> [WireId; 32] {
        let mut bits = [WireId::a(0); 32]; // placeholder
        for i in 0..32 {
            let bit = ((val >> i) & 1) as u64;
            let w = self.alloc_input(bit);
            self.bool_constrain(w);
            bits[i] = w;
        }
        bits
    }

    // -----------------------------------------------------------------------
    // 32-bit word gadgets
    // -----------------------------------------------------------------------

    /// XOR two 32-bit words (represented as arrays of boolean WireIds, LSB first).
    pub fn xor_word32(&mut self, a: &[WireId; 32], b: &[WireId; 32]) -> [WireId; 32] {
        let mut out = [WireId::a(0); 32];
        for i in 0..32 {
            out[i] = self.xor_bit(a[i], b[i]);
        }
        out
    }

    /// AND two 32-bit words.
    pub fn and_word32(&mut self, a: &[WireId; 32], b: &[WireId; 32]) -> [WireId; 32] {
        let mut out = [WireId::a(0); 32];
        for i in 0..32 {
            out[i] = self.and_bit(a[i], b[i]);
        }
        out
    }

    /// NOT a 32-bit word.
    pub fn not_word32(&mut self, a: &[WireId; 32]) -> [WireId; 32] {
        let mut out = [WireId::a(0); 32];
        for i in 0..32 {
            out[i] = self.not_bit(a[i]);
        }
        out
    }

    /// Rotate right a 32-bit word by `n` positions. FREE (just index remapping).
    pub fn rotr32(&self, bits: &[WireId; 32], n: usize) -> [WireId; 32] {
        let mut out = [WireId::a(0); 32];
        for i in 0..32 {
            out[i] = bits[(i + n) % 32];
        }
        out
    }

    /// Shift right a 32-bit word by `n` positions (zero-fill). FREE for existing bits.
    pub fn shr32(&mut self, bits: &[WireId; 32], n: usize) -> [WireId; 32] {
        let zero = self.constant(0);
        let mut out = [WireId::a(0); 32];
        for i in 0..32 {
            if i + n < 32 {
                out[i] = bits[i + n];
            } else {
                out[i] = zero;
            }
        }
        out
    }

    /// 32-bit addition mod 2^32 using ripple-carry adder.
    /// Returns 32 boolean WireIds (LSB first) representing (a + b) mod 2^32.
    pub fn add_mod32(&mut self, a: &[WireId; 32], b: &[WireId; 32]) -> [WireId; 32] {
        let mut sum = [WireId::a(0); 32];
        let zero = self.constant(0);
        let mut carry = zero; // carry_0 = 0

        for i in 0..32 {
            // p = a[i] XOR b[i]
            let p = self.xor_bit(a[i], b[i]);

            if i == 0 {
                // sum[0] = p (since carry_0 = 0)
                sum[0] = p;
                // carry_1 = a[0] AND b[0]
                carry = self.and_bit(a[0], b[0]);
            } else if i < 31 {
                // sum[i] = p XOR carry
                sum[i] = self.xor_bit(p, carry);
                // g = a[i] AND b[i]  → gives -(g) from mul gate
                let neg_g = self.mul_gate(a[i], b[i]);
                // carry * p → gives -(carry*p) from mul gate
                let neg_cp = self.mul_gate(carry, p);
                // carry_{i+1} = g + carry*p
                // add(-(g), -(carry*p)) → -(-(g) + -(carry*p)) = g + carry*p ✓
                carry = self.add_gate(neg_g, neg_cp);
            } else {
                // Last bit: just compute sum, no carry needed
                sum[31] = self.xor_bit(p, carry);
            }
        }
        sum
    }

    /// SHA-256 Ch function: Ch(e,f,g) = (e AND f) XOR (NOT(e) AND g).
    /// Optimized: Ch = ef + g - eg (4 gates per bit).
    pub fn ch_word32(
        &mut self,
        e: &[WireId; 32],
        f: &[WireId; 32],
        g: &[WireId; 32],
    ) -> [WireId; 32] {
        let mut out = [WireId::a(0); 32];
        for i in 0..32 {
            // mul(e, f) → neg_ef = -(ef)
            let neg_ef = self.mul_gate(e[i], f[i]);
            // mul(e, g) → neg_eg = -(eg)
            let neg_eg = self.mul_gate(e[i], g[i]);
            // add(neg_eg, g) → -(-(eg) + g) = eg - g
            let eg_minus_g = self.add_gate(neg_eg, g[i]);
            // add(neg_ef, eg_minus_g) → -(-(ef) + (eg-g)) = ef - eg + g = Ch
            out[i] = self.add_gate(neg_ef, eg_minus_g);

            debug_assert!({
                let ev = self.get_val(e[i]);
                let fv = self.get_val(f[i]);
                let gv = self.get_val(g[i]);
                let expected = (ev & fv) ^ ((!ev & 1) & gv);
                self.get_val(out[i]) == expected
            });
        }
        out
    }

    /// SHA-256 Maj function: Maj(a,b,c) = (a AND b) XOR (a AND c) XOR (b AND c).
    /// Optimized: Maj = ab + (a XOR b)*c (6 gates per bit).
    pub fn maj_word32(
        &mut self,
        a: &[WireId; 32],
        b: &[WireId; 32],
        c: &[WireId; 32],
    ) -> [WireId; 32] {
        let mut out = [WireId::a(0); 32];
        for i in 0..32 {
            // XOR(a, b): 4 gates → p (positive, value = a^b)
            let p = self.xor_bit(a[i], b[i]);
            // mul(p, c) → neg_pc = -(p*c)
            let neg_pc = self.mul_gate(p, c[i]);
            // mul(a, b) → neg_ab = -(ab) [reused from XOR internal? no, separate]
            let neg_ab = self.mul_gate(a[i], b[i]);
            // add(neg_ab, neg_pc) → -(-(ab) + -(pc)) = ab + pc = Maj
            out[i] = self.add_gate(neg_ab, neg_pc);

            debug_assert!({
                let av = self.get_val(a[i]);
                let bv = self.get_val(b[i]);
                let cv = self.get_val(c[i]);
                let expected = (av & bv) ^ (av & cv) ^ (bv & cv);
                self.get_val(out[i]) == expected
            });
        }
        out
    }

    /// Sigma function: XOR of three rotated/shifted versions of a word.
    /// sigma(x, r1, r2, r3, shift3) = ROTR(r1) XOR ROTR(r2) XOR (SHIFT ? SHR(r3) : ROTR(r3))
    pub fn sigma_word32(
        &mut self,
        x: &[WireId; 32],
        r1: usize,
        r2: usize,
        r3: usize,
        is_shr: bool,
    ) -> [WireId; 32] {
        let t1 = self.rotr32(x, r1);
        let t2 = self.rotr32(x, r2);
        let t3 = if is_shr {
            self.shr32(x, r3)
        } else {
            self.rotr32(x, r3)
        };
        let tmp = self.xor_word32(&t1, &t2);
        self.xor_word32(&tmp, &t3)
    }

    // -----------------------------------------------------------------------
    // 64-bit word gadgets (used by Keccak-f[1600])
    // -----------------------------------------------------------------------

    /// Decompose a u64 value into 64 boolean WireIds (LSB first).
    pub fn decompose_u64(&mut self, val: u64) -> [WireId; 64] {
        let mut bits = [WireId::a(0); 64];
        for i in 0..64 {
            let bit = (val >> i) & 1;
            let w = self.alloc_input(bit);
            self.bool_constrain(w);
            bits[i] = w;
        }
        bits
    }

    /// XOR two 64-bit words.
    pub fn xor_word64(&mut self, a: &[WireId; 64], b: &[WireId; 64]) -> [WireId; 64] {
        let mut out = [WireId::a(0); 64];
        for i in 0..64 {
            out[i] = self.xor_bit(a[i], b[i]);
        }
        out
    }

    /// AND two 64-bit words.
    pub fn and_word64(&mut self, a: &[WireId; 64], b: &[WireId; 64]) -> [WireId; 64] {
        let mut out = [WireId::a(0); 64];
        for i in 0..64 {
            out[i] = self.and_bit(a[i], b[i]);
        }
        out
    }

    /// NOT a 64-bit word.
    pub fn not_word64(&mut self, a: &[WireId; 64]) -> [WireId; 64] {
        let mut out = [WireId::a(0); 64];
        for i in 0..64 {
            out[i] = self.not_bit(a[i]);
        }
        out
    }

    /// Rotate LEFT a 64-bit word by `n` positions. FREE (index remap).
    pub fn rotl64(&self, bits: &[WireId; 64], n: usize) -> [WireId; 64] {
        let n = n % 64;
        let mut out = [WireId::a(0); 64];
        for i in 0..64 {
            // left rotation: out[i] = bits[(i - n) mod 64]
            out[i] = bits[(i + 64 - n) % 64];
        }
        out
    }

    /// Rotate RIGHT a 64-bit word by `n` positions. FREE (index remap).
    pub fn rotr64(&self, bits: &[WireId; 64], n: usize) -> [WireId; 64] {
        let n = n % 64;
        let mut out = [WireId::a(0); 64];
        for i in 0..64 {
            out[i] = bits[(i + n) % 64];
        }
        out
    }

    // -----------------------------------------------------------------------
    // Field-element helpers (used by Poseidon2)
    // -----------------------------------------------------------------------

    /// Positive field addition: returns a wire whose value is
    /// `(a_val + b_val) mod P`. Costs 2 gates.
    pub fn field_add_pos(&mut self, a: WireId, b: WireId) -> WireId {
        let neg_sum = self.add_gate(a, b);
        self.negate(neg_sum)
    }

    /// Positive sum of many wires. Returns a wire carrying
    /// `sum(wires) mod P`. Cost: `2 * (len - 1)` gates.
    pub fn field_sum(&mut self, wires: &[WireId]) -> WireId {
        assert!(!wires.is_empty(), "field_sum on empty slice");
        let mut acc = wires[0];
        for &w in &wires[1..] {
            acc = self.field_add_pos(acc, w);
        }
        acc
    }

    /// Positive scalar multiplication: returns a wire whose value is
    /// `(k_val * get_val(w)) mod P`. Costs 2 gates plus an amortized
    /// constant-pool lookup.
    pub fn field_const_mul(&mut self, w: WireId, k_val: u64) -> WireId {
        let k_wire = self.constant(k_val);
        let neg_kw = self.mul_gate(w, k_wire);
        self.negate(neg_kw)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

// The test module exercises the MamaBear build path + DeepFold PCS (x86_64-only).
#[cfg(all(test, target_arch = "x86_64"))]
mod tests {
    use super::*;

    #[test]
    fn boolean_gadgets() {
        let mut b = CircuitBuilder::new();
        let zero = b.constant(0);
        let one = b.constant(1);

        // XOR
        let r = b.xor_bit(zero, zero); assert_eq!(b.get_val(r), 0);
        let r = b.xor_bit(zero, one);  assert_eq!(b.get_val(r), 1);
        let r = b.xor_bit(one, zero);  assert_eq!(b.get_val(r), 1);
        let r = b.xor_bit(one, one);   assert_eq!(b.get_val(r), 0);

        // AND
        let r = b.and_bit(zero, zero); assert_eq!(b.get_val(r), 0);
        let r = b.and_bit(zero, one);  assert_eq!(b.get_val(r), 0);
        let r = b.and_bit(one, zero);  assert_eq!(b.get_val(r), 0);
        let r = b.and_bit(one, one);   assert_eq!(b.get_val(r), 1);

        // NOT
        let r = b.not_bit(zero); assert_eq!(b.get_val(r), 1);
        let r = b.not_bit(one);  assert_eq!(b.get_val(r), 0);
    }

    #[test]
    fn word64_gadgets_roundtrip() {
        let mut b = CircuitBuilder::new();
        let patterns: [u64; 6] = [
            0x0000_0000_0000_0000,
            0xFFFF_FFFF_FFFF_FFFF,
            0xDEAD_BEEF_CAFE_BABE,
            0x0123_4567_89AB_CDEF,
            0x8000_0000_0000_0001,
            0xA5A5_A5A5_5A5A_5A5A,
        ];

        for &p in &patterns {
            for &q in &patterns {
                let x = b.decompose_u64(p);
                let y = b.decompose_u64(q);

                // XOR
                let z = b.xor_word64(&x, &y);
                let got_xor: u64 = (0..64).map(|i| (b.get_val(z[i]) as u64) << i).sum();
                assert_eq!(got_xor, p ^ q, "xor_word64({:#x}, {:#x})", p, q);

                // AND
                let z = b.and_word64(&x, &y);
                let got_and: u64 = (0..64).map(|i| (b.get_val(z[i]) as u64) << i).sum();
                assert_eq!(got_and, p & q, "and_word64({:#x}, {:#x})", p, q);
            }

            // NOT
            let x = b.decompose_u64(p);
            let z = b.not_word64(&x);
            let got_not: u64 = (0..64).map(|i| (b.get_val(z[i]) as u64) << i).sum();
            assert_eq!(got_not, !p, "not_word64({:#x})", p);

            // Rotations (left)
            for &n in &[0usize, 1, 7, 16, 32, 63] {
                let x = b.decompose_u64(p);
                let z = b.rotl64(&x, n);
                let got_rotl: u64 = (0..64).map(|i| (b.get_val(z[i]) as u64) << i).sum();
                assert_eq!(got_rotl, p.rotate_left(n as u32), "rotl64({:#x}, {})", p, n);

                let z = b.rotr64(&x, n);
                let got_rotr: u64 = (0..64).map(|i| (b.get_val(z[i]) as u64) << i).sum();
                assert_eq!(got_rotr, p.rotate_right(n as u32), "rotr64({:#x}, {})", p, n);
            }
        }
    }

    #[test]
    fn field_helpers_smoke() {
        let mut b = CircuitBuilder::new();
        let x = b.alloc_input(100);
        let y = b.alloc_input(200);
        let z = b.alloc_input(50);

        // field_add_pos
        let sum2 = b.field_add_pos(x, y);
        assert_eq!(b.get_val(sum2), 300);

        // field_sum
        let sum3 = b.field_sum(&[x, y, z]);
        assert_eq!(b.get_val(sum3), 350);

        // field_const_mul
        let scaled = b.field_const_mul(x, 7);
        assert_eq!(b.get_val(scaled), 700);
    }

    #[test]
    fn add_mod32_basic() {
        let mut b = CircuitBuilder::new();
        let x = b.decompose_u32(0xFFFF_FFFF);
        let y = b.decompose_u32(1);
        let z = b.add_mod32(&x, &y);

        // 0xFFFFFFFF + 1 = 0 (mod 2^32)
        let result: u32 = (0..32)
            .map(|i| (b.get_val(z[i]) as u32) << i)
            .sum();
        assert_eq!(result, 0);
    }

}
