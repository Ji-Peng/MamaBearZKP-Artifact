//! Plonky3 BabyBear AVX-512 throughput benches.
//!
//! Benches the native plonky3 `PackedBabyBearAVX512` (16 u32 lanes per ZMM)
//! and its degree-4 binomial extension `PackedBinomialExtensionField<_,_,4>`.
//! This is the AVX-512 path; the scalar wrapper used by `babybear_bench`
//! does NOT exercise it, as documented in `arithmetic/src/field/babybear.rs`.
//!
//! AVX-512 VERIFICATION: We verified via disassembly of the release bench
//! binary (`target/release/deps/babybear_avx512_bench-*`, built with
//! `RUSTFLAGS="-C target-cpu=native"`) that the add / sub / mul kernels
//! lower to genuine AVX-512 instructions operating on `zmm` registers
//! (16-lane u32). The bench fns get inlined into `main`, so the signal is
//! the per-mnemonic count across the whole binary:
//!
//!     objdump -d --no-show-raw-insn <bench-bin> | \
//!         grep -oE '\b(vpaddd|vpsubd|vpmuludq|vpminud|vmovdqu64)\b[^,]*zmm[0-9]+' | \
//!         awk '{print $1}' | sort | uniq -c
//!
//! Observed on this host:
//!   vpaddd    408    (16-lane u32 add)
//!   vpsubd    156    (16-lane u32 sub)
//!   vpmuludq  268    (signature of plonky3 MontyField31 mul on AVX-512)
//!   vpminud   244    (conditional subtract via min, used in Montgomery reduce)
//!   vmovdqu64 411    (16-lane u64 load/store)
//!
//! In particular, `vpmuludq ..., %zmm*, %zmm*` is the hot-loop kernel of
//! `PackedMontyField31AVX512::mul` (see
//! `external/plonky3/monty-31/src/x86_64_avx512/packing.rs`), so its
//! presence confirms we are exercising the 16-lane AVX-512 path and not the
//! scalar / AVX2 fallback.
//!
//! Env vars: BENCH_WARMUP (default 10), BENCH_SAMPLES (default 100),
//! BENCH_OUTPUT_FILE (optional append path).

use std::hint::black_box;
use std::time::Instant;

use p3_baby_bear::BabyBear;
use p3_field::extension::{BinomialExtensionField, PackedBinomialExtensionField};
use p3_field::{BasedVectorSpace, Field, PackedFieldExtension, PackedValue};
use rand::rngs::SmallRng;
use rand::{RngCore, SeedableRng};

mod measure;
use measure::bench_op;

// Must take packing via `<BabyBear as Field>::Packing` (NOT on an extension
// type) — on AVX-512 hosts with `target-cpu=native` this resolves to
// PackedBabyBearAVX512 (WIDTH = 16).
type PackedBB = <BabyBear as Field>::Packing;
type Ext4 = BinomialExtensionField<BabyBear, 4>;
type PBBExt4 = PackedBinomialExtensionField<BabyBear, PackedBB, 4>;

/// BabyBear prime. Duplicated here (instead of using
/// `arithmetic::field::babybear::P`) to keep this bench decoupled from the
/// scalar wrapper crate surface.
const BB_P: u32 = 2013265921; // 2^31 - 2^27 + 1

#[inline]
fn rand_babybear(rng: &mut SmallRng) -> BabyBear {
    BabyBear::new(rng.next_u32() % BB_P)
}

/// Build one packed BabyBear with 16 lane-independent random scalars.
#[inline]
fn rand_packed_bb(rng: &mut SmallRng) -> PackedBB {
    PackedBB::from_fn(|_| rand_babybear(rng))
}

/// Build one packed Ext4 element: 16 lane-independent random Ext4 scalars,
/// transposed into the `[PackedBB; 4]` component layout by `from_ext_slice`.
/// Bypasses rand's `StandardUniform::Distribution<Ext4>` (which is provided by
/// plonky3 but bound to rand 0.10, whereas this crate's dev deps are on 0.9).
#[inline]
fn rand_packed_bb_ext4(rng: &mut SmallRng) -> PBBExt4 {
    let lane_scalars: [Ext4; 16] = std::array::from_fn(|_| {
        let coeffs: [BabyBear; 4] = std::array::from_fn(|_| rand_babybear(rng));
        let mut iter = coeffs.into_iter();
        Ext4::from_basis_coefficients_fn(|_| iter.next().unwrap())
    });
    PBBExt4::from_ext_slice(&lane_scalars)
}

// ---------------------------------------------------------------------------
// PackedBB (16-lane u32) add / sub / mul throughput
// ---------------------------------------------------------------------------

fn bench_add_packed_bb<const N: usize>(field: &str, count: usize) {
    bench_op(field, "add", count, move || {
        let mut rng = SmallRng::seed_from_u64(1);
        let (mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h, mut i, mut j) = (
            rand_packed_bb(&mut rng), rand_packed_bb(&mut rng),
            rand_packed_bb(&mut rng), rand_packed_bb(&mut rng),
            rand_packed_bb(&mut rng), rand_packed_bb(&mut rng),
            rand_packed_bb(&mut rng), rand_packed_bb(&mut rng),
            rand_packed_bb(&mut rng), rand_packed_bb(&mut rng),
        );
        let start = Instant::now();
        for _ in 0..N {
            (a, b, c, d, e, f, g, h, i, j) = (
                a + b, b + c, c + d, d + e, e + f,
                f + g, g + h, h + i, i + j, j + a,
            );
        }
        let elapsed = start.elapsed();
        let _ = black_box((a, b, c, d, e, f, g, h, i, j));
        elapsed
    });
}

fn bench_sub_packed_bb<const N: usize>(field: &str, count: usize) {
    bench_op(field, "sub", count, move || {
        let mut rng = SmallRng::seed_from_u64(1);
        let (mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h, mut i, mut j) = (
            rand_packed_bb(&mut rng), rand_packed_bb(&mut rng),
            rand_packed_bb(&mut rng), rand_packed_bb(&mut rng),
            rand_packed_bb(&mut rng), rand_packed_bb(&mut rng),
            rand_packed_bb(&mut rng), rand_packed_bb(&mut rng),
            rand_packed_bb(&mut rng), rand_packed_bb(&mut rng),
        );
        let start = Instant::now();
        for _ in 0..N {
            (a, b, c, d, e, f, g, h, i, j) = (
                a - b, b - c, c - d, d - e, e - f,
                f - g, g - h, h - i, i - j, j - a,
            );
        }
        let elapsed = start.elapsed();
        let _ = black_box((a, b, c, d, e, f, g, h, i, j));
        elapsed
    });
}

fn bench_mul_packed_bb<const N: usize>(field: &str, count: usize) {
    bench_op(field, "mul", count, move || {
        let mut rng = SmallRng::seed_from_u64(1);
        let (mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h, mut i, mut j) = (
            rand_packed_bb(&mut rng), rand_packed_bb(&mut rng),
            rand_packed_bb(&mut rng), rand_packed_bb(&mut rng),
            rand_packed_bb(&mut rng), rand_packed_bb(&mut rng),
            rand_packed_bb(&mut rng), rand_packed_bb(&mut rng),
            rand_packed_bb(&mut rng), rand_packed_bb(&mut rng),
        );
        let start = Instant::now();
        for _ in 0..N {
            (a, b, c, d, e, f, g, h, i, j) = (
                a * b, b * c, c * d, d * e, e * f,
                f * g, g * h, h * i, i * j, j * a,
            );
        }
        let elapsed = start.elapsed();
        let _ = black_box((a, b, c, d, e, f, g, h, i, j));
        elapsed
    });
}

// ---------------------------------------------------------------------------
// PBBExt4 (PackedBinomialExtensionField<BabyBear, PackedBB, 4>) add / sub / mul
// ---------------------------------------------------------------------------

fn bench_add_packed_bb_ext4<const N: usize>(field: &str, count: usize) {
    bench_op(field, "add", count, move || {
        let mut rng = SmallRng::seed_from_u64(1);
        let (mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h, mut i, mut j) = (
            rand_packed_bb_ext4(&mut rng), rand_packed_bb_ext4(&mut rng),
            rand_packed_bb_ext4(&mut rng), rand_packed_bb_ext4(&mut rng),
            rand_packed_bb_ext4(&mut rng), rand_packed_bb_ext4(&mut rng),
            rand_packed_bb_ext4(&mut rng), rand_packed_bb_ext4(&mut rng),
            rand_packed_bb_ext4(&mut rng), rand_packed_bb_ext4(&mut rng),
        );
        let start = Instant::now();
        for _ in 0..N {
            (a, b, c, d, e, f, g, h, i, j) = (
                a + b, b + c, c + d, d + e, e + f,
                f + g, g + h, h + i, i + j, j + a,
            );
        }
        let elapsed = start.elapsed();
        let _ = black_box((a, b, c, d, e, f, g, h, i, j));
        elapsed
    });
}

fn bench_sub_packed_bb_ext4<const N: usize>(field: &str, count: usize) {
    bench_op(field, "sub", count, move || {
        let mut rng = SmallRng::seed_from_u64(1);
        let (mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h, mut i, mut j) = (
            rand_packed_bb_ext4(&mut rng), rand_packed_bb_ext4(&mut rng),
            rand_packed_bb_ext4(&mut rng), rand_packed_bb_ext4(&mut rng),
            rand_packed_bb_ext4(&mut rng), rand_packed_bb_ext4(&mut rng),
            rand_packed_bb_ext4(&mut rng), rand_packed_bb_ext4(&mut rng),
            rand_packed_bb_ext4(&mut rng), rand_packed_bb_ext4(&mut rng),
        );
        let start = Instant::now();
        for _ in 0..N {
            (a, b, c, d, e, f, g, h, i, j) = (
                a - b, b - c, c - d, d - e, e - f,
                f - g, g - h, h - i, i - j, j - a,
            );
        }
        let elapsed = start.elapsed();
        let _ = black_box((a, b, c, d, e, f, g, h, i, j));
        elapsed
    });
}

fn bench_mul_packed_bb_ext4<const N: usize>(field: &str, count: usize) {
    bench_op(field, "mul", count, move || {
        let mut rng = SmallRng::seed_from_u64(1);
        let (mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h, mut i, mut j) = (
            rand_packed_bb_ext4(&mut rng), rand_packed_bb_ext4(&mut rng),
            rand_packed_bb_ext4(&mut rng), rand_packed_bb_ext4(&mut rng),
            rand_packed_bb_ext4(&mut rng), rand_packed_bb_ext4(&mut rng),
            rand_packed_bb_ext4(&mut rng), rand_packed_bb_ext4(&mut rng),
            rand_packed_bb_ext4(&mut rng), rand_packed_bb_ext4(&mut rng),
        );
        let start = Instant::now();
        for _ in 0..N {
            (a, b, c, d, e, f, g, h, i, j) = (
                a * b, b * c, c * d, d * e, e * f,
                f * g, g * h, h * i, i * j, j * a,
            );
        }
        let elapsed = start.elapsed();
        let _ = black_box((a, b, c, d, e, f, g, h, i, j));
        elapsed
    });
}

fn main() {
    measure::init_from_env();

    // Compile-time assert that we really are on the 16-lane AVX-512 path.
    // PackedBabyBearAVX512 has WIDTH=16 (see
    // external/plonky3/monty-31/src/x86_64_avx512/packing.rs:34). If this
    // type alias resolved to a narrower packing (AVX2 = 8, scalar = 1), the
    // count denominator below would be wrong and results would be misleading.
    const _: () = assert!(
        <PackedBB as PackedValue>::WIDTH == 16,
        "babybear_avx512_bench must be built on an AVX-512 host; \
         build with RUSTFLAGS=\"-C target-cpu=native\" on a CPU with AVX-512F"
    );

    // 16 lanes per PackedBabyBearAVX512. Pick REPS=10_000_000/16 so that
    // count = REPS * 10 ops-per-iter * 16 lanes = 100_000_000, matching the
    // scalar and MamaBear AVX-512 benches.
    const REPS_AVX512_BB: usize = 10_000_000 / 16;
    const LANES: usize = 16;
    const COUNT: usize = REPS_AVX512_BB * 10 * LANES;

    let name = "BabyBearAVX-512";
    bench_add_packed_bb::<REPS_AVX512_BB>(name, COUNT);
    bench_sub_packed_bb::<REPS_AVX512_BB>(name, COUNT);
    bench_mul_packed_bb::<REPS_AVX512_BB>(name, COUNT);

    let name = "BabyBearAVX-512Ext4";
    bench_add_packed_bb_ext4::<REPS_AVX512_BB>(name, COUNT);
    bench_sub_packed_bb_ext4::<REPS_AVX512_BB>(name, COUNT);
    bench_mul_packed_bb_ext4::<REPS_AVX512_BB>(name, COUNT);
}
