//! Generic throughput bench helpers over `arithmetic::field::Field`.
//!
//! Pattern: each bench builds a 10-element dependency chain
//!
//!     (a, b, c, d, e, f, g, h, i, j) = (a op b, b op c, ..., j op a)
//!
//! inside an outer loop of `N` iterations. The chain is intentional — it
//! breaks back-to-back data dependencies (each output feeds the next-next
//! operand), exposing enough ILP for the CPU to sustain throughput without
//! collapsing into a single critical-path-bound latency measurement.
//!
//! Setup (RNG + 10 random operands) is performed inside the closure but
//! outside the timer span (`Instant::now()` sits between setup and the inner
//! loop), matching criterion's `BatchSize::SmallInput` semantics.
//!
//! `count` is the caller-supplied logical operation count for the reported
//! row (typically `N * 10 * LANES` so cross-type results are comparable).

use std::hint::black_box;
use std::time::Instant;

// `LazyReduction` lives in the x86-only MamaBear module and is used only by the
// lazy_/reduce_ helpers below (exercised solely by the x86 `mamabear_bench`).
// Gated so the shared helper -- and the Goldilocks / BabyBear-NEON field
// baselines that pull it in -- build on aarch64.
#[cfg(target_arch = "x86_64")]
use arithmetic::field::mamabear::LazyReduction;
use arithmetic::field::Field;
use rand::rngs::SmallRng;
use rand::SeedableRng;

use super::measure::bench_op;

pub fn benchmark_add_throughput<F: Field + Copy, const N: usize>(field: &str, count: usize) {
    bench_op(field, "add", count, move || {
        let mut rng = SmallRng::seed_from_u64(1);
        let (mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h, mut i, mut j) = (
            F::random(&mut rng),
            F::random(&mut rng),
            F::random(&mut rng),
            F::random(&mut rng),
            F::random(&mut rng),
            F::random(&mut rng),
            F::random(&mut rng),
            F::random(&mut rng),
            F::random(&mut rng),
            F::random(&mut rng),
        );
        let start = Instant::now();
        for _ in 0..N {
            (a, b, c, d, e, f, g, h, i, j) = (
                a + b,
                b + c,
                c + d,
                d + e,
                e + f,
                f + g,
                g + h,
                h + i,
                i + j,
                j + a,
            );
        }
        let elapsed = start.elapsed();
        black_box((a, b, c, d, e, f, g, h, i, j));
        elapsed
    });
}

pub fn benchmark_sub_throughput<F: Field + Copy, const N: usize>(field: &str, count: usize) {
    bench_op(field, "sub", count, move || {
        let mut rng = SmallRng::seed_from_u64(1);
        let (mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h, mut i, mut j) = (
            F::random(&mut rng),
            F::random(&mut rng),
            F::random(&mut rng),
            F::random(&mut rng),
            F::random(&mut rng),
            F::random(&mut rng),
            F::random(&mut rng),
            F::random(&mut rng),
            F::random(&mut rng),
            F::random(&mut rng),
        );
        let start = Instant::now();
        for _ in 0..N {
            (a, b, c, d, e, f, g, h, i, j) = (
                a - b,
                b - c,
                c - d,
                d - e,
                e - f,
                f - g,
                g - h,
                h - i,
                i - j,
                j - a,
            );
        }
        let elapsed = start.elapsed();
        black_box((a, b, c, d, e, f, g, h, i, j));
        elapsed
    });
}

pub fn benchmark_mul_throughput<F: Field + Copy, const N: usize>(field: &str, count: usize) {
    bench_op(field, "mul", count, move || {
        let mut rng = SmallRng::seed_from_u64(1);
        let (mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h, mut i, mut j) = (
            F::random(&mut rng),
            F::random(&mut rng),
            F::random(&mut rng),
            F::random(&mut rng),
            F::random(&mut rng),
            F::random(&mut rng),
            F::random(&mut rng),
            F::random(&mut rng),
            F::random(&mut rng),
            F::random(&mut rng),
        );
        let start = Instant::now();
        for _ in 0..N {
            (a, b, c, d, e, f, g, h, i, j) = (
                a * b,
                b * c,
                c * d,
                d * e,
                e * f,
                f * g,
                g * h,
                h * i,
                i * j,
                j * a,
            );
        }
        let elapsed = start.elapsed();
        black_box((a, b, c, d, e, f, g, h, i, j));
        elapsed
    });
}

#[cfg(target_arch = "x86_64")]
#[allow(dead_code)]
pub fn benchmark_lazy_add_throughput<F: Field + LazyReduction + Copy, const N: usize>(
    field: &str,
    count: usize,
) {
    bench_op(field, "lazy_add", count, move || {
        let mut rng = SmallRng::seed_from_u64(1);
        let (mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h, mut i, mut j) = (
            F::random(&mut rng),
            F::random(&mut rng),
            F::random(&mut rng),
            F::random(&mut rng),
            F::random(&mut rng),
            F::random(&mut rng),
            F::random(&mut rng),
            F::random(&mut rng),
            F::random(&mut rng),
            F::random(&mut rng),
        );
        let start = Instant::now();
        for _ in 0..N {
            (a, b, c, d, e, f, g, h, i, j) = (
                a.lazy_add(b),
                b.lazy_add(c),
                c.lazy_add(d),
                d.lazy_add(e),
                e.lazy_add(f),
                f.lazy_add(g),
                g.lazy_add(h),
                h.lazy_add(i),
                i.lazy_add(j),
                j.lazy_add(a),
            );
        }
        let elapsed = start.elapsed();
        black_box((a, b, c, d, e, f, g, h, i, j));
        elapsed
    });
}

#[cfg(target_arch = "x86_64")]
#[allow(dead_code)]
pub fn benchmark_lazy_sub_throughput<F: Field + LazyReduction + Copy, const N: usize>(
    field: &str,
    count: usize,
) {
    bench_op(field, "lazy_sub", count, move || {
        let mut rng = SmallRng::seed_from_u64(1);
        let (mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h, mut i, mut j) = (
            F::random(&mut rng),
            F::random(&mut rng),
            F::random(&mut rng),
            F::random(&mut rng),
            F::random(&mut rng),
            F::random(&mut rng),
            F::random(&mut rng),
            F::random(&mut rng),
            F::random(&mut rng),
            F::random(&mut rng),
        );
        let start = Instant::now();
        for _ in 0..N {
            (a, b, c, d, e, f, g, h, i, j) = (
                a.lazy_sub(b),
                b.lazy_sub(c),
                c.lazy_sub(d),
                d.lazy_sub(e),
                e.lazy_sub(f),
                f.lazy_sub(g),
                g.lazy_sub(h),
                h.lazy_sub(i),
                i.lazy_sub(j),
                j.lazy_sub(a),
            );
        }
        let elapsed = start.elapsed();
        black_box((a, b, c, d, e, f, g, h, i, j));
        elapsed
    });
}

#[cfg(target_arch = "x86_64")]
#[allow(dead_code)]
pub fn benchmark_reduce_fast_throughput<F: Field + LazyReduction + Copy, const N: usize>(
    field: &str,
    count: usize,
) {
    bench_op(field, "reduce_fast", count, move || {
        let mut rng = SmallRng::seed_from_u64(1);
        let (mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h, mut i, mut j) = (
            F::random(&mut rng),
            F::random(&mut rng),
            F::random(&mut rng),
            F::random(&mut rng),
            F::random(&mut rng),
            F::random(&mut rng),
            F::random(&mut rng),
            F::random(&mut rng),
            F::random(&mut rng),
            F::random(&mut rng),
        );
        let start = Instant::now();
        for _ in 0..N {
            (a, b, c, d, e, f, g, h, i, j) = (
                a.reduce_fast(),
                b.reduce_fast(),
                c.reduce_fast(),
                d.reduce_fast(),
                e.reduce_fast(),
                f.reduce_fast(),
                g.reduce_fast(),
                h.reduce_fast(),
                i.reduce_fast(),
                j.reduce_fast(),
            );
        }
        let elapsed = start.elapsed();
        black_box((a, b, c, d, e, f, g, h, i, j));
        elapsed
    });
}

#[cfg(target_arch = "x86_64")]
#[allow(dead_code)]
pub fn benchmark_reduce_throughput<F: Field + LazyReduction + Copy, const N: usize>(
    field: &str,
    count: usize,
) {
    bench_op(field, "reduce", count, move || {
        let mut rng = SmallRng::seed_from_u64(1);
        let (mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h, mut i, mut j) = (
            F::random(&mut rng),
            F::random(&mut rng),
            F::random(&mut rng),
            F::random(&mut rng),
            F::random(&mut rng),
            F::random(&mut rng),
            F::random(&mut rng),
            F::random(&mut rng),
            F::random(&mut rng),
            F::random(&mut rng),
        );
        let start = Instant::now();
        for _ in 0..N {
            (a, b, c, d, e, f, g, h, i, j) = (
                a.reduce(),
                b.reduce(),
                c.reduce(),
                d.reduce(),
                e.reduce(),
                f.reduce(),
                g.reduce(),
                h.reduce(),
                i.reduce(),
                j.reduce(),
            );
        }
        let elapsed = start.elapsed();
        black_box((a, b, c, d, e, f, g, h, i, j));
        elapsed
    });
}
