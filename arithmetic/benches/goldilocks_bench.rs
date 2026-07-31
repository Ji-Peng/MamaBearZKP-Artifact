//! Goldilocks64 field throughput benches (custom measure harness).
//!
//! Env vars: BENCH_WARMUP (default 10), BENCH_SAMPLES (default 100),
//! BENCH_OUTPUT_FILE (optional append path).

use arithmetic::field::goldilocks64;

mod measure;
mod utils;
use utils::*;

type Gold64 = goldilocks64::Goldilocks64;
type Gold64Ext = goldilocks64::Goldilocks64Ext;

fn main() {
    measure::init_from_env();

    // REPS * 10 ops per iter * 1 lane -> 100_000_000 ops per sample. Same
    // COUNT for every row so results are directly comparable against the
    // packed (lane-parallel) benches in mamabear_bench / babybear_avx512_bench.
    const REPS: usize = 10_000_000;
    const LANES: usize = 1;
    const COUNT: usize = REPS * 10 * LANES;

    let name = "Goldilocks64";
    benchmark_add_throughput::<Gold64, REPS>(name, COUNT);
    benchmark_sub_throughput::<Gold64, REPS>(name, COUNT);
    benchmark_mul_throughput::<Gold64, REPS>(name, COUNT);

    let name = "Goldilocks64Ext2";
    benchmark_add_throughput::<Gold64Ext, REPS>(name, COUNT);
    benchmark_sub_throughput::<Gold64Ext, REPS>(name, COUNT);
    benchmark_mul_throughput::<Gold64Ext, REPS>(name, COUNT);
}
