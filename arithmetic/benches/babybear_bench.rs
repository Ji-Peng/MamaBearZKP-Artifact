//! BabyBear (scalar wrapper) field throughput benches.
//!
//! Benches the arithmetic-crate wrapper `BabyBearField` / `BabyBearExt4`,
//! which delegates to Plonky3 scalar `BabyBear`. No AVX-512 path through
//! this wrapper — see `babybear_avx512_bench` for the packed plonky3 path.
//!
//! Env vars: BENCH_WARMUP (default 10), BENCH_SAMPLES (default 100),
//! BENCH_OUTPUT_FILE (optional append path).

use arithmetic::field::babybear;

mod measure;
mod utils;
use utils::*;

type BB = babybear::BabyBearField;
type BBExt4 = babybear::BabyBearExt4;

fn main() {
    measure::init_from_env();

    const REPS: usize = 10_000_000;
    const LANES: usize = 1;
    const COUNT: usize = REPS * 10 * LANES;

    let name = "BabyBear";
    benchmark_add_throughput::<BB, REPS>(name, COUNT);
    benchmark_sub_throughput::<BB, REPS>(name, COUNT);
    benchmark_mul_throughput::<BB, REPS>(name, COUNT);

    let name = "BabyBearExt4";
    benchmark_add_throughput::<BBExt4, REPS>(name, COUNT);
    benchmark_sub_throughput::<BBExt4, REPS>(name, COUNT);
    benchmark_mul_throughput::<BBExt4, REPS>(name, COUNT);
}
