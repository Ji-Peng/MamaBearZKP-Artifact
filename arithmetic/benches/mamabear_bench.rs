//! MamaBear AVX-512 field throughput benches (PackedMamaBearAVX512 and its
//! degree-3 extension packed variant, all 8 u64 lanes per ZMM).
//!
//! Env vars: BENCH_WARMUP (default 10), BENCH_SAMPLES (default 100),
//! BENCH_OUTPUT_FILE (optional append path).

use arithmetic::field::mamabear;

mod measure;
mod utils;
use utils::*;

type MamaBear = mamabear::PackedMamaBearAVX512;
type MamaBearExt3 = mamabear::PackedMamaBearAVX512Ext3;

fn main() {
    measure::init_from_env();

    // 8 lanes per PackedMamaBearAVX512. Pick REPS=10_000_000/8 so that
    // count = REPS * 10 ops-per-iter * 8 lanes = 100_000_000, matching the
    // scalar benches' COUNT. lazy_sub now uses the same 10-element chain as
    // the other ops, so one COUNT covers every row.
    const REPS_AVX512: usize = 10_000_000 / 8;
    const LANES: usize = 8;
    const COUNT: usize = REPS_AVX512 * 10 * LANES;

    let name = "MamaBearAVX-512";
    benchmark_add_throughput::<MamaBear, REPS_AVX512>(name, COUNT);
    benchmark_sub_throughput::<MamaBear, REPS_AVX512>(name, COUNT);
    benchmark_mul_throughput::<MamaBear, REPS_AVX512>(name, COUNT);
    benchmark_lazy_add_throughput::<MamaBear, REPS_AVX512>(name, COUNT);
    benchmark_lazy_sub_throughput::<MamaBear, REPS_AVX512>(name, COUNT);
    benchmark_reduce_fast_throughput::<MamaBear, REPS_AVX512>(name, COUNT);
    benchmark_reduce_throughput::<MamaBear, REPS_AVX512>(name, COUNT);

    let name = "MamaBearAVX-512Ext3";
    benchmark_add_throughput::<MamaBearExt3, REPS_AVX512>(name, COUNT);
    benchmark_sub_throughput::<MamaBearExt3, REPS_AVX512>(name, COUNT);
    benchmark_mul_throughput::<MamaBearExt3, REPS_AVX512>(name, COUNT);
    benchmark_lazy_add_throughput::<MamaBearExt3, REPS_AVX512>(name, COUNT);
    benchmark_lazy_sub_throughput::<MamaBearExt3, REPS_AVX512>(name, COUNT);
    benchmark_reduce_fast_throughput::<MamaBearExt3, REPS_AVX512>(name, COUNT);
    benchmark_reduce_throughput::<MamaBearExt3, REPS_AVX512>(name, COUNT);
}
