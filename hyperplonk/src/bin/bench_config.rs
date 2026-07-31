//! `bench_config` — print the RESOLVED SNARK configuration as `#`-prefixed
//! header lines, for the `reproduce_*.sh` drivers to embed in every result file.
//!
//! Usage:
//! ```text
//! cargo run --release -q -p hyperplonk --bin bench_config -- \
//!     [--split N] [--nv N]...
//! ```

use std::process::ExitCode;

use util::params::{babybear, goldilocks, mamabear, CODE_RATE_LOG};

const Q_BITS_PER_QUERY: f64 = 1.2776;
const BCIKS_M: f64 = 3.0;

fn proximity_gap_coeff(code_rate_log: u32) -> f64 {
    let rho = 1.0 / f64::from(1u32 << code_rate_log);
    let sqrt_rho = rho.sqrt();
    let eta = sqrt_rho / (2.0 * BCIKS_M);
    let delta = 1.0 - sqrt_rho - eta;
    let m_half = BCIKS_M + 0.5;
    (2.0 * m_half.powi(5) + 3.0 * m_half * delta * rho) / (3.0 * rho.powf(1.5))
}

fn commit_bound_bits(log2_field_ext: f64, nv: u32, code_rate_log: u32) -> f64 {
    let log2_n = f64::from(nv + code_rate_log);
    log2_field_ext - log2_n - log2_n.log2() - proximity_gap_coeff(code_rate_log).log2()
}

fn usage() -> &'static str {
    "usage: bench_config [--split N] [--nv N]...\n\
     Prints the resolved SNARK configuration as '#'-prefixed header lines."
}

fn main() -> ExitCode {
    let mut split: String = String::from("3");
    let mut nvs: Vec<u32> = Vec::new();

    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--split" => {
                i += 1;
                match args.get(i) {
                    Some(v) if !v.is_empty() => split = v.clone(),
                    _ => {
                        eprintln!("bench_config: --split needs a value\n{}", usage());
                        return ExitCode::from(2);
                    }
                }
            }
            "--nv" => {
                i += 1;
                match args.get(i).and_then(|s| s.parse::<u32>().ok()) {
                    Some(v) => nvs.push(v),
                    None => {
                        eprintln!("bench_config: --nv needs a number\n{}", usage());
                        return ExitCode::from(2);
                    }
                }
            }
            "-h" | "--help" => {
                println!("{}", usage());
                return ExitCode::SUCCESS;
            }
            other => {
                eprintln!("bench_config: unexpected argument {other:?}\n{}", usage());
                return ExitCode::from(2);
            }
        }
        i += 1;
    }

    if nvs.is_empty() {
        nvs = vec![19, 20];
    }

    let rate_log = CODE_RATE_LOG as u32;
    let queries = mamabear::QUERY_NUM_PROV_QUERY128;
    let grinding = mamabear::GRINDING_BITS_EXT3_PROV_QUERY128;
    const EXT_DEG: u32 = 3;
    let log2_p = ((1u64 << 49) - (1u64 << 34) + 1) as f64;
    let log2_ext = log2_p.log2() * f64::from(EXT_DEG);

    let q_bits = queries as f64 * Q_BITS_PER_QUERY;
    let s_query = f64::from(grinding) + q_bits;

    println!(
        "# backend:   MamaBear  P = 2^49 - 2^34 + 1 (49-bit Solinas), R = 2^52 Montgomery, AVX-512IFMA"
    );
    println!(
        "# extension: Ext{EXT_DEG}  F_p[X]/(X^3 - X - 1)  log2|F_ext| = {log2_ext:.1}"
    );
    println!(
        "# pcs:       DeepFold  rate 1/{} (CODE_RATE_LOG={})  split={}",
        1usize << rate_log,
        rate_log,
        split
    );
    println!(
        "# fri:       PROV_QUERY128  queries={queries}  grinding={grinding} bits  (m={BCIKS_M}, \
         {Q_BITS_PER_QUERY} bits/query)"
    );
    println!(
        "# fri-query: S_query = zeta_q + Q = {grinding} + {queries}*{Q_BITS_PER_QUERY} = \
         {s_query:.1} bits"
    );

    let mut sorted_nvs = nvs.clone();
    sorted_nvs.sort_unstable();
    sorted_nvs.dedup();
    let c_pg = proximity_gap_coeff(rate_log);
    let cells: Vec<String> = sorted_nvs
        .iter()
        .map(|&nv| {
            let c = commit_bound_bits(log2_ext, nv, rate_log);
            format!("nv={nv}: C={c:.1}")
        })
        .collect();
    println!(
        "# fri-commit: C = log2|F_ext| - log2(n) - log2(log2 n) - log2(c_PG),  c_PG = {c_pg:.1} \
         (m={BCIKS_M}, rate 1/{}),  n = 2^(nv+{rate_log});  {}",
        1usize << rate_log,
        cells.join("  ")
    );

    let mut any_commit = false;
    let mut any_query = false;
    let s_prov: Vec<String> = sorted_nvs
        .iter()
        .map(|&nv| {
            let c = commit_bound_bits(log2_ext, nv, rate_log);
            let commit_binds = c < s_query;
            if commit_binds {
                any_commit = true;
            } else {
                any_query = true;
            }
            let side = if commit_binds { "commit" } else { "query" };
            format!("nv={nv}: {:.1} ({side}-bound)", c.min(s_query))
        })
        .collect();
    let consequence = match (any_commit, any_query) {
        (true, false) => "adding queries cannot lift a commit-bound total",
        (false, true) => "commit term exceeds the query budget, so more queries WOULD lift it",
        _ => "mixed across this nv range: more queries lift only the query-bound rows",
    };
    println!(
        "# fri-prov:  S_prov = min(C, S_query) = {}  ({consequence})",
        s_prov.join("  ")
    );
    println!(
        "# piop-sec:  ~124-bit conjectured: the PIOP soundness error is Schwartz-Zippel over \
         Ext3 (log2|F_ext| = {log2_ext:.0}), capped by the 2^-124 transcript-challenge budget"
    );
    println!(
        "# baselines: Goldilocks PROV_QUERY128 queries={}  BabyBear PROV_QUERY128 queries={}  \
         (same rate 1/{}, no grinding)",
        goldilocks::QUERY_NUM_PROV_QUERY128,
        babybear::QUERY_NUM_PROV_QUERY128,
        1usize << rate_log
    );

    ExitCode::SUCCESS
}
