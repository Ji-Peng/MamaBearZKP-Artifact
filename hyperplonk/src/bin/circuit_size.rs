#[cfg(not(target_arch = "x86_64"))]
fn main() {}
#[cfg(target_arch = "x86_64")]
fn main() {
    x86_impl::main();
}
#[cfg(target_arch = "x86_64")]
mod x86_impl {
//! Circuit-size reporter.
//!
//! Measures the raw gate count of one cryptographic operation for each of
//! the currently-supported circuits and prints a concise summary plus a
//! table of `nv` vs `number-of-operations-that-fit-in-2^nv-gates`.
//!
//! Usage:
//!   cargo run --release -p hyperplonk --bin circuit_size -- <variant>
//!
//! Variants:
//!   sha256, aes128, blake3, keccakf,
//!   poseidon2_x11, poseidon2_native_mamabear,
//!   poseidon2_native_goldilocks, poseidon2_native_babybear,
//!   all

use std::fs::OpenOptions;
use std::io::Write as _;

use hyperplonk::{
    aes_circuit::aes128_gates_per_call,
    blake3_circuit::blake3_gates_per_permutation,
    keccakf_circuit::keccakf_gates_per_permutation,
    poseidon2_circuit::{
        poseidon2_native_gates_per_permutation_babybear,
        poseidon2_native_gates_per_permutation_goldilocks,
        poseidon2_native_gates_per_permutation_mamabear,
        poseidon2_x11_gates_per_permutation,
    },
    sha256_circuit::sha256_gates_per_block,
};

const NV_MIN_DEFAULT: usize = 16;
const NV_MAX_DEFAULT: usize = 28;

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn fmt_commas(n: u64) -> String {
    let s = n.to_string();
    let bytes = s.as_bytes();
    let mut out = String::with_capacity(s.len() + s.len() / 3);
    for (i, &c) in bytes.iter().enumerate() {
        if i > 0 && (bytes.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(c as char);
    }
    out
}

fn fmt_log2(n: u64) -> String {
    debug_assert!(n > 0);
    format!("{:.2}", (n as f64).log2())
}

fn summary_sentence(circuit: &str, unit: &str, gates: usize) -> String {
    format!(
        "For {circuit}: 1 {unit} corresponds to {} (2^{}) gates.",
        fmt_commas(gates as u64),
        fmt_log2(gates as u64),
    )
}

fn render_table(
    out: &mut Vec<String>,
    title: &str,
    unit_col: &str,
    gates_per_unit: usize,
    nv_min: usize,
    nv_max: usize,
) {
    out.push(String::new());
    out.push(format!("{title}"));
    let hdr = format!(" {:>3} | {:>18} | {:>14}", "nv", "2^nv", unit_col);
    let sep = format!("-----+--------------------+----------------");
    out.push(hdr);
    out.push(sep);
    for nv in nv_min..=nv_max {
        let size: u128 = 1u128 << nv;
        let n_units = size / (gates_per_unit as u128);
        out.push(format!(
            " {:>3} | {:>18} | {:>14}",
            nv,
            fmt_commas(size as u64),
            fmt_commas(n_units as u64),
        ));
    }
}

struct Entry {
    variant: &'static str,
    circuit: &'static str,
    unit: &'static str,
    unit_col: &'static str,
    title: &'static str,
    gates: usize,
}

fn all_entries() -> Vec<Entry> {
    vec![
        Entry {
            variant: "sha256",
            circuit: "SHA256",
            unit: "block",
            unit_col: "#blocks",
            title: "SHA256 circuit: circuit size 2^nv vs number of SHA-256 compression blocks",
            gates: sha256_gates_per_block(),
        },
        Entry {
            variant: "aes128",
            circuit: "AES128",
            unit: "call",
            unit_col: "#calls",
            title: "AES128 circuit: circuit size 2^nv vs number of AES-128 encryption calls",
            gates: aes128_gates_per_call(),
        },
        Entry {
            variant: "blake3",
            circuit: "Blake3",
            unit: "Blake3 permutation",
            unit_col: "#permutations",
            title: "Blake3 circuit: circuit size 2^nv vs number of BLAKE3 permutations",
            gates: blake3_gates_per_permutation(),
        },
        Entry {
            variant: "keccakf",
            circuit: "Keccak-f[1600]",
            unit: "Keccak-f[1600] permutation",
            unit_col: "#permutations",
            title: "Keccak-f[1600] circuit: circuit size 2^nv vs number of permutations",
            gates: keccakf_gates_per_permutation(),
        },
        Entry {
            variant: "poseidon2_x11",
            circuit: "Poseidon2 (uniform x^11)",
            unit: "Poseidon2 permutation",
            unit_col: "#permutations",
            title: "Poseidon2 x^11 circuit (width 16, uniform S-box): \
                    circuit size 2^nv vs number of permutations",
            gates: poseidon2_x11_gates_per_permutation(),
        },
        Entry {
            variant: "poseidon2_native_mamabear",
            circuit: "Poseidon2 (MamaBear native, x^5)",
            unit: "Poseidon2 permutation",
            unit_col: "#permutations",
            title: "Poseidon2 native MamaBear circuit (width 16, x^5 S-box): \
                    circuit size 2^nv vs number of permutations",
            gates: poseidon2_native_gates_per_permutation_mamabear(),
        },
        Entry {
            variant: "poseidon2_native_goldilocks",
            circuit: "Poseidon2 (Goldilocks native, x^7)",
            unit: "Poseidon2 permutation",
            unit_col: "#permutations",
            title: "Poseidon2 native Goldilocks circuit (width 16, x^7 S-box): \
                    circuit size 2^nv vs number of permutations",
            gates: poseidon2_native_gates_per_permutation_goldilocks(),
        },
        Entry {
            variant: "poseidon2_native_babybear",
            circuit: "Poseidon2 (BabyBear native, x^7)",
            unit: "Poseidon2 permutation",
            unit_col: "#permutations",
            title: "Poseidon2 native BabyBear circuit (width 16, x^7 S-box): \
                    circuit size 2^nv vs number of permutations",
            gates: poseidon2_native_gates_per_permutation_babybear(),
        },
    ]
}

fn run(variants: &[&str]) -> Vec<String> {
    let nv_min = env_usize("BENCH_NV_MIN", NV_MIN_DEFAULT);
    let nv_max = env_usize("BENCH_NV_MAX", NV_MAX_DEFAULT);
    assert!(
        nv_min <= nv_max,
        "BENCH_NV_MIN ({nv_min}) must be <= BENCH_NV_MAX ({nv_max})"
    );

    let entries: Vec<Entry> = all_entries()
        .into_iter()
        .filter(|e| variants.iter().any(|v| *v == e.variant))
        .collect();

    let mut out: Vec<String> = Vec::new();

    for e in &entries {
        out.push(summary_sentence(e.circuit, e.unit, e.gates));
    }

    for e in &entries {
        render_table(&mut out, e.title, e.unit_col, e.gates, nv_min, nv_max);
    }

    out
}

const ALL_VARIANTS: &[&str] = &[
    "sha256",
    "aes128",
    "blake3",
    "keccakf",
    "poseidon2_x11",
    "poseidon2_native_mamabear",
    "poseidon2_native_goldilocks",
    "poseidon2_native_babybear",
];

pub fn main() {
    let variant = std::env::args()
        .nth(1)
        .expect("usage: circuit_size <sha256|aes128|blake3|keccakf|poseidon2_x11|poseidon2_native_{mamabear,goldilocks,babybear}|all>");

    let variants: Vec<&str> = match variant.as_str() {
        "sha256"
        | "aes128"
        | "blake3"
        | "keccakf"
        | "poseidon2_x11"
        | "poseidon2_native_mamabear"
        | "poseidon2_native_goldilocks"
        | "poseidon2_native_babybear" => vec![variant_as_static(&variant)],
        "all" => ALL_VARIANTS.to_vec(),
        other => panic!(
            "unknown variant {other:?}; expected one of: {:?} or all",
            ALL_VARIANTS
        ),
    };

    let lines = run(&variants);

    for line in &lines {
        println!("{line}");
    }

    if let Ok(path) = std::env::var("BENCH_OUTPUT_FILE") {
        let mut f = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .unwrap_or_else(|e| panic!("BENCH_OUTPUT_FILE={path:?} open failed: {e}"));
        let _ = writeln!(f, "=== circuit_size ({variant}) ===");
        for line in &lines {
            let _ = writeln!(f, "{line}");
        }
        let _ = writeln!(f);
        let _ = f.flush();
    }
}

fn variant_as_static(s: &str) -> &'static str {
    for v in ALL_VARIANTS {
        if *v == s {
            return v;
        }
    }
    unreachable!("variant_as_static called with unknown variant {s:?}")
}
}
