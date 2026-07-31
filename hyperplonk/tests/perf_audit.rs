//! Performance-audit harness for the fair-gate benchmark.
//!
//! Measures wall-clock prove (Ext3 serial + parallel), verify, and proof size
//! for the add+mul degree-3 Full SNARK across target circuit sizes.
//!
//! Gated behind `PERF_AUDIT=1` so the normal `cargo test` run skips it. Run:
//!
//! ```text
//! PERF_AUDIT=1 PERF_PROV128=1 PERF_SAMPLES=5 BENCH_FAIRGATE_NVS=18,19,20 \
//!   RUSTFLAGS="-C target-cpu=native" \
//!   cargo test -p hyperplonk --release --test perf_audit perf_audit_fair_gate \
//!   -- --nocapture --test-threads=1
//! ```

use std::hint::black_box;
use std::time::{Duration, Instant};

use arithmetic::field::mamabear::{MamaBearScalarExt3, PackedMamaBearAVX512};

use hyperplonk::prover_mamabear::AlignedPoly;

use poly_commit::deepfold_mamabear::DeepFoldMamaBearParam;
use util::params::{
    mamabear::{GRINDING_BITS_EXT3_PROV_QUERY128, QUERY_NUM_CONJ96, QUERY_NUM_PROV_QUERY128},
    CODE_RATE_LOG,
};

type SEF3 = MamaBearScalarExt3;
#[allow(unused)]
type PBF = PackedMamaBearAVX512;

// ---------------------------------------------------------------------------
// Timing scaffold
// ---------------------------------------------------------------------------

fn samples() -> usize {
    std::env::var("PERF_SAMPLES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(3)
}

fn median<F: FnMut() -> Duration>(mut f: F) -> f64 {
    let n = samples();
    let _ = f(); // warmup
    let mut ts: Vec<Duration> = (0..n).map(|_| f()).collect();
    ts.sort();
    ts[n / 2].as_secs_f64() * 1000.0
}

fn record(line: &str) {
    println!("{line}");
    if let Ok(p) = std::env::var("PERF_OUT") {
        use std::io::Write as _;
        if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(&p) {
            let _ = writeln!(f, "{line}");
        }
    }
}

fn pp_at(nv: usize) -> DeepFoldMamaBearParam {
    if std::env::var("PERF_CONJ96").is_ok() {
        DeepFoldMamaBearParam::new(nv, CODE_RATE_LOG, QUERY_NUM_CONJ96, 3)
    } else {
        let mut pp = DeepFoldMamaBearParam::new(nv, CODE_RATE_LOG, QUERY_NUM_PROV_QUERY128, 3);
        pp.grinding_bits = GRINDING_BITS_EXT3_PROV_QUERY128;
        pp
    }
}

/// Largest native (add+mul) `num_perms` whose plain circuit still fits `2^nv`.
fn poseidon2_native_max_perms(nv: usize, per_perm: usize) -> usize {
    use hyperplonk::poseidon2_circuit::build_poseidon2_native_circuit;
    let mut np = (1usize << nv) / per_perm;
    let prev = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let max_np = loop {
        if np == 0 {
            break 0;
        }
        let n = np;
        let ok = std::panic::catch_unwind(|| {
            let _ = build_poseidon2_native_circuit::<SEF3>(n, nv);
        })
        .is_ok();
        if ok {
            break n;
        }
        np -= 1;
    };
    std::panic::set_hook(prev);
    max_np
}

// ---------------------------------------------------------------------------
// fair-gate: the matched degree-3 gate benchmark
// ---------------------------------------------------------------------------

#[test]
fn perf_audit_fair_gate() {
    if std::env::var("PERF_AUDIT").is_err() {
        eprintln!("perf_audit_fair_gate skipped (set PERF_AUDIT=1 to run)");
        return;
    }
    use hyperplonk::poseidon2_circuit::{
        build_poseidon2_native_circuit, poseidon2_native_gates_per_permutation_mamabear,
    };
    use hyperplonk::prover_mamabear::{setup_mamabear, ProverMamaBear};
    use hyperplonk::verifier_mamabear::VerifierMamaBear;

    let nvs: Vec<usize> = std::env::var("BENCH_FAIRGATE_NVS")
        .ok()
        .map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![18, 19, 20]);
    let prov = if std::env::var("PERF_PROV128").is_ok() { "prov128" } else { "conj96" };
    record(&format!(
        "# fair-gate ours: add+mul degree-3 Full SNARK (ProverMamaBear), Ext3, {}, samples={} (median)",
        prov,
        samples()
    ));
    let per_perm = poseidon2_native_gates_per_permutation_mamabear();
    for &nv in &nvs {
        let num_perms = poseidon2_native_max_perms(nv, per_perm);
        assert!(num_perms > 0, "fair-gate nv={nv}: no add+mul gate fits");
        let (circuit, witness) = build_poseidon2_native_circuit::<SEF3>(num_perms, nv);
        let pp = pp_at(nv);
        let (pk, vk) = setup_mamabear::<SEF3>(&circuit, &pp);
        let prover = ProverMamaBear { prover_key: pk };
        let verifier = VerifierMamaBear { verifier_key: vk };
        let build_ap = || {
            [
                AlignedPoly::from_sbf(&witness[0]),
                AlignedPoly::from_sbf(&witness[1]),
                AlignedPoly::from_sbf(&witness[2]),
            ]
        };
        let p_sanity = prover.prove(&pp, nv, build_ap());
        let proof_bytes = p_sanity.bytes.len();
        assert!(
            verifier.verify(&pp, nv, p_sanity.clone()),
            "fair-gate nv={nv}: honest verify rejected"
        );
        let prove_ser_ms = median(|| {
            let ap = build_ap();
            let t = Instant::now();
            let p = prover.prove(&pp, nv, ap);
            let e = t.elapsed();
            black_box(&p);
            e
        });
        let prove_par_ms = median(|| {
            let ap = build_ap();
            let t = Instant::now();
            let p = prover.prove_par(&pp, nv, ap);
            let e = t.elapsed();
            black_box(&p);
            e
        });
        let verify_ms = median(|| {
            let pc = p_sanity.clone();
            let t = Instant::now();
            let r = verifier.verify(&pp, nv, pc);
            let e = t.elapsed();
            black_box(r);
            e
        });
        let speedup = if prove_par_ms > 0.0 {
            prove_ser_ms / prove_par_ms
        } else {
            0.0
        };
        record(&format!(
            "fairgate addmul  ext3 nv={:<2} rows={:<8} prove_ser={:>9.2}ms prove_par={:>9.2}ms par_speedup={:>4.2}x verify={:>8.3}ms proof={}B",
            nv,
            num_perms * per_perm,
            prove_ser_ms,
            prove_par_ms,
            speedup,
            verify_ms,
            proof_bytes
        ));
    }
    record("# done fair-gate");
}
