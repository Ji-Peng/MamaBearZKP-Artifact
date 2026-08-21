# MamaBearZKP Artifact

Artifact for the CCS 2026 paper **"MamaBearZKP: A Holistic Co-design of Prime Fields and Proving Stacks for High-Throughput ZKP on Modern CPUs"** by Jipeng Zhang, Yanpei Guo, Tao Lu, Hao Cheng, and Jiaheng Zhang (National University of Singapore; Shandong University).

This project is built upon [DeepFold-Hyperplonk](https://github.com/paulguoyanpei/DeepFold-Hyperplonk) (commit `4148026`, Jan 23 2025). From a protocol perspective, our DeepFold implementation (`poly_commit/src/deepfold*.rs`) follows the protocol description in the DeepFold paper more closely than that starting point does.

Licensed under the MIT License (see `LICENSE`).

## What this artifact does

MamaBearZKP is a HyperPlonk polynomial IOP paired with the DeepFold polynomial commitment scheme, instantiated over the **MamaBear** prime `P = 2^49 - 2^34 + 1` in a unified `R = 2^52` Montgomery representation and optimized end to end for AVX-512IFMA.

The artifact reproduces **every measured table in the paper**: field microbenchmarks, the ZeroCheck and ProductCheck sub-protocols, DeepFold commit/open, end-to-end proving time, proof size, peak memory, gate counts, and the Plonky3 baselines. Two shell drivers regenerate all the data, and one Python script turns that data into a PDF whose tables carry **the paper's own numbers, in the paper's own order**, for direct side-by-side comparison.

## Requirements

### Hardware

- **x86_64 CPU with AVX-512IFMA.** This is a hard requirement, not an optimization: the MamaBear kernels have no scalar fallback.
- **RAM.** About 123 GiB runs the full `mu = 18..27` sweep on the reference machine; with less, cap the sweep via `BENCH_NV_MAX` (see [Running on a smaller machine](#running-on-a-smaller-machine)). Per-`mu` growth and which `mu = 27` cells do not fit are detailed in [Memory notes and the `mu = 27` cells](#memory-notes-and-the-mu--27-cells).
- **Disk.** About 18 GB free: roughly 10 GB for this workspace's `target/`, and 6 GB for the patched Plonky3 checkout under `external/` and its build artifacts.
- **Network.** Needed once, at first build, for `cargo fetch` and for the `git clone` of Plonky3. The benchmarks themselves run offline.

### Reference machine

All numbers in `results/` were measured on **Google Cloud `c4d-highmem-16`**: AMD EPYC 9B45 (Zen 5), 16 vCPU = **8 physical cores with SMT enabled** (16 logical CPUs), 123 GiB RAM, Ubuntu 24.04 LTS. (`123 GiB` is what `free` reports and what every `results/` header records; Google's machine-type page lists the same capacity as `126 GB` in decimal units.)

We strongly recommend renting exactly this instance type to reproduce as closely as possible. Other AVX-512IFMA x86_64 machines will run the artifact correctly, but absolute timings will differ.

### Software

| Component | Version used | Notes |
| --- | --- | --- |
| OS | Ubuntu 24.04.3 LTS | kernel 6.17.0-1020-gcp |
| Rust | `rustc 1.98.0-nightly (c397dae80 2026-07-02)` | **nightly required** (AVX-512IFMA intrinsics) |
| Python | 3.10+ | for `results/plot_tables.py`; standard library only |
| LaTeX | TeX Live 2023 or newer | only to render `results/tables.tex`; needs `acmart`, `tabularray`, `subcaption`, `booktabs` |

The exact toolchain of every recorded run is in the header of each file under `results/` (`# rustc:`, `# kernel:`, `# cpu:`, and so on).

```bash
rustup toolchain install nightly
rustup default nightly
# Debian/Ubuntu, for rendering the tables:
sudo apt-get install -y texlive-latex-recommended texlive-latex-extra texlive-fonts-recommended
```

## Build

`target-cpu=native` is required; without it the AVX-512IFMA paths are not generated and every MamaBear number will be wrong.

```bash
export RUSTFLAGS="-C target-cpu=native"
cargo build --release
```

## Reproducing the results

Three steps, in order. Everything a reviewer needs is here; the later sections are background.

### Step 0 -- smoke test (about 10 minutes)

Before committing to a long run, confirm the toolchain and AVX-512IFMA are working:

```bash
export RUSTFLAGS="-C target-cpu=native"
./reproduce_addmul.sh field
```

This regenerates `results/MamaBear/field.txt`, the field microbenchmark table. If it completes and the numbers land near the shipped file, the build is good.

### Step 1 -- run the two drivers (about 13 hours total)

```bash
export RUSTFLAGS="-C target-cpu=native"

./reproduce_addmul.sh all    # MamaBear + Goldilocks/BabyBear baselines   ~11 h
./reproduce_p3_rand.sh       # Plonky3 baselines (defaults to "all")      ~2 h
```

These two commands produce **all** the data behind the paper's tables. Nothing else needs to be run.

Wall clock on the reference machine, derived from the recorded per-cell timings at the default sweep (1 warmup + 5 timed samples per cell, one process per cell):

| Command | Approx. wall clock |
| --- | ---: |
| `./reproduce_addmul.sh all` | **~11 h** |
| `./reproduce_p3_rand.sh` (includes ~15 min of Plonky3 builds) | **~2 h** |
| **Total** | **~13 h** |

Budget more on a slower machine, and see [Running on a smaller machine](#running-on-a-smaller-machine) if RAM is the constraint. The last few `mu` values dominate: each `+1` in `mu` roughly doubles the work, so everything up to `mu = 24` finishes in a small fraction of the total. A per-category breakdown is in [Individual categories](#individual-categories-optional), which you do not need to read.

Both scripts write into `results/`, first saving any file they are about to regenerate as `<name>.committed`, so you can diff your run against the shipped measurement:

```bash
diff results/Plonky3/rand/time.txt.committed results/Plonky3/rand/time.txt
```

Every output file carries a self-describing header recording the CPU, memory, kernel, `rustc`, FRI configuration, warmup and sample counts, and worker count of the run that produced it. The two drivers word it slightly differently: files under `results/MamaBear/` carry `# rayon:` and state the serial estimator on their `# samples:` line, while `results/Plonky3/rand/*` carry `# threads:` plus an explicit `# estimator:` block covering both the serial and the parallel estimator.

### Step 2 -- generate the tables

```bash
cd results
python3 plot_tables.py      # reads results/**, writes tables.tex
pdflatex tables.tex          # -> tables.pdf
```

`tables.pdf` contains **10 tables, one per page**. They carry **the paper's table numbers and appear in the paper's order** -- Tables 1, 4, 5, 6, 9, 10, 11, 12, 13, 14 -- so Table 6 in `tables.pdf` is Table 6 in the paper and can be compared line by line. (The gaps are the paper's Tables 2, 3, 7 and 8, which are descriptive or analytically derived and correspond to no benchmark run.) Captions and labels are the paper's own.

This step needs no benchmark run: it works on the shipped `results/`, so you can check the whole rendering pipeline first, then re-run it after Step 1 to see your own measurements.

### Interpreting differences

**Timings move between runs. Movement of a few percent -- and up to about 5% -- is normal and is not a failed reproduction.** It comes from scheduling, DRAM and cache state, and thermal behaviour, not from the code. On different hardware, absolute times will differ substantially, which is expected. Serial cells on the reference machine are the steadiest, typically within 1-2%; parallel cells move more, which is why they are estimated as a minimum over eight processes (see [Measurement methodology](#measurement-methodology)).

**Proof sizes are deterministic.** Given the same code and FRI parameters they reproduce bit-for-bit on any machine. A changed proof size means something is genuinely different and is worth investigating.

**The claims to check are the ratios** -- the `M. vs. G./B./P.` speedup columns -- which are far more stable across machines than the absolute milliseconds, and which are what the paper's text asserts.

### Which table comes from where

You do not need this to reproduce; it is here so any single number can be traced to the command that produced it.

| Paper table | Content | Result file | Command |
| --- | --- | --- | --- |
| Table 1 | Representative prover times (intro teaser) | `MamaBear/hp_df.txt` | `reproduce_addmul.sh hp_df` |
| Table 4 | ZeroCheck / ProductCheck | `MamaBear/hp_zc.txt`, `hp_pc.txt` | `reproduce_addmul.sh zc pc` |
| Table 5 | DeepFold commit / open | `MamaBear/df.txt` | `reproduce_addmul.sh df` |
| Table 6 | End-to-end prover time (5-level split) | `MamaBear/hp_df.txt`, `Plonky3/rand/time.txt` | `reproduce_addmul.sh hp_df` + `reproduce_p3_rand.sh time` |
| Table 9 | Field microbenchmarks | `MamaBear/field.txt` | `reproduce_addmul.sh field` |
| Table 10 | Circuit gate counts | `MamaBear/hp_df_circuit.txt` | `reproduce_addmul.sh circuit` |
| Table 11 | Proof size (5-level split) | `MamaBear/hp_df_proof_size.txt`, `Plonky3/rand/proof_size.txt` | `reproduce_addmul.sh size` + `reproduce_p3_rand.sh proof_size` |
| Table 12 | End-to-end prover time (3-level split) | `MamaBear/hp_df.txt`, `Plonky3/rand/time.txt` | as Table 6 |
| Table 13 | Proof size (3-level split) | `MamaBear/hp_df_proof_size.txt` | as Table 11 |
| Table 14 | Peak memory | `MamaBear/peak_mem.txt`, `Plonky3/rand/peak_mem.txt` | `reproduce_addmul.sh mem` + `reproduce_p3_rand.sh peak_memory` |

`results/MamaBear/grind.txt` records FRI grinding cost; it is supporting data and feeds no table.

### Running on a smaller machine

Cap the sweep rather than adding swap. `BENCH_NV_MAX` clamps every category's ceiling:

```bash
BENCH_NV_MAX=24 ./reproduce_addmul.sh all   # ~36 GiB peak, roughly a quarter of the runtime
BENCH_NV_MAX=22 ./reproduce_addmul.sh all   # ~9 GiB peak
```

Rows above the cap are simply absent, and `plot_tables.py` renders them as `--`, exactly as the paper does for its own out-of-memory cells. Other useful knobs: `BENCH_NV_MAX`, `BENCH_SPLITS` (FFT split levels, default `3,4,5`), `BENCH_SAMPLES`, `BENCH_WARMUP`, `BENCH_ATTEMPT_OOM_CELLS` (see [Memory notes and the `mu = 27` cells](#memory-notes-and-the-mu--27-cells)).

`BENCH_NV_MIN` is read by `reproduce_p3_rand.sh` only. `reproduce_addmul.sh` always starts at `mu = 18`, so setting it there has no effect -- use `BENCH_NV_MAX` to shorten that sweep.

### Memory notes and the `mu = 27` cells

Peak resident set grows about 2x per `+1` in `mu` (the log circuit size). Measured for the end-to-end MamaBear prover at the default `split=5`:

| `mu` | 22 | 23 | 24 | 25 | 26 | 27 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| serial | 2.0 GiB | 4.0 GiB | 8.0 GiB | 16.0 GiB | 32.0 GiB | 64.0 GiB |
| parallel | 2.7 GiB | 5.3 GiB | 10.6 GiB | 21.3 GiB | 42.5 GiB | 85.0 GiB |

The Goldilocks and BabyBear baselines are about 4x heavier at the same `mu` (35.6 and 32.9 GiB at `mu = 24`) and stop there, which is why the sweep continues on MamaBear alone above 24. Do not read the two backends as one growth curve.

Those are the prover's own peak figures, measured by the dedicated `peak_memory` binary. The benchmark harness needs **more** than that at the same `mu`, because it holds the proving key and the witness across the timed samples: at `mu = 27` the harness peaks near 110 GiB where the prover alone peaks at 85 GiB.

At `mu = 27` the reference machine therefore completes only some cells, and the results file records exactly those:

| `mu = 27` cell | outcome on a 123 GiB machine |
| --- | --- |
| end-to-end serial, `BENCH_SPLITS=5` | completes (~110 GiB) |
| end-to-end serial, `BENCH_SPLITS=3` or `4` | does not fit -- **skipped** |
| end-to-end parallel, any split | does not fit -- **skipped** |
| proof size, all splits | complete |
| DeepFold serial, all splits | complete |
| DeepFold parallel, `split=4`/`5` | complete |
| DeepFold parallel, `split=3` | **OOM-killed after its `Commit Par` row** |
| peak memory serial, all splits | complete |
| peak memory parallel, `split=5` | complete |
| peak memory parallel, `split=3`/`4` | **OOM-killed after `witness_setup` + `pp_setup`** |

The two OOM-killed rows above are why `results/MamaBear/df.txt` has no `Open Par NV=27 split=3` line, and why the `split=3` and `split=4` parallel `nv=27` blocks of `results/MamaBear/peak_mem.txt` stop after two lines. **Those short blocks are the expected output, not a corrupted file.** They were re-measured on the reference machine and reproduce exactly: the kernel kills the process at an anonymous RSS of about 122.6 GiB out of 123 GiB (`Out of memory: Killed process ... (deepfold_mamabe)`), whereas the `split=5` parallel cell peaks at 111.6 GiB and survives. A partial cell still emits every row it reached before the kill, which is why they are attempted rather than skipped.

`reproduce_addmul.sh` **skips** the cells marked above rather than attempting them: the row ends up absent either way, but an attempt costs a full witness and proving-key setup before the kernel kills it, and the parallel cells would pay that eight times over. The run prints an `[omit]` line for each, so a skipped cell is distinguishable from one that was tried and failed. Set `BENCH_ATTEMPT_OOM_CELLS=1` to attempt them anyway -- the right knob if your machine has more memory than the reference one, since these ceilings are a measurement rather than a property of the protocol.

The DeepFold and peak-memory sweeps are deliberately **not** skipped at `mu = 27`, even though some of their cells die as tabulated above: those emit useful rows before they do (`Commit Par`, and the `witness_setup` / `pp_setup` high-water marks), so attempting them is not wasted. The end-to-end cells are skipped precisely because they emit nothing before dying.

A shallower FFT split leaves a larger round-0 codeword, so the LOW splits are the memory-hungry ones -- `split=3` needs more than `split=5`, not less.

With less than about 123 GiB, cap the sweep instead of letting it OOM -- see [Running on a smaller machine](#running-on-a-smaller-machine). **Do not add swap to get past a cap, and do not add swap to recover the OOM-killed `mu = 27` cells:** any run that touches swap produces a timing that is not comparable to the paper's. A prover whose 110 GiB working set is partly on disk is measuring the disk. The absent rows are the honest result, and the paper prints them as `--`.

## Measurement methodology

Background on how the numbers above are produced. Nothing here needs to be configured; both drivers set it themselves.

### Threading

Every parallel benchmark runs with **8 rayon workers** -- one per physical core of the reference machine (8 cores, 16 logical CPUs with SMT). Both drivers pin that count, and each result file records it on its `# rayon:` header line.

The pin is deliberate. rayon's own default is one worker per *logical* CPU, which would make the recorded thread count a property of whatever host you run on rather than a stated parameter of the experiment, and would let the two sides of a comparison table run at different worker budgets.

`RAYON_NUM_THREADS` still overrides it, and the scripts say so when it does -- useful for sweeping thread counts, but the resulting rows will not match `results/`. Measured on the reference machine at `mu = 20`, the MamaBear parallel prover moves only about 5% between 8 and 16 workers (it is close to the DRAM roofline, as the paper discusses), whereas Plonky3 moves about 40%. So on the baseline side especially, a different worker count reads as a failed reproduction rather than as a configuration difference.

### How parallel cells are estimated

Serial cells report the median of five timed runs inside one process. **Parallel cells report the minimum over eight separate processes** (`BENCH_PAR_REPS`, default 8), because the parallel timings on this platform vary *between* processes rather than within one: repeated runs of the Plonky3 parallel prover at `mu = 20` cluster at roughly 1.51, 1.75 and 2.80 s on an otherwise idle machine, so a within-process median reports whichever cluster that process happened to land in, and two consecutive runs of the same script disagreed by 1.9x. CPU pinning does not remove it.

The noise is one-sided -- scheduling can only make a run slower than the hardware allows -- which is when the minimum is the right estimator. It is applied to **both sides** of every parallel-vs-parallel comparison, so it cannot favour either. Our own parallel prover is steady to about 8%, so this barely moves our numbers; it is there because the baseline needs it.

The `results/Plonky3/rand/*` headers spell this out on an `# estimator:` line; the `results/MamaBear/*` headers record the warmup and sample counts and the serial estimator, and the parallel cells there use the same minimum-over-processes rule described here.

One exception is the field microbenchmarks: the shipped `results/MamaBear/field.txt` was captured with 25 timed samples per cell (warmup 3) to resolve a bimodal `BabyBearAVX-512Ext4` add cell. Regenerating it with the default 1 warmup + 5 samples is fine -- the reported medians are stable across both settings -- but the header will then record `samples: 5` instead of `samples: 25`.

### Security configuration

Every benchmark runs the same FRI regime: **88 queries + 16 grinding bits at code rate 1/8** (`PROV_QUERY128`). The name refers to the query-path target `S_query = 16 + 88 x 1.2776 = 128.4` bits; the total provable level is `min(commit, query)`, and for MamaBear over `F_p^3` it is commit-bound at roughly 106-109 bits over `mu = 18..20`. Each result file's header carries the full derivation, including which of the two bounds binds.

### Challenge sampling

One detail of the Fiat-Shamir implementation is worth stating explicitly, because it is the kind of thing a careful reader will compute for themselves and it looks worse at first glance than it is.

Field challenges are drawn by reducing transcript bytes modulo `P` (`MamaBearScalar::from_uniform_bytes` in `arithmetic/src/field/mamabear.rs`). Reducing a power-of-two range modulo a non-power-of-two is never exactly uniform. Writing `2^64 = qP + r`, the `r` smallest residues get one extra preimage each, and because `P = 2^49 - 2^34 + 1` is a Solinas prime we have `r = 2^64 mod P = 2^34 - 2^15 - 1`, so `r/P = 2^-15`. That is larger than a random prime of this size would give, and it is forced by exactly the sparsity that makes reduction cheap.

The measure that matters for soundness is the pointwise probability ratio, which is `1 + 2^-15` at worst (and `1 - 2^-30` at best -- most residues are essentially exactly uniform). For `k` independent components this composes multiplicatively, so for any event `B`, including "the verifier accepts a false statement":

```
Pr_biased[B] <= (1 + 2^-15)^k * Pr_uniform[B]
```

The total variation distance is `2^-30`, but reading that additively gives a valid and badly loose answer -- it would say a `2^-100` cheating probability becomes `2^-30`. The bias here is diffuse rather than concentrated, which is precisely the case where the multiplicative bound is the right one.

Measured by instrumenting the transcript over a full prove: at `mu = 20` the add/mul SNARK draws 315 extension-field challenges, each consuming three base components, so `k = 945` and the factor is `(1 + 2^-15)^945 = 1.0293`. **The total soundness loss is 0.042 bit.** (The count does not depend on the query count -- `query_num` feeds only the query-position draw, never a field-challenge loop -- so this holds at the 88 queries every benchmark here uses.) Charging that `mu = 20` budget against the `mu = 19` provable level of 107.6 bits over-states the loss, since `mu = 19` draws fewer challenges, and even so 107.6 - 0.042 = 107.558 still prints as 107.6. No figure in `results/` or in the paper moves.

What is *not* affected: the FRI query positions carry no bias at all. They are reduced modulo the evaluation-domain size, which is a power of two, so that reduction is an exact truncation. The query-path term `S_query = 128.4` and the grinding are therefore exactly as stated; only the commit-side bound `C` sees biased randomness.

We have deliberately not changed this in the artifact. Widening the draw is cheap -- the function already receives 32 uniform bytes and uses 8, so 10 bytes per component would fit three components in one digest with no extra hashing and take the ratio to `1 + 2^-31`. But changing a challenge *value* changes the Fiat-Shamir chain, hence the query indices, hence Merkle-path deduplication, hence the proof bytes and the recorded proof sizes. An artifact whose job is to reproduce the paper's numbers should not stop reproducing them in exchange for a fraction of a bit that does not survive rounding. The full derivation, including the measured per-`mu` counts, is in the source at `arithmetic/src/field/mamabear.rs` and `util/src/params.rs`.

One caveat if you are adapting this code rather than reproducing it: the 0.042 bit is a *soundness* figure and relies on this system being a succinct argument with no zero-knowledge claim. A distinguishing claim needs the additive `k * TV = 2^-20.1` instead. If you add blinding on top of this sampler, widen the draw first.

### Individual categories (optional)

**Skip this section if you ran Step 1.** The two commands there already run every category; nothing below is additionally required. It is here only for re-running one table after a failure, or for a partial reproduction on a time budget.

`reproduce_addmul.sh` takes one or more **space-separated** specs, each `CATEGORY[,SUB,...]`. The categories are `zc`, `pc`, `df`, `hp_df`, `size`, `mem`, `circuit`, `grind`, `field`, and `all`. For `zc`, `pc`, `df`, and `hp_df` the optional subs select the backend: `baby`, `gold`, `own` (MamaBear), or `all` (the default). `reproduce_p3_rand.sh` takes a single argument: `time`, `proof_size`, `peak_memory`, or `all`.

```bash
./reproduce_addmul.sh                 # no argument -> prints the full usage text
./reproduce_addmul.sh field           # quick smoke test, exercises AVX-512IFMA
./reproduce_addmul.sh zc pc           # ZeroCheck + ProductCheck (two specs, space-separated)
./reproduce_addmul.sh zc,own          # ZeroCheck, MamaBear only (one spec with a sub)
./reproduce_addmul.sh hp_df           # end-to-end HyperPlonk + DeepFold, all backends
```

Note the comma is a category/sub separator, not a category list: `zc,pc` asks for ZeroCheck with a sub named `pc`, which does not exist and exits with `Unknown zc sub-option: pc`.

Per-category wall clock on the reference machine:

The `mu` ranges below differ per backend, because the Goldilocks and BabyBear baselines exhaust memory well before MamaBear does. The MamaBear ceiling is given first, the baseline ceiling in parentheses.

| Command | Category | `mu` range (MamaBear / baselines) | Approx. wall clock |
| --- | --- | --- | ---: |
| `reproduce_addmul.sh` | `zc` (ZeroCheck) | 18-28 / 18-28 | 1.5 h |
| | `pc` (ProductCheck) | 18-28 / 18-27 | 2.1 h |
| | `df` (DeepFold commit/open) | 18-27 / 18-24 | 2.0 h |
| | `hp_df` (end-to-end) | 18-27 / 18-24 | 2.6 h |
| | `size` + `mem` (proof size, peak memory) | 18-27 / 18-24 | 1.5-2.5 h |
| | `circuit` + `grind` + `field` | n/a | 0.5 h |
| `reproduce_p3_rand.sh` | `time` + `peak_memory` | 18-24 (Plonky3) | 1.5-2 h |

At `mu = 27` the `hp_df` sweep additionally skips the cells that cannot fit; see [Memory notes and the `mu = 27` cells](#memory-notes-and-the-mu--27-cells).

## Running the tests

```bash
export RUSTFLAGS="-C target-cpu=native"
cargo test --release
```

Expected: `arithmetic` 55 passed, `hyperplonk` 79 passed, `poly_commit` 30 passed, `util` 21 passed, `perf_audit` 1 passed, all doctests clean, and **0 failed** anywhere. The suite takes a few minutes, most of it in `hyperplonk`.

A handful of memory-hungry measurement sweeps are marked `#[ignore]` and excluded from that run; they print cost tables and assert nothing. Run them explicitly with `cargo test --release -- --ignored --nocapture` on a machine with ample RAM.

## Code structure

```
.cargo/            Cargo build configuration
arithmetic/        Field arithmetic: MamaBear (P = 2^49-2^34+1), BabyBear, Goldilocks;
                   AVX-512IFMA kernels and FFT
hyperplonk/        HyperPlonk PIOP: sumcheck, ZeroCheck, ProductCheck, prover, verifier
  benches/         Benchmark harnesses, grouped into piop/, pcs/, e2e/
  src/bin/         Utility binaries: circuit_size, peak_memory, proof_size, bench_config,
                   profile_hp_df_{goldilocks,mamabear}
  tests/           perf_audit.rs -- gated prove/verify/proof-size harness for
                   the degree-3 add+mul gate (PERF_AUDIT=1; not part of a
                   normal test run and not needed for any table)
poly_commit/       Polynomial commitments: DeepFold (base + extension field, serial + par),
                   BaseFold, KZG
util/              Fiat-Shamir transcript, Merkle trees, FRI parameters
patches/           Patches applied to Plonky3 for the baseline comparison
results/           Recorded benchmark results + plot_tables.py (generates all paper tables)
scripts/lib/       Shared shell harness used by both reproduce scripts
reproduce_addmul.sh    Main driver: MamaBear plus the Goldilocks/BabyBear baselines
reproduce_p3_rand.sh   Plonky3 baseline driver: degree-4 random-AIR sweep
```

## Contact

- Jipeng Zhang -- jp-zhang@outlook.com
- Yanpei Guo -- guo.yanpei@u.nus.edu
- Tao Lu -- lutaocc2020@gmail.com
- Hao Cheng -- hao.cheng@sdu.edu.cn
- Jiaheng Zhang -- jhzhang@nus.edu.sg
