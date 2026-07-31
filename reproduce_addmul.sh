#!/usr/bin/env bash
set -euo pipefail

# reproduce_addmul.sh -- core PIOP / PCS / end-to-end benchmark driver.
#
# Regenerates the per-metric result files for the MamaBear prover backend:
#
#   results/MamaBear/   AVX-512IFMA, P = 2^49 - 2^34 + 1, x86_64
#
# Usage:
#   ./reproduce_addmul.sh SPEC [SPEC ...]
#
# SPEC is CATEGORY[,SUB[,SUB...]]; a bare CATEGORY means CATEGORY,all.
#
#   zc        ZeroCheck              subs: baby / gold / own / all
#   pc        ProductCheck           subs: baby / gold / own / all
#   df        DeepFold PCS alone     subs: baby / gold / own / all
#   hp_df     HyperPlonk + DeepFold  subs: baby / gold / own / all
#   size      proof size             (machine-independent)
#   mem       peak heap memory
#   circuit   gate counts            subs: a circuit name, or all
#                                    (machine- AND field-independent)
#   grind     FRI grinding cost      subs: 16 / 20 / 24 / all
#   field     field-arithmetic microbench
#   all       every category
#
# "own" means MamaBear; "baby" and "gold" are the BabyBear and Goldilocks
# comparison baselines.
#
# Security regime: PROV_QUERY128 only (88 FRI queries + 16 grinding bits at code
# rate 1/8). It is the default in the benches, so nothing here passes
# BENCH_SECURITY. The weaker conj96 / prov97 regimes were dropped from this
# driver: they understate every cost and are not what the paper reports.
#
# Environment:
#   BENCH_NV_MAX            global clamp on every per-category nv ceiling
#   BENCH_SPLITS            DeepFold split levels to sweep (default 3,4,5)
#   BENCH_WARMUP            untimed warmup iterations (default 1)
#   BENCH_SAMPLES           measured iterations, median reported (default 5)
#   RAYON_NUM_THREADS       worker count for the parallel variants (default 8,
#                           one per physical core; overriding it means the rows
#                           will not match results/)
#   BENCH_MACHINE_CLASS     free-text machine description for the file header
#   BENCH_FREQ_NOTE         free-text core-frequency note for the file header
#   BENCH_APPEND=1          append to existing files instead of truncating
#
# Output files are truncated once per run and carry a self-describing header:
# machine, toolchain, git revision, and the resolved SNARK configuration read
# out of the binary rather than written here by hand.

ORIG_ARGS="$*"
SCRIPT_NAME="reproduce_addmul.sh"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck source=scripts/lib/bench_common.sh
. "$SCRIPT_DIR/scripts/lib/bench_common.sh"

cd "$SCRIPT_DIR"

BACKEND_RESOLVED="$(bc_detect_backend)"
# BENCH_RESULTS_DIR redirects the whole output tree. Verification runs use it to
# write into a scratch directory, so a structural smoke test at tiny nv can never
# overwrite a recorded measurement.
RESULTS_ROOT="${BENCH_RESULTS_DIR:-$SCRIPT_DIR/results}"
OUTDIR="$RESULTS_ROOT/$(bc_backend_outdir "$BACKEND_RESOLVED")"

NV_MIN=18
NV_MAX="${BENCH_NV_MAX:-28}"
SPLITS="${BENCH_SPLITS:-3,4,5}"

export RUSTFLAGS="${RUSTFLAGS:--C target-cpu=native}"
# The heavy end-to-end and PoP paths recurse deeply enough to overflow the
# default 8 MiB thread stack.
export RUST_MIN_STACK="${RUST_MIN_STACK:-536870912}"

# Export the harness knobs with explicit defaults. The bench targets have
# DIFFERENT built-in defaults (5 for most, 10 for the field microbenches), so
# leaving these unset would let each cell pick its own while the file header
# asserted a single number.
export BENCH_SAMPLES="${BENCH_SAMPLES:-5}"
export BENCH_WARMUP="${BENCH_WARMUP:-1}"

START_EPOCH=$(date +%s)

# ---------------------------------------------------------------------------
# Output files
# ---------------------------------------------------------------------------
ZC_OUT="$OUTDIR/hp_zc.txt"
PC_OUT="$OUTDIR/hp_pc.txt"
DF_OUT="$OUTDIR/df.txt"
HP_DF_OUT="$OUTDIR/hp_df.txt"
SIZE_OUT="$OUTDIR/hp_df_proof_size.txt"
MEM_OUT="$OUTDIR/peak_mem.txt"
CIRCUIT_OUT="$OUTDIR/hp_df_circuit.txt"
GRIND_OUT="$OUTDIR/grind.txt"
FIELD_OUT="$OUTDIR/field.txt"

# ---------------------------------------------------------------------------
# Cell runners
# ---------------------------------------------------------------------------

# cap CEILING -- clamp a per-category nv ceiling by the global BENCH_NV_MAX.
cap() { local c="$1"; (( c < NV_MAX )) && echo "$c" || echo "$NV_MAX"; }

# bench_nv BENCH OUTFILE NV [ENV=VAL ...]
#   One nv per process. Isolating each cell keeps heap fragmentation from one
#   large nv out of the next, and lets a single OOM-killed cell leave its row
#   absent without taking the rest of the sweep with it.
bench_nv() {
    local bench="$1" out="$2" nv="$3"; shift 3
    bc_cell "$bench nv=$nv ${*:-}" -- \
        env "$@" \
            BENCH_NV_MIN="$nv" BENCH_NV_MAX="$nv" \
            BENCH_OUTPUT_FILE="$out" \
            cargo bench -p hyperplonk --bench "$bench"
}

# bench_nv_min BENCH OUTFILE NV [ENV=VAL ...]
#   Same cell, but reported as the MINIMUM over BC_PAR_REPS processes rather
#   than the median within one -- see the estimator note in bench_common.sh.
#   For parallel cells only: the noise they carry is between processes, so a
#   within-process median reports whichever cluster that process landed in.
#
#   Each repetition runs one warmup and one timed sample, so the repetition
#   budget moves from within-process samples to across-process repetitions
#   rather than adding to it. Repetitions land in a scratch file and only the
#   per-label minimum reaches the result file.
bench_nv_min() {
    local bench="$1" out="$2" nv="$3"; shift 3
    local tmp rep
    tmp="$(mktemp)"
    for rep in $(seq 1 "$BC_PAR_REPS"); do
        bc_cell "$bench nv=$nv ${*:-} [rep $rep/$BC_PAR_REPS]" -- \
            env "$@" \
                BENCH_NV_MIN="$nv" BENCH_NV_MAX="$nv" \
                BENCH_SAMPLES=1 BENCH_WARMUP=1 \
                BENCH_OUTPUT_FILE="$tmp" \
                cargo bench -p hyperplonk --bench "$bench"
    done
    bc_min_rows "$out" "$tmp" >/dev/null
    rm -f "$tmp"
}

# bin_nv BIN VARIANT OUTFILE NV [ENV=VAL ...]
bin_nv() {
    local bin="$1" variant="$2" out="$3" nv="$4"; shift 4
    bc_cell "$bin $variant nv=$nv ${*:-}" -- \
        env "$@" \
            BENCH_NV_MIN="$nv" BENCH_NV_MAX="$nv" \
            BENCH_OUTPUT_FILE="$out" \
            cargo run --release -q -p hyperplonk --bin "$bin" -- "$variant"
}

# expand_subs SUBS ALL_SET
expand_subs() {
    if [[ "$1" == "all" ]]; then echo "$2"; else echo "$1" | tr ',' ' '; fi
}

# splits_list -- BENCH_SPLITS as a space-separated list.
splits_list() { echo "$SPLITS" | tr ',' ' '; }

# begin_block FILE
#   Start a new row group with a blank-line separator.
begin_block() {
    bc_begin_block "$1"
}

# skip_category CATEGORY REASON
skip_category() {
    echo "  [skip] category '$1' is not available on backend '$BACKEND_RESOLVED': $2"
}

# =========================================================================
# zc -- ZeroCheck
# =========================================================================

zc_baby() {
    begin_block "$ZC_OUT"
    local c; c=$(cap 28)
    for nv in $(seq "$NV_MIN" "$c"); do bench_nv zerocheck_babybear "$ZC_OUT" "$nv"; done
}

zc_gold() {
    begin_block "$ZC_OUT"
    local c; c=$(cap 28)
    for nv in $(seq "$NV_MIN" "$c"); do bench_nv zerocheck_goldilocks "$ZC_OUT" "$nv"; done
}

# zc_mama_variants -- one pass over the nv range per prover variant.
#
# The variant flags are presence-only: setting one restricts the bench to that
# kind. Running them as separate passes is what groups the file by variant
# (all nv for one variant, then all nv for the next) instead of interleaving
# every variant at each nv.
# One pass per variant the bench implements. The Legacy pass is not optional
# bookkeeping: results/MamaBear/hp_zc.txt carries 11 `Legacy Ext3` rows, and
# without this loop the driver cannot regenerate its own result file.
zc_mama_variants() {
    local c; c=$(cap 28)
    for nv in $(seq "$NV_MIN" "$c"); do
        bench_nv zerocheck_mamabear "$ZC_OUT" "$nv" BENCH_LEGACY_EXT3_ONLY=1
    done
    local ell0 pass
    for ell0 in 1 2 3; do
        for pass in BENCH_OPT_EXT3_ONLY BENCH_OPT_EXT3_PAR_ONLY; do
            for nv in $(seq "$NV_MIN" "$c"); do
                bench_nv zerocheck_mamabear "$ZC_OUT" "$nv" "$pass=1" ELL0="$ell0"
            done
        done
    done
}

zc_own() {
    begin_block "$ZC_OUT"
    zc_mama_variants
}

dispatch_zc() {
    local subs; subs=$(expand_subs "$1" "baby gold own")
    for sub in $subs; do
        case "$sub" in
            baby) zc_baby ;;
            gold) zc_gold ;;
            own)  zc_own ;;
            *) echo "Unknown zc sub-option: $sub" >&2; exit 1 ;;
        esac
    done
}

# =========================================================================
# pc -- ProductCheck
# =========================================================================

pc_baby() {
    begin_block "$PC_OUT"
    local c; c=$(cap 27)
    for nv in $(seq "$NV_MIN" "$c"); do bench_nv prodcheck_babybear "$PC_OUT" "$nv"; done
}

pc_gold() {
    begin_block "$PC_OUT"
    local c; c=$(cap 27)
    for nv in $(seq "$NV_MIN" "$c"); do bench_nv prodcheck_goldilocks "$PC_OUT" "$nv"; done
}

pc_own() {
    begin_block "$PC_OUT"
    local c; c=$(cap 28)
    # One process per nv emits every per-wire variant for that nv, which is why
    # the rows come out grouped by nv rather than by variant.
    for nv in $(seq "$NV_MIN" "$c"); do bench_nv prodcheck_mamabear "$PC_OUT" "$nv"; done
}

dispatch_pc() {
    local subs; subs=$(expand_subs "$1" "baby gold own")
    for sub in $subs; do
        case "$sub" in
            baby) pc_baby ;;
            gold) pc_gold ;;
            own)  pc_own ;;
            *) echo "Unknown pc sub-option: $sub" >&2; exit 1 ;;
        esac
    done
}

# =========================================================================
# df -- DeepFold PCS in isolation (commit + open)
# =========================================================================

df_baby() {
    begin_block "$DF_OUT"
    # The 31-bit baselines are ~8x slower per cell than the wide-field prover
    # and OOM past nv=24 on the 123 GiB reference machine; that ceiling is
    # measured, not chosen.
    local c; c=$(cap 24)
    for nv in $(seq "$NV_MIN" "$c"); do bench_nv deepfold_babybear "$DF_OUT" "$nv"; done
}

df_gold() {
    begin_block "$DF_OUT"
    local c; c=$(cap 24)
    for nv in $(seq "$NV_MIN" "$c"); do bench_nv deepfold_goldilocks "$DF_OUT" "$nv"; done
}

df_own() {
    # MamaBear sweeps the split level: one block per split, serial rows then
    # parallel rows within the block.
    local split c
    c=$(cap 27)
    for split in $(splits_list); do
        begin_block "$DF_OUT"
        for nv in $(seq "$NV_MIN" "$c"); do
            bench_nv deepfold_mamabear "$DF_OUT" "$nv" BENCH_SPLIT="$split"
        done
        for nv in $(seq "$NV_MIN" "$c"); do
            bench_nv deepfold_mamabear_par "$DF_OUT" "$nv" BENCH_SPLIT="$split"
        done
    done
}

dispatch_df() {
    local subs; subs=$(expand_subs "$1" "baby gold own")
    for sub in $subs; do
        case "$sub" in
            baby) df_baby ;;
            gold) df_gold ;;
            own)  df_own ;;
            *) echo "Unknown df sub-option: $sub" >&2; exit 1 ;;
        esac
    done
}

# =========================================================================
# hp_df -- HyperPlonk + DeepFold end to end
# =========================================================================

# The baseline sweeps prove a random add/mul circuit at nv=18 and SHA-256 from
# nv=19 up, because below nv=19 a SHA-256 circuit holds zero full blocks.
hp_df_baseline_bench() {
    local field="$1" nv="$2"
    if (( nv == 18 )); then echo "hp_df_${field}"; else echo "hp_df_${field}_sha256"; fi
}

hp_df_baby() {
    begin_block "$HP_DF_OUT"
    local c; c=$(cap 24)
    for nv in $(seq "$NV_MIN" "$c"); do
        bench_nv "$(hp_df_baseline_bench babybear "$nv")" "$HP_DF_OUT" "$nv"
    done
}

hp_df_gold() {
    begin_block "$HP_DF_OUT"
    local c; c=$(cap 24)
    for nv in $(seq "$NV_MIN" "$c"); do
        bench_nv "$(hp_df_baseline_bench goldilocks "$nv")" "$HP_DF_OUT" "$nv"
    done
}

# The one split whose SERIAL end-to-end cell still fits at nv=27.
HP_DF_NV27_SERIAL_SPLIT=5

# hp_df_mama_ceiling MODE SPLIT CEILING
#   The highest nv the (MODE, SPLIT) pair actually completes at, given the
#   sweep ceiling CEILING.
#
#   nv=27 is where this benchmark runs out of memory on the reference machine
#   (GCP c4d-highmem-16, 123 GiB), and it does NOT run out uniformly, so the
#   sweep is shaped rather than square. Measured:
#
#       mode      split    nv=27
#       -------------------------------------------------
#       serial    5        completes, peaks near 110 GiB
#       serial    3, 4     OOM-killed
#       parallel  any      OOM-killed
#
#   Two things drive that, and both are counter-intuitive enough to write down.
#
#   (1) A SHALLOWER split needs MORE memory, not less: it leaves a larger
#       round-0 codeword. So split=3 is the hungry one and split=5 the lean one,
#       which is the opposite of the ordering the split level suggests.
#
#   (2) The benchmark harness needs more than the prover. results/MamaBear/
#       peak_mem.txt reports 64 GiB for the serial prove and 85 GiB for the
#       parallel prove at nv=27, but those come from a dedicated binary that
#       proves once. This harness holds the proving key and witness across the
#       timed samples and peaks near 110 GiB, so cells that look affordable in
#       the peak-memory table are still killed here.
#
#   These cells are SKIPPED rather than attempted. The outcome is the same
#   either way -- an absent row, which is this file's convention for a cell that
#   did not complete -- but attempting costs a full witness and proving-key
#   setup before the kernel steps in, and the parallel cells pay that
#   BENCH_PAR_REPS times over. Skipping turned roughly an hour of guaranteed
#   waste into nothing.
#
#   Do NOT add swap to recover them: a ~110 GiB working set partly on disk
#   measures the disk, not the prover.
#
#   BENCH_ATTEMPT_OOM_CELLS=1 attempts them anyway. That is the right knob on a
#   machine with more memory than the reference one -- the ceilings below are a
#   measurement, not a property of the protocol.
hp_df_mama_ceiling() {
    local mode="$1" split="$2" ceiling="$3"
    if [[ "${BENCH_ATTEMPT_OOM_CELLS:-0}" == "1" ]] || (( ceiling < 27 )); then
        echo "$ceiling"; return
    fi
    if [[ "$mode" == serial && "$split" == "$HP_DF_NV27_SERIAL_SPLIT" ]]; then
        echo 27
    else
        echo 26
    fi
}

# hp_df_mama_announce_skip MODE SPLIT FROM TO
#   An omitted cell and a failed cell both leave the row absent, so say which
#   this was; a silent gap reads as an untested configuration.
hp_df_mama_announce_skip() {
    local mode="$1" split="$2" from="$3" to="$4"
    (( from > to )) && return 0
    echo "  [omit] hp_df mamabear $mode split=$split nv=$from..$to --" \
         "OOM-killed on the reference machine; not attempted" \
         "(BENCH_ATTEMPT_OOM_CELLS=1 to try anyway)"
}

hp_df_mama() {
    local split c nv bench ser_c par_c
    c=$(cap 27)
    for split in $(splits_list); do
        ser_c=$(hp_df_mama_ceiling serial   "$split" "$c")
        par_c=$(hp_df_mama_ceiling parallel "$split" "$c")

        # Serial and parallel are separate blocks here (unlike df), matching the
        # committed layout.
        begin_block "$HP_DF_OUT"
        for nv in $(seq "$NV_MIN" "$ser_c"); do
            if (( nv == 18 )); then bench="hp_df_mamabear"; else bench="hp_df_mamabear_sha256"; fi
            bench_nv "$bench" "$HP_DF_OUT" "$nv" BENCH_SPLIT="$split"
        done
        hp_df_mama_announce_skip serial "$split" "$((ser_c + 1))" "$c"

        # Parallel cells use the min-of-BC_PAR_REPS estimator: this column is
        # compared directly against Plonky3's parallel column, which carries
        # large between-process variance, so both sides must be estimated the
        # same way. See bench_common.sh.
        begin_block "$HP_DF_OUT"
        for nv in $(seq "$NV_MIN" "$par_c"); do
            if (( nv == 18 )); then bench="hp_df_mamabear_par"; else bench="hp_df_mamabear_sha256_par"; fi
            bench_nv_min "$bench" "$HP_DF_OUT" "$nv" BENCH_SPLIT="$split"
        done
        hp_df_mama_announce_skip parallel "$split" "$((par_c + 1))" "$c"
    done
}

dispatch_hp_df() {
    local subs; subs=$(expand_subs "$1" "baby gold own")
    for sub in $subs; do
        case "$sub" in
            baby) hp_df_baby ;;
            gold) hp_df_gold ;;
            own)  hp_df_mama ;;
            *) echo "Unknown hp_df sub-option: $sub" >&2; exit 1 ;;
        esac
    done
}

# =========================================================================
# size -- proof size (deterministic, machine-independent)
# =========================================================================

dispatch_size() {
    bc_open_file "$SIZE_OUT"
    # No blank-line separators here: the committed file is one contiguous run.
    local nv c split variant
    c=$(cap 24)
    for nv in $(seq "$NV_MIN" "$c"); do
        bin_nv proof_size "$(hp_df_baseline_bench goldilocks "$nv")" "$SIZE_OUT" "$nv"
    done
    for nv in $(seq "$NV_MIN" "$c"); do
        bin_nv proof_size "$(hp_df_baseline_bench babybear "$nv")" "$SIZE_OUT" "$nv"
    done
    # Proof size is identical for the serial and parallel provers -- they are
    # held byte-identical -- so only the serial variants are measured.
    c=$(cap 27)
    for split in $(splits_list); do
        for nv in $(seq "$NV_MIN" "$c"); do
            if (( nv == 18 )); then variant="hp_df_mamabear"; else variant="hp_df_mamabear_sha256"; fi
            bin_nv proof_size "$variant" "$SIZE_OUT" "$nv" BENCH_SPLIT="$split"
        done
    done
}

# =========================================================================
# mem -- peak heap memory per phase
# =========================================================================

dispatch_mem() {
    bc_open_file "$MEM_OUT"
    local nv c split variant
    c=$(cap 24)
    for nv in $(seq "$NV_MIN" "$c"); do
        bin_nv peak_memory "$(hp_df_baseline_bench goldilocks "$nv")" "$MEM_OUT" "$nv"
    done
    for nv in $(seq "$NV_MIN" "$c"); do
        bin_nv peak_memory "$(hp_df_baseline_bench babybear "$nv")" "$MEM_OUT" "$nv"
    done
    c=$(cap 27)
    for split in $(splits_list); do
        for nv in $(seq "$NV_MIN" "$c"); do
            if (( nv == 18 )); then variant="hp_df_mamabear"; else variant="hp_df_mamabear_sha256"; fi
            bin_nv peak_memory "$variant" "$MEM_OUT" "$nv" BENCH_SPLIT="$split"
        done
        for nv in $(seq "$NV_MIN" "$c"); do
            if (( nv == 18 )); then variant="hp_df_mamabear_par"; else variant="hp_df_mamabear_sha256_par"; fi
            bin_nv peak_memory "$variant" "$MEM_OUT" "$nv" BENCH_SPLIT="$split"
        done
    done
}

# =========================================================================
# circuit -- gate counts
# =========================================================================
#
# Gate counts are pure circuit structure: no field, no FRI instance, no machine.

CIRCUIT_ALL="sha256 aes128 blake3 keccakf poseidon2_x11 poseidon2_native_mamabear poseidon2_native_goldilocks poseidon2_native_babybear"

dispatch_circuit() {
    bc_open_file "$CIRCUIT_OUT"
    local variants; variants=$(expand_subs "$1" "$CIRCUIT_ALL")
    local v
    for v in $variants; do
        bc_cell "circuit_size $v" -- \
            env BENCH_OUTPUT_FILE="$CIRCUIT_OUT" \
                cargo run --release -q -p hyperplonk --bin circuit_size -- "$v"
    done
}

# =========================================================================
# grind -- FRI grinding (proof-of-work) cost
# =========================================================================

dispatch_grind() {
    bc_open_file "$GRIND_OUT"
    local bits; bits=$(expand_subs "$1" "16 20 24")
    local b
    for b in $bits; do
        bc_cell "grinding_bench bits=$b" -- \
            env GRINDING_BITS="$b" GRINDING_MODE=both \
                BENCH_OUTPUT_FILE="$GRIND_OUT" \
                cargo bench -p poly_commit --bench grinding_bench
    done
}

# =========================================================================
# field -- field-arithmetic microbenchmarks
# =========================================================================

dispatch_field() {
    bc_open_file "$FIELD_OUT"
    local benches="goldilocks_bench babybear_bench mamabear_bench babybear_avx512_bench"
    local b
    for b in $benches; do
        bc_cell "arithmetic $b" -- \
            env BENCH_OUTPUT_FILE="$FIELD_OUT" \
                cargo bench -p arithmetic --bench "$b"
    done
}

# =========================================================================
# Main
# =========================================================================

usage() { sed -n '4,60p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

run_all() {
    # Cheapest first, so a setup problem shows up in minutes rather than hours.
    dispatch_field all
    dispatch_grind all
    dispatch_circuit all
    dispatch_size all
    dispatch_mem all
    dispatch_zc all
    dispatch_pc all
    dispatch_df all
    dispatch_hp_df all
}

if [[ $# -eq 0 ]]; then usage; exit 0; fi

bc_pin_regime
bc_pin_threads
BC_RUN_HEADER="$(bc_init_header "$SCRIPT_NAME" "$BACKEND_RESOLVED" "$ORIG_ARGS")"
echo "$BC_RUN_HEADER"
echo "# backend:   $BACKEND_RESOLVED  ->  $OUTDIR"

trap 'bc_summary "$SCRIPT_NAME" "$START_EPOCH"' EXIT

for spec in "$@"; do
    if [[ "$spec" == "all" ]]; then run_all; continue; fi
    category="${spec%%,*}"
    if [[ "$spec" == *,* ]]; then rest="${spec#*,}"; else rest="all"; fi
    case "$category" in
        zc)       dispatch_zc       "$rest" ;;
        pc)       dispatch_pc       "$rest" ;;
        df)       dispatch_df       "$rest" ;;
        hp_df)    dispatch_hp_df    "$rest" ;;
        size)     dispatch_size     "$rest" ;;
        mem)      dispatch_mem      "$rest" ;;
        circuit)  dispatch_circuit  "$rest" ;;
        grind)    dispatch_grind    "$rest" ;;
        field)    dispatch_field    "$rest" ;;
        *) echo "Unknown category: $category" >&2; exit 1 ;;
    esac
done

echo "Done."
