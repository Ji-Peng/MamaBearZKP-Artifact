#!/usr/bin/env bash
set -euo pipefail

# reproduce_p3_rand.sh - run Plonky3 with a HyperPlonk-style degree-3 gate AIR
# as a fair comparison baseline for HyperPlonk + DeepFold.
#
# Rationale: benchmarking Plonky3 with its Blake3 AIR is not apples-to-apples,
# because one row of the Blake3 trace equals one
# full Blake3 permutation, which in our HyperPlonk gate system is ~149k
# add/mul gates. Plonky3 also allows degree-7 constraints and arbitrary
# inter-row structure, whereas HyperPlonk here runs a single degree-3 gate
# h(X) = (1-S)(L+R) + S*L*R - O with 3 witness columns and 1 preprocessed
# selector, independently per row of a size-2^nv domain.
#
# This script pins Plonky3 to the same commit, stacks the base benchmark
# patch (conj96/prov97/prov128 FRI presets, bench_prove median harness,
# PLONKY3_* markers) on top of a new patch that adds a HyperPlonkGate AIR
# and wires it into --objective hyper-plonk-gate. Same num_queries,
# log_blowup, PoW, threading and warmup/samples protocol as the Blake3
# sweep - so the only remaining difference is PIOP+PCS.
#
# Writes:
#   results/Plonky3/rand/time.txt        prove + verify time (ms, median)
#   results/Plonky3/rand/proof_size.txt  postcard proof size (KB)
#   results/Plonky3/rand/peak_mem.txt    peak heap (prove GiB, verify MiB)
#
# Usage:
#   ./reproduce_p3_rand.sh [time|proof_size|peak_memory|all]
#
# Env vars:
#   BENCH_NV_MIN            minimum NV (default 18)
#   BENCH_NV_MAX            maximum NV (default 24)
#   BENCH_SAMPLES           timed samples per cell, median reported (default 5)
#   BENCH_WARMUP            warmup runs per cell (default 1)
#   RAYON_NUM_THREADS       `par` worker count for both legs (default 8, one per
#                           physical core; overriding it means the rows will not
#                           match results/)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
# shellcheck source=scripts/lib/bench_common.sh
. "$SCRIPT_DIR/scripts/lib/bench_common.sh"

PLONKY3_COMMIT="0f87f2b543a01880274965c410bf804c124f5046"
PLONKY3_URL="https://github.com/Plonky3/Plonky3.git"
PLONKY3_DIR="$SCRIPT_DIR/external/plonky3_rand_circuit"
PATCH_FILES=(
    "$SCRIPT_DIR/patches/plonky3-benchmark.patch"
    "$SCRIPT_DIR/patches/plonky3-rand-circuit.patch"
)
OUTDIR="${BENCH_RESULTS_DIR:-$SCRIPT_DIR/results}/Plonky3/rand"
mkdir -p "$OUTDIR"

NV_MIN="${BENCH_NV_MIN:-18}"
NV_MAX="${BENCH_NV_MAX:-24}"
# Warmup / samples are read by the patched Plonky3 example itself so that
# a single process warms its caches and allocator across iterations.
SAMPLES="${BENCH_SAMPLES:-5}"
WARMUP="${BENCH_WARMUP:-1}"
# ---- Thread counts: one default per leg, each matching its committed rows ----
#
# Both legs measure at one worker per PHYSICAL core of the benchmark machine
# (8 cores / 16 logical CPUs), which is what every driver in this repo pins --
# see bc_pin_threads in scripts/lib/bench_common.sh. rayon's own default is one
# worker per LOGICAL CPU, which would make the recorded thread count a property
# of the host rather than of the experiment, and would give the two columns of
# the matched-gate table different worker budgets.
#
# The count matters more here than on our side: at mu = 20 Plonky3 moves 1.40x
# between 8 and 16 workers, where MamaBear moves 1.05x, so running this leg at
# the wrong count produces rows that read as a failed reproduction.
#
# RAYON_NUM_THREADS overrides it, which is the right knob for sweeping thread
# counts -- but then the rows will not match results/.
THREADS="${RAYON_NUM_THREADS:-8}"
export BENCH_SAMPLES BENCH_WARMUP
: "${BENCH_SAMPLES:=$SAMPLES}"
: "${BENCH_WARMUP:=$WARMUP}"

# One regime. The weaker presets understate every cost and are not what this
# suite reports; conj96/prov97 rows already on disk predate this and are left
# alone rather than mixed with fresh prov128 rows.
SECURITIES=(prov128)
MODES=(serial par)

# --------------------------------------------------------------------------
# Setup: clone + checkout + apply the patch stack idempotently.
#
# The two patches overlap in proofs.rs (the rand-circuit patch rewrites
# lines the base patch introduced), so `git apply --reverse --check` can't
# reliably identify which patches are already applied once both are on.
# Instead we use file-level sentinels:
#   - `SecurityOptions` enum in examples/src/parsers.rs  -> base patch applied
#   - `HyperPlonkGate` variant in examples/src/parsers.rs -> rand-circuit applied
# --------------------------------------------------------------------------
setup_plonky3() {
    mkdir -p "$SCRIPT_DIR/external"
    if [ ! -d "$PLONKY3_DIR/.git" ]; then
        echo "Cloning Plonky3 -> $PLONKY3_DIR ..."
        git clone "$PLONKY3_URL" "$PLONKY3_DIR"
    fi
    (
        cd "$PLONKY3_DIR"
        if [ "$(git rev-parse HEAD)" != "$PLONKY3_COMMIT" ]; then
            echo "Checking out pinned commit $PLONKY3_COMMIT ..."
            git fetch origin "$PLONKY3_COMMIT" --depth 1 2>/dev/null || git fetch origin --depth 100
            git checkout "$PLONKY3_COMMIT"
        fi

        local base_applied=0 new_applied=0
        grep -q 'pub enum SecurityOptions' examples/src/parsers.rs 2>/dev/null && base_applied=1
        grep -q 'HyperPlonkGate' examples/src/parsers.rs 2>/dev/null && new_applied=1

        if [ "$base_applied" = 1 ] && [ "$new_applied" = 1 ]; then
            echo "Both patches already applied; skipping."
        elif [ "$base_applied" = 1 ] && [ "$new_applied" = 0 ]; then
            echo "Base patch present; applying rand-circuit patch ..."
            git apply "${PATCH_FILES[1]}"
        elif [ "$base_applied" = 0 ] && [ "$new_applied" = 0 ]; then
            echo "Clean tree; applying base patch + rand-circuit patch ..."
            git apply "${PATCH_FILES[0]}"
            git apply "${PATCH_FILES[1]}"
        else
            echo "ERROR: unexpected patch state (new applied without base)." >&2
            echo "       To reset, run:" >&2
            echo "         (cd $PLONKY3_DIR && git reset --hard $PLONKY3_COMMIT)" >&2
            echo "       then re-run this script." >&2
            exit 1
        fi
    )
}

# --------------------------------------------------------------------------
# Build helpers. Each unique feature combination is built once up front so
# per-sample `cargo run` calls are no-ops on rebuild.
# --------------------------------------------------------------------------
features_for() {
    # $1 = mode (serial|par), $2 = include_peak (0|1). Prints "" or
    # "--features X,Y" safe for word-splitting in an unquoted expansion.
    local feats=""
    if [ "$1" = par ]; then feats+=",parallel"; fi
    if [ "$2" = 1 ];   then feats+=",measure-peak"; fi
    if [ -n "$feats" ]; then
        echo "--features ${feats#,}"
    fi
}

build_example() {
    local mode="$1" include_peak="$2"
    # shellcheck disable=SC2046
    (cd "$PLONKY3_DIR" && RUSTFLAGS="-Ctarget-cpu=native" \
        cargo build --release -q -p p3-examples --example prove_prime_field_31 \
        $(features_for "$mode" "$include_peak"))
}

# Run the patched example once; stdout captured by caller.
run_once() {
    local mode="$1" security="$2" nv="$3" include_peak="$4"
    local env_prefix=()
    [ "$mode" = par ] && env_prefix+=("RAYON_NUM_THREADS=$THREADS")
    # shellcheck disable=SC2046
    (cd "$PLONKY3_DIR" && env "${env_prefix[@]}" RUSTFLAGS="-Ctarget-cpu=native" \
        cargo run --release -q -p p3-examples --example prove_prime_field_31 \
        $(features_for "$mode" "$include_peak") -- \
            --field baby-bear \
            --objective hyper-plonk-gate \
            --log-trace-length "$nv" \
            --discrete-fourier-transform radix-2-dit-parallel \
            --merkle-hash keccak-f \
            --security "$security") 2>&1
}

# --------------------------------------------------------------------------
# Output formatting matches hyperplonk/benches/*.rs
# ("{label:<60} {value:>10.3} {unit}") and the other drivers in this repo.
# --------------------------------------------------------------------------
append_line() {
    local out="$1" label="$2" value="$3" unit="$4"
    printf "%-60s %16s %s\n" "$label" "$value" "$unit" | tee -a "$out"
}

# --------------------------------------------------------------------------
# Timing + proof size. The patched Plonky3 example runs BENCH_WARMUP +
# BENCH_SAMPLES iterations inside a single process and emits the median
# prove / verify times as PLONKY3_PROVE_MS / PLONKY3_VERIFY_MS. Proof size
# is recorded once per (security, nv) from the serial run, in KB.
# --------------------------------------------------------------------------
# rand_leg_header OUT
#   Write the provenance header for a rand-leg output file.
#
#   backend="none" suppresses this repo's SNARK-configuration block: these rows
#   come from a BabyBear uni-STARK with its own FRI configuration, and stamping
#   DeepFold's parameters on them would describe a prover that did not run. The
#   Plonky3-specific block below carries the configuration that DID run.
rand_leg_header() {
    local out="$1"
    BC_RUN_HEADER="$(RAYON_NUM_THREADS="$THREADS" \
        bc_init_header "reproduce_p3_rand.sh" "none" "${BC_RUN_ARGS:-all}")"
    bc_open_file "$out"
    {
        printf '# Plonky3 degree-3 random-gate AIR baseline (--objective hyper-plonk-gate)\n'
        printf '# plonky3:  %s + %s\n' "${PLONKY3_COMMIT:0:12}" \
            "$(basename "${PATCH_FILES[0]}") + $(basename "${PATCH_FILES[1]}")"
        printf '# config:   --field baby-bear --security prov128 (default degree-4 BabyBear extension)\n'
        printf '# nv:       %s..%s   threads(par)=%s\n' "$NV_MIN" "$NV_MAX" "$THREADS"
        printf '# note:     extension degree 4 gives a ~124-bit challenge field.\n'
        # The estimator is not derivable from the warmup/sample counts above:
        # parallel rows are a minimum ACROSS processes, not a median within one.
        # Spell it out, so a row can be interpreted without reading this script.
        cat <<'EOF'
# estimator: serial = median of BENCH_SAMPLES samples in one process;
#            par    = MINIMUM over BENCH_PAR_REPS processes (1 warmup + 1
#            sample each). Parallel timings on this host vary between
#            processes rather than within one, so a within-process median
#            reports whichever cluster that process landed in. The MamaBear
#            column this is compared against is estimated the same way.
EOF
        printf '#\n'
    } >> "$out"
}

bytes_to_kb() { awk -v b="$1" 'BEGIN { printf "%.2f", b / 1024.0 }'; }

dispatch_time() {
    local out="$OUTDIR/time.txt"
    local psize_out="$OUTDIR/proof_size.txt"
    # Truncate once per run (via rand_leg_header). Every writer below APPENDS,
    # so without this a repeated run silently stacks fresh rows on top of stale
    # ones and the file still looks plausible.
    rand_leg_header "$out"
    rand_leg_header "$psize_out"
    echo "=== Building (serial) ==="
    build_example serial 0
    echo "=== Building (parallel) ==="
    build_example par 0

    for sec in "${SECURITIES[@]}"; do
        for mode in "${MODES[@]}"; do
            local prove_op verify_op
            if [ "$mode" = serial ]; then
                prove_op=prove;     verify_op=verify
            else
                prove_op=prove_par; verify_op=verify_par
            fi
            for nv in $(seq "$NV_MIN" "$NV_MAX"); do
                echo "--- time $sec $mode NV=$nv (warmup=$BENCH_WARMUP samples=$BENCH_SAMPLES) ---"
                # Serial: one process. Parallel: the minimum over BENCH_PAR_REPS
                # processes, because the parallel measurement's noise lives
                # BETWEEN processes on this host -- repeated runs of this very
                # cell at nv=20 clustered at 1.51 / 1.75 / 2.80 s. The MamaBear
                # column this is compared against is estimated the same way.
                local reps=1
                [ "$mode" = par ] && reps="${BENCH_PAR_REPS:-8}"
                # One timed sample per repetition when repeating, so the budget
                # moves from within-process samples to across-process
                # repetitions rather than multiplying with them.
                local rep_samples="$SAMPLES"
                [ "$reps" -gt 1 ] && rep_samples=1
                local rep tmp prove_ms verify_ms proof_bytes cur
                prove_ms=""; verify_ms=""; proof_bytes=""
                for rep in $(seq 1 "$reps"); do
                    tmp=$(mktemp)
                    BENCH_SAMPLES="$rep_samples" run_once "$mode" "$sec" "$nv" 0 > "$tmp"
                    cur=$(awk '/^PLONKY3_PROVE_MS:/ { print $2 }' "$tmp")
                    if [ -n "$cur" ] && { [ -z "$prove_ms" ] || awk "BEGIN{exit !($cur < $prove_ms)}"; }; then
                        prove_ms="$cur"
                    fi
                    cur=$(awk '/^PLONKY3_VERIFY_MS:/ { print $2 }' "$tmp")
                    if [ -n "$cur" ] && { [ -z "$verify_ms" ] || awk "BEGIN{exit !($cur < $verify_ms)}"; }; then
                        verify_ms="$cur"
                    fi
                    [ -z "$proof_bytes" ] && proof_bytes=$(awk '/^PLONKY3_PROOF_BYTES:/ { print $2 }' "$tmp")
                    rm -f "$tmp"
                done
                append_line "$out" "Plonky3 BabyBear RAND $sec $prove_op NV=$nv" \
                    "$(printf "%.3f" "$prove_ms")" "ms"
                append_line "$out" "Plonky3 BabyBear RAND $sec $verify_op NV=$nv" \
                    "$(printf "%.3f" "$verify_ms")" "ms"
                if [ "$mode" = serial ]; then
                    append_line "$psize_out" "Plonky3 BabyBear RAND $sec proof_size NV=$nv" \
                        "$(bytes_to_kb "$proof_bytes")" "KB"
                fi
            done
        done
    done
}

# --------------------------------------------------------------------------
# Proof size only: one run per (security, nv), no parallel variant.
# Output in KB for readability.
# --------------------------------------------------------------------------
dispatch_proof_size() {
    local out="$OUTDIR/proof_size.txt"
    # Truncate once per run (via rand_leg_header). Every writer below APPENDS,
    # so without this a repeated run silently stacks fresh rows on top of stale
    # ones and the file still looks plausible.
    rand_leg_header "$out"
    echo "=== Building (serial) ==="
    build_example serial 0
    for sec in "${SECURITIES[@]}"; do
        for nv in $(seq "$NV_MIN" "$NV_MAX"); do
            echo "--- proof_size $sec NV=$nv ---"
            local tmp; tmp=$(mktemp)
            BENCH_WARMUP=0 BENCH_SAMPLES=1 run_once serial "$sec" "$nv" 0 > "$tmp"
            local pb
            pb=$(awk '/^PLONKY3_PROOF_BYTES:/ { print $2 }' "$tmp")
            rm -f "$tmp"
            append_line "$out" "Plonky3 BabyBear RAND $sec proof_size NV=$nv" \
                "$(bytes_to_kb "$pb")" "KB"
        done
    done
}

# --------------------------------------------------------------------------
# Peak memory: single measured run per cell with peak_alloc global allocator.
# Emits TWO lines per cell: prove peak (GiB) and verify peak (MiB), matching the
# unit convention of the MamaBear-side peak_memory binary. Both divide by powers
# of 1024, so these are binary units; they were previously labelled GB/MB, which
# named a decimal quantity the code never computed. Serial and par builds are
# compiled separately (different feature set).
# --------------------------------------------------------------------------
bytes_to_gib() { awk -v b="$1" 'BEGIN { printf "%.3f", b / (1024.0 * 1024.0 * 1024.0) }'; }
bytes_to_mib() { awk -v b="$1" 'BEGIN { printf "%.3f", b / (1024.0 * 1024.0) }'; }

dispatch_peak() {
    local out="$OUTDIR/peak_mem.txt"
    # Truncate once per run (via rand_leg_header). Every writer below APPENDS,
    # so without this a repeated run silently stacks fresh rows on top of stale
    # ones and the file still looks plausible.
    rand_leg_header "$out"
    echo "=== Building (serial + measure-peak) ==="
    build_example serial 1
    echo "=== Building (parallel + measure-peak) ==="
    build_example par 1
    for sec in "${SECURITIES[@]}"; do
        for mode in "${MODES[@]}"; do
            local prove_op verify_op
            if [ "$mode" = serial ]; then
                prove_op=prove;     verify_op=verify
            else
                prove_op=prove_par; verify_op=verify_par
            fi
            for nv in $(seq "$NV_MIN" "$NV_MAX"); do
                echo "--- peak $sec $mode NV=$nv ---"
                local tmp; tmp=$(mktemp)
                # 1 warmup + 1 sample: warmup primes allocator arenas; the
                # patched code resets peak_usage before the measured prove
                # and again before the measured verify so each number
                # reflects only one complete phase.
                BENCH_WARMUP=1 BENCH_SAMPLES=1 run_once "$mode" "$sec" "$nv" 1 > "$tmp"
                local prove_peak verify_peak
                prove_peak=$(awk '/^PLONKY3_PROVE_PEAK_BYTES:/ { print $2 }' "$tmp")
                verify_peak=$(awk '/^PLONKY3_VERIFY_PEAK_BYTES:/ { print $2 }' "$tmp")
                rm -f "$tmp"
                append_line "$out" "Plonky3 BabyBear RAND $sec $prove_op NV=$nv" \
                    "$(bytes_to_gib "$prove_peak")" "GiB"
                append_line "$out" "Plonky3 BabyBear RAND $sec $verify_op NV=$nv" \
                    "$(bytes_to_mib "$verify_peak")" "MiB"
            done
        done
    done
}

main() {
    local cmd="${1:-all}"
    # Recorded verbatim on the "args:" header line of every file this run writes.
    BC_RUN_ARGS="$cmd"
    case "$cmd" in
        time|proof_size|peak_memory|all) setup_plonky3 ;;
        *)
            echo "Usage: $0 [time|proof_size|peak_memory|all]" >&2
            exit 2
            ;;
    esac
    echo ""
    case "$cmd" in
        time)        dispatch_time ;;
        proof_size)  dispatch_proof_size ;;
        peak_memory) dispatch_peak ;;
        all)         dispatch_time; dispatch_peak ;;
    esac
    echo ""
    echo "Done. Outputs in $OUTDIR:"
    ls -la "$OUTDIR"/*.txt 2>/dev/null || true
}

main "$@"
