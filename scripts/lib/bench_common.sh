#!/usr/bin/env bash
# bench_common.sh -- shared plumbing for the reproduce_*.sh benchmark drivers.
#
# Source this from a driver:
#
#     SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
#     . "$SCRIPT_DIR/scripts/lib/bench_common.sh"
#
# What it provides, and why each piece exists:
#
#   bc_detect_backend   Which prover stack this machine runs. The MamaBear
#                       (AVX-512IFMA) stack is x86_64-only. Drivers use this
#                       instead of asking the operator.
#
#   bc_init_header      Builds the run header ONCE, including the SNARK
#                       configuration block, which is captured from the
#                       `bench_config` binary rather than written here. A header
#                       maintained by hand in a shell script drifts from the code
#                       it describes; this repo has already shipped one that
#                       reported a sample count its own run did not use.
#
#   bc_open_file        Truncate-once-per-run plus header-once-per-run. Every
#                       bench opens BENCH_OUTPUT_FILE in APPEND mode, so without
#                       this a second run silently interleaves fresh rows with
#                       stale ones and the file still looks plausible.
#
#   bc_begin_block      Writes the blank-line separator between row groups. The
#                       Rust harness never emits blank lines, so group structure
#                       is the driver's responsibility.
#
#   bc_cell             Runs one measurement cell, tolerating a cell that dies.
#                       Large-nv cells get OOM-killed; the convention is that an
#                       absent row means "did not complete", so a dead cell must
#                       leave no row but must still be announced on stdout and
#                       must not abort the remaining sweep.
#
# Every function is prefixed `bc_` and every variable `BC_` so a driver can use
# its own names without collision.

# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------

# bc_detect_backend
#   Echoes "mamabear" on x86_64. Honours an explicit BACKEND override.
bc_detect_backend() {
    if [[ -n "${BACKEND:-}" && "${BACKEND}" != "auto" ]]; then
        echo "$BACKEND"
        return
    fi
    case "$(uname -m)" in
        x86_64)          echo "mamabear" ;;
        *)
            echo "bench_common: unsupported machine $(uname -m); this artifact targets x86_64 only" >&2
            return 1
            ;;
    esac
}

# bc_backend_outdir BACKEND
#   The results subtree a backend owns.
bc_backend_outdir() {
    case "$1" in
        mamabear) echo "MamaBear" ;;
        *) echo "bench_common: unknown backend '$1'" >&2; return 1 ;;
    esac
}

# ---------------------------------------------------------------------------
# Regime pinning
# ---------------------------------------------------------------------------

# bc_pin_regime
#   Drop any inherited FRI-regime override from the environment.
#
#   These drivers report exactly one configuration and their headers say so, so
#   an operator's leftover BENCH_SECURITY=conj96 must not be able to make the
#   cells measure something weaker under a header claiming PROV_QUERY128.
#   Unsetting beats exporting a value: the benches default to PROV_QUERY128 and
#   PANIC on an unrecognised string, so "absent" is both correct and self-checking.
#
#   MUST be called from the driver's own shell, NOT from inside a command
#   substitution. `BC_RUN_HEADER="$(bc_init_header ...)"` runs in a subshell, so
#   an unset performed there dies with the subshell and every child cell still
#   inherits the override -- which is exactly the bug this guards against.
bc_pin_regime() {
    if [[ -n "${BENCH_SECURITY:-}" || -n "${PERF_CONJ96:-}" ]]; then
        echo "note: ignoring inherited BENCH_SECURITY/PERF_CONJ96;" \
             "this driver reports PROV_QUERY128 only" >&2
    fi
    unset BENCH_SECURITY PERF_CONJ96
}

# The parallel worker count every driver in this repo measures at.
#
# 8 is not `nproc`. The benchmark machines expose twice their physical core
# count as logical CPUs (SMT/SMT-equivalent), and rayon's own default is one
# worker per LOGICAL CPU. Letting that default through makes the recorded
# thread count a property of the host rather than of the experiment: the same
# driver then measures 8-way on one box and 16-way on another, and two files in
# the same results/ tree can disagree without either being wrong. Pinning one
# number makes every `par` row in this repo directly comparable, and makes the
# figure the papers quote a stated experimental parameter instead of whatever
# the host happened to offer.
#
# 8 rather than 16 because it is one worker per PHYSICAL core, which is the
# honest reading of "8 threads", and because it is the count the external
# baselines are measured at -- a parallel comparison is only meaningful when
# both sides get the same worker budget.
BC_DEFAULT_THREADS=8

# bc_pin_threads
#   Export RAYON_NUM_THREADS, defaulting to BC_DEFAULT_THREADS.
#
#   An explicit RAYON_NUM_THREADS in the environment is honoured, so sweeping
#   thread counts stays a one-variable change; it is announced, because the rows
#   it produces will not match the committed ones. Exporting unconditionally
#   (rather than only when unset) is the point: every child cell then inherits
#   the same number, and the "# rayon:" header line reports what actually ran
#   instead of falling back to nproc.
#
#   Call from the driver's own shell, before any cell runs -- see bc_pin_regime.
bc_pin_threads() {
    if [[ -n "${RAYON_NUM_THREADS:-}" && "${RAYON_NUM_THREADS}" != "$BC_DEFAULT_THREADS" ]]; then
        echo "note: RAYON_NUM_THREADS=$RAYON_NUM_THREADS overrides this driver's" \
             "default of $BC_DEFAULT_THREADS; rows will not match results/" >&2
    fi
    export RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-$BC_DEFAULT_THREADS}"
}

# ---------------------------------------------------------------------------
# Estimator for parallel cells
# ---------------------------------------------------------------------------
#
# Serial cells report the median of BENCH_SAMPLES timed runs inside ONE process.
# That is the right estimator when the noise lives within a process, which is
# where serial noise lives: repeated serial measurements on the benchmark
# machine agree to about 1.5%.
#
# Parallel cells do not behave that way. Measured on the reference machine
# (GCP c4d-highmem-16, idle, steal=0), the Plonky3 parallel prover at nv=20
# returns times clustered around 1.51 s, 1.75 s and 2.80 s across repeated
# PROCESSES, a 1.9x spread, while the samples inside any one process agree
# closely. A within-process median therefore reports whichever cluster that
# process happened to land in, and consecutive runs of the same driver
# disagreed by up to 1.9x. CPU pinning (one worker per physical core) does not
# remove it, and it is not host contention. Our own parallel prover is stable
# (about 8%), so a single-process median silently favours whichever side draws
# the fast cluster.
#
# So parallel cells take the MINIMUM over BC_PAR_REPS separate processes. The
# noise here is one-sided -- contention and scheduling can only make a run
# slower than the hardware allows -- which is exactly when the minimum is the
# standard estimator. It is applied to BOTH sides of every parallel-vs-parallel
# comparison, so it cannot favour either.
#
# Each repetition is one warmup plus one timed run, so the repetition budget
# moves from within-process samples to across-process repetitions instead of
# adding to it.
BC_PAR_REPS="${BENCH_PAR_REPS:-8}"

# bc_min_rows OUTFILE TMPFILE
#   Reduce a file holding several repetitions of the same rows to one row per
#   label, keeping the smallest value, and append the result to OUTFILE.
#   Row format is the suite's usual "<label...> <value> <unit>". Label order is
#   the order of first appearance, so the output keeps the sweep's row order.
bc_min_rows() {
    local out="$1" tmp="$2"
    awk '
        /^[[:space:]]*$/ || /^#/ { next }
        {
            unit = $NF; val = $(NF-1)
            label = $0
            sub(/[[:space:]]+[^[:space:]]+[[:space:]]+[^[:space:]]+[[:space:]]*$/, "", label)
            if (!(label in best) || val + 0 < best[label] + 0) {
                best[label] = val; u[label] = unit
            }
            if (!(label in seen)) { seen[label] = 1; order[++n] = label }
        }
        END { for (i = 1; i <= n; i++) printf "%-60s %10s %s\n", order[i], best[order[i]], u[order[i]] }
    ' "$tmp" | tee -a "$out"
}

# ---------------------------------------------------------------------------
# Run header
# ---------------------------------------------------------------------------

# bc_init_header SCRIPT_NAME BACKEND ARGS_STRING
#   Populates BC_RUN_HEADER. Call once, before any output file is touched.
#
#   The SNARK-configuration block comes from `cargo run --bin bench_config`. If
#   that binary cannot be built (a bare machine mid-setup, say), the header says
#   so explicitly rather than silently omitting the configuration -- a result
#   file with no regime recorded is worse than one that admits the gap.
bc_init_header() {
    local script_name="$1" backend="$2" args="$3"


    local ts cpu mem_total sockets cores_ps phys logical os_name kernel rust_v git_rev git_state
    ts="$(date -Is)"
    cpu="$(lscpu 2>/dev/null | awk -F: '/^Model name:/ {sub(/^[ \t]+/, "", $2); print $2; exit}')"
    # aarch64 /proc/cpuinfo carries no "Model name"; fall back through the part
    # number and finally the bare architecture.
    if [[ -z "$cpu" ]]; then
        cpu="$(lscpu 2>/dev/null | awk -F: '/^Vendor ID:|^Model:/ {sub(/^[ \t]+/, "", $2); print $2; exit}')"
    fi
    [[ -z "$cpu" ]] && cpu="$(uname -m)"

    mem_total="$(free -h 2>/dev/null | awk '/^Mem:/ {print $2; exit}')"
    [[ -z "$mem_total" ]] && mem_total="unknown"
    sockets="$(lscpu 2>/dev/null | awk -F: '/^Socket\(s\):/ {gsub(/[ \t]/,"",$2); print $2; exit}')"
    cores_ps="$(lscpu 2>/dev/null | awk -F: '/^Core\(s\) per socket:/ {gsub(/[ \t]/,"",$2); print $2; exit}')"
    if [[ -n "$sockets" && -n "$cores_ps" ]]; then
        phys=$((sockets * cores_ps))
    else
        phys="unknown"
    fi
    logical="$(nproc 2>/dev/null || echo unknown)"
    if [[ -r /etc/os-release ]]; then
        os_name="$(. /etc/os-release 2>/dev/null; echo "${PRETTY_NAME:-unknown}")"
    else
        os_name="unknown"
    fi
    kernel="$(uname -srv 2>/dev/null || echo unknown)"
    rust_v="$(rustc --version 2>/dev/null || echo unknown)"

    # Provenance of the code that ran.
    #
    # Three cases, and they must not be confused. In a git checkout we record the
    # commit and say plainly whether the tree was modified, because a dirty tree
    # behind a recorded measurement is a provenance defect. On the benchmark
    # machines the source arrives as a `git archive` tarball with NO .git at all,
    # so `git rev-parse` fails; BENCH_GIT_REV carries the shipped commit and the
    # state is "source archive", not "DIRTY". Reporting a clean archive as DIRTY
    # would put a false defect marker on every cloud-measured file.
    if git rev-parse --git-dir >/dev/null 2>&1; then
        git_rev="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
        if git diff --quiet HEAD -- 2>/dev/null; then
            git_state="clean"
        else
            git_state="DIRTY"
        fi
    elif [[ -n "${BENCH_GIT_REV:-}" ]]; then
        git_rev="$BENCH_GIT_REV"
        git_state="source archive"
    else
        git_rev="unknown"
        git_state="no git metadata"
    fi

    # backend "none" suppresses the SNARK-configuration block entirely. It is for
    # drivers that measure a DIFFERENT proof system (the Plonky3 baselines are a
    # BabyBear uni-STARK with their own FRI configuration): stamping this repo's
    # DeepFold parameters onto those files would describe a prover that did not
    # run, which is worse than describing nothing.
    local snark_cfg=""
    if [[ "$backend" != "none" ]]; then
        if ! snark_cfg="$(cargo run --release -q -p hyperplonk --bin bench_config -- \
                            --split "${BENCH_SPLIT_HEADER:-3}" 2>/dev/null)"; then
            snark_cfg="# (SNARK configuration unavailable: bench_config failed to build)"
        fi
        [[ -z "$snark_cfg" ]] && snark_cfg="# (SNARK configuration unavailable: bench_config produced no output)"
    fi

    {
        printf '# ==== %s run ====\n' "$script_name"
        printf '# timestamp:  %s\n' "$ts"
        printf '# cpu:        %s\n' "$cpu"
        [[ -n "${BENCH_MACHINE_CLASS:-}" ]] && printf '# machine:    %s\n' "$BENCH_MACHINE_CLASS"
        # Optional free-text note on how core frequency was established, for
        # hosts where it cannot simply be read from a counter.
        [[ -n "${BENCH_FREQ_NOTE:-}" ]] && printf '# freq:       %s\n' "$BENCH_FREQ_NOTE"
        printf '# memory:     %s\n' "$mem_total"
        printf '# cores:      %s physical, %s logical\n' "$phys" "$logical"
        printf '# os:         %s\n' "$os_name"
        printf '# kernel:     %s\n' "$kernel"
        printf '# rustc:      %s\n' "$rust_v"
        printf '# git:        %s (%s)\n' "$git_rev" "$git_state"
        printf '# args:       %s\n' "$args"
        if [[ -n "$snark_cfg" ]]; then
            printf '# ---- SNARK configuration ----\n'
            printf '%s\n' "$snark_cfg"
        fi
        printf '# ---- harness ----\n'
        printf '# warmup:     %s\n' "${BENCH_WARMUP:-1}"
        printf '# samples:    %s (median of the sorted samples)\n' "${BENCH_SAMPLES:-5}"
        printf '# rayon:      RAYON_NUM_THREADS=%s\n' "${RAYON_NUM_THREADS:-$(nproc 2>/dev/null || echo unknown)}"
        printf '# ============================\n'
    }
}

# ---------------------------------------------------------------------------
# Output-file lifecycle
# ---------------------------------------------------------------------------

# Files this run has already opened, so each is truncated and headered exactly
# once no matter how many categories write to it.
declare -A BC_OPENED=()
# Files that already hold at least one row group this run, so bc_begin_block
# knows whether a separator is wanted.
declare -A BC_HAS_BLOCK=()

# bc_open_file FILE
#   Truncate (unless BENCH_APPEND=1) and write the run header. Idempotent within
#   a run.
#
#   Truncation is the default because a driver's contract is "one full run
#   reproduces this file", and the benches themselves only ever append. Set
#   BENCH_APPEND=1 to build a file up across several partial invocations; that
#   is useful during development and dangerous for a recorded result, which is
#   why it is opt-in.
#
#   Before the first truncation, the shipped file is copied aside to
#   "<name>.committed" (once per file, so a second run cannot overwrite the
#   saved original with its own output). Truncating is what makes a run
#   reproduce the file exactly, but it also destroys the measurement the
#   reviewer is trying to compare against -- and that comparison, shipped vs.
#   regenerated, is the point of the run. "*.committed" is gitignored.
bc_open_file() {
    local f="$1"
    [[ -n "${BC_OPENED[$f]+set}" ]] && return 0
    mkdir -p "$(dirname "$f")"
    if [[ -f "$f" && ! -f "$f.committed" ]]; then
        cp -p "$f" "$f.committed"
        echo "  [saved] $f -> $f.committed (shipped copy, for diffing)"
    fi
    if [[ "${BENCH_APPEND:-0}" == "1" ]]; then
        # Separate this run's rows from whatever is already there.
        [[ -s "$f" ]] && printf '\n' >> "$f"
    else
        : > "$f"
    fi
    printf '%s\n' "$BC_RUN_HEADER" >> "$f"
    BC_OPENED[$f]=1
    return 0
}

# bc_begin_block FILE
#   Start a new row group. Writes the blank-line separator only when the file
#   already holds a group, so a file never begins with a stray blank line.
bc_begin_block() {
    local f="$1"
    bc_open_file "$f"
    if [[ -n "${BC_HAS_BLOCK[$f]+set}" ]]; then
        printf '\n' >> "$f"
    fi
    BC_HAS_BLOCK[$f]=1
}

# ---------------------------------------------------------------------------
# Cell execution
# ---------------------------------------------------------------------------

BC_CELLS_RUN=0
BC_CELLS_FAILED=0
BC_FAILED_CELLS=()

# bc_cell DESCRIPTION -- COMMAND...
#   Run one measurement cell. On failure, record and continue.
#
#   The high-nv cells of the larger sweeps get OOM-killed by the kernel. The
#   committed files record that by simply not having the row, so this must NOT
#   write anything into the results file on failure -- but it must say so on
#   stdout, because a sweep that lost half its cells should not look like a
#   clean run. The end-of-run summary reprints every failure.
bc_cell() {
    local desc="$1"; shift
    [[ "${1:-}" == "--" ]] && shift
    BC_CELLS_RUN=$((BC_CELLS_RUN + 1))
    echo "  [cell] $desc"
    local rc=0
    "$@" || rc=$?
    if (( rc != 0 )); then
        BC_CELLS_FAILED=$((BC_CELLS_FAILED + 1))
        BC_FAILED_CELLS+=("$desc (exit $rc)")
        # Attribute an out-of-memory kill, which on these sweeps is essentially
        # always the cause rather than a logic failure.
        #
        # It arrives two different ways depending on how the cell was launched,
        # and BOTH have to be recognised:
        #
        #   137  the child was SIGKILLed and the status reached us directly.
        #        This is what `cargo run --bin ...` produces.
        #   101  cargo's OWN failure code. `cargo bench` does not propagate the
        #        child's signal; it catches the SIGKILL, prints "(signal: 9,
        #        SIGKILL: kill)" and exits 101. Treating 101 as a generic
        #        failure lost the OOM attribution on every bench-launched cell
        #        -- which is most of them.
        #
        # 101 is also cargo's exit code for a genuine build or runtime error, so
        # the message says "likely" rather than asserting it. The distinction
        # does not change what happens to the row (absent either way); it
        # changes whether the operator goes looking for a bug that is not there.
        if (( rc == 137 )); then
            echo "  [SKIP] $desc -- killed (exit 137, SIGKILL: OOM); leaving the row absent" >&2
        elif (( rc == 101 )); then
            echo "  [SKIP] $desc -- exit 101 (cargo failure; at high nv this is" \
                 "almost always an OOM-killed child, check for 'signal: 9' above);" \
                 "leaving the row absent" >&2
        else
            echo "  [SKIP] $desc -- exit $rc; leaving the row absent" >&2
        fi
    fi
    # Give the allocator a moment to return pages between cells; the large-nv
    # cells otherwise start against a still-shrinking RSS.
    sleep 0.3
    return 0
}

# bc_summary SCRIPT_NAME START_EPOCH
#   End-of-run summary: elapsed time plus every cell that did not complete.
#   Wire it to an EXIT trap so it prints on Ctrl-C and on error too:
#
#       trap 'bc_summary "$SCRIPT_NAME" "$START_EPOCH"' EXIT
#
#   Use PLAIN variable expansions in that trap string. A command substitution
#   there -- `trap 'bc_summary "$0" "$(date +%s)"' EXIT` -- runs while the
#   arguments are being expanded and overwrites $? before the function is
#   entered, so the run's real exit status is lost and every failure reports
#   as success.
bc_summary() {
    # MUST be the first statement: `local x="$1"` is itself a successful command
    # and would overwrite $? with 0, so capturing the exit status any later
    # reports every aborted run as a clean one. That is exactly the lie this
    # summary exists to prevent -- a six-hour sweep that died at hour five would
    # print "cells: N run, 0 failed" and nothing else.
    local exit_code=$?
    local script_name="$1" start_epoch="$2"
    local end_epoch elapsed h m s
    end_epoch=$(date +%s)
    elapsed=$((end_epoch - start_epoch))
    h=$((elapsed / 3600)); m=$(( (elapsed % 3600) / 60 )); s=$((elapsed % 60))
    printf '\n# ==== %s summary ====\n' "$script_name"
    printf '# elapsed:  %dh%02dm%02ds (%ds)\n' "$h" "$m" "$s" "$elapsed"
    printf '# cells:    %d run, %d failed\n' "$BC_CELLS_RUN" "$BC_CELLS_FAILED"
    if (( BC_CELLS_FAILED > 0 )); then
        printf '# The following cells did not complete; their rows are ABSENT from the\n'
        printf '# results files (this is the convention, but check they were expected):\n'
        local c
        for c in "${BC_FAILED_CELLS[@]}"; do printf '#   - %s\n' "$c"; done
    fi
    (( exit_code != 0 )) && printf '# exit:     non-zero (%d)\n' "$exit_code"
    printf '# ==============================\n'
}
