#!/usr/bin/env python3
"""Generate LaTeX tables from bench_results txt files.

Usage:
    python3 plot_tables.py               # write tables.tex to current directory
    python3 plot_tables.py -p <dir>      # write tables.tex to <dir>

All tables are collected into a single, self-contained ``tables.tex`` document
that can be compiled directly with ``pdflatex tables.tex``. The output matches
the paper's exact LaTeX formatting (acmart sigconf, tabularray, subcaption).
"""
import argparse
import re
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path


def round_half_up(x, ndigits):
    """Traditional round-half-up to `ndigits` decimal places (not banker's rounding)."""
    if x is None:
        return None
    q = Decimal(1).scaleb(-ndigits)
    return float(Decimal(str(x)).quantize(q, rounding=ROUND_HALF_UP))

BENCH_DIR = Path(__file__).resolve().parent

# ============================================================================
# Benchmark txt input files (all under BENCH_DIR == this script's directory).
# ============================================================================

# --- ZeroCheck (sumcheck-zerocheck) benchmarks ---
ZEROCHECK_GOLDILOCKS = BENCH_DIR / 'MamaBear/hp_zc.txt'
ZEROCHECK_BABYBEAR   = BENCH_DIR / 'MamaBear/hp_zc.txt'
ZEROCHECK_MAMABEAR   = BENCH_DIR / 'MamaBear/hp_zc.txt'

# --- ProductCheck benchmarks ---
PRODCHECK_GOLDILOCKS = BENCH_DIR / 'MamaBear/hp_pc.txt'
PRODCHECK_BABYBEAR   = BENCH_DIR / 'MamaBear/hp_pc.txt'
PRODCHECK_MAMABEAR   = BENCH_DIR / 'MamaBear/hp_pc.txt'

# --- DeepFold PCS (commit / open) benchmarks ---
DEEPFOLD_GOLDILOCKS = BENCH_DIR / 'MamaBear/df.txt'
DEEPFOLD_BABYBEAR   = BENCH_DIR / 'MamaBear/df.txt'
DEEPFOLD_MAMABEAR   = BENCH_DIR / 'MamaBear/df.txt'

# --- HyperPlonk--DeepFold end-to-end prover benchmarks ---
HP_DF_GOLDILOCKS = BENCH_DIR / 'MamaBear/hp_df.txt'
HP_DF_BABYBEAR   = BENCH_DIR / 'MamaBear/hp_df.txt'
HP_DF_MAMABEAR   = BENCH_DIR / 'MamaBear/hp_df.txt'
PLONKY3_PROVE    = BENCH_DIR / 'Plonky3/rand/time.txt'
HP_DF_CIRCUIT    = BENCH_DIR / 'MamaBear/hp_df_circuit.txt'

# --- Peak memory ---
PEAK_MEMORY         = BENCH_DIR / 'MamaBear/peak_mem.txt'
PLONKY3_PEAK_MEMORY = BENCH_DIR / 'Plonky3/rand/peak_mem.txt'

# --- Proof size ---
PROOF_SIZE         = BENCH_DIR / 'MamaBear/hp_df_proof_size.txt'
PLONKY3_PROOF_SIZE = BENCH_DIR / 'Plonky3/rand/proof_size.txt'

# --- Field arithmetic microbenchmarks ---
FIELD_BENCH = BENCH_DIR / 'MamaBear/field.txt'

# Fail loudly on a missing input.
_REQUIRED_INPUTS = [
    ZEROCHECK_MAMABEAR, PRODCHECK_MAMABEAR, DEEPFOLD_MAMABEAR, HP_DF_MAMABEAR,
    PLONKY3_PROVE, HP_DF_CIRCUIT, PEAK_MEMORY, PLONKY3_PEAK_MEMORY,
    PROOF_SIZE, PLONKY3_PROOF_SIZE, FIELD_BENCH,
]
_missing = [str(p) for p in _REQUIRED_INPUTS if not p.exists()]
if _missing:
    raise SystemExit(
        'plot_tables.py: missing results input(s):\n  '
        + '\n  '.join(_missing)
        + '\n\nRefusing to run: every table would render as "--", which the captions\n'
          'define as out-of-memory. Regenerate the results or fix the paths above.'
    )


def parse_ml(path, pattern):
    """Parse benchmark file line by line with re.match (line-anchored).
    Returns dict[nv -> time_ms]."""
    out = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        m = re.match(pattern, line)
        if m:
            out[int(m.group('nv'))] = float(m.group('t'))
    return out


def fmt_time(t):
    """Returns (display_string, display_value_in_ms_after_rounding).
    The returned ms value is used downstream for speedup ratios so the
    ratio is consistent with what the user sees in the table."""
    if t is None:
        return '--', None
    if t > 100.0:
        val_s = t / 1000.0
        if val_s > 100.0:
            val_s = round_half_up(val_s, 1)
            return f'{val_s:.1f} s', val_s * 1000.0
        val_s = round_half_up(val_s, 2)
        return f'{val_s:.2f} s', val_s * 1000.0
    val_ms = round_half_up(t, 1)
    return f'{val_ms:.1f} ms', val_ms


def fmt_speedup(mama_ms, ref_ms):
    if mama_ms is None or ref_ms is None or ref_ms == 0 or mama_ms == 0:
        return '--'
    ratio = round_half_up(ref_ms / mama_ms, 0)
    return f'{int(ratio)}x'


def build_row_single_backend(n, g, b, m, mp):
    g_s,  g_v  = fmt_time(g)
    b_s,  b_v  = fmt_time(b)
    m_s,  m_v  = fmt_time(m)
    mp_s, mp_v = fmt_time(mp)
    return ' & '.join([
        str(n), g_s, b_s,
        m_s,  f'{fmt_speedup(m_v,  g_v)}/{fmt_speedup(m_v,  b_v)}',
        mp_s, f'{fmt_speedup(mp_v, g_v)}/{fmt_speedup(mp_v, b_v)}',
    ]) + r' \\'


# ============================================================================
# LaTeX rendering helpers for component tables (7-column tblr).
# ============================================================================

# The tblr body shared by zerocheck/prodcheck/deepfold side-by-side subtables.
_COMPONENT_TBLR_SPEC = r"""\small
\begin{tblr}{
    width = \linewidth,
    colspec = {Q[c,m] X[c,m] X[c,m] X[c,m] X[c,m] X[c,m] X[c,m]},
    row{1} = {font=\bfseries}, row{2} = {font=\bfseries}, rowsep = 1.5pt,
    colsep = 3pt,
    cell{1}{1} = {r=2}{c,m}, cell{1}{2} = {r=2}{c,m}, cell{1}{3} = {r=2}{c,m},
    cell{1}{4} = {c=2}{c,m}, cell{1}{6} = {c=2}{c,m},
}
\toprule
$\mu$ & {Gold.\\$\F_{p^2}$} & {Baby.\\$\F_{p^4}$} & {MamaBear $\F_{p^3}$} & & {MamaBear-Par $\F_{p^3}$} & \\
& & & Time & {M.\ vs.\ G./B.} & Time & {M.\ vs.\ G./B.} \\
\midrule
"""


def render_component_tblr_body(rows):
    """Render the inner tblr for a 7-column component table (no caption/label)."""
    return _COMPONENT_TBLR_SPEC + '\n'.join(rows) + '\n\\bottomrule\n\\end{tblr}\n'


# ============================================================================
# Table generators.
# ============================================================================

def gen_zerocheck_prodcheck_side_by_side():
    """Generate the side-by-side ZeroCheck + ProductCheck table (table* with two subtables)."""
    gold_z = parse_ml(ZEROCHECK_GOLDILOCKS,
                      r'Zerocheck Goldilocks64Ext NV=(?P<nv>\d+)\s+(?P<t>[\d.]+) ms')
    baby_z = parse_ml(ZEROCHECK_BABYBEAR,
                      r'Zerocheck BabyBearExt4 NV=(?P<nv>\d+)\s+(?P<t>[\d.]+) ms')
    m3_z = parse_ml(ZEROCHECK_MAMABEAR, r'Optimized Ext3 ell0=2 NV=(?P<nv>\d+)\s+(?P<t>[\d.]+) ms')
    m3p_z = parse_ml(ZEROCHECK_MAMABEAR, r'Optimized Ext3 Par ell0=2 NV=(?P<nv>\d+)\s+(?P<t>[\d.]+) ms')
    zero_rows = [build_row_single_backend(n, gold_z.get(n), baby_z.get(n), m3_z.get(n), m3p_z.get(n))
                 for n in range(18, 29)]

    gold_p = parse_ml(PRODCHECK_GOLDILOCKS,
                      r'ProdEqCheck Goldilocks NV=(?P<nv>\d+)\s+(?P<t>[\d.]+) ms')
    baby_p = parse_ml(PRODCHECK_BABYBEAR,
                      r'ProdEqCheck BabyBear NV=(?P<nv>\d+)\s+(?P<t>[\d.]+) ms')
    m3_p = parse_ml(PRODCHECK_MAMABEAR, r'ProdEqCheck PerWire Ext3 NV=(?P<nv>\d+)\s+(?P<t>[\d.]+) ms')
    m3p_p = parse_ml(PRODCHECK_MAMABEAR, r'ProdEqCheck PerWirePar Ext3 NV=(?P<nv>\d+)\s+(?P<t>[\d.]+) ms')
    prod_rows = [build_row_single_backend(n, gold_p.get(n), baby_p.get(n), m3_p.get(n), m3p_p.get(n))
                 for n in range(18, 29)]

    zero_tblr = render_component_tblr_body(zero_rows)
    prod_tblr = render_component_tblr_body(prod_rows)

    latex = (
        r"""\begin{table*}[t]
\centering
\caption{ZeroCheck and ProductCheck time across circuit sizes $2^{\mu}$ for $\mu = 18$ to $28$. Goldilocks and BabyBear serve as baselines. ``M.\ vs.\ G./B.'' reports the MamaBear speedup over Goldilocks / BabyBear. ``-Par'' columns use 8 threads. ``--'' indicates data not reported due to out-of-memory (OOM).}
\label{tab:zerocheck-prodcheck}
\begin{subtable}[t]{0.49\textwidth}
\centering
\caption{ZeroCheck time over $\F_{p^3}$ with $\ell_0{=}2$ (\S\ref{sec:zerocheck-ell0}).}
\label{tab:zerocheck}
"""
        + zero_tblr
        + r"""
\end{subtable}\hfill
\begin{subtable}[t]{0.49\textwidth}
\centering
\caption{ProductCheck time using the per-wire design (\S\ref{sec:perwire}) over $\F_{p^3}$.}
\label{tab:prodcheck}
"""
        + prod_tblr
        + r"""
\end{subtable}
\end{table*}
"""
    )
    return ('tab:zerocheck-prodcheck', latex)


def build_row4(n, g, b, m, mp):
    g_s,  g_v  = fmt_time(g)
    b_s,  b_v  = fmt_time(b)
    m_s,  m_v  = fmt_time(m)
    mp_s, mp_v = fmt_time(mp)
    return ' & '.join([
        str(n), g_s, b_s,
        m_s,  f'{fmt_speedup(m_v,  g_v)}/{fmt_speedup(m_v,  b_v)}',
        mp_s, f'{fmt_speedup(mp_v, g_v)}/{fmt_speedup(mp_v, b_v)}',
    ]) + r' \\'


def gen_deepfold_commit_open_side_by_side():
    """Generate the side-by-side DeepFold Commit + Open table (table* with two subtables)."""
    commit_gold = parse_ml(DEEPFOLD_GOLDILOCKS,
                           r'Goldilocks prov128 Commit NV=(?P<nv>\d+)\s+(?P<t>[\d.]+) ms')
    commit_baby = parse_ml(DEEPFOLD_BABYBEAR,
                           r'BabyBear prov128 Commit NV=(?P<nv>\d+)\s+(?P<t>[\d.]+) ms')
    commit_mama = parse_ml(DEEPFOLD_MAMABEAR,
                           r'Ext3 prov128 Commit NV=(?P<nv>\d+)\s+split=5\s+(?P<t>[\d.]+) ms')
    commit_mama_par = parse_ml(DEEPFOLD_MAMABEAR,
                               r'Ext3 prov128 Commit Par NV=(?P<nv>\d+)\s+split=5\s+(?P<t>[\d.]+) ms')
    commit_rows = [build_row4(n, commit_gold.get(n), commit_baby.get(n), commit_mama.get(n), commit_mama_par.get(n))
                   for n in range(18, 28)]

    open_gold = parse_ml(DEEPFOLD_GOLDILOCKS,
                         r'Goldilocks prov128 Open NV=(?P<nv>\d+)\s+(?P<t>[\d.]+) ms')
    open_baby = parse_ml(DEEPFOLD_BABYBEAR,
                         r'BabyBear prov128 Open NV=(?P<nv>\d+)\s+(?P<t>[\d.]+) ms')
    open_mama = parse_ml(DEEPFOLD_MAMABEAR,
                         r'Ext3 prov128 Open NV=(?P<nv>\d+)\s+split=5\s+(?P<t>[\d.]+) ms')
    open_mama_par = parse_ml(DEEPFOLD_MAMABEAR,
                             r'Ext3 prov128 Open Par NV=(?P<nv>\d+)\s+split=5\s+(?P<t>[\d.]+) ms')
    open_rows = [build_row4(n, open_gold.get(n), open_baby.get(n), open_mama.get(n), open_mama_par.get(n))
                 for n in range(18, 28)]

    commit_tblr = render_component_tblr_body(commit_rows)
    open_tblr = render_component_tblr_body(open_rows)

    latex = (
        r"""\begin{table*}[t]
\centering
\caption{DeepFold performance across circuit sizes $2^{\mu}$ for $\mu = 18$ to $27$. Goldilocks and BabyBear serve as baselines. ``M.\ vs.\ G./B.'' reports the MamaBear speedup over Goldilocks / BabyBear. ``-Par'' columns use 8 threads. Our MamaBear implementations use 5-level splitting FFT. ``--'' indicates data not reported due to out-of-memory (OOM).}
\label{tab:deepfold-commit-open}
\begin{subtable}[t]{0.49\textwidth}
\centering
\caption{DeepFold-Commit time.}
\label{tab:deepfold-commit}
"""
        + commit_tblr
        + r"""
\end{subtable}\hfill
\begin{subtable}[t]{0.49\textwidth}
\centering
\caption{DeepFold-Open time.}
\label{tab:deepfold-open-prov128}
"""
        + open_tblr
        + r"""
\end{subtable}
\end{table*}
"""
    )
    return ('tab:deepfold-commit-open', latex)


def parse_with_tokens(path, required, forbidden=()):
    """Token-based parser (case-insensitive substring match on each line).
    A line matches iff all `required` tokens appear and none of `forbidden` do.
    Extracts NV=<n> and the final "<x> ms" timing. Returns dict[nv -> ms]."""
    out = {}
    if not path.exists():
        return out
    req = [t.lower() for t in required]
    fbd = [t.lower() for t in forbidden]
    for line in path.read_text().splitlines():
        ll = line.lower()
        if not all(t in ll for t in req):
            continue
        if any(t in ll for t in fbd):
            continue
        nv_m = re.search(r'NV=(\d+)', line, re.IGNORECASE)
        t_m = re.search(r'([\d.]+)\s*(?:ms|KB|KiB)\s*$', line.rstrip())
        if nv_m and t_m:
            out[int(nv_m.group(1))] = float(t_m.group(1))
    return out


def build_row_hp(n, g, b, p, pp, m, mp):
    g_s,  g_v  = fmt_time(g)
    b_s,  b_v  = fmt_time(b)
    p_s,  p_v  = fmt_time(p)
    pp_s, pp_v = fmt_time(pp)
    m_s,  m_v  = fmt_time(m)
    mp_s, mp_v = fmt_time(mp)
    return ' & '.join([
        str(n), g_s, b_s, p_s, pp_s,
        m_s,  f'{fmt_speedup(m_v,  g_v)}/{fmt_speedup(m_v,  b_v)}/{fmt_speedup(m_v,  p_v)}',
        mp_s, f'{fmt_speedup(mp_v, g_v)}/{fmt_speedup(mp_v, b_v)}/{fmt_speedup(mp_v, pp_v)}',
    ]) + r' \\'


def gen_hp_df_prove(*, security, mama_ext, caption, label, split=5):
    """Generate the HyperPlonk--DeepFold prover time table (9-column tblr in table*)."""
    gold_path   = HP_DF_GOLDILOCKS
    baby_path   = HP_DF_BABYBEAR
    mama_path   = HP_DF_MAMABEAR
    plonky_path = PLONKY3_PROVE

    split_tok = f'split={split}'
    gold       = parse_with_tokens(gold_path,   ['goldilocks', security, 'prove'], forbidden=['par'])
    baby       = parse_with_tokens(baby_path,   ['babybear',   security, 'prove'], forbidden=['par'])
    plonky     = parse_with_tokens(plonky_path, ['plonky3',    security, 'prove'], forbidden=['par'])
    plonky_par = parse_with_tokens(plonky_path, ['plonky3',    security, 'prove', 'par'])
    mama       = parse_with_tokens(mama_path,   ['mamabear', mama_ext, security, 'prove', split_tok], forbidden=['par'])
    mama_par   = parse_with_tokens(mama_path,   ['mamabear', mama_ext, security, 'prove', 'par', split_tok])

    rows = [build_row_hp(n, gold.get(n), baby.get(n), plonky.get(n), plonky_par.get(n),
                         mama.get(n), mama_par.get(n))
            for n in range(18, 28)]

    latex = (
        r"""%\begin{noIndentBlock}
\begin{table*}[t]
\centering
\caption{""" + caption + r"""}
\label{""" + label + r"""}
\small
\begin{tblr}{
  width = \linewidth,
  colspec = {Q[c,m] X[c,m] X[c,m] X[c,m] X[c,m] X[c,m] X[c,m] X[c,m] X[c,m]},
  row{1} = {font=\bfseries}, row{2} = {font=\bfseries}, rowsep = 1.5pt,
  cell{1}{1} = {r=2}{c,m}, cell{1}{2} = {r=2}{c,m}, cell{1}{3} = {r=2}{c,m}, cell{1}{4} = {r=2}{c,m}, cell{1}{5} = {r=2}{c,m},
  cell{1}{6} = {c=2}{c,m}, cell{1}{8} = {c=2}{c,m},
}
\toprule
$\mu$ & {Goldilocks\\$\F_{p^2}$} & {BabyBear\\$\F_{p^4}$} & {Plonky3\\B. $\F_{p^4}$} & {Plonky3-Par\\B. $\F_{p^4}$} & {MamaBear $\F_{p^3}$} & & {MamaBear-Par $\F_{p^3}$} & \\
& & & & & Time & {M.\ vs.\ G./B./P.} & Time & {M.\ vs.\ G./B./P.} \\
\midrule
"""
        + '\n'.join(rows)
        + r"""
\bottomrule
\end{tblr}
\end{table*}
%\end{noIndentBlock}
"""
    )
    return (label, latex)


def gen_hp_df_prove_prov128():
    return gen_hp_df_prove(
        security='prov128', mama_ext='ext3',
        caption=r"HyperPlonk--DeepFold prover time across circuit sizes $2^{\mu}$ for $\mu = 18$ to $27$. Goldilocks, BabyBear, and Plonky3 serve as baselines. ``M.\ vs.\ G./B./P.'' reports the MamaBear speedup over Goldilocks / BabyBear / Plonky3. ``-Par'' columns use 8 threads. Our MamaBear implementations use 5-level splitting FFT. ``-'' indicates data not reported due to out-of-memory (OOM).",
        label='tab:hp-df-prove-prov128',
        split=5,
    )


def gen_intro_teaser():
    """Generate the introduction's representative-prover-time table (paper Table 1).

    Same source rows as the end-to-end table (paper Table 6, `gen_hp_df_prove_prov128`),
    restricted to mu in {19, 21, 23} and reshaped to the intro's five columns. It is
    regenerated here rather than left to hand transcription so that a reviewer can
    check the paper's most visible table against measured data like any other."""
    label = 'tab:intro-hp-df-teaser'
    sec, split_tok = 'prov128', 'split=5'
    base = parse_with_tokens(HP_DF_GOLDILOCKS, ['goldilocks', sec, 'prove'], forbidden=['par'])
    mama = parse_with_tokens(HP_DF_MAMABEAR,
                             ['mamabear', 'ext3', sec, 'prove', split_tok], forbidden=['par'])
    mama_par = parse_with_tokens(HP_DF_MAMABEAR,
                                 ['mamabear', 'ext3', sec, 'prove', 'par', split_tok])

    rows = []
    for n in (19, 21, 23):
        b_s, b_v = fmt_time(base.get(n))
        m_s, m_v = fmt_time(mama.get(n))
        p_s, p_v = fmt_time(mama_par.get(n))
        rows.append(' & '.join([
            str(n), b_s, m_s, fmt_speedup(m_v, b_v), p_s, fmt_speedup(p_v, b_v),
        ]) + r' \\')

    latex = (
        r"""\begin{table}[!t]
\centering
\caption{Representative end-to-end prover times. Baseline is Goldilocks; both
``This work'' columns are MamaBear $\F_{p^3}$ with a 5-level FFT split. Rows are
$\mu \in \{19, 21, 23\}$ of Table~\ref{tab:hp-df-prove-prov128}.}
\label{""" + label + r"""}
\begin{tblr}{
        width = 0.98\linewidth,
        colspec = {Q[c,m] X[c,m] X[c,m] X[c,m] X[c,m] X[c,m]},
        row{1} = {font=\bfseries}, rowsep = 1.5pt,
    }
    \toprule
    $\mu$ & {Baseline} & This work & Speed-up & {This work\\Parallel} & Speed-up \\
    \midrule
    """ + '\n    '.join(rows) + r"""
    \bottomrule
\end{tblr}
\end{table}
"""
    )
    return (label, latex)


def gen_hp_df_prove_prov128_s3():
    return gen_hp_df_prove(
        security='prov128', mama_ext='ext3',
        caption=r"HyperPlonk--DeepFold prover time across circuit sizes $2^{\mu}$ for $\mu = 18$ to $27$. Goldilocks, BabyBear, and Plonky3 serve as baselines. ``M.\ vs.\ G./B./P.'' reports the MamaBear speedup over Goldilocks / BabyBear / Plonky3. ``-Par'' columns use 8 threads. Our MamaBear implementations use 3-level splitting FFT. ``-'' indicates data not reported due to out-of-memory (OOM).",
        label='tab:hp-df-prove-prov128-s3',
        split=3,
    )


def parse_circuit_size_exp(path):
    """Parse representative primitive circuit sizes as exponent strings."""
    out = {}
    if not path.exists():
        return out
    pattern = re.compile(
        r'^For (?P<name>.+?): 1 .+? corresponds to [\d,]+ \((?P<exp>2\^[\d.]+)\) gates\.$'
    )
    for line in path.read_text().splitlines():
        match = pattern.match(line)
        if match:
            out[match.group('name')] = match.group('exp')
    return out


def gen_primitive_circuit_size():
    """Generate the representative circuit sizes table."""
    parsed = parse_circuit_size_exp(HP_DF_CIRCUIT)
    rows_spec = [
        ('SHA256', 'SHA256', '1 block'),
        ('AES128', 'AES128', '1 call'),
        ('Blake3', 'BLAKE3', '1 permutation'),
        ('Keccak-f[1600]', 'Keccak-f[1600]', '1 permutation'),
        ('Poseidon2 (uniform x^11)', 'Poseidon2', '1 permutation'),
    ]

    def fmt_exp(exp):
        if exp is None:
            return '--'
        _, exponent = exp.split('^', 1)
        return f'$2^{{{exponent}}}$'

    rows = [
        f'{label} & {op} & {fmt_exp(parsed.get(key))} \\\\'
        for key, label, op in rows_spec
    ]

    latex = (
        r"""\begin{table}[H]
\centering
\caption{Representative circuit sizes.}
\label{tab:circuit-size}
\small
\begin{tblr}{
  width = \linewidth,
  colspec = {X[l,m] X[c,m] X[c,m]},
  row{1} = {font=\bfseries},
}
\toprule
Primitive & Operation & Circuit Size \\
\midrule
"""
        + '\n'.join(rows)
        + r"""
\bottomrule
\end{tblr}
\end{table}
"""
    )
    return ('tab:circuit-size', latex)


def parse_memory_gib(path, required, forbidden=()):
    """Like parse_with_tokens but the trailing value is a memory quantity.

    Returns GiB. Every producer in this repo divides by powers of 1024, so the
    suffix is read as a binary unit: a `GB` spelling is treated as GiB rather
    than converted from decimal. That is deliberate, not an oversight -- the
    decimal reading would silently inflate a Plonky3 row by 7.4% against the
    MamaBear rows it is compared with. Do not add a decimal branch here without
    first changing what the producers emit."""
    out = {}
    if not path.exists():
        return out
    req = [t.lower() for t in required]
    fbd = [t.lower() for t in forbidden]
    for line in path.read_text().splitlines():
        ll = line.lower()
        if not all(t in ll for t in req):
            continue
        if any(t in ll for t in fbd):
            continue
        nv_m = re.search(r'NV=(\d+)', line, re.IGNORECASE)
        t_m = re.search(r'([\d.]+)\s*(KB|KiB|MB|MiB|GB|GiB)\s*$', line.rstrip())
        if nv_m and t_m:
            v = float(t_m.group(1))
            u = t_m.group(2).lower()
            if u in ('kb', 'kib'):
                v /= 1024.0 * 1024.0
            elif u in ('mb', 'mib'):
                v /= 1024.0
            out[int(nv_m.group(1))] = v
    return out


def gen_hp_df_peak_memory():
    """Generate the peak memory table."""
    mem_path    = PEAK_MEMORY
    plonky_path = PLONKY3_PEAK_MEMORY

    gold   = parse_memory_gib(mem_path,    ['goldilocks', 'prov128', 'prove'], forbidden=['par'])
    baby   = parse_memory_gib(mem_path,    ['babybear',   'prov128', 'prove'], forbidden=['par'])
    plonky = parse_memory_gib(plonky_path, ['plonky3',    'prov128', 'prove'], forbidden=['par'])
    m3     = parse_memory_gib(mem_path,    ['mamabear',   'prov128', 'prove', 'split=5'], forbidden=['par'])
    m3p    = parse_memory_gib(mem_path,    ['mamabear',   'prov128', 'prove', 'par', 'split=5'])

    def fmt(v):
        return '--' if v is None else f'{round_half_up(v, 2):.2f}'

    rows = []
    for n in range(18, 28):
        g_v, b_v, p_v = gold.get(n), baby.get(n), plonky.get(n)
        m3_v, m3p_v = m3.get(n), m3p.get(n)
        def sp(mama_v):
            return '/'.join([fmt_speedup(mama_v, g_v),
                             fmt_speedup(mama_v, b_v),
                             fmt_speedup(mama_v, p_v)])
        cells = [
            str(n),
            fmt(g_v), fmt(b_v), fmt(p_v),
            fmt(m3_v),  sp(m3_v),
            fmt(m3p_v), sp(m3p_v),
        ]
        rows.append(' & '.join(cells) + r' \\')

    latex = (
        r"""\begin{table}[H]
\centering
\caption{HyperPlonk--DeepFold prover peak memory (in GiB) across circuit sizes $2^{\mu}$ for $\mu = 18$ to $27$. ``M.\ vs.\ G./B./P.'' reports the MamaBear memory reduction factor over Goldilocks / BabyBear / Plonky3. The MamaBear configuration uses the 5-level FFT split described in \S\ref{sec:split-analysis}. ``-'' indicates data not reported due to out-of-memory (OOM).}
\label{tab:hp-df-peak-memory}
\small
\begin{tblr}{
  width = \linewidth,
  colspec = {Q[c,m] X[c,m] X[c,m] X[c,m] X[c,m] X[1.35,c,m] X[c,m] X[1.35,c,m]},
    row{1} = {font=\bfseries}, row{2} = {font=\bfseries}, rowsep = 1.5pt,
    cell{1}{1} = {r=2}{c,m}, cell{1}{2} = {r=2}{c,m}, cell{1}{3} = {r=2}{c,m}, cell{1}{4} = {r=2}{c,m},
    cell{1}{5} = {c=2}{c,m}, cell{1}{7} = {c=2}{c,m},
}
\toprule
$\mu$ & Gold. & Baby. & Plonky3 & MamaBear & & MamaBear-Par & \\
& & & & Mem. & {M.\ vs.\ G./B./P.} & Mem. & {M.\ vs.\ G./B./P.} \\
\midrule
"""
        + '\n'.join(rows)
        + r"""
\bottomrule
\end{tblr}
\end{table}
"""
    )
    return ('tab:hp-df-peak-memory', latex)


def gen_hp_df_proof_size(*, split=5, caption=None, label=None):
    """Generate the proof size table."""
    size_path   = PROOF_SIZE
    plonky_path = PLONKY3_PROOF_SIZE

    split_tok = f'split={split}'
    backends = [
        ('Goldilocks', size_path,   ['goldilocks', 'prove'],          ['par']),
        ('BabyBear',   size_path,   ['babybear',   'prove'],          ['par']),
        ('Plonky3',    plonky_path, ['plonky3',    'proof_size'],     []),
        ('MamaBear',   size_path,   ['mamabear',   'prove', split_tok], ['par']),
    ]
    security = 'prov128'
    data = {b: parse_with_tokens(p, req + [security], fbd)
            for (b, p, req, fbd) in backends}

    def fmt(v):
        return '--' if v is None else f'{round_half_up(v, 2):.2f}'

    if label is None:
        label = 'tab:hp-df-proof-size'
    if caption is None:
        caption = (r"HyperPlonk--DeepFold proof size (in KiB) across circuit sizes $2^{\mu}$ "
                   r"for $\mu = 18$ to $27$. The MamaBear configuration uses the "
                   + f'{split}' + r"-level FFT split described in \S\ref{sec:split-analysis}. "
                   r"``-'' indicates data not reported due to out-of-memory (OOM).")

    def fmt_pct(mama_v, gold_v):
        if mama_v is None or gold_v is None or gold_v == 0:
            return '--'
        pct = (mama_v - gold_v) / gold_v * 100.0
        pct_r = round_half_up(pct, 1)
        sign = '+' if pct_r > 0 else ''
        return f'{sign}{pct_r:.1f}\\%'

    backend_order = ['Goldilocks', 'BabyBear', 'Plonky3', 'MamaBear']
    rows = []
    for n in range(18, 28):
        cells = [str(n)]
        for backend in backend_order:
            cells.append(fmt(data[backend].get(n)))
        cells.append(fmt_pct(data['MamaBear'].get(n), data['Goldilocks'].get(n)))
        rows.append(' & '.join(cells) + r' \\')

    latex = (
        r"""\begin{table}[H]
\centering
\caption{""" + caption + r"""}
\label{""" + label + r"""}
\small
\begin{tblr}{
  width = \linewidth,
  colspec = {Q[c,m] X[c,m] X[c,m] X[c,m] X[c,m] X[c,m]},
  row{1} = {font=\bfseries}, rowsep = 1.5pt,
}
\toprule
$\mu$ & Goldilocks & BabyBear & Plonky3 & MamaBear & {M.\ vs.\ G.} \\
\midrule
"""
        + '\n'.join(rows)
        + r"""
\bottomrule
\end{tblr}
\end{table}
"""
    )
    return (label, latex)


def gen_hp_df_proof_size_s3():
    return gen_hp_df_proof_size(
        split=3,
        label='tab:hp-df-proof-size-s3',
    )


def parse_field_op(path, backend, op):
    """Parse field.txt: lines look like
       'Goldilocks64Ext2       add             100000000      62.977 ms'
    Returns time in ms or None."""
    if not path.exists():
        return None
    pat = re.compile(
        r'^' + re.escape(backend) + r'\s+' + re.escape(op) +
        r'\s+\d+\s+(?P<t>[\d.]+)\s*ms\s*$'
    )
    for line in path.read_text().splitlines():
        m = pat.match(line)
        if m:
            return float(m.group('t'))
    return None


def gen_field_bench():
    """Generate the field arithmetic microbenchmark table."""
    path = FIELD_BENCH

    def fmt_ms(t):
        if t is None:
            return '--'
        return f'{round_half_up(t, 2):.2f}'

    def speedup(mama, ref):
        if mama is None or ref is None or mama == 0:
            return '--'
        ratio = round_half_up(ref / mama, 1)
        return f'{ratio:.1f}x'

    rows_spec = [
        ('Add', 'Goldilocks64Ext2', 'add',
                'BabyBearExt4', 'add',
                'BabyBearAVX-512Ext4', 'add',
                'MamaBearAVX-512Ext3', 'lazy_add'),
        ('Mul', 'Goldilocks64Ext2', 'mul',
                'BabyBearExt4', 'mul',
                'BabyBearAVX-512Ext4', 'mul',
                'MamaBearAVX-512Ext3', 'mul'),
    ]

    rendered_rows = []
    for label, gb, gop, bb, bop, pb, pop, mb, mop in rows_spec:
        g = parse_field_op(path, gb, gop)
        b = parse_field_op(path, bb, bop)
        p = parse_field_op(path, pb, pop)
        m = parse_field_op(path, mb, mop)
        sp = '/'.join([speedup(m, g), speedup(m, b), speedup(m, p)])
        cells = [label, fmt_ms(g), fmt_ms(b), fmt_ms(p), fmt_ms(m), sp]
        rendered_rows.append(' & '.join(cells) + r' \\')

    rf = parse_field_op(path, 'MamaBearAVX-512Ext3', 'reduce_fast')
    rf_cells = ['reduce\\_fast', '--', '--', '--', fmt_ms(rf), '--']
    rendered_rows.append(' & '.join(rf_cells) + r' \\')

    caption = (
        r"Field arithmetic microbenchmark: time (ms) for $10^8$ operations on the "
        r"$\F_{p^d}$ extension used by each backend (configuration matches "
        r"Table~\ref{tab:eval-config}). For our MamaBear addition we report the "
        r"$\mathsf{lazy\_add}()$ variant from \S\ref{sec:lazy-reduction}. "
        r"``M.\ vs.\ G./B./P.'' is the MamaBear speedup over Goldilocks / BabyBear / Plonky3."
    )

    latex = (
        r"""\begin{table}[t]
\centering
\caption{""" + caption + r"""}
\label{tab:field-bench}
\small
\begin{tblr}{
  width = \linewidth,
  colspec = {Q[c,m] X[c,m] X[c,m] X[c,m] X[c,m] X[1.4,c,m]},
  row{1} = {font=\bfseries}, rowsep = 1.5pt,
}
\toprule
Op & Gold. & Baby. & Plonky3 & MamaBear & {M.\ vs.\ G./B./P.} \\
\midrule
"""
        + '\n'.join(rendered_rows)
        + r"""
\bottomrule
\end{tblr}
\end{table}
"""
    )
    return ('tab:field-bench', latex)


# ============================================================================
# Single-document assembly.
# ============================================================================

# The table number each label carries IN THE PAPER. The generated document
# forces these numbers rather than counting from 1, so a reviewer can read
# "Table 6" here and "Table 6" in the paper and be looking at the same numbers.
#
# The gaps are real: paper Tables 2, 3, 7 and 8 are descriptive or analytically
# derived (field and parameter summaries, an eq-handling comparison, and
# asymptotic proof-size formulas) and correspond to no benchmark run, so
# nothing here generates them.
PAPER_TABLE_NUMBERS = {
    'tab:intro-hp-df-teaser':      1,
    'tab:zerocheck-prodcheck':     4,
    'tab:deepfold-commit-open':    5,
    'tab:hp-df-prove-prov128':     6,
    'tab:field-bench':             9,
    'tab:circuit-size':           10,
    'tab:hp-df-proof-size':       11,
    'tab:hp-df-prove-prov128-s3': 12,
    'tab:hp-df-proof-size-s3':    13,
    'tab:hp-df-peak-memory':      14,
}

# Preamble matching the paper's acmart sigconf format.
_PREAMBLE = r"""\documentclass[sigconf]{acmart}
\usepackage{amsmath,amsfonts}
\usepackage{booktabs}
\usepackage{subcaption}
\usepackage{tabularray}
\usepackage{float}
\UseTblrLibrary{booktabs}
\newcommand{\F}{\mathbb{F}}
\setcopyright{none}
\settopmatter{printacmref=false, printccs=false, printfolios=true}
\renewcommand\footnotetextcopyrightpermission[1]{}
\pagestyle{plain}
\providecommand{\shorttitle}[1]{}
\shorttitle{}
\begin{document}

\begin{center}
\textbf{\large MamaBearZKP artifact --- measured tables}
\end{center}

\noindent
Every table below is generated from the files under \texttt{results/} by
\texttt{plot\_tables.py}, and is \textbf{numbered and ordered as in the paper}:
Table~6 here is Table~6 there. Paper Tables~2, 3, 7 and~8 are descriptive or
analytically derived and correspond to no benchmark run, so the numbering skips
them. Each table starts a new page, in paper order.
\clearpage
"""

_POSTAMBLE = r"""
\end{document}
"""


def build_tables_document(tables):
    """Assemble (label, latex) pairs into one compilable LaTeX document.

    Two properties are forced here, both so the document can be read side by
    side with the paper:

    * Each table carries its PAPER number, via ``\setcounter`` before the float.
      ``\caption`` steps the counter where the float is *defined*, not where it
      is placed, so this holds however LaTeX later positions the float.
    * Tables appear in paper order. Left to itself, LaTeX defers a ``table*``
      in a two-column document to the top of a later page and can emit it after
      a single-column ``table`` that was defined later; ``\clearpage`` between
      tables flushes each float before the next is defined, which is what makes
      the order in this file the order in the PDF.
    """
    unnumbered = [label for label, _ in tables if label not in PAPER_TABLE_NUMBERS]
    if unnumbered:
        raise SystemExit(
            'plot_tables.py: no paper table number for: ' + ', '.join(unnumbered)
            + '\nAdd it to PAPER_TABLE_NUMBERS so the generated document keeps '
              'matching the paper.'
        )

    parts = [_PREAMBLE]
    for i, (label, latex) in enumerate(tables):
        parts.append('%% paper Table %d\n' % PAPER_TABLE_NUMBERS[label])
        parts.append('\\setcounter{table}{%d}\n' % (PAPER_TABLE_NUMBERS[label] - 1))
        parts.append(latex)
        if not latex.endswith('\n'):
            parts.append('\n')
        if i != len(tables) - 1:
            parts.append('\\clearpage\n')
    parts.append(_POSTAMBLE)
    return ''.join(parts)


def main():
    parser = argparse.ArgumentParser(description='Generate LaTeX tables from bench_results.')
    parser.add_argument('-p', '--path', dest='path_flag', default=None,
                        help='Output directory for tables.tex.')
    parser.add_argument('path_pos', nargs='?', default=None,
                        help='Output directory (positional alias for -p).')
    args = parser.parse_args()

    out_dir = Path(args.path_flag or args.path_pos or '.').resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Each generator returns a (label, latex) pair. Listed in PAPER order --
    # the numbers on the right are the paper's table numbers, and they are what
    # the generated document prints; see PAPER_TABLE_NUMBERS.
    tables = [
        gen_intro_teaser(),                     # 1
        gen_zerocheck_prodcheck_side_by_side(),  # 4
        gen_deepfold_commit_open_side_by_side(),  # 5
        gen_hp_df_prove_prov128(),              # 6
        gen_field_bench(),                      # 9
        gen_primitive_circuit_size(),           # 10
        gen_hp_df_proof_size(),                 # 11
        gen_hp_df_prove_prov128_s3(),           # 12
        gen_hp_df_proof_size_s3(),              # 13
        gen_hp_df_peak_memory(),                # 14
    ]

    document = build_tables_document(tables)
    out_path = out_dir / 'tables.tex'
    out_path.write_text(document)
    print(f'Written {len(tables)} tables to: {out_path}')


if __name__ == '__main__':
    main()
