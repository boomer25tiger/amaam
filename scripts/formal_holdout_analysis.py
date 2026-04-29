"""
Formal Holdout Analysis: 2024-01-01 through cfg.backtest_end.

This script is run EXACTLY ONCE and never re-run based on its results.
It reports AMAAM performance on the truly unseen 2024-2026 window,
comparing against:
  - IS period  (2004-2017) — design/training window
  - OOS period (2018-2023) — out-of-sample period used in earlier diagnostics
  - SPY buy-and-hold benchmark

Sections
--------
1. Summary performance table (Ann Return, Sharpe, MDD, Calmar, Sortino, Vol)
2. Annual returns: 2024, 2025, 2026-YTD vs SPY
3. Monthly return detail (all months in holdout)
4. Drawdown analysis — max drawdown depth and duration
5. Regime analysis — how the trend filter and selection shifted during 2024-2026
6. 2025 tariff-shock stress window (2025-02 through 2025-05)
7. IS → OOS → Holdout performance progression
8. Alpha / Beta vs SPY in holdout window
"""

import sys
sys.path.insert(0, '/Users/GualyCr/Desktop/AMAAM/amaam/.claude/worktrees/reverent-mayer')

import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from config.default_config import ModelConfig
from src.backtest.engine import run_backtest
from src.data.loader import load_validated_data
from src.backtest.metrics import compute_drawdown_series

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# 0.  Setup
# ─────────────────────────────────────────────────────────────────────────────

DATA_DIR = Path(
    '/Users/GualyCr/Desktop/AMAAM/amaam/.claude/worktrees/reverent-mayer/data/processed'
)

data_dict = load_validated_data(DATA_DIR)
cfg       = ModelConfig()
result    = run_backtest(data_dict, cfg)

# Align AMAAM monthly returns to period index
mr = result.monthly_returns.copy()
mr.index = mr.index.to_period('M')

# SPY monthly returns — same alignment
spy_daily = data_dict['SPY']['Close']
spy_me    = spy_daily.resample('ME').last()
spy_ret   = spy_me.pct_change().dropna()
spy_ret.index = spy_ret.index.to_period('M')

# Align on common dates
common = mr.index.intersection(spy_ret.index)
mr  = mr.reindex(common)
spy = spy_ret.reindex(common)

# ── Period masks ─────────────────────────────────────────────────────────────
is_mask      = common.year <= 2017
oos_mask     = (common.year >= 2018) & (common.year <= 2023)
holdout_mask = common.year >= 2024

mr_is   = mr[is_mask];   spy_is   = spy[is_mask]
mr_oos  = mr[oos_mask];  spy_oos  = spy[oos_mask]
mr_ho   = mr[holdout_mask]; spy_ho = spy[holdout_mask]


# ─────────────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────────────

def ann_return(r: pd.Series) -> float:
    n = len(r)
    if n == 0:
        return np.nan
    tot = float((1.0 + r).prod() - 1.0)
    return float((1.0 + tot) ** (12.0 / n) - 1.0)


def ann_vol(r: pd.Series) -> float:
    return float(r.std() * math.sqrt(12))


def sharpe(r: pd.Series, rf: float = 0.02) -> float:
    if len(r) == 0:
        return np.nan
    v = ann_vol(r)
    if v == 0:
        return np.nan
    period_rf = (1.0 + rf) ** (1.0 / 12.0) - 1.0
    excess    = r - period_rf
    return float(excess.mean() * 12 / (r.std() * math.sqrt(12)))


def sortino(r: pd.Series, rf: float = 0.02) -> float:
    if len(r) == 0:
        return np.nan
    period_rf = (1.0 + rf) ** (1.0 / 12.0) - 1.0
    excess    = r - period_rf
    downside  = excess[excess < 0]
    if len(downside) == 0:
        return np.nan
    dd_dev = float(math.sqrt((downside ** 2).mean()) * math.sqrt(12))
    if dd_dev == 0:
        return np.nan
    return float((ann_return(r) - rf) / dd_dev)


def mdd(r: pd.Series) -> float:
    if len(r) == 0:
        return np.nan
    return float(compute_drawdown_series(r).min())


def calmar(r: pd.Series) -> float:
    ar = ann_return(r)
    md = mdd(r)
    if md == 0 or np.isnan(md):
        return np.nan
    return ar / abs(md)


def mdd_duration(r: pd.Series) -> int:
    """Longest consecutive run of months in drawdown (below prior peak)."""
    dd = compute_drawdown_series(r)
    max_dur, cur = 0, 0
    for v in dd:
        if v < 0:
            cur += 1
            max_dur = max(max_dur, cur)
        else:
            cur = 0
    return max_dur


def pct_positive(r: pd.Series) -> float:
    return float((r > 0).mean())


def ols_alpha_beta(r_a: pd.Series, r_b: pd.Series):
    """Return (annualised alpha, beta, R², t_alpha, p_alpha)."""
    x = r_b.values
    y = r_a.values
    X = np.column_stack([np.ones(len(x)), x])
    coeffs, *_ = np.linalg.lstsq(X, y, rcond=None)
    alpha_m, beta = coeffs[0], coeffs[1]
    alpha_ann = (1.0 + alpha_m) ** 12 - 1.0

    y_hat = X @ coeffs
    resid = y - y_hat
    n, k  = len(y), 2
    s2    = np.sum(resid ** 2) / (n - k)
    se    = np.sqrt(np.diag(np.linalg.inv(X.T @ X) * s2))
    t_a   = alpha_m / se[0]
    p_a   = 2.0 * stats.t.sf(abs(t_a), n - k)

    ss_tot = np.sum((y - y.mean()) ** 2)
    r2     = 1.0 - np.sum(resid ** 2) / ss_tot if ss_tot > 0 else 0.0
    return alpha_ann, beta, r2, t_a, p_a


def metrics_row(label: str, r: pd.Series, r_spy: pd.Series) -> dict:
    return {
        "Period":         label,
        "N Months":       len(r),
        "Ann Return":     ann_return(r),
        "Ann Vol":        ann_vol(r),
        "Sharpe":         sharpe(r),
        "Sortino":        sortino(r),
        "Max Drawdown":   mdd(r),
        "Calmar":         calmar(r),
        "MDD Duration":   mdd_duration(r),
        "% Pos Months":   pct_positive(r),
        "SPY Ann Return": ann_return(r_spy),
        "SPY Sharpe":     sharpe(r_spy),
        "SPY Max DD":     mdd(r_spy),
    }


def fmt_pct(x) -> str:
    if np.isnan(x):
        return "  N/A "
    return f"{x * 100:+6.1f}%"


def fmt_f(x, decimals=2) -> str:
    if np.isnan(x):
        return "  N/A "
    return f"{x:+.{decimals}f}"


W   = 76
SEP = "=" * W


# ─────────────────────────────────────────────────────────────────────────────
# Section 1 — Performance comparison table
# ─────────────────────────────────────────────────────────────────────────────

rows = [
    metrics_row("IS  (2004–2017)",   mr_is,  spy_is),
    metrics_row("OOS (2018–2023)",   mr_oos, spy_oos),
    metrics_row("HOLDOUT (2024–26)", mr_ho,  spy_ho),
]

print(SEP)
print(" FORMAL HOLDOUT ANALYSIS — AMAAM vs SPY")
print(f" Holdout window: 2024-01-01 -> {cfg.backtest_end}  |  Run: ONCE, never re-run")
print(SEP)
print()
print("SECTION 1: PERFORMANCE SUMMARY")
print()

COLS = [
    ("Ann Ret",  "Ann Return",    lambda x: fmt_pct(x)),
    ("Ann Vol",  "Ann Vol",       lambda x: fmt_pct(x)),
    ("Sharpe",   "Sharpe",        lambda x: fmt_f(x)),
    ("Sortino",  "Sortino",       lambda x: fmt_f(x)),
    ("Max DD",   "Max Drawdown",  lambda x: fmt_pct(x)),
    ("Calmar",   "Calmar",        lambda x: fmt_f(x)),
    ("% Pos",    "% Pos Months",  lambda x: fmt_pct(x)),
]

# Header
hdr = f"{'Period':<22}" + "".join(f"{c[0]:>9}" for c in COLS)
print(hdr)
print("-" * W)

for row in rows:
    line = f"{row['Period']:<22}"
    for _, key, fmt_fn in COLS:
        line += f"{fmt_fn(row[key]):>9}"
    print(line)

print()
print("AMAAM vs SPY:")
spy_cols = [
    ("Ann Ret",  "SPY Ann Return",  lambda x: fmt_pct(x)),
    ("Sharpe",   "SPY Sharpe",      lambda x: fmt_f(x)),
    ("Max DD",   "SPY Max DD",      lambda x: fmt_pct(x)),
]
hdr2 = f"{'Period':<22}" + "".join(f"{c[0]:>9}" for c in spy_cols)
print(hdr2)
print("-" * 40)

for row in rows:
    line = f"{row['Period']:<22}"
    for _, key, fmt_fn in spy_cols:
        line += f"{fmt_fn(row[key]):>9}"
    print(line)

print()

# ─────────────────────────────────────────────────────────────────────────────
# Section 2 — Annual returns
# ─────────────────────────────────────────────────────────────────────────────

print("SECTION 2: ANNUAL RETURNS — HOLDOUT PERIOD")
print()

def annual_returns(r: pd.Series) -> pd.Series:
    return r.groupby(r.index.year).apply(lambda g: float((1.0 + g).prod() - 1.0))

ho_annual    = annual_returns(mr_ho)
spy_annual   = annual_returns(spy_ho)
oos_annual   = annual_returns(mr_oos)
spy_oos_ann  = annual_returns(spy_oos)
is_annual    = annual_returns(mr_is)
spy_is_ann   = annual_returns(spy_is)

print(f"{'Year':<8}{'AMAAM':>10}{'SPY':>10}{'Excess':>10}{'# Months':>10}")
print("-" * 48)

for yr in sorted(ho_annual.index):
    a_ret = ho_annual.get(yr, np.nan)
    s_ret = spy_annual.get(yr, np.nan)
    exc   = a_ret - s_ret if not (np.isnan(a_ret) or np.isnan(s_ret)) else np.nan
    n_mo  = int((mr_ho.index.year == yr).sum())
    tag   = " (partial)" if yr == 2026 else ""
    print(
        f"{yr}{tag:<10}"
        f"{fmt_pct(a_ret):>10}"
        f"{fmt_pct(s_ret):>10}"
        f"{fmt_pct(exc):>10}"
        f"{n_mo:>10}"
    )

print()

# ─────────────────────────────────────────────────────────────────────────────
# Section 3 — Monthly return detail
# ─────────────────────────────────────────────────────────────────────────────

print("SECTION 3: MONTHLY RETURN DETAIL — HOLDOUT PERIOD")
print()
print(f"{'Month':<10}{'AMAAM':>9}{'SPY':>9}{'Excess':>9}  {'Notes'}")
print("-" * 60)

# Tariff shock window
tariff_months = {
    pd.Period('2025-02', 'M'), pd.Period('2025-03', 'M'),
    pd.Period('2025-04', 'M'), pd.Period('2025-05', 'M'),
}

for per in mr_ho.index:
    a = mr_ho[per]
    s = spy_ho.get(per, np.nan)
    e = a - s if not np.isnan(s) else np.nan
    note = ""
    if per in tariff_months:
        note = "<-- tariff shock"
    elif abs(a) > 0.06:
        note = "<-- notable"
    print(
        f"{str(per):<10}"
        f"{fmt_pct(a):>9}"
        f"{fmt_pct(s) if not np.isnan(s) else '  N/A ':>9}"
        f"{fmt_pct(e) if not np.isnan(e) else '  N/A ':>9}"
        f"  {note}"
    )

print()

# ─────────────────────────────────────────────────────────────────────────────
# Section 4 — Drawdown analysis
# ─────────────────────────────────────────────────────────────────────────────

print("SECTION 4: DRAWDOWN ANALYSIS")
print()

dd_ho     = compute_drawdown_series(mr_ho)
dd_spy_ho = compute_drawdown_series(spy_ho)

print(f"  AMAAM max drawdown (holdout):      {mdd(mr_ho)*100:+.2f}%")
print(f"  SPY   max drawdown (holdout):      {mdd(spy_ho)*100:+.2f}%")
print()
print(f"  AMAAM MDD duration (holdout):      {mdd_duration(mr_ho)} months")
print(f"  AMAAM % months in drawdown:        {(dd_ho < 0).mean()*100:.1f}%")
print()

# Worst drawdown period
min_idx = dd_ho.idxmin()
print(f"  Deepest drawdown trough:           {str(min_idx)} ({dd_ho[min_idx]*100:+.2f}%)")

print()

# ─────────────────────────────────────────────────────────────────────────────
# Section 5 — Regime analysis: holdings / trend signal during 2024-2026
# ─────────────────────────────────────────────────────────────────────────────

print("SECTION 5: HOLDINGS AND REGIME ANALYSIS")
print()

# Use allocations DataFrame (index = month-end date, columns = tickers, values = weights)
allocs = result.allocations.copy()
allocs.index = pd.to_datetime(allocs.index)

ho_start = pd.Timestamp('2024-01-01')
ho_end   = pd.Timestamp(cfg.backtest_end)

allocs_ho = allocs.loc[(allocs.index >= ho_start) & (allocs.index <= ho_end)]

# Identify hedging sleeve tickers: GLD/TLT/IEF/SH/UUP/SHY
HEDGING_TICKERS = {'GLD', 'TLT', 'IEF', 'SH', 'UUP', 'SHY'}
ALL_TICKERS     = set(allocs.columns.tolist())
MAIN_TICKERS    = ALL_TICKERS - HEDGING_TICKERS

if not allocs_ho.empty:
    print(f"  {'Month':<10}  {'Main sleeve (6)':<52}  {'Hedge (2)'}")
    print("  " + "-" * 80)
    for date, row in allocs_ho.iterrows():
        per      = pd.Period(date, 'M')
        main_sel = sorted([t for t in MAIN_TICKERS if row.get(t, 0) > 0.001])
        hdg_sel  = sorted([t for t in HEDGING_TICKERS if row.get(t, 0) > 0.001])
        main_str = ", ".join(main_sel) if main_sel else "(none)"
        hdg_str  = ", ".join(hdg_sel)  if hdg_sel  else "(none)"
        print(f"  {str(per):<10}  {main_str:<52}  {hdg_str}")
else:
    print("  No holdout allocations found in result.")

print()

# ─────────────────────────────────────────────────────────────────────────────
# Section 6 — 2025 tariff-shock stress window
# ─────────────────────────────────────────────────────────────────────────────

print("SECTION 6: 2025 TARIFF-SHOCK STRESS WINDOW (Feb–May 2025)")
print()

tariff_periods = [pd.Period(f'2025-{m:02d}', 'M') for m in range(2, 6)]
mr_tariff  = mr_ho.reindex(tariff_periods).dropna()
spy_tariff = spy_ho.reindex(tariff_periods).dropna()

if len(mr_tariff) > 0:
    amaam_cum = float((1 + mr_tariff).prod() - 1)
    spy_cum   = float((1 + spy_tariff).prod() - 1) if len(spy_tariff) > 0 else np.nan
    print(f"  AMAAM cumulative (Feb–May 2025): {amaam_cum*100:+.2f}%")
    if not np.isnan(spy_cum):
        print(f"  SPY   cumulative (Feb–May 2025): {spy_cum*100:+.2f}%")
        print(f"  Excess return:                   {(amaam_cum - spy_cum)*100:+.2f}%")
    print()
    print(f"  {'Month':<10}  {'AMAAM':>8}  {'SPY':>8}  {'Excess':>8}")
    print("  " + "-" * 38)
    for per in tariff_periods:
        a = mr_tariff.get(per, np.nan)
        s = spy_tariff.get(per, np.nan) if per in spy_tariff.index else np.nan
        e = a - s if not (np.isnan(a) or np.isnan(s)) else np.nan
        print(
            f"  {str(per):<10}"
            f"  {fmt_pct(a):>8}"
            f"  {fmt_pct(s) if not np.isnan(s) else '  N/A ':>8}"
            f"  {fmt_pct(e) if not np.isnan(e) else '  N/A ':>8}"
        )
else:
    print("  Tariff-shock period not yet available in data.")

print()

# ─────────────────────────────────────────────────────────────────────────────
# Section 7 — IS → OOS → Holdout progression
# ─────────────────────────────────────────────────────────────────────────────

print("SECTION 7: IS → OOS → HOLDOUT PROGRESSION")
print()

prog_rows = [
    ("IS  (2004–2017)",   mr_is,  spy_is,  len(mr_is)),
    ("OOS (2018–2023)",   mr_oos, spy_oos, len(mr_oos)),
    ("Holdout (2024–26)", mr_ho,  spy_ho,  len(mr_ho)),
]

print(f"{'Period':<22}{'N Mo':>6}{'Ann Ret':>9}{'Sharpe':>9}{'Max DD':>9}{'Calmar':>9}  SPY SR")
print("-" * W)

for lbl, r, r_spy, n in prog_rows:
    ar  = ann_return(r)
    sr  = sharpe(r)
    md_ = mdd(r)
    cal = calmar(r)
    sspy = sharpe(r_spy)
    print(
        f"{lbl:<22}"
        f"{n:>6}"
        f"{fmt_pct(ar):>9}"
        f"{fmt_f(sr):>9}"
        f"{fmt_pct(md_):>9}"
        f"{fmt_f(cal):>9}"
        f"  {fmt_f(sspy)}"
    )

# Degradation metric
print()
print("  Sharpe progression (IS → OOS → Holdout):")
vals = [(lbl, sharpe(r)) for lbl, r, _, _ in prog_rows]
for lbl, sr in vals:
    bar_n = max(0, int((sr / 1.5) * 30))
    bar   = "█" * bar_n
    print(f"    {lbl:<22}  {sr:+.3f}  {bar}")

print()

# ─────────────────────────────────────────────────────────────────────────────
# Section 8 — Alpha / Beta in holdout window
# ─────────────────────────────────────────────────────────────────────────────

print("SECTION 8: ALPHA / BETA DECOMPOSITION")
print()

common_ho = mr_ho.index.intersection(spy_ho.index)
mr_ho_aligned  = mr_ho.reindex(common_ho)
spy_ho_aligned = spy_ho.reindex(common_ho)

if len(mr_ho_aligned) >= 12:
    alpha_ann, beta, r2, t_a, p_a = ols_alpha_beta(mr_ho_aligned, spy_ho_aligned)
    print(f"  Holdout Alpha (annualised): {alpha_ann*100:+.2f}% / yr"
          f"    (t = {t_a:+.2f}, p = {p_a:.4f})")
    print(f"  Holdout Beta:               {beta:+.3f}")
    print(f"  Holdout R²:                 {r2:.3f}")
    print()
    print("  For comparison (from prior analysis):")

    # IS alpha/beta
    common_is = mr_is.index.intersection(spy_is.index)
    a_is, b_is, r2_is, t_is, p_is = ols_alpha_beta(
        mr_is.reindex(common_is), spy_is.reindex(common_is)
    )
    # OOS alpha/beta
    common_oos = mr_oos.index.intersection(spy_oos.index)
    a_oos, b_oos, r2_oos, t_oos, p_oos = ols_alpha_beta(
        mr_oos.reindex(common_oos), spy_oos.reindex(common_oos)
    )
    print(f"  IS  Alpha: {a_is*100:+.2f}%/yr,  Beta: {b_is:+.3f},  R²: {r2_is:.3f}")
    print(f"  OOS Alpha: {a_oos*100:+.2f}%/yr,  Beta: {b_oos:+.3f},  R²: {r2_oos:.3f}")
    print(f"  HOL Alpha: {alpha_ann*100:+.2f}%/yr,  Beta: {beta:+.3f},  R²: {r2:.3f}")
else:
    print(f"  Insufficient holdout months ({len(mr_ho_aligned)}) for regression.")

print()

# ─────────────────────────────────────────────────────────────────────────────
# Summary verdict
# ─────────────────────────────────────────────────────────────────────────────

print(SEP)
print(" HOLDOUT VERDICT")
print(SEP)
print()

ho_sr   = sharpe(mr_ho)
oos_sr  = sharpe(mr_oos)
is_sr   = sharpe(mr_is)
ho_mdd  = mdd(mr_ho)
oos_mdd = mdd(mr_oos)

# Did the model degrade substantially?
degrade_threshold = 0.30  # Sharpe drop > 0.30 considered meaningful degradation

sr_delta_ho_vs_oos = ho_sr - oos_sr if not (np.isnan(ho_sr) or np.isnan(oos_sr)) else np.nan
sr_delta_ho_vs_is  = ho_sr - is_sr  if not (np.isnan(ho_sr) or np.isnan(is_sr))  else np.nan

def verdict_str(delta: float) -> str:
    if np.isnan(delta):
        return "N/A"
    if delta > 0.05:
        return "IMPROVED"
    if delta > -degrade_threshold:
        return "STABLE"
    return "DEGRADED"

print(f"  Holdout Sharpe:           {ho_sr:+.3f}")
print(f"  vs OOS Sharpe ({oos_sr:+.3f}):    {sr_delta_ho_vs_oos:+.3f}  →  {verdict_str(sr_delta_ho_vs_oos)}")
print(f"  vs IS  Sharpe ({is_sr:+.3f}):    {sr_delta_ho_vs_is:+.3f}  →  {verdict_str(sr_delta_ho_vs_is)}")
print()
print(f"  Holdout Ann Return:       {ann_return(mr_ho)*100:+.2f}%")
print(f"  Holdout Max Drawdown:     {ho_mdd*100:+.2f}%")
print(f"  Holdout Calmar:           {calmar(mr_ho):+.2f}")
print()

# Beat SPY?
spy_ho_sr = sharpe(spy_ho)
print(f"  SPY Holdout Sharpe:       {spy_ho_sr:+.3f}")
beat_spy = (ho_sr > spy_ho_sr) if not (np.isnan(ho_sr) or np.isnan(spy_ho_sr)) else None
print(f"  AMAAM {'BEAT' if beat_spy else 'TRAILED'} SPY on Sharpe in holdout")
print()
print("  (This analysis was run once and is final. No parameters will be")
print("  adjusted based on these holdout results.)")
print()
print(SEP)
