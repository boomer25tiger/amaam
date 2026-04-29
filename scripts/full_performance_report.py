"""
Full performance report for the AMAAM baseline model.

Produces a plain-text report covering:
  1. Full-period summary vs SPY buy-and-hold
  2. In-sample / out-of-sample split
  3. Annual returns table
  4. Drawdown analysis
  5. Stress period analysis
  6. Return distribution

No modifications to the model or config are made.
"""

import sys

sys.path.insert(0, '/Users/GualyCr/Desktop/AMAAM/amaam/.claude/worktrees/reverent-mayer')

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from config.default_config import ModelConfig
from src.backtest.engine import run_backtest
from src.data.loader import load_validated_data

# ---------------------------------------------------------------------------
# Data and backtest
# ---------------------------------------------------------------------------

DATA_DIR = Path('/Users/GualyCr/Desktop/AMAAM/amaam/.claude/worktrees/reverent-mayer/data/processed')
data_dict = load_validated_data(DATA_DIR)
cfg = ModelConfig()
result = run_backtest(data_dict, cfg)

monthly_returns: pd.Series = result.monthly_returns

# SPY buy-and-hold monthly returns aligned to the same dates
spy_close = data_dict["SPY"]["Close"]
spy_monthly_returns = spy_close.resample("ME").last().pct_change().dropna()

# Align both series to the model's actual date range
model_start = monthly_returns.index.min()
model_end = monthly_returns.index.max()
spy_aligned = spy_monthly_returns.loc[
    (spy_monthly_returns.index >= model_start) & (spy_monthly_returns.index <= model_end)
]

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

PERIODS_PER_YEAR = 12.0


def ann_return(rets: pd.Series) -> float:
    """Annualised geometric return from monthly return series."""
    n = len(rets)
    if n == 0:
        return float("nan")
    total = np.prod(1.0 + rets.values)
    return float(total ** (PERIODS_PER_YEAR / n) - 1.0)


def ann_vol(rets: pd.Series) -> float:
    """Annualised volatility of monthly return series."""
    return float(rets.std(ddof=1) * np.sqrt(PERIODS_PER_YEAR))


def sharpe(rets: pd.Series, rf: float = 0.02) -> float:
    """Annualised Sharpe ratio (monthly risk-free rate derived from annual rf)."""
    rf_monthly = (1 + rf) ** (1 / PERIODS_PER_YEAR) - 1
    excess = rets - rf_monthly
    vol = ann_vol(rets)
    if vol == 0:
        return float("nan")
    return float(ann_return(rets) - rf) / vol


def sortino(rets: pd.Series, rf: float = 0.02) -> float:
    """Annualised Sortino ratio using downside deviation of negative monthly returns."""
    rf_monthly = (1 + rf) ** (1 / PERIODS_PER_YEAR) - 1
    downside = rets[rets < 0]
    if len(downside) == 0:
        return float("nan")
    downside_vol = float(downside.std(ddof=1) * np.sqrt(PERIODS_PER_YEAR))
    if downside_vol == 0:
        return float("nan")
    return float(ann_return(rets) - rf) / downside_vol


def max_drawdown(rets: pd.Series) -> float:
    """Maximum peak-to-trough drawdown from monthly return series."""
    equity = (1 + rets).cumprod()
    peak = equity.cummax()
    dd = (equity - peak) / peak
    return float(dd.min())


def calmar(rets: pd.Series) -> float:
    """Calmar ratio = Ann.Return / abs(Max Drawdown)."""
    mdd = abs(max_drawdown(rets))
    if mdd == 0:
        return float("nan")
    return ann_return(rets) / mdd


def pct_positive(rets: pd.Series) -> float:
    """Percentage of months with positive returns."""
    return float((rets > 0).sum() / len(rets) * 100)


def stats_block(rets: pd.Series) -> dict:
    """Compute the full stats block for a return series."""
    return {
        "ann_ret": ann_return(rets),
        "ann_vol": ann_vol(rets),
        "sharpe": sharpe(rets),
        "sortino": sortino(rets),
        "mdd": max_drawdown(rets),
        "calmar": calmar(rets),
        "pct_pos": pct_positive(rets),
        "best": float(rets.max()),
        "worst": float(rets.min()),
    }


def fmt_pct(v: float, decimals: int = 2) -> str:
    if np.isnan(v):
        return "    N/A"
    return f"{v * 100:+.{decimals}f}%"


def fmt_float(v: float, decimals: int = 2) -> str:
    if np.isnan(v):
        return "    N/A"
    return f"{v:+.{decimals}f}"


def print_separator(char: str = "─", width: int = 72) -> None:
    print(char * width)


def print_header(title: str, width: int = 72) -> None:
    print()
    print_separator("═", width)
    print(f"  {title}")
    print_separator("═", width)


# ---------------------------------------------------------------------------
# Section 1 — Full-period summary
# ---------------------------------------------------------------------------

print_header("SECTION 1 — FULL-PERIOD SUMMARY (2004–2026)")

m_stats = stats_block(monthly_returns)
s_stats = stats_block(spy_aligned)

col1_w = 30
col2_w = 16
col3_w = 16

header = f"{'Metric':<{col1_w}} {'AMAAM Model':>{col2_w}} {'SPY B&H':>{col3_w}}"
print(header)
print_separator("-", col1_w + col2_w + col3_w + 2)

rows = [
    ("Ann. Return",       fmt_pct(m_stats["ann_ret"]),  fmt_pct(s_stats["ann_ret"])),
    ("Ann. Volatility",   fmt_pct(m_stats["ann_vol"]),  fmt_pct(s_stats["ann_vol"])),
    ("Sharpe Ratio",      fmt_float(m_stats["sharpe"]), fmt_float(s_stats["sharpe"])),
    ("Sortino Ratio",     fmt_float(m_stats["sortino"]),fmt_float(s_stats["sortino"])),
    ("Max Drawdown",      fmt_pct(m_stats["mdd"]),      fmt_pct(s_stats["mdd"])),
    ("Calmar Ratio",      fmt_float(m_stats["calmar"]), fmt_float(s_stats["calmar"])),
    ("% Months Positive", f"{m_stats['pct_pos']:>+.1f}%", f"{s_stats['pct_pos']:>+.1f}%"),
    ("Best Month",        fmt_pct(m_stats["best"]),     fmt_pct(s_stats["best"])),
    ("Worst Month",       fmt_pct(m_stats["worst"]),    fmt_pct(s_stats["worst"])),
]

for label, mv, sv in rows:
    print(f"{label:<{col1_w}} {mv:>{col2_w}} {sv:>{col3_w}}")

print(f"\n  Period: {model_start.date()} → {model_end.date()}  ({len(monthly_returns)} months)")

# ---------------------------------------------------------------------------
# Section 2 — IS vs OOS split
# ---------------------------------------------------------------------------

print_header("SECTION 2 — IN-SAMPLE vs OUT-OF-SAMPLE SPLIT")

IS_END   = pd.Timestamp("2017-12-31")
OOS_START = pd.Timestamp("2018-01-01")

is_rets  = monthly_returns[monthly_returns.index <= IS_END]
oos_rets = monthly_returns[monthly_returns.index >= OOS_START]

is_spy  = spy_aligned[spy_aligned.index <= IS_END]
oos_spy = spy_aligned[spy_aligned.index >= OOS_START]

is_stats  = stats_block(is_rets)
oos_stats = stats_block(oos_rets)
is_spy_s  = stats_block(is_spy)
oos_spy_s = stats_block(oos_spy)

print(f"  IS  period: 2004-01-01 → 2017-12-31  ({len(is_rets)} months)")
print(f"  OOS period: 2018-01-01 → {model_end.date()}  ({len(oos_rets)} months)")
print()

hdr = f"{'Metric':<22} {'IS Model':>12} {'OOS Model':>12} {'IS SPY':>12} {'OOS SPY':>12}"
print(hdr)
print_separator("-", 72)

split_rows = [
    ("Ann. Return",       fmt_pct(is_stats["ann_ret"]),  fmt_pct(oos_stats["ann_ret"]),
                          fmt_pct(is_spy_s["ann_ret"]),  fmt_pct(oos_spy_s["ann_ret"])),
    ("Ann. Volatility",   fmt_pct(is_stats["ann_vol"]),  fmt_pct(oos_stats["ann_vol"]),
                          fmt_pct(is_spy_s["ann_vol"]),  fmt_pct(oos_spy_s["ann_vol"])),
    ("Sharpe Ratio",      fmt_float(is_stats["sharpe"]), fmt_float(oos_stats["sharpe"]),
                          fmt_float(is_spy_s["sharpe"]), fmt_float(oos_spy_s["sharpe"])),
    ("Max Drawdown",      fmt_pct(is_stats["mdd"]),      fmt_pct(oos_stats["mdd"]),
                          fmt_pct(is_spy_s["mdd"]),      fmt_pct(oos_spy_s["mdd"])),
    ("% Months Positive", f"{is_stats['pct_pos']:.1f}%",  f"{oos_stats['pct_pos']:.1f}%",
                          f"{is_spy_s['pct_pos']:.1f}%",  f"{oos_spy_s['pct_pos']:.1f}%"),
]

for row in split_rows:
    label = row[0]
    vals  = row[1:]
    print(f"{label:<22} {vals[0]:>12} {vals[1]:>12} {vals[2]:>12} {vals[3]:>12}")

is_sr  = is_stats["sharpe"]
oos_sr = oos_stats["sharpe"]
if not (np.isnan(is_sr) or is_sr == 0):
    degradation = (is_sr - oos_sr) / is_sr * 100
    print(f"\n  IS→OOS Sharpe degradation: ({is_sr:.2f} → {oos_sr:.2f}) = {degradation:+.1f}%")
else:
    print("\n  IS→OOS Sharpe degradation: N/A")

# ---------------------------------------------------------------------------
# Section 3 — Annual returns table
# ---------------------------------------------------------------------------

print_header("SECTION 3 — ANNUAL RETURNS TABLE")

# Compute calendar-year returns from monthly series
def annual_returns(rets: pd.Series) -> pd.Series:
    """Compound monthly returns to calendar-year returns."""
    return rets.groupby(rets.index.year).apply(lambda x: np.prod(1 + x) - 1)

model_annual = annual_returns(monthly_returns)
spy_annual   = annual_returns(spy_aligned)

all_years = sorted(set(model_annual.index) | set(spy_annual.index))

print(f"{'Year':>6}  {'AMAAM Model':>13}  {'SPY B&H':>13}  {'Difference':>13}")
print_separator("-", 52)

for yr in all_years:
    m_r = model_annual.get(yr, float("nan"))
    s_r = spy_annual.get(yr, float("nan"))
    diff = m_r - s_r if not (np.isnan(m_r) or np.isnan(s_r)) else float("nan")
    m_s = fmt_pct(m_r) if not np.isnan(m_r) else "     N/A"
    s_s = fmt_pct(s_r) if not np.isnan(s_r) else "     N/A"
    d_s = fmt_pct(diff) if not np.isnan(diff) else "     N/A"
    print(f"{yr:>6}  {m_s:>13}  {s_s:>13}  {d_s:>13}")

# ---------------------------------------------------------------------------
# Section 4 — Drawdown analysis
# ---------------------------------------------------------------------------

print_header("SECTION 4 — DRAWDOWN ANALYSIS")


def compute_drawdowns(rets: pd.Series):
    """
    Identify all drawdown episodes from a monthly return series.

    Returns list of dicts: start, trough, recovery, depth, duration_months.
    Duration is from start to recovery (or last date if ongoing).
    """
    equity = (1 + rets).cumprod()
    equity.index = pd.DatetimeIndex(equity.index)

    drawdowns = []
    peak_val  = equity.iloc[0]
    peak_date = equity.index[0]
    in_dd     = False
    dd_start  = None
    trough_val  = None
    trough_date = None

    for date, val in equity.items():
        if val >= peak_val:
            if in_dd:
                # Recovered
                depth = (trough_val - peak_val) / peak_val
                dur   = (date - dd_start).days / 30.5
                drawdowns.append({
                    "start":    dd_start,
                    "trough":   trough_date,
                    "recovery": date,
                    "depth":    depth,
                    "duration": dur,
                    "ongoing":  False,
                })
                in_dd = False
            peak_val  = val
            peak_date = date
        else:
            if not in_dd:
                in_dd      = True
                dd_start   = peak_date
                trough_val  = val
                trough_date = date
            elif val < trough_val:
                trough_val  = val
                trough_date = date

    # Check for ongoing drawdown at the end
    if in_dd:
        depth = (trough_val - peak_val) / peak_val
        dur   = (equity.index[-1] - dd_start).days / 30.5
        drawdowns.append({
            "start":    dd_start,
            "trough":   trough_date,
            "recovery": equity.index[-1],
            "depth":    depth,
            "duration": dur,
            "ongoing":  True,
        })

    return sorted(drawdowns, key=lambda d: d["depth"])


dds = compute_drawdowns(monthly_returns)
top5 = dds[:5]

print(f"  {'Start':<12}  {'Trough':<12}  {'Recovery':<14}  {'Depth':>10}  {'Duration':>10}")
print_separator("-", 68)
for dd in top5:
    rec_str = dd["recovery"].strftime("%Y-%m-%d") + (" (ongoing)" if dd["ongoing"] else "")
    print(
        f"  {dd['start'].strftime('%Y-%m-%d'):<12}  "
        f"{dd['trough'].strftime('%Y-%m-%d'):<12}  "
        f"{rec_str:<22}  "
        f"{dd['depth'] * 100:>+.2f}%  "
        f"{dd['duration']:>7.1f} mo"
    )

# Time in drawdown
equity_m = (1 + monthly_returns).cumprod()
peak_m   = equity_m.cummax()
in_dd_mask = equity_m < peak_m
pct_in_dd  = in_dd_mask.sum() / len(in_dd_mask) * 100
print(f"\n  Time in drawdown: {pct_in_dd:.1f}% of months ({in_dd_mask.sum()} of {len(in_dd_mask)})")

# ---------------------------------------------------------------------------
# Section 5 — Stress period analysis
# ---------------------------------------------------------------------------

print_header("SECTION 5 — STRESS PERIOD ANALYSIS")

STRESS_PERIODS = [
    ("GFC Peak-to-Trough",    "2007-10-01", "2009-03-31"),
    ("Euro Crisis",           "2010-04-01", "2011-09-30"),
    ("Taper Tantrum",         "2013-05-01", "2013-06-30"),
    ("China Selloff",         "2015-08-01", "2016-02-29"),
    ("COVID Crash",           "2020-02-01", "2020-03-31"),
    ("2022 Rate Shock",       "2022-01-01", "2022-12-31"),
    ("2025 Tariff Shock",     "2025-02-01", cfg.backtest_end),
]

print(f"  {'Stress Period':<28}  {'AMAAM Cum.Ret':>15}  {'SPY Cum.Ret':>13}")
print_separator("-", 62)

for label, start, end in STRESS_PERIODS:
    ts = pd.Timestamp(start)
    te = pd.Timestamp(end)

    m_slice = monthly_returns[(monthly_returns.index >= ts) & (monthly_returns.index <= te)]
    s_slice = spy_aligned[(spy_aligned.index >= ts) & (spy_aligned.index <= te)]

    m_cum = float(np.prod(1 + m_slice) - 1) if len(m_slice) > 0 else float("nan")
    s_cum = float(np.prod(1 + s_slice) - 1) if len(s_slice) > 0 else float("nan")

    m_str = fmt_pct(m_cum) if not np.isnan(m_cum) else "     N/A"
    s_str = fmt_pct(s_cum) if not np.isnan(s_cum) else "     N/A"

    print(f"  {label:<28}  {m_str:>15}  {s_str:>13}")

# ---------------------------------------------------------------------------
# Section 6 — Return distribution
# ---------------------------------------------------------------------------

print_header("SECTION 6 — RETURN DISTRIBUTION")

m_vals = monthly_returns.values
skew = float(sp_stats.skew(m_vals))
kurt = float(sp_stats.kurtosis(m_vals))  # excess kurtosis

print(f"  Skewness (monthly):         {skew:+.4f}")
print(f"  Excess Kurtosis (monthly):  {kurt:+.4f}")
print()

buckets = [
    ("< -5%",         -np.inf,  -0.05),
    ("-5% to -3%",    -0.05,    -0.03),
    ("-3% to -1%",    -0.03,    -0.01),
    ("-1% to  0%",    -0.01,     0.00),
    ("  0% to  1%",   0.00,      0.01),
    ("  1% to  3%",   0.01,      0.03),
    ("  3% to  5%",   0.03,      0.05),
    ("> 5%",           0.05,    np.inf),
]

total_months = len(monthly_returns)
print(f"  {'Bucket':<20}  {'Count':>7}  {'% of Months':>13}")
print_separator("-", 46)

for label, lo, hi in buckets:
    mask  = (monthly_returns > lo) & (monthly_returns <= hi)
    count = int(mask.sum())
    pct   = count / total_months * 100
    print(f"  {label:<20}  {count:>7}  {pct:>12.1f}%")

print_separator("-", 46)
print(f"  {'Total':<20}  {total_months:>7}  {100.0:>12.1f}%")

print()
print_separator("═", 72)
print("  END OF REPORT")
print_separator("═", 72)
print()
