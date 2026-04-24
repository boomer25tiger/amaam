"""
Biweekly vs Monthly rebalancing comparison for AMAAM.

Runs two full backtests (A = monthly baseline, B = biweekly) and prints a
structured comparison covering: full-period metrics, IS/OOS split, annual
returns table, stress-period drawdowns, transaction cost analysis, and the
top-5 drawdowns for the biweekly config.
"""

import sys
import math
sys.path.insert(0, '/Users/GualyCr/Desktop/AMAAM/amaam/.claude/worktrees/reverent-mayer')

from pathlib import Path

import numpy as np
import pandas as pd

from config.default_config import ModelConfig
from src.backtest.engine import run_backtest
from src.data.loader import load_validated_data

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def pct(x: float, decimals: int = 2) -> str:
    """Format decimal as percentage string."""
    return f"{x * 100:.{decimals}f}%"


def signed_pct(x: float, decimals: int = 2) -> str:
    """Format delta as signed percentage string."""
    sign = "+" if x >= 0 else ""
    return f"{sign}{x * 100:.{decimals}f}%"


def ann_return(rets: pd.Series) -> float:
    """Annualised return from a monthly-frequency return series."""
    n = len(rets)
    if n == 0:
        return float("nan")
    total = float((1.0 + rets).prod() - 1.0)
    return float((1.0 + total) ** (12.0 / n) - 1.0)


def ann_vol(rets: pd.Series) -> float:
    """Annualised volatility from a monthly return series."""
    return float(rets.std() * math.sqrt(12.0))


def sharpe(rets: pd.Series, rf: float = 0.02) -> float:
    """Sharpe ratio (monthly data, annualised, rf = 2%)."""
    v = ann_vol(rets)
    if v == 0 or math.isnan(v):
        return float("nan")
    return float((ann_return(rets) - rf) / v)


def sortino(rets: pd.Series, rf: float = 0.02) -> float:
    """Sortino ratio (downside deviation below rf)."""
    period_rf = (1.0 + rf) ** (1.0 / 12.0) - 1.0
    excess = rets - period_rf
    down = excess[excess < 0]
    if len(down) == 0:
        return float("nan")
    dd_dev = float(math.sqrt((down ** 2).mean()) * math.sqrt(12.0))
    if dd_dev == 0:
        return float("nan")
    return float((ann_return(rets) - rf) / dd_dev)


def max_dd(rets: pd.Series) -> float:
    """Maximum drawdown (most negative value, decimal)."""
    equity = (1.0 + rets).cumprod()
    peak = equity.cummax()
    dd_series = (equity - peak) / peak
    return float(dd_series.min())


def calmar(rets: pd.Series) -> float:
    mdd = max_dd(rets)
    ar = ann_return(rets)
    if mdd >= 0:
        return float("nan")
    return float(ar / abs(mdd))


def period_return(rets: pd.Series, start: str, end: str) -> float:
    """
    Cumulative return over a date slice.

    Uses the monthly returns index; months that overlap the stress period
    boundaries are included.  Returns NaN if no data in range.
    """
    sub = rets.loc[start:end]
    if sub.empty:
        return float("nan")
    return float((1.0 + sub).prod() - 1.0)


def top5_drawdowns(rets: pd.Series):
    """
    Identify the 5 worst drawdown episodes in a monthly return series.

    Returns a list of dicts with keys: start, trough, recovery, depth, duration.
    """
    equity = (1.0 + rets).cumprod()
    peak = equity.cummax()
    dd_series = (equity - peak) / peak

    episodes = []
    in_dd = False
    ep_start = None
    ep_peak_val = None

    for date, dd_val in dd_series.items():
        if not in_dd and dd_val < 0:
            in_dd = True
            ep_start = date
            ep_peak_val = peak.loc[date]
            ep_trough_date = date
            ep_trough_val = dd_val
        elif in_dd:
            if dd_val < ep_trough_val:
                ep_trough_val = dd_val
                ep_trough_date = date
            if dd_val >= 0:
                episodes.append({
                    "start": ep_start,
                    "trough": ep_trough_date,
                    "recovery": date,
                    "depth": ep_trough_val,
                    "duration": len(dd_series.loc[ep_start:date]),
                })
                in_dd = False
                ep_start = None

    # Handle drawdown still open at end of series
    if in_dd:
        episodes.append({
            "start": ep_start,
            "trough": ep_trough_date,
            "recovery": None,
            "depth": ep_trough_val,
            "duration": len(dd_series.loc[ep_start:]),
        })

    # Sort by depth (worst first), take top 5
    episodes.sort(key=lambda e: e["depth"])
    return episodes[:5]


# ---------------------------------------------------------------------------
# Load data and run backtests
# ---------------------------------------------------------------------------

data_dir = Path('/Users/GualyCr/Desktop/AMAAM/amaam/.claude/worktrees/reverent-mayer/data/processed')
print("Loading data…")
data_dict = load_validated_data(data_dir)

cfg_a = ModelConfig()  # monthly baseline
cfg_b = ModelConfig()
cfg_b.rebalancing_frequency = "biweekly"

print("Running backtest A (monthly)…")
result_a = run_backtest(data_dict, cfg_a)

print("Running backtest B (biweekly)…")
result_b = run_backtest(data_dict, cfg_b)

ra = result_a.monthly_returns
rb = result_b.monthly_returns

print(f"  A: {len(ra)} return periods  ({ra.index.min().date()} → {ra.index.max().date()})")
print(f"  B: {len(rb)} return periods  ({rb.index.min().date()} → {rb.index.max().date()})")

# ---------------------------------------------------------------------------
# IS / OOS boundaries
# ---------------------------------------------------------------------------
IS_START  = "2004-01-01"
IS_END    = "2017-12-31"
OOS_START = "2018-01-01"
OOS_END   = "2026-04-10"

ra_is  = ra.loc[IS_START:IS_END]
ra_oos = ra.loc[OOS_START:OOS_END]
rb_is  = rb.loc[IS_START:IS_END]
rb_oos = rb.loc[OOS_START:OOS_END]

# ---------------------------------------------------------------------------
# Print report
# ---------------------------------------------------------------------------

print()
print("=" * 70)
print("=== BIWEEKLY vs MONTHLY REBALANCING ===")
print("=" * 70)

# ── Full period ──────────────────────────────────────────────────────────────
ar_a = ann_return(ra); ar_b = ann_return(rb)
av_a = ann_vol(ra);    av_b = ann_vol(rb)
sh_a = sharpe(ra);     sh_b = sharpe(rb)
so_a = sortino(ra);    so_b = sortino(rb)
md_a = max_dd(ra);     md_b = max_dd(rb)
ca_a = calmar(ra);     ca_b = calmar(rb)
pp_a = float((ra > 0).mean()); pp_b = float((rb > 0).mean())
bm_a = float(ra.max()); bm_b = float(rb.max())
wm_a = float(ra.min()); wm_b = float(rb.min())

print()
print("FULL PERIOD:")
print(f"{'Metric':<28}{'A Monthly':>14}{'B Biweekly':>14}{'Delta':>12}")
print("-" * 70)
print(f"{'Ann. Return':<28}{pct(ar_a):>14}{pct(ar_b):>14}{signed_pct(ar_b - ar_a):>12}")
print(f"{'Ann. Volatility':<28}{pct(av_a):>14}{pct(av_b):>14}{signed_pct(av_b - av_a):>12}")
print(f"{'Sharpe':<28}{sh_a:>14.3f}{sh_b:>14.3f}{sh_b - sh_a:>+12.3f}")
print(f"{'Sortino':<28}{so_a:>14.3f}{so_b:>14.3f}{so_b - so_a:>+12.3f}")
print(f"{'Max Drawdown':<28}{pct(md_a):>14}{pct(md_b):>14}{signed_pct(md_b - md_a):>12}")
print(f"{'Calmar':<28}{ca_a:>14.3f}{ca_b:>14.3f}{ca_b - ca_a:>+12.3f}")
print(f"{'% Months Positive':<28}{pct(pp_a, 1):>14}{pct(pp_b, 1):>14}{signed_pct(pp_b - pp_a, 1):>12}")
print(f"{'Best Month':<28}{pct(bm_a):>14}{pct(bm_b):>14}{signed_pct(bm_b - bm_a):>12}")
print(f"{'Worst Month':<28}{pct(wm_a):>14}{pct(wm_b):>14}{signed_pct(wm_b - wm_a):>12}")

# ── IS / OOS ──────────────────────────────────────────────────────────────────
sh_a_is  = sharpe(ra_is);  sh_b_is  = sharpe(rb_is)
sh_a_oos = sharpe(ra_oos); sh_b_oos = sharpe(rb_oos)
ar_a_is  = ann_return(ra_is);  ar_b_is  = ann_return(rb_is)
ar_a_oos = ann_return(ra_oos); ar_b_oos = ann_return(rb_oos)
md_a_is  = max_dd(ra_is);  md_b_is  = max_dd(rb_is)
md_a_oos = max_dd(ra_oos); md_b_oos = max_dd(rb_oos)

# IS→OOS degradation = (OOS Sharpe - IS Sharpe) / |IS Sharpe| * 100%
deg_a = (sh_a_oos - sh_a_is) / abs(sh_a_is) * 100 if sh_a_is != 0 else float("nan")
deg_b = (sh_b_oos - sh_b_is) / abs(sh_b_is) * 100 if sh_b_is != 0 else float("nan")

print()
print(f"IS (2004–2017):")
print(f"{'Metric':<28}{'A Monthly':>14}{'B Biweekly':>14}{'ΔSharpe':>12}")
print("-" * 70)
print(f"{'Sharpe':<28}{sh_a_is:>14.3f}{sh_b_is:>14.3f}{sh_b_is - sh_a_is:>+12.3f}")
print(f"{'Ann. Return':<28}{pct(ar_a_is):>14}{pct(ar_b_is):>14}{signed_pct(ar_b_is - ar_a_is):>12}")
print(f"{'MDD':<28}{pct(md_a_is):>14}{pct(md_b_is):>14}{signed_pct(md_b_is - md_a_is):>12}")

print()
print(f"OOS (2018–2026):")
print(f"{'Metric':<28}{'A Monthly':>14}{'B Biweekly':>14}{'ΔSharpe':>12}")
print("-" * 70)
print(f"{'Sharpe':<28}{sh_a_oos:>14.3f}{sh_b_oos:>14.3f}{sh_b_oos - sh_a_oos:>+12.3f}")
print(f"{'Ann. Return':<28}{pct(ar_a_oos):>14}{pct(ar_b_oos):>14}{signed_pct(ar_b_oos - ar_a_oos):>12}")
print(f"{'MDD':<28}{pct(md_a_oos):>14}{pct(md_b_oos):>14}{signed_pct(md_b_oos - md_a_oos):>12}")

print()
print(f"IS→OOS degradation:    A: {deg_a:+.1f}%    B: {deg_b:+.1f}%")

# ── Annual returns ────────────────────────────────────────────────────────────
print()
print("ANNUAL RETURNS:")
print(f"{'Year':<8}{'A Monthly':>14}{'B Biweekly':>14}{'Delta':>12}")
print("-" * 50)

# Build calendar-year return series for both
def annual_rets(rets: pd.Series) -> pd.Series:
    return rets.groupby(rets.index.to_period("Y")).apply(
        lambda r: float((1.0 + r).prod() - 1.0)
    )

ann_a = annual_rets(ra)
ann_b = annual_rets(rb)

all_years = sorted(set(ann_a.index.tolist()) | set(ann_b.index.tolist()))
# Filter to 2005–2026
for yr in all_years:
    if int(str(yr)) < 2005:
        continue
    va = ann_a.get(yr, float("nan"))
    vb = ann_b.get(yr, float("nan"))
    delta = vb - va if not (math.isnan(va) or math.isnan(vb)) else float("nan")
    da = pct(va) if not math.isnan(va) else "N/A"
    db = pct(vb) if not math.isnan(vb) else "N/A"
    dd_str = signed_pct(delta) if not math.isnan(delta) else "N/A"
    print(f"{str(yr):<8}{da:>14}{db:>14}{dd_str:>12}")

# ── Stress periods ────────────────────────────────────────────────────────────
stress_periods = [
    ("GFC",               "2007-10-01", "2009-03-31"),
    ("Euro Crisis",       "2010-04-01", "2011-09-30"),
    ("Taper Tantrum",     "2013-05-01", "2013-06-30"),
    ("China Selloff",     "2015-08-01", "2016-02-29"),
    ("COVID",             "2020-02-01", "2020-03-31"),
    ("2022 Rate Shock",   "2022-01-01", "2022-12-31"),
    ("2025 Tariff Shock", "2025-02-01", "2026-04-10"),
]

print()
print("STRESS PERIODS:")
print(f"{'Period':<24}{'A Monthly':>14}{'B Biweekly':>14}{'Delta':>12}")
print("-" * 66)

for name, start, end in stress_periods:
    va = period_return(ra, start, end)
    vb = period_return(rb, start, end)
    delta = vb - va if not (math.isnan(va) or math.isnan(vb)) else float("nan")
    da = pct(va) if not math.isnan(va) else "N/A"
    db = pct(vb) if not math.isnan(vb) else "N/A"
    dd_str = signed_pct(delta) if not math.isnan(delta) else "N/A"
    print(f"{name:<24}{da:>14}{db:>14}{dd_str:>12}")

# ── Transaction cost analysis ─────────────────────────────────────────────────
# Turnover is indexed on signal dates; we need to align to a per-month cadence
# for the monthly-cost drag computation.
# Monthly cost drag = avg per-rebalance turnover * transaction_cost * rebalances_per_month
# For monthly: ~1 rebalance/month; for biweekly: ~25/12 ≈ 2.08 rebalances/month.
tc = cfg_a.transaction_cost  # 0.0005 (same for both configs)

to_a = result_a.turnover
to_b = result_b.turnover

# Average turnover per rebalancing event (sum of |Δw| / 2 is already the
# one-way half-turnover convention — but the engine stores the full round-trip
# sum.  Cost formula: cost = turnover * tc / 2 (one-way).  So the avg monthly
# cost drag = mean_rebalance_turnover * tc / 2 * rebalances_per_month.
avg_to_a_event = float(to_a.mean()) if len(to_a) > 0 else 0.0  # per event
avg_to_b_event = float(to_b.mean()) if len(to_b) > 0 else 0.0

# Annualised rebalancing events
years_a = (ra.index.max() - ra.index.min()).days / 365.25 if len(ra) > 1 else 1.0
years_b = (rb.index.max() - rb.index.min()).days / 365.25 if len(rb) > 1 else 1.0
events_per_year_a = len(to_a) / years_a if years_a > 0 else 12.0
events_per_year_b = len(to_b) / years_b if years_b > 0 else 25.0

# Monthly equivalent turnover = (total_turnover_per_year / 12)
monthly_to_a = avg_to_a_event * events_per_year_a / 12.0
monthly_to_b = avg_to_b_event * events_per_year_b / 12.0

# Annual cost drag = avg annual turnover * tc / 2 (one-way cost)
annual_cost_a = avg_to_a_event * events_per_year_a * tc / 2.0
annual_cost_b = avg_to_b_event * events_per_year_b * tc / 2.0

print()
print("TRANSACTION COST ANALYSIS:")
print(f"{'Config':<18}{'Est. Monthly Turnover':>24}{'Annual Cost Drag':>18}")
print("-" * 62)
print(f"{'A Monthly':<18}{pct(monthly_to_a, 1):>24}{pct(annual_cost_a, 3):>18}")
print(f"{'B Biweekly':<18}{pct(monthly_to_b, 1):>24}{pct(annual_cost_b, 3):>18}")
extra = annual_cost_b - annual_cost_a
print(f"Additional drag from biweekly: {pct(extra, 3)} per year")

# ── Top 5 drawdowns for B ─────────────────────────────────────────────────────
print()
print("TOP 5 DRAWDOWNS (B Biweekly):")
print(f"{'#':<4}{'Start':<14}{'Trough':<14}{'Recovery':<16}{'Depth':>10}{'Duration':>12}")
print("-" * 72)

episodes = top5_drawdowns(rb)
for i, ep in enumerate(episodes, 1):
    start_str    = ep["start"].strftime("%Y-%m") if ep["start"] else "—"
    trough_str   = ep["trough"].strftime("%Y-%m") if ep["trough"] else "—"
    recovery_str = ep["recovery"].strftime("%Y-%m") if ep["recovery"] else "ongoing"
    depth_str    = pct(ep["depth"])
    dur_str      = str(ep["duration"]) + " periods"
    print(f"{i:<4}{start_str:<14}{trough_str:<14}{recovery_str:<16}{depth_str:>10}{dur_str:>12}")

print()
print("=" * 70)
