"""
BTAL addition test: compare configs A (baseline), B (+BTAL top-2), C (+BTAL top-3).

Aligned window aligned to BTAL inception (Sep 2011) plus warm-up:
  IS:  2012-06-01 → 2018-01-01
  OOS: 2018-01-01 → 2026-04-23
"""

import sys
import math

sys.path.insert(0, '/Users/GualyCr/Desktop/AMAAM/amaam/.claude/worktrees/reverent-mayer')

import numpy as np
import pandas as pd

from config.default_config import ModelConfig
import config.etf_universe as universe_mod
import src.backtest.engine as engine_mod
from src.backtest.engine import run_backtest
from src.data.loader import load_validated_data
from pathlib import Path

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
data_dir = Path('/Users/GualyCr/Desktop/AMAAM/amaam/.claude/worktrees/reverent-mayer/data/processed')
data_dict = load_validated_data(data_dir)

# Load BTAL
btal = pd.read_csv(data_dir / 'BTAL.csv', index_col=0, parse_dates=True)
if isinstance(btal.columns, pd.MultiIndex):
    btal.columns = btal.columns.get_level_values(0)
data_dict['BTAL'] = btal

# ---------------------------------------------------------------------------
# Universe definitions
# ---------------------------------------------------------------------------
BASE_HEDGE = ['GLD', 'TLT', 'IEF', 'SH', 'UUP', 'SHY']
BTAL_HEDGE = ['GLD', 'TLT', 'IEF', 'SH', 'UUP', 'SHY', 'BTAL']

IS_START  = '2012-06-01'
IS_END    = '2018-01-01'
OOS_START = '2018-01-01'
OOS_END   = '2026-04-23'

# ---------------------------------------------------------------------------
# Helper: metrics over a return slice
# ---------------------------------------------------------------------------

def compute_metrics(returns: pd.Series) -> dict:
    """Compute annualised performance metrics from monthly returns."""
    r = returns.dropna()
    if len(r) == 0:
        return dict(ann_ret=np.nan, ann_vol=np.nan, sharpe=np.nan, mdd=np.nan)
    ann_ret = (1 + r.mean()) ** 12 - 1
    ann_vol = r.std() * math.sqrt(12)
    sharpe  = ann_ret / ann_vol if ann_vol != 0 else np.nan
    equity  = (1 + r).cumprod()
    roll_max = equity.cummax()
    dd = equity / roll_max - 1
    mdd = dd.min()
    return dict(ann_ret=ann_ret, ann_vol=ann_vol, sharpe=sharpe, mdd=mdd)


def period_return(returns: pd.Series, start: str, end: str) -> float:
    """Cumulative return over a date range."""
    r = returns.loc[start:end].dropna()
    if len(r) == 0:
        return np.nan
    return (1 + r).prod() - 1


# ---------------------------------------------------------------------------
# Run backtest for a given config, returning BacktestResult
# ---------------------------------------------------------------------------

def run_config(hedge_tickers: list, top_n: int) -> object:
    """Monkey-patch the hedging sleeve, run backtest, restore."""
    orig_universe = universe_mod.HEDGING_SLEEVE_TICKERS
    orig_engine   = engine_mod.HEDGING_SLEEVE_TICKERS
    try:
        universe_mod.HEDGING_SLEEVE_TICKERS = hedge_tickers
        engine_mod.HEDGING_SLEEVE_TICKERS   = hedge_tickers

        cfg = ModelConfig(
            backtest_start=IS_START,
            backtest_end=OOS_END,
            holdout_start=OOS_START,
            hedging_sleeve_top_n=top_n,
        )
        result = run_backtest(data_dict, cfg)
    finally:
        universe_mod.HEDGING_SLEEVE_TICKERS = orig_universe
        engine_mod.HEDGING_SLEEVE_TICKERS   = orig_engine
    return result


# ---------------------------------------------------------------------------
# Run all three configs
# ---------------------------------------------------------------------------
print("Running Config A (baseline, 6 assets, top-2)…")
res_a = run_config(BASE_HEDGE, top_n=2)

print("Running Config B (+BTAL, 7 assets, top-2)…")
res_b = run_config(BTAL_HEDGE, top_n=2)

print("Running Config C (+BTAL, 7 assets, top-3)…")
res_c = run_config(BTAL_HEDGE, top_n=3)

# ---------------------------------------------------------------------------
# Slice IS / OOS returns
# ---------------------------------------------------------------------------

def slice_returns(result, start: str, end: str) -> pd.Series:
    r = result.monthly_returns
    return r.loc[(r.index >= start) & (r.index < end)]


r_a_is  = slice_returns(res_a, IS_START, IS_END)
r_a_oos = slice_returns(res_a, OOS_START, OOS_END)

r_b_is  = slice_returns(res_b, IS_START, IS_END)
r_b_oos = slice_returns(res_b, OOS_START, OOS_END)

r_c_is  = slice_returns(res_c, IS_START, IS_END)
r_c_oos = slice_returns(res_c, OOS_START, OOS_END)

m_a_is  = compute_metrics(r_a_is)
m_a_oos = compute_metrics(r_a_oos)
m_b_is  = compute_metrics(r_b_is)
m_b_oos = compute_metrics(r_b_oos)
m_c_is  = compute_metrics(r_c_is)
m_c_oos = compute_metrics(r_c_oos)

# ---------------------------------------------------------------------------
# BTAL selection frequency
# ---------------------------------------------------------------------------

def btal_selection_freq(result, start: str, end: str) -> tuple:
    """Return (selected_months, total_months) for BTAL in the hedge sleeve."""
    alloc = result.allocations
    if alloc.empty or 'BTAL' not in alloc.columns:
        return 0, 0
    mask = (alloc.index >= start) & (alloc.index < end)
    sub  = alloc.loc[mask, 'BTAL'].fillna(0)
    total    = len(sub)
    selected = int((sub > 0).sum())
    return selected, total


b_is_sel,  b_is_tot  = btal_selection_freq(res_b, IS_START, IS_END)
b_oos_sel, b_oos_tot = btal_selection_freq(res_b, OOS_START, OOS_END)
c_is_sel,  c_is_tot  = btal_selection_freq(res_c, IS_START, IS_END)
c_oos_sel, c_oos_tot = btal_selection_freq(res_c, OOS_START, OOS_END)

# ---------------------------------------------------------------------------
# Stress period returns
# ---------------------------------------------------------------------------
IS_STRESS_PERIODS = {
    'Eurozone 2012':     ('2012-01-01', '2012-07-01'),
    'Taper 2013':        ('2013-05-01', '2013-07-01'),
    'HY Stress 2015-16': ('2015-08-01', '2016-03-01'),
}
OOS_STRESS_PERIODS = {
    'COVID 2020':        ('2020-02-01', '2020-04-01'),
    '2022 Rate Shock':   ('2022-01-01', '2023-01-01'),
    '2025 Tariff Shock': ('2025-02-01', '2026-05-01'),
}

# ---------------------------------------------------------------------------
# Print results
# ---------------------------------------------------------------------------
print()
print("=== BTAL TEST: TOP-2 vs TOP-3, IS + OOS ===")
print(f"Aligned window: IS {IS_START[:7]}–{IS_END[:7]} | OOS {OOS_START[:7]}–{OOS_END[:7]}")
print()

# Summary table
print(f"{'Config':<12} {'Assets':>6} {'TopN':>4} {'SelRatio':>9}  {'IS Sharpe':>10}  {'OOS Sharpe':>10}  {'IS MDD':>9}  {'OOS MDD':>9}")
print("-" * 80)
for label, n_assets, top_n, sel_ratio, m_is, m_oos in [
    ("A Base",  6, 2, "33%", m_a_is, m_a_oos),
    ("B +BTAL", 7, 2, "29%", m_b_is, m_b_oos),
    ("C +BTAL", 7, 3, "43%", m_c_is, m_c_oos),
]:
    print(
        f"{label:<12} {n_assets:>6} {top_n:>4} {sel_ratio:>9}  "
        f"{m_is['sharpe']:>10.3f}  {m_oos['sharpe']:>10.3f}  "
        f"{m_is['mdd']*100:>8.2f}%  {m_oos['mdd']*100:>8.2f}%"
    )

# IS Detail
print()
print("IS Detail (2012–2018):")
print(f"{'Config':<10} {'Ann.Ret':>8} {'Ann.Vol':>8} {'Sharpe':>7} {'MDD':>9} {'ΔSharpe':>9}")
print("-" * 58)
for label, m in [("A", m_a_is), ("B", m_b_is), ("C", m_c_is)]:
    delta = ""
    if label == "A":
        delta_str = "—"
    else:
        ref = m_a_is['sharpe']
        diff = m['sharpe'] - ref
        delta_str = f"{diff:+.3f}"
    print(
        f"{label:<10} {m['ann_ret']*100:>7.2f}%  {m['ann_vol']*100:>7.2f}%  "
        f"{m['sharpe']:>6.3f}  {m['mdd']*100:>8.2f}%  {delta_str:>9}"
    )

# OOS Detail
print()
print("OOS Detail (2018–2026):")
print(f"{'Config':<10} {'Ann.Ret':>8} {'Ann.Vol':>8} {'Sharpe':>7} {'MDD':>9} {'ΔSharpe':>9}")
print("-" * 58)
for label, m in [("A", m_a_oos), ("B", m_b_oos), ("C", m_c_oos)]:
    if label == "A":
        delta_str = "—"
    else:
        diff = m['sharpe'] - m_a_oos['sharpe']
        delta_str = f"{diff:+.3f}"
    print(
        f"{label:<10} {m['ann_ret']*100:>7.2f}%  {m['ann_vol']*100:>7.2f}%  "
        f"{m['sharpe']:>6.3f}  {m['mdd']*100:>8.2f}%  {delta_str:>9}"
    )

# IS Stress
print()
print("IS Stress Periods:")
print(f"{'Period':<22} {'A(6/2)':>8} {'B(7/2)':>8} {'C(7/3)':>8}")
print("-" * 50)
for period, (s, e) in IS_STRESS_PERIODS.items():
    ra = period_return(r_a_is, s, e)
    rb = period_return(r_b_is, s, e)
    rc = period_return(r_c_is, s, e)
    def fmt(v):
        return f"{v*100:>7.2f}%" if not np.isnan(v) else "    N/A"
    print(f"{period:<22} {fmt(ra)} {fmt(rb)} {fmt(rc)}")

# OOS Stress
print()
print("OOS Stress Periods:")
print(f"{'Period':<22} {'A(6/2)':>8} {'B(7/2)':>8} {'C(7/3)':>8}")
print("-" * 50)
for period, (s, e) in OOS_STRESS_PERIODS.items():
    ra = period_return(r_a_oos, s, e)
    rb = period_return(r_b_oos, s, e)
    rc = period_return(r_c_oos, s, e)
    def fmt(v):
        return f"{v*100:>7.2f}%" if not np.isnan(v) else "    N/A"
    print(f"{period:<22} {fmt(ra)} {fmt(rb)} {fmt(rc)}")

# BTAL selection frequency
print()
print("BTAL Selection Frequency:")
print(f"{'Config':<10} {'IS months selected / total':<30} {'OOS months selected / total'}")
print("-" * 72)
for label, is_sel, is_tot, oos_sel, oos_tot in [
    ("B(7/2)", b_is_sel, b_is_tot, b_oos_sel, b_oos_tot),
    ("C(7/3)", c_is_sel, c_is_tot, c_oos_sel, c_oos_tot),
]:
    is_pct  = b_is_sel  / b_is_tot  * 100 if label == "B(7/2)" else c_is_sel  / c_is_tot  * 100
    oos_pct = b_oos_sel / b_oos_tot * 100 if label == "B(7/2)" else c_oos_sel / c_oos_tot * 100
    is_str  = f"{is_sel}/{is_tot} ({is_pct:.1f}%)"
    oos_str = f"{oos_sel}/{oos_tot} ({oos_pct:.1f}%)"
    print(f"{label:<10} {is_str:<30} {oos_str}")
