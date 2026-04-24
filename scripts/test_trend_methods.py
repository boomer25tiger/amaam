"""
Trend method comparison: Keltner vs SMA200 vs SMA_ratio.

Runs three full backtests (identical config except trend_method) and prints
a side-by-side performance report covering full period, IS/OOS split, annual
returns, and stress-period drawdowns.

  A  Keltner   — what the model has physically been computing (hardcoded default)
  B  SMA200    — Faber (2007) simple 200-day SMA
  C  SMA_ratio — Close/SMA(200) ±1% buffer (what config.trend_method claimed)
"""

import sys
sys.path.insert(0, '/Users/GualyCr/Desktop/AMAAM/amaam/.claude/worktrees/reverent-mayer')

from pathlib import Path

import numpy as np
import pandas as pd

from config.default_config import ModelConfig
from src.backtest.engine import run_backtest
from src.data.loader import load_validated_data

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

DATA_DIR = Path('/Users/GualyCr/Desktop/AMAAM/amaam/.claude/worktrees/reverent-mayer/data/processed')
data_dict = load_validated_data(DATA_DIR)

# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------

cfg_a = ModelConfig()
cfg_a.trend_method = "keltner"

cfg_b = ModelConfig()
cfg_b.trend_method = "sma200"

cfg_c = ModelConfig()
cfg_c.trend_method = "sma_ratio"

# ---------------------------------------------------------------------------
# Run backtests
# ---------------------------------------------------------------------------

print("Running backtest A: Keltner ...", flush=True)
res_a = run_backtest(data_dict, cfg_a)

print("Running backtest B: SMA200  ...", flush=True)
res_b = run_backtest(data_dict, cfg_b)

print("Running backtest C: SMA_ratio ...", flush=True)
res_c = run_backtest(data_dict, cfg_c)

# ---------------------------------------------------------------------------
# SPY benchmark
# ---------------------------------------------------------------------------

spy_close = data_dict["SPY"]["Close"]
spy_monthly = spy_close.resample("ME").last().pct_change().dropna()

# Align SPY to model date range
model_start = res_a.monthly_returns.index.min()
model_end   = res_a.monthly_returns.index.max()
spy = spy_monthly.loc[
    (spy_monthly.index >= model_start) & (spy_monthly.index <= model_end)
]

# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def ann_ret(r: pd.Series) -> float:
    """Annualised geometric return from monthly series."""
    if len(r) == 0:
        return float("nan")
    return (1.0 + r).prod() ** (12.0 / len(r)) - 1.0


def ann_vol(r: pd.Series) -> float:
    """Annualised volatility from monthly series."""
    return r.std() * np.sqrt(12.0)


def sharpe(r: pd.Series) -> float:
    v = ann_vol(r)
    return ann_ret(r) / v if v > 0 else float("nan")


def mdd(r: pd.Series) -> float:
    """Maximum drawdown (negative number)."""
    eq = (1.0 + r).cumprod()
    return (eq / eq.cummax() - 1.0).min()


def sortino(r: pd.Series) -> float:
    down = r[r < 0]
    if len(down) < 2:
        return float("nan")
    dv = down.std() * np.sqrt(12.0)
    return ann_ret(r) / dv if dv > 0 else float("nan")


def calmar(r: pd.Series) -> float:
    d = abs(mdd(r))
    return ann_ret(r) / d if d > 0 else float("nan")


def pct(x: float, decimals: int = 2) -> str:
    return f"{x * 100:.{decimals}f}%"


def fmt3(x: float) -> str:
    return f"{x:.3f}"


# ---------------------------------------------------------------------------
# Slice helpers
# ---------------------------------------------------------------------------

IS_START  = "2004-01-01"
IS_END    = "2017-12-31"
OOS_START = "2018-01-01"

def slice_period(r: pd.Series, start: str, end: str | None = None) -> pd.Series:
    if end:
        return r.loc[(r.index >= start) & (r.index <= end)]
    return r.loc[r.index >= start]


# Monthly return series for each config
ra = res_a.monthly_returns
rb = res_b.monthly_returns
rc = res_c.monthly_returns

# ---------------------------------------------------------------------------
# Section 1: Full period summary
# ---------------------------------------------------------------------------

print()
print("=" * 72)
print("  TREND METHOD COMPARISON: KELTNER vs SMA200 vs SMA_RATIO")
print("=" * 72)

header = f"{'Metric':<18} {'A Keltner':>12} {'B SMA200':>12} {'C SMA_Ratio':>12} {'SPY':>12}"
sep    = "-" * len(header)

print()
print("FULL PERIOD:")
print(header)
print(sep)

rows = [
    ("Ann. Return",  pct(ann_ret(ra)),  pct(ann_ret(rb)),  pct(ann_ret(rc)),  pct(ann_ret(spy))),
    ("Ann. Vol",     pct(ann_vol(ra)),  pct(ann_vol(rb)),  pct(ann_vol(rc)),  pct(ann_vol(spy))),
    ("Sharpe",       fmt3(sharpe(ra)),  fmt3(sharpe(rb)),  fmt3(sharpe(rc)),  fmt3(sharpe(spy))),
    ("Sortino",      fmt3(sortino(ra)), fmt3(sortino(rb)), fmt3(sortino(rc)), fmt3(sortino(spy))),
    ("MDD",          pct(mdd(ra)),      pct(mdd(rb)),      pct(mdd(rc)),      pct(mdd(spy))),
    ("Calmar",       fmt3(calmar(ra)),  fmt3(calmar(rb)),  fmt3(calmar(rc)),  fmt3(calmar(spy))),
]

for label, va, vb, vc, vs in rows:
    print(f"{label:<18} {va:>12} {vb:>12} {vc:>12} {vs:>12}")

# ---------------------------------------------------------------------------
# Section 2: IS / OOS split
# ---------------------------------------------------------------------------

def is_oos_row(label: str, r: pd.Series, ref_sharpe_is: float | None, ref_sharpe_oos: float | None):
    r_is  = slice_period(r, IS_START, IS_END)
    r_oos = slice_period(r, OOS_START)
    sh_is  = sharpe(r_is)
    sh_oos = sharpe(r_oos)
    delta_is  = f"{sh_is  - ref_sharpe_is :.3f}" if ref_sharpe_is  is not None else "  —  "
    delta_oos = f"{sh_oos - ref_sharpe_oos:.3f}" if ref_sharpe_oos is not None else "  —  "
    return (label,
            fmt3(sh_is),  pct(ann_ret(r_is)),  pct(mdd(r_is)),  delta_is,
            fmt3(sh_oos), pct(ann_ret(r_oos)), pct(mdd(r_oos)), delta_oos)

sh_a_is  = sharpe(slice_period(ra, IS_START, IS_END))
sh_a_oos = sharpe(slice_period(ra, OOS_START))

rows_is_oos = [
    is_oos_row("A Keltner",   ra, None,      None),
    is_oos_row("B SMA200",    rb, sh_a_is,   sh_a_oos),
    is_oos_row("C SMA_Ratio", rc, sh_a_is,   sh_a_oos),
]

is_hdr  = f"{'Config':<14} {'Sharpe':>8} {'Ann.Ret':>9} {'MDD':>9} {'ΔSharpe':>9}"
oos_hdr = f"{'Config':<14} {'Sharpe':>8} {'Ann.Ret':>9} {'MDD':>9} {'ΔSharpe':>9}"

print()
print(f"IS (2004–2017):")
print(is_hdr)
print("-" * len(is_hdr))
for row in rows_is_oos:
    label, sh_is, ret_is, mdd_is, d_is, sh_oos, ret_oos, mdd_oos, d_oos = row
    print(f"{label:<14} {sh_is:>8} {ret_is:>9} {mdd_is:>9} {d_is:>9}")

print()
print(f"OOS (2018–2026):")
print(oos_hdr)
print("-" * len(oos_hdr))
for row in rows_is_oos:
    label, sh_is, ret_is, mdd_is, d_is, sh_oos, ret_oos, mdd_oos, d_oos = row
    print(f"{label:<14} {sh_oos:>8} {ret_oos:>9} {mdd_oos:>9} {d_oos:>9}")

# ---------------------------------------------------------------------------
# Section 3: Annual returns
# ---------------------------------------------------------------------------

def annual_returns(r: pd.Series) -> pd.Series:
    """Compound monthly returns into calendar-year returns."""
    return r.groupby(r.index.year).apply(lambda x: (1.0 + x).prod() - 1.0)


ann_a   = annual_returns(ra)
ann_b   = annual_returns(rb)
ann_c   = annual_returns(rc)
ann_spy = annual_returns(spy)

all_years = sorted(set(ann_a.index) | set(ann_b.index) | set(ann_c.index))

print()
print("ANNUAL RETURNS:")
ar_hdr = f"{'Year':>6} {'A Keltner':>12} {'B SMA200':>12} {'C SMA_Ratio':>12} {'SPY':>10}"
print(ar_hdr)
print("-" * len(ar_hdr))

for yr in all_years:
    va   = pct(ann_a.get(yr,   float("nan")))
    vb   = pct(ann_b.get(yr,   float("nan")))
    vc   = pct(ann_c.get(yr,   float("nan")))
    vs   = pct(ann_spy.get(yr, float("nan")))
    print(f"{yr:>6} {va:>12} {vb:>12} {vc:>12} {vs:>10}")

# ---------------------------------------------------------------------------
# Section 4: Stress periods
# ---------------------------------------------------------------------------

STRESS_PERIODS = [
    ("GFC",              "2007-10-01", "2009-03-31"),
    ("Euro Crisis",      "2010-04-01", "2011-09-30"),
    ("China/HY",         "2015-08-01", "2016-02-29"),
    ("COVID",            "2020-02-01", "2020-03-31"),
    ("2022 Rate Shock",  "2022-01-01", "2022-12-31"),
    ("2025 Tariff Shock","2025-02-01", "2026-04-30"),
]

def stress_return(r: pd.Series, start: str, end: str) -> float:
    window = slice_period(r, start, end)
    if len(window) == 0:
        return float("nan")
    return (1.0 + window).prod() - 1.0


print()
print("STRESS PERIODS:")
st_hdr = f"{'Period':<22} {'A Keltner':>12} {'B SMA200':>12} {'C SMA_Ratio':>12} {'SPY':>10}"
print(st_hdr)
print("-" * len(st_hdr))

for name, start, end in STRESS_PERIODS:
    va = pct(stress_return(ra,  start, end))
    vb = pct(stress_return(rb,  start, end))
    vc = pct(stress_return(rc,  start, end))
    vs = pct(stress_return(spy, start, end))
    print(f"{name:<22} {va:>12} {vb:>12} {vc:>12} {vs:>10}")

print()
print("=" * 72)
print("Done.")
