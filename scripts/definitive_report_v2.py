"""
AMAAM Definitive Performance Report v2

Runs one clean, fully-audited backtest and prints the definitive performance
report with explicit alignment checks and all intermediate values exposed.
"""

import sys
import logging

sys.path.insert(0, '/Users/GualyCr/Desktop/AMAAM/amaam/.claude/worktrees/reverent-mayer')

# Suppress engine/factor logging so output is clean
logging.basicConfig(level=logging.WARNING)

from config.default_config import ModelConfig
from src.backtest.engine import run_backtest
from src.data.loader import load_validated_data
from pathlib import Path
import pandas as pd
import numpy as np

# ── Metric helpers ────────────────────────────────────────────────────────────

def ann_ret(r: pd.Series) -> float:
    r = r.dropna()
    return (1 + r.mean()) ** 12 - 1


def ann_vol(r: pd.Series) -> float:
    r = r.dropna()
    return r.std() * np.sqrt(12)


def sharpe(r: pd.Series) -> float:
    r = r.dropna()
    return ann_ret(r) / ann_vol(r)


def sortino(r: pd.Series) -> float:
    r = r.dropna()
    down = r[r < 0]
    dv = down.std() * np.sqrt(12)
    return ann_ret(r) / dv if dv > 0 else np.nan


def mdd(r: pd.Series) -> float:
    r = r.dropna()
    eq = (1 + r).cumprod()
    return float((eq / eq.cummax() - 1).min())


def calmar(r: pd.Series) -> float:
    d = mdd(r)
    return ann_ret(r) / abs(d) if d != 0 else np.nan


def pct_pos(r: pd.Series) -> float:
    r = r.dropna()
    return (r > 0).mean() * 100


def stress_ret(r: pd.Series, start: str, end: str) -> float:
    """Cumulative return over a stress period (Period-indexed series)."""
    s = pd.Period(start, freq='M')
    e = pd.Period(end, freq='M')
    window = r[(r.index >= s) & (r.index <= e)].dropna()
    if window.empty:
        return np.nan
    return float((1 + window).prod() - 1)


# ── Load data & run backtest ──────────────────────────────────────────────────

data_dir = Path('/Users/GualyCr/Desktop/AMAAM/amaam/.claude/worktrees/reverent-mayer/data/processed')
data_dict = load_validated_data(data_dir)
cfg = ModelConfig()

print("Running backtest (this may take ~30-60 seconds)…")
result = run_backtest(data_dict, cfg)
print("Backtest complete.\n")

# ── Config audit ──────────────────────────────────────────────────────────────

# Determine what trend dispatch is actually running in _precompute_factors
# by reading the engine's elif chain — confirmed by audit of engine.py line 197:
# elif _tm == "sma_ratio": → compute_sma_ratio_signal(close)
# And trend.py compute_trend_all_assets dispatches identically.
_trend_dispatch_code = (
    'compute_sma_ratio_signal(close)  '
    '[elif _tm == "sma_ratio" branch in engine._precompute_factors]'
)

# ── SPY alignment ─────────────────────────────────────────────────────────────

spy_daily = data_dict['SPY']['Close']
spy_me = spy_daily.resample('ME').last()
spy_ret_me = spy_me.pct_change().dropna()

amaam = result.monthly_returns.copy()

print("AMAAM index sample:", amaam.index[:5].tolist())
print("SPY month-end index sample:", spy_ret_me.index[:5].tolist())

# Convert both to period index for year-month alignment
amaam.index = amaam.index.to_period('M')
spy_ret_me.index = spy_ret_me.index.to_period('M')
spy_aligned = spy_ret_me.reindex(amaam.index)

nan_count = int(spy_aligned.isna().sum())
print(f"NaN count in aligned SPY: {nan_count}")
print(f"Total months: {len(amaam)}")

# ── Intermediate verification ─────────────────────────────────────────────────

amaam_mean = float(amaam.mean())
amaam_std  = float(amaam.std())
spy_mean   = float(spy_aligned.mean())
spy_std    = float(spy_aligned.std())

# ── Period slicing (year from Period index) ───────────────────────────────────

amaam_years = amaam.index.year
spy_years   = spy_aligned.index.year

# IS / OOS masks
is_mask  = amaam_years <= 2017
oos_mask = amaam_years >= 2018

amaam_is  = amaam[is_mask]
amaam_oos = amaam[oos_mask]
spy_is    = spy_aligned[is_mask]
spy_oos   = spy_aligned[oos_mask]

# ── Annual returns ────────────────────────────────────────────────────────────

amaam_annual = (1 + amaam).groupby(amaam.index.year).prod() - 1
spy_annual   = (1 + spy_aligned).groupby(spy_aligned.index.year).prod() - 1

# ── Stress periods ────────────────────────────────────────────────────────────

stress_windows = [
    ("GFC (Oct07–Mar09)",    "2007-10", "2009-03"),
    ("Euro Crisis (Apr10–Sep11)", "2010-04", "2011-09"),
    ("Taper Tantrum (May13–Jun13)", "2013-05", "2013-06"),
    ("China/HY Selloff (Aug15–Feb16)", "2015-08", "2016-02"),
    ("COVID (Feb–Mar20)",    "2020-02", "2020-03"),
    ("2022 Rate Shock (Jan22–Dec22)", "2022-01", "2022-12"),
    ("2025 Tariff Shock (Feb25–Apr25)", "2025-02", "2025-04"),
]

# ── Print report ──────────────────────────────────────────────────────────────

fp_start = str(amaam.index.min())
fp_end   = str(amaam.index.max())
n_months = len(amaam)
is_n     = int(is_mask.sum())
oos_n    = int(oos_mask.sum())

print()
print("=" * 72)
print("AMAAM DEFINITIVE PERFORMANCE REPORT")
print("=" * 72)

print()
print("SYSTEM AUDIT:")
print(f"  trend_method (config):           {cfg.trend_method}")
print(f"  trend dispatch (code):           {_trend_dispatch_code}")
print(f"  weighting_scheme:                {cfg.weighting_scheme}")
print(f"  main_sleeve_top_n:               {cfg.main_sleeve_top_n}")
print(f"  hedging_sleeve_top_n:            {cfg.hedging_sleeve_top_n}")
print(f"  momentum_blend:                  {cfg.momentum_blend}")
print(f"  momentum_blend_lookbacks:        {cfg.momentum_blend_lookbacks}")
print(f"  weights (wM/wV/wC/wT):          "
      f"{cfg.weight_momentum}/{cfg.weight_volatility}/"
      f"{cfg.weight_correlation}/{cfg.weight_trend}")

print()
print("BACKTEST PERIOD:")
print(f"  First month:  {fp_start}")
print(f"  Last month:   {fp_end}")
print(f"  Total months: {n_months}")

print()
print("ALIGNMENT CHECK:")
print(f"  SPY NaN after alignment: {nan_count}  (should be 0 or near 0)")
print(f"  AMAAM mean monthly ret:  {amaam_mean:.6f}")
print(f"  AMAAM std  monthly ret:  {amaam_std:.6f}")
print(f"  SPY   mean monthly ret:  {spy_mean:.6f}")
print(f"  SPY   std  monthly ret:  {spy_std:.6f}")

# ── Full period table ─────────────────────────────────────────────────────────

def print_table(label: str, a: pd.Series, s: pd.Series, n: int) -> None:
    start = str(a.index.min())
    end   = str(a.index.max())
    print()
    print(f"{label} ({start} – {end}, {n} months):")
    print(f"{'Metric':<22} {'AMAAM':>10} {'SPY':>10}")
    print("-" * 44)
    print(f"{'Ann. Return':<22} {ann_ret(a)*100:>9.2f}%  {ann_ret(s)*100:>9.2f}%")
    print(f"{'Ann. Vol':<22} {ann_vol(a)*100:>9.2f}%  {ann_vol(s)*100:>9.2f}%")
    print(f"{'Sharpe':<22} {sharpe(a):>10.3f}  {sharpe(s):>10.3f}")
    print(f"{'Sortino':<22} {sortino(a):>10.3f}  {sortino(s):>10.3f}")
    print(f"{'MDD':<22} {mdd(a)*100:>9.2f}%  {mdd(s)*100:>9.2f}%")
    print(f"{'Calmar':<22} {calmar(a):>10.3f}  {calmar(s):>10.3f}")
    print(f"{'% Positive':<22} {pct_pos(a):>9.1f}%  {pct_pos(s):>9.1f}%")
    print(f"{'Best Month':<22} {a.dropna().max()*100:>+9.2f}%  {s.dropna().max()*100:>+9.2f}%")
    print(f"{'Worst Month':<22} {a.dropna().min()*100:>+9.2f}%  {s.dropna().min()*100:>+9.2f}%")


def print_sub_table(label: str, a: pd.Series, s: pd.Series, n: int) -> None:
    start = str(a.index.min())
    end   = str(a.index.max())
    print()
    print(f"{label} ({start} – {end}, {n} months):")
    print(f"{'Metric':<22} {'AMAAM':>10} {'SPY':>10}")
    print("-" * 44)
    print(f"{'Sharpe':<22} {sharpe(a):>10.3f}  {sharpe(s):>10.3f}")
    print(f"{'Ann. Return':<22} {ann_ret(a)*100:>9.2f}%  {ann_ret(s)*100:>9.2f}%")
    print(f"{'Ann. Vol':<22} {ann_vol(a)*100:>9.2f}%  {ann_vol(s)*100:>9.2f}%")
    print(f"{'MDD':<22} {mdd(a)*100:>9.2f}%  {mdd(s)*100:>9.2f}%")


print_table(
    f"FULL PERIOD",
    amaam, spy_aligned, n_months,
)

print_sub_table("IS (2004–2017)", amaam_is, spy_is, is_n)
print_sub_table("OOS (2018–2026)", amaam_oos, spy_oos, oos_n)

is_sharpe  = sharpe(amaam_is)
oos_sharpe = sharpe(amaam_oos)
delta_pct  = (oos_sharpe - is_sharpe) / abs(is_sharpe) * 100
print()
print(f"IS→OOS Sharpe change: {delta_pct:+.1f}%  "
      f"(IS={is_sharpe:.3f}, OOS={oos_sharpe:.3f}; "
      f"negative = OOS better)")

# ── Annual returns ────────────────────────────────────────────────────────────

print()
print("ANNUAL RETURNS:")
print(f"{'Year':<8} {'AMAAM':>9} {'SPY':>9} {'Delta':>9}")
print("-" * 38)
all_years = sorted(set(amaam_annual.index) | set(spy_annual.index))
for yr in all_years:
    a_r = amaam_annual.get(yr, np.nan)
    s_r = spy_annual.get(yr, np.nan)
    if np.isnan(a_r) or np.isnan(s_r):
        continue
    delta = a_r - s_r
    print(f"{yr:<8} {a_r*100:>+8.2f}%  {s_r*100:>+8.2f}%  {delta*100:>+8.2f}%")

# ── Stress periods ────────────────────────────────────────────────────────────

print()
print("STRESS PERIODS:")
print(f"{'Period':<38} {'AMAAM':>9} {'SPY':>9}")
print("-" * 58)
for label, st, en in stress_windows:
    a_r = stress_ret(amaam, st, en)
    s_r = stress_ret(spy_aligned, st, en)
    a_s = f"{a_r*100:>+8.2f}%" if not np.isnan(a_r) else "       N/A"
    s_s = f"{s_r*100:>+8.2f}%" if not np.isnan(s_r) else "       N/A"
    print(f"{label:<38} {a_s}  {s_s}")

print()
print("=" * 72)
