"""
Comprehensive IS-only comparison of all implemented trend methods in AMAAM.

Runs each of the 11 trend methods over the in-sample period 2004–2017,
prints a ranked summary table and stress-period breakdown.
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

data_dir = Path('/Users/GualyCr/Desktop/AMAAM/amaam/.claude/worktrees/reverent-mayer/data/processed')
data_dict = load_validated_data(data_dir)

IS_START = '2004-01-01'
IS_END   = '2017-12-31'

# ---------------------------------------------------------------------------
# Metric helpers (monthly return series expected)
# ---------------------------------------------------------------------------

def ann_ret(r: pd.Series) -> float:
    """Annualised geometric return from a monthly return series."""
    return (1 + r.mean()) ** 12 - 1


def ann_vol(r: pd.Series) -> float:
    """Annualised volatility from a monthly return series."""
    return r.std() * np.sqrt(12)


def sharpe(r: pd.Series) -> float:
    """Annualised Sharpe ratio (risk-free = 0)."""
    vol = ann_vol(r)
    return ann_ret(r) / vol if vol > 0 else np.nan


def sortino(r: pd.Series) -> float:
    """Annualised Sortino ratio (downside deviation denominator)."""
    down = r[r < 0]
    dv = down.std() * np.sqrt(12)
    return ann_ret(r) / dv if dv > 0 else np.nan


def mdd(r: pd.Series) -> float:
    """Maximum drawdown (negative number)."""
    eq = (1 + r).cumprod()
    return (eq / eq.cummax() - 1).min()


def calmar(r: pd.Series) -> float:
    """Calmar ratio = annualised return / abs(max drawdown)."""
    dd = abs(mdd(r))
    return ann_ret(r) / dd if dd > 0 else np.nan


def is_slice(result) -> pd.Series:
    """Extract IS monthly returns as a period-indexed Series."""
    mr = result.monthly_returns.copy()
    mr.index = mr.index.to_period('M')
    return mr[
        (mr.index >= pd.Period(IS_START, 'M')) &
        (mr.index <= pd.Period(IS_END, 'M'))
    ]


def stress_return(r: pd.Series, start: str, end: str) -> float:
    """Cumulative return over a stress window (period-indexed monthly returns)."""
    window = r[
        (r.index >= pd.Period(start, 'M')) &
        (r.index <= pd.Period(end, 'M'))
    ]
    if window.empty:
        return np.nan
    return (1 + window).prod() - 1


# ---------------------------------------------------------------------------
# Stress-period definitions (IS only — all fall within 2004–2017)
# ---------------------------------------------------------------------------

STRESS_PERIODS = {
    'GFC':       ('2007-10', '2009-03'),
    'Euro Cris': ('2010-04', '2011-09'),
    'Taper':     ('2013-05', '2013-06'),
    'China/HY':  ('2015-08', '2016-02'),
}

# ---------------------------------------------------------------------------
# Method descriptions
# ---------------------------------------------------------------------------

METHOD_DESC = {
    'keltner':       'Asymmetric Keltner Channel — EMA63 upper / EMA105 lower / ATR42; medium speed (~4 mo)',
    'sma200':        'Faber (2007) 200-day SMA; T=+2 if Close > SMA else -2; no buffer, slow (~10 mo)',
    'sma_ratio':     'Close/SMA200 with ±1% buffer + persistent carry; reduces whipsaws vs plain SMA200; slow',
    'dual_sma':      'Golden/death cross — SMA50 vs SMA200; T=+2 if fast > slow; slow (~10 mo)',
    'donchian':      '200-day Donchian channel breakout with carry; fires only on new extremes; slow',
    'tsmom':         '12-month return sign (Moskowitz et al. 2012); T=+2 if 252-day ROC > 0; slow',
    'rolling_sharpe':'126-day rolling Sharpe > 0 (Baz et al. 2015 / AQR); vol-normalised; medium speed',
    'r2_trend':      'OLS slope > 0 AND R²≥0.65 on 126-day log-price regression; suppresses choppy markets',
    'macd':          'MACD(12,26) zero-line cross — EMA12 > EMA26; fast (~1 mo); Appel (1979)',
    'paper_atr':     'Paper-literal ATR: LB = EMA(High,105) + ATR (not close − ATR); high bar for bearish signal',
    'ensemble':      'Equal-weight composite: MACD (fast) + Keltner (medium) + SMA_ratio (slow); diversified',
}

# ---------------------------------------------------------------------------
# Run backtests
# ---------------------------------------------------------------------------

METHODS = [
    'keltner', 'sma200', 'sma_ratio', 'dual_sma', 'donchian',
    'tsmom', 'rolling_sharpe', 'r2_trend', 'macd', 'paper_atr', 'ensemble',
]

summary_rows = []
stress_rows  = []

for method in METHODS:
    print(f"Running {method}...")
    cfg = ModelConfig()
    cfg.trend_method = method
    result = run_backtest(data_dict, cfg)
    is_r = is_slice(result)

    # Core metrics
    row = {
        'method':   method,
        'sharpe':   sharpe(is_r),
        'ann_ret':  ann_ret(is_r),
        'ann_vol':  ann_vol(is_r),
        'sortino':  sortino(is_r),
        'mdd':      mdd(is_r),
        'calmar':   calmar(is_r),
        'pct_pos':  (is_r > 0).mean(),
    }
    summary_rows.append(row)

    # Stress periods
    srow = {'method': method}
    for label, (s, e) in STRESS_PERIODS.items():
        srow[label] = stress_return(is_r, s, e)
    stress_rows.append(srow)

summary_df = pd.DataFrame(summary_rows)
stress_df  = pd.DataFrame(stress_rows)

# ---------------------------------------------------------------------------
# Baseline delta
# ---------------------------------------------------------------------------

baseline_sharpe = summary_df.loc[summary_df['method'] == 'sma_ratio', 'sharpe'].iloc[0]
summary_df['delta_sharpe'] = summary_df['sharpe'] - baseline_sharpe
summary_df = summary_df.sort_values('sharpe', ascending=False).reset_index(drop=True)
summary_df.index += 1  # 1-based rank

# Reorder stress_df to match summary rank order
rank_order = summary_df['method'].tolist()
stress_df  = stress_df.set_index('method').reindex(rank_order)

# ---------------------------------------------------------------------------
# Print output
# ---------------------------------------------------------------------------

print()
print("=" * 88)
print("=== ALL TREND METHODS — IS PERFORMANCE (2004–2017) ===")
print(f"Baseline (sma_ratio): Sharpe {baseline_sharpe:.3f}")
print("=" * 88)

# Summary table header
print()
print("SUMMARY TABLE (ranked by IS Sharpe):")
hdr = (
    f"{'Rank':<5} {'Method':<16} {'Sharpe':>7} {'Ann.Ret':>8} {'Ann.Vol':>8} "
    f"{'Sortino':>8} {'MDD':>8} {'Calmar':>7} {'%Pos':>6} {'ΔSharpe':>8}"
)
print(hdr)
print("-" * len(hdr))

for rank, row in summary_df.iterrows():
    delta_str = f"+{row['delta_sharpe']:.3f}" if row['delta_sharpe'] >= 0 else f"{row['delta_sharpe']:.3f}"
    print(
        f"{rank:<5} {row['method']:<16} {row['sharpe']:>7.3f} {row['ann_ret']:>7.2%} "
        f"{row['ann_vol']:>7.2%} {row['sortino']:>8.3f} {row['mdd']:>7.2%} "
        f"{row['calmar']:>7.3f} {row['pct_pos']:>6.1%} {delta_str:>8}"
    )

# Stress table
print()
print("STRESS PERIODS (IS):")
stress_hdr = f"{'Method':<16} {'GFC':>10} {'Euro Cris':>11} {'Taper':>8} {'China/HY':>10}"
print(stress_hdr)
print("-" * len(stress_hdr))

for method in rank_order:
    sr = stress_df.loc[method]
    def fmt(v):
        if np.isnan(v):
            return '   N/A'
        sign = '+' if v >= 0 else ''
        return f"{sign}{v:.2%}"
    print(
        f"{method:<16} {fmt(sr['GFC']):>10} {fmt(sr['Euro Cris']):>11} "
        f"{fmt(sr['Taper']):>8} {fmt(sr['China/HY']):>10}"
    )

# Description
print()
print("DESCRIPTION OF EACH METHOD:")
for m in METHODS:
    print(f"  {m}: {METHOD_DESC[m]}")

print()
print("Done.")
