"""
IS-only sweep: trend_rank_scale (True/False) x wT {0.5, 1.0, 1.5, 2.0, 3.0}.

Tests 10 configurations over the in-sample window 2004-01-01 to 2017-12-31 and
reports Sharpe, annualised return, annualised volatility, and max drawdown.
All other config settings are kept at ModelConfig defaults (sma_ratio trend
method, equal weighting, blended momentum).
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

# ---------------------------------------------------------------------------
# IS window
# ---------------------------------------------------------------------------
IS_START = '2004-01-01'
IS_END   = '2017-12-31'


def ann_ret(r: pd.Series) -> float:
    """Annualised geometric return from monthly returns."""
    return (1 + r.mean()) ** 12 - 1


def ann_vol(r: pd.Series) -> float:
    """Annualised volatility from monthly returns."""
    return r.std() * np.sqrt(12)


def sharpe(r: pd.Series) -> float:
    """Sharpe ratio (no risk-free rate)."""
    return ann_ret(r) / ann_vol(r)


def mdd(r: pd.Series) -> float:
    """Maximum drawdown (negative fraction)."""
    eq = (1 + r).cumprod()
    return (eq / eq.cummax() - 1).min()


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------
wT_values = [0.5, 1.0, 1.5, 2.0, 3.0]
results = []

for scale in [False, True]:
    for wt in wT_values:
        cfg = ModelConfig()
        cfg.trend_rank_scale = scale
        cfg.weight_trend = wt

        r = run_backtest(data_dict, cfg)
        mr = r.monthly_returns
        mr.index = mr.index.to_period('M')

        is_r = mr[
            (mr.index >= pd.Period(IS_START, 'M')) &
            (mr.index <= pd.Period(IS_END, 'M'))
        ]

        results.append({
            'scale': scale,
            'wT': wt,
            'sharpe': sharpe(is_r),
            'ann_ret': ann_ret(is_r),
            'ann_vol': ann_vol(is_r),
            'mdd': mdd(is_r),
        })

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
df = pd.DataFrame(results)

# Baseline: scale=False, wT=1.0
baseline = df[(df['scale'] == False) & (df['wT'] == 1.0)].iloc[0]
baseline_sharpe = baseline['sharpe']

print()
print("=== TREND RANK SCALE + wT SWEEP (IS ONLY: 2004-2017) ===")
print(f"Baseline (scale=False, wT=1.0): Sharpe {baseline_sharpe:.3f}")
print()

header = f"{'wT':<6} {'Sharpe':>7} {'Ann.Ret':>9} {'Ann.Vol':>9} {'MDD':>9}  {'DSharpe vs baseline':>20}"

for scale in [False, True]:
    label = "False (raw T values +-2)" if not scale else "True (T ranked 1..N cross-sectionally)"
    print(f"trend_rank_scale = {label}:")
    print(header)

    sub = df[df['scale'] == scale].sort_values('wT')
    for _, row in sub.iterrows():
        tag = "  <- baseline" if (not scale and row['wT'] == 1.0) else ""
        delta = row['sharpe'] - baseline_sharpe
        print(
            f"{row['wT']:<6.1f} {row['sharpe']:>7.3f} "
            f"{row['ann_ret']*100:>8.2f}% "
            f"{row['ann_vol']*100:>8.2f}% "
            f"{row['mdd']*100:>8.2f}%  "
            f"{delta:>+.3f}{tag}"
        )
    print()

# Best configuration
best = df.loc[df['sharpe'].idxmax()]
print(
    f"Best IS configuration: scale={best['scale']}, wT={best['wT']:.1f} "
    f"--> Sharpe {best['sharpe']:.3f}"
)
