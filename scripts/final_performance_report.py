"""
Definitive baseline performance report for AMAAM.

Runs the model exactly as configured (no modifications) and produces a
comprehensive 7-section analysis comparing AMAAM to SPY buy-and-hold.
"""

import sys
sys.path.insert(0, '/Users/GualyCr/Desktop/AMAAM/amaam/.claude/worktrees/reverent-mayer')

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

from config.default_config import ModelConfig
from config.etf_universe import HEDGING_SLEEVE_TICKERS, MAIN_SLEEVE_TICKERS
from src.backtest.engine import run_backtest
from src.data.loader import load_validated_data


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

data_dir = Path('/Users/GualyCr/Desktop/AMAAM/amaam/.claude/worktrees/reverent-mayer/data/processed')
data_dict = load_validated_data(data_dir)
cfg = ModelConfig()
result = run_backtest(data_dict, cfg)

# SPY month-end returns, then align to AMAAM's execution-date index by year-month.
# AMAAM's monthly_returns are indexed by execution date (first trading day of
# the holding month), while SPY resamples to month-end.  We join on the
# year-month period so each SPY month-end return is paired with the AMAAM
# return that was earned during the same calendar month.
_spy_me = data_dict['SPY']['Close'].resample('ME').last().pct_change().dropna()
_spy_me.index = _spy_me.index.to_period('M')
_amaam_periods = result.monthly_returns.copy()
_amaam_periods.index = _amaam_periods.index.to_period('M')
spy_monthly = _spy_me.reindex(_amaam_periods.index)
spy_monthly.index = result.monthly_returns.index  # restore original datetime index


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def ann_ret(r):
    return (1 + r.mean())**12 - 1

def ann_vol(r):
    return r.std() * np.sqrt(12)

def sharpe(r):
    return ann_ret(r) / ann_vol(r)

def mdd(r):
    eq = (1 + r).cumprod()
    return (eq / eq.cummax() - 1).min()

def sortino(r):
    down = r[r < 0]
    return ann_ret(r) / (down.std() * np.sqrt(12)) if len(down) > 0 else np.nan

def calmar(r):
    return ann_ret(r) / abs(mdd(r))


# ---------------------------------------------------------------------------
# Config block
# ---------------------------------------------------------------------------

print("=" * 72)
print("AMAAM — DEFINITIVE BASELINE PERFORMANCE REPORT")
print("=" * 72)
print()
print("CONFIG")
print("-" * 72)
print(f"  backtest_start             : {cfg.backtest_start}")
print(f"  backtest_end               : {cfg.backtest_end}")
print(f"  hedging_sleeve_tickers     : {HEDGING_SLEEVE_TICKERS}")
print(f"  hedging_sleeve_top_n       : {cfg.hedging_sleeve_top_n}")
print(f"  main_sleeve_tickers        : {MAIN_SLEEVE_TICKERS}")
print(f"  main_sleeve_top_n          : {cfg.main_sleeve_top_n}")
print(f"  momentum_blend             : {cfg.momentum_blend}")
print(f"  momentum_blend_lookbacks   : {cfg.momentum_blend_lookbacks}")
print(f"  weight_momentum            : {cfg.weight_momentum}")
print(f"  weight_volatility          : {cfg.weight_volatility}")
print(f"  weight_correlation         : {cfg.weight_correlation}")
print(f"  weight_trend               : {cfg.weight_trend}")
print(f"  trend_method               : {cfg.trend_method}")
print(f"  trend_rank_scale           : {cfg.trend_rank_scale}")
print(f"  weighting_scheme           : {cfg.weighting_scheme}")
print(f"  rebalancing_frequency      : {cfg.rebalancing_frequency}")
print(f"  correlation_lookback       : {cfg.correlation_lookback}")
print(f"  vol_blend                  : {cfg.vol_blend}")
print(f"  correlation_blend          : {cfg.correlation_blend}")
print()


# ---------------------------------------------------------------------------
# Section 1: Full Period Summary
# ---------------------------------------------------------------------------

r_a = result.monthly_returns.dropna()
r_s = spy_monthly.dropna()

start_date = r_a.index.min()
end_date   = r_a.index.max()
n_months   = len(r_a)

print("=" * 72)
print("SECTION 1: FULL PERIOD SUMMARY")
print("=" * 72)
print(f"  Period : {start_date.strftime('%Y-%m-%d')} → {end_date.strftime('%Y-%m-%d')}")
print(f"  Months : {n_months}")
print()

def full_stats(r, label):
    ar     = ann_ret(r)
    av     = ann_vol(r)
    sh     = sharpe(r)
    so     = sortino(r)
    md     = mdd(r)
    ca     = calmar(r)
    pct_up = (r > 0).mean() * 100
    best   = r.max()
    worst  = r.min()
    skew   = stats.skew(r)
    kurt   = stats.kurtosis(r)   # excess kurtosis (Fisher)
    return {
        'label': label,
        'Ann. Return':      ar,
        'Ann. Volatility':  av,
        'Sharpe Ratio':     sh,
        'Sortino Ratio':    so,
        'Max Drawdown':     md,
        'Calmar Ratio':     ca,
        '% Months Positive': pct_up,
        'Best Month':       best,
        'Worst Month':      worst,
        'Skewness':         skew,
        'Excess Kurtosis':  kurt,
    }

s_amaam = full_stats(r_a, 'AMAAM')
s_spy   = full_stats(r_s, 'SPY')

metrics_order = [
    'Ann. Return', 'Ann. Volatility', 'Sharpe Ratio', 'Sortino Ratio',
    'Max Drawdown', 'Calmar Ratio', '% Months Positive',
    'Best Month', 'Worst Month', 'Skewness', 'Excess Kurtosis',
]

fmt_pct  = {'Ann. Return', 'Ann. Volatility', 'Max Drawdown', 'Best Month', 'Worst Month'}
fmt_pct2 = {'% Months Positive'}

print(f"  {'Metric':<22}  {'AMAAM':>12}  {'SPY':>12}")
print(f"  {'-'*22}  {'-'*12}  {'-'*12}")
for m in metrics_order:
    a_val = s_amaam[m]
    s_val = s_spy[m]
    if m in fmt_pct:
        print(f"  {m:<22}  {a_val*100:>11.2f}%  {s_val*100:>11.2f}%")
    elif m in fmt_pct2:
        print(f"  {m:<22}  {a_val:>11.1f}%  {s_val:>11.1f}%")
    else:
        print(f"  {m:<22}  {a_val:>12.3f}  {s_val:>12.3f}")
print()


# ---------------------------------------------------------------------------
# Section 2: IS / OOS
# ---------------------------------------------------------------------------

IS_START  = '2004-01-01'
IS_END    = '2017-12-31'
OOS_START = '2018-01-01'

r_a_is  = r_a.loc[IS_START:IS_END]
r_s_is  = r_s.loc[IS_START:IS_END]
r_a_oos = r_a.loc[OOS_START:]
r_s_oos = r_s.loc[OOS_START:]

print("=" * 72)
print("SECTION 2: IN-SAMPLE (IS) / OUT-OF-SAMPLE (OOS)")
print("=" * 72)
print(f"  IS  : {IS_START} → {IS_END}  ({len(r_a_is)} months)")
print(f"  OOS : {OOS_START} → {end_date.strftime('%Y-%m-%d')}  ({len(r_a_oos)} months)")
print()

def period_stats(r, label):
    return {
        'label': label,
        'Ann. Return':    ann_ret(r),
        'Ann. Vol':       ann_vol(r),
        'Sharpe':         sharpe(r),
        'Sortino':        sortino(r),
        'MDD':            mdd(r),
    }

ps = {
    'AMAAM IS':  period_stats(r_a_is,  'AMAAM IS'),
    'AMAAM OOS': period_stats(r_a_oos, 'AMAAM OOS'),
    'SPY IS':    period_stats(r_s_is,  'SPY IS'),
    'SPY OOS':   period_stats(r_s_oos, 'SPY OOS'),
}

sub_metrics = ['Ann. Return', 'Ann. Vol', 'Sharpe', 'Sortino', 'MDD']
fmt_sub_pct  = {'Ann. Return', 'Ann. Vol', 'MDD'}

print(f"  {'Metric':<18}  {'AMAAM IS':>10}  {'AMAAM OOS':>10}  {'SPY IS':>10}  {'SPY OOS':>10}")
print(f"  {'-'*18}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")
for m in sub_metrics:
    vals = [ps[k][m] for k in ['AMAAM IS', 'AMAAM OOS', 'SPY IS', 'SPY OOS']]
    if m in fmt_sub_pct:
        row = "  ".join(f"{v*100:>9.2f}%" for v in vals)
    else:
        row = "  ".join(f"{v:>10.3f}" for v in vals)
    print(f"  {m:<18}  {row}")

is_sr  = ps['AMAAM IS']['Sharpe']
oos_sr = ps['AMAAM OOS']['Sharpe']
degradation = (is_sr - oos_sr) / is_sr * 100 if is_sr != 0 else float('nan')
print()
print(f"  IS→OOS Sharpe degradation : {degradation:+.1f}%", end="")
if degradation < 0:
    print("  (negative = OOS beats IS)", end="")
print()
print()


# ---------------------------------------------------------------------------
# Section 3: Annual Returns
# ---------------------------------------------------------------------------

print("=" * 72)
print("SECTION 3: ANNUAL RETURNS")
print("=" * 72)
print()

annual_a = r_a.groupby(r_a.index.year).apply(lambda x: (1 + x).prod() - 1)
annual_s = r_s.groupby(r_s.index.year).apply(lambda x: (1 + x).prod() - 1)

years = sorted(set(annual_a.index) | set(annual_s.index))

annual_df = pd.DataFrame({
    'AMAAM': annual_a,
    'SPY':   annual_s,
}, index=annual_a.index)
annual_df['Delta'] = annual_df['AMAAM'] - annual_df['SPY']

# Flag best 3 and worst 3 AMAAM years
amaam_vals = annual_df['AMAAM'].dropna().sort_values()
worst3 = set(amaam_vals.head(3).index)
best3  = set(amaam_vals.tail(3).index)

print(f"  {'Year':<6}  {'AMAAM':>10}  {'SPY':>10}  {'Delta':>10}  {'Flag'}")
print(f"  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*4}")
for yr in sorted(annual_df.index):
    a_val = annual_df.loc[yr, 'AMAAM']
    s_val = annual_df.loc[yr, 'SPY']
    d_val = annual_df.loc[yr, 'Delta']
    a_str = f"{a_val*100:>9.2f}%" if not np.isnan(a_val) else "        N/A"
    s_str = f"{s_val*100:>9.2f}%" if not np.isnan(s_val) else "        N/A"
    d_str = f"{d_val*100:>9.2f}%" if not np.isnan(d_val) else "        N/A"
    flag = ""
    if yr in best3:
        flag = " ★"
    elif yr in worst3:
        flag = " ✗"
    print(f"  {yr:<6}  {a_str}  {s_str}  {d_str}  {flag}")
print()
print("  ★ = Best 3 AMAAM years   ✗ = Worst 3 AMAAM years")
print()


# ---------------------------------------------------------------------------
# Section 4: Rolling Performance
# ---------------------------------------------------------------------------

print("=" * 72)
print("SECTION 4: ROLLING PERFORMANCE (12-MONTH WINDOWS)")
print("=" * 72)
print()

# Rolling 12-month return
roll12_a = r_a.rolling(12).apply(lambda x: (1 + x).prod() - 1, raw=False)
roll12_s = r_s.rolling(12).apply(lambda x: (1 + x).prod() - 1, raw=False)

# Rolling 12-month annualised Sharpe
def rolling_sharpe_12(r):
    roll_ret = r.rolling(12).mean()
    roll_std = r.rolling(12).std()
    return (roll_ret * 12) / (roll_std * np.sqrt(12))

roll_sh_a = rolling_sharpe_12(r_a)

roll12_a_clean = roll12_a.dropna()
roll12_s_clean = roll12_s.dropna()
roll_sh_a_clean = roll_sh_a.dropna()

# Align
common_idx = roll12_a_clean.index.intersection(roll12_s_clean.index)
r12a_c = roll12_a_clean.loc[common_idx]
r12s_c = roll12_s_clean.loc[common_idx]

amaam_beats = (r12a_c > r12s_c).mean() * 100
amaam_trails_10 = (r12s_c - r12a_c > 0.10).mean() * 100

print(f"  Rolling 12m Sharpe (AMAAM)")
print(f"    Min    : {roll_sh_a_clean.min():.3f}")
print(f"    Median : {roll_sh_a_clean.median():.3f}")
print(f"    Max    : {roll_sh_a_clean.max():.3f}")
print()
print(f"  Rolling 12m Return — AMAAM")
print(f"    Min    : {roll12_a_clean.min()*100:.2f}%")
print(f"    Median : {roll12_a_clean.median()*100:.2f}%")
print(f"    Max    : {roll12_a_clean.max()*100:.2f}%")
print()
print(f"  Rolling 12m Return — SPY")
print(f"    Min    : {roll12_s_clean.min()*100:.2f}%")
print(f"    Median : {roll12_s_clean.median()*100:.2f}%")
print(f"    Max    : {roll12_s_clean.max()*100:.2f}%")
print()
print(f"  % of rolling 12m windows AMAAM beats SPY    : {amaam_beats:.1f}%")
print(f"  % of rolling 12m windows AMAAM trails SPY >10pp : {amaam_trails_10:.1f}%")
print()


# ---------------------------------------------------------------------------
# Section 5: Drawdown Analysis
# ---------------------------------------------------------------------------

print("=" * 72)
print("SECTION 5: DRAWDOWN ANALYSIS")
print("=" * 72)
print()

def find_top_drawdowns(r, n=5):
    """Find the top-N drawdowns by depth, returning start, trough, recovery, depth, duration."""
    eq = (1 + r).cumprod()
    eq.index = range(len(eq))  # use integer index for arithmetic
    dates = r.index

    drawdowns = []
    i = 0
    while i < len(eq):
        if eq.iloc[i] < eq.iloc[:i+1].max():
            # we're in a drawdown — find the peak before this point
            peak_idx = eq.iloc[:i+1].idxmax()
            # find trough from peak onward
            remaining = eq.iloc[peak_idx:]
            trough_idx = remaining.idxmin()
            trough_val = remaining.min()
            peak_val   = eq.iloc[peak_idx]
            depth      = trough_val / peak_val - 1

            # find recovery: first point after trough >= peak
            after_trough = eq.iloc[trough_idx:]
            recovered = after_trough[after_trough >= peak_val]
            if len(recovered) > 0:
                recovery_idx = recovered.index[0]
                recovery_date = dates[recovery_idx]
                ongoing = False
            else:
                recovery_idx = len(eq) - 1
                recovery_date = None
                ongoing = True

            duration = trough_idx - peak_idx

            drawdowns.append({
                'start':    dates[peak_idx],
                'trough':   dates[trough_idx],
                'recovery': recovery_date,
                'depth':    depth,
                'duration': duration,
                'ongoing':  ongoing,
            })
            # advance past trough
            i = recovery_idx + 1 if not ongoing else len(eq)
        else:
            i += 1

    # Sort by depth (most negative first) and take top-N
    drawdowns.sort(key=lambda x: x['depth'])
    return drawdowns[:n]

top_dd = find_top_drawdowns(r_a, n=5)

# Time in drawdown
eq_a = (1 + r_a).cumprod()
in_dd = (eq_a < eq_a.cummax()).mean() * 100

print(f"  Top 5 Drawdowns (AMAAM)")
print()
print(f"  {'#':<3}  {'Start':<12}  {'Trough':<12}  {'Recovery':<12}  {'Depth':>8}  {'Duration':>10}")
print(f"  {'-'*3}  {'-'*12}  {'-'*12}  {'-'*12}  {'-'*8}  {'-'*10}")
for i, dd in enumerate(top_dd, 1):
    rec_str = dd['recovery'].strftime('%Y-%m-%d') if dd['recovery'] else 'Ongoing'
    print(f"  {i:<3}  {dd['start'].strftime('%Y-%m-%d'):<12}  "
          f"{dd['trough'].strftime('%Y-%m-%d'):<12}  "
          f"{rec_str:<12}  "
          f"{dd['depth']*100:>7.2f}%  "
          f"{dd['duration']:>8}mo")
print()
print(f"  Time in drawdown : {in_dd:.1f}% of months")
print()


# ---------------------------------------------------------------------------
# Section 6: Stress Periods
# ---------------------------------------------------------------------------

print("=" * 72)
print("SECTION 6: STRESS PERIOD ANALYSIS")
print("=" * 72)
print()

stress_periods = [
    ("GFC",               "2007-10-01", "2009-03-31"),
    ("Euro Crisis",       "2010-04-01", "2011-09-30"),
    ("Taper Tantrum",     "2013-05-01", "2013-06-30"),
    ("China/HY Selloff",  "2015-08-01", "2016-02-29"),
    ("COVID Crash",       "2020-02-01", "2020-03-31"),
    ("2022 Rate Shock",   "2022-01-01", "2022-12-31"),
    ("2025 Tariff Shock", "2025-02-01", cfg.backtest_end),
]

print(f"  {'Period':<22}  {'AMAAM Cum. Ret':>14}  {'SPY Cum. Ret':>13}")
print(f"  {'-'*22}  {'-'*14}  {'-'*13}")
for name, start, end in stress_periods:
    ra_sub = r_a.loc[start:end].dropna()
    rs_sub = r_s.loc[start:end].dropna()
    if len(ra_sub) == 0:
        a_cum_str = "        N/A"
    else:
        a_cum = (1 + ra_sub).prod() - 1
        a_cum_str = f"{a_cum*100:>12.2f}%"
    if len(rs_sub) == 0:
        s_cum_str = "       N/A"
    else:
        s_cum = (1 + rs_sub).prod() - 1
        s_cum_str = f"{s_cum*100:>11.2f}%"
    print(f"  {name:<22}  {a_cum_str}  {s_cum_str}")
print()


# ---------------------------------------------------------------------------
# Section 7: Return Distribution
# ---------------------------------------------------------------------------

print("=" * 72)
print("SECTION 7: RETURN DISTRIBUTION")
print("=" * 72)
print()

bins = [
    ("< -5%",       lambda r: r < -0.05),
    ("-5% to -3%",  lambda r: (r >= -0.05) & (r < -0.03)),
    ("-3% to -1%",  lambda r: (r >= -0.03) & (r < -0.01)),
    ("-1% to  0%",  lambda r: (r >= -0.01) & (r < 0.00)),
    (" 0% to  1%",  lambda r: (r >= 0.00)  & (r < 0.01)),
    (" 1% to  3%",  lambda r: (r >= 0.01)  & (r < 0.03)),
    (" 3% to  5%",  lambda r: (r >= 0.03)  & (r < 0.05)),
    ("> 5%",        lambda r: r >= 0.05),
]

total_a = len(r_a)
total_s = len(r_s)

print(f"  {'Bucket':<14}  {'AMAAM Count':>12}  {'AMAAM %':>9}  {'SPY Count':>10}  {'SPY %':>7}")
print(f"  {'-'*14}  {'-'*12}  {'-'*9}  {'-'*10}  {'-'*7}")
for label, condition in bins:
    cnt_a = condition(r_a).sum()
    cnt_s = condition(r_s).sum()
    pct_a = cnt_a / total_a * 100
    pct_s = cnt_s / total_s * 100
    print(f"  {label:<14}  {cnt_a:>12d}  {pct_a:>8.1f}%  {cnt_s:>10d}  {pct_s:>6.1f}%")

print()
print(f"  Total months : AMAAM={total_a}  SPY={total_s}")
print()
print(f"  AMAAM Skewness        : {stats.skew(r_a):.4f}")
print(f"  AMAAM Excess Kurtosis : {stats.kurtosis(r_a):.4f}")
print(f"  SPY   Skewness        : {stats.skew(r_s):.4f}")
print(f"  SPY   Excess Kurtosis : {stats.kurtosis(r_s):.4f}")
print()
print("=" * 72)
print("END OF REPORT")
print("=" * 72)
