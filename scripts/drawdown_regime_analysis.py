"""
Deep Drawdown Regime Analysis — AMAAM vs SPY.

Classifies every backtest month into one of four drawdown-regime buckets,
identifies model-specific underperformance episodes, provides month-by-month
underwater tables for the top-5 drawdown events, computes rolling 12-month
return comparisons, and correlates model returns with the SPY bull/bear regime.

Sections
--------
1. Bucket classification (4-regime taxonomy)
2. Model-only drawdown episodes (Bucket 2)
3. Top-5 underwater duration tables
4. Rolling 12-month return comparison
5. Correlation with SPY bull/bear market regime
"""

import sys
from pathlib import Path

# Allow running from the project root without installing the package.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from config.default_config import ModelConfig
from src.backtest.engine import run_backtest
from src.data.loader import load_validated_data

# ---------------------------------------------------------------------------
# 0. Load data and run backtest
# ---------------------------------------------------------------------------

DATA_DIR = PROJECT_ROOT / "data" / "processed"

print("Loading data …")
data_dict = load_validated_data(DATA_DIR)

print("Running backtest …")
cfg = ModelConfig()
result = run_backtest(data_dict, cfg)

monthly_model: pd.Series = result.monthly_returns   # indexed by exec date (first of month)

# SPY monthly returns: resample to month-end, then align by year-month period
# because the engine's execution dates are first-of-month (one-day lag from signal)
# while SPY resample('ME') produces last-of-month timestamps.  A direct reindex()
# will find zero matches, so we match by period instead.
spy_close = data_dict["SPY"]["Close"].resample("ME").last()
spy_ret_raw = spy_close.pct_change().dropna()

# Build a period-keyed lookup, then map model's exec-date periods to SPY returns.
spy_by_period = spy_ret_raw.copy()
spy_by_period.index = spy_by_period.index.to_period("M")
model_periods = monthly_model.index.to_period("M")

spy_returns = pd.Series(
    spy_by_period.reindex(model_periods).values,
    index=monthly_model.index,
    name="spy_return",
)

# Also build SPY daily-close for the 200-day SMA computation (Part 5).
spy_daily_close = data_dict["SPY"]["Close"]

# Drop any months where either series has NaN (warm-up spillover).
valid_mask = monthly_model.notna() & spy_returns.notna()
monthly_model = monthly_model[valid_mask]
spy_returns = spy_returns[valid_mask]

# ---------------------------------------------------------------------------
# Helper: drawdown series from returns
# ---------------------------------------------------------------------------

def to_drawdown(returns: pd.Series) -> pd.Series:
    """Compute the rolling drawdown (negative, 0 = at peak) from a return series."""
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    return cumulative / running_max - 1.0


model_dd = to_drawdown(monthly_model)
spy_dd   = to_drawdown(spy_returns)

# Threshold for "at/near peak" = within 2% of the peak.
PEAK_THRESHOLD = -0.02

# ---------------------------------------------------------------------------
# 1. Bucket classification
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("PART 1 — MONTHLY REGIME BUCKET CLASSIFICATION")
print("=" * 70)

model_in_dd = model_dd < PEAK_THRESHOLD
spy_in_dd   = spy_dd   < PEAK_THRESHOLD

bucket = pd.Series(index=monthly_model.index, dtype=int)
bucket[ model_in_dd &  spy_in_dd] = 1   # Both in drawdown
bucket[ model_in_dd & ~spy_in_dd] = 2   # Model only in drawdown
bucket[~model_in_dd &  spy_in_dd] = 3   # SPY only in drawdown
bucket[~model_in_dd & ~spy_in_dd] = 4   # Both at/near peak

BUCKET_LABELS = {
    1: "Both in drawdown",
    2: "Model-only in drawdown",
    3: "SPY-only in drawdown",
    4: "Both at/near peak",
}

total = len(bucket)
print(f"\nTotal months analysed: {total}")
print(f"Date range: {monthly_model.index.min().date()} → {monthly_model.index.max().date()}\n")

MONTHS_PER_YEAR = 12

header = (
    f"{'Bucket':<5} {'Description':<26} {'Count':>6} {'%Tot':>6} "
    f"{'AvgRet(M)':>10} {'AvgRet(S)':>10} "
    f"{'Ann(M)%':>8} {'Ann(S)%':>8} "
    f"{'AvgDD(M)':>9} {'AvgDD(S)':>9}"
)
print(header)
print("-" * len(header))

for b in [1, 2, 3, 4]:
    mask = bucket == b
    n = mask.sum()
    if n == 0:
        print(f"  {b}   {BUCKET_LABELS[b]:<26}  {0:>5}  {0:>5.1f}%  — (no months)")
        continue

    pct  = 100 * n / total
    mr   = monthly_model[mask].mean()
    sr   = spy_returns[mask].mean()
    ann_m = ((1 + mr) ** MONTHS_PER_YEAR - 1) * 100
    ann_s = ((1 + sr) ** MONTHS_PER_YEAR - 1) * 100
    add_m = model_dd[mask].mean() * 100
    add_s = spy_dd[mask].mean() * 100

    print(
        f"  {b}   {BUCKET_LABELS[b]:<26}  {n:>5}  {pct:>5.1f}%  "
        f"{mr*100:>+9.2f}%  {sr*100:>+9.2f}%  "
        f"{ann_m:>+7.1f}%  {ann_s:>+7.1f}%  "
        f"{add_m:>+8.2f}%  {add_s:>+8.2f}%"
    )

# ---------------------------------------------------------------------------
# 2. Model-only drawdown episodes (Bucket 2)
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("PART 2 — MODEL-ONLY DRAWDOWN EPISODES (BUCKET 2)")
print("=" * 70)

b2_mask = bucket == 2

# Identify contiguous episodes.
def find_episodes(mask: pd.Series):
    """Return list of (start_idx, end_idx) for runs of True in mask."""
    episodes = []
    in_ep = False
    start = None
    for i, (dt, val) in enumerate(mask.items()):
        if val and not in_ep:
            in_ep = True
            start = dt
        elif not val and in_ep:
            in_ep = False
            episodes.append((start, dt))
    if in_ep:
        episodes.append((start, mask.index[-1]))
    return episodes

episodes = find_episodes(b2_mask)

# Allocations are indexed by signal date; we need to map them to the monthly
# return execution dates.  The engine records signal_date → alloc but returns
# are indexed by exec_date (next day).  Build a best-effort mapping.
alloc_df = result.allocations
has_alloc = not alloc_df.empty

from config.etf_universe import MAIN_SLEEVE_TICKERS

print(f"\nFound {len(episodes)} Bucket-2 episode(s)\n")

for ep_num, (start, end) in enumerate(episodes, 1):
    ep_mask = (monthly_model.index >= start) & (monthly_model.index <= end)
    ep_model = monthly_model[ep_mask]
    ep_spy   = spy_returns[ep_mask]

    n_months   = len(ep_model)
    cum_model  = (1 + ep_model).prod() - 1
    cum_spy    = (1 + ep_spy).prod() - 1
    peak_dd    = model_dd[ep_mask].min()

    print(f"  Episode {ep_num}: {start.date()} → {end.date()} ({n_months} month(s))")
    print(f"    Peak model drawdown : {peak_dd*100:+.2f}%")
    print(f"    Cumulative model return: {cum_model*100:+.2f}%")
    print(f"    Cumulative SPY return  : {cum_spy*100:+.2f}%")

    # Infer market context from year.
    year = start.year
    if   year in range(2007, 2010): context = "Global Financial Crisis / Bear market"
    elif year in range(2011, 2012): context = "European Debt Crisis / US Downgrade"
    elif year in range(2013, 2015): context = "Post-QE reflation — strong risk-on trend"
    elif year in range(2015, 2016): context = "China devaluation scare / commodity collapse"
    elif year in range(2016, 2017): context = "Election-driven cyclical rotation (Trump rally)"
    elif year in range(2017, 2019): context = "Synchronized global growth / low-vol melt-up"
    elif year in range(2019, 2020): context = "Late-cycle US equity momentum before COVID"
    elif year in range(2020, 2021): context = "COVID crash followed by rapid V-shaped recovery"
    elif year in range(2021, 2023): context = "Post-COVID growth boom / inflation spike"
    elif year in range(2023, 2025): context = "AI/tech-led rally; rate normalisation"
    else: context = "N/A"
    print(f"    Market context      : {context}")

    # Top-3 main sleeve holdings during episode (use signal dates ≤ end).
    if has_alloc:
        sig_dates_in_ep = alloc_df.index[
            (alloc_df.index >= start - pd.offsets.MonthBegin(2)) &
            (alloc_df.index <= end)
        ]
        if len(sig_dates_in_ep) > 0:
            ep_alloc = alloc_df.loc[sig_dates_in_ep, [t for t in MAIN_SLEEVE_TICKERS if t in alloc_df.columns]]
            avg_weights = ep_alloc.mean().sort_values(ascending=False)
            top3 = avg_weights[avg_weights > 0].head(3)
            if not top3.empty:
                top3_str = ", ".join(f"{t} ({w*100:.1f}%)" for t, w in top3.items())
                print(f"    Top-3 main sleeve (avg): {top3_str}")
    print()

# ---------------------------------------------------------------------------
# 3. Top-5 drawdown episodes — underwater duration tables
# ---------------------------------------------------------------------------

print("=" * 70)
print("PART 3 — TOP-5 MODEL DRAWDOWN EPISODES (MONTH-BY-MONTH)")
print("=" * 70)

def find_drawdown_episodes_by_depth(dd_series: pd.Series, top_n: int = 5):
    """
    Identify the top-N drawdown episodes by trough depth.

    An episode runs from the last time the cumulative NAV was at a peak
    (dd == 0) through the full recovery (dd returns to 0), or to the end
    of the series.  We split on zero-crossings and pick the deepest troughs.
    """
    episodes = []
    in_ep = False
    ep_start = None

    dd_arr = dd_series.values
    idx    = dd_series.index

    for i in range(len(dd_arr)):
        val = dd_arr[i]
        if not in_ep and val < 0:
            in_ep = True
            ep_start = i
        elif in_ep and val >= 0:
            episodes.append((ep_start, i - 1))
            in_ep = False

    if in_ep:
        episodes.append((ep_start, len(dd_arr) - 1))

    # Sort by trough depth (most negative first).
    episodes.sort(key=lambda x: dd_arr[x[0]:x[1]+1].min())
    return episodes[:top_n]

top5_ep = find_drawdown_episodes_by_depth(model_dd, top_n=5)

for ep_rank, (i0, i1) in enumerate(top5_ep, 1):
    ep_idx    = model_dd.index[i0:i1+1]
    ep_mdd    = model_dd.iloc[i0:i1+1]
    ep_sdd    = spy_dd.loc[ep_idx]
    ep_mr     = monthly_model.loc[ep_idx]
    ep_sr     = spy_returns.loc[ep_idx]
    trough_dd = ep_mdd.min()

    print(f"\n  Episode {ep_rank}: {ep_idx[0].date()} → {ep_idx[-1].date()} "
          f"| Peak DD = {trough_dd*100:.2f}%  ({len(ep_idx)} months)")

    col_w = [12, 10, 10, 11, 11, 18]
    hdr = (
        f"  {'Date':<{col_w[0]}} {'Model DD':>{col_w[1]}} {'SPY DD':>{col_w[2]}} "
        f"{'Model Ret':>{col_w[3]}} {'SPY Ret':>{col_w[4]}} {'vs SPY':>{col_w[5]}}"
    )
    print(hdr)
    print("  " + "-" * (sum(col_w) + 5 * 2))

    for dt in ep_idx:
        mdd_v = ep_mdd.loc[dt] * 100
        sdd_v = ep_sdd.loc[dt] * 100
        mr_v  = ep_mr.loc[dt]  * 100
        sr_v  = ep_sr.loc[dt]  * 100
        beat  = "BEATING SPY" if mr_v > sr_v else f"lagging ({mr_v - sr_v:+.2f}pp)"

        print(
            f"  {str(dt.date()):<{col_w[0]}} {mdd_v:>+{col_w[1]}.2f}% "
            f"{sdd_v:>+{col_w[2]}.2f}%  "
            f"{mr_v:>+{col_w[3]-1}.2f}%  {sr_v:>+{col_w[4]-1}.2f}%  {beat}"
        )

# ---------------------------------------------------------------------------
# 4. Rolling 12-month return comparison
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("PART 4 — ROLLING 12-MONTH RETURN COMPARISON")
print("=" * 70)

ROLL = 12

def rolling_ann_return(returns: pd.Series, window: int) -> pd.Series:
    """Annualised return over a rolling *window*-month window."""
    cumret = returns.rolling(window).apply(lambda x: (1 + x).prod() - 1, raw=True)
    return cumret

roll_model = rolling_ann_return(monthly_model, ROLL).dropna()
roll_spy   = rolling_ann_return(spy_returns,   ROLL).reindex(roll_model.index)
delta      = roll_model - roll_spy

pct_beat  = (delta > 0).mean() * 100
pct_lag10 = (delta < -0.10).mean() * 100

print(f"\n  Rolling 12-month windows analysed  : {len(delta)}")
print(f"  Model beats SPY                    : {pct_beat:.1f}% of windows")
print(f"  Model trails SPY by > 10pp         : {pct_lag10:.1f}% of windows")

# Worst 10 rolling 12-month periods for the model.
worst10 = roll_model.nsmallest(10)

print("\n  Worst 10 rolling 12-month periods (by model return):")
hdr2 = f"  {'End Date':<14} {'Model 12m':>10} {'SPY 12m':>10} {'Delta (pp)':>12}"
print(hdr2)
print("  " + "-" * (len(hdr2) - 2))
for dt, m_ret in worst10.items():
    s_ret = roll_spy.loc[dt] if dt in roll_spy.index else float("nan")
    d_pp  = (m_ret - s_ret) * 100 if not np.isnan(s_ret) else float("nan")
    print(
        f"  {str(dt.date()):<14} {m_ret*100:>+9.2f}%  {s_ret*100:>+9.2f}%  "
        f"{d_pp:>+11.2f}pp"
    )

# ---------------------------------------------------------------------------
# 5. Correlation with SPY bull/bear regime
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("PART 5 — MODEL RETURNS vs SPY BULL/BEAR REGIME")
print("=" * 70)

# Compute 200-day SMA of SPY on daily prices, then sample to month-end.
spy_sma200 = spy_daily_close.rolling(200).mean()
spy_ratio  = spy_daily_close / spy_sma200

# Sample ratio at month-end and align to model's execution-date index via period.
spy_ratio_me = spy_ratio.resample("ME").last()
spy_ratio_me.index = spy_ratio_me.index.to_period("M")
spy_ratio_monthly = pd.Series(
    spy_ratio_me.reindex(monthly_model.index.to_period("M")).values,
    index=monthly_model.index,
)

bull_mask = spy_ratio_monthly >= 1.0
bear_mask = spy_ratio_monthly <  1.0

# Some months may have NaN ratio (first 200 days); drop them.
valid_regime = spy_ratio_monthly.notna()

m_bull = monthly_model[bull_mask & valid_regime]
m_bear = monthly_model[bear_mask & valid_regime]
s_bull = spy_returns[bull_mask & valid_regime]
s_bear = spy_returns[bear_mask & valid_regime]

print(f"\n  SPY bull months (SPY > 200d SMA)  : {bull_mask.sum()} months")
print(f"  SPY bear months (SPY ≤ 200d SMA)  : {bear_mask.sum()} months")

def _ann(avg_monthly: float) -> float:
    return ((1 + avg_monthly) ** MONTHS_PER_YEAR - 1) * 100

print("\n  Average monthly return (model):")
print(f"    Bull-market months : {m_bull.mean()*100:+.3f}%  "
      f"(ann. {_ann(m_bull.mean()):+.1f}%)  |  n={len(m_bull)}")
print(f"    Bear-market months : {m_bear.mean()*100:+.3f}%  "
      f"(ann. {_ann(m_bear.mean()):+.1f}%)  |  n={len(m_bear)}")

print("\n  Average monthly return (SPY):")
print(f"    Bull-market months : {s_bull.mean()*100:+.3f}%  "
      f"(ann. {_ann(s_bull.mean()):+.1f}%)")
print(f"    Bear-market months : {s_bear.mean()*100:+.3f}%  "
      f"(ann. {_ann(s_bear.mean()):+.1f}%)")

# Drawdown frequency: % of months spent in drawdown, split by regime.
bull_dd_freq = model_dd[bull_mask & valid_regime].lt(PEAK_THRESHOLD).mean() * 100
bear_dd_freq = model_dd[bear_mask & valid_regime].lt(PEAK_THRESHOLD).mean() * 100

print("\n  Model drawdown frequency (% of months in DD > 2%):")
print(f"    Bull-market months : {bull_dd_freq:.1f}%")
print(f"    Bear-market months : {bear_dd_freq:.1f}%")

# --- summary correlation stat ---
# Pearson correlation between model_dd depth and monthly SPY return.
corr_dd_spy = np.corrcoef(model_dd[valid_regime], spy_ratio_monthly[valid_regime])[0, 1]
print(f"\n  Pearson corr (model DD depth vs SPY/SMA200 ratio): {corr_dd_spy:.3f}")
print("  (Negative = model drawdowns deepen when SPY is below trend — expected)")

print("\n" + "=" * 70)
print("Analysis complete.")
print("=" * 70)
