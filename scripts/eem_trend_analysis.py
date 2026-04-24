"""
Deep analysis of the sma_ratio trend signal on EEM across the full backtest period.

Covers:
  1. Price action overview
  2. Signal distribution (daily)
  3. Monthly signal at rebalance dates
  4. Signal vs price action quality check (major events)
  5. Buffer zone analysis
  6. Cost of being bearish (correct vs wrong exclusions)

Run with:
  python3.13 scripts/eem_trend_analysis.py
"""

import sys
sys.path.insert(0, '/Users/GualyCr/Desktop/AMAAM/amaam/.claude/worktrees/reverent-mayer')

from pathlib import Path

import numpy as np
import pandas as pd

from config.default_config import ModelConfig
from src.data.loader import load_validated_data
from src.factors.trend import compute_sma_ratio_signal

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
DATA_DIR = Path('/Users/GualyCr/Desktop/AMAAM/amaam/.claude/worktrees/reverent-mayer/data/processed')
data_dict = load_validated_data(DATA_DIR)
cfg = ModelConfig()

eem = data_dict['EEM']
close  = eem['Close']
high   = eem['High']
low    = eem['Low']

# Compute signal with default parameters (period=200, upper=1.01, lower=0.99)
PERIOD = 200
UPPER  = 1.01
LOWER  = 0.99

signal  = compute_sma_ratio_signal(close, period=PERIOD, upper_threshold=UPPER, lower_threshold=LOWER)
sma200  = close.rolling(PERIOD, min_periods=PERIOD).mean()
ratio   = close / sma200

# Drop warm-up NaNs for most analyses
sig_valid  = signal.dropna()
close_full = close  # keep full series for price stats

SEP = "=" * 72


# ===========================================================================
# SECTION 1 — PRICE ACTION OVERVIEW
# ===========================================================================
print(SEP)
print("SECTION 1 — PRICE ACTION OVERVIEW")
print(SEP)

first_date = close_full.index[0]
last_date  = close_full.index[-1]
total_days = len(close_full)

price_min_date = close_full.idxmin()
price_max_date = close_full.idxmax()
price_min      = close_full.min()
price_max      = close_full.max()

first_price = close_full.iloc[0]
last_price  = close_full.iloc[-1]
total_return = (last_price / first_price - 1) * 100

daily_rets = close_full.pct_change().dropna()
ann_vol    = daily_rets.std() * np.sqrt(252) * 100

print(f"First date          : {first_date.date()}")
print(f"Last date           : {last_date.date()}")
print(f"Total trading days  : {total_days:,}")
print(f"First price         : ${first_price:.2f}")
print(f"Last price          : ${last_price:.2f}")
print(f"Total return        : {total_return:+.1f}%")
print(f"Price minimum       : ${price_min:.2f}  on {price_min_date.date()}")
print(f"Price maximum       : ${price_max:.2f}  on {price_max_date.date()}")
print(f"Annualised vol      : {ann_vol:.1f}%")

# Major milestones — inspect key regions
milestones = [
    ("2007 pre-GFC peak",    "2007-01-01", "2008-01-01"),
    ("GFC trough",           "2008-07-01", "2009-04-01"),
    ("Post-GFC recovery",    "2009-01-01", "2011-06-01"),
    ("Euro crisis trough",   "2011-01-01", "2012-01-01"),
    ("China selloff trough", "2015-01-01", "2016-06-01"),
    ("EM selloff trough",    "2018-01-01", "2019-01-01"),
    ("COVID peak",           "2020-01-01", "2020-04-01"),
    ("Post-COVID peak",      "2020-07-01", "2022-01-01"),
    ("2022 trough",          "2021-10-01", "2022-12-31"),
]

print("\nMajor price milestones:")
print(f"  {'Label':<26} {'Date':<12} {'Price':>8}")
print(f"  {'-'*26} {'-'*12} {'-'*8}")
for label, start, end in milestones:
    window = close_full.loc[start:end]
    if window.empty:
        continue
    if "peak" in label.lower() or "recovery" in label.lower():
        idx = window.idxmax()
        val = window.max()
    else:
        idx = window.idxmin()
        val = window.min()
    print(f"  {label:<26} {str(idx.date()):<12} ${val:>7.2f}")


# ===========================================================================
# SECTION 2 — SIGNAL DISTRIBUTION (DAILY)
# ===========================================================================
print()
print(SEP)
print("SECTION 2 — SIGNAL DISTRIBUTION (DAILY)")
print(SEP)

n_bull  = (sig_valid == 2.0).sum()
n_bear  = (sig_valid == -2.0).sum()
n_total = len(sig_valid)
pct_bull = n_bull / n_total * 100
pct_bear = n_bear / n_total * 100

print(f"Valid signal days   : {n_total:,}")
print(f"Bullish (+2) days   : {n_bull:,}  ({pct_bull:.1f}%)")
print(f"Bearish (-2) days   : {n_bear:,}  ({pct_bear:.1f}%)")

# Signal flips
sig_shift  = sig_valid.shift(1)
flips      = (sig_valid != sig_shift) & sig_shift.notna()
flip_dates = sig_valid.index[flips]
n_flips    = flips.sum()
print(f"\nTotal signal flips  : {n_flips}")

# Flips per year
flip_series = pd.Series(1, index=flip_dates)
flips_per_year = flip_series.groupby(flip_series.index.year).sum()
print("\nFlips per year:")
for yr, cnt in flips_per_year.items():
    print(f"  {yr}: {cnt}")

# Streak analysis
def _streak_stats(sig: pd.Series, val: float) -> dict:
    """Return streak statistics for a given signal value."""
    mask     = (sig == val).astype(int)
    groups   = (mask != mask.shift()).cumsum()
    streaks  = []
    for g, grp in sig.groupby(groups):
        if len(grp) > 0 and grp.iloc[0] == val:
            streaks.append((grp.index[0], grp.index[-1], len(grp)))
    if not streaks:
        return {"count": 0, "avg_dur": 0, "longest": None}
    durations = [s[2] for s in streaks]
    longest   = max(streaks, key=lambda x: x[2])
    return {
        "count":   len(streaks),
        "avg_dur": np.mean(durations),
        "longest": longest,
    }

bull_stats = _streak_stats(sig_valid, 2.0)
bear_stats = _streak_stats(sig_valid, -2.0)

print(f"\nBullish streaks:")
print(f"  Count              : {bull_stats['count']}")
print(f"  Average duration   : {bull_stats['avg_dur']:.1f} days")
if bull_stats["longest"]:
    ls = bull_stats["longest"]
    print(f"  Longest streak     : {ls[2]} days  ({ls[0].date()} → {ls[1].date()})")

print(f"\nBearish streaks:")
print(f"  Count              : {bear_stats['count']}")
print(f"  Average duration   : {bear_stats['avg_dur']:.1f} days")
if bear_stats["longest"]:
    ls = bear_stats["longest"]
    print(f"  Longest streak     : {ls[2]} days  ({ls[0].date()} → {ls[1].date()})")


# ===========================================================================
# SECTION 3 — MONTHLY SIGNAL AT REBALANCE DATES
# ===========================================================================
print()
print(SEP)
print("SECTION 3 — MONTHLY SIGNAL AT REBALANCE DATES")
print(SEP)

# Last trading day of each month
monthly_last = (
    pd.Series(sig_valid.index, index=sig_valid.index)
    .groupby(sig_valid.index.to_period("M"))
    .last()
)
monthly_signal = sig_valid.reindex(monthly_last.values).dropna()

n_m_bull = (monthly_signal == 2.0).sum()
n_m_bear = (monthly_signal == -2.0).sum()
n_m_tot  = len(monthly_signal)
print(f"Total month-end samples : {n_m_tot}")
print(f"Bullish months          : {n_m_bull}  ({n_m_bull/n_m_tot*100:.1f}%)")
print(f"Bearish months          : {n_m_bear}  ({n_m_bear/n_m_tot*100:.1f}%)")

# Year-by-year
print("\nYear-by-year bullish vs bearish months:")
print(f"  {'Year':<6} {'Bull':>5} {'Bear':>5}")
print(f"  {'-'*6} {'-'*5} {'-'*5}")
for yr, grp in monthly_signal.groupby(monthly_signal.index.year):
    b  = (grp == 2.0).sum()
    br = (grp == -2.0).sum()
    print(f"  {yr:<6} {b:>5} {br:>5}")

# Transition months (signal changed vs prior month-end)
m_shift       = monthly_signal.shift(1)
m_transitions = monthly_signal[
    (monthly_signal != m_shift) & m_shift.notna()
]
print(f"\nTransition months ({len(m_transitions)} total):")
print(f"  {'Date':<12} {'From':>5} {'To':>5} {'Direction'}")
print(f"  {'-'*12} {'-'*5} {'-'*5} {'-'*16}")
for dt, val in m_transitions.items():
    prev = m_shift.loc[dt]
    direction = "BULL→BEAR" if val == -2.0 else "BEAR→BULL"
    print(f"  {str(dt.date()):<12} {int(prev):>5} {int(val):>5} {direction}")


# ===========================================================================
# SECTION 4 — SIGNAL VS PRICE ACTION QUALITY CHECK
# ===========================================================================
print()
print(SEP)
print("SECTION 4 — SIGNAL VS PRICE ACTION QUALITY CHECK")
print(SEP)

events = [
    ("GFC 2008",           "2007-06-01", "2007-12-31", "2008-06-01", "2009-03-31"),
    ("Euro crisis 2011",   "2011-01-01", "2011-06-30", "2011-06-01", "2012-03-31"),
    ("China selloff 2015", "2015-01-01", "2015-07-31", "2015-06-01", "2016-06-30"),
    ("EM selloff 2018",    "2017-07-01", "2018-04-30", "2018-04-01", "2019-01-31"),
    ("COVID 2020",         "2019-10-01", "2020-02-29", "2020-02-01", "2020-06-30"),
    ("2022 downturn",      "2021-06-01", "2021-12-31", "2021-10-01", "2022-12-31"),
]

def first_flip_after(date: pd.Timestamp, to_val: float, sig: pd.Series) -> pd.Timestamp | None:
    """Return the first date after `date` where signal == to_val."""
    subset = sig[sig.index > date]
    hits   = subset[subset == to_val]
    return hits.index[0] if len(hits) > 0 else None

def first_flip_after_bear(date: pd.Timestamp, sig: pd.Series) -> pd.Timestamp | None:
    return first_flip_after(date, -2.0, sig)

def first_flip_after_bull(date: pd.Timestamp, sig: pd.Series) -> pd.Timestamp | None:
    return first_flip_after(date, 2.0, sig)

# We want: first SUSTAINED flip — i.e. date of first bar that changed TO the value
# (not just a visit). For sma_ratio with carry, once it flips it tends to stay.
# We also want to find the actual first FLIP (state change), not just first occurrence.
def first_state_change_to(after_date: pd.Timestamp, to_val: float, sig: pd.Series) -> pd.Timestamp | None:
    """Return the first date after `after_date` where signal changes TO to_val."""
    subset    = sig[sig.index > after_date]
    prev_sub  = subset.shift(1)
    changes   = subset[(subset == to_val) & (prev_sub != to_val) & prev_sub.notna()]
    return changes.index[0] if len(changes) > 0 else None

for name, peak_start, peak_end, trough_start, trough_end in events:
    print(f"\n--- {name} ---")

    # Actual peak in the peak window
    peak_window  = close.loc[peak_start:peak_end]
    trough_window = close.loc[trough_start:trough_end]
    if peak_window.empty or trough_window.empty:
        print("  (insufficient data)")
        continue

    peak_date    = peak_window.idxmax()
    peak_price   = peak_window.max()
    trough_date  = trough_window.idxmin()
    trough_price = trough_window.min()
    drawdown     = (trough_price / peak_price - 1) * 100

    print(f"  Price peak         : {peak_date.date()}  ${peak_price:.2f}")
    print(f"  Price trough       : {trough_date.date()}  ${trough_price:.2f}  (drawdown {drawdown:.1f}%)")

    # When did signal first flip to bearish AFTER the peak?
    bear_flip = first_state_change_to(peak_date, -2.0, sig_valid)
    if bear_flip is not None:
        lag_bear = (bear_flip - peak_date).days
        price_at_bear = close.loc[bear_flip]
        price_move_to_bear = (price_at_bear / peak_price - 1) * 100
        print(f"  Bear signal fired  : {bear_flip.date()}  ${price_at_bear:.2f}  "
              f"(lag {lag_bear} days, price {price_move_to_bear:+.1f}% from peak)")
    else:
        print(f"  Bear signal fired  : never (within period)")
        bear_flip = None

    # When did signal first flip to bullish AFTER the trough?
    bull_flip = first_state_change_to(trough_date, 2.0, sig_valid)
    if bull_flip is not None:
        lag_bull = (bull_flip - trough_date).days
        price_at_bull = close.loc[bull_flip]
        price_move_to_bull = (price_at_bull / trough_price - 1) * 100
        print(f"  Bull signal fired  : {bull_flip.date()}  ${price_at_bull:.2f}  "
              f"(lag {lag_bull} days, price {price_move_to_bull:+.1f}% from trough)")
    else:
        print(f"  Bull signal fired  : never (within period)")
        bull_flip = None

    # Drawdown captured while bearish
    # = portion of the peak-to-trough range that occurred while signal == -2
    if bear_flip is not None and bear_flip <= trough_date:
        bear_period_close = close.loc[bear_flip:trough_date]
        bear_low          = bear_period_close.min()
        captured_dd       = (bear_low / peak_price - 1) * 100
        # What fraction of total drawdown was "missed" before bear signal fired
        price_at_flip_bear = close.loc[bear_flip]
        missed_pct         = (price_at_flip_bear / peak_price - 1) / (trough_price / peak_price - 1) * 100
        print(f"  Drawdown at bear flip: {(price_at_flip_bear/peak_price-1)*100:.1f}%  "
              f"(missed {missed_pct:.1f}% of total drawdown before signal)")
    else:
        if bear_flip is None:
            print(f"  Drawdown captured  : signal never flipped bearish during event")
        else:
            print(f"  Drawdown captured  : bear signal fired AFTER trough ({bear_flip.date()})")

    # Signal quality summary
    if bear_flip is not None and bear_flip <= trough_date:
        print(f"  Assessment         : CAUGHT — signal turned bearish before trough")
    elif bear_flip is not None and bear_flip > trough_date:
        days_late = (bear_flip - trough_date).days
        print(f"  Assessment         : LATE — signal turned bearish {days_late} days AFTER trough")
    else:
        print(f"  Assessment         : MISSED — no bearish signal during event")


# ===========================================================================
# SECTION 5 — BUFFER ZONE ANALYSIS
# ===========================================================================
print()
print(SEP)
print("SECTION 5 — BUFFER ZONE ANALYSIS")
print(SEP)

ratio_valid = ratio.dropna()

# Days where ratio is in buffer zone (0.99 < ratio <= 1.01)
in_buffer = (ratio_valid > LOWER) & (ratio_valid < UPPER)
n_buffer_days = in_buffer.sum()
pct_buffer    = n_buffer_days / len(ratio_valid) * 100
print(f"Buffer zone days    : {n_buffer_days:,}  ({pct_buffer:.1f}% of valid days)")

# Each time the ratio enters the buffer zone, does it lead to a flip or carry?
# "Enter buffer" = prior bar was outside buffer, current bar is inside buffer
was_outside   = ~((ratio_valid.shift(1) > LOWER) & (ratio_valid.shift(1) < UPPER))
just_entered  = in_buffer & was_outside & ratio_valid.shift(1).notna()
entry_dates   = ratio_valid.index[just_entered]
n_entries     = just_entered.sum()
print(f"Buffer zone entries : {n_entries}")

# For each entry, check if the signal flipped within the next 30 bars
n_flip   = 0
n_carry  = 0
for dt in entry_dates:
    sig_at_entry = sig_valid.loc[dt] if dt in sig_valid.index else np.nan
    if np.isnan(sig_at_entry):
        continue
    # Look ahead up to 30 bars
    future_sig = sig_valid[sig_valid.index > dt].iloc[:30]
    if len(future_sig) == 0:
        continue
    if (future_sig != sig_at_entry).any():
        n_flip += 1
    else:
        n_carry += 1

print(f"  Led to flip        : {n_flip}")
print(f"  Carried through    : {n_carry}")
if n_entries > 0:
    print(f"  Flip rate          : {n_flip/(n_flip+n_carry)*100:.1f}%")

# Distribution of ratio values at month-end dates
print(f"\nRatio distribution at month-end dates:")
me_ratio = ratio_valid.reindex(monthly_last.values).dropna()
pcts = [5, 10, 25, 50, 75, 90, 95]
print(f"  {'Pct':<6} {'Ratio':>8}")
for p in pcts:
    print(f"  {p:>3}%   {np.percentile(me_ratio.values, p):8.4f}")
print(f"  Mean   {me_ratio.mean():8.4f}")
print(f"  Std    {me_ratio.std():8.4f}")
print(f"  Min    {me_ratio.min():8.4f}  ({me_ratio.idxmin().date()})")
print(f"  Max    {me_ratio.max():8.4f}  ({me_ratio.idxmax().date()})")


# ===========================================================================
# SECTION 6 — COST OF BEING BEARISH
# ===========================================================================
print()
print(SEP)
print("SECTION 6 — COST OF BEING BEARISH")
print(SEP)

# Identify all bearish streaks at the monthly-rebalance level
# A "streak" is a contiguous run of month-end signals = -2
def _monthly_bear_streaks(monthly_sig: pd.Series) -> list:
    """Return list of (start_date, end_date, eem_return, n_months) for each bearish streak."""
    streaks   = []
    in_streak = False
    start_dt  = None
    for dt, val in monthly_sig.items():
        if val == -2.0 and not in_streak:
            in_streak = True
            start_dt  = dt
        elif val == 2.0 and in_streak:
            end_dt = dt  # first bullish month-end after streak
            streaks.append((start_dt, end_dt))
            in_streak = False
    # If streak extends to end of data
    if in_streak and start_dt is not None:
        streaks.append((start_dt, monthly_sig.index[-1]))
    return streaks

bear_streaks = _monthly_bear_streaks(monthly_signal)

print(f"Total bearish streaks (month-end level): {len(bear_streaks)}")
print()

results = []
for start_dt, end_dt in bear_streaks:
    # Compute EEM price return from the close on start_dt to close on end_dt
    if start_dt not in close.index or end_dt not in close.index:
        # Find nearest available
        try:
            p_start = close.asof(start_dt)
            p_end   = close.asof(end_dt)
        except Exception:
            continue
    else:
        p_start = close.loc[start_dt]
        p_end   = close.loc[end_dt]

    if p_start == 0 or np.isnan(p_start) or np.isnan(p_end):
        continue

    eem_ret  = (p_end / p_start - 1) * 100
    n_months = len(monthly_signal.loc[start_dt:end_dt])
    correct  = eem_ret < 0  # signal was correct if EEM fell
    results.append({
        "start": start_dt,
        "end":   end_dt,
        "n_months": n_months,
        "eem_ret":  eem_ret,
        "correct":  correct,
    })

print(f"  {'Start':<12} {'End':<12} {'Months':>7} {'EEM Ret':>9} {'Verdict'}")
print(f"  {'-'*12} {'-'*12} {'-'*7} {'-'*9} {'-'*16}")
for r in results:
    verdict = "CORRECT (saved loss)" if r["correct"] else "WRONG  (missed gain)"
    print(f"  {str(r['start'].date()):<12} {str(r['end'].date()):<12} "
          f"{r['n_months']:>7} {r['eem_ret']:>8.1f}% {verdict}")

# Top 3 correct exclusions (most negative EEM return = most saved)
correct_results = [r for r in results if r["correct"]]
wrong_results   = [r for r in results if not r["correct"]]

correct_sorted = sorted(correct_results, key=lambda x: x["eem_ret"])        # most negative first
wrong_sorted   = sorted(wrong_results, key=lambda x: x["eem_ret"], reverse=True)  # most positive first

print()
print("TOP 3 CORRECT EXCLUSIONS (bearish signal saved the most):")
for i, r in enumerate(correct_sorted[:3], 1):
    print(f"  {i}. {r['start'].date()} → {r['end'].date()}  "
          f"EEM returned {r['eem_ret']:+.1f}%  ({r['n_months']} months)")

print()
print("TOP 3 WRONG EXCLUSIONS (bearish signal missed the most upside):")
for i, r in enumerate(wrong_sorted[:3], 1):
    print(f"  {i}. {r['start'].date()} → {r['end'].date()}  "
          f"EEM returned {r['eem_ret']:+.1f}%  ({r['n_months']} months)")

# Summary statistics
if results:
    all_rets  = [r["eem_ret"] for r in results]
    n_correct = sum(r["correct"] for r in results)
    n_wrong   = len(results) - n_correct
    print(f"\nSummary:")
    print(f"  Correct exclusions  : {n_correct} / {len(results)}  "
          f"({n_correct/len(results)*100:.1f}%)")
    print(f"  Wrong exclusions    : {n_wrong} / {len(results)}  "
          f"({n_wrong/len(results)*100:.1f}%)")
    print(f"  Avg EEM return during bearish streaks: {np.mean(all_rets):+.1f}%")
    print(f"  Median EEM return during bearish streaks: {np.median(all_rets):+.1f}%")

print()
print(SEP)
print("END OF ANALYSIS")
print(SEP)
