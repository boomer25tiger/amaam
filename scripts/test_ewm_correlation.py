"""
Exponentially Weighted Correlation Test.

Compares three correlation estimation methods:
  A  Rolling-126:  current implementation, 126-day equal-weight rolling window
  B  EWM-126:      exponentially weighted, span=126 (same effective lookback, recent-weighted)
  C  EWM-63:       exponentially weighted, span=63 (faster decay, more crisis-responsive)

IS period: 2004-01-01 → 2017-12-31
Walk-forward windows: 7 two-year OOS windows spanning 2010-2023.

The EWM estimator weights recent observations more heavily, reducing the
hard-cliff effect of rolling windows where a single event entering/exiting
the lookback window causes a step change in correlation estimates.  This
test determines whether that property improves ranking quality.
"""

import sys
sys.path.insert(0, '/Users/GualyCr/Desktop/AMAAM/amaam/.claude/worktrees/reverent-mayer')

from pathlib import Path
from copy import deepcopy

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
# Walk-forward windows
# ---------------------------------------------------------------------------

WF_WINDOWS = [
    ("WF1", "2010-01-01", "2011-12-31"),
    ("WF2", "2012-01-01", "2013-12-31"),
    ("WF3", "2014-01-01", "2015-12-31"),
    ("WF4", "2016-01-01", "2017-12-31"),
    ("WF5", "2018-01-01", "2019-12-31"),
    ("WF6", "2020-01-01", "2021-12-31"),
    ("WF7", "2022-01-01", "2023-12-31"),
]

IS_START = "2004-01-01"
IS_END   = "2017-12-31"

# ---------------------------------------------------------------------------
# Configs — only correlation_method and correlation_ewm_span differ
# ---------------------------------------------------------------------------

cfg_a = ModelConfig()
cfg_a.correlation_method = "pairwise"   # rolling-126 (default)
cfg_a.correlation_lookback = 126

cfg_b = ModelConfig()
cfg_b.correlation_method = "ewm"
cfg_b.correlation_ewm_span = 126        # EWM span=126

cfg_c = ModelConfig()
cfg_c.correlation_method = "ewm"
cfg_c.correlation_ewm_span = 63         # EWM span=63 (faster decay)

# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def ann_ret(r: pd.Series) -> float:
    """Annualised geometric return from monthly series."""
    if len(r) < 2:
        return float("nan")
    return (1.0 + r).prod() ** (12.0 / len(r)) - 1.0


def ann_vol(r: pd.Series) -> float:
    """Annualised volatility from monthly series."""
    if len(r) < 2:
        return float("nan")
    return r.std() * np.sqrt(12.0)


def sharpe(r: pd.Series) -> float:
    """Annualised Sharpe (no risk-free rate adjustment for comparability)."""
    v = ann_vol(r)
    return ann_ret(r) / v if (v > 0 and not np.isnan(v)) else float("nan")


def mdd(r: pd.Series) -> float:
    """Maximum drawdown (negative number)."""
    if len(r) == 0:
        return float("nan")
    eq = (1.0 + r).cumprod()
    return (eq / eq.cummax() - 1.0).min()


def slice_period(r: pd.Series, start: str, end: str) -> pd.Series:
    return r.loc[(r.index >= start) & (r.index <= end)]


# ---------------------------------------------------------------------------
# Run full-period backtests (used for IS slice + walk-forward loop)
# ---------------------------------------------------------------------------

print("Running backtest A: Rolling-126 ...", flush=True)
res_a = run_backtest(data_dict, cfg_a)

print("Running backtest B: EWM-126    ...", flush=True)
res_b = run_backtest(data_dict, cfg_b)

print("Running backtest C: EWM-63     ...", flush=True)
res_c = run_backtest(data_dict, cfg_c)

ra = res_a.monthly_returns
rb = res_b.monthly_returns
rc = res_c.monthly_returns

# ---------------------------------------------------------------------------
# IS performance
# ---------------------------------------------------------------------------

ra_is = slice_period(ra, IS_START, IS_END)
rb_is = slice_period(rb, IS_START, IS_END)
rc_is = slice_period(rc, IS_START, IS_END)

sh_a_is = sharpe(ra_is)
sh_b_is = sharpe(rb_is)
sh_c_is = sharpe(rc_is)

# ---------------------------------------------------------------------------
# Walk-forward OOS slices
# ---------------------------------------------------------------------------

def wf_metrics(r: pd.Series, start: str, end: str):
    s = slice_period(r, start, end)
    return sharpe(s), ann_ret(s), mdd(s)


wf_results = {}   # key: (wf_label, method) → (sharpe, ann_ret, mdd)
for label, start, end in WF_WINDOWS:
    wf_results[(label, "A")] = wf_metrics(ra, start, end)
    wf_results[(label, "B")] = wf_metrics(rb, start, end)
    wf_results[(label, "C")] = wf_metrics(rc, start, end)

# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def summary_stats(method: str):
    sharpes = [wf_results[(lbl, method)][0] for lbl, _, _ in WF_WINDOWS]
    sharpes_clean = [s for s in sharpes if not np.isnan(s)]
    mean   = np.mean(sharpes_clean) if sharpes_clean else float("nan")
    median = np.median(sharpes_clean) if sharpes_clean else float("nan")
    std    = np.std(sharpes_clean, ddof=1) if len(sharpes_clean) > 1 else float("nan")
    return mean, median, std, sharpes


def win_rate(base_sharpes, chal_sharpes):
    """Fraction of WF windows where challenger beats base (ignoring NaN pairs)."""
    wins = sum(
        1 for b, c in zip(base_sharpes, chal_sharpes)
        if not (np.isnan(b) or np.isnan(c)) and c > b
    )
    total = sum(
        1 for b, c in zip(base_sharpes, chal_sharpes)
        if not (np.isnan(b) or np.isnan(c))
    )
    return wins / total if total > 0 else float("nan")


def avg_rank(methods_sharpes_per_window):
    """
    Average rank of each method across WF windows.
    Rank 1 = best (highest Sharpe), rank N = worst.
    """
    n_methods = len(methods_sharpes_per_window)
    n_windows = len(WF_WINDOWS)
    ranks = [[] for _ in range(n_methods)]
    for w in range(n_windows):
        row = [methods_sharpes_per_window[m][w] for m in range(n_methods)]
        # Rank: highest sharpe gets rank 1.  NaN treated as worst.
        sorted_idx = sorted(range(n_methods),
                            key=lambda i: row[i] if not np.isnan(row[i]) else -1e9,
                            reverse=True)
        for r_pos, m_idx in enumerate(sorted_idx):
            ranks[m_idx].append(r_pos + 1)
    return [np.mean(r) if r else float("nan") for r in ranks]


mean_a, med_a, std_a, sharpes_a = summary_stats("A")
mean_b, med_b, std_b, sharpes_b = summary_stats("B")
mean_c, med_c, std_c, sharpes_c = summary_stats("C")

wr_b = win_rate(sharpes_a, sharpes_b)
wr_c = win_rate(sharpes_a, sharpes_c)

avg_ranks = avg_rank([sharpes_a, sharpes_b, sharpes_c])

# ---------------------------------------------------------------------------
# Print report
# ---------------------------------------------------------------------------

print()
print("=" * 72)
print("  EXPONENTIALLY WEIGHTED CORRELATION TEST")
print("=" * 72)
print()
print("What we're testing:")
print("  A  Rolling-126:  current implementation, 126-day equal-weight rolling window")
print("  B  EWM-126:      exponentially weighted, span=126 (same effective lookback, recent-weighted)")
print("  C  EWM-63:       exponentially weighted, span=63 (faster decay, more crisis-responsive)")

# IS performance table
print()
print("IS PERFORMANCE (2004–2017):")
hdr = f"{'Method':<14} {'Sharpe':>8} {'Ann.Ret':>9} {'Ann.Vol':>9} {'MDD':>10} {'ΔSharpe':>9}"
print(hdr)
print("-" * len(hdr))

for label, r_is, sh_is in [
    ("A Rolling",   ra_is, sh_a_is),
    ("B EWM-126",   rb_is, sh_b_is),
    ("C EWM-63",    rc_is, sh_c_is),
]:
    delta = f"{sh_is - sh_a_is:+.3f}" if label != "A Rolling" else "   —"
    print(
        f"{label:<14} {sh_is:>8.3f} {ann_ret(r_is)*100:>8.2f}% "
        f"{ann_vol(r_is)*100:>8.2f}% {mdd(r_is)*100:>9.2f}% {delta:>9}"
    )

# Walk-forward OOS Sharpe table
print()
print("WALK-FORWARD OOS SHARPE:")
wf_hdr = f"{'Window':<6} {'Period':<13} {'A Rolling':>11} {'B EWM-126':>11} {'C EWM-63':>10} {'Best':>7}"
print(wf_hdr)
print("-" * len(wf_hdr))

for label, start, end in WF_WINDOWS:
    sa = wf_results[(label, "A")][0]
    sb = wf_results[(label, "B")][0]
    sc = wf_results[(label, "C")][0]

    vals = {"A Rolling": sa, "B EWM-126": sb, "C EWM-63": sc}
    valid = {k: v for k, v in vals.items() if not np.isnan(v)}
    best = max(valid, key=valid.get) if valid else "n/a"

    period = f"{start[:4]}–{end[:4]}"
    print(
        f"{label:<6} {period:<13} {sa:>11.3f} {sb:>11.3f} {sc:>10.3f} {best:>7}"
    )

# Walk-forward OOS MDD table
print()
print("WALK-FORWARD OOS MDD:")
mdd_hdr = f"{'Window':<6} {'Period':<13} {'A Rolling':>11} {'B EWM-126':>11} {'C EWM-63':>10}"
print(mdd_hdr)
print("-" * len(mdd_hdr))

for label, start, end in WF_WINDOWS:
    ma = wf_results[(label, "A")][2]
    mb = wf_results[(label, "B")][2]
    mc = wf_results[(label, "C")][2]
    period = f"{start[:4]}–{end[:4]}"
    print(
        f"{label:<6} {period:<13} {ma*100:>10.2f}% {mb*100:>10.2f}% {mc*100:>9.2f}%"
    )

# Summary
print()
print("SUMMARY:")
sum_hdr = f"{'Metric':<22} {'A Rolling':>11} {'B EWM-126':>11} {'C EWM-63':>10}"
print(sum_hdr)
print("-" * len(sum_hdr))

print(f"{'Mean OOS Sharpe':<22} {mean_a:>11.3f} {mean_b:>11.3f} {mean_c:>10.3f}")
print(f"{'Median OOS Sharpe':<22} {med_a:>11.3f} {med_b:>11.3f} {med_c:>10.3f}")
print(f"{'Sharpe Std Dev':<22} {std_a:>11.3f} {std_b:>11.3f} {std_c:>10.3f}")
print(f"{'Win rate vs base':<22} {'  —':>11} {wr_b*100:>10.1f}% {wr_c*100:>9.1f}%")
print(f"{'Avg rank':<22} {avg_ranks[0]:>11.1f} {avg_ranks[1]:>11.1f} {avg_ranks[2]:>10.1f}")

# Verdict
winner = "A (Rolling-126)"
winner_mean = mean_a
if mean_b > winner_mean:
    winner = "B (EWM-126)"
    winner_mean = mean_b
if mean_c > winner_mean:
    winner = "C (EWM-63)"
    winner_mean = mean_c

ewm_wins_is = (sh_b_is > sh_a_is or sh_c_is > sh_a_is)
both_ewm_win_oos = (mean_b > mean_a and mean_c > mean_a)

if winner == "A (Rolling-126)":
    verdict = (
        f"Rolling-126 wins (mean OOS Sharpe {mean_a:.3f} vs EWM-126 {mean_b:.3f} / "
        f"EWM-63 {mean_c:.3f}).  Equal-weight rolling window is sufficient for the "
        f"pairwise correlation factor at the current parameter settings; the exponential "
        f"decay does not provide a consistent ranking advantage."
    )
elif winner == "B (EWM-126)":
    verdict = (
        f"EWM-126 wins (mean OOS Sharpe {mean_b:.3f} vs Rolling {mean_a:.3f} / "
        f"EWM-63 {mean_c:.3f}).  Matching the effective lookback but applying "
        f"exponential decay improves OOS Sharpe — recent-observation weighting helps "
        f"without the instability of the shorter 63-day span."
    )
else:
    verdict = (
        f"EWM-63 wins (mean OOS Sharpe {mean_c:.3f} vs Rolling {mean_a:.3f} / "
        f"EWM-126 {mean_b:.3f}).  Faster decay (span=63) provides the best ranking "
        f"signal — the correlation factor benefits from higher responsiveness to "
        f"regime shifts such as 2008, 2020, and 2022."
    )

print()
print(f"VERDICT: {verdict}")
print()
print("=" * 72)
print("Done.")
