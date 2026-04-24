"""
Marginal Portfolio Variance Contribution (MCPV) — C factor replacement test.

Compares two correlation estimation methods:
  A  Pairwise:   average pairwise Pearson correlation with all 16 sleeve
                 candidates (current default).
  B  Portfolio:  Pearson correlation with the equal-weight sleeve portfolio
                 (MCPV — marginal contribution to portfolio variance).

For an equal-weight portfolio of N assets, MCPV_i = Cov(r_i, r_p) which is
proportional to the row mean of the covariance matrix. Computing
corr(r_i, r_p) with r_p = (1/N) Σ_j r_j gives the normalised, scale-free
version.  Lower value = more diversifying = higher C rank (same direction as
pairwise).

Academic basis:
* Sharpe (1964) CAPM / Merton (1972): marginal variance contribution is the
  correct measure of an asset's risk within a portfolio.
* Keller & Butler (2014) EAA: replaced pairwise correlation with correlation
  to the equal-weight portfolio of candidates.
* Ang & Chen (2002): pairwise correlations collapse to 1.0 in crises; the
  portfolio correlation is comparatively more stable.

IS period: 2004-01-01 → 2017-12-31
Walk-forward: 7 two-year OOS windows spanning 2010–2023.
"""

import sys
sys.path.insert(0, '/Users/GualyCr/Desktop/AMAAM/amaam/.claude/worktrees/reverent-mayer')

from pathlib import Path
from copy import deepcopy

import numpy as np
import pandas as pd

from config.default_config import ModelConfig
from config.etf_universe import MAIN_SLEEVE_TICKERS
from src.backtest.engine import run_backtest
from src.data.loader import load_validated_data
from src.factors.correlation import (
    compute_correlation_all_assets,
    compute_portfolio_correlation_all_assets,
)

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

DATA_DIR = Path(
    '/Users/GualyCr/Desktop/AMAAM/amaam/.claude/worktrees/reverent-mayer/data/processed'
)
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
# Configs
# ---------------------------------------------------------------------------

cfg_a = ModelConfig()
cfg_a.correlation_method = "pairwise"     # rolling 126-day average pairwise (default)
cfg_a.correlation_lookback = 126

cfg_b = ModelConfig()
cfg_b.correlation_method = "portfolio_all"  # MCPV: corr with equal-weight portfolio
cfg_b.correlation_lookback = 126

# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def ann_ret(r: pd.Series) -> float:
    """Annualised geometric return from a monthly return series."""
    if len(r) < 2:
        return float("nan")
    return (1.0 + r).prod() ** (12.0 / len(r)) - 1.0


def ann_vol(r: pd.Series) -> float:
    """Annualised volatility from a monthly return series."""
    if len(r) < 2:
        return float("nan")
    return r.std() * np.sqrt(12.0)


def sharpe(r: pd.Series) -> float:
    """Annualised Sharpe (no risk-free adjustment, for relative comparison)."""
    v = ann_vol(r)
    return ann_ret(r) / v if (v > 0 and not np.isnan(v)) else float("nan")


def mdd(r: pd.Series) -> float:
    """Maximum drawdown (negative number)."""
    if len(r) == 0:
        return float("nan")
    eq = (1.0 + r).cumprod()
    return (eq / eq.cummax() - 1.0).min()


def slice_period(r: pd.Series, start: str, end: str) -> pd.Series:
    """Slice a return series to a date range."""
    return r.loc[(r.index >= start) & (r.index <= end)]


# ---------------------------------------------------------------------------
# DIAGNOSTIC: C rankings for Dec 2008 (crisis month)
# ---------------------------------------------------------------------------
# The diagnostic shows whether pairwise and MCPV produce different rankings
# during market stress — the period when diversification matters most.

DIAGNOSTIC_DATE = "2008-12-31"
LOOKBACK = 126

main_tickers_avail = [t for t in MAIN_SLEEVE_TICKERS if t in data_dict]

# Build raw C scores as of Dec 2008 using the full time-series methods.
corr_pairwise_df = compute_correlation_all_assets(
    {t: data_dict[t] for t in main_tickers_avail},
    main_tickers_avail,
    LOOKBACK,
)
corr_portfolio_df = compute_portfolio_correlation_all_assets(
    {t: data_dict[t] for t in main_tickers_avail},
    main_tickers_avail,
    LOOKBACK,
)

# Find the last available date at or before Dec 2008.
diag_dates_pw = corr_pairwise_df.index[corr_pairwise_df.index <= DIAGNOSTIC_DATE]
diag_dates_po = corr_portfolio_df.index[corr_portfolio_df.index <= DIAGNOSTIC_DATE]

if len(diag_dates_pw) > 0 and len(diag_dates_po) > 0:
    diag_date_pw = diag_dates_pw[-1]
    diag_date_po = diag_dates_po[-1]

    pw_scores = corr_pairwise_df.loc[diag_date_pw]
    po_scores = corr_portfolio_df.loc[diag_date_po]

    # Rank: higher C score = higher correlation = LOWER rank (ascending=False means
    # rank 1 = highest C, which in the engine maps to lowest TRank contribution).
    # We want to show rank such that rank 1 = most diversifying (lowest C score),
    # consistent with how the engine uses rC (ascending=False → rank 1 = highest C
    # value, then TRank is ADDED, so lower rank number = higher TRank contribution).
    # Display convention: rank 1 = most diversifying = lowest C score.
    pw_rank = pw_scores.rank(ascending=True).astype(int)   # rank 1 = lowest corr = most diversifying
    po_rank = po_scores.rank(ascending=True).astype(int)

    diag_ok = True
else:
    diag_ok = False

# ---------------------------------------------------------------------------
# Run full-period backtests
# ---------------------------------------------------------------------------

print("Running backtest A: Pairwise 126-day ...", flush=True)
res_a = run_backtest(data_dict, cfg_a)

print("Running backtest B: Portfolio MCPV   ...", flush=True)
res_b = run_backtest(data_dict, cfg_b)

ra = res_a.monthly_returns
rb = res_b.monthly_returns

# ---------------------------------------------------------------------------
# IS slices
# ---------------------------------------------------------------------------

ra_is = slice_period(ra, IS_START, IS_END)
rb_is = slice_period(rb, IS_START, IS_END)

sh_a_is = sharpe(ra_is)
sh_b_is = sharpe(rb_is)

# ---------------------------------------------------------------------------
# Walk-forward OOS metrics
# ---------------------------------------------------------------------------

def wf_metrics(r: pd.Series, start: str, end: str):
    """Return (sharpe, ann_ret, mdd) for a slice of the return series."""
    s = slice_period(r, start, end)
    return sharpe(s), ann_ret(s), mdd(s)


wf_results = {}
for label, start, end in WF_WINDOWS:
    wf_results[(label, "A")] = wf_metrics(ra, start, end)
    wf_results[(label, "B")] = wf_metrics(rb, start, end)

# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def summary_stats(method: str):
    """Return (mean, median, std, list_of_sharpes) for a method key."""
    sharpes = [wf_results[(lbl, method)][0] for lbl, _, _ in WF_WINDOWS]
    clean = [s for s in sharpes if not np.isnan(s)]
    mean   = np.mean(clean)   if clean else float("nan")
    median = np.median(clean) if clean else float("nan")
    std    = np.std(clean, ddof=1) if len(clean) > 1 else float("nan")
    return mean, median, std, sharpes


def win_rate(base_sharpes, chal_sharpes):
    """Fraction of WF windows where challenger > baseline (ignoring NaN pairs)."""
    pairs = [(b, c) for b, c in zip(base_sharpes, chal_sharpes)
             if not (np.isnan(b) or np.isnan(c))]
    if not pairs:
        return float("nan")
    return sum(1 for b, c in pairs if c > b) / len(pairs)


def avg_rank(methods_sharpes_per_window):
    """Average rank per method (rank 1 = highest Sharpe). NaN → treated as worst."""
    n_methods = len(methods_sharpes_per_window)
    n_windows = len(WF_WINDOWS)
    ranks = [[] for _ in range(n_methods)]
    for w in range(n_windows):
        row = [methods_sharpes_per_window[m][w] for m in range(n_methods)]
        sorted_idx = sorted(
            range(n_methods),
            key=lambda i: row[i] if not np.isnan(row[i]) else -1e9,
            reverse=True,
        )
        for r_pos, m_idx in enumerate(sorted_idx):
            ranks[m_idx].append(r_pos + 1)
    return [np.mean(r) if r else float("nan") for r in ranks]


mean_a, med_a, std_a, sharpes_a = summary_stats("A")
mean_b, med_b, std_b, sharpes_b = summary_stats("B")

wr_b = win_rate(sharpes_a, sharpes_b)

avg_ranks = avg_rank([sharpes_a, sharpes_b])

# ---------------------------------------------------------------------------
# Print report
# ---------------------------------------------------------------------------

print()
print("=" * 72)
print("  MARGINAL PORTFOLIO VARIANCE CONTRIBUTION TEST")
print("=" * 72)
print()
print("What we're testing:")
print("  A  Pairwise:   avg correlation to all 16 sleeve candidates (current)")
print("  B  Portfolio:  correlation with equal-weight sleeve portfolio (MCPV)")
print()
print("Academic basis for B:")
print("  MCPV_i = Cov(r_i, r_p) = row mean of covariance matrix (Sharpe 1964).")
print("  corr(r_i, (1/N) Σ_j r_j) is the normalised, scale-free version.")
print("  Keller & Butler (2014) EAA use the same principle.")

# ── Diagnostic ────────────────────────────────────────────────────────────
print()
diag_header = f"DIAGNOSTIC — C rankings for Dec 2008 (crisis month, date used: {diag_date_pw.date() if diag_ok else 'N/A'}):"
print(diag_header)
print()

if diag_ok:
    # Build a combined table, sorted by pairwise rank.
    diag_data = []
    for t in main_tickers_avail:
        pw_s = pw_scores.get(t, float("nan"))
        po_s = po_scores.get(t, float("nan"))
        pw_r = int(pw_rank.get(t, 0))
        po_r = int(po_rank.get(t, 0))
        delta = po_r - pw_r
        diag_data.append((t, pw_s, pw_r, po_s, po_r, delta))

    # Sort by pairwise rank ascending (rank 1 first = most diversifying).
    diag_data.sort(key=lambda x: x[2])

    hdr_d = (
        f"{'Ticker':<8} {'Pairwise C':>11} {'PW Rank':>8} "
        f"{'Portfolio C':>12} {'Port Rank':>10} {'Δ rank':>7}"
    )
    print(hdr_d)
    print("-" * len(hdr_d))
    for t, pw_s, pw_r, po_s, po_r, delta in diag_data:
        delta_str = f"{delta:+d}" if delta != 0 else "  0"
        flag = "  *" if abs(delta) >= 3 else ""
        print(
            f"{t:<8} {pw_s:>11.4f} {pw_r:>8} "
            f"{po_s:>12.4f} {po_r:>10} {delta_str:>7}{flag}"
        )

    large_changes = [(t, d) for t, _, _, _, _, d in diag_data if abs(d) >= 3]
    print()
    if large_changes:
        print(f"  * Assets with |Δ rank| >= 3 (meaningful re-ranking): "
              f"{', '.join(t for t, _ in large_changes)}")
    else:
        print("  No assets re-ranked by 3 or more positions (methods agree closely).")

    # Summary of agreement.
    deltas = [abs(d) for _, _, _, _, _, d in diag_data]
    spearman = pd.Series(
        [pw_r for _, _, pw_r, _, _, _ in diag_data]
    ).corr(pd.Series([po_r for _, _, _, _, po_r, _ in diag_data]),
           method="spearman")
    print(f"  Rank correlation (Spearman) pairwise vs portfolio: {spearman:.3f}")
    print(f"  Mean |Δ rank|: {np.mean(deltas):.2f}   Max |Δ rank|: {max(deltas)}")
else:
    print("  Diagnostic not available: insufficient data before Dec 2008.")

# ── IS performance ─────────────────────────────────────────────────────────
print()
print(f"IS PERFORMANCE ({IS_START[:4]}–{IS_END[:4]}):")
hdr = f"{'Method':<16} {'Sharpe':>8} {'Ann.Ret':>9} {'Ann.Vol':>9} {'MDD':>10} {'ΔSharpe':>9}"
print(hdr)
print("-" * len(hdr))

for label, r_is, sh_is in [
    ("A Pairwise",   ra_is, sh_a_is),
    ("B Portfolio",  rb_is, sh_b_is),
]:
    delta = f"{sh_is - sh_a_is:+.3f}" if label.startswith("B") else "    —"
    print(
        f"{label:<16} {sh_is:>8.3f} {ann_ret(r_is)*100:>8.2f}% "
        f"{ann_vol(r_is)*100:>8.2f}% {mdd(r_is)*100:>9.2f}% {delta:>9}"
    )

# ── Walk-forward OOS Sharpe ────────────────────────────────────────────────
print()
print("WALK-FORWARD OOS SHARPE:")
wf_hdr = (
    f"{'Window':<7} {'Period':<13} {'A Pairwise':>12} "
    f"{'B Portfolio':>13} {'Best':>10}"
)
print(wf_hdr)
print("-" * len(wf_hdr))

for label, start, end in WF_WINDOWS:
    sa = wf_results[(label, "A")][0]
    sb = wf_results[(label, "B")][0]

    vals = {"A Pairwise": sa, "B Portfolio": sb}
    valid = {k: v for k, v in vals.items() if not np.isnan(v)}
    best = max(valid, key=valid.get) if valid else "n/a"

    period = f"{start[:4]}–{end[:4]}"
    print(
        f"{label:<7} {period:<13} {sa:>12.3f} {sb:>13.3f} {best:>10}"
    )

# ── Walk-forward OOS Ann.Ret ──────────────────────────────────────────────
print()
print("WALK-FORWARD OOS ANN.RET:")
ar_hdr = (
    f"{'Window':<7} {'Period':<13} {'A Pairwise':>12} {'B Portfolio':>13}"
)
print(ar_hdr)
print("-" * len(ar_hdr))

for label, start, end in WF_WINDOWS:
    ara = wf_results[(label, "A")][1]
    arb = wf_results[(label, "B")][1]
    period = f"{start[:4]}–{end[:4]}"
    print(
        f"{label:<7} {period:<13} {ara*100:>11.2f}% {arb*100:>12.2f}%"
    )

# ── Walk-forward OOS MDD ──────────────────────────────────────────────────
print()
print("WALK-FORWARD OOS MDD:")
mdd_hdr = (
    f"{'Window':<7} {'Period':<13} {'A Pairwise':>12} {'B Portfolio':>13}"
)
print(mdd_hdr)
print("-" * len(mdd_hdr))

for label, start, end in WF_WINDOWS:
    mda = wf_results[(label, "A")][2]
    mdb = wf_results[(label, "B")][2]
    period = f"{start[:4]}–{end[:4]}"
    print(
        f"{label:<7} {period:<13} {mda*100:>11.2f}% {mdb*100:>12.2f}%"
    )

# ── Summary ────────────────────────────────────────────────────────────────
print()
print("SUMMARY:")
sum_hdr = f"{'Metric':<24} {'A Pairwise':>12} {'B Portfolio':>13}"
print(sum_hdr)
print("-" * len(sum_hdr))

print(f"{'Mean OOS Sharpe':<24} {mean_a:>12.3f} {mean_b:>13.3f}")
print(f"{'Median OOS Sharpe':<24} {med_a:>12.3f} {med_b:>13.3f}")
print(f"{'Sharpe Std Dev':<24} {std_a:>12.3f} {std_b:>13.3f}")
wr_str = f"{wr_b*100:.1f}%" if not np.isnan(wr_b) else "  n/a"
print(f"{'Win rate vs base':<24} {'   —':>12} {wr_str:>13}")
print(f"{'Avg rank':<24} {avg_ranks[0]:>12.1f} {avg_ranks[1]:>13.1f}")

# ── Verdict ────────────────────────────────────────────────────────────────
print()

b_wins_oos = mean_b > mean_a
b_wins_is  = sh_b_is > sh_a_is

if b_wins_oos and b_wins_is:
    verdict = (
        f"MCPV (Portfolio) wins both IS and OOS. "
        f"Mean OOS Sharpe: B {mean_b:.3f} vs A {mean_a:.3f} "
        f"({wr_b*100:.0f}% WF win rate). "
        f"IS Sharpe: B {sh_b_is:.3f} vs A {sh_a_is:.3f}. "
        f"The all-assets portfolio correlation is a superior C factor: "
        f"it is theoretically grounded (MCPV), more statistically efficient "
        f"(one correlation vs N-1 pairwise), and improves portfolio performance."
    )
elif b_wins_oos and not b_wins_is:
    verdict = (
        f"MCPV wins OOS (mean Sharpe: B {mean_b:.3f} vs A {mean_a:.3f}, "
        f"{wr_b*100:.0f}% WF win rate) but not IS "
        f"(B {sh_b_is:.3f} vs A {sh_a_is:.3f}). "
        f"OOS improvement with IS neutrality is a positive signal — "
        f"the MCPV method generalises better out-of-sample."
    )
elif not b_wins_oos and b_wins_is:
    verdict = (
        f"MCPV wins IS (B {sh_b_is:.3f} vs A {sh_a_is:.3f}) but not OOS "
        f"(mean Sharpe: B {mean_b:.3f} vs A {mean_a:.3f}, "
        f"{wr_b*100:.0f}% WF win rate). "
        f"IS-only improvement suggests in-sample fitting; pairwise remains "
        f"the safer default."
    )
else:
    verdict = (
        f"Pairwise wins (mean OOS Sharpe: A {mean_a:.3f} vs B {mean_b:.3f}; "
        f"IS Sharpe: A {sh_a_is:.3f} vs B {sh_b_is:.3f}). "
        f"Despite the theoretical advantage of MCPV, average pairwise "
        f"correlation provides equal or better performance in this universe. "
        f"The C factor weight is low (0.10), limiting the impact of any "
        f"C method change on overall strategy performance."
    )

print(f"VERDICT: {verdict}")
print()
print("=" * 72)
print("Done.")
