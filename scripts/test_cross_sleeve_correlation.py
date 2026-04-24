"""
Cross-sleeve correlation penalty test.

Evaluates whether penalising main-sleeve assets for co-moving with the
hedging sleeve improves IS and OOS performance vs the baseline pairwise-only
correlation method.

Configs tested
--------------
A: pairwise only (baseline)
B: cross_sleeve, lambda=0.5   -> C_adj = C_within + 0.5 * C_cross
C: cross_sleeve, lambda=1.0   -> C_adj = C_within + 1.0 * C_cross

IS  : 2004-01-01 → 2017-12-31
WF windows (OOS):
    WF1 2010-2011, WF2 2012-2013, WF3 2014-2015, WF4 2016-2017
    WF5 2018-2019, WF6 2020-2021, WF7 2022-2023
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from config.default_config import ModelConfig
from config.etf_universe import (
    HEDGING_SLEEVE_TICKERS,
    MAIN_SLEEVE_TICKERS,
)
from src.backtest.engine import run_backtest
from src.data.loader import load_validated_data
from src.factors.correlation import (
    compute_correlation_all_assets,
    compute_cross_sleeve_correlation,
)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

# Processed CSVs may live in the worktree or the main repo; check both.
_WORKTREE_DATA = Path(ROOT) / "data" / "processed"
_MAIN_DATA     = Path("/Users/GualyCr/Desktop/AMAAM/amaam/data/processed")
DATA_DIR = _WORKTREE_DATA if _WORKTREE_DATA.exists() and any(_WORKTREE_DATA.iterdir()) else _MAIN_DATA

if not DATA_DIR.exists():
    print(f"ERROR: data directory not found at {DATA_DIR}")
    sys.exit(1)

data_dict: Dict[str, pd.DataFrame] = load_validated_data(DATA_DIR)
print(f"Loaded data for {len(data_dict)} tickers from {DATA_DIR}.")

# ---------------------------------------------------------------------------
# Walk-forward windows
# ---------------------------------------------------------------------------

WF_WINDOWS: List[Tuple[str, str, str]] = [
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
# Base config (shared settings across all three runs)
# ---------------------------------------------------------------------------

BASE_CFG = ModelConfig(
    backtest_start=IS_START,
    backtest_end=IS_END,
    momentum_blend=True,
    correlation_method="pairwise",   # overridden per config
    cross_sleeve_lambda=0.5,         # overridden per config C
)

CONFIGS = {
    "A": replace(BASE_CFG, correlation_method="pairwise"),
    "B": replace(BASE_CFG, correlation_method="cross_sleeve", cross_sleeve_lambda=0.5),
    "C": replace(BASE_CFG, correlation_method="cross_sleeve", cross_sleeve_lambda=1.0),
}

LABELS = {
    "A": "Pairwise",
    "B": "Cross λ=0.5",
    "C": "Cross λ=1.0",
}

# ---------------------------------------------------------------------------
# Helper: extract metrics from a BacktestResult
# ---------------------------------------------------------------------------

def _metrics(result) -> Dict[str, float]:
    return {
        "sharpe":  result.metrics.get("Sharpe Ratio",          float("nan")),
        "ret":     result.metrics.get("Annualized Return",     float("nan")),
        "vol":     result.metrics.get("Annualized Volatility", float("nan")),
        "mdd":     result.metrics.get("Max Drawdown",          float("nan")),
    }


def _run_window(cfg: ModelConfig, start: str, end: str) -> Dict[str, float]:
    """Run a single backtest window; returns metrics dict."""
    c = replace(cfg, backtest_start=start, backtest_end=end)
    res = run_backtest(data_dict, c)
    return _metrics(res)


# ---------------------------------------------------------------------------
# DIAGNOSTIC: Dec 2008
# ---------------------------------------------------------------------------

DIAG_DATE_STR = "2008-12-31"

def _diagnostic_dec2008() -> None:
    """
    For Dec 2008 (GFC), show C_within, C_cross, C_adj for each main asset.
    Hedging sleeve at this point holds TLT+GLD (plus others).
    """
    LOOKBACK = BASE_CFG.correlation_lookback  # 126 days

    main = [t for t in MAIN_SLEEVE_TICKERS if t in data_dict]
    hedge = [t for t in HEDGING_SLEEVE_TICKERS if t in data_dict]

    # Find last trading date in Dec 2008
    sample_ticker = main[0]
    dates = data_dict[sample_ticker].index
    dec2008 = dates[(dates >= "2008-12-01") & (dates <= "2008-12-31")]
    if len(dec2008) == 0:
        print("  [WARNING] No trading dates found in Dec 2008 for diagnostic.")
        return
    diag_date = dec2008[-1]

    # Compute within-sleeve pairwise C
    c_within_df = compute_correlation_all_assets(
        {t: data_dict[t] for t in main}, main, LOOKBACK
    )
    # Compute cross-sleeve C
    c_cross_df = compute_cross_sleeve_correlation(data_dict, main, hedge, LOOKBACK)

    # Snap to diagnostic date (last valid row at or before date)
    def snap(df: pd.DataFrame, date: pd.Timestamp) -> pd.Series:
        sub = df.loc[:date].dropna(how="all")
        if sub.empty:
            return pd.Series(dtype=float)
        return sub.iloc[-1]

    c_within = snap(c_within_df, diag_date)
    c_cross  = snap(c_cross_df,  diag_date)

    c_adj_05 = c_within + 0.5 * c_cross
    c_adj_10 = c_within + 1.0 * c_cross

    # Build table
    rows = []
    for t in main:
        rows.append({
            "Ticker":       t,
            "C_within":     c_within.get(t, float("nan")),
            "C_cross":      c_cross.get(t, float("nan")),
            "C_adj(λ=0.5)": c_adj_05.get(t, float("nan")),
            "C_adj(λ=1.0)": c_adj_10.get(t, float("nan")),
        })
    tbl = pd.DataFrame(rows).set_index("Ticker")
    tbl_sorted = tbl.sort_values("C_cross", ascending=False)

    print(f"DIAGNOSTIC (Dec 2008 — last valid date used: {diag_date.date()}):")
    print(f"  Hedging sleeve: {', '.join(hedge)}")
    print()
    print(f"  {'Ticker':<8}  {'C_within':>8}  {'C_cross':>8}  {'C_adj(λ=0.5)':>13}  {'C_adj(λ=1.0)':>13}")
    print("  " + "-" * 60)

    # Most correlated with hedges (top 5)
    print("  Most correlated with hedges (penalised most):")
    for t, row in tbl_sorted.head(5).iterrows():
        print(
            f"  {t:<8}  {row['C_within']:>8.4f}  {row['C_cross']:>8.4f}"
            f"  {row['C_adj(λ=0.5)']:>13.4f}  {row['C_adj(λ=1.0)']:>13.4f}"
        )
    print()

    # Least correlated with hedges (bottom 5)
    print("  Least correlated with hedges (penalised least):")
    for t, row in tbl_sorted.tail(5).iterrows():
        print(
            f"  {t:<8}  {row['C_within']:>8.4f}  {row['C_cross']:>8.4f}"
            f"  {row['C_adj(λ=0.5)']:>13.4f}  {row['C_adj(λ=1.0)']:>13.4f}"
        )
    print()

    # Full table
    print("  Full table (sorted by C_cross descending):")
    print(f"  {'Ticker':<8}  {'C_within':>8}  {'C_cross':>8}  {'C_adj(λ=0.5)':>13}  {'C_adj(λ=1.0)':>13}")
    print("  " + "-" * 60)
    for t, row in tbl_sorted.iterrows():
        print(
            f"  {t:<8}  {row['C_within']:>8.4f}  {row['C_cross']:>8.4f}"
            f"  {row['C_adj(λ=0.5)']:>13.4f}  {row['C_adj(λ=1.0)']:>13.4f}"
        )
    print()


# ---------------------------------------------------------------------------
# IS performance
# ---------------------------------------------------------------------------

def _run_is() -> Dict[str, Dict]:
    print("Running IS (2004–2017)…")
    results = {}
    for key, cfg in CONFIGS.items():
        print(f"  Config {key} ({LABELS[key]})…")
        results[key] = _run_window(cfg, IS_START, IS_END)
    return results


# ---------------------------------------------------------------------------
# Walk-forward OOS
# ---------------------------------------------------------------------------

def _run_wf() -> Dict[str, List[Dict]]:
    """Return {config_key: [metrics per WF window]}"""
    wf_results: Dict[str, List[Dict]] = {k: [] for k in CONFIGS}
    for wf_label, start, end in WF_WINDOWS:
        print(f"  {wf_label} ({start[:4]}–{end[:4]})…")
        for key, cfg in CONFIGS.items():
            m = _run_window(cfg, start, end)
            m["wf"] = wf_label
            m["period"] = f"{start[:4]}-{end[2:4]}"
            wf_results[key].append(m)
    return wf_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("=== CROSS-SLEEVE CORRELATION PENALTY TEST ===")
    print("=" * 60)
    print()
    print("What we're testing:")
    print("  A  Pairwise only:          standard within-sleeve correlation (baseline)")
    print("  B  Cross-sleeve λ=0.5:     C_adj = C_within + 0.5 * C_cross")
    print("  C  Cross-sleeve λ=1.0:     C_adj = C_within + 1.0 * C_cross")
    print()

    # ── Diagnostic ─────────────────────────────────────────────────────────
    _diagnostic_dec2008()

    # ── IS ─────────────────────────────────────────────────────────────────
    is_res = _run_is()

    print()
    print("IS PERFORMANCE (2004–2017):")
    hdr = f"{'Method':<20}  {'Sharpe':>7}  {'Ann.Ret':>8}  {'Ann.Vol':>8}  {'MDD':>8}  {'ΔSharpe':>8}"
    print(hdr)
    print("-" * 72)
    base_sharpe = is_res["A"]["sharpe"]
    for key in ["A", "B", "C"]:
        m = is_res[key]
        label = f"{key} {LABELS[key]}"
        ds = f"{m['sharpe'] - base_sharpe:+.3f}" if key != "A" else "—"
        print(
            f"{label:<20}  {m['sharpe']:>7.3f}  {m['ret']*100:>7.2f}%  "
            f"{m['vol']*100:>7.2f}%  {m['mdd']*100:>7.2f}%  {ds:>8}"
        )
    print()

    # ── Walk-forward ───────────────────────────────────────────────────────
    print("Running Walk-Forward OOS…")
    wf_res = _run_wf()

    print()
    print("WALK-FORWARD OOS SHARPE:")
    hdr2 = (
        f"{'Window':<6}  {'Period':<9}  {'A Pairwise':>10}  "
        f"{'B λ=0.5':>8}  {'C λ=1.0':>8}  {'Best':<12}"
    )
    print(hdr2)
    print("-" * 60)

    for i, (wf_label, start, end) in enumerate(WF_WINDOWS):
        mA = wf_res["A"][i]["sharpe"]
        mB = wf_res["B"][i]["sharpe"]
        mC = wf_res["C"][i]["sharpe"]
        period = f"{start[:4]}-{end[2:4]}"
        best_val = max(mA, mB, mC)
        best_lbl = (
            "A Pairwise"  if best_val == mA else
            "B λ=0.5"     if best_val == mB else
            "C λ=1.0"
        )
        print(
            f"{wf_label:<6}  {period:<9}  {mA:>10.3f}  "
            f"{mB:>8.3f}  {mC:>8.3f}  {best_lbl:<12}"
        )

    # ── Summary ────────────────────────────────────────────────────────────
    print()
    print("SUMMARY:")
    hdr3 = f"{'Metric':<22}  {'A Pairwise':>10}  {'B λ=0.5':>8}  {'C λ=1.0':>8}"
    print(hdr3)
    print("-" * 56)

    sharpes = {k: np.array([m["sharpe"] for m in wf_res[k]]) for k in CONFIGS}
    base_s = sharpes["A"]

    def fmt(arr: np.ndarray) -> str:
        return f"{np.nanmean(arr):>8.3f}"

    # Mean OOS Sharpe
    print(
        f"{'Mean OOS Sharpe':<22}  {fmt(sharpes['A']):>10}  "
        f"{fmt(sharpes['B']):>8}  {fmt(sharpes['C']):>8}"
    )
    # Median OOS Sharpe
    print(
        f"{'Median OOS Sharpe':<22}  {np.nanmedian(sharpes['A']):>10.3f}  "
        f"{np.nanmedian(sharpes['B']):>8.3f}  {np.nanmedian(sharpes['C']):>8.3f}"
    )
    # Sharpe Std Dev
    print(
        f"{'Sharpe Std Dev':<22}  {np.nanstd(sharpes['A']):>10.3f}  "
        f"{np.nanstd(sharpes['B']):>8.3f}  {np.nanstd(sharpes['C']):>8.3f}"
    )
    # Win rate vs A
    n = len(WF_WINDOWS)
    wr_b = np.nansum(sharpes["B"] > sharpes["A"]) / n * 100
    wr_c = np.nansum(sharpes["C"] > sharpes["A"]) / n * 100
    print(
        f"{'Win rate vs base':<22}  {'—':>10}  "
        f"{wr_b:>7.1f}%  {wr_c:>7.1f}%"
    )
    # Avg rank (1 = best)
    ranks = np.zeros((len(WF_WINDOWS), 3))
    for i in range(len(WF_WINDOWS)):
        vals = [sharpes["A"][i], sharpes["B"][i], sharpes["C"][i]]
        sorted_desc = sorted(range(3), key=lambda x: -vals[x])
        for rank_pos, idx in enumerate(sorted_desc):
            ranks[i, idx] = rank_pos + 1
    avg_ranks = ranks.mean(axis=0)
    print(
        f"{'Avg rank':<22}  {avg_ranks[0]:>10.1f}  "
        f"{avg_ranks[1]:>8.1f}  {avg_ranks[2]:>8.1f}"
    )
    print()

    # ── Verdict ────────────────────────────────────────────────────────────
    best_mean = max(np.nanmean(sharpes[k]) for k in ["A", "B", "C"])
    best_cfg = [k for k in ["A", "B", "C"] if abs(np.nanmean(sharpes[k]) - best_mean) < 1e-9][0]

    print("VERDICT:")
    mean_a = np.nanmean(sharpes["A"])
    mean_b = np.nanmean(sharpes["B"])
    mean_c = np.nanmean(sharpes["C"])
    delta_b = mean_b - mean_a
    delta_c = mean_c - mean_a

    if best_cfg == "A":
        print(
            f"  Cross-sleeve penalty does NOT improve OOS performance. "
            f"Baseline pairwise wins with mean OOS Sharpe {mean_a:.3f}. "
            f"B ({delta_b:+.3f}) and C ({delta_c:+.3f}) underperform. "
            f"The penalty likely over-penalises defensives that provide genuine "
            f"risk-off diversification within the main sleeve, reducing performance."
        )
    elif best_cfg == "B":
        print(
            f"  Cross-sleeve λ=0.5 is the best method. "
            f"Mean OOS Sharpe {mean_b:.3f} vs baseline {mean_a:.3f} "
            f"(delta {delta_b:+.3f}). "
            f"Moderate penalty improves diversification without over-penalising "
            f"defensives. Win rate {wr_b:.1f}%. Avg rank {avg_ranks[1]:.1f}. "
            f"λ=1.0 ({delta_c:+.3f} vs base) is {'better' if delta_c > 0 else 'worse'} "
            f"but {'stronger' if delta_c > delta_b else 'weaker'} than λ=0.5."
        )
    else:  # C
        print(
            f"  Cross-sleeve λ=1.0 is the best method. "
            f"Mean OOS Sharpe {mean_c:.3f} vs baseline {mean_a:.3f} "
            f"(delta {delta_c:+.3f}). "
            f"Full equal-weight penalty maximises OOS Sharpe. Win rate {wr_c:.1f}%. "
            f"Avg rank {avg_ranks[2]:.1f}. Both λ values improve on the baseline "
            f"(B: {delta_b:+.3f}, C: {delta_c:+.3f}), suggesting cross-sleeve "
            f"awareness genuinely improves main-sleeve selection."
        )
    print()


if __name__ == "__main__":
    main()
