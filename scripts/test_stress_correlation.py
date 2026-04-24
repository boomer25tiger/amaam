"""
Stress-conditional correlation test: IS + walk-forward validation.

Compares four correlation estimation methods:
  A  Pairwise:      standard 126-day rolling (baseline)
  B  Stress-Vol:    correlation on high-volatility days only
  C  Stress-DD:     correlation on SPY bear market days only
  D  Stress-Blend:  50/50 pairwise + stress-vol

IS period : 2004-01-01 → 2017-12-31
WF windows: 7 two-year windows from 2010 through 2023

Additionally produces a diagnostic for Dec 2008 (crisis month) showing
which methods rank defensive assets higher and cyclicals lower.
"""

import sys
import warnings

sys.path.insert(0, '/Users/GualyCr/Desktop/AMAAM/amaam/.claude/worktrees/reverent-mayer')
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd

from config.default_config import ModelConfig
from src.backtest.engine import run_backtest
from src.data.loader import load_validated_data
from src.factors.correlation import (
    compute_correlation_all_assets,
    compute_stress_correlation_all_assets,
    compute_stress_blend_correlation_all_assets,
)
from config.etf_universe import MAIN_SLEEVE_TICKERS

DATA_DIR = Path('/Users/GualyCr/Desktop/AMAAM/amaam/.claude/worktrees/reverent-mayer/data/processed')

# ---------------------------------------------------------------------------
# Helper: run backtest for a date window and return metrics
# ---------------------------------------------------------------------------

def run_window(
    data_dict: dict,
    method: str,
    start: str,
    end: str,
) -> dict:
    """Run AMAAM backtest for a given correlation method and date window."""
    cfg = ModelConfig(
        backtest_start=start,
        backtest_end=end,
        correlation_method=method,
        correlation_lookback=126,
        stress_vol_multiplier=1.5,
    )
    result = run_backtest(data_dict, cfg)
    return result.metrics


# ---------------------------------------------------------------------------
# Helper: format metrics for display
# ---------------------------------------------------------------------------

def fmt_metrics(metrics: dict) -> tuple:
    """Extract (sharpe, ann_ret_pct, ann_vol_pct, mdd_pct) from metrics dict."""
    sharpe  = metrics.get("Sharpe Ratio", float("nan"))
    ann_ret = metrics.get("Annualized Return", float("nan")) * 100
    ann_vol = metrics.get("Annualized Volatility", float("nan")) * 100
    mdd     = metrics.get("Max Drawdown", float("nan")) * 100
    return sharpe, ann_ret, ann_vol, mdd


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Loading data…")
    data_dict = load_validated_data(DATA_DIR)
    print(f"  Loaded {len(data_dict)} tickers.\n")

    methods = {
        "A": "pairwise",
        "B": "stress_vol",
        "C": "stress_drawdown",
        "D": "stress_blend",
    }
    labels = {
        "A": "Pairwise",
        "B": "Stress-Vol",
        "C": "Stress-DD",
        "D": "Stress-Blend",
    }

    IS_START = "2004-01-01"
    IS_END   = "2017-12-31"

    WF_WINDOWS = [
        ("WF1", "2010-01-01", "2011-12-31"),
        ("WF2", "2012-01-01", "2013-12-31"),
        ("WF3", "2014-01-01", "2015-12-31"),
        ("WF4", "2016-01-01", "2017-12-31"),
        ("WF5", "2018-01-01", "2019-12-31"),
        ("WF6", "2020-01-01", "2021-12-31"),
        ("WF7", "2022-01-01", "2023-12-31"),
    ]

    # -----------------------------------------------------------------------
    # IS backtest
    # -----------------------------------------------------------------------
    print("Running IS backtests (2004–2017)…")
    is_metrics = {}
    for key, method in methods.items():
        print(f"  Method {key} ({method})…")
        is_metrics[key] = run_window(data_dict, method, IS_START, IS_END)

    # -----------------------------------------------------------------------
    # Walk-forward backtests
    # -----------------------------------------------------------------------
    print("Running walk-forward backtests…")
    wf_sharpes: dict[str, dict] = {key: {} for key in methods}
    for wf_label, wf_start, wf_end in WF_WINDOWS:
        print(f"  {wf_label} ({wf_start[:4]}–{wf_end[:4]})…")
        for key, method in methods.items():
            m = run_window(data_dict, method, wf_start, wf_end)
            wf_sharpes[key][wf_label] = m.get("Sharpe Ratio", float("nan"))

    # -----------------------------------------------------------------------
    # DIAGNOSTIC: Dec 2008 rankings under all 4 methods
    # -----------------------------------------------------------------------
    # Compute full-history correlation frames for main sleeve tickers only.
    print("\nBuilding diagnostic correlation frames for Dec 2008…")
    DIAG_DATE = pd.Timestamp("2008-12-31")
    main_tickers = [t for t in MAIN_SLEEVE_TICKERS if t in data_dict]
    LOOKBACK = 126

    corr_frames = {}

    # A — pairwise
    corr_frames["A"] = compute_correlation_all_assets(
        {t: data_dict[t] for t in main_tickers}, main_tickers, LOOKBACK
    )

    # B — stress_vol
    corr_frames["B"] = compute_stress_correlation_all_assets(
        data_dict, main_tickers, LOOKBACK,
        stress_method="vol", vol_multiplier=1.5,
    )

    # C — stress_drawdown
    corr_frames["C"] = compute_stress_correlation_all_assets(
        data_dict, main_tickers, LOOKBACK,
        stress_method="drawdown", vol_multiplier=1.5,
    )

    # D — stress_blend
    corr_frames["D"] = compute_stress_blend_correlation_all_assets(
        data_dict, main_tickers, LOOKBACK, vol_multiplier=1.5,
    )

    def get_ranks_at_date(df: pd.DataFrame, date: pd.Timestamp) -> pd.Series:
        """Return ordinal ranks (1=lowest corr = most diversifying) at date."""
        # Find the nearest available date at or before the requested date.
        available = df.index[df.index <= date]
        if len(available) == 0:
            return pd.Series(dtype=float)
        snap = df.loc[available[-1]]
        # Rank ascending: rank 1 = lowest correlation = most diversifying = best.
        # NaN tickers are ranked last.
        return snap.rank(ascending=True, na_option="bottom").astype(int)

    ranks_at_dec08: dict[str, pd.Series] = {}
    for key in ["A", "B", "C", "D"]:
        ranks_at_dec08[key] = get_ranks_at_date(corr_frames[key], DIAG_DATE)

    # Find the 5 tickers with the largest rank difference between C and A.
    rank_df = pd.DataFrame(ranks_at_dec08)
    rank_df.columns = ["A", "B", "C", "D"]
    rank_df["delta_C_vs_A"] = (rank_df["C"] - rank_df["A"]).abs()
    top5_delta = rank_df.sort_values("delta_C_vs_A", ascending=False).head(5)

    # -----------------------------------------------------------------------
    # Print output
    # -----------------------------------------------------------------------
    print()
    print("=" * 70)
    print("=== STRESS-CONDITIONAL CORRELATION TEST ===")
    print("=" * 70)
    print()
    print("What we're testing:")
    print("  A  Pairwise:      standard 126-day rolling (baseline)")
    print("  B  Stress-Vol:    correlation on high-volatility days only")
    print("  C  Stress-DD:     correlation on SPY bear market days only")
    print("  D  Stress-Blend:  50/50 pairwise + stress-vol")
    print()

    # --- DIAGNOSTIC --------------------------------------------------------
    print("DIAGNOSTIC (Dec 2008 — crisis month):")
    print("Top 5 assets with largest C rank changes vs pairwise:")
    print(f"{'Ticker':<8}  {'A Pairwise':>10}  {'B Stress-Vol':>12}  "
          f"{'C Stress-DD':>11}  {'D Blend':>8}  {'|C-A|':>6}")
    print("-" * 64)
    for ticker, row in top5_delta.iterrows():
        a_r = int(row["A"]) if not np.isnan(row["A"]) else "N/A"
        b_r = int(row["B"]) if not np.isnan(row["B"]) else "N/A"
        c_r = int(row["C"]) if not np.isnan(row["C"]) else "N/A"
        d_r = int(row["D"]) if not np.isnan(row["D"]) else "N/A"
        delta = int(row["delta_C_vs_A"])
        print(f"{ticker:<8}  {str(a_r):>10}  {str(b_r):>12}  "
              f"{str(c_r):>11}  {str(d_r):>8}  {delta:>6}")

    # Show all-ticker rankings as context.
    print()
    print("  Full ranking snapshot (all main-sleeve assets, Dec 2008):")
    print(f"  {'Ticker':<8}  {'A':>4}  {'B':>4}  {'C':>4}  {'D':>4}")
    print("  " + "-" * 30)
    for ticker in sorted(rank_df.index):
        row = rank_df.loc[ticker]
        print(f"  {ticker:<8}  {int(row['A']):>4}  {int(row['B']):>4}  "
              f"{int(row['C']):>4}  {int(row['D']):>4}")
    print()

    # --- IS PERFORMANCE ----------------------------------------------------
    base_sharpe = fmt_metrics(is_metrics["A"])[0]
    print("IS PERFORMANCE (2004–2017):")
    print(
        f"{'Method':<20}  {'Sharpe':>8}  {'Ann.Ret':>8}  "
        f"{'Ann.Vol':>8}  {'MDD':>9}  {'DSharpe':>8}"
    )
    print("-" * 74)
    for key, lbl in labels.items():
        s, r, v, d = fmt_metrics(is_metrics[key])
        delta_s = s - base_sharpe if key != "A" else float("nan")
        delta_str = f"{delta_s:+.3f}" if not np.isnan(delta_s) else "—"
        method_str = f"{key} {lbl}"
        print(
            f"{method_str:<20}  {s:>8.3f}  {r:>7.2f}%  "
            f"{v:>7.2f}%  {d:>8.2f}%  {delta_str:>8}"
        )
    print()

    # --- WALK-FORWARD OOS SHARPE -------------------------------------------
    print("WALK-FORWARD OOS SHARPE:")
    header = (
        f"{'Window':<8}  {'Period':<10}  {'A Pairwise':>10}  "
        f"{'B Stress-Vol':>12}  {'C Stress-DD':>11}  {'D Blend':>8}  {'Best':<12}"
    )
    print(header)
    print("-" * 76)
    for wf_label, wf_start, wf_end in WF_WINDOWS:
        period_str = f"{wf_start[:4]}-{wf_end[2:4]}"
        sharpes = {k: wf_sharpes[k][wf_label] for k in methods}
        best_key = max(
            sharpes, key=lambda k: sharpes[k] if not np.isnan(sharpes[k]) else -999
        )
        best_str = f"{best_key} {labels[best_key]}"
        vals = {k: (f"{sharpes[k]:.3f}" if not np.isnan(sharpes[k]) else "N/A")
                for k in methods}
        print(
            f"{wf_label:<8}  {period_str:<10}  {vals['A']:>10}  "
            f"{vals['B']:>12}  {vals['C']:>11}  {vals['D']:>8}  {best_str:<12}"
        )
    print()

    # --- SUMMARY -----------------------------------------------------------
    print("SUMMARY:")

    def wf_series(key: str) -> np.ndarray:
        return np.array([wf_sharpes[key][wf] for _, wf, _ in
                         [(w, w, e) for w, s, e in WF_WINDOWS]])

    # Rebuild properly.
    def wf_arr(key: str) -> np.ndarray:
        return np.array([wf_sharpes[key][wf_label] for wf_label, _, _ in WF_WINDOWS])

    base_arr = wf_arr("A")

    summary_rows = []
    for key, lbl in labels.items():
        arr = wf_arr(key)
        valid = arr[~np.isnan(arr)]
        mean_s   = np.mean(valid) if len(valid) > 0 else float("nan")
        median_s = np.median(valid) if len(valid) > 0 else float("nan")
        std_s    = np.std(valid, ddof=1) if len(valid) > 1 else float("nan")
        if key == "A":
            win_rate = float("nan")
            avg_rank_val = float("nan")
        else:
            # Win rate: fraction of WF windows where this method beats baseline.
            wins = np.sum(arr > base_arr)
            win_rate = wins / len(arr) * 100
            # Average rank across windows (1 = best of 4 methods that window).
            avg_rank_val = float("nan")
        summary_rows.append((key, lbl, mean_s, median_s, std_s, win_rate))

    # Compute per-window ranks across methods.
    method_keys = list(methods.keys())
    wf_rank_matrix = {}
    for wf_label, _, _ in WF_WINDOWS:
        window_sharpes = {k: wf_sharpes[k][wf_label] for k in method_keys}
        sorted_keys = sorted(
            method_keys,
            key=lambda k: window_sharpes[k] if not np.isnan(window_sharpes[k]) else -999,
            reverse=True,
        )
        for rank_pos, k in enumerate(sorted_keys, start=1):
            wf_rank_matrix.setdefault(k, []).append(rank_pos)

    avg_ranks = {k: np.mean(wf_rank_matrix[k]) for k in method_keys}

    print(
        f"{'Metric':<22}  {'A Pairwise':>10}  {'B Stress-Vol':>12}  "
        f"{'C Stress-DD':>11}  {'D Blend':>8}"
    )
    print("-" * 70)

    # Mean OOS Sharpe.
    row_vals = [f"{np.mean(wf_arr(k)[~np.isnan(wf_arr(k))]):>10.3f}" for k in method_keys]
    print(f"{'Mean OOS Sharpe':<22}  {'  '.join(row_vals)}")

    # Median OOS Sharpe.
    row_vals = [f"{np.median(wf_arr(k)[~np.isnan(wf_arr(k))]):>10.3f}" for k in method_keys]
    print(f"{'Median OOS Sharpe':<22}  {'  '.join(row_vals)}")

    # Sharpe Std Dev.
    row_vals = []
    for k in method_keys:
        arr = wf_arr(k)
        valid = arr[~np.isnan(arr)]
        std = np.std(valid, ddof=1) if len(valid) > 1 else float("nan")
        row_vals.append(f"{std:>10.3f}" if not np.isnan(std) else "       N/A")
    print(f"{'Sharpe Std Dev':<22}  {'  '.join(row_vals)}")

    # Win rate vs baseline.
    row_vals = []
    for k in method_keys:
        if k == "A":
            row_vals.append(f"{'—':>10}")
        else:
            arr = wf_arr(k)
            wins = int(np.sum(arr > base_arr))
            rate = wins / len(arr) * 100
            row_vals.append(f"{rate:>9.1f}%")
    print(f"{'Win rate vs base':<22}  {'  '.join(row_vals)}")

    # Average rank.
    row_vals = [f"{avg_ranks[k]:>10.1f}" for k in method_keys]
    print(f"{'Avg rank':<22}  {'  '.join(row_vals)}")
    print()

    # --- VERDICT -----------------------------------------------------------
    # Determine winner by mean OOS Sharpe.
    mean_oos = {k: np.mean(wf_arr(k)[~np.isnan(wf_arr(k))]) for k in method_keys}
    best_key = max(mean_oos, key=lambda k: mean_oos[k])
    best_lbl = labels[best_key]
    base_mean = mean_oos["A"]
    improvement = mean_oos[best_key] - base_mean

    # Win rates.
    def win_rate(k: str) -> float:
        arr = wf_arr(k)
        return np.sum(arr > base_arr) / len(arr) * 100

    stress_adds_value = any(
        win_rate(k) > 50 for k in ["B", "C", "D"]
    )
    best_stress = max(["B", "C", "D"], key=lambda k: mean_oos[k])
    best_stress_lbl = labels[best_stress]

    print("VERDICT:")
    print(f"  Best method by mean OOS Sharpe: {best_key} ({best_lbl}), "
          f"mean={mean_oos[best_key]:.3f} "
          f"({'+'if improvement >= 0 else ''}{improvement:.3f} vs baseline).")

    if stress_adds_value:
        print(f"  Stress-conditioning ADDS value: {best_stress_lbl} wins {win_rate(best_stress):.0f}% "
              f"of OOS windows vs the pairwise baseline.")
        print(f"  Using only high-stress observations sharpens the diversification")
        print(f"  signal by filtering out noise from low-correlation bull-market days.")
    else:
        print(f"  Stress-conditioning does NOT reliably add value OOS: all stress methods")
        print(f"  win fewer than 50% of walk-forward windows vs the pairwise baseline.")
        print(f"  The additional estimation noise from fewer observations outweighs the")
        print(f"  benefit of focusing on crisis regimes.")

    # Diagnostic note on defensive vs cyclical re-ranking.
    defensive = [t for t in ["GLD", "TLT", "IEF", "XLU", "XLP", "XLV"] if t in rank_df.index]
    cyclical  = [t for t in ["XLF", "XLE", "XLY", "XLK", "EEM"] if t in rank_df.index]
    if defensive and cyclical:
        def_rank_A = rank_df.loc[defensive, "A"].mean()
        def_rank_C = rank_df.loc[defensive, "C"].mean()
        cyc_rank_A = rank_df.loc[cyclical,  "A"].mean()
        cyc_rank_C = rank_df.loc[cyclical,  "C"].mean()
        def_improved = def_rank_C < def_rank_A  # lower rank = more diversifying
        cyc_worsened = cyc_rank_C > cyc_rank_A
        if def_improved and cyc_worsened:
            print(f"  Diagnostic confirms theory: in Dec 2008, Stress-DD ranks defensive assets")
            print(f"  (avg rank {def_rank_C:.1f} vs {def_rank_A:.1f} under pairwise) more favourably")
            print(f"  and cyclicals less favourably ({cyc_rank_C:.1f} vs {cyc_rank_A:.1f}).")
        else:
            print(f"  Diagnostic (Dec 2008): rank shifts do not fully conform to the")
            print(f"  defensive-up/cyclical-down hypothesis; results are mixed.")
    print()


if __name__ == "__main__":
    main()
