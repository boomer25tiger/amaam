"""
Walk-forward validation for top IS trend methods vs sma_ratio baseline.

Expanding IS window, fixed 2-year OOS windows (2010–2023).
7 independent OOS windows; each method is always tested on data it was
never trained on.  This script runs all 4 methods × 1 full-period backtest
each, then slices the monthly returns to each OOS window for metrics.

Methods tested:
    sma_ratio (baseline/current), tsmom, donchian, dual_sma
"""

import sys

sys.path.insert(0, "/Users/GualyCr/Desktop/AMAAM/amaam/.claude/worktrees/reverent-mayer")

from pathlib import Path

import numpy as np
import pandas as pd

from config.default_config import ModelConfig
from src.backtest.engine import run_backtest
from src.data.loader import load_validated_data

# ---------------------------------------------------------------------------
# Helper metrics
# ---------------------------------------------------------------------------

def ann_ret(r: pd.Series) -> float:
    """Annualised arithmetic return from monthly returns."""
    r = r.dropna()
    if len(r) == 0:
        return float("nan")
    return (1 + r.mean()) ** 12 - 1


def ann_vol(r: pd.Series) -> float:
    """Annualised volatility from monthly returns."""
    r = r.dropna()
    if len(r) == 0:
        return float("nan")
    return r.std() * np.sqrt(12)


def sharpe(r: pd.Series) -> float:
    """Annualised Sharpe ratio (no risk-free rate) from monthly returns."""
    r = r.dropna()
    if len(r) < 2:
        return float("nan")
    v = ann_vol(r)
    if v == 0:
        return float("nan")
    return ann_ret(r) / v


def mdd(r: pd.Series) -> float:
    """Maximum drawdown from monthly returns."""
    r = r.dropna()
    if len(r) == 0:
        return float("nan")
    eq = (1 + r).cumprod()
    return float((eq / eq.cummax() - 1).min())


def slice_returns(result, start: str, end: str) -> pd.Series:
    """
    Slice monthly returns to a given period.

    Parameters
    ----------
    result : BacktestResult
    start  : str  e.g. '2010-01'
    end    : str  e.g. '2011-12'
    """
    mr = result.monthly_returns.copy()
    mr.index = mr.index.to_period("M")
    mask = (mr.index >= pd.Period(start, "M")) & (mr.index <= pd.Period(end, "M"))
    return mr[mask]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    data_dir = Path(
        "/Users/GualyCr/Desktop/AMAAM/amaam/.claude/worktrees/reverent-mayer/data/processed"
    )
    data_dict = load_validated_data(data_dir)

    methods = ["sma_ratio", "tsmom", "donchian", "dual_sma"]

    print("Running 4 full-period backtests …")
    all_results: dict = {}
    for method in methods:
        cfg = ModelConfig()
        cfg.trend_method = method
        print(f"  [{method}] …", end=" ", flush=True)
        all_results[method] = run_backtest(data_dict, cfg)
        print("done")

    # Walk-forward windows: (name, IS_start, IS_end, OOS_start, OOS_end)
    wf_windows = [
        ("WF1", "2004-01", "2009-12", "2010-01", "2011-12"),
        ("WF2", "2004-01", "2011-12", "2012-01", "2013-12"),
        ("WF3", "2004-01", "2013-12", "2014-01", "2015-12"),
        ("WF4", "2004-01", "2015-12", "2016-01", "2017-12"),
        ("WF5", "2004-01", "2017-12", "2018-01", "2019-12"),
        ("WF6", "2004-01", "2019-12", "2020-01", "2021-12"),
        ("WF7", "2004-01", "2021-12", "2022-01", "2023-12"),
    ]

    # Build result tables: sharpe_tbl[method][window_name] = value
    sharpe_tbl: dict = {m: {} for m in methods}
    ret_tbl:    dict = {m: {} for m in methods}
    mdd_tbl:    dict = {m: {} for m in methods}

    for wname, _is0, _is1, oos0, oos1 in wf_windows:
        for method in methods:
            r = slice_returns(all_results[method], oos0, oos1)
            sharpe_tbl[method][wname] = sharpe(r)
            ret_tbl[method][wname]    = ann_ret(r)
            mdd_tbl[method][wname]    = mdd(r)

    # Ranking per window (1=best Sharpe)
    rank_tbl: dict = {m: {} for m in methods}
    for wname, *_ in wf_windows:
        window_sharpes = {m: sharpe_tbl[m][wname] for m in methods}
        sorted_methods = sorted(methods, key=lambda m: window_sharpes[m], reverse=True)
        for rank, method in enumerate(sorted_methods, start=1):
            rank_tbl[method][wname] = rank

    # ---------------------------------------------------------------------------
    # Print results
    # ---------------------------------------------------------------------------

    print()
    print("=" * 78)
    print("  WALK-FORWARD VALIDATION: TREND METHODS")
    print("=" * 78)
    print(f"  Methods: sma_ratio (baseline), tsmom, donchian, dual_sma")
    print("  7 expanding-window OOS periods, 2 years each (2010–2023)")
    print()

    # OOS Sharpe table
    print("OOS SHARPE BY WINDOW:")
    hdr = f"{'Window':<8} {'Period':<12} {'sma_ratio':>10} {'tsmom':>8} {'donchian':>10} {'dual_sma':>10}   Best method"
    print(hdr)
    print("-" * len(hdr))
    period_labels = {
        "WF1": "2010–2011", "WF2": "2012–2013", "WF3": "2014–2015",
        "WF4": "2016–2017", "WF5": "2018–2019", "WF6": "2020–2021", "WF7": "2022–2023",
    }
    for wname, *_ in wf_windows:
        vals = {m: sharpe_tbl[m][wname] for m in methods}
        best = max(vals, key=lambda m: vals[m])
        row = (
            f"{wname:<8} {period_labels[wname]:<12}"
            f" {vals['sma_ratio']:>10.3f}"
            f" {vals['tsmom']:>8.3f}"
            f" {vals['donchian']:>10.3f}"
            f" {vals['dual_sma']:>10.3f}"
            f"   {best}"
        )
        print(row)

    print()
    print("OOS RETURN BY WINDOW:")
    hdr2 = f"{'Window':<8} {'Period':<12} {'sma_ratio':>10} {'tsmom':>8} {'donchian':>10} {'dual_sma':>10}"
    print(hdr2)
    print("-" * len(hdr2))
    for wname, *_ in wf_windows:
        vals = {m: ret_tbl[m][wname] for m in methods}
        row = (
            f"{wname:<8} {period_labels[wname]:<12}"
            f" {vals['sma_ratio']*100:>9.2f}%"
            f" {vals['tsmom']*100:>7.2f}%"
            f" {vals['donchian']*100:>9.2f}%"
            f" {vals['dual_sma']*100:>9.2f}%"
        )
        print(row)

    print()
    print("OOS MDD BY WINDOW:")
    hdr3 = f"{'Window':<8} {'Period':<12} {'sma_ratio':>10} {'tsmom':>8} {'donchian':>10} {'dual_sma':>10}"
    print(hdr3)
    print("-" * len(hdr3))
    for wname, *_ in wf_windows:
        vals = {m: mdd_tbl[m][wname] for m in methods}
        row = (
            f"{wname:<8} {period_labels[wname]:<12}"
            f" {vals['sma_ratio']*100:>9.2f}%"
            f" {vals['tsmom']*100:>7.2f}%"
            f" {vals['donchian']*100:>9.2f}%"
            f" {vals['dual_sma']*100:>9.2f}%"
        )
        print(row)

    print()
    print("RANKING BY WINDOW (1=best Sharpe):")
    hdr4 = f"{'Window':<8} {'sma_ratio':>10} {'tsmom':>8} {'donchian':>10} {'dual_sma':>10}"
    print(hdr4)
    print("-" * len(hdr4))
    for wname, *_ in wf_windows:
        row = (
            f"{wname:<8}"
            f" {rank_tbl['sma_ratio'][wname]:>10}"
            f" {rank_tbl['tsmom'][wname]:>8}"
            f" {rank_tbl['donchian'][wname]:>10}"
            f" {rank_tbl['dual_sma'][wname]:>10}"
        )
        print(row)

    # Average ranks
    avg_ranks = {
        m: np.mean(list(rank_tbl[m].values())) for m in methods
    }
    print(
        f"{'Avg rank:':<8}"
        f" {avg_ranks['sma_ratio']:>10.1f}"
        f" {avg_ranks['tsmom']:>8.1f}"
        f" {avg_ranks['donchian']:>10.1f}"
        f" {avg_ranks['dual_sma']:>10.1f}"
    )

    # ---------------------------------------------------------------------------
    # Summary statistics
    # ---------------------------------------------------------------------------

    print()
    print("SUMMARY STATISTICS:")
    hdr5 = f"{'Metric':<22} {'sma_ratio':>10} {'tsmom':>8} {'donchian':>10} {'dual_sma':>10}"
    print(hdr5)
    print("-" * len(hdr5))

    wnames = [w[0] for w in wf_windows]

    mean_sh  = {m: np.mean([sharpe_tbl[m][w] for w in wnames]) for m in methods}
    med_sh   = {m: np.median([sharpe_tbl[m][w] for w in wnames]) for m in methods}
    std_sh   = {m: np.std([sharpe_tbl[m][w] for w in wnames], ddof=1) for m in methods}
    best_win = {m: max(wnames, key=lambda w: sharpe_tbl[m][w]) for m in methods}
    worst_win= {m: min(wnames, key=lambda w: sharpe_tbl[m][w]) for m in methods}

    # Win rate vs baseline (sma_ratio)
    win_rate = {}
    for m in methods:
        if m == "sma_ratio":
            win_rate[m] = None
        else:
            wins = sum(
                1 for w in wnames
                if sharpe_tbl[m][w] > sharpe_tbl["sma_ratio"][w]
            )
            win_rate[m] = wins / len(wnames)

    def _wr(m: str) -> str:
        if win_rate[m] is None:
            return f"{'—':>10}"
        return f"{win_rate[m]*100:>9.1f}%"

    print(
        f"{'Mean OOS Sharpe':<22}"
        f" {mean_sh['sma_ratio']:>10.3f}"
        f" {mean_sh['tsmom']:>8.3f}"
        f" {mean_sh['donchian']:>10.3f}"
        f" {mean_sh['dual_sma']:>10.3f}"
    )
    print(
        f"{'Median OOS Sharpe':<22}"
        f" {med_sh['sma_ratio']:>10.3f}"
        f" {med_sh['tsmom']:>8.3f}"
        f" {med_sh['donchian']:>10.3f}"
        f" {med_sh['dual_sma']:>10.3f}"
    )
    print(
        f"{'Sharpe Std Dev':<22}"
        f" {std_sh['sma_ratio']:>10.3f}"
        f" {std_sh['tsmom']:>8.3f}"
        f" {std_sh['donchian']:>10.3f}"
        f" {std_sh['dual_sma']:>10.3f}"
    )
    print(
        f"{'Win rate vs base':<22}"
        f"{_wr('sma_ratio')}"
        f"{_wr('tsmom')}"
        f"{_wr('donchian')}"
        f"{_wr('dual_sma')}"
    )
    print(
        f"{'Avg rank':<22}"
        f" {avg_ranks['sma_ratio']:>10.1f}"
        f" {avg_ranks['tsmom']:>8.1f}"
        f" {avg_ranks['donchian']:>10.1f}"
        f" {avg_ranks['dual_sma']:>10.1f}"
    )
    print(
        f"{'Best window':<22}"
        f" {best_win['sma_ratio']:>10}"
        f" {best_win['tsmom']:>8}"
        f" {best_win['donchian']:>10}"
        f" {best_win['dual_sma']:>10}"
    )
    print(
        f"{'Worst window':<22}"
        f" {worst_win['sma_ratio']:>10}"
        f" {worst_win['tsmom']:>8}"
        f" {worst_win['donchian']:>10}"
        f" {worst_win['dual_sma']:>10}"
    )

    # ---------------------------------------------------------------------------
    # Verdict
    # ---------------------------------------------------------------------------

    print()
    print("VERDICT:")

    # Overall winner by mean Sharpe
    winner = max(methods, key=lambda m: mean_sh[m])
    winner_mean = mean_sh[winner]
    base_mean   = mean_sh["sma_ratio"]

    # IS ranking from spec/memory context: tsmom > donchian > dual_sma > sma_ratio
    is_order = ["tsmom", "donchian", "dual_sma", "sma_ratio"]

    # Check if OOS ranking matches IS ranking (excluding baseline)
    challengers = ["tsmom", "donchian", "dual_sma"]
    oos_ranked = sorted(challengers, key=lambda m: mean_sh[m], reverse=True)

    is_order_challengers = [m for m in is_order if m in challengers]
    ranking_holds = oos_ranked == is_order_challengers

    print(f"  Walk-forward winner (mean OOS Sharpe): {winner} ({winner_mean:.3f})")
    print(f"  Baseline (sma_ratio) mean OOS Sharpe:  {base_mean:.3f}")
    print()
    print(f"  OOS ranking (mean Sharpe, best→worst):")
    all_oos_ranked = sorted(methods, key=lambda m: mean_sh[m], reverse=True)
    for rank_pos, m in enumerate(all_oos_ranked, start=1):
        wr_str = ""
        if m != "sma_ratio":
            wr_val = win_rate[m] * 100
            wr_str = f"  (beats baseline {wr_val:.0f}% of windows)"
        marker = " <-- WINNER" if m == winner else ""
        print(f"    {rank_pos}. {m:<12}  mean Sharpe {mean_sh[m]:.3f}{wr_str}{marker}")

    print()
    if ranking_holds:
        print("  IS ranking (tsmom > donchian > dual_sma) HOLDS out-of-sample.")
    else:
        print(
            f"  IS ranking does NOT fully hold OOS.  "
            f"OOS order of challengers: {' > '.join(oos_ranked)}"
        )

    beats_baseline = [m for m in challengers if mean_sh[m] > base_mean]
    if beats_baseline:
        print(
            f"  Method(s) beating baseline on mean OOS Sharpe: "
            + ", ".join(beats_baseline)
        )
    else:
        print(
            "  No challenger method beats sma_ratio on mean OOS Sharpe — "
            "baseline remains preferred."
        )

    print("=" * 78)


if __name__ == "__main__":
    main()
