"""
LQD and XLRE addition test.

Scenarios:
  A. Baseline      — original 16-asset main sleeve
  B. +LQD          — add LQD (full history from 2002)
  C. +XLRE         — add XLRE (data from Oct 2015; NaN-ranked before warm-up)
  D. +LQD +XLRE    — both added

Metrics reported per scenario:
  - IS  (2007-08 → 2017-12)
  - OOS (2018-01 → 2026-04)
  - FULL period
  - Year-by-year OOS returns
  - LQD / XLRE selection frequency and average return when held
"""

import sys
sys.path.insert(0, "/Users/GualyCr/Desktop/AMAAM/amaam/.claude/worktrees/reverent-mayer")

import numpy as np
import pandas as pd

from config.default_config import ModelConfig
import config.etf_universe as universe_mod
import src.backtest.engine as eng
from src.backtest.engine import run_backtest
from src.data.loader import load_validated_data


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_scenario(data_dict: dict, extra_tickers: list, label: str):
    """Run backtest with extra tickers appended to the main sleeve.

    Tickers with data predating the common fleet start (2007-08-01) are trimmed
    to that date so warm-up periods are synchronised across the full sleeve.
    Without trimming, a ticker with pre-2007 history bypasses the engine's
    NaN-guard during the hedge warm-up window and triggers a ValueError.
    """
    COMMON_START = "2007-08-01"
    trimmed_dict = {
        t: df.loc[df.index >= COMMON_START] if df.index[0] < pd.Timestamp(COMMON_START) else df
        for t, df in data_dict.items()
    }

    original_main = universe_mod.MAIN_SLEEVE_TICKERS[:]
    universe_mod.MAIN_SLEEVE_TICKERS = original_main + extra_tickers
    eng.MAIN_SLEEVE_TICKERS = universe_mod.MAIN_SLEEVE_TICKERS

    try:
        result = run_backtest(trimmed_dict, ModelConfig())
    finally:
        universe_mod.MAIN_SLEEVE_TICKERS = original_main
        eng.MAIN_SLEEVE_TICKERS = original_main

    return result


def slice_metrics(result, start: str, end: str) -> dict:
    ret = result.monthly_returns
    ret = ret[(ret.index >= start) & (ret.index <= end)]
    if len(ret) < 3:
        return dict(ret=float("nan"), sharpe=float("nan"),
                    maxdd=float("nan"), calmar=float("nan"), n=0)
    ann_ret = (1 + ret).prod() ** (12 / len(ret)) - 1
    ann_vol = ret.std() * np.sqrt(12)
    sharpe  = (ann_ret - 0.02) / ann_vol
    eq      = (1 + ret).cumprod()
    maxdd   = (eq / eq.cummax() - 1).min()
    calmar  = ann_ret / abs(maxdd) if maxdd != 0 else float("nan")
    return dict(ret=ann_ret*100, sharpe=sharpe, maxdd=maxdd*100,
                calmar=calmar, n=len(ret))


def holding_stats(result, data_dict: dict, ticker: str) -> dict:
    """Frequency, avg weight, and avg monthly return when ticker is held."""
    allocs = result.allocations
    if ticker not in allocs.columns:
        return dict(months=0, pct=0, avg_w=0, avg_ret=float("nan"))

    held = allocs[allocs[ticker] > 0]
    if held.empty:
        return dict(months=0, pct=0, avg_w=0, avg_ret=float("nan"))

    total_months = len(result.monthly_returns)
    closes = pd.DataFrame({ticker: data_dict[ticker]["Close"]})
    exec_dates = result.monthly_returns.index.tolist()
    signal_dates = allocs.index.tolist()

    rets = []
    for i, sig in enumerate(signal_dates):
        if i >= len(exec_dates) - 1:
            break
        if allocs.at[sig, ticker] <= 0:
            continue
        e0, e1 = exec_dates[i], exec_dates[i + 1]
        if e0 in closes.index and e1 in closes.index:
            p0, p1 = closes.at[e0, ticker], closes.at[e1, ticker]
            if p0 > 0:
                rets.append((p1 / p0) - 1)

    avg_ret = np.mean(rets) * 100 if rets else float("nan")
    return dict(months=len(held), pct=len(held)/total_months*100,
                avg_w=held[ticker].mean(), avg_ret=avg_ret)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading data…")
    data_dict = load_validated_data("data/processed")

    cfg = ModelConfig()
    IS_START, IS_END   = cfg.backtest_start, "2017-12-31"
    OOS_START, OOS_END = cfg.holdout_start,  cfg.backtest_end

    scenarios = [
        ("A. Baseline",   []),
        ("B. +LQD",       ["LQD"]),
        ("C. +XLRE",      ["XLRE"]),
        ("D. +LQD+XLRE",  ["LQD", "XLRE"]),
    ]

    results = {}
    for label, extras in scenarios:
        print(f"Running {label}…")
        results[label] = run_scenario(data_dict, extras, label)

    # ── Summary table ────────────────────────────────────────────────────────
    print("\n" + "=" * 88)
    print("SCENARIO COMPARISON — IS / OOS / FULL")
    print("=" * 88)
    print(f"{'Scenario':<20} {'Period':<6} {'Ret%':>7} {'Sharpe':>8} {'MaxDD%':>9} {'Calmar':>8} {'Mo':>5}")
    print("-" * 88)

    for label, _ in scenarios:
        res = results[label]
        for period, s, e in [("IS", IS_START, IS_END),
                              ("OOS", OOS_START, OOS_END),
                              ("FULL", IS_START, OOS_END)]:
            m = slice_metrics(res, s, e)
            marker = " ←" if period == "OOS" and label != "A. Baseline" and \
                             m["sharpe"] > slice_metrics(results["A. Baseline"], s, e)["sharpe"] else ""
            print(f"{label:<20} {period:<6} {m['ret']:>7.2f} {m['sharpe']:>8.3f} "
                  f"{m['maxdd']:>9.2f} {m['calmar']:>8.3f} {m['n']:>5}{marker}")
        print()

    # ── OOS year-by-year ─────────────────────────────────────────────────────
    print("=" * 88)
    print("YEAR-BY-YEAR OOS RETURNS (2018–2026)")
    print("-" * 88)
    labels = [s[0] for s in scenarios]
    print(f"{'Year':<6}" + "".join(f"{l:>18}" for l in labels))
    print("-" * 88)

    for year in range(2018, 2027):
        row = [str(year)]
        for label, _ in scenarios:
            ret = results[label].monthly_returns
            yr  = ret[ret.index.year == year]
            if len(yr) == 0:
                row.append("    —")
            else:
                ann = (1 + yr).prod() ** (12 / len(yr)) - 1
                row.append(f"{ann*100:+.1f}%")
        print(f"{row[0]:<6}" + "".join(f"{v:>18}" for v in row[1:]))

    # ── Holding stats for LQD and XLRE ───────────────────────────────────────
    print("\n" + "=" * 88)
    print("LQD AND XLRE SELECTION STATISTICS")
    print("-" * 88)

    for ticker, scenario_label in [("LQD",  "B. +LQD"),
                                    ("LQD",  "D. +LQD+XLRE"),
                                    ("XLRE", "C. +XLRE"),
                                    ("XLRE", "D. +LQD+XLRE")]:
        res = results[scenario_label]
        stats = holding_stats(res, data_dict, ticker)
        if stats["months"] == 0:
            print(f"  {ticker} in {scenario_label}: never selected")
        else:
            print(f"  {ticker} in {scenario_label}: "
                  f"selected {stats['months']} months ({stats['pct']:.1f}%), "
                  f"avg weight {stats['avg_w']:.1%}, "
                  f"avg monthly return {stats['avg_ret']:+.2f}%")

    # ── Sharpe delta vs baseline ─────────────────────────────────────────────
    print("\n" + "=" * 88)
    print("SHARPE DELTA vs BASELINE (A)")
    print("-" * 88)
    base_is   = slice_metrics(results["A. Baseline"], IS_START, IS_END)["sharpe"]
    base_oos  = slice_metrics(results["A. Baseline"], OOS_START, OOS_END)["sharpe"]
    base_full = slice_metrics(results["A. Baseline"], IS_START, OOS_END)["sharpe"]

    for label, _ in scenarios[1:]:
        res = results[label]
        d_is   = slice_metrics(res, IS_START, IS_END)["sharpe"]   - base_is
        d_oos  = slice_metrics(res, OOS_START, OOS_END)["sharpe"] - base_oos
        d_full = slice_metrics(res, IS_START, OOS_END)["sharpe"]  - base_full
        print(f"  {label:<20}  IS: {d_is:+.3f}   OOS: {d_oos:+.3f}   FULL: {d_full:+.3f}")


if __name__ == "__main__":
    main()
