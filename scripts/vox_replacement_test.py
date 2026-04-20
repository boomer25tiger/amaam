"""
VOX removal and replacement scenarios.

Scenarios:
  A. Baseline       — original 16-asset sleeve (includes VOX)
  B. -VOX           — 15 assets, VOX removed, nothing added
  C. VOX → QQQ      — swap VOX for Nasdaq-100 (full history 1999)
  D. VOX → INDA     — swap VOX for iShares MSCI India (history from Feb 2012)

Metrics: IS / OOS / FULL Sharpe, Return, MaxDD, Calmar
+ year-by-year OOS returns
+ selection stats for QQQ and INDA when applicable
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

COMMON_START = "2007-08-01"


def trim(data_dict):
    """Align all tickers to the common fleet start to keep warmup synchronised."""
    return {
        t: df.loc[df.index >= COMMON_START] if df.index[0] < pd.Timestamp(COMMON_START) else df
        for t, df in data_dict.items()
    }


def run_scenario(data_dict, remove=None, add=None):
    original = universe_mod.MAIN_SLEEVE_TICKERS[:]
    sleeve = [t for t in original if t != remove]
    if add:
        sleeve = sleeve + [add]
    universe_mod.MAIN_SLEEVE_TICKERS = sleeve
    eng.MAIN_SLEEVE_TICKERS = sleeve
    try:
        result = run_backtest(data_dict, ModelConfig())
    finally:
        universe_mod.MAIN_SLEEVE_TICKERS = original
        eng.MAIN_SLEEVE_TICKERS = original
    return result


def slice_metrics(result, start, end):
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
    return dict(ret=ann_ret*100, sharpe=sharpe, maxdd=maxdd*100, calmar=calmar, n=len(ret))


def holding_stats(result, data_dict, ticker):
    allocs = result.allocations
    if ticker not in allocs.columns:
        return None
    held = allocs[allocs[ticker] > 0]
    if held.empty:
        return None
    closes = pd.DataFrame({ticker: data_dict[ticker]["Close"]})
    exec_dates  = result.monthly_returns.index.tolist()
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
    total = len(result.monthly_returns)
    return dict(months=len(held), pct=len(held)/total*100,
                avg_ret=np.mean(rets)*100 if rets else float("nan"))


def vox_stats(result, data_dict):
    """How often was VOX selected and at what return (where applicable)."""
    return holding_stats(result, data_dict, "VOX")


def main():
    print("Loading and trimming data…")
    data_dict = trim(load_validated_data("data/processed"))
    cfg = ModelConfig()

    IS_START, IS_END   = cfg.backtest_start, "2017-12-31"
    OOS_START, OOS_END = cfg.holdout_start,  cfg.backtest_end

    scenarios = [
        ("A. Baseline",  None,  None),
        ("B. -VOX",      "VOX", None),
        ("C. VOX→QQQ",   "VOX", "QQQ"),
        ("D. VOX→INDA",  "VOX", "INDA"),
    ]

    results = {}
    for label, rem, add in scenarios:
        print(f"Running {label}…")
        results[label] = run_scenario(data_dict, remove=rem, add=add)

    # ── Summary table ────────────────────────────────────────────────────────
    print("\n" + "=" * 88)
    print("SCENARIO COMPARISON — IS / OOS / FULL")
    print("=" * 88)
    print(f"{'Scenario':<18} {'Period':<6} {'Ret%':>7} {'Sharpe':>8} {'MaxDD%':>9} {'Calmar':>8} {'Mo':>5}")
    print("-" * 88)

    for label, _, _ in scenarios:
        for period, s, e in [("IS", IS_START, IS_END),
                              ("OOS", OOS_START, OOS_END),
                              ("FULL", IS_START, OOS_END)]:
            m = slice_metrics(results[label], s, e)
            base_s = slice_metrics(results["A. Baseline"], s, e)["sharpe"]
            marker = " ←" if label != "A. Baseline" and m["sharpe"] > base_s else ""
            print(f"{label:<18} {period:<6} {m['ret']:>7.2f} {m['sharpe']:>8.3f} "
                  f"{m['maxdd']:>9.2f} {m['calmar']:>8.3f} {m['n']:>5}{marker}")
        print()

    # ── Sharpe delta vs baseline ─────────────────────────────────────────────
    print("=" * 88)
    print("SHARPE DELTA vs BASELINE (A)")
    print("-" * 88)
    for label, _, _ in scenarios[1:]:
        res = results[label]
        d_is   = slice_metrics(res, IS_START, IS_END)["sharpe"]   - slice_metrics(results["A. Baseline"], IS_START, IS_END)["sharpe"]
        d_oos  = slice_metrics(res, OOS_START, OOS_END)["sharpe"] - slice_metrics(results["A. Baseline"], OOS_START, OOS_END)["sharpe"]
        d_full = slice_metrics(res, IS_START, OOS_END)["sharpe"]  - slice_metrics(results["A. Baseline"], IS_START, OOS_END)["sharpe"]
        print(f"  {label:<18}  IS: {d_is:+.3f}   OOS: {d_oos:+.3f}   FULL: {d_full:+.3f}")

    # ── Year-by-year OOS ─────────────────────────────────────────────────────
    print("\n" + "=" * 88)
    print("YEAR-BY-YEAR RETURNS (full period, all scenarios)")
    print("-" * 88)
    labels = [s[0] for s in scenarios]
    print(f"{'Year':<6}" + "".join(f"{l:>20}" for l in labels))
    print("-" * 88)

    all_ret = {label: results[label].monthly_returns for label, _, _ in scenarios}
    min_year = min(r.index.year.min() for r in all_ret.values())

    for year in range(min_year, 2027):
        row = [str(year)]
        any_data = False
        for label, _, _ in scenarios:
            ret = all_ret[label]
            yr  = ret[ret.index.year == year]
            if len(yr) == 0:
                row.append("    —")
            else:
                ann = (1 + yr).prod() ** (12 / len(yr)) - 1
                row.append(f"{ann*100:+.1f}%")
                any_data = True
        if any_data:
            print(f"{row[0]:<6}" + "".join(f"{v:>20}" for v in row[1:]))

    # ── QQQ and INDA selection stats ─────────────────────────────────────────
    print("\n" + "=" * 88)
    print("REPLACEMENT ASSET SELECTION STATS")
    print("-" * 88)
    for ticker, scenario_label in [("QQQ", "C. VOX→QQQ"), ("INDA", "D. VOX→INDA")]:
        stats = holding_stats(results[scenario_label], data_dict, ticker)
        if stats is None:
            print(f"  {ticker}: never selected")
        else:
            print(f"  {ticker} in {scenario_label}: "
                  f"selected {stats['months']} months ({stats['pct']:.1f}%), "
                  f"avg monthly return {stats['avg_ret']:+.2f}%")

    # VOX stats in baseline for reference
    vox = vox_stats(results["A. Baseline"], data_dict)
    if vox:
        print(f"  VOX in baseline:          "
              f"selected {vox['months']} months ({vox['pct']:.1f}%), "
              f"avg monthly return {vox['avg_ret']:+.2f}%")

    # ── QQQ year-by-year when held ───────────────────────────────────────────
    print("\n" + "=" * 88)
    print("QQQ vs VOX: YEAR-BY-YEAR SELECTION COUNT AND AVG RETURN")
    print(f"{'Year':<6} {'QQQ sel':>8} {'QQQ ret%':>10} {'VOX sel':>8} {'VOX ret%':>10}")
    print("-" * 88)

    closes = pd.DataFrame({t: data_dict[t]["Close"] for t in ["QQQ", "VOX"] if t in data_dict})
    for res_label, ticker in [("C. VOX→QQQ", "QQQ"), ("A. Baseline", "VOX")]:
        pass  # computed below

    res_qqq  = results["C. VOX→QQQ"]
    res_base = results["A. Baseline"]
    exec_qqq  = res_qqq.monthly_returns.index.tolist()
    exec_base = res_base.monthly_returns.index.tolist()

    def yearly_stats(res, ticker, exec_dates):
        allocs = res.allocations
        if ticker not in allocs.columns:
            return {}
        signal_dates = allocs.index.tolist()
        year_data = {}
        for i, sig in enumerate(signal_dates):
            if i >= len(exec_dates) - 1:
                break
            w = allocs.at[sig, ticker] if ticker in allocs.columns else 0
            if w <= 0:
                continue
            e0, e1 = exec_dates[i], exec_dates[i + 1]
            year = e0.year
            if ticker in closes.columns and e0 in closes.index and e1 in closes.index:
                p0, p1 = closes.at[e0, ticker], closes.at[e1, ticker]
                ret = (p1/p0 - 1) if p0 > 0 else float("nan")
            else:
                ret = float("nan")
            year_data.setdefault(year, []).append(ret)
        return {y: (len(v), np.mean(v)*100) for y, v in year_data.items()}

    qqq_yr  = yearly_stats(res_qqq,  "QQQ", exec_qqq)
    vox_yr  = yearly_stats(res_base, "VOX", exec_base)

    all_years = sorted(set(qqq_yr) | set(vox_yr))
    for year in all_years:
        q = qqq_yr.get(year)
        v = vox_yr.get(year)
        qs = f"{q[0]:>4}  {q[1]:>+8.2f}%" if q else f"{'—':>4}  {'—':>9}"
        vs = f"{v[0]:>4}  {v[1]:>+8.2f}%" if v else f"{'—':>4}  {'—':>9}"
        print(f"{year:<6} {qs}    {vs}")


if __name__ == "__main__":
    main()
