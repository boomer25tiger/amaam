"""
Compare blended-momentum lookback windows: [21, 63, 126] vs [63, 126, 252].

Motivation: the 21-day component (≈1 month) measures near-term noise at monthly
rebalancing frequency.  Replacing it with 252 days (≈12 months) shifts all three
windows to established medium-to-long-term horizons, potentially reducing
TRank rank-boundary churn without slowing exits.

Reports IS, holdout, walk-forward (6 folds), and stacked OOS.

Usage
-----
    python3.13 scripts/compare_momentum_blend.py
    python3.13 scripts/compare_momentum_blend.py --data-dir data/processed
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import logging
import numpy as np
import pandas as pd
from dataclasses import replace

from config.default_config import ModelConfig
from src.data.loader import load_validated_data
from src.backtest.engine import run_backtest

logging.basicConfig(level=logging.WARNING,
                    format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S")

# ── Walk-forward fold definitions (identical to walk_forward.py) ─────────────
FOLDS = [
    ("Fold 1", "2007-08-01", "2012-12-31", "2013-01-01", "2014-12-31"),
    ("Fold 2", "2007-08-01", "2014-12-31", "2015-01-01", "2016-12-31"),
    ("Fold 3", "2007-08-01", "2016-12-31", "2017-01-01", "2018-12-31"),
    ("Fold 4", "2007-08-01", "2018-12-31", "2019-01-01", "2020-12-31"),
    ("Fold 5", "2007-08-01", "2020-12-31", "2021-01-01", "2022-12-31"),
    ("Fold 6", "2007-08-01", "2022-12-31", "2023-01-01", "2024-12-31"),
]

_METRIC_ORDER = [
    "Annualized Return", "Annualized Volatility", "Sharpe Ratio",
    "Sortino Ratio", "Calmar Ratio", "Max Drawdown", "Max Drawdown Duration",
    "Best Year", "Worst Year", "% Positive Months", "Avg Annual Turnover",
]
_PCT_METRICS = {
    "Annualized Return", "Annualized Volatility", "Max Drawdown",
    "Best Year", "Worst Year", "% Positive Months", "Avg Annual Turnover",
}


def _fmt(metric: str, value: float) -> str:
    import math
    if math.isnan(value):
        return "N/A"
    if metric == "Max Drawdown Duration":
        return f"{int(value)}mo"
    if metric in _PCT_METRICS:
        return f"{value * 100:+.2f}%"
    return f"{value:.3f}"


def _print_metrics_table(results: dict, title: str) -> None:
    col_w = 22
    strategies = list(results.keys())
    header = f"{'Metric':<30}" + "".join(f"{s:>{col_w}}" for s in strategies)
    sep = "=" * len(header)
    print(f"\n{title}")
    print(f"{sep}\n{header}\n{sep}")
    for metric in _METRIC_ORDER:
        row = f"{metric:<30}"
        for s in strategies:
            val = results[s].get(metric, float("nan"))
            row += f"{_fmt(metric, val):>{col_w}}"
        print(row)
    print(sep)


def _slice_metrics(rets: pd.Series, start: str, end: str,
                   turnover: pd.Series | None = None,
                   rf: float = 0.02) -> dict:
    """Compute summary metrics on a slice of monthly returns."""
    r = rets[(rets.index >= start) & (rets.index <= end)]
    if len(r) < 6:
        return {m: float("nan") for m in _METRIC_ORDER}

    ann_ret = (1 + r).prod() ** (12 / len(r)) - 1
    ann_vol = r.std() * np.sqrt(12)
    sr      = (ann_ret - rf) / ann_vol if ann_vol > 0 else float("nan")
    eq      = (1 + r).cumprod()
    maxdd   = (eq / eq.cummax() - 1).min()
    calmar  = ann_ret / abs(maxdd) if maxdd != 0 else float("nan")

    # Sortino
    downside = r[r < 0]
    sortino_vol = downside.std() * np.sqrt(12)
    sortino = (ann_ret - rf) / sortino_vol if sortino_vol > 0 else float("nan")

    # Drawdown duration
    running_max = eq.cummax()
    in_dd = (eq < running_max)
    dd_dur = 0
    max_dd_dur = 0
    for v in in_dd:
        dd_dur = dd_dur + 1 if v else 0
        max_dd_dur = max(max_dd_dur, dd_dur)

    # Annual stats
    annual = r.resample("YE").apply(lambda x: (1 + x).prod() - 1)
    best_year  = annual.max() if not annual.empty else float("nan")
    worst_year = annual.min() if not annual.empty else float("nan")
    pct_pos    = (r > 0).mean()

    avg_to = float("nan")
    if turnover is not None:
        to_slice = turnover[(turnover.index >= start) & (turnover.index <= end)]
        avg_to = float(to_slice.mean()) * 12 if not to_slice.empty else float("nan")

    return {
        "Annualized Return":    ann_ret,
        "Annualized Volatility": ann_vol,
        "Sharpe Ratio":         sr,
        "Sortino Ratio":        sortino,
        "Calmar Ratio":         calmar,
        "Max Drawdown":         maxdd,
        "Max Drawdown Duration": max_dd_dur,
        "Best Year":            best_year,
        "Worst Year":           worst_year,
        "% Positive Months":    pct_pos,
        "Avg Annual Turnover":  avg_to,
    }


def _sharpe(rets: pd.Series, start: str, end: str, rf: float = 0.02) -> float:
    r = rets[(rets.index >= start) & (rets.index <= end)]
    if len(r) < 6:
        return float("nan")
    ann_ret = (1 + r).prod() ** (12 / len(r)) - 1
    ann_vol = r.std() * np.sqrt(12)
    return (ann_ret - rf) / ann_vol if ann_vol > 0 else float("nan")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/processed")
    args = parser.parse_args()

    print("Loading data…")
    data = load_validated_data(args.data_dir)

    base_cfg = ModelConfig()
    cfg_current = base_cfg                                          # [21, 63, 126]
    cfg_new     = replace(base_cfg, momentum_blend_lookbacks=[63, 126, 252])

    configs = {
        "Current [21,63,126]": cfg_current,
        "New    [63,126,252]": cfg_new,
    }

    print("Running full-history backtests…")
    results = {}
    for label, cfg in configs.items():
        results[label] = run_backtest(data, cfg)
        print(f"  {label} done.")

    is_start  = base_cfg.backtest_start
    is_end    = base_cfg.holdout_start       # "2018-01-01"
    oos_start = base_cfg.holdout_start
    oos_end   = base_cfg.backtest_end

    # ── IS metrics ────────────────────────────────────────────────────────────
    _print_metrics_table(
        {lbl: _slice_metrics(r.monthly_returns, is_start, is_end, r.turnover)
         for lbl, r in results.items()},
        f"IN-SAMPLE  ({is_start[:7]} → {is_end[:7]})",
    )

    # ── Holdout metrics ───────────────────────────────────────────────────────
    _print_metrics_table(
        {lbl: _slice_metrics(r.monthly_returns, oos_start, oos_end, r.turnover)
         for lbl, r in results.items()},
        f"HOLDOUT  ({oos_start[:7]} → {oos_end[:7]})",
    )

    # ── Turnover deep-dive ────────────────────────────────────────────────────
    print("\nTURNOVER DETAIL  (full history)")
    print(f"  {'Config':<24}  {'Avg monthly':>12}  {'Annual (%)':>11}  "
          f"{'Annual cost @5bps':>18}")
    print("  " + "-" * 70)
    for lbl, r in results.items():
        avg = float(r.turnover.mean())
        print(f"  {lbl:<24}  {avg * 100:>11.2f}%  {avg * 12 * 100:>10.2f}%  "
              f"{avg * 12 * 5:>17.1f} bps")

    # ── Walk-forward ──────────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("WALK-FORWARD  (6 expanding folds, 2-year test windows)")
    print("=" * 78)

    rets = {lbl: r.monthly_returns for lbl, r in results.items()}
    labels = list(rets.keys())
    lbl_curr = labels[0]
    lbl_new  = labels[1]

    fold_records = []
    for fold_name, tr_s, tr_e, te_s, te_e in FOLDS:
        train = {lbl: _sharpe(rets[lbl], tr_s, tr_e) for lbl in labels}
        test  = {lbl: _sharpe(rets[lbl], te_s, te_e) for lbl in labels}
        delta = test[lbl_new] - test[lbl_curr]
        winner = lbl_new if test[lbl_new] > test[lbl_curr] else lbl_curr

        print(f"\n{fold_name}  train → {tr_e[:7]}  |  test {te_s[:7]} → {te_e[:7]}")
        print(f"  Train SR:  Current={train[lbl_curr]:.3f}   New={train[lbl_new]:.3f}")
        print(f"  Test  SR:  Current={test[lbl_curr]:.3f}   New={test[lbl_new]:.3f}"
              f"   Δ={delta:+.3f}  → {'NEW WINS' if delta > 0 else 'CURRENT WINS'}")

        fold_records.append(dict(
            fold=fold_name, te_s=te_s, te_e=te_e,
            curr=test[lbl_curr], new=test[lbl_new], delta=delta,
        ))

    fr = pd.DataFrame(fold_records)
    new_wins  = (fr["delta"] > 0).sum()
    avg_delta = fr["delta"].mean()

    print(f"\n{'='*78}")
    print(f"FOLD SUMMARY — New [63,126,252] vs Current [21,63,126] on test windows")
    print(f"{'Fold':<8}  {'Test window':<20}  {'Current SR':>10}  "
          f"{'New SR':>8}  {'Δ SR':>8}  {'Winner'}")
    print("-" * 78)
    for _, row in fr.iterrows():
        winner_tag = "NEW" if row["delta"] > 0 else "current"
        print(f"{row['fold']:<8}  {row['te_s'][:7]}→{row['te_e'][:7]:<10}  "
              f"{row['curr']:>10.3f}  {row['new']:>8.3f}  "
              f"{row['delta']:>+8.3f}  {winner_tag}")

    print(f"\n  New wins: {new_wins}/6 folds   avg Δ SR = {avg_delta:+.3f}")

    # ── Stacked OOS ───────────────────────────────────────────────────────────
    print(f"\n{'='*78}")
    print("STACKED OOS — all six test windows concatenated")
    stacked = {lbl: [] for lbl in labels}
    for _, row in fr.iterrows():
        for lbl in labels:
            r = rets[lbl]
            stacked[lbl].append(r[(r.index >= row["te_s"]) & (r.index <= row["te_e"])])

    for lbl in labels:
        combined = pd.concat(stacked[lbl]).sort_index()
        m = _sharpe(combined,
                    combined.index.min().strftime("%Y-%m-%d"),
                    combined.index.max().strftime("%Y-%m-%d"))
        ann_ret = (1 + combined).prod() ** (12 / len(combined)) - 1
        eq = (1 + combined).cumprod()
        maxdd = (eq / eq.cummax() - 1).min()
        calmar = ann_ret / abs(maxdd) if maxdd != 0 else float("nan")
        print(f"  {lbl}  SR={m:.3f}  Ret={ann_ret*100:+.2f}%  "
              f"MaxDD={maxdd*100:.2f}%  Calmar={calmar:.3f}  n={len(combined)}mo")

    # ── Verdict ───────────────────────────────────────────────────────────────
    print(f"\n{'='*78}")
    print("VERDICT")
    stacked_srs = {}
    for lbl in labels:
        combined = pd.concat(stacked[lbl]).sort_index()
        stacked_srs[lbl] = _sharpe(combined,
                                   combined.index.min().strftime("%Y-%m-%d"),
                                   combined.index.max().strftime("%Y-%m-%d"))

    curr_sr = stacked_srs[lbl_curr]
    new_sr  = stacked_srs[lbl_new]
    adopt   = new_wins >= 4 and new_sr > curr_sr
    print(f"  New [63,126,252]: wins {new_wins}/6 folds | "
          f"stacked OOS SR {new_sr:.3f} vs {curr_sr:.3f} (current) | "
          f"avg Δ SR {avg_delta:+.3f}")
    print(f"  → {'ADOPT ✓' if adopt else 'REJECT ✗'}")
    print(f"  (Adoption threshold: ≥ 4/6 fold wins AND stacked OOS SR > current)")


if __name__ == "__main__":
    main()
