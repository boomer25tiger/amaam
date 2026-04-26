"""
Head-to-head comparison: AMAAM baseline vs selection-hysteresis variants.

Selection hysteresis relaxes the exit rule so that an incumbent asset stays in
the portfolio until it falls outside the top-(N + exit_buffer) zone rather than
exiting immediately when it drops out of top-N.  Entry still requires a strict
top-N rank — only the exit threshold is widened.

Runs three configurations using the canonical wM=0.65/wV=0.25/wC=0.10 weights:
  1. Buffer-0 — no hysteresis (current production behaviour; exit_buffer=0)
  2. Buffer-1 — incumbent retained until ranked > N+1 (exit_buffer=1)
  3. Buffer-2 — incumbent retained until ranked > N+2 (exit_buffer=2)

Usage
-----
    python3.13 scripts/compare_hysteresis.py
    python3.13 scripts/compare_hysteresis.py --data-dir data/processed
"""

import logging
import os
import sys
from dataclasses import replace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config.default_config import ModelConfig
from src.backtest.engine import run_backtest
from src.data.loader import load_validated_data

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

_METRIC_ORDER = [
    "Annualized Return",
    "Annualized Volatility",
    "Sharpe Ratio",
    "Sortino Ratio",
    "Calmar Ratio",
    "Max Drawdown",
    "Max Drawdown Duration",
    "Best Year",
    "Worst Year",
    "% Positive Months",
    "Avg Annual Turnover",
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


def _print_table(results: dict) -> None:
    col_w = 18
    strategies = list(results.keys())
    header = f"{'Metric':<32}" + "".join(f"{s:>{col_w}}" for s in strategies)
    sep = "=" * len(header)
    print(f"\n{sep}\n{header}\n{sep}")
    for metric in _METRIC_ORDER:
        row = f"{metric:<32}"
        for s in strategies:
            val = results[s].get(metric, float("nan"))
            row += f"{_fmt(metric, val):>{col_w}}"
        print(row)
    print(sep)


def _turnover_stats(result) -> dict:
    """Return monthly average turnover and annualised round-trip cost at 5 bps/leg."""
    if result.turnover.empty:
        return {"avg_monthly_turnover": float("nan"), "annual_cost_bps": float("nan")}
    avg = float(result.turnover.mean())
    # 12 months × avg monthly turnover × 5 bps per leg = annual cost in bps.
    annual_cost_bps = avg * 12 * 5.0
    return {"avg_monthly_turnover": avg, "annual_cost_bps": annual_cost_bps}


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument(
        "--is-only", action="store_true",
        help="Restrict backtest to the in-sample window (backtest_start → holdout_start).",
    )
    parser.add_argument(
        "--holdout-only", action="store_true",
        help="Restrict backtest to the holdout window (holdout_start → backtest_end). "
             "Run this at most once per design decision; results inform go/no-go only.",
    )
    args = parser.parse_args()

    print("Loading data…")
    data = load_validated_data(args.data_dir)

    base_cfg = ModelConfig()
    if args.is_only:
        base_cfg = replace(base_cfg, backtest_end=base_cfg.holdout_start)
        print(f"IS-only mode: {base_cfg.backtest_start} → {base_cfg.backtest_end}")
    elif args.holdout_only:
        base_cfg = replace(base_cfg, backtest_start=base_cfg.holdout_start)
        print(f"Holdout-only mode: {base_cfg.backtest_start} → {base_cfg.backtest_end}")

    configs = {
        "Buffer-0 (baseline)": base_cfg,
        "Buffer-1": replace(base_cfg, selection_exit_buffer=1),
        "Buffer-2": replace(base_cfg, selection_exit_buffer=2),
    }

    results = {}
    for label, cfg in configs.items():
        print(f"Running {label}…")
        results[label] = run_backtest(data, cfg)

    _print_table({label: r.metrics for label, r in results.items()})

    # Turnover deep-dive: show monthly average and estimated annual cost.
    print("\nTurnover detail (one-way 5 bps/leg assumed):")
    header = f"  {'Config':<22}  {'Avg monthly TO':>16}  {'Annual cost (bps)':>18}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for label, result in results.items():
        stats = _turnover_stats(result)
        avg_to = stats["avg_monthly_turnover"]
        cost   = stats["annual_cost_bps"]
        print(f"  {label:<22}  {avg_to * 100:>15.2f}%  {cost:>17.1f} bps")

    # Relative comparison: how much does each buffer reduce turnover vs baseline?
    baseline_to = results["Buffer-0 (baseline)"].turnover.mean()
    print("\nTurnover reduction vs baseline:")
    for label, result in results.items():
        if label == "Buffer-0 (baseline)":
            continue
        avg_to = result.turnover.mean()
        reduction_pct = (baseline_to - avg_to) / baseline_to * 100
        print(f"  {label}: {reduction_pct:+.1f}% change in avg monthly turnover")


if __name__ == "__main__":
    main()
