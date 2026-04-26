"""
Head-to-head comparison: AMAAM baseline vs portfolio vol-targeted variants.

Runs four configurations using the canonical wM=0.65/wV=0.25/wC=0.10 weights:
  1. Baseline     — no vol targeting (current production model)
  2. VT-10%-NoLev — vol target 10%, max leverage 1.0 (scale down only)
  3. VT-10%-1.5x  — vol target 10%, max leverage 1.5 (modest leverage allowed)
  4. VT-12%-NoLev — vol target 12% (closer to current realised vol of 12.4%)

Usage
-----
    python3.13 scripts/compare_vol_targeting.py
    python3.13 scripts/compare_vol_targeting.py --data-dir data/processed
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
    level=logging.WARNING,          # suppress per-month INFO noise during sweep
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
    col_w = 20
    strategies = list(results.keys())
    header = f"{'Metric':<30}" + "".join(f"{s:>{col_w}}" for s in strategies)
    sep = "=" * len(header)
    print(f"\n{sep}\n{header}\n{sep}")
    for metric in _METRIC_ORDER:
        row = f"{metric:<30}"
        for s in strategies:
            val = results[s].get(metric, float("nan"))
            row += f"{_fmt(metric, val):>{col_w}}"
        print(row)
    print(sep)


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/processed")
    args = parser.parse_args()

    print("Loading data…")
    data = load_validated_data(args.data_dir)

    base_cfg = ModelConfig()

    configs = {
        "Baseline": base_cfg,
        "VT-10% NoLev": replace(
            base_cfg,
            vol_targeting=True,
            vol_target=0.10,
            vol_target_lookback=21,
            vol_target_max_leverage=1.0,
        ),
        "VT-10% 1.5×Lev": replace(
            base_cfg,
            vol_targeting=True,
            vol_target=0.10,
            vol_target_lookback=21,
            vol_target_max_leverage=1.5,
        ),
        "VT-12% NoLev": replace(
            base_cfg,
            vol_targeting=True,
            vol_target=0.12,
            vol_target_lookback=21,
            vol_target_max_leverage=1.0,
        ),
    }

    results = {}
    for label, cfg in configs.items():
        print(f"Running {label}…")
        result = run_backtest(data, cfg)
        results[label] = result.metrics

    _print_table(results)

    # Print scale statistics for vol-targeted runs.
    print("\nVol-targeting scale statistics (how often / how much scaling occurs):")
    for label, cfg in configs.items():
        if not cfg.vol_targeting:
            continue
        result = run_backtest(data, cfg)
        # Infer average cash_proxy weight as a proxy for average scale reduction.
        if not result.allocations.empty and "SHY" in result.allocations.columns:
            avg_shy = result.allocations["SHY"].mean()
            print(f"  {label}: avg SHY weight (incl. baseline hedge) = {avg_shy * 100:.1f}%")


if __name__ == "__main__":
    main()
