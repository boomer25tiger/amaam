"""
Main backtest execution script for AMAAM.

Loads validated data, instantiates ModelConfig with the desired parameters,
runs the full backtest via src/backtest/engine.py, computes all performance
metrics, and prints a summary table. Accepts command-line arguments for
transaction cost scenario, weighting scheme, and rebalancing frequency so that
multiple configurations can be run without editing source files.

Usage
-----
    python scripts/run_backtest.py
    python scripts/run_backtest.py --cost 0.0000 --scheme equal --output results/
    python scripts/run_backtest.py --start 2010-01-01 --end 2020-12-31
"""

import argparse
import logging
import os
import sys
from dataclasses import replace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd

from config.default_config import ModelConfig
from src.backtest.benchmarks import (
    compute_seven_twelve,
    compute_sixty_forty,
    compute_spy_benchmark,
)
from src.backtest.engine import run_backtest
from src.backtest.metrics import compute_all_metrics
from src.data.loader import load_validated_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Metrics printed in this fixed order.
_METRIC_ORDER = [
    "Annualized Return",
    "Annualized Volatility",
    "Sharpe Ratio",
    "Sortino Ratio",
    "Calmar Ratio",
    "Max Drawdown",
    "Max Drawdown Duration",
    "Best Month",
    "Worst Month",
    "Best Year",
    "Worst Year",
    "% Positive Months",
    "% Positive Years",
    "Total Return",
    "Avg Monthly Turnover",
    "Avg Annual Turnover",
]


def _fmt(metric: str, value: float) -> str:
    """Format a metric value as a human-readable string."""
    import math
    if math.isnan(value):
        return "N/A"
    pct_metrics = {
        "Annualized Return", "Annualized Volatility", "Max Drawdown",
        "Best Month", "Worst Month", "Best Year", "Worst Year", "Total Return",
        "Avg Monthly Turnover", "Avg Annual Turnover",
        "% Positive Months", "% Positive Years",
    }
    dur_metrics = {"Max Drawdown Duration"}
    if metric in pct_metrics:
        return f"{value * 100:+.2f}%"
    if metric in dur_metrics:
        return f"{int(value)} months"
    return f"{value:.3f}"


def _print_table(results: dict[str, dict]) -> None:
    """Print a side-by-side metric table for all strategies."""
    col_w = 22
    strategies = list(results.keys())
    header = f"{'Metric':<30}" + "".join(f"{s:>{col_w}}" for s in strategies)
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))
    for metric in _METRIC_ORDER:
        row = f"{metric:<30}"
        for s in strategies:
            val = results[s].get(metric, float("nan"))
            row += f"{_fmt(metric, val):>{col_w}}"
        print(row)
    print("=" * len(header))


def _save_outputs(result, output_dir: str, prefix: str) -> None:
    """Save equity curve, monthly returns, and allocations to CSV."""
    os.makedirs(output_dir, exist_ok=True)
    result.equity_curve.to_csv(os.path.join(output_dir, f"{prefix}_equity.csv"), header=True)
    result.monthly_returns.to_csv(os.path.join(output_dir, f"{prefix}_returns.csv"), header=True)
    if not result.allocations.empty:
        result.allocations.to_csv(os.path.join(output_dir, f"{prefix}_allocations.csv"))
    logger.info("Saved %s outputs to %s/", prefix, output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AMAAM backtest and benchmarks.")
    parser.add_argument("--data-dir",  default="data/processed",  help="Processed data directory")
    parser.add_argument("--output",    default="",               help="Directory to save CSV outputs (optional)")
    parser.add_argument("--start",     default=None,             help="Override backtest start date (YYYY-MM-DD)")
    parser.add_argument("--end",       default=None,             help="Override backtest end date (YYYY-MM-DD)")
    parser.add_argument("--cost",      type=float, default=None, help="Transaction cost override (e.g. 0.001)")
    parser.add_argument("--scheme",    default=None,             choices=["equal", "inverse_volatility"],
                        help="Weighting scheme override")
    parser.add_argument("--no-benchmarks", action="store_true",  help="Skip benchmark computation")
    args = parser.parse_args()

    # ── Load data ────────────────────────────────────────────────────────────
    logger.info("Loading processed data from %s …", args.data_dir)
    data = load_validated_data(args.data_dir)
    logger.info("Loaded %d tickers.", len(data))

    # ── Build config ─────────────────────────────────────────────────────────
    cfg = ModelConfig()
    overrides: dict = {}
    if args.start:  overrides["backtest_start"] = args.start
    if args.end:    overrides["backtest_end"]   = args.end
    if args.cost is not None:  overrides["transaction_cost"] = args.cost
    if args.scheme: overrides["weighting_scheme"] = args.scheme
    if overrides:
        cfg = replace(cfg, **overrides)

    logger.info(
        "Config: start=%s  end=%s  cost=%.0f bps  scheme=%s",
        cfg.backtest_start, cfg.backtest_end,
        cfg.transaction_cost * 10_000, cfg.weighting_scheme,
    )

    # ── Run AMAAM ────────────────────────────────────────────────────────────
    logger.info("Running AMAAM backtest …")
    result = run_backtest(data, cfg)

    all_metrics: dict[str, dict] = {"AMAAM": result.metrics}

    if args.output:
        _save_outputs(result, args.output, "amaam")

    # ── Benchmarks ───────────────────────────────────────────────────────────
    if not args.no_benchmarks:
        bm_fns = {
            "SPY B&H":    compute_spy_benchmark,
            "60/40":      compute_sixty_forty,
            "7Twelve":    compute_seven_twelve,
        }
        for label, fn in bm_fns.items():
            logger.info("Computing benchmark: %s …", label)
            try:
                bm_returns = fn(data, cfg.backtest_start, cfg.backtest_end)
                bm_metrics = compute_all_metrics(bm_returns, risk_free_rate=0.02)
                all_metrics[label] = bm_metrics
            except Exception as exc:
                logger.warning("Benchmark %s failed: %s", label, exc)

    # ── Print results ─────────────────────────────────────────────────────────
    _print_table(all_metrics)
    print(f"\nAMAAM equity curve: {result.equity_curve.iloc[-1]:.4f}× initial capital")
    print(f"Holding period:     {result.monthly_returns.index[0].date()} → "
          f"{result.monthly_returns.index[-1].date()}")
    n_months = len(result.monthly_returns)
    print(f"Months simulated:   {n_months}")


if __name__ == "__main__":
    main()
