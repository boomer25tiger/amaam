"""
Sensitivity analysis execution script for AMAAM.

Runs all four sensitivity sweeps defined in Section 6 of the specification:
factor weight grid, selection count variants, weighting scheme comparison, and
rebalancing frequency comparison. Saves results to reports/summary/ as CSV
files for inspection and as inputs to the visualization pipeline. All sweeps
use the development period only (through December 2017).
"""

import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd

from config.default_config import ModelConfig
from src.analysis.sensitivity import (
    run_rebalancing_sensitivity,
    run_selection_sensitivity,
    run_weight_sensitivity,
)
from src.data.loader import load_validated_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Display floats with four decimal places so small differences remain visible.
pd.set_option("display.float_format", lambda x: f"{x:.4f}")
pd.set_option("display.max_columns", 20)
pd.set_option("display.width", 120)

_REPORT_DIR = "reports/sensitivity"
_DATA_DIR = "data/processed"


def _print_section(title: str, df: pd.DataFrame) -> None:
    """Print a labelled section separator followed by the DataFrame."""
    sep = "=" * 80
    print(f"\n{sep}")
    print(f"=== {title} ===")
    print(sep)
    print(df.to_string())
    print()


def _save_csv(df: pd.DataFrame, filename: str) -> None:
    """Save *df* to *_REPORT_DIR*/*filename* if the directory can be created."""
    try:
        os.makedirs(_REPORT_DIR, exist_ok=True)
        path = os.path.join(_REPORT_DIR, filename)
        df.to_csv(path)
        logger.info("Saved %s", path)
    except OSError as exc:
        # Non-fatal — the analysis results have already been printed to stdout.
        logger.warning("Could not save %s: %s", filename, exc)


def main() -> None:
    """Run all sensitivity sweeps and print results."""
    # ── Load data ────────────────────────────────────────────────────────────
    logger.info("Loading processed data from %s …", _DATA_DIR)
    data = load_validated_data(_DATA_DIR)
    logger.info("Loaded %d tickers.", len(data))

    # ── Base config ──────────────────────────────────────────────────────────
    # All sweeps operate on the development period only (through 2017-12-31) so
    # that out-of-sample data is never contaminated by parameter selection.
    cfg = ModelConfig()
    logger.info(
        "Base config: start=%s  end=%s  cost=%.0f bps  scheme=%s",
        cfg.backtest_start, cfg.backtest_end,
        cfg.transaction_cost * 10_000, cfg.weighting_scheme,
    )

    # ── Sweep 1: Factor weight sensitivity ───────────────────────────────────
    logger.info("Running factor weight sensitivity sweep …")
    df_weights = run_weight_sensitivity(data, cfg)
    _print_section("Factor Weight Sensitivity", df_weights)
    _save_csv(df_weights, "weight_sensitivity.csv")

    # ── Sweep 2: Selection count sensitivity ─────────────────────────────────
    logger.info("Running selection count sensitivity sweep …")
    df_selection = run_selection_sensitivity(data, cfg)
    _print_section("Selection Count Sensitivity", df_selection)
    _save_csv(df_selection, "selection_sensitivity.csv")

    # ── Sweep 3: Rebalancing frequency × transaction cost ────────────────────
    logger.info("Running rebalancing frequency sensitivity sweep …")
    df_rebal = run_rebalancing_sensitivity(data, cfg)
    _print_section("Rebalancing Frequency Sensitivity", df_rebal)
    _save_csv(df_rebal, "rebalancing_sensitivity.csv")

    logger.info("All sensitivity sweeps complete.")


if __name__ == "__main__":
    main()
