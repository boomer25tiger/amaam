"""
Sensitivity analysis for AMAAM.

Runs systematic parameter sweeps to characterize the robustness of the model:
(1) factor weight sensitivity (wM from 0.20 to 0.60 in 0.05 steps, with wV and
wC absorbing the remainder equally), (2) selection count sensitivity (top 4–7
from the main sleeve), (3) weighting scheme comparison (equal vs inverse-vol),
and (4) rebalancing frequency comparison (monthly vs bi-weekly). All analysis
is performed on the development period only (through December 2017). See
Section 6 of the specification.
"""

import logging
from dataclasses import replace
from typing import Dict, List

import numpy as np
import pandas as pd

from config.default_config import ModelConfig
from src.backtest.engine import run_backtest

logger = logging.getLogger(__name__)

# Columns extracted from BacktestResult.metrics for each sweep table.
_WEIGHT_COLS = [
    "Sharpe Ratio", "Max Drawdown", "Annualized Return",
    "Annualized Volatility", "Calmar Ratio",
]
_SELECTION_COLS = [
    "Sharpe Ratio", "Max Drawdown", "Annualized Return",
    "Annualized Volatility", "Calmar Ratio", "Total Return",
    "Best Year", "Worst Year", "% Positive Months",
]
_REBAL_COLS = [
    "Sharpe Ratio", "Max Drawdown", "Annualized Return",
    "Annualized Volatility", "Calmar Ratio", "Avg Monthly Turnover",
]


def run_weight_sensitivity(
    data_dict: Dict[str, pd.DataFrame],
    config: ModelConfig,
    wm_values: List[float] | None = None,
) -> pd.DataFrame:
    """Run a factor-weight sweep varying wM and letting wV = wC = (1 - wM) / 2.

    The sweep covers wM ∈ [0.20, 0.60] in 0.05 increments plus a special
    equal-weight baseline (wM = wV = wC = 1/3).  All other config parameters
    are unchanged, so each row isolates the effect of momentum weighting.

    Parameters
    ----------
    data_dict : Dict[str, pd.DataFrame]
        Processed OHLCV data keyed by ticker, as returned by load_validated_data.
    config : ModelConfig
        Base configuration to clone for each sweep run.  The backtest date
        range should already be restricted to the development period.
    wm_values : List[float] or None, optional
        Explicit list of wM values to sweep.  Defaults to np.arange(0.20, 0.65, 0.05)
        rounded to two decimal places.

    Returns
    -------
    pd.DataFrame
        One row per configuration, indexed by the "label" column.  Columns are
        ``["wM", "wV", "wC", "label", "Sharpe Ratio", "Max Drawdown",
        "Annualized Return", "Annualized Volatility", "Calmar Ratio"]``.
    """
    if wm_values is None:
        wm_values = [round(x, 2) for x in np.arange(0.20, 0.65, 0.05)]

    records: List[dict] = []

    for wm in wm_values:
        remainder = 1.0 - wm
        wv = wc = remainder / 2.0
        label = f"wM={wm:.2f}"
        logger.info("Weight sweep: %s (wM=%.2f, wV=%.2f, wC=%.2f)", label, wm, wv, wc)

        cfg = replace(
            config,
            weight_momentum=wm,
            weight_volatility=wv,
            weight_correlation=wc,
        )
        result = run_backtest(data_dict, cfg)

        row: dict = {"wM": wm, "wV": wv, "wC": wc, "label": label}
        for col in _WEIGHT_COLS:
            row[col] = result.metrics.get(col, float("nan"))
        records.append(row)

    # Equal-weight baseline: wM = wV = wC = 1/3.
    # Included so readers can see where equal weighting sits relative to the sweep.
    ew = round(1.0 / 3.0, 10)
    logger.info("Weight sweep: Equal (1/3) baseline")
    cfg_eq = replace(
        config,
        weight_momentum=ew,
        weight_volatility=ew,
        weight_correlation=ew,
    )
    result_eq = run_backtest(data_dict, cfg_eq)
    row_eq: dict = {"wM": ew, "wV": ew, "wC": ew, "label": "Equal (1/3)"}
    for col in _WEIGHT_COLS:
        row_eq[col] = result_eq.metrics.get(col, float("nan"))
    records.append(row_eq)

    df = pd.DataFrame(records)
    df = df.set_index("label")
    return df[["wM", "wV", "wC"] + _WEIGHT_COLS]


def run_selection_sensitivity(
    data_dict: Dict[str, pd.DataFrame],
    config: ModelConfig,
    n_values: List[int] | None = None,
) -> pd.DataFrame:
    """Run a selection-count sweep varying the number of assets chosen from the main sleeve.

    Holding all other parameters constant, this sweep quantifies how concentrating
    or diversifying the main-sleeve selection affects risk-adjusted returns.

    Parameters
    ----------
    data_dict : Dict[str, pd.DataFrame]
        Processed OHLCV data keyed by ticker.
    config : ModelConfig
        Base configuration to clone.  The ``main_sleeve_top_n`` field is
        overridden for each run.
    n_values : List[int] or None, optional
        List of Top N values to test.  Defaults to [4, 5, 6, 7].

    Returns
    -------
    pd.DataFrame
        One row per N value, indexed by "Top N".  Columns are
        ``["Sharpe Ratio", "Max Drawdown", "Annualized Return",
        "Annualized Volatility", "Calmar Ratio", "Total Return",
        "Best Year", "Worst Year", "% Positive Months"]``.
    """
    if n_values is None:
        n_values = [4, 5, 6, 7]

    records: List[dict] = []

    for n in n_values:
        logger.info("Selection sweep: Top N = %d", n)
        cfg = replace(config, main_sleeve_top_n=n)
        result = run_backtest(data_dict, cfg)

        row: dict = {"Top N": n}
        for col in _SELECTION_COLS:
            row[col] = result.metrics.get(col, float("nan"))
        records.append(row)

    df = pd.DataFrame(records)
    df = df.set_index("Top N")
    return df[_SELECTION_COLS]


def run_rebalancing_sensitivity(
    data_dict: Dict[str, pd.DataFrame],
    config: ModelConfig,
) -> pd.DataFrame:
    """Compare monthly vs bi-weekly rebalancing across three transaction-cost scenarios.

    Tests 2 frequencies × 3 cost levels = 6 combinations.  Higher rebalancing
    frequency can improve signal responsiveness but amplifies transaction costs,
    so the cross-product makes the trade-off explicit.

    Parameters
    ----------
    data_dict : Dict[str, pd.DataFrame]
        Processed OHLCV data keyed by ticker.
    config : ModelConfig
        Base configuration to clone.  ``rebalancing_frequency`` and
        ``transaction_cost`` are overridden for each combination.

    Returns
    -------
    pd.DataFrame
        One row per (frequency, cost) combination, indexed by a label of the
        form ``"<freq> / <bps> bps"``.  Columns are
        ``["Frequency", "Cost (bps)", "Sharpe Ratio", "Max Drawdown",
        "Annualized Return", "Annualized Volatility", "Calmar Ratio",
        "Avg Monthly Turnover"]``.
    """
    frequencies = ["monthly", "biweekly"]
    # Zero-cost baseline exposes pure frequency effect; 10 bps is default;
    # 15 bps stress-tests higher-cost environments.
    costs = [0.0, 0.0010, 0.0015]

    records: List[dict] = []

    for freq in frequencies:
        for cost in costs:
            cost_bps = int(cost * 10_000)
            label = f"{freq} / {cost_bps} bps"
            logger.info("Rebalancing sweep: %s", label)

            cfg = replace(
                config,
                rebalancing_frequency=freq,
                transaction_cost=cost,
            )
            result = run_backtest(data_dict, cfg)

            row: dict = {
                "label": label,
                "Frequency": freq,
                "Cost (bps)": cost_bps,
            }
            for col in _REBAL_COLS:
                row[col] = result.metrics.get(col, float("nan"))
            records.append(row)

    df = pd.DataFrame(records)
    df = df.set_index("label")
    return df[["Frequency", "Cost (bps)"] + _REBAL_COLS]
