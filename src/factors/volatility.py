"""
Volatility Model factor (V) for AMAAM.

Estimates realized volatility using the J.P. Morgan RiskMetrics EWMA variance
model (lambda=0.94), followed by a 10-day SMA smoothing step and annualization
by sqrt(252). Lower volatility assets receive higher TRank scores, implementing
the risk-management dimension of the ranking engine. See Section 3.3 of the
specification.
"""

import logging
from typing import Dict

import numpy as np
import pandas as pd

from config.default_config import ModelConfig

logger = logging.getLogger(__name__)

# 252 trading days per year — standard equity market convention.
_ANNUALIZATION_FACTOR: int = 252


def compute_ewma_variance(
    returns: pd.Series,
    lambda_param: float,
    init_window: int,
) -> pd.Series:
    """
    Compute EWMA variance using the J.P. Morgan RiskMetrics recursion.

    ``σ²_t = λ · σ²_{t-1} + (1 − λ) · r²_{t-1}``

    Initialisation uses the population variance of the first ``init_window``
    valid returns (Zangari 1996).  The recursive formula is then applied
    forward from position ``init_window + 1``.

    Parameters
    ----------
    returns : pd.Series
        Daily log returns.  The first entry is typically NaN (no prior close).
    lambda_param : float
        Exponential decay factor.  RiskMetrics daily default is 0.94.
    init_window : int
        Number of returns used to seed the variance before the recursion.

    Returns
    -------
    pd.Series
        Daily EWMA variance estimates.  The first ``init_window`` rows are NaN.
        Same index as *returns*.
    """
    r = returns.values.astype(float)
    n = len(r)
    var = np.full(n, np.nan)

    # Locate the first non-NaN return; returns[0] is always NaN for a
    # close-to-close log return series.
    valid_mask = ~np.isnan(r)
    valid_idx = np.where(valid_mask)[0]
    if len(valid_idx) < init_window + 1:
        logger.warning(
            "Insufficient data for EWMA initialisation: need %d valid returns, "
            "got %d.",
            init_window + 1, len(valid_idx),
        )
        return pd.Series(var, index=returns.index)

    start = valid_idx[0]
    init_end = start + init_window  # exclusive upper bound for init slice

    if init_end > n:
        return pd.Series(var, index=returns.index)

    # Seed: population variance of the first init_window returns.
    # Population variance (ddof=0) matches the RiskMetrics convention of
    # treating the full init window as the reference distribution.
    init_returns = r[start:init_end]
    var[init_end - 1] = np.var(init_returns, ddof=0)

    # Recursive EWMA forward from the init_window-th position.
    # σ²_t uses the return from the PREVIOUS day (r_{t-1}).
    for i in range(init_end, n):
        prev_r = r[i - 1]
        if np.isnan(prev_r) or np.isnan(var[i - 1]):
            var[i] = var[i - 1]  # propagate; should not occur on aligned data
        else:
            var[i] = lambda_param * var[i - 1] + (1.0 - lambda_param) * prev_r ** 2

    return pd.Series(var, index=returns.index)


def compute_volatility_model(
    prices: pd.Series,
    lambda_param: float,
    init_window: int,
    smoothing_window: int,
) -> pd.Series:
    """
    Full volatility pipeline: log returns → EWMA variance → SMA smooth → annualise.

    The SMA smoothing step is applied to the *variance* series (not to the
    square-root vol) as specified in the RAAM paper.  Smoothing on variance
    is less distorting than smoothing on volatility because it keeps the
    estimator in variance space throughout.

    Parameters
    ----------
    prices : pd.Series
        Daily closing prices (adjusted).
    lambda_param : float
        EWMA decay factor (0.94 for RiskMetrics daily).
    init_window : int
        Number of returns for EWMA seed (spec default: 20).
    smoothing_window : int
        SMA window applied to the EWMA variance series (spec default: 10).

    Returns
    -------
    pd.Series
        Annualised volatility.  The first ``init_window + smoothing_window - 1``
        rows are NaN.  Same index as *prices*.
    """
    log_returns = np.log(prices / prices.shift(1))
    log_returns.name = prices.name

    ewma_var = compute_ewma_variance(log_returns, lambda_param, init_window)

    # SMA on variance: min_periods=smoothing_window so we only report once
    # we have a full smoothing window of valid variance estimates.
    smoothed_var = ewma_var.rolling(
        window=smoothing_window, min_periods=smoothing_window
    ).mean()

    annualised_vol = np.sqrt(smoothed_var * _ANNUALIZATION_FACTOR)
    annualised_vol.name = prices.name
    return annualised_vol


def compute_volatility_all_assets(
    data_dict: Dict[str, pd.DataFrame],
    config: ModelConfig,
) -> pd.DataFrame:
    """
    Compute volatility for every asset in a data dictionary.

    Parameters
    ----------
    data_dict : Dict[str, pd.DataFrame]
        Mapping of ticker → OHLCV DataFrame (must contain a ``Close`` column).
    config : ModelConfig
        Model configuration supplying ``volatility_lambda``,
        ``volatility_init_window``, and ``volatility_smoothing``.

    Returns
    -------
    pd.DataFrame
        DataFrame with dates as index and tickers as columns.
    """
    series = {
        ticker: compute_volatility_model(
            df["Close"],
            config.volatility_lambda,
            config.volatility_init_window,
            config.volatility_smoothing,
        )
        for ticker, df in data_dict.items()
    }
    return pd.DataFrame(series)
