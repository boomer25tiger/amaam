"""
Volatility factor (V) for AMAAM — see spec Section 3.3.

Default estimator is Yang-Zhang (OHLC-based, drift-independent, ~8× more
efficient than close-to-close). The legacy RiskMetrics EWMA estimator is
retained for comparison. Lower vol → higher TRank → more likely selected.
"""

import logging
from typing import Dict, List

import numpy as np
import pandas as pd

from config.default_config import ModelConfig

logger = logging.getLogger(__name__)

# 252 trading days per year — standard equity market convention.
_ANNUALIZATION_FACTOR: int = 252


# =============================================================================
# Yang-Zhang estimator (default)
# =============================================================================

def compute_yang_zhang_vol(
    ohlc: pd.DataFrame,
    window: int,
) -> pd.Series:
    """
    Compute rolling Yang-Zhang annualised volatility for a single asset.

    Notes
    -----
    YZ decomposes variance into three orthogonal terms:
    ``σ²_YZ = σ²_overnight + k·σ²_open_close + (1−k)·σ²_rogers_satchell``
    where k = 0.34 / (1.34 + (n+1)/(n−1)) is the Chou-Wang (2006) MSE-optimal
    weight. The Rogers-Satchell term makes the estimator drift-independent.
    """
    o = np.log(ohlc["Open"])
    h = np.log(ohlc["High"])
    l = np.log(ohlc["Low"])
    c = np.log(ohlc["Close"])
    c_prev = c.shift(1)

    # Chou-Wang (2006) optimal weighting for the open-to-close component.
    k = 0.34 / (1.34 + (window + 1) / (window - 1))

    overnight = o - c_prev
    var_overnight = overnight.rolling(window, min_periods=window).var(ddof=1)

    open_close = c - o
    var_open_close = open_close.rolling(window, min_periods=window).var(ddof=1)

    # Rogers-Satchell (1991): drift-independent intraday variance.
    rs = (h - c) * (h - o) + (l - c) * (l - o)
    var_rs = rs.rolling(window, min_periods=window).mean()

    var_yz = var_overnight + k * var_open_close + (1 - k) * var_rs

    # clip(lower=0) guards against floating-point rounding producing tiny
    # negatives in the RS term when H ≈ L ≈ O ≈ C.
    return np.sqrt(var_yz.clip(lower=0) * _ANNUALIZATION_FACTOR)


# =============================================================================
# RiskMetrics EWMA estimator (legacy / alternative)
# =============================================================================

def compute_ewma_variance(
    returns: pd.Series,
    lambda_param: float,
    init_window: int,
) -> pd.Series:
    """
    Compute RiskMetrics EWMA variance: ``σ²_t = λ·σ²_{t-1} + (1−λ)·r²_{t-1}``.

    Notes
    -----
    Seeded with population variance (ddof=0) over the first ``init_window``
    returns, per the Zangari (1996) RiskMetrics convention. The recursion uses
    the *previous* day's return, so the first ``init_window`` rows are NaN.
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
    Legacy EWMA volatility pipeline: log returns → EWMA variance → SMA smooth → annualise.

    Retained for backward-compatibility; the default pipeline uses Yang-Zhang.

    Notes
    -----
    SMA smoothing is applied to the variance series (not the square-root vol)
    to stay in variance space and avoid Jensen's inequality distortion.
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


# =============================================================================
# Public dispatcher — used by the backtest engine
# =============================================================================

def compute_volatility_all_assets(
    data_dict: Dict[str, pd.DataFrame],
    config: ModelConfig,
) -> pd.DataFrame:
    """
    Dispatch Yang-Zhang volatility computation across all assets and assemble into a DataFrame.

    This is the primary entry point used by the backtest engine.
    """
    series = {
        ticker: compute_yang_zhang_vol(df, config.yang_zhang_window)
        for ticker, df in data_dict.items()
    }
    return pd.DataFrame(series)


def compute_blended_yang_zhang_vol(
    data_dict: Dict[str, pd.DataFrame],
    lookbacks: List[int],
) -> pd.DataFrame:
    """
    Equal-weight blend of Yang-Zhang volatility across multiple window lengths.

    Mirrors the multi-horizon philosophy of blended momentum to reduce
    sensitivity to any single lookback choice.

    Notes
    -----
    Output is NaN until every component window has cleared its warm-up period,
    so the longest lookback governs when valid values begin.
    """
    frames = {
        lb: pd.DataFrame({
            ticker: compute_yang_zhang_vol(df, lb)
            for ticker, df in data_dict.items()
        })
        for lb in lookbacks
    }
    stacked = np.stack([frames[lb].values for lb in lookbacks], axis=0)
    all_valid = np.all(~np.isnan(stacked), axis=0)
    blended_values = np.nanmean(stacked, axis=0)
    blended_values[~all_valid] = np.nan
    ref = next(iter(frames.values()))
    return pd.DataFrame(blended_values, index=ref.index, columns=ref.columns)
