"""
Volatility Model factor (V) for AMAAM.

Default estimator: Yang-Zhang (YZ), which uses OHLC data and explicitly
captures the overnight gap component (Close → next Open).  YZ is
drift-independent, ~8× more statistically efficient than close-to-close, and
outperformed both the RiskMetrics EWMA baseline and Garman-Klass on IS/OOS
Sharpe in all three measurement windows.

The legacy RiskMetrics EWMA estimator (lambda=0.94) is retained as a named
alternative for comparison and backward-compatibility.

Lower annualised volatility → higher TRank → more likely to be selected.
See Section 3.3 of the specification.
"""

import logging
from typing import Dict

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
    Compute Yang-Zhang annualised volatility for a single asset.

    Yang-Zhang (2000) decomposes total variance into three orthogonal
    components, each estimated from a different aspect of daily price data:

    ``σ²_YZ = σ²_overnight + k · σ²_open_close + (1 − k) · σ²_rogers_satchell``

    Components:
    * **Overnight** — variance of ln(Open_t / Close_{t-1}); captures gap risk
      from macro announcements, earnings, Fed decisions, etc.
    * **Open-to-close** — variance of ln(Close_t / Open_t); intraday drift.
    * **Rogers-Satchell** — Σ[ln(H/C)·ln(H/O) + ln(L/C)·ln(L/O)] / n;
      captures intraday volatility without assuming zero drift.

    The optimal weighting coefficient k (Chou & Wang 2006) minimises the
    estimator's mean-squared error:
    ``k = 0.34 / (1.34 + (n+1)/(n−1))``

    All three components are computed over a *window*-day rolling window
    (min_periods = window so that early NaN rows are not reported).
    The final series is annualised by √252.

    Parameters
    ----------
    ohlc : pd.DataFrame
        Must contain ``Open``, ``High``, ``Low``, ``Close`` columns.
        Index is a DatetimeIndex of trading days.
    window : int
        Rolling window length in trading days (spec default: 84 — matched to
        the momentum lookback so both factors share the same horizon).

    Returns
    -------
    pd.Series
        Annualised Yang-Zhang volatility.  The first ``window − 1`` rows are
        NaN.  Same index as *ohlc*.
    """
    o = np.log(ohlc["Open"])
    h = np.log(ohlc["High"])
    l = np.log(ohlc["Low"])
    c = np.log(ohlc["Close"])
    c_prev = c.shift(1)

    # Chou-Wang (2006) optimal weighting for the open-to-close component.
    k = 0.34 / (1.34 + (window + 1) / (window - 1))

    # Component 1: overnight return variance  ln(O_t / C_{t-1})
    overnight = o - c_prev
    var_overnight = overnight.rolling(window, min_periods=window).var(ddof=1)

    # Component 2: open-to-close return variance  ln(C_t / O_t)
    open_close = c - o
    var_open_close = open_close.rolling(window, min_periods=window).var(ddof=1)

    # Component 3: Rogers-Satchell intraday variance (drift-independent)
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
    Legacy EWMA volatility pipeline: log returns → EWMA variance → SMA smooth → annualise.

    Retained for backward-compatibility and comparative analysis.  The default
    pipeline now uses Yang-Zhang via ``compute_volatility_all_assets``.

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


# =============================================================================
# Public dispatcher — used by the backtest engine
# =============================================================================

def compute_volatility_all_assets(
    data_dict: Dict[str, pd.DataFrame],
    config: ModelConfig,
) -> pd.DataFrame:
    """
    Compute Yang-Zhang volatility for every asset in a data dictionary.

    Yang-Zhang is the default estimator (IS/OOS testing confirmed it
    outperforms both EWMA and Garman-Klass on Sharpe across all measurement
    windows, with no IS-to-OOS rank reversal).

    Parameters
    ----------
    data_dict : Dict[str, pd.DataFrame]
        Mapping of ticker → OHLCV DataFrame.  Must contain ``Open``,
        ``High``, ``Low``, and ``Close`` columns.
    config : ModelConfig
        Model configuration supplying ``yang_zhang_window`` (default: 84).

    Returns
    -------
    pd.DataFrame
        Annualised YZ volatility with dates as index and tickers as columns.
    """
    series = {
        ticker: compute_yang_zhang_vol(df, config.yang_zhang_window)
        for ticker, df in data_dict.items()
    }
    return pd.DataFrame(series)
