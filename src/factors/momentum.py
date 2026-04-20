"""
Absolute Momentum factor (M) for AMAAM.

Computes the Rate of Change on daily closing prices. Two modes are supported:

* **Single lookback** (default): 4-month (84 trading day) ROC, matching the
  original Keller/Giordano specification.
* **Blended lookback**: equal-weight average of ROC across multiple horizons
  (e.g. 1/3/6/12 months). Averaging across horizons reduces sensitivity to
  the specific window choice and captures momentum at different time scales,
  following the approach in Faber (2007) and Antonacci (2014).

The momentum value serves two roles: (1) an input to the TRank ranking formula
and (2) the binary filter that determines whether a selected asset retains its
portfolio weight or redirects it to the hedging sleeve. See Section 3.2 of the
specification.
"""

import logging
from typing import Dict, List

import pandas as pd

logger = logging.getLogger(__name__)


def compute_absolute_momentum(prices: pd.Series, lookback: int) -> pd.Series:
    """
    Compute 4-month Rate of Change (ROC) on daily closing prices.

    ``M = (Price_today / Price_{today - lookback}) - 1``

    The first ``lookback`` rows are NaN because there is no prior price
    ``lookback`` days back.  The month-end value of this series is the one
    consumed by TRank and the momentum filter; daily values are stored so
    the caller can choose any evaluation date.

    Parameters
    ----------
    prices : pd.Series
        Daily closing prices with ``DatetimeIndex``.
    lookback : int
        Number of trading days for the ROC window (spec default: 84).

    Returns
    -------
    pd.Series
        Daily momentum values.  Same index as *prices*.
        Positive value → asset has appreciated over the lookback window.
        Negative value → asset has declined; triggers the momentum filter
        in the allocation module.
    """
    mom = prices / prices.shift(lookback) - 1.0
    mom.name = prices.name
    return mom


def compute_blended_momentum(
    prices: pd.Series,
    lookbacks: List[int],
) -> pd.Series:
    """
    Compute equal-weight average ROC across multiple lookback horizons.

    Each horizon contributes an independent ROC reading; averaging them reduces
    the strategy's sensitivity to any single lookback choice and incorporates
    both short-term (1-month) and long-term (12-month) momentum signals in a
    single factor value.

    ``M_blend = mean(ROC_1m, ROC_3m, ROC_6m, ROC_12m)``

    The first valid value appears at the longest lookback; earlier rows are NaN.
    The sign of the blended value still drives the momentum filter (positive →
    hold, non-positive → redirect to hedging sleeve).

    Parameters
    ----------
    prices : pd.Series
        Daily closing prices with ``DatetimeIndex``.
    lookbacks : List[int]
        Lookback windows in trading days, e.g. [21, 63, 126, 252] for the
        1/3/6/12-month blend.

    Returns
    -------
    pd.Series
        Daily blended momentum values, same index as *prices*.
        NaN until the longest lookback is satisfied.
    """
    rocs = pd.concat(
        [prices / prices.shift(lb) - 1.0 for lb in lookbacks],
        axis=1,
    )
    blended = rocs.mean(axis=1)
    blended.name = prices.name
    return blended


def compute_momentum_all_assets(
    data_dict: Dict[str, pd.DataFrame],
    lookback: int,
) -> pd.DataFrame:
    """
    Compute absolute momentum for every asset in a data dictionary.

    Parameters
    ----------
    data_dict : Dict[str, pd.DataFrame]
        Mapping of ticker → OHLCV DataFrame (must contain a ``Close`` column).
    lookback : int
        Lookback in trading days.

    Returns
    -------
    pd.DataFrame
        DataFrame with dates as index and tickers as columns.  All tickers
        share the same index because the data has been calendar-aligned by
        :func:`~src.data.validator.align_trading_calendar`.
    """
    series = {
        ticker: compute_absolute_momentum(df["Close"], lookback)
        for ticker, df in data_dict.items()
    }
    return pd.DataFrame(series)
