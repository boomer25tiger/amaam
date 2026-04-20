"""
Absolute Momentum factor (M) for AMAAM.

Computes the 4-month (84 trading day) Rate of Change on daily closing prices.
The momentum value serves two roles: (1) an input to the TRank ranking formula
and (2) the binary filter that determines whether a selected asset retains its
portfolio weight or redirects it to the hedging sleeve. See Section 3.2 of the
specification.
"""

import logging
from typing import Dict

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
