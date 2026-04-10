"""
Data loader for AMAAM.

Single entry point for all downstream modules that need market data. Loads
validated, calendar-aligned CSV files from the processed data directory,
computes daily log or simple returns, and provides helpers for extracting
month-end rebalancing dates. See Section 9.5 of the specification.
"""

import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def load_validated_data(data_dir: Path) -> Dict[str, pd.DataFrame]:
    """
    Load validated, calendar-aligned OHLCV data from the processed directory.

    This is the single entry point for all downstream modules (factor
    computation, backtesting, live signal generation).  Routing all reads
    through here ensures that any future change to the storage format only
    requires updating this one function.

    Parameters
    ----------
    data_dir : Path
        Directory containing processed ``{TICKER}.csv`` files produced by
        ``scripts/download_data.py``.

    Returns
    -------
    Dict[str, pd.DataFrame]
        Mapping of ticker → OHLCV DataFrame with ``DatetimeIndex``.

    Raises
    ------
    FileNotFoundError
        If *data_dir* does not exist or contains no CSV files.
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Processed data directory not found: {data_dir}\n"
            "Run scripts/download_data.py first."
        )

    csv_files = sorted(data_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in {data_dir}.\n"
            "Run scripts/download_data.py first."
        )

    data: Dict[str, pd.DataFrame] = {}
    for csv_path in csv_files:
        ticker = csv_path.stem
        df = pd.read_csv(csv_path, index_col="Date", parse_dates=True)
        data[ticker] = df
        logger.debug("Loaded %s: %d rows.", ticker, len(df))

    logger.info("Loaded %d tickers from %s.", len(data), data_dir)
    return data


def get_returns(
    data_dict: Dict[str, pd.DataFrame],
    return_type: str = "log",
) -> Dict[str, pd.Series]:
    """
    Compute daily returns from the ``Close`` column of each ticker's DataFrame.

    Log returns are the default because they are time-additive and make the
    EWMA variance recursion in the volatility factor numerically stable.
    Simple (arithmetic) returns are needed for the backtesting engine's equity
    curve and performance metric calculations, where dollar compounding applies.

    Parameters
    ----------
    data_dict : Dict[str, pd.DataFrame]
        Mapping of ticker → OHLCV DataFrame.
    return_type : str, optional
        ``"log"`` for continuously compounded returns (default) or
        ``"simple"`` for arithmetic percentage returns.

    Returns
    -------
    Dict[str, pd.Series]
        Mapping of ticker → daily return ``Series``.  The first row of each
        series is ``NaN`` by construction (no prior close available).

    Raises
    ------
    ValueError
        If *return_type* is not ``"log"`` or ``"simple"``.
    """
    if return_type not in ("log", "simple"):
        raise ValueError(
            f"return_type must be 'log' or 'simple', got '{return_type!r}'."
        )

    returns: Dict[str, pd.Series] = {}
    for ticker, df in data_dict.items():
        close = df["Close"]
        if return_type == "log":
            ret = np.log(close / close.shift(1))
        else:
            ret = close.pct_change()
        ret.name = ticker
        returns[ticker] = ret

    return returns


def get_monthly_dates(data_dict: Dict[str, pd.DataFrame]) -> List[pd.Timestamp]:
    """
    Return the last trading day of each calendar month present in the dataset.

    Groups the actual dates in the data by (year, month) and takes the maximum
    date within each group.  Because the data is already aligned to the NYSE
    calendar by :func:`~src.data.validator.align_trading_calendar`, the
    per-group maximum is guaranteed to be a real trading day.

    Parameters
    ----------
    data_dict : Dict[str, pd.DataFrame]
        Mapping of ticker → OHLCV DataFrame.  All DataFrames must share the
        same ``DatetimeIndex`` (guaranteed after calendar alignment).

    Returns
    -------
    List[pd.Timestamp]
        Timestamps of the last trading day of each month, sorted ascending.
    """
    if not data_dict:
        return []

    # All tickers share the same index after alignment; use any one.
    index: pd.DatetimeIndex = next(iter(data_dict.values())).index

    date_series = pd.Series(index, index=index)
    # to_period("M") groups by calendar month regardless of day count.
    monthly_last = (
        date_series
        .groupby(date_series.index.to_period("M"))
        .last()
    )

    return list(monthly_last.values)
