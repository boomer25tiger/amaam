"""
Benchmark portfolio construction for AMAAM.

Builds the three comparison benchmarks used throughout the backtest analysis:
(1) SPY buy-and-hold, (2) 60/40 SPY/AGG portfolio rebalanced monthly, and
(3) equal-weighted passive 7Twelve portfolio (12 ETFs) rebalanced monthly.
All benchmarks use the same execution assumptions as AMAAM (signal on last
trading day, implemented next close). See Sections 2.4 and 9.15 of the
specification.
"""

import logging
from typing import Dict, List

import pandas as pd

logger = logging.getLogger(__name__)

# Canonical 7Twelve ETF list from Section 2.4 / config/etf_universe.py.
# IGOV is excluded from the processed dataset (zero-volume issues); the
# remaining 11 tickers are equal-weighted and renormalised to sum to 1.0.
_SEVEN_TWELVE_TICKERS: List[str] = [
    "VV", "IJH", "IJR", "EFA", "EEM", "RWR",
    "DBC", "AGG", "TIP", "SHY", "GLD",
    # "IGOV" excluded — not present in processed data (see Phase 1 notes)
]


def _build_close_matrix(data_dict: Dict[str, pd.DataFrame], tickers: List[str]) -> pd.DataFrame:
    """Return a DataFrame of adjusted close prices for *tickers*, forward-filled."""
    available = [t for t in tickers if t in data_dict]
    missing = [t for t in tickers if t not in data_dict]
    if missing:
        logger.warning("Benchmark tickers not in data_dict (skipped): %s", missing)
    closes = pd.DataFrame({t: data_dict[t]["Close"] for t in available})
    return closes.ffill()


def _monthly_rebalanced_returns(
    closes: pd.DataFrame,
    weights: Dict[str, float],
    start: str,
    end: str,
) -> pd.Series:
    """
    Compute monthly returns for a fixed-weight portfolio rebalanced monthly.

    Each month's return is computed using the ONE-DAY lag rule from Section 5.3:
    signal is the last trading day of month M, allocation is executed at the
    next trading day's close, and held until the following month's execution day.

    Parameters
    ----------
    closes : pd.DataFrame
        Daily adjusted-close prices, columns = tickers.
    weights : Dict[str, float]
        Target weights (must sum to 1.0; normalised internally if not).
    start, end : str
        Backtest window (inclusive).

    Returns
    -------
    pd.Series
        Monthly portfolio returns indexed by the execution date (first trading
        day of each holding month).
    """
    tickers = [t for t in weights if t in closes.columns]
    w_series = pd.Series({t: weights[t] for t in tickers})
    w_series /= w_series.sum()   # normalise in case of missing tickers

    # Narrow to the requested window plus one day of buffer for the lag.
    closes = closes.loc[start:end, tickers].ffill()
    if closes.empty:
        return pd.Series(dtype=float)

    # Month-end (signal) dates within the window.
    signal_dates = [
        g.index.max()
        for _, g in closes.groupby(closes.index.to_period("M"))
    ]

    # Map each signal date to the next trading day (execution date).
    sorted_dates = closes.index.tolist()
    date_to_idx = {d: i for i, d in enumerate(sorted_dates)}

    def _next_day(d: pd.Timestamp) -> pd.Timestamp | None:
        idx = date_to_idx.get(d)
        return sorted_dates[idx + 1] if idx is not None and idx + 1 < len(sorted_dates) else None

    exec_dates = {sig: _next_day(sig) for sig in signal_dates}

    monthly_returns = {}
    for i, sig in enumerate(signal_dates[:-1]):
        exec0 = exec_dates.get(sig)
        exec1 = exec_dates.get(signal_dates[i + 1])
        if exec0 is None or exec1 is None:
            continue
        if exec0 not in date_to_idx or exec1 not in date_to_idx:
            continue

        asset_rets = closes.loc[exec1] / closes.loc[exec0] - 1.0
        port_ret = float((w_series * asset_rets).sum())
        monthly_returns[exec1] = port_ret

    return pd.Series(monthly_returns)


def compute_spy_benchmark(
    data_dict: Dict[str, pd.DataFrame],
    start: str,
    end: str,
) -> pd.Series:
    """
    Buy-and-hold SPY monthly return series.

    No rebalancing; the portfolio is 100 % SPY throughout.  Returns use the
    same one-day execution lag as AMAAM (Section 5.3).

    Parameters
    ----------
    data_dict : Dict[str, pd.DataFrame]
        Processed OHLCV data dictionary.
    start, end : str
        Backtest window (inclusive).

    Returns
    -------
    pd.Series
        Monthly SPY returns indexed by execution date.
    """
    closes = _build_close_matrix(data_dict, ["SPY"])
    if "SPY" not in closes.columns:
        raise KeyError("SPY not found in data_dict; cannot build buy-and-hold benchmark.")
    return _monthly_rebalanced_returns(closes, {"SPY": 1.0}, start, end)


def compute_sixty_forty(
    data_dict: Dict[str, pd.DataFrame],
    start: str,
    end: str,
) -> pd.Series:
    """
    60/40 SPY+AGG monthly-rebalanced benchmark.

    Targets 60 % SPY / 40 % AGG, rebalanced back to target at each month-end
    signal date with the one-day execution lag.

    Parameters
    ----------
    data_dict : Dict[str, pd.DataFrame]
        Processed OHLCV data dictionary.
    start, end : str
        Backtest window (inclusive).

    Returns
    -------
    pd.Series
        Monthly 60/40 portfolio returns indexed by execution date.
    """
    closes = _build_close_matrix(data_dict, ["SPY", "AGG"])
    return _monthly_rebalanced_returns(closes, {"SPY": 0.60, "AGG": 0.40}, start, end)


def compute_seven_twelve(
    data_dict: Dict[str, pd.DataFrame],
    start: str,
    end: str,
) -> pd.Series:
    """
    Passive 7Twelve equal-weighted monthly-rebalanced benchmark.

    The 7Twelve portfolio normally comprises 12 ETFs; IGOV is excluded from
    this implementation because it is not present in the processed dataset
    (zero-volume issues identified in Phase 1).  The remaining 11 tickers are
    equal-weighted and renormalised to sum to 1.0.

    Parameters
    ----------
    data_dict : Dict[str, pd.DataFrame]
        Processed OHLCV data dictionary.
    start, end : str
        Backtest window (inclusive).

    Returns
    -------
    pd.Series
        Monthly 7Twelve portfolio returns indexed by execution date.
    """
    available = [t for t in _SEVEN_TWELVE_TICKERS if t in data_dict]
    if not available:
        raise KeyError("No 7Twelve tickers found in data_dict.")
    w = 1.0 / len(available)
    weights = {t: w for t in available}
    closes = _build_close_matrix(data_dict, available)
    return _monthly_rebalanced_returns(closes, weights, start, end)
