"""
ATR Trend/Breakout System factor (T) for AMAAM.

Implements a daily ATR-based breakout signal that assigns T = +2 (uptrend) when
today's high exceeds the upper band, T = -2 (downtrend) when today's low falls
below the lower band, and retains the prior value otherwise. The signal captures
directional bias and enters TRank as a raw value (not ranked). See Section 3.5
of the specification for band definitions and sign-convention resolution notes.
"""

import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from config.default_config import ModelConfig

logger = logging.getLogger(__name__)

# Signal values as specified in Section 3.5.
_SIGNAL_UP: float = 2.0
_SIGNAL_DOWN: float = -2.0

# Conservative initial state: assume no trend until an uptrend breakout
# is explicitly confirmed.  -2 means the asset is not in a confirmed uptrend,
# so it receives no trend bonus in TRank until it earns it.
_SIGNAL_INIT: float = -2.0


def compute_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int,
) -> pd.Series:
    """
    Compute Wilder's Average True Range (ATR).

    True Range at bar t:
    ``TR_t = max(H_t − L_t, |H_t − C_{t-1}|, |L_t − C_{t-1}|)``

    ATR is initialised as the SMA of the first *period* True Range values,
    then updated via Wilder's exponential smoothing:
    ``ATR_t = (ATR_{t-1} · (period − 1) + TR_t) / period``

    This is equivalent to EMA with α = 1/period.  The SMA seed is the
    standard Wilder initialisation (Wilder 1978).

    Parameters
    ----------
    high : pd.Series
        Daily high prices.
    low : pd.Series
        Daily low prices.
    close : pd.Series
        Daily closing prices.
    period : int
        ATR smoothing period (spec default: 42).

    Returns
    -------
    pd.Series
        ATR values.  The first *period* rows are NaN.  Same index as *close*.
    """
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    n = len(close)
    atr = np.full(n, np.nan)
    tr_vals = tr.values

    # Need at least period+1 bars (1 for prev_close + period for SMA seed).
    if n <= period:
        return pd.Series(atr, index=close.index)

    # TR[0] is NaN (no previous close).  Seed with SMA of TR[1..period].
    atr[period] = np.nanmean(tr_vals[1 : period + 1])

    # Wilder's smoothing from period+1 onward.
    for i in range(period + 1, n):
        if not np.isnan(tr_vals[i]) and not np.isnan(atr[i - 1]):
            atr[i] = (atr[i - 1] * (period - 1) + tr_vals[i]) / period
        else:
            atr[i] = atr[i - 1]

    return pd.Series(atr, index=close.index)


def compute_atr_bands(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    atr_period: int,
    upper_lookback: int,
    lower_lookback: int,
) -> Tuple[pd.Series, pd.Series]:
    """
    Compute the upper and lower ATR breakout bands.

    ``Upper Band = ATR(atr_period) + Highest Close(upper_lookback)``
    ``Lower Band = ATR(atr_period) + Highest Low(lower_lookback)``

    The "Highest Close" and "Highest Low" are rolling maxima over the
    respective lookback windows.  Adding the ATR provides a volatility-scaled
    buffer that reduces false breakout signals in choppy markets.

    The lower band uses the HIGHEST LOW (rolling max of lows) rather than the
    lowest low, so a breakdown is signalled when price falls below what was
    recently the *best* support level — a stronger bearish indication than
    simply making a new low.

    Parameters
    ----------
    high, low, close : pd.Series
        Daily OHLC components.
    atr_period : int
        Wilder's ATR period (spec default: 42).
    upper_lookback : int
        Rolling window for highest close (spec default: 63 trading days).
    lower_lookback : int
        Rolling window for highest low (spec default: 105 trading days).

    Returns
    -------
    Tuple[pd.Series, pd.Series]
        ``(upper_band, lower_band)`` with the same index as *close*.
    """
    atr = compute_atr(high, low, close, atr_period)

    highest_close = close.rolling(window=upper_lookback, min_periods=upper_lookback).max()
    highest_low = low.rolling(window=lower_lookback, min_periods=lower_lookback).max()

    upper_band = atr + highest_close
    lower_band = atr + highest_low

    return upper_band, lower_band


def compute_trend_signal(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    atr_period: int,
    upper_lookback: int,
    lower_lookback: int,
) -> pd.Series:
    """
    Compute the daily ATR Trend/Breakout signal T ∈ {−2, +2}.

    State machine rules (evaluated at each bar's close):
    * ``High_t > Upper Band_t`` → T = +2 (uptrend breakout confirmed)
    * ``Low_t  < Lower Band_t`` → T = −2 (downtrend / breakdown confirmed)
    * Otherwise                 → T retains its previous value

    The signal is initialised to ``_SIGNAL_INIT`` (−2) at the first bar
    where valid band values are available.  Per Section 3.5, the signal
    computed from bar t takes effect at bar t (not shifted), because the
    backtest engine already implements a one-day execution lag between
    signal computation and order placement (Section 3.7).

    Sign-convention note (Section 3.5):
    In TRank the T term appears as ``−wT · T``.  T = +2 (uptrend) subtracts
    more from TRank, which *lowers* the score.  Since higher TRank = better,
    this would *penalise* uptrending assets — the opposite of the intent.
    During Phase 3 (trank.py), validate the sign convention against a known
    uptrend period (e.g., XLK 2019) and flip the sign in the TRank formula
    if needed.  The raw signal values here (+2 / −2) are correct per the
    paper; the resolution lives in trank.py, not here.

    Parameters
    ----------
    high, low, close : pd.Series
        Daily OHLC components.
    atr_period : int
        Wilder's ATR period.
    upper_lookback : int
        Lookback for highest close in upper band.
    lower_lookback : int
        Lookback for highest low in lower band.

    Returns
    -------
    pd.Series
        Daily T values (+2 or −2).  NaN before the first valid band date.
        Same index as *close*.
    """
    upper_band, lower_band = compute_atr_bands(
        high, low, close, atr_period, upper_lookback, lower_lookback
    )

    h = high.values
    l = low.values
    ub = upper_band.values
    lb = lower_band.values
    n = len(close)

    signal = np.full(n, np.nan)

    # Find the first position where both bands are valid.
    first_valid = np.where(~(np.isnan(ub) | np.isnan(lb)))[0]
    if len(first_valid) == 0:
        return pd.Series(signal, index=close.index)

    start = first_valid[0]
    signal[start] = _SIGNAL_INIT

    for i in range(start + 1, n):
        if np.isnan(ub[i]) or np.isnan(lb[i]):
            signal[i] = signal[i - 1]
            continue

        if h[i] > ub[i]:
            signal[i] = _SIGNAL_UP
        elif l[i] < lb[i]:
            signal[i] = _SIGNAL_DOWN
        else:
            signal[i] = signal[i - 1]  # persist current state

    return pd.Series(signal, index=close.index)


def compute_trend_all_assets(
    data_dict: Dict[str, pd.DataFrame],
    config: ModelConfig,
) -> pd.DataFrame:
    """
    Compute the ATR trend signal for every asset in a data dictionary.

    Parameters
    ----------
    data_dict : Dict[str, pd.DataFrame]
        Mapping of ticker → OHLCV DataFrame (must contain ``High``, ``Low``,
        ``Close`` columns).
    config : ModelConfig
        Model configuration supplying ``atr_period``, ``atr_upper_lookback``,
        and ``atr_lower_lookback``.

    Returns
    -------
    pd.DataFrame
        DataFrame with dates as index and tickers as columns.
        Each cell is +2.0 or −2.0 (or NaN during the warm-up period).
    """
    series = {
        ticker: compute_trend_signal(
            df["High"],
            df["Low"],
            df["Close"],
            config.atr_period,
            config.atr_upper_lookback,
            config.atr_lower_lookback,
        )
        for ticker, df in data_dict.items()
    }
    return pd.DataFrame(series)
