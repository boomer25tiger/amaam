"""
Keltner Channel Trend Direction factor (T) for AMAAM.

Replaces the original ATR rolling-max breakout system with an asymmetric
Keltner Channel (Variant A, multiplier k = 1.0).  The upper band uses a
faster 63-period EMA so the system confirms uptrends more readily; the lower
band uses a slower 105-period EMA so a sustained decline is required to flip
the signal to −2.  ATR is a simple 42-period rolling average of True Range
(SMA, not Wilder's smoothing).  Both bands widen when volatility rises, so a
more decisive price move is required to change state in turbulent markets.

The signal is persistent (carry-forward): it is evaluated at every daily bar
but only the last value of each calendar month enters TRank.  See Section 3.5
of the specification.

Band formulas
─────────────
  ATR_t  = (1/42) × Σ_{i=0}^{41} TR_{t-i}   (SMA of True Range)
  UB_t   = EMA(Close, 63)_t + 1.0 × ATR_t
  LB_t   = EMA(Close, 105)_t − 1.0 × ATR_t

Signal rules (persistent carry)
────────────────────────────────
  T_t = +2   if High_t  > UB_t    (uptrend breakout)
  T_t = −2   if Low_t   < LB_t    (downtrend breakdown)
  T_t = T_{t-1}  otherwise        (state unchanged)

Key behavioural property
────────────────────────
With EMA-centred bands the lower band sits BELOW the EMA centre line,
meaning price must fall a full ATR below its 105-day moving average to flip
T to −2.  Conversely, the upper band sits ABOVE the 63-day EMA, meaning
price must rally a full ATR above it to flip T to +2.  Once flipped, the
signal persists until the opposite band is breached — unlike the original
rolling-max formulation where T = −2 was structurally permanent.
"""

import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from config.default_config import ModelConfig

logger = logging.getLogger(__name__)

# Signal values as specified in Section 3.5.
_SIGNAL_UP:   float = 2.0
_SIGNAL_DOWN: float = -2.0

# Conservative initial state: assume no confirmed trend until a breakout fires.
_SIGNAL_INIT: float = -2.0

# Keltner Channel ATR multiplier (k = 1.0 per spec Section 3.5).
_KELTNER_MULTIPLIER: float = 1.0


def compute_atr(
    high:   pd.Series,
    low:    pd.Series,
    close:  pd.Series,
    period: int,
) -> pd.Series:
    """
    Compute the 42-period simple moving average of True Range.

    True Range at bar t:
    ``TR_t = max(H_t − L_t, |H_t − C_{t-1}|, |L_t − C_{t-1}|)``

    ATR is the rolling SMA of TR over *period* bars (min_periods = period so
    that the series is NaN until a full window is available).  This differs
    from Wilder's exponential smoothing; the SMA form matches the spec formula
    ``ATR_t = (1/42) Σ TR``.

    Parameters
    ----------
    high : pd.Series
        Daily high prices.
    low : pd.Series
        Daily low prices.
    close : pd.Series
        Daily closing prices.
    period : int
        Rolling window length (spec default: 42).

    Returns
    -------
    pd.Series
        ATR values.  The first ``period − 1`` rows are NaN.  Note that
        TR[0] = H[0] − L[0] is always valid (the intraday range requires no
        prior close); ``pandas max(axis=1, skipna=True)`` returns the HL
        component even when the gap terms are NaN.  Therefore the rolling
        window accumulates *period* valid TRs at bar ``period − 1`` (not
        ``period``), and the first valid ATR is at that position.
        Same index as *close*.
    """
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low  - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    # Rolling SMA: pandas counts non-NaN values; because TR[0] is always NaN
    # (no previous close), the first complete window of `period` valid TRs
    # occurs at bar `period`, matching the spec's summation index.
    return tr.rolling(window=period, min_periods=period).mean()


def compute_atr_bands(
    high:           pd.Series,
    low:            pd.Series,
    close:          pd.Series,
    atr_period:     int,
    upper_ema_span: int,
    lower_ema_span: int,
) -> Tuple[pd.Series, pd.Series]:
    """
    Compute the asymmetric Keltner Channel upper and lower bands.

    ``UB_t = EMA(Close, upper_ema_span)_t + k × ATR_t``
    ``LB_t = EMA(Close, lower_ema_span)_t − k × ATR_t``

    where ``k = _KELTNER_MULTIPLIER = 1.0``.

    The asymmetry (different EMA spans for each band) is intentional: the
    faster upper EMA (63 periods) makes the system quicker to confirm
    uptrends, while the slower lower EMA (105 periods) requires a more
    sustained decline before the downtrend signal fires.

    Parameters
    ----------
    high, low, close : pd.Series
        Daily OHLC components.
    atr_period : int
        SMA window for ATR (spec default: 42).
    upper_ema_span : int
        EMA span for the upper band (spec default: 63).
    lower_ema_span : int
        EMA span for the lower band (spec default: 105).

    Returns
    -------
    Tuple[pd.Series, pd.Series]
        ``(upper_band, lower_band)`` with the same index as *close*.
        Values are NaN until both EMA and ATR have enough history
        (binding constraint: ``lower_ema_span`` bars).
    """
    atr = compute_atr(high, low, close, atr_period)

    # min_periods enforces that each band is NaN until the full EMA span
    # has been observed, preventing unreliable early estimates from entering
    # the signal logic.
    upper_ema = close.ewm(span=upper_ema_span, adjust=False,
                          min_periods=upper_ema_span).mean()
    lower_ema = close.ewm(span=lower_ema_span, adjust=False,
                          min_periods=lower_ema_span).mean()

    upper_band = upper_ema + _KELTNER_MULTIPLIER * atr
    lower_band = lower_ema - _KELTNER_MULTIPLIER * atr

    return upper_band, lower_band


def compute_trend_signal(
    high:           pd.Series,
    low:            pd.Series,
    close:          pd.Series,
    atr_period:     int,
    upper_ema_span: int,
    lower_ema_span: int,
) -> pd.Series:
    """
    Compute the daily Keltner Channel Trend Direction signal T ∈ {−2, +2}.

    State machine rules (evaluated at each bar's close):
    * ``High_t > UB_t`` → T = +2 (uptrend breakout confirmed)
    * ``Low_t  < LB_t`` → T = −2 (downtrend breakdown confirmed)
    * Otherwise         → T retains its previous value (persistent carry)

    The signal is initialised to ``_SIGNAL_INIT`` (−2) at the first bar
    where both bands carry valid values.  The one-day execution lag between
    signal computation and order placement (Section 5.3) is handled by the
    backtest engine, not here.

    Parameters
    ----------
    high, low, close : pd.Series
        Daily OHLC components.
    atr_period : int
        ATR rolling window (spec default: 42).
    upper_ema_span : int
        EMA span for the upper band (spec default: 63).
    lower_ema_span : int
        EMA span for the lower band (spec default: 105).

    Returns
    -------
    pd.Series
        Daily T values (+2 or −2).  NaN before the first valid band date.
        Same index as *close*.
    """
    upper_band, lower_band = compute_atr_bands(
        high, low, close, atr_period, upper_ema_span, lower_ema_span
    )

    h_vals  = high.values
    l_vals  = low.values
    ub_vals = upper_band.values
    lb_vals = lower_band.values
    n       = len(close)

    signal = np.full(n, np.nan)

    # Find the first position where both bands are simultaneously valid.
    first_valid = np.where(~(np.isnan(ub_vals) | np.isnan(lb_vals)))[0]
    if len(first_valid) == 0:
        return pd.Series(signal, index=close.index)

    start = first_valid[0]
    signal[start] = _SIGNAL_INIT

    for i in range(start + 1, n):
        if np.isnan(ub_vals[i]) or np.isnan(lb_vals[i]):
            signal[i] = signal[i - 1]
            continue

        if h_vals[i] > ub_vals[i]:
            signal[i] = _SIGNAL_UP
        elif l_vals[i] < lb_vals[i]:
            signal[i] = _SIGNAL_DOWN
        else:
            signal[i] = signal[i - 1]   # persist current state

    return pd.Series(signal, index=close.index)


def compute_trend_all_assets(
    data_dict: Dict[str, pd.DataFrame],
    config:    ModelConfig,
) -> pd.DataFrame:
    """
    Compute the Keltner Channel trend signal for every asset in a data dict.

    Parameters
    ----------
    data_dict : Dict[str, pd.DataFrame]
        Mapping of ticker → OHLCV DataFrame (must contain ``High``, ``Low``,
        and ``Close`` columns).
    config : ModelConfig
        Model configuration supplying ``atr_period``, ``atr_upper_lookback``
        (= upper EMA span, 63), and ``atr_lower_lookback`` (= lower EMA
        span, 105).

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
