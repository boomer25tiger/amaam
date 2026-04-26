"""
Keltner Channel Trend Direction factor (T) for AMAAM (Section 3.5).

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
    Compute the SMA of True Range over *period* bars per the spec formula ``ATR_t = (1/42) Σ TR``.

    Notes
    -----
    Uses SMA, not Wilder's exponential smoothing.  TR[0] = H[0] − L[0] is
    always valid (no prior close needed), so pandas accumulates *period* valid
    TRs at bar ``period − 1`` and the first non-NaN ATR appears one bar earlier
    than a naive count would suggest.
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
    Build the asymmetric Keltner Channel bands: ``UB = EMA(Close, upper_span) + ATR``,
    ``LB = EMA(Close, lower_span) − ATR``.

    Notes
    -----
    Different EMA spans per band are intentional — the faster upper EMA
    (63 bars) confirms uptrends sooner, while the slower lower EMA (105 bars)
    demands a sustained decline before firing the downtrend signal.
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
    Compute the primary Keltner Channel trend signal T ∈ {−2, +2} with persistent carry.

    Notes
    -----
    Initialised to −2 (conservative) at the first bar where both bands are valid.
    Execution lag (signal → order) is handled by the backtest engine, not here.
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


def compute_paper_atr_signal(
    high:           pd.Series,
    low:            pd.Series,
    close:          pd.Series,
    atr_period:     int,
    upper_ema_span: int,
    lower_ema_span: int,
) -> pd.Series:
    """
    Paper-literal ATR breakout signal, used to validate against the original publication.

    Notes
    -----
    The lower band diverges from the standard Keltner formulation in two ways:
    its EMA is computed on *High* prices (not Close), and ATR is **added**
    rather than subtracted — placing the band above recent highs so only a
    genuine intraday collapse can fire T = −2.
    """
    atr = compute_atr(high, low, close, atr_period)

    upper_ema = close.ewm(span=upper_ema_span, adjust=False,
                          min_periods=upper_ema_span).mean()
    # Paper-literal: lower band EMA is computed on HIGH prices, ATR added.
    lower_ema = high.ewm(span=lower_ema_span, adjust=False,
                         min_periods=lower_ema_span).mean()

    upper_band = upper_ema + _KELTNER_MULTIPLIER * atr
    lower_band = lower_ema + _KELTNER_MULTIPLIER * atr   # + not -

    h_vals  = high.values
    l_vals  = low.values
    ub_vals = upper_band.values
    lb_vals = lower_band.values
    n       = len(close)

    signal = np.full(n, np.nan)
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
            signal[i] = signal[i - 1]

    return pd.Series(signal, index=close.index)


def compute_sma200_signal(close: pd.Series, period: int = 200) -> pd.Series:
    """
    Faber (2007) SMA trend filter: T = +2 if Close > SMA(period), else T = −2.

    No carry logic — every bar is independently classified with no neutral zone.
    """
    sma = close.rolling(window=period, min_periods=period).mean()
    signal = pd.Series(np.where(close > sma, _SIGNAL_UP, _SIGNAL_DOWN),
                       index=close.index, dtype=float)
    signal[sma.isna()] = np.nan
    return signal


def compute_sma_ratio_signal(
    close: pd.Series,
    period: int = 200,
    upper_threshold: float = 1.01,
    lower_threshold: float = 0.99,
) -> pd.Series:
    """
    SMA trend filter with a ±1 % buffer zone and persistent carry to reduce whipsawing.

    Notes
    -----
    Without a buffer (thresholds = 1.0) this is identical to ``compute_sma_carry_signal``.
    Initialised to −2 at the first valid bar, matching the Keltner conservative default.
    """
    sma = close.rolling(window=period, min_periods=period).mean()
    ratio = close / sma

    n = len(close)
    signal = np.full(n, np.nan)
    first_valid = ratio.first_valid_index()
    if first_valid is None:
        return pd.Series(signal, index=close.index)

    start = close.index.get_loc(first_valid)
    signal[start] = _SIGNAL_INIT

    ratio_vals = ratio.values
    for i in range(start + 1, n):
        if np.isnan(ratio_vals[i]):
            signal[i] = signal[i - 1]
        elif ratio_vals[i] > upper_threshold:
            signal[i] = _SIGNAL_UP
        elif ratio_vals[i] < lower_threshold:
            signal[i] = _SIGNAL_DOWN
        else:
            signal[i] = signal[i - 1]

    return pd.Series(signal, index=close.index)


def compute_sma_carry_signal(
    close: pd.Series,
    period: int = 200,
) -> pd.Series:
    """
    SMA trend signal with carry but no buffer zone, isolating the carry effect from the buffer.
    """
    sma = close.rolling(window=period, min_periods=period).mean()
    ratio = close / sma
    n = len(close)
    signal = np.full(n, np.nan)
    first_valid = ratio.first_valid_index()
    if first_valid is None:
        return pd.Series(signal, index=close.index)
    start = close.index.get_loc(first_valid)
    signal[start] = _SIGNAL_INIT
    ratio_vals = ratio.values
    for i in range(start + 1, n):
        if np.isnan(ratio_vals[i]):
            signal[i] = signal[i - 1]
        elif ratio_vals[i] > 1.0:
            signal[i] = _SIGNAL_UP
        else:
            signal[i] = _SIGNAL_DOWN
    return pd.Series(signal, index=close.index)


def compute_dual_sma_signal(
    close: pd.Series,
    fast: int = 50,
    slow: int = 200,
) -> pd.Series:
    """
    Golden/death cross signal: T = +2 if SMA(fast) > SMA(slow), else T = −2.
    """
    sma_fast = close.rolling(window=fast, min_periods=fast).mean()
    sma_slow = close.rolling(window=slow, min_periods=slow).mean()
    signal = pd.Series(
        np.where(sma_fast > sma_slow, _SIGNAL_UP, _SIGNAL_DOWN),
        index=close.index, dtype=float,
    )
    signal[sma_slow.isna()] = np.nan
    return signal


def compute_donchian_signal(close: pd.Series, period: int = 200) -> pd.Series:
    """
    Donchian channel breakout signal: T = +2 on a new *period*-day high, T = −2 on a new low,
    otherwise carry previous state.
    """
    roll_max = close.rolling(window=period, min_periods=period).max()
    roll_min = close.rolling(window=period, min_periods=period).min()

    n = len(close)
    signal = np.full(n, np.nan)
    first_valid = int(np.where(~np.isnan(roll_max.values))[0][0]) if roll_max.notna().any() else None
    if first_valid is None:
        return pd.Series(signal, index=close.index)

    signal[first_valid] = _SIGNAL_INIT
    c_vals   = close.values
    max_vals = roll_max.values
    min_vals = roll_min.values

    for i in range(first_valid + 1, n):
        if np.isnan(max_vals[i]):
            signal[i] = signal[i - 1]
        elif c_vals[i] >= max_vals[i]:
            signal[i] = _SIGNAL_UP
        elif c_vals[i] <= min_vals[i]:
            signal[i] = _SIGNAL_DOWN
        else:
            signal[i] = signal[i - 1]

    return pd.Series(signal, index=close.index)


def compute_tsmom_signal(close: pd.Series, lookback: int = 252) -> pd.Series:
    """
    Moskowitz, Ooi & Pedersen (2012) TSMOM: T = sign of the trailing *lookback*-day return.
    """
    roc = close / close.shift(lookback) - 1.0
    signal = pd.Series(
        np.where(roc > 0, _SIGNAL_UP, _SIGNAL_DOWN),
        index=close.index, dtype=float,
    )
    signal[roc.isna()] = np.nan
    return signal


def compute_rolling_sharpe_signal(
    close: pd.Series,
    lookback: int = 126,
) -> pd.Series:
    """
    Trend signal based on the sign of the rolling Sharpe ratio of daily returns.

    Notes
    -----
    Normalising by rolling volatility means a steady low-vol uptrend and a
    sharp high-vol uptrend of equal Sharpe score identically — unlike TSMOM,
    which only looks at raw return sign.
    """
    rets   = close.pct_change(fill_method=None)
    roll_m = rets.rolling(window=lookback, min_periods=lookback).mean()
    roll_s = rets.rolling(window=lookback, min_periods=lookback).std()
    sharpe = roll_m / roll_s
    signal = pd.Series(
        np.where(sharpe > 0, _SIGNAL_UP, _SIGNAL_DOWN),
        index=close.index, dtype=float,
    )
    signal[sharpe.isna()] = np.nan
    return signal


def compute_r2_trend_signal(
    close: pd.Series,
    lookback: int = 126,
    r2_threshold: float = 0.65,
) -> pd.Series:
    """
    Trend quality signal: T = +2 only when a rolling OLS of log-prices on time shows both
    positive slope *and* R² ≥ r2_threshold, suppressing signals in choppy sideways markets.

    Notes
    -----
    The R² gate is the key difference from TSMOM: a near-zero positive slope in a noisy
    market does not fire +2 unless the price path is actually consistent with a trend.
    """
    log_prices = np.log(close.values.astype(float))
    n          = len(close)
    signal     = np.full(n, np.nan)

    # Pre-build the time regressor and its mean/variance for speed.
    t_idx   = np.arange(lookback, dtype=float)
    t_mean  = t_idx.mean()
    t_var   = ((t_idx - t_mean) ** 2).sum()

    for i in range(lookback - 1, n):
        y      = log_prices[i - lookback + 1 : i + 1]
        if np.any(np.isnan(y)):
            signal[i] = signal[i - 1] if i > 0 else _SIGNAL_INIT
            continue
        y_mean = y.mean()
        cov_ty = ((t_idx - t_mean) * (y - y_mean)).sum()
        beta   = cov_ty / t_var
        y_var  = ((y - y_mean) ** 2).sum()
        r2     = (cov_ty ** 2) / (t_var * y_var) if y_var > 1e-12 else 0.0
        if beta > 0 and r2 >= r2_threshold:
            signal[i] = _SIGNAL_UP
        else:
            signal[i] = _SIGNAL_DOWN

    return pd.Series(signal, index=close.index)


def compute_macd_signal(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
) -> pd.Series:
    """
    MACD zero-line crossover: T = +2 if EMA(fast) > EMA(slow), else T = −2.

    Notes
    -----
    Uses zero-line cross rather than signal-line cross to minimise free
    parameters and reduce in-sample overfitting risk.
    """
    ema_fast = close.ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_slow = close.ewm(span=slow, adjust=False, min_periods=slow).mean()
    macd     = ema_fast - ema_slow
    signal   = pd.Series(
        np.where(macd > 0, _SIGNAL_UP, _SIGNAL_DOWN),
        index=close.index, dtype=float,
    )
    signal[ema_slow.isna()] = np.nan
    return signal


def compute_trend_ensemble(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    atr_period: int = 42,
    atr_upper_lookback: int = 63,
    atr_lower_lookback: int = 105,
    sma_period: int = 200,
    macd_fast: int = 12,
    macd_slow: int = 26,
) -> pd.Series:
    """
    Equal-weight average of MACD, Keltner, and SMA-ratio signals across three time horizons.

    Notes
    -----
    Output lies in [−2, +2]; intermediate values (±0.67) arise when exactly
    two of three components agree.  NaN is propagated until all three have
    completed warm-up so the average is never computed on a partial ensemble.
    """
    t_macd    = compute_macd_signal(close, fast=macd_fast, slow=macd_slow)
    t_keltner = compute_trend_signal(
        high, low, close, atr_period, atr_upper_lookback, atr_lower_lookback
    )
    t_sma     = compute_sma_ratio_signal(close, period=sma_period)

    # Stack into a DataFrame and take the row-wise mean.  skipna=False ensures
    # the composite is NaN whenever any component is still in warm-up, so the
    # three-signal average is never computed on an incomplete ensemble.
    components = pd.DataFrame(
        {"macd": t_macd, "keltner": t_keltner, "sma": t_sma},
        index=close.index,
    )
    return components.mean(axis=1, skipna=False)


def compute_trend_all_assets(
    data_dict: Dict[str, pd.DataFrame],
    config:    ModelConfig,
) -> pd.DataFrame:
    """
    Apply the trend method named by ``config.trend_method`` to every asset and return a
    tickers × dates DataFrame of T values.
    """
    method = config.trend_method

    def _signal_for(ticker: str, df: pd.DataFrame) -> pd.Series:
        close = df["Close"]

        if method == "keltner":
            return compute_trend_signal(
                df["High"], df["Low"], close,
                config.atr_period, config.atr_upper_lookback, config.atr_lower_lookback,
            )
        if method == "paper_atr":
            return compute_paper_atr_signal(
                df["High"], df["Low"], close,
                config.atr_period, config.atr_upper_lookback, config.atr_lower_lookback,
            )
        if method == "sma200":
            return compute_sma200_signal(close)
        if method == "sma_ratio":
            return compute_sma_ratio_signal(close)
        if method == "sma_carry":
            return compute_sma_carry_signal(close, period=200)
        if method == "dual_sma":
            return compute_dual_sma_signal(close)
        if method == "donchian":
            return compute_donchian_signal(close)
        if method == "tsmom":
            return compute_tsmom_signal(close)
        if method == "rolling_sharpe":
            return compute_rolling_sharpe_signal(close)
        if method == "r2_trend":
            return compute_r2_trend_signal(close)
        if method == "macd":
            return compute_macd_signal(close)
        if method == "ensemble":
            return compute_trend_ensemble(
                df["High"], df["Low"], close,
                config.atr_period, config.atr_upper_lookback, config.atr_lower_lookback,
            )
        raise ValueError(
            f"Unknown trend_method '{method}'. "
            f"Choose from: keltner, paper_atr, sma200, sma_ratio, sma_carry, dual_sma, "
            f"donchian, tsmom, rolling_sharpe, r2_trend, macd, ensemble."
        )

    series = {ticker: _signal_for(ticker, df) for ticker, df in data_dict.items()}
    return pd.DataFrame(series)
