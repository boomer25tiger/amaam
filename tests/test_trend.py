"""
Unit tests for src/factors/trend.py (Keltner Channel Trend Direction).

Covers: ATR calculation against hand computation (SMA form), Keltner band
construction, uptrend signal flip (high > upper band → T = +2), downtrend
signal flip (low < lower band → T = −2), and signal persistence (no breakout
→ T retains prior value). See Section 3.5 and Section 10.1 of the
specification.

Key formula reminder:
  ATR = rolling SMA of True Range (period = 42, default)
  Upper Band = EMA(Close, upper_ema_span) + k * ATR     [k = 1.0]
  Lower Band = EMA(Close, lower_ema_span) − k * ATR
"""

import numpy as np
import pandas as pd
import pytest

from src.factors.trend import (
    compute_atr,
    compute_atr_bands,
    compute_trend_signal,
    compute_trend_all_assets,
    _SIGNAL_UP,
    _SIGNAL_DOWN,
    _SIGNAL_INIT,
)
from config.default_config import ModelConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ohlc(high, low, close, start="2020-01-02"):
    idx = pd.bdate_range(start, periods=len(close))
    return (
        pd.Series(high, index=idx, dtype=float),
        pd.Series(low, index=idx, dtype=float),
        pd.Series(close, index=idx, dtype=float),
    )


def _constant_ohlc(n, h=110.0, l=90.0, c=100.0, start="2020-01-02"):
    return _make_ohlc([h] * n, [l] * n, [c] * n, start)


# ---------------------------------------------------------------------------
# ATR: hand computation
# ---------------------------------------------------------------------------

class TestComputeAtr:
    def test_constant_ohlc_atr_equals_range(self):
        """
        With constant H=110, L=90, C=100:
        - TR[0] = H[0]−L[0] = 20 (HL component is always defined; pandas
          max(axis=1, skipna=True) returns 20 even when the gap terms are NaN).
        - TR[1..] = max(20, 10, 10) = 20 for every subsequent bar.
        - ATR is the rolling SMA of TR over `period` bars.
        - First valid ATR is at bar `period−1` (all `period` TRs are available).
        """
        period = 5
        n = period + 5
        high, low, close = _constant_ohlc(n)
        atr = compute_atr(high, low, close, period)

        # Bars 0..period-2 have fewer than `period` TRs in the window → NaN.
        assert atr.iloc[:period - 1].isna().all()
        # From bar period-1 onward every rolling window contains `period` valid
        # TRs, all equal to 20 → ATR = 20.
        valid = atr.iloc[period - 1:]
        assert (abs(valid - 20.0) < 1e-10).all(), f"Expected 20.0, got {valid.values}"

    def test_atr_hand_calculation_three_bars(self):
        """
        Manual SMA ATR with period=2.

        Prices: C=[100, 105, 95, 110]
                H=[105, 108, 100, 115]
                L=[ 98, 102,  92, 108]

        TR[0] = max(H[0]−L[0], |H[0]−NaN|, |L[0]−NaN|) = max(7, NaN, NaN) = 7
                (pandas max(skipna=True) returns the HL component)
        TR[1] = max(108−102, |108−100|, |102−100|) = max(6, 8, 2) = 8
        TR[2] = max(100− 92, |100−105|, | 92−105|) = max(8, 5,13) = 13
        TR[3] = max(115−108, |115− 95|, |108− 95|) = max(7,20,13) = 20

        SMA ATR (period=2) — rolling window rolls over exactly 2 consecutive TRs:
          ATR[0] = NaN  (window [TR[0]], only 1 value < min_periods=2)
          ATR[1] = mean(TR[0]=7, TR[1]=8) = 7.5
          ATR[2] = mean(TR[1]=8, TR[2]=13) = 10.5
          ATR[3] = mean(TR[2]=13, TR[3]=20) = 16.5
        """
        period = 2
        close = [100.0, 105.0, 95.0, 110.0]
        high  = [105.0, 108.0, 100.0, 115.0]
        low   = [ 98.0, 102.0,  92.0, 108.0]
        h, l, c = _make_ohlc(high, low, close)
        atr = compute_atr(h, l, c, period)

        assert pd.isna(atr.iloc[0]),           f"ATR[0] should be NaN, got {atr.iloc[0]}"
        assert abs(atr.iloc[1] - 7.5)  < 1e-10, f"ATR[1]: {atr.iloc[1]}"
        assert abs(atr.iloc[2] - 10.5) < 1e-10, f"ATR[2]: {atr.iloc[2]}"
        assert abs(atr.iloc[3] - 16.5) < 1e-10, f"ATR[3]: {atr.iloc[3]}"

    def test_atr_series_length_and_index(self):
        n = 20
        period = 5
        high, low, close = _constant_ohlc(n)
        atr = compute_atr(high, low, close, period)
        assert len(atr) == n
        assert atr.index.equals(close.index)

    def test_insufficient_data_all_nan(self):
        """With period−1 bars, every rolling window has fewer than `period`
        values, so min_periods=period is never satisfied → all NaN.

        Note: TR[0] = H[0]−L[0] is valid (no prior close needed), so `period`
        bars IS enough for the last ATR (all `period` TRs available at bar
        period−1).  To keep ATR all-NaN we use one fewer bar.
        """
        period = 10
        high, low, close = _constant_ohlc(period - 1)   # period−1 bars → all NaN
        atr = compute_atr(high, low, close, period)
        assert atr.isna().all()


# ---------------------------------------------------------------------------
# Keltner bands
# ---------------------------------------------------------------------------

class TestComputeAtrBands:
    def test_bands_keltner_channel_values(self):
        """
        With constant H=110, L=90, C=100, after warm-up:
          TR = max(110−90, |110−100|, |90−100|) = max(20, 10, 10) = 20 → ATR = 20
          EMA(Close, upper_ema_span) → 100  →  UB = 100 + 20 = 120
          EMA(Close, lower_ema_span) → 100  →  LB = 100 − 20 =  80
        """
        atr_period = 3
        upper_ema_span = 4
        lower_ema_span = 5
        n = upper_ema_span + atr_period + 5
        high, low, close = _constant_ohlc(n, h=110, l=90, c=100)
        ub, lb = compute_atr_bands(high, low, close, atr_period, upper_ema_span, lower_ema_span)

        # Binding warm-up: max(atr_period, upper_ema_span−1, lower_ema_span−1).
        first_valid = max(atr_period, upper_ema_span - 1, lower_ema_span - 1)
        valid_ub = ub.iloc[first_valid:].dropna()
        valid_lb = lb.iloc[first_valid:].dropna()

        assert (abs(valid_ub - 120.0) < 1e-10).all(), f"Upper band: {valid_ub.values}"
        assert (abs(valid_lb - 80.0) < 1e-10).all(),  f"Lower band: {valid_lb.values}"

    def test_upper_band_above_lower_band(self):
        """UB > LB at every valid bar (a fundamental sanity check)."""
        atr_period = 3
        upper_ema_span = 4
        lower_ema_span = 5
        n = 30
        high, low, close = _constant_ohlc(n)
        ub, lb = compute_atr_bands(high, low, close, atr_period, upper_ema_span, lower_ema_span)
        valid = ~(ub.isna() | lb.isna())
        assert (ub[valid] > lb[valid]).all()

    def test_bands_have_same_index_as_close(self):
        n = 30
        high, low, close = _constant_ohlc(n)
        ub, lb = compute_atr_bands(high, low, close, 3, 5, 7)
        assert ub.index.equals(close.index)
        assert lb.index.equals(close.index)


# ---------------------------------------------------------------------------
# Signal: uptrend flip
# ---------------------------------------------------------------------------

class TestUptrendFlip:
    def test_high_above_upper_band_sets_signal_up(self):
        """
        Build a price series where the last bar's high clearly breaches the
        upper Keltner band.  With constant C=100, H=110, L=90:
          ATR ≈ 20,  EMA(Close) → 100  →  UB ≈ 120.
        A spike to H=200 on the last bar triggers T = +2.
        """
        atr_period = 3
        upper_ema_span = 4
        lower_ema_span = 5
        n = 25

        close = [100.0] * n
        high  = [110.0] * n
        low   = [ 90.0] * n

        # Spike the last bar's high well above upper_band (≈120).
        high[-1] = 200.0

        h, l, c = _make_ohlc(high, low, close)
        signal = compute_trend_signal(h, l, c, atr_period, upper_ema_span, lower_ema_span)

        assert signal.iloc[-1] == _SIGNAL_UP, (
            f"Expected {_SIGNAL_UP} (uptrend), got {signal.iloc[-1]}"
        )

    def test_signal_up_value_is_positive_two(self):
        assert _SIGNAL_UP == 2.0


# ---------------------------------------------------------------------------
# Signal: downtrend flip
# ---------------------------------------------------------------------------

class TestDowntrendFlip:
    def test_low_below_lower_band_sets_signal_down(self):
        """
        Verify T = −2 fires when low < lower band.

        Because _SIGNAL_INIT = −2 the test must first drive the signal to +2
        (via a high spike), wait for ATR to normalise, then drive it back to
        −2 via a low spike.  That way the −2 on the final check is a genuine
        flip, not just the initial value.

        After the high spike at bar 30 (H=300 >> UB≈120), T flips to +2.
        ATR normalises within ~5 bars.  At bar 60 a low spike (L=1 << LB≈80)
        drives T back to −2.
        """
        atr_period     = 5
        upper_ema_span = 10
        lower_ema_span = 10
        n = 80

        close = [100.0] * n
        high  = [110.0] * n
        low   = [ 90.0] * n

        # Bar 30: trigger T = +2 (H=300 >> UB ≈ ATR spike + EMA ≈ 100)
        high[30] = 300.0
        # Bar 60: trigger T = −2 (L=1 << LB ≈ EMA − ATR ≈ 80)
        low[60]  = 1.0

        h, l, c = _make_ohlc(high, low, close)
        signal = compute_trend_signal(h, l, c, atr_period, upper_ema_span, lower_ema_span)

        assert signal.iloc[30] == _SIGNAL_UP,   (
            f"Expected {_SIGNAL_UP} at bar 30, got {signal.iloc[30]}"
        )
        assert signal.iloc[60] == _SIGNAL_DOWN, (
            f"Expected {_SIGNAL_DOWN} at bar 60, got {signal.iloc[60]}"
        )


# ---------------------------------------------------------------------------
# Signal: persistence (no breakout)
# ---------------------------------------------------------------------------

class TestSignalPersistence:
    def test_signal_retains_value_with_no_breakout(self):
        """
        After warm-up, if no bar breaks either band, the signal must stay at
        the initial value throughout.

        With tight noise (±0.5 around 100) and H/L only ±2 from close, the
        ATR ≈ 4, bands are ≈ [96, 104].  Highs (≈102) and lows (≈98) stay
        comfortably inside — no breakout fires.
        """
        atr_period     = 3
        upper_ema_span = 4
        lower_ema_span = 5
        n = 40

        np.random.seed(0)
        noise = np.random.uniform(-0.5, 0.5, n)
        close = [100.0 + x for x in noise]
        high  = [c + 2.0 for c in close]
        low   = [c - 2.0 for c in close]

        h, l, c = _make_ohlc(high, low, close)
        signal = compute_trend_signal(h, l, c, atr_period, upper_ema_span, lower_ema_span)

        valid = signal.dropna()
        assert (valid == valid.iloc[0]).all(), (
            f"Signal changed without a breakout: {valid.values}"
        )

    def test_signal_is_plus_two_on_uptrend_bar(self):
        """
        A one-bar high spike above the upper band sets T = +2 on that bar.

        With the Keltner Channel formula, a high spike raises ATR, which
        LOWERS the lower band (LB = EMA − ATR).  So the signal CAN persist
        at +2 after the trigger bar — unlike the old rolling-max formula
        where the lower band was structurally above the current low.
        This test only verifies the trigger bar itself (bar 20).
        """
        atr_period     = 3
        upper_ema_span = 4
        lower_ema_span = 5
        n = 40

        close = [100.0] * n
        high  = [110.0] * n
        low   = [ 90.0] * n

        # Trigger an uptrend on bar 20 by spiking high well above upper_band (≈120).
        high[20] = 300.0

        h, l, c = _make_ohlc(high, low, close)
        signal = compute_trend_signal(h, l, c, atr_period, upper_ema_span, lower_ema_span)

        assert signal.iloc[20] == _SIGNAL_UP, (
            f"Expected {_SIGNAL_UP} on trigger bar, got {signal.iloc[20]}"
        )

    def test_uptrend_signal_persists_after_spike(self):
        """
        After a high spike drives T to +2, the signal should persist on
        subsequent bars with normal prices because the inflated ATR pushes
        the lower band (LB = EMA − ATR) below the current lows.
        Confirm T = +2 still holds several bars after the spike.
        """
        atr_period     = 5
        upper_ema_span = 10
        lower_ema_span = 10
        n = 60

        close = [100.0] * n
        high  = [110.0] * n
        low   = [ 90.0] * n

        # Spike at bar 20 drives T = +2.
        high[20] = 300.0

        h, l, c = _make_ohlc(high, low, close)
        signal = compute_trend_signal(h, l, c, atr_period, upper_ema_span, lower_ema_span)

        # T = +2 must hold on the trigger bar and at least 3 bars afterward.
        for i in range(20, 24):
            assert signal.iloc[i] == _SIGNAL_UP, (
                f"Expected {_SIGNAL_UP} at bar {i} (post-spike persist), "
                f"got {signal.iloc[i]}"
            )

    def test_nan_before_warm_up(self):
        """
        Signal must be NaN before the first bar where both Keltner bands are
        simultaneously valid.

        Binding warm-up is max(atr_period−1, upper_ema_span−1, lower_ema_span−1):
        - ATR first valid at bar atr_period−1 (TR[0] = H−L is always valid,
          so `period` consecutive TRs are accumulated at bar period−1).
        - EMA(span=s) first valid at bar s−1 (min_periods=s observed).
        """
        atr_period     = 3
        upper_ema_span = 4
        lower_ema_span = 5
        n = 30
        high, low, close = _constant_ohlc(n)
        signal = compute_trend_signal(
            high, low, close, atr_period, upper_ema_span, lower_ema_span
        )

        first_valid = max(atr_period - 1, upper_ema_span - 1, lower_ema_span - 1)
        assert signal.iloc[:first_valid].isna().all()
        assert not signal.iloc[first_valid:].isna().any()


# ---------------------------------------------------------------------------
# compute_trend_all_assets
# ---------------------------------------------------------------------------

class TestComputeTrendAllAssets:
    def test_returns_dataframe_correct_shape(self):
        cfg = ModelConfig(atr_period=3, atr_upper_lookback=4, atr_lower_lookback=5)
        n = 30
        high, low, close = _constant_ohlc(n)
        df = pd.DataFrame({"High": high, "Low": low, "Close": close,
                           "Open": close, "Volume": 1_000_000})
        data = {"X": df, "Y": df.copy()}
        result = compute_trend_all_assets(data, cfg)
        assert set(result.columns) == {"X", "Y"}
        assert len(result) == n

    def test_only_plus_two_and_minus_two_in_valid_region(self):
        """All valid (non-NaN) signal values must be exactly +2 or −2."""
        cfg = ModelConfig(atr_period=3, atr_upper_lookback=4, atr_lower_lookback=5)
        n = 30
        high, low, close = _constant_ohlc(n)
        df = pd.DataFrame({"High": high, "Low": low, "Close": close,
                           "Open": close, "Volume": 1_000_000})
        result = compute_trend_all_assets({"X": df}, cfg)
        valid = result["X"].dropna()
        assert set(valid.unique()).issubset({_SIGNAL_UP, _SIGNAL_DOWN}), (
            f"Unexpected signal values: {valid.unique()}"
        )
