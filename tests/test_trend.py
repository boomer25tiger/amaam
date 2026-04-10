"""
Unit tests for src/factors/trend.py.

Covers: ATR calculation against hand computation, band construction, uptrend
signal flip (high > upper band → T = +2), downtrend signal flip
(low < lower band → T = -2), and signal persistence (no breakout → T retains
prior value). See Section 10.1 of the specification.
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
        - TR_1 = max(20, |110-100|, |90-100|) = max(20,10,10) = 20 for every bar.
        - ATR seed (position `period`) = mean of first `period` TRs = 20.
        - Every subsequent ATR = 20 (constant TR, so Wilder update preserves value).
        """
        period = 5
        n = period + 5
        high, low, close = _constant_ohlc(n)
        atr = compute_atr(high, low, close, period)

        # Positions 0..period-1 should be NaN
        assert atr.iloc[:period].isna().all()
        # All valid ATR values should equal 20
        valid = atr.iloc[period:]
        assert (abs(valid - 20.0) < 1e-10).all(), f"Expected 20.0, got {valid.values}"

    def test_atr_hand_calculation_three_bars(self):
        """
        Manual 3-step Wilder ATR with period=2.

        Prices: C=[100, 105, 95, 110]
                H=[105, 108, 100, 115]
                L=[ 98, 102,  92, 108]

        TR[1] = max(108-102, |108-100|, |102-100|) = max(6, 8, 2) = 8
        TR[2] = max(100- 92, |100-105|, | 92-105|) = max(8, 5,13) = 13
        TR[3] = max(115-108, |115- 95|, |108- 95|) = max(7,20,13) = 20

        Seed (period=2): ATR[2] = mean(TR[1], TR[2]) = (8+13)/2 = 10.5
        ATR[3] = (ATR[2] * 1 + TR[3]) / 2 = (10.5 + 20) / 2 = 15.25
        """
        period = 2
        close = [100.0, 105.0, 95.0, 110.0]
        high  = [105.0, 108.0, 100.0, 115.0]
        low   = [ 98.0, 102.0,  92.0, 108.0]
        h, l, c = _make_ohlc(high, low, close)
        atr = compute_atr(h, l, c, period)

        assert pd.isna(atr.iloc[0])
        assert pd.isna(atr.iloc[1])
        assert abs(atr.iloc[2] - 10.5) < 1e-10, f"ATR seed: {atr.iloc[2]}"
        assert abs(atr.iloc[3] - 15.25) < 1e-10, f"ATR step 1: {atr.iloc[3]}"

    def test_atr_series_length_and_index(self):
        n = 20
        period = 5
        high, low, close = _constant_ohlc(n)
        atr = compute_atr(high, low, close, period)
        assert len(atr) == n
        assert atr.index.equals(close.index)

    def test_insufficient_data_all_nan(self):
        """Fewer bars than period → all NaN."""
        period = 10
        high, low, close = _constant_ohlc(period)  # exactly period bars, no room for ATR
        atr = compute_atr(high, low, close, period)
        assert atr.isna().all()


# ---------------------------------------------------------------------------
# ATR bands
# ---------------------------------------------------------------------------

class TestComputeAtrBands:
    def test_bands_equal_atr_plus_rolling_max(self):
        """
        With constant H=110, L=90, C=100:
          TR = max(110-90, |110-100|, |90-100|) = max(20, 10, 10) = 20 → ATR = 20
          rolling max close = 100  → upper_band = 20 + 100 = 120
          rolling max low   =  90  → lower_band = 20 +  90 = 110
        """
        atr_period = 3
        upper_lookback = 4
        lower_lookback = 5
        n = upper_lookback + atr_period + 5
        high, low, close = _constant_ohlc(n, h=110, l=90, c=100)
        ub, lb = compute_atr_bands(high, low, close, atr_period, upper_lookback, lower_lookback)

        # After the warm-up period, values should be constant
        first_valid = max(atr_period, upper_lookback - 1, lower_lookback - 1)
        valid_ub = ub.iloc[first_valid:].dropna()
        valid_lb = lb.iloc[first_valid:].dropna()

        assert (abs(valid_ub - 120.0) < 1e-10).all(), f"Upper band: {valid_ub.values}"
        assert (abs(valid_lb - 110.0) < 1e-10).all(), f"Lower band: {valid_lb.values}"

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
        Build a price series where the last bar's high clearly breaches the upper band.
        Confirm T = +2 on that bar.
        """
        atr_period = 3
        upper_lookback = 4
        lower_lookback = 5
        n = 25

        # Quiet period: constant prices → ATR=20, upper_band=130
        close = [100.0] * n
        high  = [110.0] * n
        low   = [ 90.0] * n

        # Spike the last bar's high well above upper_band (130)
        high[-1] = 200.0

        h, l, c = _make_ohlc(high, low, close)
        signal = compute_trend_signal(h, l, c, atr_period, upper_lookback, lower_lookback)

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
        Build a price series where the last bar's low clearly falls below the
        lower band (ATR + highest_low).  Confirm T = -2 on that bar.
        """
        atr_period = 3
        upper_lookback = 4
        lower_lookback = 4
        n = 25

        # Quiet period: ATR=10, highest_low=90, lower_band=100
        close = [100.0] * n
        high  = [105.0] * n
        low   = [ 90.0] * n

        # Spike down: low falls far below lower_band on last bar
        low[-1] = 1.0

        h, l, c = _make_ohlc(high, low, close)
        signal = compute_trend_signal(h, l, c, atr_period, upper_lookback, lower_lookback)

        assert signal.iloc[-1] == _SIGNAL_DOWN, (
            f"Expected {_SIGNAL_DOWN} (downtrend), got {signal.iloc[-1]}"
        )


# ---------------------------------------------------------------------------
# Signal: persistence (no breakout)
# ---------------------------------------------------------------------------

class TestSignalPersistence:
    def test_signal_retains_value_with_no_breakout(self):
        """
        After the warm-up period, if no bar breaks either band, the signal
        must stay at the initial value throughout.
        """
        atr_period = 3
        upper_lookback = 4
        lower_lookback = 5
        n = 40

        # Prices stay in a tight range — no breakout
        np.random.seed(0)
        noise = np.random.uniform(-0.5, 0.5, n)
        close = [100.0 + x for x in noise]
        high  = [c + 2.0 for c in close]
        low   = [c - 2.0 for c in close]

        h, l, c = _make_ohlc(high, low, close)
        signal = compute_trend_signal(h, l, c, atr_period, upper_lookback, lower_lookback)

        valid = signal.dropna()
        # All values should be the same (initial value, no breakout triggered)
        assert (valid == valid.iloc[0]).all(), (
            f"Signal changed without a breakout: {valid.values}"
        )

    def test_signal_is_plus_two_on_uptrend_bar(self):
        """
        A one-bar high spike above the upper band sets T = +2 on that bar.
        Signal does NOT necessarily persist because the spike also inflates ATR,
        which raises the lower band above subsequent lows, re-triggering T = -2
        on the next bar.  This test only asserts the trigger bar itself.
        """
        atr_period = 3
        upper_lookback = 4
        lower_lookback = 5
        n = 40

        close = [100.0] * n
        high  = [110.0] * n
        low   = [ 90.0] * n

        # Trigger an uptrend on bar 20 by spiking high well above upper_band (~120)
        high[20] = 300.0

        h, l, c = _make_ohlc(high, low, close)
        signal = compute_trend_signal(h, l, c, atr_period, upper_lookback, lower_lookback)

        assert signal.iloc[20] == _SIGNAL_UP, (
            f"Expected {_SIGNAL_UP} on trigger bar, got {signal.iloc[20]}"
        )

    def test_signal_reverts_to_down_after_uptrend_bar(self):
        """
        Documents the structural property of the ATR band formula:
        Lower Band = ATR + Highest_Low(lookback).

        Because ATR > 0 always and Highest_Low(lookback) ≥ current_low (the
        rolling max of past lows is always ≥ the current low), the lower band
        is structurally above the current low on every bar where ATR is valid.
        Therefore T=+2 can only occur on the specific bar where High > upper_band;
        on the very next bar, Low < lower_band fires and T reverts to -2.

        This is a spec-faithful test: it confirms the formula behaves as written.
        Resolution of the sign convention is deferred to Phase 3 (trank.py).
        """
        atr_period = 3
        upper_lookback = 4
        lower_lookback = 5

        n = 40
        close = [100.0] * n
        high  = [110.0] * n
        low   = [ 90.0] * n

        # Trigger an uptrend on bar 20 by spiking high above the upper band
        high[20] = 300.0

        h, l, c = _make_ohlc(high, low, close)
        signal = compute_trend_signal(h, l, c, atr_period, upper_lookback, lower_lookback)

        # The trigger bar itself must be +2
        assert signal.iloc[20] == _SIGNAL_UP, (
            f"Expected {_SIGNAL_UP} on trigger bar, got {signal.iloc[20]}"
        )
        # The bar immediately after must be -2 (lower_band > low due to ATR > 0)
        assert signal.iloc[21] == _SIGNAL_DOWN, (
            f"Expected {_SIGNAL_DOWN} on bar after trigger, got {signal.iloc[21]}"
        )

    def test_nan_before_warm_up(self):
        """Signal must be NaN before the first valid band position."""
        atr_period = 3
        upper_lookback = 4
        lower_lookback = 5
        n = 30
        high, low, close = _constant_ohlc(n)
        signal = compute_trend_signal(
            high, low, close, atr_period, upper_lookback, lower_lookback
        )

        first_valid = max(atr_period, upper_lookback - 1, lower_lookback - 1)
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
        """All valid (non-NaN) signal values must be exactly +2 or -2."""
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
