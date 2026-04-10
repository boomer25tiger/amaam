"""
Unit tests for src/factors/momentum.py.

Covers: known input/output ROC calculation, constant-price edge case (M=0),
and insufficient-data edge case (fewer than lookback days available). See
Section 10.1 of the specification.
"""

import numpy as np
import pandas as pd
import pytest

from src.factors.momentum import compute_absolute_momentum, compute_momentum_all_assets


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_prices(values, start="2020-01-02"):
    """Build a daily price Series from a list of values."""
    idx = pd.bdate_range(start, periods=len(values))
    return pd.Series(values, index=idx, dtype=float)


def _make_ohlcv(close_values, start="2020-01-02"):
    """Build a minimal OHLCV DataFrame where only Close matters for momentum."""
    idx = pd.bdate_range(start, periods=len(close_values))
    close = pd.Series(close_values, index=idx, dtype=float)
    return pd.DataFrame({"Close": close, "Open": close, "High": close, "Low": close,
                         "Volume": 1_000_000})


# ---------------------------------------------------------------------------
# Known input/output
# ---------------------------------------------------------------------------

class TestComputeAbsoluteMomentum:
    def test_known_value_positive(self):
        """ROC at position lookback should equal (price[lookback] / price[0]) - 1."""
        lookback = 5
        # price doubles over the lookback window: 100 → 200
        prices = _make_prices([100.0, 110.0, 120.0, 140.0, 160.0, 200.0])
        mom = compute_absolute_momentum(prices, lookback)
        expected = 200.0 / 100.0 - 1.0  # = 1.0
        assert abs(mom.iloc[lookback] - expected) < 1e-10

    def test_known_value_negative(self):
        """Negative momentum when price falls over the lookback window."""
        lookback = 3
        prices = _make_prices([100.0, 90.0, 80.0, 70.0])
        mom = compute_absolute_momentum(prices, lookback)
        expected = 70.0 / 100.0 - 1.0  # = -0.30
        assert abs(mom.iloc[lookback] - expected) < 1e-10

    def test_nans_before_lookback(self):
        """The first `lookback` positions must be NaN."""
        lookback = 4
        prices = _make_prices(list(range(100, 111)))  # 11 prices
        mom = compute_absolute_momentum(prices, lookback)
        assert mom.iloc[:lookback].isna().all(), "Expected NaN for first lookback rows"
        assert not mom.iloc[lookback:].isna().any(), "Expected valid values from lookback onwards"

    def test_monotone_increasing_prices(self):
        """Hand-verify 84-day momentum with a linearly increasing price series."""
        lookback = 84
        # price[0] = 100, price[84] = 100 + 84*1 = 184
        prices = _make_prices([100.0 + i for i in range(100)])
        mom = compute_absolute_momentum(prices, lookback)
        expected = (100.0 + 84.0) / 100.0 - 1.0  # 184/100 - 1 = 0.84
        assert abs(mom.iloc[lookback] - expected) < 1e-10

    def test_output_series_preserves_index(self):
        """Output index must match input index."""
        prices = _make_prices([100.0 + i for i in range(20)])
        mom = compute_absolute_momentum(prices, lookback=5)
        assert mom.index.equals(prices.index)


# ---------------------------------------------------------------------------
# Edge case: constant prices
# ---------------------------------------------------------------------------

class TestConstantPrices:
    def test_momentum_zero_for_constant_prices(self):
        """When price is constant, M = 0 for all valid positions."""
        lookback = 10
        prices = _make_prices([100.0] * 20)
        mom = compute_absolute_momentum(prices, lookback)
        valid = mom.iloc[lookback:]
        assert (valid == 0.0).all(), f"Expected all zeros, got {valid.values}"


# ---------------------------------------------------------------------------
# Edge case: insufficient data
# ---------------------------------------------------------------------------

class TestInsufficientData:
    def test_all_nan_when_fewer_rows_than_lookback(self):
        """If len(prices) <= lookback, all values must be NaN."""
        lookback = 10
        prices = _make_prices([100.0 + i for i in range(10)])  # exactly lookback rows
        mom = compute_absolute_momentum(prices, lookback)
        assert mom.isna().all(), "Expected all NaN when data length equals lookback"

    def test_single_valid_point_at_lookback(self):
        """Exactly lookback+1 prices → exactly one valid momentum value."""
        lookback = 5
        prices = _make_prices([100.0, 101.0, 102.0, 103.0, 104.0, 110.0])
        mom = compute_absolute_momentum(prices, lookback)
        assert mom.iloc[:lookback].isna().all()
        assert not pd.isna(mom.iloc[lookback])
        assert abs(mom.iloc[lookback] - (110.0 / 100.0 - 1.0)) < 1e-10


# ---------------------------------------------------------------------------
# compute_momentum_all_assets
# ---------------------------------------------------------------------------

class TestComputeMomentumAllAssets:
    def test_returns_dataframe_with_correct_shape(self):
        lookback = 5
        data = {
            "A": _make_ohlcv([100.0 + i for i in range(20)]),
            "B": _make_ohlcv([200.0 - i for i in range(20)]),
        }
        result = compute_momentum_all_assets(data, lookback)
        assert isinstance(result, pd.DataFrame)
        assert set(result.columns) == {"A", "B"}
        assert len(result) == 20

    def test_values_match_per_ticker_computation(self):
        lookback = 3
        close_a = [100.0, 110.0, 115.0, 120.0, 130.0]
        close_b = [200.0, 190.0, 180.0, 170.0, 160.0]
        data = {"A": _make_ohlcv(close_a), "B": _make_ohlcv(close_b)}
        result = compute_momentum_all_assets(data, lookback)

        # Verify A at position 3: 120/100 - 1 = 0.20
        assert abs(result["A"].iloc[3] - 0.20) < 1e-10
        # Verify B at position 3: 170/200 - 1 = -0.15
        assert abs(result["B"].iloc[3] - (-0.15)) < 1e-10
