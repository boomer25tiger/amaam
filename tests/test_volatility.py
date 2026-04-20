"""
Unit tests for src/factors/volatility.py.

Covers: EWMA recursion correctness against hand calculation for 5-10 steps,
SMA smoothing output, annualization factor (sqrt(252)), and the zero-return
edge case (variance near zero). See Section 10.1 of the specification.
"""

import numpy as np
import pandas as pd
import pytest

from src.factors.volatility import (
    compute_ewma_variance,
    compute_volatility_model,
    compute_volatility_all_assets,
)
from config.default_config import ModelConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _returns_series(values, start="2020-01-02"):
    idx = pd.bdate_range(start, periods=len(values))
    return pd.Series([np.nan] + list(values), index=pd.bdate_range(start, periods=len(values) + 1))


def _prices_from_returns(log_returns, p0=100.0, start="2020-01-02"):
    """Build a price Series consistent with a given sequence of log returns."""
    n = len(log_returns) + 1
    idx = pd.bdate_range(start, periods=n)
    prices = [p0]
    for r in log_returns:
        prices.append(prices[-1] * np.exp(r))
    return pd.Series(prices, index=idx, dtype=float)


def _reference_ewma(log_returns_list, lambda_param=0.94, init_window=20):
    """
    Pure-Python reference implementation of compute_ewma_variance.

    Returns a list (same length as log_returns_list + 1 for the leading NaN)
    with NaN before the init_window position, then EWMA variance values.
    The list is indexed to match a price Series: index 0 = NaN return.
    """
    # The full returns array has NaN at position 0 (no prev price),
    # then log_returns_list at positions 1..n.
    r = [float("nan")] + list(log_returns_list)
    n = len(r)
    var = [float("nan")] * n

    if n - 1 < init_window:
        return var

    # Seed: population variance of r[1..init_window]
    init = r[1 : init_window + 1]
    mean = sum(init) / len(init)
    var[init_window] = sum((x - mean) ** 2 for x in init) / len(init)

    for i in range(init_window + 1, n):
        var[i] = lambda_param * var[i - 1] + (1.0 - lambda_param) * r[i - 1] ** 2

    return var


# ---------------------------------------------------------------------------
# EWMA recursion
# ---------------------------------------------------------------------------

class TestEwmaVariance:
    def test_matches_reference_first_10_steps(self):
        """
        Verify compute_ewma_variance matches pure-Python reference for 10 steps
        after initialisation.
        """
        init_window = 5
        lambda_param = 0.94
        # Alternating ±0.01 for the init window, then varying values
        log_rets = [0.01, -0.01, 0.01, -0.01, 0.01,   # init window (positions 1-5)
                    0.02, -0.03, 0.015, -0.005, 0.025]  # recursion steps (positions 6-10)

        ref = _reference_ewma(log_rets, lambda_param, init_window)
        series = _returns_series(log_rets)
        result = compute_ewma_variance(series, lambda_param, init_window)

        for i in range(1, len(ref)):
            ref_val = ref[i]
            got_val = result.iloc[i]
            if ref_val != ref_val:  # NaN
                assert pd.isna(got_val), f"Expected NaN at index {i}"
            else:
                assert abs(got_val - ref_val) < 1e-14, (
                    f"Mismatch at index {i}: expected {ref_val:.8e}, got {got_val:.8e}"
                )

    def test_single_step_hand_calculation(self):
        """
        Manually verify three EWMA steps to confirm the recursion formula.

        Init returns (positions 1-4): [0.10, -0.10, 0.10, -0.10]
        population variance = mean([0.01, 0.01, 0.01, 0.01]) = 0.01
        (all r² = 0.01, mean(r) = 0, so popvar = mean(r²) - mean(r)² = 0.01)

        Step 1 (position 5): σ²_5 = 0.94 × 0.01 + 0.06 × (-0.10)² = 0.0094 + 0.0006 = 0.01
        Step 2 (position 6): σ²_6 = 0.94 × 0.01 + 0.06 × ( 0.05)² = 0.0094 + 0.00015 = 0.00955
        Step 3 (position 7): σ²_7 = 0.94 × 0.00955 + 0.06 × (-0.02)² = 0.008977 + 0.000024 = 0.009001
        """
        init_window = 4
        lam = 0.94
        log_rets = [0.10, -0.10, 0.10, -0.10,   # init: pos 1-4
                    0.05, -0.02]                  # recursion: pos 5-6 → read as r[4] and r[5]

        # Position 4 (0-indexed) in the full array = var[init_window]
        series = _returns_series(log_rets)
        result = compute_ewma_variance(series, lam, init_window)

        # var[init_window] = popvar([0.10, -0.10, 0.10, -0.10]) = 0.01
        assert abs(result.iloc[init_window] - 0.01) < 1e-10, (
            f"Seed variance wrong: {result.iloc[init_window]}"
        )
        # var[init_window+1] = 0.94*0.01 + 0.06*(-0.10)^2 = 0.0094 + 0.0006 = 0.01
        assert abs(result.iloc[init_window + 1] - 0.01) < 1e-10, (
            f"Step 1 wrong: {result.iloc[init_window + 1]}"
        )
        # var[init_window+2] = 0.94*0.01 + 0.06*(0.05)^2 = 0.0094 + 0.00015 = 0.00955
        assert abs(result.iloc[init_window + 2] - 0.00955) < 1e-10, (
            f"Step 2 wrong: {result.iloc[init_window + 2]}"
        )

    def test_nans_before_init_window(self):
        """All positions before init_window must be NaN."""
        init_window = 10
        log_rets = [0.01] * 20
        series = _returns_series(log_rets)
        result = compute_ewma_variance(series, 0.94, init_window)
        assert result.iloc[:init_window].isna().all()
        assert not result.iloc[init_window:].isna().any()

    def test_insufficient_data_all_nan(self):
        """When there are fewer returns than init_window, output is all NaN."""
        init_window = 20
        log_rets = [0.01] * 15  # fewer than init_window
        series = _returns_series(log_rets)
        result = compute_ewma_variance(series, 0.94, init_window)
        assert result.isna().all()


# ---------------------------------------------------------------------------
# SMA smoothing
# ---------------------------------------------------------------------------

class TestVolatilityModelSmoothing:
    def test_smoothing_window_delays_first_valid(self):
        """
        The first valid vol value should appear at index init_window + smoothing_window - 1.
        """
        init_window = 5
        smoothing_window = 3
        # 20 returns so there is plenty of data
        log_rets = [0.01 * ((-1) ** i) for i in range(20)]
        prices = _prices_from_returns(log_rets)
        vol = compute_volatility_model(prices, 0.94, init_window, smoothing_window)

        first_valid_idx = init_window + smoothing_window - 1
        assert vol.iloc[:first_valid_idx].isna().all(), (
            f"Expected NaN before index {first_valid_idx}"
        )
        assert not pd.isna(vol.iloc[first_valid_idx]), (
            f"Expected valid value at index {first_valid_idx}"
        )

    def test_smoothing_is_sma_of_variance(self):
        """
        Verify the SMA is applied to the variance, not the volatility.
        Manually compute three EWMA variance values and their SMA.
        """
        init_window = 4
        smoothing_window = 3
        lam = 0.94
        log_rets = [0.10, -0.10, 0.10, -0.10, 0.05, -0.02, 0.03]
        prices = _prices_from_returns(log_rets)

        # Compute reference EWMA variances
        ref_var = _reference_ewma(log_rets, lam, init_window)
        # The smoothed variance at position init_window+smoothing_window-1 = 4+3-1 = 6
        # is the mean of ref_var[4], ref_var[5], ref_var[6]
        sma_var = np.mean([v for v in ref_var[init_window: init_window + smoothing_window]
                           if v == v])  # filter NaN
        expected_vol = np.sqrt(sma_var * 252)

        vol = compute_volatility_model(prices, lam, init_window, smoothing_window)
        result_vol = vol.iloc[init_window + smoothing_window - 1]

        assert abs(result_vol - expected_vol) < 1e-10, (
            f"Expected vol {expected_vol:.6f}, got {result_vol:.6f}"
        )


# ---------------------------------------------------------------------------
# Annualisation
# ---------------------------------------------------------------------------

class TestAnnualisation:
    def test_annualisation_factor_sqrt_252(self):
        """
        With a constant log return of r every day, the EWMA variance converges
        to r².  The annualised vol should converge to |r| × sqrt(252).
        """
        r = 0.01
        # Long series so EWMA and SMA fully converge
        log_rets = [r] * 200
        prices = _prices_from_returns(log_rets)
        vol = compute_volatility_model(prices, 0.94, 20, 10)

        # Take vol at the last valid position
        last_vol = vol.dropna().iloc[-1]
        expected = r * np.sqrt(252)
        # Allow 1% relative tolerance because EWMA hasn't fully converged
        # (it approaches r² asymptotically)
        assert abs(last_vol - expected) / expected < 0.01, (
            f"Expected ~{expected:.4f}, got {last_vol:.4f}"
        )


# ---------------------------------------------------------------------------
# Edge case: zero returns
# ---------------------------------------------------------------------------

class TestZeroReturns:
    def test_variance_near_zero_for_constant_prices(self):
        """Constant price → zero log returns → EWMA variance should be zero."""
        prices = pd.Series(
            [100.0] * 50,
            index=pd.bdate_range("2020-01-02", periods=50),
        )
        vol = compute_volatility_model(prices, 0.94, 20, 10)
        valid = vol.dropna()
        assert (valid.abs() < 1e-12).all(), f"Expected ~0, got max {valid.abs().max()}"


# ---------------------------------------------------------------------------
# compute_volatility_all_assets
# ---------------------------------------------------------------------------

class TestComputeVolatilityAllAssets:
    def test_returns_dataframe_correct_shape(self):
        # Use a short yang_zhang_window so the test data (40 rows) is long
        # enough to produce at least some non-NaN values.  The legacy EWMA
        # parameters (volatility_init_window, volatility_smoothing) are not
        # used by compute_volatility_all_assets which routes to Yang-Zhang.
        cfg_small = ModelConfig(yang_zhang_window=10)
        log_rets = [0.01 * ((-1) ** i) for i in range(40)]
        prices = _prices_from_returns(log_rets)
        df = pd.DataFrame({"Open": prices, "High": prices, "Low": prices,
                           "Close": prices, "Volume": 1_000_000})
        data = {"A": df, "B": df.copy()}
        result = compute_volatility_all_assets(data, cfg_small)
        assert set(result.columns) == {"A", "B"}
        assert len(result) == len(prices)
