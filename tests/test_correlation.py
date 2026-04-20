"""
Unit tests for src/factors/correlation.py.

Covers: correlation matrix correctness for a 3-asset × 10-day synthetic
dataset, average pairwise correlation calculation, perfectly correlated
assets (C=1.0), and perfectly uncorrelated assets (C≈0.0).
See Section 10.1 of the specification.
"""

import numpy as np
import pandas as pd
import pytest

from src.factors.correlation import (
    compute_average_relative_correlation,
    compute_correlation_all_assets,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_returns(matrix, start="2020-01-02"):
    """
    Build a returns_dict from a 2-D array (rows=days, cols=assets).
    Also wrap as a data_dict with OHLCV DataFrames (Close only matters).
    """
    tickers = [f"A{i}" for i in range(matrix.shape[1])]
    idx = pd.bdate_range(start, periods=matrix.shape[0])

    returns_dict = {t: pd.Series(matrix[:, j], index=idx) for j, t in enumerate(tickers)}

    # Build price series by cumulative product (needed for data_dict)
    data_dict = {}
    for j, t in enumerate(tickers):
        price = pd.Series(np.cumprod(1.0 + matrix[:, j]) * 100.0, index=idx)
        data_dict[t] = pd.DataFrame({
            "Open": price, "High": price, "Low": price,
            "Close": price, "Volume": 1_000_000,
        })

    return returns_dict, data_dict, tickers


# ---------------------------------------------------------------------------
# Manual correlation check: 3 assets × 10 days
# ---------------------------------------------------------------------------

class TestManualCorrelation:
    def test_average_relative_correlation_matches_numpy(self):
        """
        Build a 3-asset × 10-day return matrix, compute average pairwise
        correlations manually with numpy, and verify compute_average_relative_correlation
        matches to numerical precision.
        """
        np.random.seed(42)
        matrix = np.random.randn(15, 3) * 0.01   # 15 days, 3 assets
        lookback = 10
        returns_dict, _, tickers = _make_returns(matrix)

        # Target date = last available date
        date = pd.bdate_range("2020-01-02", periods=15)[-1]
        result = compute_average_relative_correlation(returns_dict, tickers, lookback, date)

        # Manual numpy reference: last 10 rows of the matrix
        window = matrix[-lookback:]
        corr = np.corrcoef(window.T)  # shape (3, 3)
        n = 3
        for j, t in enumerate(tickers):
            expected = (corr[j].sum() - 1.0) / (n - 1)
            assert abs(result[t] - expected) < 1e-8, (
                f"{t}: expected {expected:.6f}, got {result[t]:.6f}"
            )

    def test_average_pairwise_uses_correct_formula(self):
        """C_i = (sum of row i in corr matrix − 1) / (N − 1)."""
        # Construct returns where we know the correlation exactly
        n_days = 30
        # A1 and A2 have a known correlation structure
        t = np.linspace(0, 2 * np.pi, n_days)
        r1 = np.sin(t) * 0.01
        r2 = np.cos(t) * 0.01
        r3 = np.random.RandomState(7).randn(n_days) * 0.01
        matrix = np.column_stack([r1, r2, r3])
        lookback = 20
        returns_dict, _, tickers = _make_returns(matrix)
        date = pd.bdate_range("2020-01-02", periods=n_days)[-1]

        result = compute_average_relative_correlation(returns_dict, tickers, lookback, date)
        window = matrix[-lookback:]
        corr = np.corrcoef(window.T)
        for j, t_name in enumerate(tickers):
            expected = (corr[j].sum() - 1.0) / 2.0
            assert abs(result[t_name] - expected) < 1e-8


# ---------------------------------------------------------------------------
# Perfectly correlated assets (C = 1.0)
# ---------------------------------------------------------------------------

class TestPerfectlyCorrelated:
    def test_average_correlation_is_one(self):
        """When all assets move identically, every pairwise corr = 1, so C_i = 1."""
        n_days = 30
        lookback = 20
        identical_returns = np.random.RandomState(1).randn(n_days) * 0.01
        # Three identical series
        matrix = np.column_stack([identical_returns] * 3)
        returns_dict, _, tickers = _make_returns(matrix)
        date = pd.bdate_range("2020-01-02", periods=n_days)[-1]

        result = compute_average_relative_correlation(returns_dict, tickers, lookback, date)
        for t in tickers:
            assert abs(result[t] - 1.0) < 1e-8, f"{t}: expected 1.0, got {result[t]}"


# ---------------------------------------------------------------------------
# Perfectly uncorrelated assets (C ≈ 0)
# ---------------------------------------------------------------------------

class TestPerfectlyUncorrelated:
    def test_average_correlation_near_zero(self):
        """
        Orthogonal sine-wave returns produce zero pairwise Pearson correlation.
        C_i should be 0.0 (up to floating-point precision).
        """
        # Use sine waves at different frequencies that are orthogonal over the window
        n_days = 40
        lookback = 40
        t = np.linspace(0, 2 * np.pi, n_days, endpoint=False)
        r1 = np.sin(1 * t)
        r2 = np.sin(2 * t)
        r3 = np.sin(3 * t)
        matrix = np.column_stack([r1, r2, r3])
        returns_dict, _, tickers = _make_returns(matrix)
        date = pd.bdate_range("2020-01-02", periods=n_days)[-1]

        result = compute_average_relative_correlation(returns_dict, tickers, lookback, date)
        # Sine waves with integer frequency ratios are orthogonal over a full period
        for t_name in tickers:
            assert abs(result[t_name]) < 0.05, (
                f"{t_name}: expected near 0, got {result[t_name]:.4f}"
            )

    def test_compute_correlation_all_assets_shape(self):
        """compute_correlation_all_assets returns correct shape and valid values."""
        np.random.seed(0)
        n_days = 50
        lookback = 20
        matrix = np.random.randn(n_days, 3) * 0.01
        _, data_dict, tickers = _make_returns(matrix)

        result = compute_correlation_all_assets(data_dict, tickers, lookback)
        assert isinstance(result, pd.DataFrame)
        assert set(result.columns) == set(tickers)
        assert len(result) == n_days
        # First lookback-1 rows should be NaN
        # pct_change gives NaN at row 0, so rolling(min_periods=lookback) needs
        # rows 1..lookback to all be valid → first non-NaN result is at row `lookback`.
        assert result.iloc[:lookback].isna().all().all()
        # Rows from lookback onward should be valid
        valid_rows = result.iloc[lookback:]
        assert not valid_rows.isna().any().any()


# ---------------------------------------------------------------------------
# compute_correlation_all_assets vs compute_average_relative_correlation
# ---------------------------------------------------------------------------

class TestConsistencyBetweenFunctions:
    def test_all_assets_matches_point_in_time(self):
        """
        compute_correlation_all_assets at a specific date must match
        compute_average_relative_correlation called for that same date.
        """
        np.random.seed(99)
        n_days = 40
        lookback = 20
        matrix = np.random.randn(n_days, 3) * 0.01
        returns_dict, data_dict, tickers = _make_returns(matrix)

        all_corr = compute_correlation_all_assets(data_dict, tickers, lookback)
        idx = data_dict[tickers[0]].index

        # Compare at the last valid date
        date = idx[-1]
        point_in_time = compute_average_relative_correlation(
            returns_dict, tickers, lookback, date
        )
        for t in tickers:
            expected = point_in_time[t]
            got = all_corr.loc[date, t]
            assert abs(got - expected) < 1e-8, (
                f"{t} at {date}: batch={got:.6f}, point-in-time={expected:.6f}"
            )
