"""
Unit tests for src/backtest/metrics.py.

Covers: Sharpe ratio against known analytical values, max drawdown against a
hand-constructed equity curve with a known peak-to-trough decline, annualization
scaling (sqrt(12) for monthly), and rolling metric output shapes and values.
See Section 10.1 of the specification.
"""

import math

import numpy as np
import pandas as pd
import pytest

from src.backtest.metrics import (
    compute_all_metrics,
    compute_drawdown_series,
    compute_rolling_metrics,
)


def _monthly_index(n: int, start: str = "2015-01") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq="MS")


# ---------------------------------------------------------------------------
# compute_drawdown_series
# ---------------------------------------------------------------------------

class TestComputeDrawdownSeries:
    def test_new_high_gives_zero_drawdown(self):
        # Each period is a new high — no drawdown at any point.
        rets = pd.Series([0.05, 0.05, 0.05], index=_monthly_index(3))
        dd = compute_drawdown_series(rets)
        assert (dd.abs() < 1e-12).all()

    def test_known_drawdown_after_decline(self):
        # +10%, +10%, -20%, +10%
        # Equity: 1.1 → 1.21 → 0.968 → 1.0648
        # Peak at 1.21; trough at 0.968
        # Drawdown at period 3 = (0.968 - 1.21) / 1.21 = -0.2 exactly
        rets = pd.Series([0.1, 0.1, -0.2, 0.1], index=_monthly_index(4))
        dd = compute_drawdown_series(rets)
        assert dd.iloc[0] == pytest.approx(0.0, abs=1e-12)
        assert dd.iloc[1] == pytest.approx(0.0, abs=1e-12)
        assert dd.iloc[2] == pytest.approx(-0.2, rel=1e-9)
        assert dd.iloc[3] > -0.2  # recovering

    def test_all_values_nonpositive(self):
        np.random.seed(7)
        rets = pd.Series(np.random.normal(0.005, 0.04, 60), index=_monthly_index(60))
        dd = compute_drawdown_series(rets)
        assert (dd <= 0.0).all()

    def test_single_period(self):
        rets = pd.Series([0.1], index=_monthly_index(1))
        dd = compute_drawdown_series(rets)
        assert len(dd) == 1
        assert dd.iloc[0] == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# compute_all_metrics
# ---------------------------------------------------------------------------

class TestComputeAllMetrics:
    def test_empty_returns_gives_empty_dict(self):
        rets = pd.Series(dtype=float)
        result = compute_all_metrics(rets)
        assert result == {}

    def test_annualized_return_exactly_one_year(self):
        # 12 months of flat +1%/month → ann_return = (1.01)^12 - 1
        rets = pd.Series([0.01] * 12, index=_monthly_index(12))
        metrics = compute_all_metrics(rets)
        expected = (1.01) ** 12 - 1.0
        assert metrics["Annualized Return"] == pytest.approx(expected, rel=1e-9)

    def test_annualized_return_two_year_series(self):
        # 24 months of flat +2%/month → ann_return = (1.02)^12 - 1
        rets = pd.Series([0.02] * 24, index=_monthly_index(24))
        metrics = compute_all_metrics(rets)
        total = (1.02) ** 24 - 1.0
        expected = (1.0 + total) ** (12.0 / 24) - 1.0
        assert metrics["Annualized Return"] == pytest.approx(expected, rel=1e-9)

    def test_sharpe_ratio_formula_correctness(self):
        # Sharpe = (E[r] - rf_monthly) * sqrt(12) / std(r)
        # Verify the implementation matches this analytic definition exactly.
        np.random.seed(42)
        rets = pd.Series(np.random.normal(0.01, 0.04, 60), index=_monthly_index(60))
        metrics = compute_all_metrics(rets, risk_free_rate=0.02)

        monthly_rf = (1.02) ** (1.0 / 12) - 1.0
        expected = (rets.mean() - monthly_rf) * math.sqrt(12) / rets.std()
        assert metrics["Sharpe Ratio"] == pytest.approx(expected, rel=1e-9)

    def test_sharpe_zero_vol_is_nan(self):
        # Constant returns → zero standard deviation → Sharpe is nan.
        rets = pd.Series([0.01] * 24, index=_monthly_index(24))
        metrics = compute_all_metrics(rets)
        assert math.isnan(metrics["Sharpe Ratio"])

    def test_max_drawdown_known_value(self):
        # Peak-to-trough of -20% as constructed in TestComputeDrawdownSeries.
        rets = pd.Series([0.1, 0.1, -0.2, 0.1], index=_monthly_index(4))
        metrics = compute_all_metrics(rets)
        assert metrics["Max Drawdown"] == pytest.approx(-0.2, rel=1e-9)

    def test_calmar_ratio_definition(self):
        # Calmar = ann_return / |max_drawdown|
        np.random.seed(5)
        rets = pd.Series(np.random.normal(0.008, 0.035, 48), index=_monthly_index(48))
        metrics = compute_all_metrics(rets)
        if metrics["Max Drawdown"] < 0:
            expected = metrics["Annualized Return"] / abs(metrics["Max Drawdown"])
            assert metrics["Calmar Ratio"] == pytest.approx(expected, rel=1e-9)

    def test_best_and_worst_month(self):
        rets = pd.Series([0.05, -0.03, 0.02, -0.07, 0.04], index=_monthly_index(5))
        metrics = compute_all_metrics(rets)
        assert metrics["Best Month"] == pytest.approx(0.05, rel=1e-9)
        assert metrics["Worst Month"] == pytest.approx(-0.07, rel=1e-9)

    def test_pct_positive_months(self):
        # 3 positive out of 5.
        rets = pd.Series([0.01, -0.02, 0.03, 0.01, -0.01], index=_monthly_index(5))
        metrics = compute_all_metrics(rets)
        assert metrics["% Positive Months"] == pytest.approx(3 / 5, rel=1e-9)

    def test_total_return_compound_product(self):
        rets = pd.Series([0.1, -0.05, 0.2], index=_monthly_index(3))
        metrics = compute_all_metrics(rets)
        expected = 1.1 * 0.95 * 1.2 - 1.0
        assert metrics["Total Return"] == pytest.approx(expected, rel=1e-9)

    def test_turnover_metrics_included_when_provided(self):
        rets = pd.Series([0.01] * 12, index=_monthly_index(12))
        to = pd.Series([0.8] * 12, index=_monthly_index(12))
        metrics = compute_all_metrics(rets, turnover=to)
        assert "Avg Monthly Turnover" in metrics
        assert "Avg Annual Turnover" in metrics
        assert metrics["Avg Monthly Turnover"] == pytest.approx(0.8, rel=1e-9)

    def test_turnover_metrics_absent_without_turnover(self):
        rets = pd.Series([0.01] * 12, index=_monthly_index(12))
        metrics = compute_all_metrics(rets)
        assert "Avg Monthly Turnover" not in metrics
        assert "Avg Annual Turnover" not in metrics

    def test_annualized_vol_formula(self):
        # ann_vol = monthly_std * sqrt(12)
        np.random.seed(1)
        rets = pd.Series(np.random.normal(0.01, 0.04, 36), index=_monthly_index(36))
        metrics = compute_all_metrics(rets)
        expected_vol = rets.std() * math.sqrt(12)
        assert metrics["Annualized Volatility"] == pytest.approx(expected_vol, rel=1e-9)


# ---------------------------------------------------------------------------
# compute_rolling_metrics
# ---------------------------------------------------------------------------

class TestComputeRollingMetrics:
    def test_output_shape_and_columns(self):
        n = 24
        np.random.seed(2)
        rets = pd.Series(np.random.normal(0.01, 0.03, n), index=_monthly_index(n))
        result = compute_rolling_metrics(rets, window=12)
        assert result.shape == (n, 3)
        assert set(result.columns) == {"rolling_sharpe", "rolling_vol", "rolling_max_dd"}

    def test_first_window_minus_one_rows_are_nan(self):
        n, window = 24, 12
        np.random.seed(3)
        rets = pd.Series(np.random.normal(0.01, 0.03, n), index=_monthly_index(n))
        result = compute_rolling_metrics(rets, window=window)
        # Rows 0 through window-2 should all be NaN (insufficient history).
        assert result.iloc[: window - 1].isna().all(axis=None)

    def test_rows_from_window_onward_are_not_all_nan(self):
        n, window = 24, 12
        np.random.seed(4)
        rets = pd.Series(np.random.normal(0.01, 0.03, n), index=_monthly_index(n))
        result = compute_rolling_metrics(rets, window=window)
        assert not result.iloc[window - 1 :].isna().all(axis=None)

    def test_rolling_vol_matches_pandas_rolling_std(self):
        np.random.seed(0)
        rets = pd.Series(np.random.normal(0.01, 0.03, 36), index=_monthly_index(36))
        result = compute_rolling_metrics(rets, window=12)
        expected = rets.rolling(12, min_periods=12).std() * math.sqrt(12)
        pd.testing.assert_series_equal(
            result["rolling_vol"].rename(None),
            expected.rename(None),
            check_names=False,
            rtol=1e-9,
        )

    def test_rolling_max_dd_is_nonpositive(self):
        np.random.seed(9)
        rets = pd.Series(np.random.normal(0.005, 0.04, 36), index=_monthly_index(36))
        result = compute_rolling_metrics(rets, window=12)
        valid = result["rolling_max_dd"].dropna()
        assert (valid <= 0.0).all()
