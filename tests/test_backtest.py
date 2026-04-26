"""
Unit and integration tests for src/backtest/engine.py.

Covers: _month_end_dates and _build_exec_date_map helpers (unit), equity curve
matching a manual calculation over a 3-month synthetic period, transaction cost
deduction on full-rebalance vs. same-allocation periods, turnover when allocation
changes, and warm-up period skipping. See Section 10.1 and 10.2.
"""

import pandas as pd
import pytest
from unittest.mock import patch

from config.default_config import ModelConfig
from config.etf_universe import HEDGING_SLEEVE_TICKERS, MAIN_SLEEVE_TICKERS
from src.backtest.engine import (
    BacktestResult,
    _build_exec_date_map,
    _month_end_dates,
    run_backtest,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_data_dict(dates: pd.DatetimeIndex, price: float = 100.0) -> dict:
    """
    Return a minimal data_dict for all 22 sleeve tickers with flat Close prices.

    The engine only reads data_dict[t]["Close"] when building the closes matrix;
    factor computation is mocked so High/Low are not required here.
    """
    all_sleeve = MAIN_SLEEVE_TICKERS + HEDGING_SLEEVE_TICKERS
    return {
        t: pd.DataFrame({"Close": pd.Series(price, index=dates)})
        for t in all_sleeve
    }


def _test_config(**kwargs) -> ModelConfig:
    """ModelConfig centred on the synthetic 2020-01 → 2020-05 window."""
    defaults = dict(backtest_start="2020-01-01", backtest_end="2020-05-31", transaction_cost=0.001)
    defaults.update(kwargs)
    return ModelConfig(**defaults)


# ---------------------------------------------------------------------------
# _month_end_dates
# ---------------------------------------------------------------------------

class TestMonthEndDates:
    def test_returns_last_bday_of_each_month(self):
        # Jan 31 = Fri, Feb 28 = Fri, Mar 31 = Tue in 2020.
        dates = pd.bdate_range("2020-01-01", "2020-03-31")
        result = _month_end_dates(dates)
        assert result == [
            pd.Timestamp("2020-01-31"),
            pd.Timestamp("2020-02-28"),
            pd.Timestamp("2020-03-31"),
        ]

    def test_single_month(self):
        dates = pd.bdate_range("2020-03-01", "2020-03-31")
        result = _month_end_dates(dates)
        assert len(result) == 1
        assert result[0] == pd.Timestamp("2020-03-31")

    def test_partial_month_uses_last_available_bday(self):
        # Only the first 10 business days of April.
        dates = pd.bdate_range("2020-04-01", "2020-04-14")
        result = _month_end_dates(dates)
        assert len(result) == 1
        assert result[0] == pd.Timestamp("2020-04-14")


# ---------------------------------------------------------------------------
# _build_exec_date_map
# ---------------------------------------------------------------------------

class TestBuildExecDateMap:
    def test_maps_signal_to_next_bday(self):
        # Jan 31 2020 is Friday; next bday is Feb 3 (Mon).
        dates = pd.bdate_range("2020-01-28", "2020-02-05").tolist()
        result = _build_exec_date_map([pd.Timestamp("2020-01-31")], dates)
        assert result[pd.Timestamp("2020-01-31")] == pd.Timestamp("2020-02-03")

    def test_last_date_has_no_mapping(self):
        # No trading day follows the final date in the series.
        dates = [pd.Timestamp("2020-01-31")]
        result = _build_exec_date_map([pd.Timestamp("2020-01-31")], dates)
        assert pd.Timestamp("2020-01-31") not in result

    def test_multiple_signals_all_mapped(self):
        dates = pd.bdate_range("2020-01-28", "2020-03-05").tolist()
        signals = [pd.Timestamp("2020-01-31"), pd.Timestamp("2020-02-28")]
        result = _build_exec_date_map(signals, dates)
        assert result[pd.Timestamp("2020-01-31")] == pd.Timestamp("2020-02-03")
        assert result[pd.Timestamp("2020-02-28")] == pd.Timestamp("2020-03-02")


# ---------------------------------------------------------------------------
# Equity curve and transaction cost tests (engine integration with mocks)
#
# Synthetic setup (2020-01-01 → 2020-05-31):
#   Signal dates:  Jan 31, Feb 28, Mar 31, Apr 30, May 29
#   Exec dates:    Feb 3,  Mar 2,  Apr 1,  May 1   (Jun 1 is outside range)
#   Holding periods: [Feb 3→Mar 2], [Mar 2→Apr 1], [Apr 1→May 1]  ← 3 periods
#
# GLD is set to $100 everywhere; we override 3 exec dates to produce exactly
# 2% return per holding period:
#   Feb 3 = 100 (flat), Mar 2 = 102, Apr 1 = 104.04, May 1 = 106.1208
# ---------------------------------------------------------------------------

_GLD_EXEC_OVERRIDES = {
    pd.Timestamp("2020-03-02"): 102.00,
    pd.Timestamp("2020-04-01"): 104.04,
    pd.Timestamp("2020-05-01"): 106.1208,
}


@pytest.fixture
def synthetic_dates():
    return pd.bdate_range("2020-01-01", "2020-05-31")


@pytest.fixture
def synthetic_data(synthetic_dates):
    data = _make_data_dict(synthetic_dates)
    for d, p in _GLD_EXEC_OVERRIDES.items():
        data["GLD"].loc[d, "Close"] = p
    return data


@pytest.fixture
def cfg():
    return _test_config()


def _run_with_fixed_alloc(data, config, alloc):
    """Run backtest with _precompute_factors mocked and a fixed allocation.

    ``_allocation_at_date`` now returns ``(alloc, top_main, top_hedge)`` so the
    mock must wrap the dict in a tuple to match the updated engine contract.
    """
    with patch("src.backtest.engine._precompute_factors") as mp, \
         patch("src.backtest.engine._allocation_at_date") as ma:
        mp.return_value = {}
        ma.return_value = (alloc, [], [])
        return run_backtest(data, config)


class TestBacktestEquityCurve:
    def test_three_periods_produced(self, synthetic_data, cfg):
        result = _run_with_fixed_alloc(synthetic_data, cfg, {"GLD": 1.0})
        assert len(result.monthly_returns) == 3
        assert len(result.equity_curve) == 3

    def test_equity_curve_matches_manual_calculation(self, synthetic_data, cfg):
        # Period 1: 100% initial rebalance, turnover=1.0, cost=1.0×0.001=0.001
        #           GLD return = 102/100 - 1 = 2%
        #           net return = 0.02 - 0.001 = 0.019
        # Period 2 & 3: same allocation, turnover=0, cost=0, net return = 2%
        result = _run_with_fixed_alloc(synthetic_data, cfg, {"GLD": 1.0})

        expected_1 = 1.0 * (1.0 + 0.02 - 0.001)
        expected_2 = expected_1 * 1.02
        expected_3 = expected_2 * 1.02

        assert result.equity_curve.iloc[0] == pytest.approx(expected_1, rel=1e-9)
        assert result.equity_curve.iloc[1] == pytest.approx(expected_2, rel=1e-9)
        assert result.equity_curve.iloc[2] == pytest.approx(expected_3, rel=1e-9)

    def test_monthly_returns_match_equity_curve_ratios(self, synthetic_data, cfg):
        result = _run_with_fixed_alloc(synthetic_data, cfg, {"GLD": 1.0})
        eq = result.equity_curve
        rets = result.monthly_returns
        # equity[i] = equity[i-1] * (1 + return[i]) for i > 0
        assert eq.iloc[1] == pytest.approx(eq.iloc[0] * (1.0 + rets.iloc[1]), rel=1e-9)
        assert eq.iloc[2] == pytest.approx(eq.iloc[1] * (1.0 + rets.iloc[2]), rel=1e-9)

    def test_result_is_backtest_result_instance(self, synthetic_data, cfg):
        result = _run_with_fixed_alloc(synthetic_data, cfg, {"GLD": 1.0})
        assert isinstance(result, BacktestResult)
        assert isinstance(result.metrics, dict)
        assert len(result.metrics) > 0


class TestTransactionCosts:
    def test_first_period_has_full_turnover(self, synthetic_data, cfg):
        # Starting from empty portfolio → 100% GLD: turnover = 1.0.
        result = _run_with_fixed_alloc(synthetic_data, cfg, {"GLD": 1.0})
        assert result.turnover.iloc[0] == pytest.approx(1.0, rel=1e-9)

    def test_unchanged_allocation_has_zero_turnover(self, synthetic_data, cfg):
        result = _run_with_fixed_alloc(synthetic_data, cfg, {"GLD": 1.0})
        assert result.turnover.iloc[1] == pytest.approx(0.0, abs=1e-12)
        assert result.turnover.iloc[2] == pytest.approx(0.0, abs=1e-12)

    def test_cost_only_deducted_on_first_period(self, synthetic_data, cfg):
        # Period 1 net return should be 0.001 below period 2 net return
        # (cost = turnover × tc = 1.0 × 0.001 = 0.001 on period 1 only).
        result = _run_with_fixed_alloc(synthetic_data, cfg, {"GLD": 1.0})
        gap = result.monthly_returns.iloc[1] - result.monthly_returns.iloc[0]
        assert gap == pytest.approx(0.001, rel=1e-6)

    def test_full_allocation_switch_generates_double_turnover(self, synthetic_data, cfg):
        # Period 1: {} → GLD=1.0 → turnover = 1.0
        # Period 2: GLD=1.0 → SHY=1.0 → turnover = 2.0 (sell all GLD, buy all SHY)
        # Period 3: SHY=1.0 → SHY=1.0 → turnover = 0.0
        # _allocation_at_date returns (alloc, top_main, top_hedge) tuples.
        allocs = [
            ({"GLD": 1.0}, ["GLD"], []),
            ({"SHY": 1.0}, ["SHY"], []),
            ({"SHY": 1.0}, ["SHY"], []),
        ]
        it = iter(allocs)

        with patch("src.backtest.engine._precompute_factors") as mp, \
             patch("src.backtest.engine._allocation_at_date") as ma:
            mp.return_value = {}
            ma.side_effect = lambda *a, **k: next(it)
            result = run_backtest(synthetic_data, cfg)

        assert result.turnover.iloc[0] == pytest.approx(1.0, rel=1e-9)
        assert result.turnover.iloc[1] == pytest.approx(2.0, rel=1e-9)
        assert result.turnover.iloc[2] == pytest.approx(0.0, abs=1e-12)

    def test_zero_transaction_cost_config(self, synthetic_data, synthetic_dates):
        # With zero cost, period 1 and period 2 net returns should both equal gross.
        cfg_no_cost = _test_config(transaction_cost=0.0)
        result = _run_with_fixed_alloc(synthetic_data, cfg_no_cost, {"GLD": 1.0})
        # All periods should return exactly 2% (gross = net).
        for r in result.monthly_returns:
            assert r == pytest.approx(0.02, rel=1e-9)


class TestWarmupBehavior:
    def test_none_allocation_skips_period(self, synthetic_data, cfg):
        # First allocation returns None (warm-up) → only 2 periods produced.
        # Non-None entries must be (alloc, top_main, top_hedge) tuples.
        allocs = [None, ({"GLD": 1.0}, ["GLD"], []), ({"GLD": 1.0}, ["GLD"], [])]
        it = iter(allocs)

        with patch("src.backtest.engine._precompute_factors") as mp, \
             patch("src.backtest.engine._allocation_at_date") as ma:
            mp.return_value = {}
            ma.side_effect = lambda *a, **k: next(it)
            result = run_backtest(synthetic_data, cfg)

        assert len(result.monthly_returns) == 2

    def test_all_none_allocations_gives_empty_result(self, synthetic_data, cfg):
        with patch("src.backtest.engine._precompute_factors") as mp, \
             patch("src.backtest.engine._allocation_at_date") as ma:
            mp.return_value = {}
            ma.return_value = None
            result = run_backtest(synthetic_data, cfg)

        assert len(result.monthly_returns) == 0
        assert len(result.equity_curve) == 0
