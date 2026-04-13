"""
Unit tests for src/portfolio/weighting.py.

Covers: equal-weight output (1/N for each asset), inverse-volatility weights
proportional to 1/V, and normalization invariant (weights sum to 1.0 in both
schemes). See Section 10.1 of the specification.
"""

import pytest
import pandas as pd

from src.portfolio.weighting import (
    apply_weighting,
    equal_weight,
    inverse_volatility_weight,
    SCHEME_EQUAL,
    SCHEME_INVERSE_VOL,
)


# ---------------------------------------------------------------------------
# TestEqualWeight
# ---------------------------------------------------------------------------

class TestEqualWeight:

    def test_each_weight_is_one_over_n(self):
        """Every ticker receives exactly 1/N."""
        w = equal_weight(["A", "B", "C"])
        assert w["A"] == pytest.approx(1 / 3)
        assert w["B"] == pytest.approx(1 / 3)
        assert w["C"] == pytest.approx(1 / 3)

    def test_six_tickers_each_one_sixth(self):
        """Canonical main-sleeve case: 6 active assets each get 1/6."""
        tickers = ["A", "B", "C", "D", "E", "F"]
        w = equal_weight(tickers)
        for t in tickers:
            assert w[t] == pytest.approx(1 / 6)

    def test_single_ticker_gets_full_weight(self):
        """One ticker → weight = 1.0."""
        w = equal_weight(["X"])
        assert w["X"] == pytest.approx(1.0)

    def test_weights_sum_to_one(self):
        """Sum of equal weights must be exactly 1.0 for any N."""
        for n in [2, 3, 6, 10, 16]:
            tickers = [str(i) for i in range(n)]
            assert sum(equal_weight(tickers).values()) == pytest.approx(1.0)

    def test_empty_tickers_raises(self):
        with pytest.raises(ValueError):
            equal_weight([])


# ---------------------------------------------------------------------------
# TestInverseVolatilityWeight
# ---------------------------------------------------------------------------

class TestInverseVolatilityWeight:

    def test_lower_volatility_gets_higher_weight(self):
        """Asset A (V=0.10) must outweigh asset B (V=0.20)."""
        vols = pd.Series({"A": 0.10, "B": 0.20})
        w = inverse_volatility_weight(["A", "B"], vols)
        assert w["A"] > w["B"]

    def test_weights_proportional_to_inverse_volatility(self):
        """
        w_A / w_B must equal V_B / V_A (inverse proportionality).

        With V_A=0.10, V_B=0.20:
          inv_A = 10.0, inv_B = 5.0, total = 15.0
          w_A = 10/15 = 2/3,  w_B = 5/15 = 1/3
        """
        vols = pd.Series({"A": 0.10, "B": 0.20})
        w = inverse_volatility_weight(["A", "B"], vols)
        assert w["A"] == pytest.approx(2 / 3)
        assert w["B"] == pytest.approx(1 / 3)

    def test_known_three_asset_calculation(self):
        """
        Three assets: V = [0.10, 0.20, 0.25].
          inv = [10, 5, 4], total = 19
          w = [10/19, 5/19, 4/19]
        """
        vols = pd.Series({"A": 0.10, "B": 0.20, "C": 0.25})
        w = inverse_volatility_weight(["A", "B", "C"], vols)
        assert w["A"] == pytest.approx(10 / 19)
        assert w["B"] == pytest.approx(5 / 19)
        assert w["C"] == pytest.approx(4 / 19)

    def test_weights_sum_to_one(self):
        vols = pd.Series({"A": 0.12, "B": 0.18, "C": 0.09, "D": 0.24})
        w = inverse_volatility_weight(list(vols.index), vols)
        assert sum(w.values()) == pytest.approx(1.0)

    def test_equal_volatilities_produce_equal_weights(self):
        """When all assets have the same volatility, weights must be equal."""
        vols = pd.Series({"A": 0.15, "B": 0.15, "C": 0.15})
        w = inverse_volatility_weight(["A", "B", "C"], vols)
        assert w["A"] == pytest.approx(1 / 3)
        assert w["B"] == pytest.approx(1 / 3)
        assert w["C"] == pytest.approx(1 / 3)

    def test_zero_volatility_raises(self):
        vols = pd.Series({"A": 0.10, "B": 0.0})
        with pytest.raises(ValueError):
            inverse_volatility_weight(["A", "B"], vols)

    def test_empty_tickers_raises(self):
        with pytest.raises(ValueError):
            inverse_volatility_weight([], pd.Series(dtype=float))


# ---------------------------------------------------------------------------
# TestApplyWeighting
# ---------------------------------------------------------------------------

class TestApplyWeighting:

    def test_equal_scheme_dispatches_correctly(self):
        w = apply_weighting(["A", "B", "C"], SCHEME_EQUAL)
        assert w == pytest.approx({"A": 1/3, "B": 1/3, "C": 1/3})

    def test_inverse_vol_scheme_dispatches_correctly(self):
        vols = pd.Series({"A": 0.10, "B": 0.20})
        w = apply_weighting(["A", "B"], SCHEME_INVERSE_VOL, volatility_values=vols)
        assert w["A"] == pytest.approx(2 / 3)
        assert w["B"] == pytest.approx(1 / 3)

    def test_inverse_vol_without_volatility_raises(self):
        with pytest.raises(ValueError, match="volatility_values"):
            apply_weighting(["A", "B"], SCHEME_INVERSE_VOL, volatility_values=None)

    def test_unknown_scheme_raises(self):
        with pytest.raises(ValueError, match="Unknown weighting scheme"):
            apply_weighting(["A"], "momentum_weighted")
