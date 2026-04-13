"""
Unit tests for src/portfolio/allocation.py.

Covers: momentum filter separation (positive vs negative M), weight redirection
(2 of 6 negative → 2/6 redirected), all-negative main sleeve (100% to hedging),
all-negative hedging sleeve (100% to SHY), and weight-sum invariant (always 1.0).
See Section 10.1 of the specification.
"""

import pytest
import pandas as pd

from config.default_config import ModelConfig
from config.etf_universe import CASH_PROXY
from src.portfolio.allocation import (
    apply_momentum_filter,
    compute_hedging_allocation,
    compute_monthly_allocation,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cfg(**overrides) -> ModelConfig:
    cfg = ModelConfig()
    for k, v in overrides.items():
        object.__setattr__(cfg, k, v)
    return cfg


def _mom(**kwargs) -> pd.Series:
    return pd.Series(kwargs)


MAIN_6  = ["A", "B", "C", "D", "E", "F"]   # canonical 6-asset main sleeve
HEDGE_2 = ["H1", "H2"]                       # canonical 2-asset hedging sleeve


# ---------------------------------------------------------------------------
# TestApplyMomentumFilter
# ---------------------------------------------------------------------------

class TestApplyMomentumFilter:

    def test_separates_positive_and_negative(self):
        """Basic split: 2 positive, 2 negative."""
        mom = _mom(A=0.05, B=-0.03, C=0.10, D=-0.01)
        active, redirected = apply_momentum_filter(["A", "B", "C", "D"], mom)
        assert set(active) == {"A", "C"}
        assert set(redirected) == {"B", "D"}

    def test_zero_momentum_is_redirected(self):
        """M = 0 is not strictly positive; the asset must be redirected."""
        mom = _mom(A=0.0, B=0.05)
        active, redirected = apply_momentum_filter(["A", "B"], mom)
        assert "A" in redirected
        assert "B" in active

    def test_all_positive_no_redirected(self):
        mom = _mom(A=0.10, B=0.05, C=0.01)
        active, redirected = apply_momentum_filter(["A", "B", "C"], mom)
        assert set(active) == {"A", "B", "C"}
        assert redirected == []

    def test_all_negative_no_active(self):
        mom = _mom(A=-0.10, B=-0.05)
        active, redirected = apply_momentum_filter(["A", "B"], mom)
        assert active == []
        assert set(redirected) == {"A", "B"}

    def test_preserves_input_order(self):
        """Active and redirected lists must follow the order of selected_tickers."""
        mom = _mom(A=0.10, B=-0.01, C=0.05, D=-0.02)
        active, redirected = apply_momentum_filter(["A", "B", "C", "D"], mom)
        assert active == ["A", "C"]
        assert redirected == ["B", "D"]

    def test_missing_ticker_treated_as_zero(self):
        """A ticker absent from momentum_values is treated as M = 0 → redirected."""
        mom = _mom(A=0.05)   # B is missing
        active, redirected = apply_momentum_filter(["A", "B"], mom)
        assert "A" in active
        assert "B" in redirected


# ---------------------------------------------------------------------------
# TestComputeHedgingAllocation
# ---------------------------------------------------------------------------

class TestComputeHedgingAllocation:

    def test_both_active_split_equally(self):
        """Both hedging ETFs pass momentum filter → each gets half the weight."""
        mom = _mom(H1=0.05, H2=0.03)
        weights = compute_hedging_allocation(HEDGE_2, mom, 1/3, _cfg())
        assert weights["H1"] == pytest.approx(1/6)
        assert weights["H2"] == pytest.approx(1/6)

    def test_one_fails_slot_goes_to_shy(self):
        """H2 fails (M ≤ 0): H1 gets its slot, SHY gets H2's slot."""
        mom = _mom(H1=0.05, H2=-0.01)
        weights = compute_hedging_allocation(HEDGE_2, mom, 0.4, _cfg())
        assert weights["H1"] == pytest.approx(0.2)
        assert weights[CASH_PROXY] == pytest.approx(0.2)

    def test_both_fail_all_to_shy(self):
        """Both hedging ETFs fail → entire redirected weight to SHY."""
        mom = _mom(H1=-0.02, H2=-0.05)
        weights = compute_hedging_allocation(HEDGE_2, mom, 0.5, _cfg())
        assert weights.get(CASH_PROXY, 0.0) == pytest.approx(0.5)
        assert "H1" not in weights
        assert "H2" not in weights

    def test_weights_sum_to_redirected_weight(self):
        """Total of returned weights must equal the redirected_weight argument."""
        for rw in [1/6, 2/6, 1.0]:
            mom = _mom(H1=0.03, H2=-0.01)
            w = compute_hedging_allocation(HEDGE_2, mom, rw, _cfg())
            assert sum(w.values()) == pytest.approx(rw)

    def test_zero_redirected_weight_returns_empty(self):
        """No weight to distribute → return empty dict, not a zero-weight entry."""
        mom = _mom(H1=0.05, H2=0.03)
        weights = compute_hedging_allocation(HEDGE_2, mom, 0.0, _cfg())
        assert weights == {}

    def test_empty_hedging_rankings_all_to_shy(self):
        """No hedging ETFs available → full redirected weight to SHY."""
        weights = compute_hedging_allocation([], _mom(), 0.25, _cfg())
        assert weights[CASH_PROXY] == pytest.approx(0.25)

    def test_three_way_tie_slots_split_in_thirds(self):
        """Keller tie-inclusion: 3 hedging ETFs selected → each gets 1/3 of weight."""
        mom = _mom(H1=0.05, H2=0.03, H3=0.01)
        weights = compute_hedging_allocation(["H1", "H2", "H3"], mom, 0.3, _cfg())
        assert weights["H1"] == pytest.approx(0.1)
        assert weights["H2"] == pytest.approx(0.1)
        assert weights["H3"] == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# TestComputeMonthlyAllocation
# ---------------------------------------------------------------------------

class TestComputeMonthlyAllocation:

    def _all_positive_main(self, tickers=MAIN_6):
        return _mom(**{t: 0.05 for t in tickers})

    def _all_positive_hedge(self, tickers=HEDGE_2):
        return _mom(**{t: 0.03 for t in tickers})

    # ── Normal operations ────────────────────────────────────────────────────

    def test_all_active_no_hedging_needed(self):
        """All 6 main assets pass filter → full weight in main, no hedging."""
        alloc = compute_monthly_allocation(
            MAIN_6, HEDGE_2,
            self._all_positive_main(), self._all_positive_hedge(),
            _cfg(),
        )
        assert CASH_PROXY not in alloc
        assert all(t not in alloc for t in HEDGE_2)
        for t in MAIN_6:
            assert alloc[t] == pytest.approx(1 / 6)

    def test_two_of_six_negative_redirects_two_sixths(self):
        """
        2 of 6 main ETFs have M ≤ 0.
        Active main: 4 × (1/6) each.
        Redirected to hedging: 2/6 total → each of H1, H2 gets 1/6.
        """
        mom_main = _mom(A=0.10, B=0.08, C=0.05, D=0.02, E=-0.01, F=-0.03)
        mom_hedge = _mom(H1=0.05, H2=0.03)
        alloc = compute_monthly_allocation(MAIN_6, HEDGE_2, mom_main, mom_hedge, _cfg())

        for t in ["A", "B", "C", "D"]:
            assert alloc[t] == pytest.approx(1 / 6), f"{t} weight wrong"
        assert alloc["H1"] == pytest.approx(1 / 6)
        assert alloc["H2"] == pytest.approx(1 / 6)
        assert "E" not in alloc
        assert "F" not in alloc

    def test_one_hedging_fails_slot_goes_to_shy(self):
        """
        2 main ETFs redirected → 2/6 to hedging.
        H2 has M ≤ 0 → H1 gets 1/6, SHY gets 1/6.
        """
        mom_main = _mom(A=0.10, B=0.08, C=0.05, D=0.02, E=-0.01, F=-0.03)
        mom_hedge = _mom(H1=0.05, H2=-0.02)
        alloc = compute_monthly_allocation(MAIN_6, HEDGE_2, mom_main, mom_hedge, _cfg())

        assert alloc["H1"] == pytest.approx(1 / 6)
        assert alloc[CASH_PROXY] == pytest.approx(1 / 6)
        assert "H2" not in alloc

    # ── Edge case 1: all main M ≤ 0 ─────────────────────────────────────────

    def test_all_main_negative_100pct_to_hedging(self):
        """All 6 main ETFs fail filter → 100% to hedging sleeve."""
        mom_main = _mom(**{t: -0.01 for t in MAIN_6})
        mom_hedge = _mom(H1=0.05, H2=0.03)
        alloc = compute_monthly_allocation(MAIN_6, HEDGE_2, mom_main, mom_hedge, _cfg())

        # No main ETFs in allocation.
        for t in MAIN_6:
            assert t not in alloc
        # H1 and H2 split the full portfolio.
        assert alloc["H1"] == pytest.approx(0.5)
        assert alloc["H2"] == pytest.approx(0.5)

    # ── Edge case 2: all hedging M ≤ 0 ──────────────────────────────────────

    def test_all_hedging_negative_redirected_weight_to_shy(self):
        """
        2 main ETFs fail (redirecting 2/6) AND both hedging ETFs fail.
        The redirected 2/6 goes entirely to SHY.
        """
        mom_main = _mom(A=0.10, B=0.08, C=0.05, D=0.02, E=-0.01, F=-0.03)
        mom_hedge = _mom(H1=-0.04, H2=-0.06)
        alloc = compute_monthly_allocation(MAIN_6, HEDGE_2, mom_main, mom_hedge, _cfg())

        assert alloc[CASH_PROXY] == pytest.approx(2 / 6)
        for t in HEDGE_2:
            assert t not in alloc

    def test_all_main_and_all_hedging_negative_100pct_shy(self):
        """Worst case: all main AND all hedging fail → 100% SHY."""
        mom_main = _mom(**{t: -0.05 for t in MAIN_6})
        mom_hedge = _mom(**{t: -0.05 for t in HEDGE_2})
        alloc = compute_monthly_allocation(MAIN_6, HEDGE_2, mom_main, mom_hedge, _cfg())

        assert alloc[CASH_PROXY] == pytest.approx(1.0)
        for t in MAIN_6 + HEDGE_2:
            assert t not in alloc

    # ── Weight-sum invariant ─────────────────────────────────────────────────

    def test_weights_sum_to_one_all_active(self):
        alloc = compute_monthly_allocation(
            MAIN_6, HEDGE_2,
            self._all_positive_main(), self._all_positive_hedge(),
            _cfg(),
        )
        assert sum(alloc.values()) == pytest.approx(1.0)

    def test_weights_sum_to_one_partial_redirection(self):
        mom_main = _mom(A=0.10, B=0.08, C=0.05, D=0.02, E=-0.01, F=-0.03)
        mom_hedge = _mom(H1=0.05, H2=-0.02)
        alloc = compute_monthly_allocation(MAIN_6, HEDGE_2, mom_main, mom_hedge, _cfg())
        assert sum(alloc.values()) == pytest.approx(1.0)

    def test_weights_sum_to_one_all_main_negative(self):
        mom_main = _mom(**{t: -0.01 for t in MAIN_6})
        alloc = compute_monthly_allocation(
            MAIN_6, HEDGE_2, mom_main, self._all_positive_hedge(), _cfg()
        )
        assert sum(alloc.values()) == pytest.approx(1.0)

    def test_weights_sum_to_one_all_negative(self):
        mom_main = _mom(**{t: -0.01 for t in MAIN_6})
        mom_hedge = _mom(**{t: -0.01 for t in HEDGE_2})
        alloc = compute_monthly_allocation(MAIN_6, HEDGE_2, mom_main, mom_hedge, _cfg())
        assert sum(alloc.values()) == pytest.approx(1.0)

    # ── Tie-inflated selection ───────────────────────────────────────────────

    def test_seven_main_assets_due_to_tie(self):
        """
        When tie-inclusion yields 7 main assets, each slot = 1/7.
        4 active → 4/7 total main weight; 3 failing → 3/7 to hedging.
        """
        main_7 = MAIN_6 + ["G"]
        mom_main = _mom(A=0.10, B=0.08, C=0.05, D=0.02, E=-0.01, F=-0.03, G=-0.02)
        mom_hedge = _mom(H1=0.05, H2=0.03)
        alloc = compute_monthly_allocation(main_7, HEDGE_2, mom_main, mom_hedge, _cfg())

        for t in ["A", "B", "C", "D"]:
            assert alloc[t] == pytest.approx(1 / 7), f"{t} weight wrong"
        assert alloc["H1"] == pytest.approx(3 / 14)
        assert alloc["H2"] == pytest.approx(3 / 14)
        assert sum(alloc.values()) == pytest.approx(1.0)

    # ── No main rankings ─────────────────────────────────────────────────────

    def test_empty_main_rankings_returns_100pct_shy(self):
        alloc = compute_monthly_allocation(
            [], HEDGE_2, _mom(), self._all_positive_hedge(), _cfg()
        )
        assert alloc == {CASH_PROXY: 1.0}

    # ── Inverse-volatility weighting ─────────────────────────────────────────

    def test_inverse_vol_weights_within_main_sleeve(self):
        """
        With inv-vol scheme and V_A=0.10, V_B=0.20 (2 active, 4 failing):
          total active weight = 2/6
          unit weights: A=2/3, B=1/3
          portfolio weights: A = (2/3)×(2/6) = 2/9, B = (1/3)×(2/6) = 1/9
        Hedging gets 4/6, split equally: H1=H2=2/6.
        """
        mom_main = _mom(A=0.10, B=0.05, C=-0.01, D=-0.02, E=-0.01, F=-0.03)
        mom_hedge = _mom(H1=0.05, H2=0.03)
        vols_main = pd.Series({"A": 0.10, "B": 0.20, "C": 0.15, "D": 0.12,
                                "E": 0.18, "F": 0.14})
        cfg = _cfg(weighting_scheme="inverse_volatility")
        alloc = compute_monthly_allocation(
            MAIN_6, HEDGE_2, mom_main, mom_hedge, cfg, main_volatility=vols_main
        )

        assert alloc["A"] == pytest.approx(2 / 9)
        assert alloc["B"] == pytest.approx(1 / 9)
        assert alloc["H1"] == pytest.approx(2 / 6)
        assert alloc["H2"] == pytest.approx(2 / 6)
        assert sum(alloc.values()) == pytest.approx(1.0)
