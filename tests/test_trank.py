"""
Unit tests for src/ranking/trank.py.

Covers all cases listed in Section 10.1 of the specification:
  - Ranking direction (higher M = higher rank; lower V/C = higher rank).
  - TRank formula produces correct composite score for known inputs.
  - Top-N selection picks the correct assets.
  - Tiebreaker (M/n) resolves equal weighted-rank sums.
  - Tie-inclusion convention when assets share the N-th score.
  - T term sign: T = +2 lowers TRank, T = -2 raises it.
"""

import math

import pandas as pd
import pytest

from config.default_config import ModelConfig
from src.ranking.trank import compute_trank, rank_assets, select_top_n


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cfg() -> ModelConfig:
    """Default config: wM=0.65, wV=0.25, wC=0.10, wT=1.0."""
    return ModelConfig()


def _s(values: dict) -> pd.Series:
    return pd.Series(values)


# ---------------------------------------------------------------------------
# TestRankAssets
# ---------------------------------------------------------------------------

class TestRankAssets:
    """Ordinal ranking helper behaves correctly for both ranking directions."""

    def test_ascending_highest_value_gets_top_rank(self):
        """Higher raw value → largest rank number (used for M)."""
        ranks = rank_assets(_s({"A": 0.20, "B": 0.10, "C": 0.05}), ascending=True)
        assert ranks["A"] == 3
        assert ranks["B"] == 2
        assert ranks["C"] == 1

    def test_descending_lowest_value_gets_top_rank(self):
        """Lower raw value → largest rank number (used for V and C)."""
        ranks = rank_assets(_s({"A": 0.10, "B": 0.20, "C": 0.30}), ascending=False)
        assert ranks["A"] == 3
        assert ranks["B"] == 2
        assert ranks["C"] == 1

    def test_tied_values_receive_min_rank(self):
        """Tied values receive the minimum (lowest) rank in the group — Keller
        tie-inclusion convention ensures both are eligible for selection."""
        ranks = rank_assets(_s({"A": 0.10, "B": 0.10, "C": 0.05}), ascending=True)
        # A and B are tied for 2nd-best; each gets rank 2, not 2.5.
        assert ranks["A"] == 2
        assert ranks["B"] == 2
        assert ranks["C"] == 1

    def test_nan_input_produces_nan_rank(self):
        """NaN factor values must propagate to NaN ranks, not receive an ordinal."""
        ranks = rank_assets(_s({"A": 0.10, "B": float("nan"), "C": 0.05}), ascending=True)
        assert ranks["A"] == 2
        assert math.isnan(ranks["B"])
        assert ranks["C"] == 1

    def test_single_asset_gets_rank_one(self):
        ranks = rank_assets(_s({"A": 0.15}), ascending=True)
        assert ranks["A"] == 1


# ---------------------------------------------------------------------------
# TestComputeTrank — formula arithmetic
# ---------------------------------------------------------------------------

class TestComputeTrank:
    """TRank formula matches hand calculations from Section 3.1."""

    def test_known_composite_score_three_assets(self):
        """
        Hand-calculated TRank for a 3-asset sleeve where A dominates all factors.

        Setup (all T = −2, wM=0.65, wV=0.25, wC=0.10, wT=1.0):
          Asset   M      V     C
          A      0.20   0.10  0.10   →  Rank(M)=3, Rank(V)=3, Rank(C)=3
          B      0.10   0.20  0.20   →  Rank(M)=2, Rank(V)=2, Rank(C)=2
          C      0.05   0.30  0.30   →  Rank(M)=1, Rank(V)=1, Rank(C)=1

        Because all three ranks are identical for each asset, the weighted sum
        collapses to (wM + wV + wC) · rank = 1.0 · rank regardless of weights:
        TRank_A = 1.0·3 − 1.0·(−2) + 0.20/3 = 3 + 2 + 0.20/3  → 5 + 0.20/3
        TRank_B = 1.0·2 + 2 + 0.10/3         → 4 + 0.10/3
        TRank_C = 1.0·1 + 2 + 0.05/3         → 3 + 0.05/3

        Subtracting the T contribution (−(−2) = +2 per asset):
        TRank_A = 1.0 + 0.20/3,  TRank_B = 0.0 + 0.10/3,  TRank_C = −1.0 + 0.05/3
        """
        cfg = _cfg()
        m_raw = _s({"A": 0.20, "B": 0.10, "C": 0.05})
        v_raw = _s({"A": 0.10, "B": 0.20, "C": 0.30})
        c_raw = _s({"A": 0.10, "B": 0.20, "C": 0.30})
        t_raw = _s({"A": -2.0, "B": -2.0, "C": -2.0})

        m_ranks = rank_assets(m_raw, ascending=True)
        v_ranks = rank_assets(v_raw, ascending=False)
        c_ranks = rank_assets(c_raw, ascending=False)

        tranks = compute_trank(m_ranks, v_ranks, c_ranks, t_raw, m_raw, cfg)

        # With T = −2 and + wT·T: contribution = +1.0·(−2) = −2 per asset.
        # Weighted rank sum for A: 0.65·3 + 0.25·3 + 0.10·3 = 3.0, plus −2 = 1.0.
        assert tranks["A"] == pytest.approx(1.0 + 0.20 / 3, rel=1e-6)
        assert tranks["B"] == pytest.approx(0.0 + 0.10 / 3, rel=1e-6)
        assert tranks["C"] == pytest.approx(-1.0 + 0.05 / 3, rel=1e-6)
        assert tranks["A"] > tranks["B"] > tranks["C"]

    def test_t_plus_two_raises_trank_by_two_wt(self):
        """
        T = +2 raises TRank relative to T = −2 by exactly 2·wT per asset.
        With default wT = 1.0 the bonus is 4 points (from +wT·(+2) vs +wT·(−2)).
        """
        cfg = _cfg()
        ranks = _s({"A": 3.0})
        m_raw = _s({"A": 0.10})

        down = compute_trank(ranks, ranks, ranks, _s({"A": -2.0}), m_raw, cfg)
        up   = compute_trank(ranks, ranks, ranks, _s({"A": +2.0}), m_raw, cfg)

        assert up["A"] - down["A"] == pytest.approx(4.0 * cfg.weight_trend)

    def test_m_over_n_tiebreaker_is_present(self):
        """
        The tiebreaker term M/n is added to the weighted-rank base.
        Verified with n=1 where M/n = M_raw exactly.
        """
        cfg = _cfg()
        ranks = _s({"A": 1.0})
        m_raw = _s({"A": 0.30})
        t_raw = _s({"A": -2.0})

        trank = compute_trank(ranks, ranks, ranks, t_raw, m_raw, cfg)

        base = (
            cfg.weight_momentum * 1.0
            + cfg.weight_volatility * 1.0
            + cfg.weight_correlation * 1.0
            + cfg.weight_trend * (-2.0)
        )
        assert trank["A"] == pytest.approx(base + 0.30 / 1, rel=1e-9)

    def test_nan_in_factor_rank_propagates_to_trank(self):
        """NaN in any input rank or trend value must yield a NaN TRank."""
        cfg = _cfg()
        m_ranks = _s({"A": 2.0, "B": float("nan")})
        v_ranks = _s({"A": 2.0, "B": 1.0})
        c_ranks = _s({"A": 2.0, "B": 1.0})
        t_raw   = _s({"A": -2.0, "B": -2.0})
        m_raw   = _s({"A": 0.10, "B": 0.05})

        tranks = compute_trank(m_ranks, v_ranks, c_ranks, t_raw, m_raw, cfg)

        assert not math.isnan(tranks["A"])
        assert math.isnan(tranks["B"])

    def test_empty_input_returns_empty_series(self):
        """Empty inputs must return an empty Series without raising."""
        cfg = _cfg()
        empty = pd.Series(dtype=float)
        tranks = compute_trank(empty, empty, empty, empty, empty, cfg)
        assert len(tranks) == 0


# ---------------------------------------------------------------------------
# TestTiebreakerResolution
# ---------------------------------------------------------------------------

class TestTiebreakerResolution:
    """M/n tiebreaker resolves equal weighted-rank sums in favour of higher M."""

    def test_tiebreaker_favours_higher_raw_momentum(self):
        """
        Two assets with equal weighted-rank sums differ only in raw M.
        The one with higher M must win.

        Engineered tie (n=4, wM=0.65, wV=0.25, wC=0.10):
          A: Rank(M)=1, Rank(V)=4, Rank(C)=1
             base = 0.65·1 + 0.25·4 + 0.10·1 = 0.65 + 1.00 + 0.10 = 1.75
          B: Rank(M)=2, Rank(V)=1, Rank(C)=2
             base = 0.65·2 + 0.25·1 + 0.10·2 = 1.30 + 0.25 + 0.20 = 1.75

        Equal base and equal T; raw M decides: B has M=0.10 > A's M=0.05.
        Expected: TRank_B − TRank_A = (0.10 − 0.05) / 4.

        C and D are dummy assets that provide the rank context; only A and B
        are checked in the assertions.
        """
        cfg = _cfg()
        # M ascending: A(0.05)=1, B(0.10)=2, C(0.20)=3, D(0.30)=4
        m_raw = _s({"A": 0.05, "B": 0.10, "C": 0.20, "D": 0.30})
        # V ascending=False (lower=better): A(0.10)=4, C(0.20)=3, D(0.30)=2, B(0.40)=1
        v_raw = _s({"A": 0.10, "C": 0.20, "D": 0.30, "B": 0.40})
        # C ascending=False (lower=better): C(0.10)=4, D(0.20)=3, B(0.30)=2, A(0.40)=1
        c_raw = _s({"C": 0.10, "D": 0.20, "B": 0.30, "A": 0.40})
        t_raw = _s({"A": -2.0, "B": -2.0, "C": -2.0, "D": -2.0})

        m_ranks = rank_assets(m_raw, ascending=True)
        v_ranks = rank_assets(v_raw, ascending=False)
        c_ranks = rank_assets(c_raw, ascending=False)

        tranks = compute_trank(m_ranks, v_ranks, c_ranks, t_raw, m_raw, cfg)

        assert tranks["B"] > tranks["A"]
        assert tranks["B"] - tranks["A"] == pytest.approx((0.10 - 0.05) / 4, rel=1e-9)

    def test_tiebreaker_divisor_is_n_assets(self):
        """
        M/n uses the total number of assets as the divisor.
        Four assets all receiving identical ranks: TRank gaps must equal the
        corresponding M gaps divided by 4.
        """
        cfg = _cfg()
        tickers = ["A", "B", "C", "D"]
        same_rank = pd.Series([2.0, 2.0, 2.0, 2.0], index=tickers)
        t_raw     = pd.Series([-2.0] * 4, index=tickers)
        m_raw     = pd.Series([0.40, 0.30, 0.20, 0.10], index=tickers)

        tranks = compute_trank(same_rank, same_rank, same_rank, t_raw, m_raw, cfg)

        assert tranks["A"] - tranks["B"] == pytest.approx((0.40 - 0.30) / 4, rel=1e-9)
        assert tranks["C"] - tranks["D"] == pytest.approx((0.20 - 0.10) / 4, rel=1e-9)


# ---------------------------------------------------------------------------
# TestSelectTopN
# ---------------------------------------------------------------------------

class TestSelectTopN:
    """Top-N selection returns correct assets and respects edge cases."""

    def test_basic_top_two_from_four(self):
        """Standard case: top-2 from 4 distinct TRank scores."""
        selected = select_top_n(_s({"A": 5.0, "B": 4.0, "C": 3.0, "D": 2.0}), n=2)
        assert set(selected) == {"A", "B"}

    def test_result_sorted_descending_by_trank(self):
        """Returned list must be ordered best → worst."""
        selected = select_top_n(_s({"A": 5.0, "B": 4.0, "C": 3.0}), n=2)
        assert selected[0] == "A"
        assert selected[1] == "B"

    def test_tie_at_cutoff_includes_all_tied_assets(self):
        """
        Keller tie-inclusion: when N-th and (N+1)-th assets share the same
        TRank, ALL of them are returned.
        Top-2 requested; B and C tie at position 2 → return A, B, C (3 assets).
        """
        selected = select_top_n(_s({"A": 5.0, "B": 4.0, "C": 4.0, "D": 3.0}), n=2)
        assert set(selected) == {"A", "B", "C"}
        assert "D" not in selected

    def test_fewer_assets_than_n_returns_all(self):
        """Sleeve smaller than top_n: return every available asset."""
        selected = select_top_n(_s({"A": 3.0, "B": 2.0}), n=6)
        assert set(selected) == {"A", "B"}

    def test_nan_trank_excluded_from_selection(self):
        """Assets with NaN TRank are invisible to the selection logic."""
        selected = select_top_n(_s({"A": 5.0, "B": float("nan"), "C": 3.0}), n=2)
        assert "B" not in selected
        assert set(selected) == {"A", "C"}

    def test_all_nan_raises_value_error(self):
        """All-NaN input cannot produce a valid selection."""
        with pytest.raises(ValueError):
            select_top_n(_s({"A": float("nan"), "B": float("nan")}), n=1)

    def test_top_one_returns_single_best(self):
        """n=1 must return exactly the asset with the highest TRank."""
        selected = select_top_n(_s({"A": 2.0, "B": 5.0, "C": 1.0}), n=1)
        assert selected == ["B"]

    def test_n_equals_total_returns_all_sorted(self):
        """Requesting exactly all assets returns them all, sorted descending."""
        selected = select_top_n(_s({"A": 3.0, "B": 1.0, "C": 2.0}), n=3)
        assert selected == ["A", "C", "B"]


# ---------------------------------------------------------------------------
# TestSelectionHysteresis — exit_buffer keeps incumbents while inside the zone
# ---------------------------------------------------------------------------

class TestSelectionHysteresis:
    """Incumbents persist until they drop below top-(n + exit_buffer)."""

    def test_no_buffer_standard_exit(self):
        """exit_buffer=0 behaves identically to the standard top-N rule."""
        tranks = _s({"A": 5.0, "B": 4.0, "C": 3.0, "D": 2.0})
        # A and B were previously selected; C is now ranked 3rd (outside top-2).
        selected = select_top_n(tranks, n=2, prev_selected=["A", "B"], exit_buffer=0)
        assert set(selected) == {"A", "B"}

    def test_incumbent_retained_within_buffer(self):
        """
        With exit_buffer=1 and n=2, exit threshold is at rank 3.
        B has dropped to rank 3 (TRank=3.0); it is retained as an incumbent.
        D is at rank 4 and never held — not retained.
        """
        tranks = _s({"A": 5.0, "C": 4.0, "B": 3.0, "D": 2.0})
        # Previously held: A and B. B slipped from rank 2 to rank 3.
        selected = select_top_n(tranks, n=2, prev_selected=["A", "B"], exit_buffer=1)
        assert "A" in selected  # new top-2 entry
        assert "B" in selected  # incumbent retained within buffer
        assert "D" not in selected  # not held and outside top-2

    def test_incumbent_dropped_outside_buffer(self):
        """
        With exit_buffer=1 and n=2, exit threshold is at rank 3.
        B has dropped to rank 4 (outside the exit zone) — must be dropped.
        """
        tranks = _s({"A": 5.0, "C": 4.0, "D": 3.0, "B": 2.0})
        selected = select_top_n(tranks, n=2, prev_selected=["A", "B"], exit_buffer=1)
        assert "A" in selected
        assert "C" in selected  # enters as new top-2 asset
        assert "B" not in selected  # incumbent dropped: rank 4 > n + buffer = 3

    def test_new_entrant_requires_top_n_rank(self):
        """
        An asset NOT in prev_selected must rank within top-N to enter,
        regardless of exit_buffer.
        """
        tranks = _s({"A": 5.0, "B": 4.0, "C": 3.0, "D": 2.0})
        # Previously held: A only. C is ranked 3rd (outside top-2) — must not enter.
        selected = select_top_n(tranks, n=2, prev_selected=["A"], exit_buffer=2)
        assert "A" in selected
        assert "B" in selected   # standard top-2 entry
        assert "C" not in selected  # ranked 3rd, not previously held

    def test_buffer_larger_than_pool_retains_all_incumbents(self):
        """
        When n + exit_buffer >= total assets, all incumbents are retained
        regardless of their current rank.
        """
        tranks = _s({"A": 5.0, "B": 4.0, "C": 3.0})
        # n=2, exit_buffer=5 → exit zone covers all 3 assets.
        selected = select_top_n(tranks, n=2, prev_selected=["A", "C"], exit_buffer=5)
        assert "A" in selected
        assert "C" in selected  # retained: exit zone covers the whole pool

    def test_none_prev_selected_is_standard_topn(self):
        """Passing prev_selected=None uses the standard top-N rule."""
        tranks = _s({"A": 5.0, "B": 4.0, "C": 3.0, "D": 2.0})
        selected_with_none  = select_top_n(tranks, n=2, prev_selected=None, exit_buffer=1)
        selected_no_args    = select_top_n(tranks, n=2)
        assert set(selected_with_none) == set(selected_no_args)

    def test_empty_prev_selected_is_standard_topn(self):
        """Passing prev_selected=[] uses the standard top-N rule."""
        tranks = _s({"A": 5.0, "B": 4.0, "C": 3.0, "D": 2.0})
        selected = select_top_n(tranks, n=2, prev_selected=[], exit_buffer=2)
        assert set(selected) == {"A", "B"}

    def test_buffer_two_keeps_incumbent_at_rank_four(self):
        """
        With n=2, exit_buffer=2, exit threshold is at rank 4.
        An incumbent at rank 4 is at the exact boundary — it should be retained
        (its TRank >= cutoff_exit, which is the 4th-best score).
        """
        tranks = _s({"A": 5.0, "C": 4.0, "D": 3.0, "B": 2.0})
        # exit_n = 4 → cutoff_exit = sorted[3] = 2.0 → B (TRank=2.0) >= 2.0 → retained.
        selected = select_top_n(tranks, n=2, prev_selected=["A", "B"], exit_buffer=2)
        assert "B" in selected

    def test_buffer_two_drops_incumbent_at_rank_five(self):
        """
        With n=2, exit_buffer=2, exit threshold is at rank 4.
        An incumbent ranked 5th is outside the exit zone — must be dropped.
        """
        tranks = _s({"A": 5.0, "C": 4.0, "D": 3.0, "E": 2.0, "B": 1.0})
        # exit_n = 4 → cutoff_exit = sorted[3] = 2.0 → B (TRank=1.0) < 2.0 → dropped.
        selected = select_top_n(tranks, n=2, prev_selected=["A", "B"], exit_buffer=2)
        assert "B" not in selected


# ---------------------------------------------------------------------------
# TestRankingDirection — end-to-end: raw factor values → ranks → TRank order
# ---------------------------------------------------------------------------

class TestRankingDirection:
    """Higher M, lower V, and lower C each independently raise TRank,
    all else equal."""

    def _baseline(self):
        """Two assets tied on every factor."""
        return (
            _s({"A": 0.10, "B": 0.10}),  # m_raw
            _s({"A": 0.20, "B": 0.20}),  # v_raw
            _s({"A": 0.30, "B": 0.30}),  # c_raw
            _s({"A": -2.0, "B": -2.0}),  # t_raw
        )

    def test_higher_momentum_raises_trank(self):
        cfg = _cfg()
        m, v, c, t = self._baseline()
        m["A"] = 0.20
        tranks = compute_trank(
            rank_assets(m, ascending=True),
            rank_assets(v, ascending=False),
            rank_assets(c, ascending=False),
            t, m, cfg,
        )
        assert tranks["A"] > tranks["B"]

    def test_lower_volatility_raises_trank(self):
        cfg = _cfg()
        m, v, c, t = self._baseline()
        v["A"] = 0.10   # lower vol → better
        tranks = compute_trank(
            rank_assets(m, ascending=True),
            rank_assets(v, ascending=False),
            rank_assets(c, ascending=False),
            t, m, cfg,
        )
        assert tranks["A"] > tranks["B"]

    def test_lower_correlation_raises_trank(self):
        cfg = _cfg()
        m, v, c, t = self._baseline()
        c["A"] = 0.10   # lower avg correlation → better
        tranks = compute_trank(
            rank_assets(m, ascending=True),
            rank_assets(v, ascending=False),
            rank_assets(c, ascending=False),
            t, m, cfg,
        )
        assert tranks["A"] > tranks["B"]

    def test_t_plus_two_raises_trank_vs_identical_peer(self):
        """
        An asset with T = +2 scores higher than an identical peer at T = −2,
        per the +wT·T term (uptrend confirmation boosts TRank → more likely selected).
        """
        cfg = _cfg()
        m, v, c, t = self._baseline()
        t["A"] = +2.0
        tranks = compute_trank(
            rank_assets(m, ascending=True),
            rank_assets(v, ascending=False),
            rank_assets(c, ascending=False),
            t, m, cfg,
        )
        assert tranks["A"] > tranks["B"]
