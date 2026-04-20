"""
Portfolio allocation logic for AMAAM.

Implements the two-sleeve allocation pipeline: applies the momentum filter to
main sleeve selections, redirects weight from negative-momentum main-sleeve
assets to the hedging sleeve, applies the hedging sleeve's own momentum filter
(replacing failures with SHY), and assembles the final portfolio weights.
Handles all edge cases: full main-sleeve momentum failure (100% to hedging),
full hedging failure (100% to SHY), and tie-splitting. See Sections 3.7 and
9.11 of the specification.
"""

import logging
from typing import Dict, List, Optional, Tuple

import pandas as pd

from config.default_config import ModelConfig
from config.etf_universe import CASH_PROXY
from src.portfolio.weighting import apply_weighting

logger = logging.getLogger(__name__)


def apply_momentum_filter(
    selected_tickers: List[str],
    momentum_values: pd.Series,
) -> Tuple[List[str], List[str]]:
    """
    Split selected tickers into momentum-positive and momentum-negative groups.

    The momentum filter is applied after TRank selection: any selected asset
    with M ≤ 0 has its portfolio slot redirected to the hedging sleeve rather
    than being held directly.  Strict positive momentum (M > 0) is required to
    remain in the active sleeve.

    Parameters
    ----------
    selected_tickers : List[str]
        Tickers chosen by ``select_top_n`` for this sleeve.
    momentum_values : pd.Series
        Raw absolute momentum values (decimal returns) indexed by ticker.
        Missing tickers are treated as M = 0 (redirected).

    Returns
    -------
    Tuple[List[str], List[str]]
        ``(active_tickers, redirected_tickers)`` where *active_tickers* have
        M > 0 and *redirected_tickers* have M ≤ 0.  Order within each list
        matches the input order of *selected_tickers*.
    """
    active: List[str] = []
    redirected: List[str] = []
    for t in selected_tickers:
        m = float(momentum_values.get(t, 0.0))
        if m > 0.0:
            active.append(t)
        else:
            redirected.append(t)
    return active, redirected


def compute_hedging_allocation(
    hedging_rankings: List[str],
    momentum_values: pd.Series,
    redirected_weight: float,
    config: ModelConfig,
) -> Dict[str, float]:
    """
    Allocate the redirected portfolio weight across the hedging sleeve.

    Per Section 3.7 steps 4–6, the hedging sleeve always uses equal-slot
    distribution: the redirected weight is divided equally among the selected
    hedging ETFs, with each slot assigned to that ETF if M > 0 or to the cash
    proxy (SHY) if M ≤ 0.

    Parameters
    ----------
    hedging_rankings : List[str]
        Top-N hedging ETFs selected by TRank (may exceed 2 when ties straddle
        the cut-off, per Keller's tie-inclusion convention).
    momentum_values : pd.Series
        Raw absolute momentum values for hedging ETFs.
    redirected_weight : float
        Total portfolio weight to distribute (e.g. 0.333 when 2 of 6 main
        sleeve assets failed the momentum filter).
    config : ModelConfig
        Model configuration (used for ``CASH_PROXY`` identity resolution and
        future extensibility).

    Returns
    -------
    Dict[str, float]
        Ticker → weight mapping.  Values sum to *redirected_weight*.
        Returns an empty dict when *redirected_weight* ≤ 0.

    Notes
    -----
    Hedging-sleeve allocation is intentionally always equal-slot, matching the
    spec's explicit "equally among the top 2" language in Section 3.7 step 5.
    The ``config.weighting_scheme`` applies only to the main sleeve.
    """
    weights: Dict[str, float] = {}

    if redirected_weight <= 0.0:
        return weights

    # No hedging ETFs available — entire redirected weight goes to cash.
    if not hedging_rankings:
        logger.warning(
            "No hedging ETFs selected; routing %.4f to %s.", redirected_weight, CASH_PROXY
        )
        weights[CASH_PROXY] = redirected_weight
        return weights

    n_selected = len(hedging_rankings)
    slot = redirected_weight / n_selected   # equal slot per position

    active, failed = apply_momentum_filter(hedging_rankings, momentum_values)

    # Active hedging ETFs each receive their slot directly.
    for ticker in active:
        weights[ticker] = slot

    # Failed hedging ETF slots flow to the cash proxy.
    # The SHY ticker may already be in `active` (as a selected hedging ETF);
    # accumulate rather than overwrite so no weight is lost.
    shy_weight = len(failed) * slot
    if shy_weight > 0.0:
        weights[CASH_PROXY] = weights.get(CASH_PROXY, 0.0) + shy_weight

    return weights


def compute_monthly_allocation(
    main_rankings: List[str],
    hedging_rankings: List[str],
    main_momentum: pd.Series,
    hedging_momentum: pd.Series,
    config: ModelConfig,
    main_volatility: Optional[pd.Series] = None,
    hedging_volatility: Optional[pd.Series] = None,
) -> Dict[str, float]:
    """
    Assemble the complete monthly portfolio allocation for both sleeves.

    Implements the full Section 3.7 pipeline in a single call:

    1. Apply the momentum filter to the main sleeve selections.
    2. Compute the redirected weight (1/N_main per failing main slot).
    3. Apply the configured weighting scheme to active main sleeve assets.
    4. Route the redirected weight to the hedging sleeve via
       ``compute_hedging_allocation``.
    5. Combine main and hedging weights into a single portfolio dict.

    Edge cases handled:

    * **All main M ≤ 0** — 100 % of the portfolio is redirected to the hedging
      sleeve (``redirected_weight = 1.0``).
    * **All hedging M ≤ 0** — the entire redirected weight goes to SHY.
    * **No main rankings** — 100 % to SHY (defensive fallback).
    * **Tie-inflated selection** — N_main derived from ``len(main_rankings)``,
      so additional tied assets reduce each slot proportionally.

    Parameters
    ----------
    main_rankings : List[str]
        Top-N main sleeve tickers from ``select_top_n`` (typically 6, may be
        more when ties straddle the cut-off).
    hedging_rankings : List[str]
        Top-N hedging sleeve tickers from ``select_top_n`` (typically 2).
    main_momentum : pd.Series
        Raw absolute momentum values for main sleeve tickers.
    hedging_momentum : pd.Series
        Raw absolute momentum values for hedging sleeve tickers.
    config : ModelConfig
        Model configuration supplying ``weighting_scheme``.
    main_volatility : pd.Series, optional
        Yang-Zhang (or alternative) annualised volatility values for main
        sleeve tickers.  Required when
        ``config.weighting_scheme == "inverse_volatility"``.
    hedging_volatility : pd.Series, optional
        Reserved for future use; hedging sleeve always uses equal-slot
        distribution per the spec.

    Returns
    -------
    Dict[str, float]
        ``{ticker: weight}`` for every holding.  Values sum to exactly 1.0
        (subject to floating-point precision; a normalization guard ensures
        this within 1 × 10⁻¹⁰ tolerance).

    Raises
    ------
    ValueError
        Propagated from ``apply_weighting`` if
        ``config.weighting_scheme == "inverse_volatility"`` but
        ``main_volatility`` is ``None``.
    """
    allocation: Dict[str, float] = {}

    # Defensive fallback: no main rankings → 100 % cash.
    if not main_rankings:
        logger.warning("No main sleeve rankings provided; allocating 100%% to %s.", CASH_PROXY)
        return {CASH_PROXY: 1.0}

    n_main = len(main_rankings)
    active_main, redirected_main = apply_momentum_filter(main_rankings, main_momentum)

    # Edge case 1 — all main-sleeve assets have non-positive momentum.
    if not active_main:
        logger.info("All main sleeve assets have M ≤ 0; routing 100%% to hedging sleeve.")
        redirected_weight = 1.0
        active_main_total = 0.0
    else:
        # Each of the n_main positions represents a 1/n_main portfolio slice.
        # Failing positions contribute their slice to redirected_weight.
        redirected_weight = len(redirected_main) / n_main
        active_main_total = len(active_main) / n_main

        # Distribute active_main_total across the active main assets using the
        # configured weighting scheme.  apply_weighting returns unit weights
        # (sum = 1.0); scale by active_main_total to get portfolio weights.
        unit_weights = apply_weighting(
            active_main,
            config.weighting_scheme,
            main_volatility,
        )
        for ticker, unit_w in unit_weights.items():
            allocation[ticker] = unit_w * active_main_total

    # Route redirected weight to the hedging sleeve.
    if redirected_weight > 0.0:
        hedging_weights = compute_hedging_allocation(
            hedging_rankings,
            hedging_momentum,
            redirected_weight,
            config,
        )
        for ticker, weight in hedging_weights.items():
            # Accumulate rather than overwrite: SHY may already be in main.
            allocation[ticker] = allocation.get(ticker, 0.0) + weight

    # Floating-point guard: the algebra guarantees sum = 1.0, but accumulated
    # rounding can drift by ε.  Normalize only when drift exceeds 1e-10.
    total = sum(allocation.values())
    if total > 0.0 and abs(total - 1.0) > 1e-10:
        logger.debug("Normalising allocation (pre-norm sum = %.15f).", total)
        allocation = {t: w / total for t, w in allocation.items()}

    return allocation
