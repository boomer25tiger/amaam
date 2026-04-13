"""
TRank ranking engine for AMAAM.

Combines the four factor scores (M, V, C, T) into a composite TRank score using
the formula from Section 3.1 of the specification. Handles ordinal ranking of
the M, V, and C factors, applies the raw T value, and adds the M/n tiebreaker
term. Selects the top-N assets per sleeve each month, respecting the tie-inclusion
convention from Keller (2012). Used identically by the main sleeve and the
hedging sleeve. See Section 9.10 of the specification.
"""

import logging
from typing import Dict, List

import pandas as pd

from config.default_config import ModelConfig

logger = logging.getLogger(__name__)


def rank_assets(values: pd.Series, ascending: bool = True) -> pd.Series:
    """
    Assign ordinal ranks 1..N to a Series of factor values.

    The convention throughout AMAAM is that rank N (the highest integer) is
    always the *best* asset regardless of whether higher or lower raw values
    are preferred.  The ``ascending`` flag controls which raw-value direction
    maps to the top rank:

    * ``ascending=True``  → higher raw value = higher rank (used for M).
    * ``ascending=False`` → lower raw value = higher rank (used for V and C).

    NaN entries receive NaN ranks and are excluded from selection.

    Parameters
    ----------
    values : pd.Series
        Raw factor values indexed by ticker.
    ascending : bool
        If True, the largest value receives rank N.  If False, the smallest
        value receives rank N.

    Returns
    -------
    pd.Series
        Integer ranks 1..N with the same index as *values*.  NaN where input
        is NaN.
    """
    # pandas rank: method="min" gives tied values the same (lowest) rank,
    # consistent with Keller's tie-inclusion convention in select_top_n.
    # na_option="keep" preserves NaN rather than assigning a rank.
    return values.rank(method="min", ascending=ascending, na_option="keep")


def compute_trank(
    momentum_ranks: pd.Series,
    volatility_ranks: pd.Series,
    correlation_ranks: pd.Series,
    trend_values: pd.Series,
    raw_momentum: pd.Series,
    config: ModelConfig,
) -> pd.Series:
    """
    Apply the TRank formula to produce a composite score for each asset.

    Formula (Section 3.1)::

        TRank = (wM·Rank(M) + wV·Rank(V) + wC·Rank(C) − wT·T) + M/n

    Where:

    * ``Rank(M)`` — ordinal rank of Absolute Momentum (ascending, best = N).
    * ``Rank(V)`` — ordinal rank of Volatility (descending, lowest vol = N).
    * ``Rank(C)`` — ordinal rank of Avg Relative Correlation (descending,
      lowest corr = N).
    * ``T`` — raw ATR Trend/Breakout signal (±2), NOT ranked.
    * ``M`` — raw Absolute Momentum value used as a tiebreaker.
    * ``n`` — number of assets in the sleeve (``len(raw_momentum)``).

    The ``M/n`` tiebreaker is small relative to the weighted-rank terms but
    guarantees that two assets with identical weighted-rank sums can still be
    ordered by momentum, preventing arbitrary selection when ranks are tied.

    Higher TRank = better.  The top-N assets by TRank are selected each month.

    Parameters
    ----------
    momentum_ranks : pd.Series
        Ordinal ranks of M (1..N, ascending), indexed by ticker.
    volatility_ranks : pd.Series
        Ordinal ranks of V (1..N, descending), indexed by ticker.
    correlation_ranks : pd.Series
        Ordinal ranks of C (1..N, descending), indexed by ticker.
    trend_values : pd.Series
        Raw T values (+2 or −2), indexed by ticker.
    raw_momentum : pd.Series
        Raw M values (decimal returns), indexed by ticker.  Used only for the
        tiebreaker; must share the same index.
    config : ModelConfig
        Supplies ``weight_momentum``, ``weight_volatility``,
        ``weight_correlation``, and ``weight_trend``.

    Returns
    -------
    pd.Series
        TRank scores indexed by ticker.  NaN where any input is NaN.
    """
    n = len(raw_momentum)
    if n == 0:
        return pd.Series(dtype=float)

    trank = (
        config.weight_momentum    * momentum_ranks
        + config.weight_volatility  * volatility_ranks
        + config.weight_correlation * correlation_ranks
        - config.weight_trend       * trend_values
        + raw_momentum / n
    )
    trank.name = "TRank"
    return trank


def select_top_n(tranks: pd.Series, n: int) -> List[str]:
    """
    Return the tickers of the top-N assets by TRank score.

    Tie-inclusion rule (Keller 2012): if multiple assets share the same TRank
    at the N-th position, ALL of them are included.  The caller is responsible
    for adjusting weights proportionally when more than N assets are returned.

    Assets with NaN TRank are excluded before selection.

    Parameters
    ----------
    tranks : pd.Series
        TRank scores indexed by ticker.
    n : int
        Target number of assets to select (e.g. 6 for main sleeve).

    Returns
    -------
    List[str]
        Tickers of the selected assets, sorted descending by TRank.
        May contain more than *n* elements if ties straddle the cut-off.

    Raises
    ------
    ValueError
        If *tranks* is empty or all values are NaN.
    """
    valid = tranks.dropna()
    if valid.empty:
        raise ValueError("Cannot select top-N: all TRank values are NaN.")

    # Sort descending: best (highest) TRank first.
    sorted_tranks = valid.sort_values(ascending=False)

    if n >= len(sorted_tranks):
        # Fewer assets than requested — return all.
        return sorted_tranks.index.tolist()

    # The cut-off score is the N-th highest value.  Include all assets that
    # tie at or above that score (Keller tie-inclusion convention).
    cutoff_score = sorted_tranks.iloc[n - 1]
    selected = sorted_tranks[sorted_tranks >= cutoff_score]
    return selected.index.tolist()


def compute_monthly_rankings(
    factor_data: Dict[str, pd.DataFrame],
    config: ModelConfig,
    sleeve_tickers: List[str],
) -> pd.DataFrame:
    """
    Compute TRank and select the top-N assets for every month-end date.

    For each month-end date present in the factor data, this function:

    1. Extracts the month-end values of M, V, C, T for all sleeve assets.
    2. Computes ordinal ranks for M (ascending), V (descending), C (descending).
    3. Applies ``compute_trank`` to produce TRank scores.
    4. Calls ``select_top_n`` with the sleeve's configured ``top_n``.

    The returned DataFrame has one row per month-end date and a fixed set of
    columns regardless of how many assets were selected.  Columns beyond the
    actual selection count are filled with ``None``.

    Parameters
    ----------
    factor_data : Dict[str, pd.DataFrame]
        Mapping with keys ``"momentum"``, ``"volatility"``, ``"correlation"``,
        ``"trend"``, each a wide DataFrame (dates × tickers).
    config : ModelConfig
        Model configuration.  ``main_sleeve_top_n`` or ``hedging_sleeve_top_n``
        is inferred from ``len(sleeve_tickers)``.
    sleeve_tickers : List[str]
        Tickers belonging to this sleeve (main or hedging).

    Returns
    -------
    pd.DataFrame
        Index: month-end dates.
        Columns: ``["rank_1", "rank_2", …, "rank_N", "trank_<ticker>", …]``
        where ``N = top_n`` from config and the ``trank_`` columns hold the
        raw TRank scores for diagnostics.
    """
    # Determine target selection count from sleeve size.
    n_sleeve = len(sleeve_tickers)
    if n_sleeve == len(config.main_sleeve_top_n.__class__.__mro__):  # type guard
        top_n = config.main_sleeve_top_n
    top_n = (
        config.main_sleeve_top_n
        if n_sleeve > config.hedging_sleeve_top_n
        else config.hedging_sleeve_top_n
    )

    mom_df  = factor_data["momentum"].loc[:, sleeve_tickers]
    vol_df  = factor_data["volatility"].loc[:, sleeve_tickers]
    corr_df = factor_data["correlation"].loc[:, sleeve_tickers]
    trend_df = factor_data["trend"].loc[:, sleeve_tickers]

    # Month-end dates are the last trading day of each calendar month.
    month_ends = mom_df.groupby(mom_df.index.to_period("M")).apply(
        lambda g: g.index.max()
    ).values

    records = []
    for date in month_ends:
        m_row    = mom_df.loc[date]
        v_row    = vol_df.loc[date]
        c_row    = corr_df.loc[date]
        t_row    = trend_df.loc[date]

        m_ranks = rank_assets(m_row, ascending=True)
        v_ranks = rank_assets(v_row, ascending=False)
        c_ranks = rank_assets(c_row, ascending=False)

        tranks = compute_trank(m_ranks, v_ranks, c_ranks, t_row, m_row, config)

        try:
            selected = select_top_n(tranks, top_n)
        except ValueError:
            logger.warning("No valid TRank values on %s — skipping.", date)
            selected = []

        row: Dict = {"date": date}
        for i, ticker in enumerate(selected, start=1):
            row[f"rank_{i}"] = ticker
        for ticker in sleeve_tickers:
            row[f"trank_{ticker}"] = tranks.get(ticker, float("nan"))
        records.append(row)

    result = pd.DataFrame(records).set_index("date")
    return result
