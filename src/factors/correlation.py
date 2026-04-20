"""
Average Relative Correlation factor (C) for AMAAM.

Computes each asset's mean pairwise Pearson correlation with all other assets
in the same sleeve over a trailing 84-trading-day window. Assets with lower
average correlation receive higher TRank scores, promoting portfolio
diversification. Correlations are computed independently within each sleeve;
there is no cross-sleeve correlation. See Section 3.4 of the specification.
"""

import logging
from typing import Dict, List

import pandas as pd

logger = logging.getLogger(__name__)


def compute_average_relative_correlation(
    returns_dict: Dict[str, pd.Series],
    tickers: List[str],
    lookback: int,
    date: pd.Timestamp,
) -> Dict[str, float]:
    """
    Compute each asset's average pairwise correlation as of a single date.

    ``C_i = (1 / (N−1)) · Σ_{j≠i} corr(r_i, r_j)``

    where the sum ranges over all other assets in *tickers* and correlations
    are Pearson over the trailing *lookback* trading days ending on *date*.

    Parameters
    ----------
    returns_dict : Dict[str, pd.Series]
        Mapping of ticker → daily return Series.
    tickers : List[str]
        Ordered list of tickers that form one sleeve.  Correlations are
        computed only within this set (no cross-sleeve correlations).
    lookback : int
        Trailing window in trading days (spec default: 84).
    date : pd.Timestamp
        Evaluation date.  The correlation window is
        ``(date − lookback + 1, date]`` inclusive.

    Returns
    -------
    Dict[str, float]
        Mapping of ticker → average pairwise correlation.  NaN if the
        ticker has fewer than *lookback* prior returns.
    """
    # Build returns DataFrame for the lookback window ending on date.
    window_returns = pd.DataFrame(
        {t: returns_dict[t] for t in tickers if t in returns_dict}
    )
    window_returns = window_returns.loc[:date].iloc[-lookback:]

    if len(window_returns) < lookback:
        logger.debug(
            "Insufficient history for correlation at %s: need %d rows, got %d.",
            date.date(), lookback, len(window_returns),
        )
        return {t: float("nan") for t in tickers}

    corr_matrix = window_returns.corr(method="pearson")
    n = len(tickers)

    result: Dict[str, float] = {}
    for t in tickers:
        if t not in corr_matrix.columns:
            result[t] = float("nan")
            continue
        # Average of off-diagonal elements in row t.
        row_sum = corr_matrix.loc[t].sum()
        result[t] = (row_sum - 1.0) / (n - 1)  # subtract self-correlation

    return result


def compute_correlation_all_assets(
    data_dict: Dict[str, pd.DataFrame],
    tickers: List[str],
    lookback: int,
) -> pd.DataFrame:
    """
    Compute average relative correlation for all assets across all dates.

    Uses ``pd.DataFrame.rolling().corr()`` to build the full trailing
    correlation matrix at every date in one vectorised pass, then derives
    per-asset average pairwise correlations from the row sums.

    ``C_i(t) = (Σ_j corr(r_i, r_j) − 1) / (N − 1)``

    where the sum over j includes the self-correlation (1.0), which is then
    subtracted.  Only assets in *tickers* enter the correlation calculation,
    enforcing the sleeve-independence rule from Section 3.4.

    Parameters
    ----------
    data_dict : Dict[str, pd.DataFrame]
        Mapping of ticker → OHLCV DataFrame.
    tickers : List[str]
        Ordered sleeve ticker list.
    lookback : int
        Rolling window in trading days (spec default: 84).

    Returns
    -------
    pd.DataFrame
        DataFrame with dates as index and tickers as columns.  The first
        ``lookback − 1`` rows are NaN (insufficient history).
    """
    available = [t for t in tickers if t in data_dict]
    if not available:
        return pd.DataFrame()

    # Simple returns are used for Pearson correlation; the choice between log
    # and simple returns is immaterial for correlation  because the scaling
    # cancels out in the Pearson formula.
    returns_df = pd.DataFrame(
        {t: data_dict[t]["Close"].pct_change(fill_method=None) for t in available}
    )

    # rolling().corr() returns MultiIndex (date, ticker) × tickers.
    # Summing across columns gives total correlation for each (date, ticker)
    # including the self-correlation diagonal which equals 1.0.
    rolling_corr = returns_df.rolling(window=lookback, min_periods=lookback).corr()
    # min_count=n ensures the sum returns NaN (not 0) when the rolling window
    # hasn't accumulated enough data yet.  Without min_count, pandas' default
    # skipna=True would turn a fully-NaN row into 0, producing spurious -0.5
    # values ((0 − 1) / (N − 1)) in the pre-warmup period.
    n = len(available)
    row_sums = rolling_corr.sum(axis=1, min_count=n)

    # Subtract diagonal (1.0) and normalise by (N-1) to get average pairwise.
    avg_corr_stacked = (row_sums - 1.0) / (n - 1)

    # Pivot from MultiIndex (date, ticker) Series to date × ticker DataFrame.
    result = avg_corr_stacked.unstack(level=1)

    # Reorder columns to match the tickers argument for consistency.
    result = result.reindex(columns=available)
    return result
