"""
Correlation factor (C) for AMAAM.

Two estimation methods are supported via ``ModelConfig.correlation_method``:

**"pairwise"** (original FAA spec):
    Each asset's mean Pearson correlation with every other asset in the same
    sleeve over a trailing window.  Promotes intra-sleeve diversification but
    suffers from (a) statistical noise on short windows, (b) poor discriminating
    power within equity-heavy sleeves where all assets share a large common
    S&P 500 factor, and (c) correlation-to-1 collapse during market stress.

**"market"** (revised, academically motivated):
    Each asset's rolling Pearson correlation with SPY (the market proxy).
    Lower market correlation → higher TRank score.

    Academic basis:
    * **Keller & Butler (2014) EAA** — replaced FAA pairwise correlation with
      correlation to the equal-weight portfolio, arguing that pairwise scores
      in equity universes primarily reflect shared market beta rather than
      genuine sleeve-level diversification.
    * **Sharpe (1964) CAPM** — marginal portfolio risk contribution is captured
      by market beta, not average pairwise covariance, making the market the
      correct reference for a diversification-oriented ranking factor.
    * **Ang & Chen (2002)** — pairwise equity correlations spike toward 1.0
      during downturns; market beta is comparatively more stable across regimes.
    * **Frazzini & Pedersen (2014) BAB** — low-beta assets deliver higher
      risk-adjusted returns across 20 markets, providing a return-predictive
      rationale beyond pure diversification.

    Practical advantage: SPY is available for the full backtest window, the
    estimate is more stable than pairwise (one correlation vs N−1), and it
    genuinely discriminates between defensive (XLU, XLP) and cyclical (XLK,
    XLY) assets within the equity-dominated main sleeve.

See Section 3.4 of the specification.
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


def compute_market_correlation(
    data_dict: Dict[str, pd.DataFrame],
    tickers: List[str],
    lookback: int,
    market_ticker: str = "SPY",
) -> pd.DataFrame:
    """
    Compute each asset's rolling Pearson correlation with the market proxy (SPY).

    Lower market correlation → higher TRank score (same ranking direction as
    pairwise correlation; assets that co-move less with SPY are preferred as
    diversifiers).

    This estimator addresses three structural weaknesses of the pairwise method
    within equity-heavy sleeves:

    1. **Discriminating power**: SPY correlation genuinely separates defensive
       sectors (XLU, XLP, GLD: β ≈ 0.3–0.5) from cyclicals (XLK, XLY: β ≈ 1.1)
       where pairwise correlation ranks them nearly identically (all ≈ 0.80).
    2. **Statistical efficiency**: one correlation per asset vs N−1 pairwise
       estimates, halving the estimation noise for a given lookback window.
    3. **Crisis stability**: market beta is more stable across regimes than
       pairwise correlations, which collapse toward 1.0 during selloffs and
       lose all discriminating power (Ang & Chen 2002).

    Parameters
    ----------
    data_dict : Dict[str, pd.DataFrame]
        Mapping of ticker → OHLCV DataFrame.  Must include *market_ticker*.
    tickers : List[str]
        Sleeve tickers to compute correlation for.
    lookback : int
        Rolling window in trading days (spec default: 84).
    market_ticker : str
        Ticker used as the market proxy.  Defaults to ``"SPY"``.

    Returns
    -------
    pd.DataFrame
        Dates × tickers DataFrame of rolling market correlations.
        First ``lookback − 1`` rows are NaN.  Range [−1, 1].

    Raises
    ------
    KeyError
        If *market_ticker* is not present in *data_dict*.
    """
    if market_ticker not in data_dict:
        raise KeyError(
            f"Market proxy '{market_ticker}' not found in data_dict. "
            "Ensure it is downloaded alongside the sleeve tickers."
        )

    available = [t for t in tickers if t in data_dict]
    if not available:
        return pd.DataFrame()

    # Build a returns DataFrame that includes both sleeve assets and the market.
    all_tickers = available + ([market_ticker] if market_ticker not in available else [])
    returns_df = pd.DataFrame(
        {t: data_dict[t]["Close"].pct_change(fill_method=None) for t in all_tickers}
    )

    market_rets = returns_df[market_ticker]

    # Compute rolling Pearson correlation of each sleeve asset with SPY.
    # Using a loop rather than rolling().corr() avoids the MultiIndex overhead
    # and is clearer for a single reference-series calculation.
    result = pd.DataFrame(index=returns_df.index, columns=available, dtype=float)
    for t in available:
        result[t] = (
            returns_df[t]
            .rolling(window=lookback, min_periods=lookback)
            .corr(market_rets)
        )

    return result
