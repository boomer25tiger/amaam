"""
Cross-asset correlation scoring for AMAAM.

Provides multiple correlation estimation methods (pairwise, portfolio,
market-relative, EWM, stress-conditioned, cross-sleeve). Each function
returns a per-asset correlation score for use as the C component of TRank.
The method is selected via ModelConfig.correlation_method; see that field
and Section 3.4 of the specification for per-method academic rationale.
"""

import logging
from typing import Dict, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_average_relative_correlation(
    returns_dict: Dict[str, pd.Series],
    tickers: List[str],
    lookback: int,
    date: pd.Timestamp,
) -> Dict[str, float]:
    """
    Return each asset's average pairwise Pearson correlation as of a single date.

    Used for one-shot evaluation at a specific date; prefer
    ``compute_correlation_all_assets`` for full time-series construction.

    Notes
    -----
    The self-correlation diagonal (always 1.0) is subtracted before dividing by
    ``N − 1`` so the result is strictly the mean of the ``N*(N-1)/2`` off-diagonal pairs.
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
    Build the full time-series of average pairwise correlations for an entire sleeve in one vectorised pass.

    Notes
    -----
    ``rolling().corr()`` returns a MultiIndex ``(date, ticker) × tickers`` DataFrame.
    Row sums include the self-correlation diagonal (1.0), which is subtracted before
    normalising by ``N − 1``.  ``min_count=n`` on the row sum is required so pre-warmup
    rows return NaN rather than the spurious ``(0 − 1) / (N − 1)`` value that would
    result from pandas' default ``skipna=True`` treating a fully-NaN row as 0.
    """
    available = [t for t in tickers if t in data_dict]
    if not available:
        return pd.DataFrame()

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


def compute_blended_correlation_all_assets(
    data_dict: Dict[str, pd.DataFrame],
    tickers: List[str],
    lookbacks: List[int],
) -> pd.DataFrame:
    """
    Average pairwise correlation across multiple lookback windows to reduce horizon sensitivity.

    Notes
    -----
    A cell is NaN until every component window has exited warm-up — partial blends are
    not used, matching the ``compute_trend_ensemble`` convention.
    """
    frames = [
        compute_correlation_all_assets(data_dict, tickers, lb)
        for lb in lookbacks
    ]
    # Stack on a new axis and take the mean — skipna=False so the blend is
    # NaN until every component window has exited warm-up (same convention
    # as compute_trend_ensemble).
    stacked = np.stack([f.values for f in frames], axis=0)
    blended_values = np.nanmean(stacked, axis=0)   # use nanmean: allow partial warm-up
    # But actually use skipna=False equivalent: only valid when ALL windows valid
    all_valid = np.all(~np.isnan(stacked), axis=0)
    blended_values[~all_valid] = np.nan
    result = pd.DataFrame(blended_values, index=frames[0].index, columns=frames[0].columns)
    return result


def compute_ewm_correlation_all_assets(
    data_dict: Dict[str, pd.DataFrame],
    tickers: List[str],
    span: int,
) -> pd.DataFrame:
    """
    Compute average pairwise correlation using exponentially weighted observations, making the estimate more responsive to recent regime shifts than a same-span rolling window.

    Notes
    -----
    Uses the same MultiIndex ``(date, ticker) × tickers`` output as ``rolling().corr()``,
    so the identical self-correlation subtraction and ``min_count`` pattern applies.
    ``min_periods`` is set to ``span // 2`` so the series starts earlier than a
    rectangular window of the same span.
    """
    available = [t for t in tickers if t in data_dict]
    if not available:
        return pd.DataFrame()

    returns_df = pd.DataFrame(
        {t: data_dict[t]["Close"].pct_change(fill_method=None) for t in available}
    )

    min_periods = max(span // 2, 2)

    ewm_corr = returns_df.ewm(span=span, min_periods=min_periods).corr()

    n = len(available)
    row_sums = ewm_corr.sum(axis=1, min_count=n)

    avg_corr_stacked = (row_sums - 1.0) / (n - 1)

    # Unstack MultiIndex → date × ticker DataFrame.
    result = avg_corr_stacked.unstack(level=1)
    result = result.reindex(columns=available)
    return result


def compute_portfolio_correlation(
    data_dict: Dict[str, pd.DataFrame],
    tickers: List[str],
    lookback: int,
) -> pd.DataFrame:
    """
    Correlate each asset with the equal-weight return stream of its sleeve peers (excluding itself).

    This is the Keller & Gilman (2012) interpretation: a single well-defined Pearson
    correlation rather than an average of N−1 non-linear pairwise coefficients, which
    would require Fisher z-transforms to be mathematically sound.
    """
    available = [t for t in tickers if t in data_dict]
    if not available:
        return pd.DataFrame()

    returns_df = pd.DataFrame(
        {t: data_dict[t]["Close"].pct_change(fill_method=None) for t in available}
    )

    result = pd.DataFrame(index=returns_df.index, columns=available, dtype=float)
    for t in available:
        # Equal-weight portfolio of all OTHER sleeve members for this asset.
        peers = [c for c in available if c != t]
        if not peers:
            result[t] = float("nan")
            continue
        peer_portfolio = returns_df[peers].mean(axis=1)
        result[t] = (
            returns_df[t]
            .rolling(window=lookback, min_periods=lookback)
            .corr(peer_portfolio)
        )

    return result


def compute_portfolio_correlation_all_assets(
    data_dict: Dict[str, pd.DataFrame],
    tickers: List[str],
    lookback: int,
) -> pd.DataFrame:
    """
    Correlate each asset with the equal-weight portfolio of all N sleeve members (including itself), the strict MCPV formulation.

    Including the asset in its own reference portfolio (vs. peers-only in
    ``compute_portfolio_correlation``) makes this the theoretically correct marginal
    contribution to portfolio variance estimator for equal-weight portfolios
    (Merton 1972, Keller & Butler 2014).
    """
    available = [t for t in tickers if t in data_dict]
    if not available:
        return pd.DataFrame()

    # Build returns DataFrame aligned on a common date index.
    closes = pd.DataFrame(
        {t: data_dict[t]["Close"] for t in available}
    ).ffill()
    returns = closes.pct_change(fill_method=None)

    # Equal-weight portfolio return (all N assets, including each asset itself).
    # Using all N rather than N−1 peers makes this the strict MCPV estimator.
    port_returns = returns.mean(axis=1)

    result = pd.DataFrame(index=returns.index, columns=available, dtype=float)
    for t in available:
        result[t] = (
            returns[t]
            .rolling(window=lookback, min_periods=lookback)
            .corr(port_returns)
        )

    return result


def compute_market_correlation(
    data_dict: Dict[str, pd.DataFrame],
    tickers: List[str],
    lookback: int,
    market_ticker: str = "SPY",
) -> pd.DataFrame:
    """
    Compute each asset's rolling Pearson correlation with the market proxy (SPY by default).

    Assets with lower market correlation rank higher as diversifiers; ranking
    direction is identical to the pairwise method.
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


def compute_stress_correlation_all_assets(
    data_dict: Dict[str, pd.DataFrame],
    tickers: List[str],
    lookback: int,
    stress_method: str = "vol",
    vol_multiplier: float = 1.5,
    market_ticker: str = "SPY",
) -> pd.DataFrame:
    """
    Compute average pairwise correlation using only stress-regime observations, capturing crisis co-movement that standard rolling correlation masks (Ang & Chen 2002).

    Notes
    -----
    Two regime filters are supported: ``"vol"`` flags days where SPY 21-day vol exceeds
    ``vol_multiplier × rolling median``; ``"drawdown"`` flags days where SPY is below its
    200-day SMA (Faber 2007).  When fewer than ``lookback // 4`` stress days fall inside
    the lookback window the function falls back to standard rolling correlation rather than
    returning NaN — this most commonly affects the ``"drawdown"`` method in prolonged
    bull markets.
    """
    if market_ticker not in data_dict:
        raise KeyError(
            f"Market proxy '{market_ticker}' not found in data_dict. "
            "Ensure it is downloaded alongside the sleeve tickers."
        )

    available = [t for t in tickers if t in data_dict]
    if not available:
        return pd.DataFrame()

    # Build returns DataFrame for sleeve assets.
    returns_df = pd.DataFrame(
        {t: data_dict[t]["Close"].pct_change(fill_method=None) for t in available}
    )

    spy_close = data_dict[market_ticker]["Close"]
    spy_rets = spy_close.pct_change(fill_method=None)

    # --- Build the stress indicator Series (1 = stress day, 0 = normal) ------
    if stress_method == "vol":
        # 21-day realized SPY vol (annualised not needed; ranking is relative).
        spy_vol_21 = spy_rets.rolling(window=21, min_periods=21).std()
        # Rolling median of that vol over the lookback window.
        spy_vol_median = spy_vol_21.rolling(window=lookback, min_periods=lookback // 2).median()
        # Stress if current 21-day vol exceeds vol_multiplier × rolling median.
        stress_mask = (spy_vol_21 > vol_multiplier * spy_vol_median).astype(float)
        stress_mask[spy_vol_21.isna() | spy_vol_median.isna()] = float("nan")
    else:  # "drawdown"
        # Faber (2007) bear-market filter: below 200-day SMA.
        sma200 = spy_close.rolling(window=200, min_periods=200).mean()
        stress_mask = (spy_close < sma200).astype(float)
        stress_mask[sma200.isna()] = float("nan")

    # Align all series to the common index of the returns DataFrame.
    common_idx = returns_df.index
    stress_mask = stress_mask.reindex(common_idx)

    # Pre-compute the full-window standard pairwise correlation as a fallback
    # so we don't leave NaN gaps on low-stress periods.
    fallback_corr = compute_correlation_all_assets(data_dict, tickers, lookback)
    fallback_corr = fallback_corr.reindex(common_idx)

    n = len(available)
    min_stress_obs = lookback // 4  # minimum stress days required for a valid estimate

    result = pd.DataFrame(index=common_idx, columns=available, dtype=float)

    # Iterate over dates from the first possible valid date.
    # Using a loop because the stress mask changes dynamically and differs by date.
    valid_dates = common_idx[lookback - 1:]
    for date in valid_dates:
        # Extract the lookback window ending on this date.
        loc_end = common_idx.get_loc(date)
        loc_start = max(0, loc_end - lookback + 1)
        window_idx = common_idx[loc_start: loc_end + 1]

        s_mask = stress_mask.loc[window_idx]
        # Stress obs: rows where mask == 1 (non-NaN and True).
        stress_days = s_mask[s_mask == 1.0].index
        n_stress = len(stress_days)

        if n_stress < min_stress_obs:
            # Insufficient stress observations — fall back to standard rolling corr.
            if date in fallback_corr.index and not fallback_corr.loc[date].isna().all():
                result.loc[date] = fallback_corr.loc[date, available]
            continue

        # Compute pairwise correlation on stress days only.
        stress_rets = returns_df.loc[stress_days]
        if stress_rets.shape[0] < 2:
            if date in fallback_corr.index:
                result.loc[date] = fallback_corr.loc[date, available]
            continue

        corr_matrix = stress_rets.corr(method="pearson")

        for t in available:
            if t not in corr_matrix.columns or corr_matrix[t].isna().all():
                result.at[date, t] = float("nan")
                continue
            row_sum = corr_matrix.loc[t].sum()
            result.at[date, t] = (row_sum - 1.0) / (n - 1)

    return result


def compute_stress_blend_correlation_all_assets(
    data_dict: Dict[str, pd.DataFrame],
    tickers: List[str],
    lookback: int,
    vol_multiplier: float = 1.5,
    market_ticker: str = "SPY",
) -> pd.DataFrame:
    """
    Blend standard pairwise rolling correlation with vol-conditional stress correlation 50/50.

    Pure stress correlation can be noisy due to few observations in low-vol regimes;
    blending with the unconditional estimate stabilises the signal without requiring
    a data-fitted blend weight.
    """
    pairwise = compute_correlation_all_assets(data_dict, tickers, lookback)
    stress_vol = compute_stress_correlation_all_assets(
        data_dict, tickers, lookback,
        stress_method="vol",
        vol_multiplier=vol_multiplier,
        market_ticker=market_ticker,
    )

    # Align to the same index.
    idx = pairwise.index.union(stress_vol.index)
    pairwise = pairwise.reindex(idx)
    stress_vol = stress_vol.reindex(idx)

    # Stack and take the mean; nanmean so partial NaN is handled gracefully.
    stacked = np.stack([pairwise.values, stress_vol.values], axis=0).astype(float)
    blended = np.nanmean(stacked, axis=0)
    # Keep fully NaN rows as NaN (not zero).
    all_nan = np.all(np.isnan(stacked), axis=0)
    blended[all_nan] = np.nan

    result = pd.DataFrame(blended, index=idx, columns=pairwise.columns)
    return result


def compute_cross_sleeve_correlation(
    data_dict: Dict[str, pd.DataFrame],
    main_tickers: List[str],
    hedge_tickers: List[str],
    lookback: int,
) -> pd.DataFrame:
    """
    Measure how much each main-sleeve asset co-moves with the hedging sleeve, identifying main-sleeve assets that would create redundant exposure alongside the hedges.

    Notes
    -----
    Prices are forward-filled before computing returns so gaps in less-liquid
    ETFs do not orphan otherwise valid correlation observations.
    """
    all_tickers = list(set(main_tickers) | set(hedge_tickers))
    available_main = [t for t in main_tickers if t in data_dict]
    available_hedge = [t for t in hedge_tickers if t in data_dict]

    if not available_main or not available_hedge:
        return pd.DataFrame()

    # Align all price series on a common date index via forward-fill so gaps
    # in less-liquid ETFs do not orphan otherwise valid correlation observations.
    closes = pd.DataFrame(
        {t: data_dict[t]["Close"] for t in all_tickers if t in data_dict}
    ).ffill()
    returns = closes.pct_change(fill_method=None)

    result = pd.DataFrame(index=returns.index, columns=available_main, dtype=float)

    for main_t in available_main:
        if main_t not in returns.columns:
            result[main_t] = float("nan")
            continue
        # Average rolling correlation with each hedge asset; then average across
        # all hedges so the score is not driven by correlation with one outlier.
        cross_corrs = []
        for hedge_t in available_hedge:
            if hedge_t not in returns.columns:
                continue
            pair_corr = (
                returns[main_t]
                .rolling(lookback, min_periods=lookback)
                .corr(returns[hedge_t])
            )
            cross_corrs.append(pair_corr)

        if not cross_corrs:
            result[main_t] = float("nan")
            continue

        # Equal-weight average across all hedge pairings for asset main_t.
        result[main_t] = pd.concat(cross_corrs, axis=1).mean(axis=1)

    return result


def compute_market_beta(
    data_dict: Dict[str, pd.DataFrame],
    tickers: List[str],
    lookback: int,
    market_ticker: str = "SPY",
) -> pd.DataFrame:
    """
    Compute each asset's rolling OLS beta against the market proxy, capturing both correlation and relative volatility in a single sensitivity measure.

    Notes
    -----
    Unlike correlation, beta is not bounded to ``[-1, 1]``; it equals
    ``ρ × (σ_asset / σ_market)``, so two assets at the same correlation level can have
    very different betas if they differ in volatility.  This matters for marginal
    portfolio-variance contribution (Frazzini & Pedersen 2014 BAB factor).
    """
    if market_ticker not in data_dict:
        raise KeyError(
            f"Market proxy '{market_ticker}' not found in data_dict. "
            "Ensure it is downloaded alongside the sleeve tickers."
        )

    available = [t for t in tickers if t in data_dict]
    if not available:
        return pd.DataFrame()

    all_tickers = available + ([market_ticker] if market_ticker not in available else [])
    returns_df = pd.DataFrame(
        {t: data_dict[t]["Close"].pct_change(fill_method=None) for t in all_tickers}
    )

    market_rets = returns_df[market_ticker]

    # Rolling Cov(asset, market) / Var(market).
    # pandas rolling().cov() is element-wise when called with another Series.
    rolling_var_market = market_rets.rolling(window=lookback, min_periods=lookback).var()

    result = pd.DataFrame(index=returns_df.index, columns=available, dtype=float)
    for t in available:
        rolling_cov = (
            returns_df[t]
            .rolling(window=lookback, min_periods=lookback)
            .cov(market_rets)
        )
        result[t] = rolling_cov / rolling_var_market

    return result
