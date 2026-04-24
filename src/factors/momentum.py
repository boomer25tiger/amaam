"""
Absolute Momentum factor (M) for AMAAM — spec Section 3.2.

Provides single-lookback and blended-lookback ROC variants. Momentum feeds
both the TRank ranking formula and the binary filter that redirects weight
to the hedging sleeve when an asset's return is negative.
"""

import logging
from typing import Dict, List

import pandas as pd

logger = logging.getLogger(__name__)


def compute_absolute_momentum(prices: pd.Series, lookback: int) -> pd.Series:
    """
    Compute Rate of Change over ``lookback`` trading days: ``(P_t / P_{t-lb}) - 1``.

    Notes
    -----
    Daily values are returned so the caller controls the evaluation date;
    the allocation engine samples at month-end.
    """
    mom = prices / prices.shift(lookback) - 1.0
    mom.name = prices.name
    return mom


def compute_blended_momentum(
    prices: pd.Series,
    lookbacks: List[int],
    skip_days: int = 0,
) -> pd.Series:
    """
    Average ROC across multiple horizons to reduce sensitivity to any single lookback choice.

    Notes
    -----
    ``skip_days=21`` applies the Jegadeesh-Titman one-month skip, offsetting every
    window backward to avoid short-term reversal contamination. When blending across
    N horizons this skip matters less than in single-lookback implementations because
    the reversal effect is already diluted by 1/N, but it is exposed as an option for
    empirical comparison.
    """
    # Each ROC starts skip_days in the past: avoids the short-term reversal
    # that would otherwise contaminate the most-recent-month component.
    rocs = pd.concat(
        [prices.shift(skip_days) / prices.shift(skip_days + lb) - 1.0
         for lb in lookbacks],
        axis=1,
    )
    blended = rocs.mean(axis=1)
    blended.name = prices.name
    return blended


def compute_momentum_all_assets(
    data_dict: Dict[str, pd.DataFrame],
    lookback: int,
) -> pd.DataFrame:
    """
    Apply ``compute_absolute_momentum`` to every ticker in the data dictionary and
    return results as a single aligned DataFrame.
    """
    series = {
        ticker: compute_absolute_momentum(df["Close"], lookback)
        for ticker, df in data_dict.items()
    }
    return pd.DataFrame(series)
