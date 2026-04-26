"""
Portfolio-level volatility targeting for AMAAM.

Scales the monthly allocation so realized portfolio volatility stays near a
target level. When realized vol exceeds the target, all weights are scaled down
and the residual is parked in the cash proxy (SHY). When vol is below target
the scale is capped at ``vol_target_max_leverage`` (default 1.0 = no leverage).

Reference: Barroso & Santa-Clara (2015) "Momentum Has Its Moments",
Journal of Financial Economics 116(1); Daniel & Moskowitz (2016) "Momentum
Crashes", Journal of Financial Economics 122(2).
"""

import logging
from typing import Dict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# 252 trading days per year — standard equity market convention.
_ANNUALIZATION_FACTOR: int = 252


def compute_realized_portfolio_vol(
    alloc: Dict[str, float],
    closes: pd.DataFrame,
    signal_date: pd.Timestamp,
    lookback: int,
) -> float:
    """
    Estimate annualised portfolio volatility over the past *lookback* trading days.

    Notes
    -----
    Applies the proposed allocation weights to historical close-to-close returns,
    giving the volatility the portfolio *would have had* under the new weights.
    Returns ``nan`` when fewer than ``lookback // 4`` observations are available
    so the caller can skip scaling rather than act on a noisy estimate.
    """
    price_window = closes.loc[:signal_date].tail(lookback + 1)
    rets = price_window.pct_change(fill_method=None).dropna(how="all")

    min_obs = max(lookback // 4, 5)
    if len(rets) < min_obs:
        logger.debug(
            "vol_targeting: only %d obs before %s (need %d) — skipping scale.",
            len(rets), signal_date.date(), min_obs,
        )
        return float("nan")

    port_rets = pd.Series(0.0, index=rets.index)
    for t, w in alloc.items():
        if t in rets.columns and w > 0.0:
            port_rets = port_rets + w * rets[t].fillna(0.0)

    return float(port_rets.std() * np.sqrt(_ANNUALIZATION_FACTOR))


def apply_vol_targeting(
    alloc: Dict[str, float],
    closes: pd.DataFrame,
    signal_date: pd.Timestamp,
    vol_target: float,
    lookback: int,
    max_leverage: float,
    cash_proxy: str,
) -> Dict[str, float]:
    """
    Scale portfolio weights so expected portfolio volatility ≈ *vol_target*.

    Notes
    -----
    ``scale = min(vol_target / realized_vol, max_leverage)``.  When scale < 1
    the residual weight ``(1 − scale)`` is added to *cash_proxy* (SHY).  When
    scale ≥ 1 and ``max_leverage = 1.0`` the allocation is returned unchanged —
    no leverage is taken in calm markets.  If realized vol cannot be estimated
    the allocation is returned unchanged.
    """
    port_vol = compute_realized_portfolio_vol(alloc, closes, signal_date, lookback)

    if np.isnan(port_vol) or port_vol <= 0.0:
        return alloc

    scale = min(vol_target / port_vol, max_leverage)

    # Skip trivial adjustments (within floating-point noise of 1.0).
    if abs(scale - 1.0) < 1e-6:
        return alloc

    logger.debug(
        "vol_targeting: date=%s port_vol=%.4f target=%.4f scale=%.4f",
        signal_date.date(), port_vol, vol_target, scale,
    )

    scaled: Dict[str, float] = {t: w * scale for t, w in alloc.items()}

    # Residual (when scale < 1) parks in cash — never negative.
    residual = 1.0 - scale
    if residual > 1e-8:
        scaled[cash_proxy] = scaled.get(cash_proxy, 0.0) + residual

    return scaled
