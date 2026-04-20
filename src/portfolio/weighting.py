"""
Weighting schemes for AMAAM portfolio construction.

Provides two weighting methods that operate after asset selection and the
momentum filter: equal-weight (1/N) and inverse-volatility weight (weight
proportional to 1/V, normalized to sum to 1.0). Both schemes are applied
within each sleeve independently before the sleeves are combined. See Section
3.8 and Section 9.12 of the specification.
"""

import logging
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Recognised scheme identifiers — kept here so allocation.py can reference
# them without a hard-coded string comparison.
SCHEME_EQUAL = "equal"
SCHEME_INVERSE_VOL = "inverse_volatility"


def equal_weight(tickers: List[str]) -> Dict[str, float]:
    """
    Assign each ticker an equal weight of 1/N.

    The returned weights sum to exactly 1.0.  The caller is responsible for
    scaling to the sleeve's actual portfolio allocation (e.g. multiplying by
    ``n_active / n_selected`` when some positions have been redirected).

    Parameters
    ----------
    tickers : List[str]
        Tickers to be weighted.  Must be non-empty.

    Returns
    -------
    Dict[str, float]
        ``{ticker: 1/N}`` for each ticker in *tickers*.

    Raises
    ------
    ValueError
        If *tickers* is empty.
    """
    if not tickers:
        raise ValueError("equal_weight requires at least one ticker.")
    w = 1.0 / len(tickers)
    return {t: w for t in tickers}


def inverse_volatility_weight(
    tickers: List[str],
    volatility_values: pd.Series,
) -> Dict[str, float]:
    """
    Assign each ticker a weight proportional to the inverse of its volatility.

    .. math::

        w_i = \\frac{1/V_i}{\\sum_j 1/V_j}

    Lower volatility assets receive higher weights, rewarding risk efficiency.
    The returned weights sum to exactly 1.0.

    Parameters
    ----------
    tickers : List[str]
        Tickers to be weighted.  Must be non-empty.
    volatility_values : pd.Series
        Annualised volatility for each ticker (output of the EWMA volatility
        model).  Values must be strictly positive.

    Returns
    -------
    Dict[str, float]
        Normalised inverse-volatility weights summing to 1.0.

    Raises
    ------
    ValueError
        If *tickers* is empty or any ticker has a non-positive volatility.
    """
    if not tickers:
        raise ValueError("inverse_volatility_weight requires at least one ticker.")

    inv_vols: Dict[str, float] = {}
    for t in tickers:
        v = float(volatility_values[t])
        if v <= 0.0:
            raise ValueError(
                f"Volatility for {t} is {v}; must be strictly positive for "
                "inverse-volatility weighting."
            )
        inv_vols[t] = 1.0 / v

    total_inv = sum(inv_vols.values())
    return {t: inv_v / total_inv for t, inv_v in inv_vols.items()}


def apply_weighting(
    tickers: List[str],
    scheme: str,
    volatility_values: Optional[pd.Series] = None,
) -> Dict[str, float]:
    """
    Dispatch to the appropriate weighting function based on *scheme*.

    This is the single entry-point used by the allocation module so that
    changing the weighting scheme in ``ModelConfig`` propagates automatically
    without touching allocation logic.

    Parameters
    ----------
    tickers : List[str]
        Active (momentum-positive) tickers to be weighted.
    scheme : str
        ``"equal"`` or ``"inverse_volatility"``.
    volatility_values : pd.Series, optional
        Required when *scheme* is ``"inverse_volatility"``.

    Returns
    -------
    Dict[str, float]
        Unit weights (sum = 1.0) for the tickers.  Caller scales by the
        sleeve's total portfolio weight.

    Raises
    ------
    ValueError
        If *scheme* is unrecognised, or if ``"inverse_volatility"`` is
        requested but *volatility_values* is ``None``.
    """
    if scheme == SCHEME_EQUAL:
        return equal_weight(tickers)

    if scheme == SCHEME_INVERSE_VOL:
        if volatility_values is None:
            raise ValueError(
                "volatility_values must be provided for 'inverse_volatility' weighting."
            )
        return inverse_volatility_weight(tickers, volatility_values)

    raise ValueError(
        f"Unknown weighting scheme '{scheme}'. "
        f"Valid options: '{SCHEME_EQUAL}', '{SCHEME_INVERSE_VOL}'."
    )
