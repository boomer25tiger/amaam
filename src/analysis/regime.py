"""
Regime-specific performance analysis for AMAAM.

Defines the six named market regimes from Section 7.1 of the specification
(2008 GFC, 2011 Euro crisis, 2015–16 commodity crash, 2018 vol spike, 2020
COVID, 2022 rate shock) and computes the full set of performance metrics for
each regime period. Used to evaluate how the model and benchmarks behave during
distinct macroeconomic episodes. See Section 9.17 of the specification.
"""

from typing import Dict, Tuple

import pandas as pd

from src.backtest.metrics import compute_all_metrics


def define_regimes() -> Dict[str, Tuple[str, str]]:
    """Return the six named market stress regimes from Section 7.1.

    Returns
    -------
    Dict[str, Tuple[str, str]]
        Regime label → (start_date, end_date) in YYYY-MM-DD format.
        Dates chosen to capture the full drawdown-to-recovery arc of each episode.
    """
    return {
        "2008 GFC":          ("2008-01-01", "2009-06-30"),
        "2011 Euro Crisis":  ("2011-05-01", "2012-01-31"),
        "2015-16 Commodity": ("2015-06-01", "2016-02-29"),
        "2018 Vol Spike":    ("2018-09-01", "2019-01-31"),
        "2020 COVID":        ("2020-02-01", "2020-06-30"),
        "2022 Rate Shock":   ("2022-01-01", "2022-12-31"),
    }


def compute_regime_metrics(
    returns_dict: Dict[str, pd.Series],
    regimes: Dict[str, Tuple[str, str]],
) -> pd.DataFrame:
    """Compute performance metrics for each strategy across each named regime.

    For each (strategy, regime) pair, the return series is sliced to the regime
    window and the full metric set is computed.  Short slices (< 2 observations)
    are skipped.

    Parameters
    ----------
    returns_dict : Dict[str, pd.Series]
        Strategy label → monthly return series (full period).
    regimes : Dict[str, Tuple[str, str]]
        Regime label → (start_date, end_date) as returned by define_regimes().

    Returns
    -------
    pd.DataFrame
        Flat structure with MultiIndex (Strategy, Regime) and columns
        [Total Return, Annualized Return, Max Drawdown, Sharpe Ratio, Worst Month].
        Rows where the slice was too short are omitted.
    """
    records = []
    for strategy, rets in returns_dict.items():
        for regime_name, (start, end) in regimes.items():
            slice_ = rets.loc[start:end].dropna()
            if len(slice_) < 2:
                continue
            m = compute_all_metrics(slice_)
            records.append({
                "Strategy":          strategy,
                "Regime":            regime_name,
                "Total Return":      m.get("Total Return",       float("nan")),
                "Annualized Return": m.get("Annualized Return",  float("nan")),
                "Max Drawdown":      m.get("Max Drawdown",       float("nan")),
                "Sharpe Ratio":      m.get("Sharpe Ratio",       float("nan")),
                "Worst Month":       m.get("Worst Month",        float("nan")),
            })

    df = pd.DataFrame(records)
    if not df.empty:
        df = df.set_index(["Strategy", "Regime"])
    return df
