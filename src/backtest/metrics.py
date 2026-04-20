"""
Performance metrics for AMAAM backtesting.

Computes the full set of risk-adjusted performance statistics defined in Section
5.6 of the specification: annualized return, annualized volatility, Sharpe ratio,
Calmar ratio, max drawdown, max drawdown duration, best/worst month and year,
percentage of positive periods, and turnover statistics. Also provides rolling
versions of key metrics (rolling Sharpe, rolling volatility, rolling max drawdown)
and a drawdown series for charting. See Section 9.14 of the specification.
"""

import math
from typing import Dict

import numpy as np
import pandas as pd


# Number of months used to annualise monthly-frequency return series.
_MONTHS_PER_YEAR: int = 12


def compute_drawdown_series(returns: pd.Series) -> pd.Series:
    """
    Compute the peak-to-trough drawdown at every point in a return series.

    At each date, the drawdown is the percentage decline from the highest
    prior equity value to the current equity value.  Values are always ≤ 0.

    Parameters
    ----------
    returns : pd.Series
        Periodic (monthly) returns as decimal fractions.

    Returns
    -------
    pd.Series
        Drawdown at each period, expressed as a decimal fraction (e.g. −0.15
        means the portfolio is 15 % below its previous high-water mark).
        Same index as *returns*.
    """
    equity = (1.0 + returns).cumprod()
    peak = equity.cummax()
    return (equity - peak) / peak


def compute_all_metrics(
    returns: pd.Series,
    risk_free_rate: float = 0.02,
    turnover: pd.Series | None = None,
    periods_per_year: float | None = None,
) -> Dict[str, float]:
    """
    Compute the full Section 5.6 performance metric table for a return series.

    Annualisation scales to the supplied *periods_per_year*.  When omitted the
    value is inferred from the date index so that monthly (12), bi-weekly (~25),
    and any other frequency all produce correctly scaled Sharpe / vol numbers.

    Parameters
    ----------
    returns : pd.Series
        Periodic portfolio returns as decimal fractions.  Index must be
        datetime-like so that calendar-year grouping and automatic frequency
        inference work correctly.
    risk_free_rate : float
        Annual risk-free rate used in the Sharpe ratio (default 2 % per year).
    turnover : pd.Series, optional
        Per-period portfolio turnover values (sum of absolute weight changes).
        If provided, average and annual turnover metrics are included.
    periods_per_year : float, optional
        Number of return periods per calendar year.  When ``None`` (default),
        inferred from the date span of *returns*: ``len(returns) / years``.
        Pass ``12`` explicitly to reproduce the original monthly-only behaviour.

    Returns
    -------
    Dict[str, float]
        Metric name → value.  Monetary values are decimal fractions; duration
        values are in periods (not necessarily months).
    """
    if len(returns) == 0:
        return {}

    n_periods = len(returns)

    # Infer periods_per_year from the actual date span when not provided.
    # This ensures correct annualisation for monthly, bi-weekly, or any other
    # rebalancing frequency without requiring the caller to know the factor.
    # Default to 12 (monthly) when no frequency hint is provided.
    # Callers with non-monthly data (e.g. biweekly) must supply the correct
    # value so that Sharpe and annualised return are properly scaled.
    if periods_per_year is None:
        periods_per_year = float(_MONTHS_PER_YEAR)

    # ── Core return / volatility ────────────────────────────────────────────
    total_return = float((1.0 + returns).prod() - 1.0)
    ann_return = float((1.0 + total_return) ** (periods_per_year / n_periods) - 1.0)
    ann_vol = float(returns.std() * math.sqrt(periods_per_year))

    # ── Risk-adjusted ratios ────────────────────────────────────────────────
    period_rf = (1.0 + risk_free_rate) ** (1.0 / periods_per_year) - 1.0
    excess_returns = returns - period_rf
    if ann_vol > 0:
        sharpe = float(excess_returns.mean() * periods_per_year / (returns.std() * math.sqrt(periods_per_year)))
    else:
        sharpe = float("nan")

    # ── Drawdown ────────────────────────────────────────────────────────────
    dd_series = compute_drawdown_series(returns)
    max_dd = float(dd_series.min())

    calmar = ann_return / abs(max_dd) if max_dd < 0 else float("nan")

    # Max drawdown duration: longest consecutive run of negative drawdown.
    max_dd_dur = 0
    cur_dur = 0
    for v in dd_series:
        if v < 0:
            cur_dur += 1
            max_dd_dur = max(max_dd_dur, cur_dur)
        else:
            cur_dur = 0

    # ── Sortino ratio ───────────────────────────────────────────────────────
    # Uses downside deviation (std of returns below the risk-free hurdle) rather
    # than total std, so it penalises only harmful volatility.  This is more
    # appropriate for asymmetric return distributions like AMAAM, which truncates
    # large losses via the momentum filter but still has a fat left tail.
    downside_returns = excess_returns[excess_returns < 0]
    if len(downside_returns) > 0:
        downside_dev = float(math.sqrt((downside_returns ** 2).mean()) * math.sqrt(periods_per_year))
        sortino = float(ann_return - risk_free_rate) / downside_dev if downside_dev > 0 else float("nan")
    else:
        sortino = float("nan")

    # ── Per-period extremes ─────────────────────────────────────────────────
    best_month = float(returns.max())
    worst_month = float(returns.min())
    pct_positive_months = float((returns > 0).mean())

    # ── Calendar-year analysis ──────────────────────────────────────────────
    if hasattr(returns.index, "to_period"):
        annual = returns.groupby(returns.index.to_period("Y")).apply(
            lambda r: float((1.0 + r).prod() - 1.0)
        )
    else:
        annual = pd.Series(dtype=float)

    best_year = float(annual.max()) if len(annual) > 0 else float("nan")
    worst_year = float(annual.min()) if len(annual) > 0 else float("nan")
    pct_positive_years = float((annual > 0).mean()) if len(annual) > 0 else float("nan")

    metrics: Dict[str, float] = {
        "Annualized Return":        ann_return,
        "Annualized Volatility":    ann_vol,
        "Sharpe Ratio":             sharpe,
        "Sortino Ratio":            sortino,
        "Calmar Ratio":             calmar,
        "Max Drawdown":             max_dd,
        "Max Drawdown Duration":    float(max_dd_dur),
        "Best Month":               best_month,
        "Worst Month":              worst_month,
        "Best Year":                best_year,
        "Worst Year":               worst_year,
        "% Positive Months":        pct_positive_months,
        "% Positive Years":         pct_positive_years,
        "Total Return":             total_return,
        "N Months":                 float(n_periods),
    }

    # ── Turnover (optional) ─────────────────────────────────────────────────
    if turnover is not None and len(turnover) > 0:
        avg_monthly_turnover = float(turnover.mean())
        # Annual turnover: sum all per-period values within each calendar year,
        # then average across years.
        if hasattr(turnover.index, "to_period"):
            annual_to = turnover.groupby(turnover.index.to_period("Y")).sum()
            avg_annual_turnover = float(annual_to.mean())
        else:
            avg_annual_turnover = avg_monthly_turnover * periods_per_year
        metrics["Avg Monthly Turnover"] = avg_monthly_turnover
        metrics["Avg Annual Turnover"]  = avg_annual_turnover

    return metrics


def compute_rolling_metrics(
    returns: pd.Series,
    window: int = 12,
) -> pd.DataFrame:
    """
    Compute rolling Sharpe ratio, rolling volatility, and rolling max drawdown.

    Parameters
    ----------
    returns : pd.Series
        Monthly return series.
    window : int
        Rolling window size in months (default 12).

    Returns
    -------
    pd.DataFrame
        Columns: ``rolling_sharpe``, ``rolling_vol``, ``rolling_max_dd``.
        Same index as *returns*.  First ``window − 1`` rows are NaN.
    """
    def _rolling_sharpe(r: pd.Series) -> float:
        v = r.std()
        return float((r.mean() / v) * math.sqrt(_MONTHS_PER_YEAR)) if v > 0 else float("nan")

    rolling_sharpe = returns.rolling(window, min_periods=window).apply(
        _rolling_sharpe, raw=False
    )
    rolling_vol = returns.rolling(window, min_periods=window).std() * math.sqrt(_MONTHS_PER_YEAR)

    dd = compute_drawdown_series(returns)
    rolling_max_dd = dd.rolling(window, min_periods=window).min()

    return pd.DataFrame(
        {"rolling_sharpe": rolling_sharpe, "rolling_vol": rolling_vol, "rolling_max_dd": rolling_max_dd}
    )
