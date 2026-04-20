"""
Backtesting engine for AMAAM.

Core event loop that iterates over monthly (or bi-weekly) rebalancing dates,
computes the full factor stack and allocation for each period, applies the
resulting weights to the following period's returns, and deducts transaction
costs based on portfolio turnover. Returns a BacktestResult dataclass containing
the equity curve, monthly returns, weight history, turnover series, and per-period
factor values. Rebalancing frequency is a configurable parameter. See Sections
5.3, 5.4, 5.5, and 9.13 of the specification.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd

from config.default_config import ModelConfig
from config.etf_universe import (
    HEDGING_SLEEVE_TICKERS,
    MAIN_SLEEVE_TICKERS,
)
from src.backtest.metrics import compute_all_metrics
from src.factors.correlation import (
    compute_correlation_all_assets,
    compute_market_correlation,
)
from src.factors.momentum import compute_absolute_momentum, compute_blended_momentum
from src.factors.trend import compute_trend_signal
from src.factors.volatility import compute_volatility_all_assets
from src.portfolio.allocation import compute_monthly_allocation
from src.ranking.trank import compute_trank, rank_assets, select_top_n

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class BacktestResult:
    """
    Container for all outputs of a completed AMAAM backtest run.

    Attributes
    ----------
    equity_curve : pd.Series
        Portfolio value starting at 1.0, indexed by execution date (the first
        trading day of each holding month).
    monthly_returns : pd.Series
        Net monthly portfolio returns (after transaction costs), indexed by
        execution date.
    allocations : pd.DataFrame
        Weight of each ticker at every signal date (index = signal dates,
        columns = all tickers ever held).  NaN means zero weight.
    turnover : pd.Series
        Sum of absolute weight changes at each rebalancing, indexed by signal
        date.
    config : ModelConfig
        The configuration used for this run.
    metrics : Dict[str, float]
        Full Section 5.6 performance table, populated after the run completes.
    """
    equity_curve:    pd.Series
    monthly_returns: pd.Series
    allocations:     pd.DataFrame
    turnover:        pd.Series
    config:          ModelConfig
    metrics:         Dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _month_end_dates(index: pd.DatetimeIndex) -> List[pd.Timestamp]:
    """Return the last trading day of each calendar month present in *index*."""
    return [
        g.index.max()
        for _, g in pd.Series(index=index, dtype=float).groupby(index.to_period("M"))
    ]


def _biweekly_signal_dates(index: pd.DatetimeIndex) -> List[pd.Timestamp]:
    """Return every 10th trading day within *index* (≈ bi-weekly, ~25 events/year).

    A step of 10 trading days approximates two calendar weeks and yields roughly
    25 rebalancing events per year, versus 12 for monthly frequency.
    """
    all_days = sorted(index.unique().tolist())
    return all_days[::10]  # step of 10 trading days


def _build_exec_date_map(
    signal_dates: List[pd.Timestamp],
    all_dates: List[pd.Timestamp],
) -> Dict[pd.Timestamp, pd.Timestamp]:
    """
    For each signal date return the next calendar trading day.

    The one-day implementation lag (Section 5.3): signals are computed on the
    last trading day of month M; allocation is executed at the close of the
    next trading day (first day of month M+1).
    """
    date_to_idx = {d: i for i, d in enumerate(all_dates)}
    result: Dict[pd.Timestamp, pd.Timestamp] = {}
    for sig in signal_dates:
        idx = date_to_idx.get(sig)
        if idx is not None and idx + 1 < len(all_dates):
            result[sig] = all_dates[idx + 1]
    return result


def _precompute_factors(
    data_dict: Dict[str, pd.DataFrame],
    main_tickers: List[str],
    hedge_tickers: List[str],
    config: ModelConfig,
) -> Dict[str, pd.DataFrame]:
    """
    Compute all four factor series for every sleeve ticker once before the loop.

    Correlation is computed within each sleeve independently (Section 3.4).
    All other factors are computed per-ticker and stored as wide DataFrames.

    Returns
    -------
    Dict with keys ``"momentum"``, ``"volatility"``, ``"corr_main"``,
    ``"corr_hedge"``, ``"trend"``.
    """
    all_tickers = main_tickers + hedge_tickers
    logger.info("Pre-computing factor series for %d tickers…", len(all_tickers))

    # Use blended 1/3/6/12-month average when configured; otherwise the
    # standard single 4-month ROC from the original spec (Section 3.2).
    if config.momentum_blend:
        momentum = pd.DataFrame({
            t: compute_blended_momentum(
                data_dict[t]["Close"], config.momentum_blend_lookbacks
            )
            for t in all_tickers
        })
    else:
        momentum = pd.DataFrame({
            t: compute_absolute_momentum(data_dict[t]["Close"], config.momentum_lookback)
            for t in all_tickers
        })

    # Yang-Zhang requires OHLC; pass the full per-ticker DataFrames.
    volatility = compute_volatility_all_assets(
        {t: data_dict[t] for t in all_tickers}, config
    )

    trend = pd.DataFrame({
        t: compute_trend_signal(
            data_dict[t]["High"], data_dict[t]["Low"], data_dict[t]["Close"],
            config.atr_period, config.atr_upper_lookback, config.atr_lower_lookback,
        )
        for t in all_tickers
    })

    # Dispatch to the configured correlation estimator.  "market" uses rolling
    # Pearson correlation with SPY, which discriminates better within the
    # equity-heavy main sleeve than average pairwise correlation (Section 3.4).
    if config.correlation_method == "market":
        corr_main = compute_market_correlation(
            data_dict, main_tickers, config.correlation_lookback,
        )
        corr_hedge = compute_market_correlation(
            data_dict, hedge_tickers, config.correlation_lookback,
        )
    else:
        corr_main = compute_correlation_all_assets(
            {t: data_dict[t] for t in main_tickers},
            main_tickers,
            config.correlation_lookback,
        )
        corr_hedge = compute_correlation_all_assets(
            {t: data_dict[t] for t in hedge_tickers},
            hedge_tickers,
            config.correlation_lookback,
        )

    return {
        "momentum":  momentum,
        "volatility": volatility,
        "corr_main":  corr_main,
        "corr_hedge": corr_hedge,
        "trend":      trend,
    }


def _allocation_at_date(
    signal_date: pd.Timestamp,
    factors: Dict[str, pd.DataFrame],
    main_tickers: List[str],
    hedge_tickers: List[str],
    config: ModelConfig,
) -> Dict[str, float] | None:
    """
    Compute the full monthly allocation for one signal date.

    Returns ``None`` if any required factor is entirely NaN on this date
    (i.e., still within the warm-up period).
    """
    def _snap(df: pd.DataFrame, tickers: List[str]) -> pd.Series:
        """Last valid row at or before *signal_date* for *tickers*."""
        sub = df.loc[:signal_date, tickers]
        valid = sub.dropna(how="all")
        return valid.iloc[-1] if not valid.empty else pd.Series(dtype=float)

    M_main  = _snap(factors["momentum"],  main_tickers)
    V_main  = _snap(factors["volatility"], main_tickers)
    C_main  = _snap(factors["corr_main"],  main_tickers)
    T_main  = _snap(factors["trend"],      main_tickers)

    M_hedge = _snap(factors["momentum"],  hedge_tickers)
    V_hedge = _snap(factors["volatility"], hedge_tickers)
    C_hedge = _snap(factors["corr_hedge"], hedge_tickers)
    T_hedge = _snap(factors["trend"],      hedge_tickers)

    # Guard: if any sleeve lacks enough data for a full ranking, skip.
    for s, name in [(M_main, "main M"), (V_main, "main V"),
                    (C_main, "main C"), (T_main, "main T"),
                    (M_hedge, "hedge M")]:
        if s.isna().all():
            logger.debug("Skipping %s — all NaN for %s", signal_date.date(), name)
            return None

    # Rank and score — main sleeve.
    rM = rank_assets(M_main, ascending=True)
    rV = rank_assets(V_main, ascending=False)
    rC = rank_assets(C_main, ascending=False)
    trank_main = compute_trank(rM, rV, rC, T_main, M_main, config)

    # Rank and score — hedging sleeve.
    rMh = rank_assets(M_hedge, ascending=True)
    rVh = rank_assets(V_hedge, ascending=False)
    rCh = rank_assets(C_hedge, ascending=False)
    trank_hedge = compute_trank(rMh, rVh, rCh, T_hedge, M_hedge, config)

    top_main  = select_top_n(trank_main,  config.main_sleeve_top_n)
    top_hedge = select_top_n(trank_hedge, config.hedging_sleeve_top_n)

    # Pass precomputed volatility so inverse_volatility weighting scheme
    # can use it; under the default "equal" scheme it is ignored.
    return compute_monthly_allocation(
        top_main, top_hedge, M_main, M_hedge, config,
        main_volatility=V_main,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_backtest(
    data_dict: Dict[str, pd.DataFrame],
    config: ModelConfig,
) -> BacktestResult:
    """
    Execute the AMAAM monthly backtest over the full configured date range.

    Implements the Section 5.3 execution model: signals computed at the last
    trading day of month M are applied at the next trading day's close and
    held for the full following month.  Transaction costs (Section 5.5) are
    deducted from each period's gross return.

    Parameters
    ----------
    data_dict : Dict[str, pd.DataFrame]
        Processed OHLCV data, keyed by ticker.  Must contain at least all
        main-sleeve and hedging-sleeve tickers that are present in
        ``config``'s universe.
    config : ModelConfig
        Full model configuration.

    Returns
    -------
    BacktestResult
        Equity curve, monthly returns, allocations, turnover, and the
        populated Section 5.6 metrics table.
    """
    main_tickers  = [t for t in MAIN_SLEEVE_TICKERS  if t in data_dict]
    hedge_tickers = [t for t in HEDGING_SLEEVE_TICKERS if t in data_dict]
    all_sleeve    = main_tickers + hedge_tickers

    # Build a single close-price matrix covering all tickers we will trade.
    closes = pd.DataFrame({t: data_dict[t]["Close"] for t in all_sleeve}).ffill()
    all_dates = closes.index.tolist()

    # Precompute all factor series once.
    factors = _precompute_factors(data_dict, main_tickers, hedge_tickers, config)

    # Signal dates: determined by rebalancing frequency from config.
    # Biweekly uses every 10th trading day (~25/year); all other values default
    # to month-end (12/year).  Frequency is kept in config so sweeps can override
    # it without touching engine logic.
    window = closes.loc[config.backtest_start:config.backtest_end]
    if config.rebalancing_frequency == "biweekly":
        signal_dates = _biweekly_signal_dates(window.index)
    else:
        signal_dates = _month_end_dates(window.index)
    exec_date_map = _build_exec_date_map(signal_dates, all_dates)

    logger.info(
        "Backtest: %s → %s  (%d signal dates, %s)",
        config.backtest_start, config.backtest_end, len(signal_dates),
        config.rebalancing_frequency,
    )

    # ── Main event loop ──────────────────────────────────────────────────────
    monthly_rets:   List[tuple] = []
    equity_pts:     List[tuple] = []
    alloc_records:  List[dict]  = []
    turnover_list:  List[tuple] = []

    equity     = 1.0
    prev_alloc: Dict[str, float] = {}

    for i, sig_date in enumerate(signal_dates[:-1]):
        next_sig = signal_dates[i + 1]

        exec0 = exec_date_map.get(sig_date)
        exec1 = exec_date_map.get(next_sig)

        if exec0 is None or exec1 is None:
            continue
        if exec0 not in closes.index or exec1 not in closes.index:
            continue

        alloc = _allocation_at_date(sig_date, factors, main_tickers, hedge_tickers, config)
        if alloc is None:
            continue   # still in warm-up

        # ── Transaction costs (Section 5.5) ──────────────────────────────
        all_keys = set(alloc) | set(prev_alloc)
        turnover = sum(abs(alloc.get(t, 0.0) - prev_alloc.get(t, 0.0)) for t in all_keys)
        cost = turnover * config.transaction_cost / 2.0

        # ── Holding-period return ─────────────────────────────────────────
        port_ret = 0.0
        for ticker, weight in alloc.items():
            if ticker in closes.columns:
                p0 = closes.at[exec0, ticker]
                p1 = closes.at[exec1, ticker]
                if p0 > 0 and not (np.isnan(p0) or np.isnan(p1)):
                    port_ret += weight * (p1 / p0 - 1.0)

        port_ret -= cost
        equity *= (1.0 + port_ret)

        monthly_rets.append((exec1, port_ret))
        equity_pts.append((exec1, equity))
        turnover_list.append((sig_date, turnover))

        row = {t: alloc.get(t, 0.0) for t in all_sleeve}
        row["date"] = sig_date
        alloc_records.append(row)

        prev_alloc = alloc

    # ── Assemble result objects ──────────────────────────────────────────────
    monthly_returns = pd.Series(
        {d: r for d, r in monthly_rets}, name="monthly_return"
    )
    equity_curve = pd.Series(
        {d: v for d, v in equity_pts}, name="equity"
    )
    turnover_series = pd.Series(
        {d: v for d, v in turnover_list}, name="turnover"
    )

    alloc_df = pd.DataFrame(alloc_records).set_index("date") if alloc_records else pd.DataFrame()

    # Derive the actual annualisation factor from the realised return series so
    # that biweekly (~25/year) and any future frequency produce correct Sharpe
    # and vol figures rather than being forced through the monthly (12/year) path.
    n_ret = len(monthly_returns)
    if n_ret > 1:
        span_days = (monthly_returns.index.max() - monthly_returns.index.min()).days
        periods_per_year: float = n_ret / (span_days / 365.25) if span_days > 0 else 12.0
    else:
        periods_per_year = 12.0

    metrics = compute_all_metrics(
        monthly_returns,
        risk_free_rate=0.02,
        turnover=turnover_series,
        periods_per_year=periods_per_year,
    )

    logger.info(
        "Backtest complete — %.0f months | Ann. return: %.2f%% | Sharpe: %.2f | Max DD: %.2f%%",
        metrics.get("N Months", 0),
        metrics.get("Annualized Return", 0) * 100,
        metrics.get("Sharpe Ratio", float("nan")),
        metrics.get("Max Drawdown", 0) * 100,
    )

    return BacktestResult(
        equity_curve=equity_curve,
        monthly_returns=monthly_returns,
        allocations=alloc_df,
        turnover=turnover_series,
        config=config,
        metrics=metrics,
    )
