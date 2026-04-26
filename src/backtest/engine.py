"""
Backtesting engine for AMAAM.

Runs the monthly (or bi-weekly) event loop: computes factor scores, builds
allocations, applies weights to the next period's returns, and deducts
transaction costs. Produces a BacktestResult with equity curve, returns,
weights, and metrics.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config.default_config import ModelConfig
from config.etf_universe import (
    CASH_PROXY,
    HEDGING_SLEEVE_TICKERS,
    MAIN_SLEEVE_TICKERS,
)
from src.backtest.metrics import compute_all_metrics
from src.factors.correlation import (
    compute_blended_correlation_all_assets,
    compute_correlation_all_assets,
    compute_cross_sleeve_correlation,
    compute_ewm_correlation_all_assets,
    compute_market_beta,
    compute_market_correlation,
    compute_portfolio_correlation,
    compute_portfolio_correlation_all_assets,
    compute_stress_blend_correlation_all_assets,
    compute_stress_correlation_all_assets,
)
from src.factors.momentum import compute_absolute_momentum, compute_blended_momentum
from src.factors.trend import compute_trend_all_assets
from src.factors.volatility import compute_blended_yang_zhang_vol, compute_volatility_all_assets
from src.portfolio.allocation import compute_monthly_allocation
from src.portfolio.volatility_targeting import apply_vol_targeting
from src.ranking.trank import compute_trank, rank_assets, select_top_n

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class BacktestResult:
    """Immutable output bundle returned by ``run_backtest``."""
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
    """Return the last trading day of each calendar month present in *index* to serve as monthly signal dates."""
    return [
        g.index.max()
        for _, g in pd.Series(index=index, dtype=float).groupby(index.to_period("M"))
    ]


def _biweekly_signal_dates(index: pd.DatetimeIndex) -> List[pd.Timestamp]:
    """Return every 10th trading day within *index* to approximate bi-weekly rebalancing (~25 events/year).

    Notes
    -----
    A fixed step of 10 trading days is used rather than a calendar-based
    fortnight so the count stays stable across months of different lengths.
    """
    all_days = sorted(index.unique().tolist())
    return all_days[::10]  # step of 10 trading days


def _build_exec_date_map(
    signal_dates: List[pd.Timestamp],
    all_dates: List[pd.Timestamp],
) -> Dict[pd.Timestamp, pd.Timestamp]:
    """Map each signal date to the immediately following trading day, implementing the one-day execution lag required by Section 5.3."""
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
    """Compute momentum, volatility, correlation, and trend series for all tickers once before the rebalancing loop, so the loop itself is just lookups.

    Notes
    -----
    Correlation is computed independently per sleeve (Section 3.4); the method
    and blend settings in ``config`` control which variant is used for each
    factor.
    """
    all_tickers = main_tickers + hedge_tickers
    logger.info("Pre-computing factor series for %d tickers…", len(all_tickers))

    # Use blended 1/3/6/12-month average when configured; otherwise the
    # standard single 4-month ROC from the original spec (Section 3.2).
    if config.momentum_blend:
        momentum = pd.DataFrame({
            t: compute_blended_momentum(
                data_dict[t]["Close"],
                config.momentum_blend_lookbacks,
                skip_days=config.momentum_skip_days,
            )
            for t in all_tickers
        })
    else:
        momentum = pd.DataFrame({
            t: compute_absolute_momentum(data_dict[t]["Close"], config.momentum_lookback)
            for t in all_tickers
        })

    # Yang-Zhang requires OHLC; pass the full per-ticker DataFrames.
    # When vol_blend is True, average YZ across multiple window lengths to
    # reduce sensitivity to any single horizon (same philosophy as blended momentum).
    if config.vol_blend:
        volatility = compute_blended_yang_zhang_vol(
            {t: data_dict[t] for t in all_tickers},
            config.vol_blend_lookbacks,
        )
    else:
        volatility = compute_volatility_all_assets(
            {t: data_dict[t] for t in all_tickers}, config
        )

    # Dispatch to the configured trend signal method via the canonical trend module.
    trend = compute_trend_all_assets({t: data_dict[t] for t in all_tickers}, config)

    if config.correlation_blend:
        corr_main = compute_blended_correlation_all_assets(
            {t: data_dict[t] for t in main_tickers},
            main_tickers,
            config.correlation_blend_lookbacks,
        )
        corr_hedge = compute_blended_correlation_all_assets(
            {t: data_dict[t] for t in hedge_tickers},
            hedge_tickers,
            config.correlation_blend_lookbacks,
        )
    elif config.correlation_method == "portfolio":
        corr_main = compute_portfolio_correlation(
            {t: data_dict[t] for t in main_tickers},
            main_tickers,
            config.correlation_lookback,
        )
        corr_hedge = compute_portfolio_correlation(
            {t: data_dict[t] for t in hedge_tickers},
            hedge_tickers,
            config.correlation_lookback,
        )
    elif config.correlation_method == "portfolio_all":
        corr_main = compute_portfolio_correlation_all_assets(
            {t: data_dict[t] for t in main_tickers},
            main_tickers,
            config.correlation_lookback,
        )
        corr_hedge = compute_portfolio_correlation_all_assets(
            {t: data_dict[t] for t in hedge_tickers},
            hedge_tickers,
            config.correlation_lookback,
        )
    elif config.correlation_method == "market":
        corr_main = compute_market_correlation(
            data_dict, main_tickers, config.correlation_lookback,
        )
        corr_hedge = compute_market_correlation(
            data_dict, hedge_tickers, config.correlation_lookback,
        )
    elif config.correlation_method == "beta":
        corr_main = compute_market_beta(
            data_dict, main_tickers, config.correlation_lookback,
        )
        corr_hedge = compute_market_beta(
            data_dict, hedge_tickers, config.correlation_lookback,
        )
    elif config.correlation_method == "ewm":
        corr_main = compute_ewm_correlation_all_assets(
            {t: data_dict[t] for t in main_tickers},
            main_tickers,
            config.correlation_ewm_span,
        )
        corr_hedge = compute_ewm_correlation_all_assets(
            {t: data_dict[t] for t in hedge_tickers},
            hedge_tickers,
            config.correlation_ewm_span,
        )
    elif config.correlation_method == "stress_vol":
        corr_main = compute_stress_correlation_all_assets(
            data_dict, main_tickers, config.correlation_lookback,
            stress_method="vol",
            vol_multiplier=config.stress_vol_multiplier,
        )
        corr_hedge = compute_stress_correlation_all_assets(
            data_dict, hedge_tickers, config.correlation_lookback,
            stress_method="vol",
            vol_multiplier=config.stress_vol_multiplier,
        )
    elif config.correlation_method == "stress_drawdown":
        corr_main = compute_stress_correlation_all_assets(
            data_dict, main_tickers, config.correlation_lookback,
            stress_method="drawdown",
            vol_multiplier=config.stress_vol_multiplier,
        )
        corr_hedge = compute_stress_correlation_all_assets(
            data_dict, hedge_tickers, config.correlation_lookback,
            stress_method="drawdown",
            vol_multiplier=config.stress_vol_multiplier,
        )
    elif config.correlation_method == "stress_blend":
        corr_main = compute_stress_blend_correlation_all_assets(
            data_dict, main_tickers, config.correlation_lookback,
            vol_multiplier=config.stress_vol_multiplier,
        )
        corr_hedge = compute_stress_blend_correlation_all_assets(
            data_dict, hedge_tickers, config.correlation_lookback,
            vol_multiplier=config.stress_vol_multiplier,
        )
    elif config.correlation_method == "cross_sleeve":
        c_within = compute_correlation_all_assets(
            {t: data_dict[t] for t in main_tickers},
            main_tickers,
            config.correlation_lookback,
        )
        c_cross = compute_cross_sleeve_correlation(
            data_dict,
            main_tickers,
            hedge_tickers,
            config.correlation_lookback,
        )
        # Align indices before combining — both should share the same date range
        # but reindex defensively to avoid silent NaN introduction from gaps.
        c_cross = c_cross.reindex(index=c_within.index, columns=c_within.columns)
        corr_main = c_within.add(c_cross.mul(config.cross_sleeve_lambda))

        corr_hedge = compute_correlation_all_assets(
            {t: data_dict[t] for t in hedge_tickers},
            hedge_tickers,
            config.correlation_lookback,
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
    prev_main_selected: Optional[List[str]] = None,
    prev_hedge_selected: Optional[List[str]] = None,
) -> Optional[Tuple[Dict[str, float], List[str], List[str]]]:
    """Rank tickers and compute sleeve allocations for a single signal date.

    Returns ``None`` during the warm-up period when factors are still all-NaN.
    Otherwise returns a 3-tuple ``(alloc, top_main, top_hedge)`` so the caller
    can maintain previous-selection state for hysteresis across periods.

    Parameters
    ----------
    signal_date : pd.Timestamp
        Month-end (or bi-weekly) signal date.
    factors : Dict[str, pd.DataFrame]
        Pre-computed factor matrices from ``_precompute_factors``.
    main_tickers : List[str]
        Tickers in the main sleeve.
    hedge_tickers : List[str]
        Tickers in the hedging sleeve.
    config : ModelConfig
        Supplies selection counts, weights, and hysteresis settings.
    prev_main_selected : List[str] or None
        Main-sleeve selections from the previous period.  Fed to ``select_top_n``
        when ``config.selection_exit_buffer > 0`` to apply exit hysteresis.
    prev_hedge_selected : List[str] or None
        Hedging-sleeve selections from the previous period.
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
    rT_main  = rank_assets(T_main,  ascending=True) if config.trend_rank_scale else T_main
    rT_hedge = rank_assets(T_hedge, ascending=True) if config.trend_rank_scale else T_hedge
    trank_main = compute_trank(rM, rV, rC, rT_main, M_main, config)

    # Rank and score — hedging sleeve.
    rMh = rank_assets(M_hedge, ascending=True)
    rVh = rank_assets(V_hedge, ascending=False)
    rCh = rank_assets(C_hedge, ascending=False)
    trank_hedge = compute_trank(rMh, rVh, rCh, rT_hedge, M_hedge, config)

    # Guard: if either TRank series is entirely NaN (can happen in the warm-up
    # period before all factors have accumulated enough history), skip this date.
    if trank_main.isna().all():
        logger.debug("Skipping %s — trank_main all NaN", signal_date.date())
        return None
    if trank_hedge.isna().all():
        logger.debug("Skipping %s — trank_hedge all NaN", signal_date.date())
        return None

    # Apply selection hysteresis when configured: incumbents survive until they
    # drop outside the wider top-(N + exit_buffer) zone rather than exiting
    # immediately when they slip out of top-N.  exit_buffer=0 is the standard
    # behaviour and is the default in ModelConfig.
    buf = config.selection_exit_buffer
    top_main  = select_top_n(
        trank_main, config.main_sleeve_top_n,
        prev_selected=prev_main_selected, exit_buffer=buf,
    )
    top_hedge = select_top_n(
        trank_hedge, config.hedging_sleeve_top_n,
        prev_selected=prev_hedge_selected, exit_buffer=buf,
    )

    # Pass precomputed volatility so inverse_volatility weighting scheme
    # can use it; under the default "equal" scheme it is ignored.
    alloc = compute_monthly_allocation(
        top_main, top_hedge, M_main, M_hedge, config,
        main_volatility=V_main,
    )
    return alloc, top_main, top_hedge


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_backtest(
    data_dict: Dict[str, pd.DataFrame],
    config: ModelConfig,
) -> BacktestResult:
    """Execute the full AMAAM backtest and return a populated ``BacktestResult``.

    Notes
    -----
    The annualisation factor for Sharpe and volatility is derived from the
    realised return frequency rather than assuming 12 periods/year, so results
    are correct for both monthly and bi-weekly runs.
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
    # Previous sleeve selections are threaded across periods so hysteresis can
    # compare the current TRank ranking against last month's holdings.  These
    # start empty (no incumbents before the first live period).
    prev_main_selected:  List[str] = []
    prev_hedge_selected: List[str] = []

    for i, sig_date in enumerate(signal_dates[:-1]):
        next_sig = signal_dates[i + 1]

        exec0 = exec_date_map.get(sig_date)
        exec1 = exec_date_map.get(next_sig)

        if exec0 is None or exec1 is None:
            continue
        if exec0 not in closes.index or exec1 not in closes.index:
            continue

        result = _allocation_at_date(
            sig_date, factors, main_tickers, hedge_tickers, config,
            prev_main_selected, prev_hedge_selected,
        )
        if result is None:
            continue   # still in warm-up
        alloc, prev_main_selected, prev_hedge_selected = result

        if config.vol_targeting:
            alloc = apply_vol_targeting(
                alloc, closes, sig_date,
                config.vol_target,
                config.vol_target_lookback,
                config.vol_target_max_leverage,
                CASH_PROXY,
            )

        all_keys = set(alloc) | set(prev_alloc)
        turnover = sum(abs(alloc.get(t, 0.0) - prev_alloc.get(t, 0.0)) for t in all_keys)
        # turnover = Σ|Δw|, which counts each leg (buy and sell) separately.
        # transaction_cost is the one-way cost per leg, so the total cost is
        # turnover × tc (no halving required).
        cost = turnover * config.transaction_cost

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
