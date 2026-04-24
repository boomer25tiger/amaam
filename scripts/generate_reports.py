"""
Report generation script for AMAAM.

Loads backtest results, sensitivity analysis outputs, and benchmark data, then
calls every chart function in src/visualization/matplotlib_charts.py (saving
PNGs to reports/figures/) and src/visualization/plotly_charts.py (saving HTML
to reports/interactive/). Also generates the summary statistics table for the
README. Requires a completed backtest run before it can be executed.
"""

import argparse
import logging
import sys
from dataclasses import replace
from pathlib import Path

# Ensure the project root is on the path so local packages resolve correctly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from itertools import product

from config.default_config import ModelConfig
from config.etf_universe import HEDGING_SLEEVE_TICKERS, MAIN_SLEEVE_TICKERS
from src.analysis.regime import compute_regime_metrics, define_regimes
from src.analysis.sensitivity import run_selection_sensitivity, run_weight_sensitivity
from src.backtest.benchmarks import (
    compute_seven_twelve,
    compute_sixty_forty,
    compute_spy_benchmark,
)
from src.backtest.engine import run_backtest
from src.data.loader import load_validated_data
import src.visualization.matplotlib_charts as mpl_charts
import src.visualization.plotly_charts as plotly_charts

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the report generation script."""
    parser = argparse.ArgumentParser(
        description="Generate all AMAAM visualisation charts (PNG + HTML).",
    )
    parser.add_argument(
        "--data-dir", default="data/processed",
        help="Path to the processed data directory (default: data/processed).",
    )
    parser.add_argument(
        "--figures-dir", default="reports/figures",
        help="Output directory for static PNG charts (default: reports/figures).",
    )
    parser.add_argument(
        "--interactive-dir", default="reports/interactive",
        help="Output directory for interactive HTML charts (default: reports/interactive).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Backtest helpers
# ---------------------------------------------------------------------------

def _run_backtests(data_dict, config):
    """Run all six backtest variants required for visualisation.

    Returns
    -------
    tuple
        (result_base, result_0bps, result_15bps, result_invvol, result_is, result_oos)
    """
    logger.info("Running base backtest (5 bps, equal weight, monthly)…")
    result_base = run_backtest(data_dict, config)

    logger.info("Running 0 bps cost variant…")
    result_0bps = run_backtest(data_dict, replace(config, transaction_cost=0.0))

    logger.info("Running 10 bps cost variant…")
    result_10bps = run_backtest(data_dict, replace(config, transaction_cost=0.001))

    logger.info("Running inverse-volatility weighting variant…")
    result_invvol = run_backtest(data_dict, replace(config, weighting_scheme="inverse_volatility"))

    logger.info("Running IS-only backtest (up to %s)…", config.holdout_start)
    result_is = run_backtest(
        data_dict,
        replace(config, backtest_end=config.holdout_start),
    )

    logger.info("Running OOS-only backtest (from %s)…", config.holdout_start)
    result_oos = run_backtest(
        data_dict,
        replace(config, backtest_start=config.holdout_start),
    )

    return result_base, result_0bps, result_10bps, result_invvol, result_is, result_oos


def _build_benchmark_returns(data_dict, config):
    """Compute monthly return series for all three benchmarks."""
    spy_returns    = compute_spy_benchmark(data_dict, config.backtest_start, config.backtest_end)
    sixty40_returns = compute_sixty_forty(data_dict,  config.backtest_start, config.backtest_end)
    seven12_returns = compute_seven_twelve(data_dict, config.backtest_start, config.backtest_end)
    return spy_returns, sixty40_returns, seven12_returns


# ---------------------------------------------------------------------------
# Data packaging helpers
# ---------------------------------------------------------------------------

def _build_equity_curves(result_base, spy_returns, sixty40_returns, seven12_returns):
    """Build normalised equity curve dict for all four strategies."""
    return {
        "AMAAM":   result_base.equity_curve / result_base.equity_curve.iloc[0],
        "SPY B&H": (1 + spy_returns).cumprod(),
        "60/40":   (1 + sixty40_returns).cumprod(),
        "7Twelve": (1 + seven12_returns).cumprod(),
    }


def _build_returns_dict(result_base, spy_returns, sixty40_returns, seven12_returns):
    """Build monthly return series dict for all four strategies."""
    return {
        "AMAAM":   result_base.monthly_returns,
        "SPY B&H": spy_returns,
        "60/40":   sixty40_returns,
        "7Twelve": seven12_returns,
    }


# ---------------------------------------------------------------------------
# Chart dispatcher
# ---------------------------------------------------------------------------

def _generate_mpl_charts(
    result_base,
    result_0bps,
    result_10bps,
    result_invvol,
    result_is,
    result_oos,
    equity_curves,
    returns_dict,
    data_dict,
    regime_df,
    weight_df,
    selection_df,
    fold_df,
    wf_stacked,
    figures_dir,
):
    """Call all matplotlib chart functions and return a list of saved paths.

    Chart 14 (factor weights) is intentionally omitted — it adds no
    information beyond what is already documented in the config file.
    """
    saved = []

    alloc = result_base.allocations

    saved.append(mpl_charts.plot_equity_curves(equity_curves, figures_dir))
    saved.append(mpl_charts.plot_drawdowns(equity_curves, figures_dir))
    saved.append(mpl_charts.plot_monthly_return_heatmap(result_base.monthly_returns, figures_dir))
    saved.append(mpl_charts.plot_annual_returns(returns_dict, figures_dir))
    saved.append(mpl_charts.plot_rolling_returns(returns_dict, figures_dir))
    saved.append(mpl_charts.plot_rolling_sharpe(returns_dict, figures_dir))
    saved.append(mpl_charts.plot_rolling_volatility(returns_dict, figures_dir))
    saved.append(mpl_charts.plot_rolling_drawdown(equity_curves, figures_dir))
    saved.append(mpl_charts.plot_return_distribution(returns_dict, figures_dir))
    saved.append(mpl_charts.plot_main_sleeve_allocation(alloc, MAIN_SLEEVE_TICKERS, figures_dir))
    saved.append(mpl_charts.plot_hedging_sleeve_allocation(
        alloc, HEDGING_SLEEVE_TICKERS, figures_dir))
    saved.append(mpl_charts.plot_hedging_weight_over_time(
        alloc, HEDGING_SLEEVE_TICKERS, figures_dir))
    saved.append(mpl_charts.plot_turnover(result_base.turnover, figures_dir))
    # Chart 14 (factor_weights) removed — static bar of fixed config params.
    saved.append(mpl_charts.plot_sleeve_return_decomposition(
        result_base.monthly_returns, alloc,
        MAIN_SLEEVE_TICKERS, HEDGING_SLEEVE_TICKERS, figures_dir,
    ))
    saved.append(mpl_charts.plot_correlation_matrix(data_dict, MAIN_SLEEVE_TICKERS, figures_dir))
    saved.append(mpl_charts.plot_regime_performance(regime_df, figures_dir))
    saved.append(mpl_charts.plot_weight_sensitivity_heatmap(weight_df, figures_dir))
    saved.append(mpl_charts.plot_selection_sensitivity(selection_df, figures_dir))
    saved.append(mpl_charts.plot_weighting_scheme_comparison(
        {"Equal Weight": result_base.metrics, "Inverse Vol": result_invvol.metrics},
        figures_dir,
    ))
    # Cost scenarios: 0 / 5 bps (base) / 10 bps.
    cost_equity = {
        "0 bps":  result_0bps.equity_curve  / result_0bps.equity_curve.iloc[0],
        "5 bps":  result_base.equity_curve  / result_base.equity_curve.iloc[0],
        "10 bps": result_10bps.equity_curve / result_10bps.equity_curve.iloc[0],
    }
    cost_metrics = {
        "0 bps":  result_0bps.metrics,
        "5 bps":  result_base.metrics,
        "10 bps": result_10bps.metrics,
    }
    saved.append(mpl_charts.plot_cost_scenarios_equity(cost_equity, figures_dir))
    saved.append(mpl_charts.plot_cost_scenarios_table(cost_metrics, figures_dir))
    saved.append(mpl_charts.plot_is_oos_equity(result_is, result_oos, figures_dir))
    saved.append(mpl_charts.plot_is_oos_stats_table(result_is, result_oos, figures_dir))

    # ── Supplementary charts (25–32) ─────────────────────────────────────────
    saved.append(mpl_charts.plot_risk_return_scatter(returns_dict, figures_dir))
    saved.append(mpl_charts.plot_beta_scatter(returns_dict, figures_dir))
    saved.append(mpl_charts.plot_rolling_spy_correlation(returns_dict, figures_dir))
    saved.append(mpl_charts.plot_drawdown_duration(equity_curves, figures_dir))
    saved.append(mpl_charts.plot_win_rate_stats(returns_dict, figures_dir))
    saved.append(mpl_charts.plot_var_cvar(returns_dict, figures_dir))
    saved.append(mpl_charts.plot_rolling_calmar(returns_dict, figures_dir))
    saved.append(mpl_charts.plot_return_autocorrelation(returns_dict, figures_dir))
    saved.append(mpl_charts.plot_walk_forward(fold_df, wf_stacked, figures_dir))

    return saved


def _generate_plotly_charts(
    result_base,
    result_0bps,
    result_10bps,
    result_invvol,
    result_is,
    result_oos,
    equity_curves,
    returns_dict,
    data_dict,
    regime_df,
    weight_df,
    selection_df,
    interactive_dir,
):
    """Call all Plotly chart functions and return a list of saved paths.

    Chart 14 (factor weights) is intentionally omitted — mirrors the
    change made to the matplotlib dispatcher.
    """
    saved = []

    alloc = result_base.allocations

    saved.append(plotly_charts.plot_equity_curves(equity_curves, interactive_dir))
    saved.append(plotly_charts.plot_drawdowns(equity_curves, interactive_dir))
    saved.append(plotly_charts.plot_monthly_return_heatmap(
        result_base.monthly_returns, interactive_dir))
    saved.append(plotly_charts.plot_annual_returns(returns_dict, interactive_dir))
    saved.append(plotly_charts.plot_rolling_returns(returns_dict, interactive_dir))
    saved.append(plotly_charts.plot_rolling_sharpe(returns_dict, interactive_dir))
    saved.append(plotly_charts.plot_rolling_volatility(returns_dict, interactive_dir))
    saved.append(plotly_charts.plot_rolling_drawdown(equity_curves, interactive_dir))
    saved.append(plotly_charts.plot_return_distribution(returns_dict, interactive_dir))
    saved.append(plotly_charts.plot_main_sleeve_allocation(
        alloc, MAIN_SLEEVE_TICKERS, interactive_dir))
    saved.append(plotly_charts.plot_hedging_sleeve_allocation(
        alloc, HEDGING_SLEEVE_TICKERS, interactive_dir))
    saved.append(plotly_charts.plot_hedging_weight_over_time(
        alloc, HEDGING_SLEEVE_TICKERS, interactive_dir))
    saved.append(plotly_charts.plot_turnover(result_base.turnover, interactive_dir))
    saved.append(plotly_charts.plot_sleeve_return_decomposition(
        result_base.monthly_returns, alloc,
        MAIN_SLEEVE_TICKERS, HEDGING_SLEEVE_TICKERS, interactive_dir,
    ))
    saved.append(plotly_charts.plot_correlation_matrix(
        data_dict, MAIN_SLEEVE_TICKERS, interactive_dir))
    saved.append(plotly_charts.plot_regime_performance(regime_df, interactive_dir))
    saved.append(plotly_charts.plot_weight_sensitivity_heatmap(weight_df, interactive_dir))
    saved.append(plotly_charts.plot_selection_sensitivity(selection_df, interactive_dir))
    saved.append(plotly_charts.plot_weighting_scheme_comparison(
        {"Equal Weight": result_base.metrics, "Inverse Vol": result_invvol.metrics},
        interactive_dir,
    ))
    # Cost scenarios: 0 / 5 bps (base) / 10 bps.
    cost_equity = {
        "0 bps":  result_0bps.equity_curve  / result_0bps.equity_curve.iloc[0],
        "5 bps":  result_base.equity_curve  / result_base.equity_curve.iloc[0],
        "10 bps": result_10bps.equity_curve / result_10bps.equity_curve.iloc[0],
    }
    cost_metrics = {
        "0 bps":  result_0bps.metrics,
        "5 bps":  result_base.metrics,
        "10 bps": result_10bps.metrics,
    }
    saved.append(plotly_charts.plot_cost_scenarios_equity(cost_equity, interactive_dir))
    saved.append(plotly_charts.plot_cost_scenarios_table(cost_metrics, interactive_dir))
    saved.append(plotly_charts.plot_is_oos_equity(result_is, result_oos, interactive_dir))
    saved.append(plotly_charts.plot_is_oos_stats_table(result_is, result_oos, interactive_dir))

    return saved


# ---------------------------------------------------------------------------
# Walk-forward validation
# ---------------------------------------------------------------------------

_WF_FOLDS = [
    ("Fold 1", "2007-08-01", "2012-12-31", "2013-01-01", "2014-12-31"),
    ("Fold 2", "2007-08-01", "2014-12-31", "2015-01-01", "2016-12-31"),
    ("Fold 3", "2007-08-01", "2016-12-31", "2017-01-01", "2018-12-31"),
    ("Fold 4", "2007-08-01", "2018-12-31", "2019-01-01", "2020-12-31"),
    ("Fold 5", "2007-08-01", "2020-12-31", "2021-01-01", "2022-12-31"),
    ("Fold 6", "2007-08-01", "2022-12-31", "2023-01-01", "2024-12-31"),
]

_WM_VALUES = [round(v, 2) for v in np.arange(0.45, 0.66, 0.05)]
_WC_VALUES = [round(v, 2) for v in np.arange(0.05, 0.46, 0.05)]
_WV_MIN    = 0.05


def _wf_sharpe(rets: pd.Series, start: str, end: str, rf: float = 0.02) -> float:
    r = rets[(rets.index >= start) & (rets.index <= end)]
    if len(r) < 6:
        return float("nan")
    ann_ret = float((1 + r).prod() ** (12 / len(r)) - 1)
    ann_vol = float(r.std() * np.sqrt(12))
    return (ann_ret - rf) / ann_vol if ann_vol > 0 else float("nan")


def _run_walk_forward(data_dict: dict, config: ModelConfig) -> tuple:
    """Run 6-fold expanding walk-forward validation.

    Returns
    -------
    tuple
        (fold_df, stacked_returns) where fold_df is a DataFrame of per-fold
        metrics and stacked_returns is a dict of concatenated OOS return series.
    """
    logger.info("Walk-forward: building %d grid configs…",
                sum(1 for wm, wc in product(_WM_VALUES, _WC_VALUES)
                    if round(1.0 - wm - wc, 2) >= _WV_MIN))

    grid_rets: list = []
    for wm, wc in product(_WM_VALUES, _WC_VALUES):
        wv = round(1.0 - wm - wc, 2)
        if wv < _WV_MIN:
            continue
        cfg = replace(config, weight_momentum=wm,
                      weight_volatility=wv, weight_correlation=wc)
        res = run_backtest(data_dict, cfg)
        grid_rets.append({"wm": wm, "wv": wv, "wc": wc, "rets": res.monthly_returns})

    df_grid     = pd.DataFrame(grid_rets)
    cand_rets   = df_grid[(df_grid["wm"] == 0.65) & (df_grid["wc"] == 0.10)].iloc[0]["rets"]
    base_rets   = df_grid[(df_grid["wm"] == 0.50) & (df_grid["wc"] == 0.25)].iloc[0]["rets"]

    fold_rows: list = []
    cand_slices: list = []
    base_slices: list = []

    for fold_name, _tr_s, tr_e, te_s, te_e in _WF_FOLDS:
        # Select best config on training window by Sharpe
        best_sr, best_row = -np.inf, None
        for _, row in df_grid.iterrows():
            sr = _wf_sharpe(row["rets"], _tr_s, tr_e)
            if not np.isnan(sr) and sr > best_sr:
                best_sr, best_row = sr, row

        fold_rows.append({
            "fold":          fold_name,
            "test_start":    te_s,
            "test_end":      te_e,
            "candidate_sr":  _wf_sharpe(cand_rets, te_s, te_e),
            "baseline_sr":   _wf_sharpe(base_rets,  te_s, te_e),
            "winner_sr":     _wf_sharpe(best_row["rets"], te_s, te_e),
        })
        cand_slices.append(cand_rets[(cand_rets.index >= te_s) & (cand_rets.index <= te_e)])
        base_slices.append(base_rets[(base_rets.index >= te_s) & (base_rets.index <= te_e)])
        logger.info("Walk-forward %s done.", fold_name)

    fold_df = pd.DataFrame(fold_rows)
    stacked_returns = {
        "AMAAM (canonical)":     pd.concat(cand_slices).sort_index(),
        "Baseline":              pd.concat(base_slices).sort_index(),
    }
    return fold_df, stacked_returns


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the full report generation pipeline."""
    args = _parse_args()
    config = ModelConfig()

    # ── Data ─────────────────────────────────────────────────────────────────
    logger.info("Loading validated data from %s…", args.data_dir)
    data_dict = load_validated_data(args.data_dir)
    logger.info("Loaded %d tickers.", len(data_dict))

    # ── Backtests ─────────────────────────────────────────────────────────────
    result_base, result_0bps, result_10bps, result_invvol, result_is, result_oos = (
        _run_backtests(data_dict, config)
    )

    # ── Benchmarks ────────────────────────────────────────────────────────────
    logger.info("Computing benchmark returns…")
    spy_returns, sixty40_returns, seven12_returns = _build_benchmark_returns(data_dict, config)

    equity_curves = _build_equity_curves(
        result_base, spy_returns, sixty40_returns, seven12_returns)
    returns_dict  = _build_returns_dict(
        result_base, spy_returns, sixty40_returns, seven12_returns)

    # ── Sensitivity sweeps (IS period only) ───────────────────────────────────
    logger.info("Running weight sensitivity sweep…")
    is_config = replace(config, backtest_end=config.holdout_start)
    weight_df = run_weight_sensitivity(data_dict, is_config)

    logger.info("Running selection sensitivity sweep…")
    selection_df = run_selection_sensitivity(data_dict, is_config)

    # ── Regime analysis ────────────────────────────────────────────────────────
    logger.info("Computing regime metrics…")
    regimes = define_regimes()
    full_returns_dict = {
        "AMAAM":   result_base.monthly_returns,
        "SPY B&H": spy_returns,
        "60/40":   sixty40_returns,
        "7Twelve": seven12_returns,
    }
    regime_df = compute_regime_metrics(full_returns_dict, regimes)

    # ── Walk-forward validation ───────────────────────────────────────────────
    logger.info("Running walk-forward validation (39 grid configs × 6 folds)…")
    fold_df, wf_stacked = _run_walk_forward(data_dict, config)

    # ── Matplotlib charts ─────────────────────────────────────────────────────
    logger.info("Generating static PNG charts → %s", args.figures_dir)
    mpl_saved = _generate_mpl_charts(
        result_base, result_0bps, result_10bps, result_invvol, result_is, result_oos,
        equity_curves, returns_dict, data_dict,
        regime_df, weight_df, selection_df,
        fold_df, wf_stacked,
        args.figures_dir,
    )

    # ── Plotly charts ─────────────────────────────────────────────────────────
    logger.info("Generating interactive HTML charts → %s", args.interactive_dir)
    plotly_saved = _generate_plotly_charts(
        result_base, result_0bps, result_10bps, result_invvol, result_is, result_oos,
        equity_curves, returns_dict, data_dict,
        regime_df, weight_df, selection_df,
        args.interactive_dir,
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    total = len(mpl_saved) + len(plotly_saved)
    logger.info("Done. %d files saved (%d PNG, %d HTML).",
                total, len(mpl_saved), len(plotly_saved))
    print(f"\nReport generation complete.")
    print(f"  Static PNGs  : {args.figures_dir}/ ({len(mpl_saved)} files)")
    print(f"  Interactive  : {args.interactive_dir}/ ({len(plotly_saved)} files)")


if __name__ == "__main__":
    main()
