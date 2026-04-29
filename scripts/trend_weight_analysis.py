"""
Trend Signal Analysis — AMAAM
=================================
Part 1 : Code review of T values, TRank formula, and scale comparison.
Part 2 : Empirical analysis of how often / how much trend impacts rankings.
Part 3 : wT sensitivity test: 1.0 vs 2.0 vs 3.0.

Usage
-----
    python3.13 scripts/trend_weight_analysis.py
"""

import copy
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config.default_config import ModelConfig
from config.etf_universe import MAIN_SLEEVE_TICKERS
from src.backtest.engine import run_backtest, _precompute_factors
from src.data.loader import load_validated_data
from src.factors.trend import (
    compute_sma_ratio_signal,
    _SIGNAL_UP,
    _SIGNAL_DOWN,
    _SIGNAL_INIT,
)
from src.ranking.trank import compute_trank, rank_assets, select_top_n


# ── Constants ─────────────────────────────────────────────────────────────────
DATA_DIR = ROOT / "data" / "processed"
IS_START  = "2004-01-01"
IS_END    = "2017-12-31"
OOS_START = "2018-01-01"
OOS_END   = ModelConfig().backtest_end

STRESS_PERIODS = {
    "GFC":           ("2007-10-01", "2009-03-31"),
    "China Selloff": ("2015-08-01", "2016-02-29"),
    "COVID":         ("2020-02-01", "2020-03-31"),
    "2022 Rates":    ("2022-01-01", "2022-12-31"),
    "2025 Tariffs":  ("2025-02-01", ModelConfig().backtest_end),
}


# ── Helper: compute_metrics_for_slice ─────────────────────────────────────────

def _metrics(rets: pd.Series, periods_per_year: float = 12.0) -> dict:
    """Annualised return, vol, Sharpe, MDD from a monthly return slice."""
    if rets.empty or len(rets) < 2:
        return {"ann_ret": float("nan"), "ann_vol": float("nan"),
                "sharpe": float("nan"), "mdd": float("nan")}
    ann_ret = (1 + rets).prod() ** (periods_per_year / len(rets)) - 1
    ann_vol = rets.std() * np.sqrt(periods_per_year)
    sharpe  = (ann_ret - 0.02) / ann_vol if ann_vol > 0 else float("nan")
    equity  = (1 + rets).cumprod()
    peak    = equity.cummax()
    mdd     = ((equity - peak) / peak).min()
    return {"ann_ret": ann_ret, "ann_vol": ann_vol, "sharpe": sharpe, "mdd": mdd}


def _slice_rets(rets: pd.Series, start: str, end: str) -> pd.Series:
    return rets.loc[start:end]


def _stress_return(rets: pd.Series, start: str, end: str) -> float:
    """Cumulative return over a stress window (monthly returns, simple product)."""
    sl = rets.loc[start:end]
    if sl.empty:
        return float("nan")
    return (1 + sl).prod() - 1


# ── Part 1 : Code review ───────────────────────────────────────────────────────

def print_code_review(cfg: ModelConfig, n_main: int = 16) -> None:
    print("=" * 70)
    print("=== TREND SIGNAL ANALYSIS ===")
    print("=" * 70)
    print()
    print("CODE REVIEW:")
    print(f"  Trend method active      : {cfg.trend_method}")
    print(f"  _SIGNAL_UP               : {_SIGNAL_UP:+.1f}")
    print(f"  _SIGNAL_DOWN             : {_SIGNAL_DOWN:+.1f}")
    print(f"  _SIGNAL_INIT             : {_SIGNAL_INIT:+.1f}")
    print(f"  T values produced        : {{{_SIGNAL_DOWN:+.1f}, {_SIGNAL_UP:+.1f}}}")
    print()
    print("  TRank formula (Section 3.1):")
    print("    TRank = wM*Rank(M) + wV*Rank(V) + wC*Rank(C) + wT*T + M/n")
    print()
    print(f"  Config weights (default):")
    print(f"    wM={cfg.weight_momentum}, wV={cfg.weight_volatility}, "
          f"wC={cfg.weight_correlation}, wT={cfg.weight_trend}")
    print(f"    trend_rank_scale = {cfg.trend_rank_scale}")
    print()

    # Scale analysis with n=16 main sleeve assets
    n = n_main
    # MVC rank range: ranks run 1..n, so max MVC contribution = wM*n + wV*n + wC*n
    max_mvc = (cfg.weight_momentum + cfg.weight_volatility + cfg.weight_correlation) * n
    min_mvc = (cfg.weight_momentum + cfg.weight_volatility + cfg.weight_correlation) * 1
    mvc_spread = max_mvc - min_mvc

    # T contribution span: wT * (+2 - (-2)) = wT * 4
    max_t_contribution = cfg.weight_trend * _SIGNAL_UP    # wT * +2
    min_t_contribution = cfg.weight_trend * _SIGNAL_DOWN  # wT * -2
    t_spread = max_t_contribution - min_t_contribution    # wT * 4

    # TRank full range (ignoring M/n tiebreaker)
    trank_max = max_mvc + max_t_contribution
    trank_min = min_mvc + min_t_contribution
    trank_range = trank_max - trank_min

    t_pct_of_trank_range = 100 * t_spread / trank_range if trank_range > 0 else 0.0

    print(f"  With n={n} main sleeve assets and wT={cfg.weight_trend}:")
    print(f"    Max MVC sum  (best asset):  wM*{n}+wV*{n}+wC*{n} = "
          f"{max_mvc:.3f}")
    print(f"    Min MVC sum  (worst asset): wM*1+wV*1+wC*1  = "
          f"{min_mvc:.3f}")
    print(f"    MVC cross-sectional spread: {mvc_spread:.3f}")
    print(f"    Max T contribution (+2):    wT*2 = {max_t_contribution:+.3f}")
    print(f"    Min T contribution (-2):    wT*2 = {min_t_contribution:+.3f}")
    print(f"    T cross-sectional spread:   wT*4 = {t_spread:.3f}")
    print(f"    Full TRank range:           {trank_range:.3f}")
    print(f"    T as %% of TRank range:      {t_pct_of_trank_range:.1f}%%")
    print()
    print("  NOTE: With trend_rank_scale=False (default), T enters TRank as a")
    print("  raw ±2 value (not ranked 1..N). wT=1.0 gives T a cross-sectional")
    print("  spread of 4 vs an MVC spread of ~15 (for n=16). T adds ~21% of")
    print("  total TRank range.")
    print()


# ── Part 2 : Empirical analysis ────────────────────────────────────────────────

def run_empirical_analysis(
    data_dict: dict,
    cfg: ModelConfig,
) -> None:
    print("-" * 70)
    print("EMPIRICAL ANALYSIS:")
    print()

    main_tickers = [t for t in MAIN_SLEEVE_TICKERS if t in data_dict]
    hedge_tickers = [t for t in ["GLD", "TLT", "IEF", "SH", "UUP", "SHY"] if t in data_dict]
    n = len(main_tickers)

    # Precompute all factors once
    factors = _precompute_factors(data_dict, main_tickers, hedge_tickers, cfg)

    trend_df = factors["trend"][main_tickers]
    mom_df   = factors["momentum"][main_tickers]
    vol_df   = factors["volatility"][main_tickers]
    corr_df  = factors["corr_main"][main_tickers]

    # Month-end sampling
    month_ends = [
        g.index.max()
        for _, g in pd.Series(index=trend_df.index, dtype=float).groupby(
            trend_df.index.to_period("M")
        )
    ]

    # Filter to backtest window and drop NaN months
    valid_months = [
        d for d in month_ends
        if cfg.backtest_start <= str(d.date()) <= cfg.backtest_end
        and trend_df.loc[d].notna().all()
        and mom_df.loc[d].notna().all()
        and vol_df.loc[d].notna().all()
        and corr_df.loc[d].notna().all()
    ]

    if not valid_months:
        print("  No valid months found — check data directory.")
        return

    # Collect T values at each month-end
    t_matrix = trend_df.loc[valid_months]  # shape: (months, 16)

    # 1. T value distribution
    t_flat = t_matrix.values.flatten()
    t_flat = t_flat[~np.isnan(t_flat)]
    n_total = len(t_flat)
    up_count   = (t_flat == _SIGNAL_UP).sum()
    down_count = (t_flat == _SIGNAL_DOWN).sum()
    other_count = n_total - up_count - down_count

    print(f"  T distribution (asset×month, n={n_total}):")
    print(f"    T={_SIGNAL_UP:+.0f}:  {up_count:5d} ({100*up_count/n_total:5.1f}%)  — bullish")
    if other_count > 0:
        print(f"    T= 0:  {other_count:5d} ({100*other_count/n_total:5.1f}%)  — neutral")
    print(f"    T={_SIGNAL_DOWN:+.0f}:  {down_count:5d} ({100*down_count/n_total:5.1f}%)  — bearish")
    print()

    # 2. Months where ALL 16 main assets have identical T
    n_months = len(valid_months)
    identical_t_count = 0
    for d in valid_months:
        row = t_matrix.loc[d].values
        if len(set(row[~np.isnan(row)])) == 1:
            identical_t_count += 1
    print(f"  Months where all {n} main assets have identical T:")
    print(f"    {identical_t_count}/{n_months} ({100*identical_t_count/n_months:.1f}%)"
          " → zero cross-sectional ranking effect those months")
    print()

    # 3 & 4. MVC spread vs T spread per month
    mvc_spreads = []
    t_spreads   = []

    for d in valid_months:
        m_row = mom_df.loc[d]
        v_row = vol_df.loc[d]
        c_row = corr_df.loc[d]
        t_row = t_matrix.loc[d]

        rM = rank_assets(m_row, ascending=True)
        rV = rank_assets(v_row, ascending=False)
        rC = rank_assets(c_row, ascending=False)

        mvc = cfg.weight_momentum * rM + cfg.weight_volatility * rV + cfg.weight_correlation * rC
        mvc_spreads.append(mvc.max() - mvc.min())

        wt_vals = cfg.weight_trend * t_row
        t_spreads.append(wt_vals.max() - wt_vals.min())

    mvc_spreads = np.array(mvc_spreads)
    t_spreads   = np.array(t_spreads)

    median_mvc = np.nanmedian(mvc_spreads)
    median_t   = np.nanmedian(t_spreads)
    # Avoid divide-by-zero for months where all T are the same (t_spread = 0)
    nonzero_t  = t_spreads[t_spreads > 0]
    pct_of_mvc = 100 * median_t / median_mvc if median_mvc > 0 else 0.0

    print(f"  Cross-sectional spreads (median over {n_months} valid months):")
    print(f"    MVC spread  (wM*rM + wV*rV + wC*rC)  : {median_mvc:.3f}")
    print(f"    T spread    (wT * T)                   : {median_t:.3f}")
    print(f"    T as %% of MVC spread                  : {pct_of_mvc:.1f}%%")
    print()

    # 5. % months where T changes top-6 selection
    top_n = cfg.main_sleeve_top_n
    changed_count = 0

    for d in valid_months:
        m_row = mom_df.loc[d]
        v_row = vol_df.loc[d]
        c_row = corr_df.loc[d]
        t_row = t_matrix.loc[d]

        rM = rank_assets(m_row, ascending=True)
        rV = rank_assets(v_row, ascending=False)
        rC = rank_assets(c_row, ascending=False)

        # With T
        trank_with = compute_trank(rM, rV, rC, t_row, m_row, cfg)
        # Without T (wT set to 0 via a zero-T series)
        zero_t = pd.Series(0.0, index=t_row.index)
        cfg_no_t = replace(cfg, weight_trend=0.0)
        trank_no  = compute_trank(rM, rV, rC, zero_t, m_row, cfg_no_t)

        try:
            top_with = set(select_top_n(trank_with, top_n))
            top_no   = set(select_top_n(trank_no,   top_n))
        except ValueError:
            continue

        if top_with != top_no:
            changed_count += 1

    print(f"  Months where trend signal changes top-{top_n} selection:")
    print(f"    {changed_count}/{n_months} ({100*changed_count/n_months:.1f}%%)")
    print()

    # 6. Assets most often bearish
    bearish_counts = {}
    for ticker in main_tickers:
        col = t_matrix[ticker]
        bearish_counts[ticker] = (col == _SIGNAL_DOWN).sum()

    sorted_bearish = sorted(bearish_counts.items(), key=lambda x: -x[1])
    print(f"  Most common assets with bearish T (T={_SIGNAL_DOWN:+.0f}):")
    for ticker, cnt in sorted_bearish[:8]:
        pct = 100 * cnt / n_months
        print(f"    {ticker:<6}: {cnt:3d} months ({pct:.0f}%)")
    print()


# ── Part 3 : wT sensitivity backtest ──────────────────────────────────────────

def run_weight_test(
    data_dict: dict,
    base_cfg: ModelConfig,
) -> None:
    """Run wT = 1.0 / 2.0 / 3.0 and compare all metrics."""

    configs = {
        "A Base": replace(base_cfg, weight_trend=1.0),
        "B":      replace(base_cfg, weight_trend=2.0),
        "C":      replace(base_cfg, weight_trend=3.0),
    }

    results = {}
    for label, cfg in configs.items():
        print(f"  Running backtest for {label} (wT={cfg.weight_trend:.1f})…", flush=True)
        res = run_backtest(data_dict, cfg)
        results[label] = res

    print()
    print("-" * 70)
    print("WEIGHT TEST (wT = 1 / 2 / 3):")
    print()

    # Full-period metrics
    print("Full Period:")
    hdr = f"{'Config':<10} {'wT':>4}  {'Ann.Ret':>8} {'Ann.Vol':>8} {'Sharpe':>8} {'MDD':>9}"
    print(hdr)
    print("-" * len(hdr))

    wt_values = {"A Base": 1.0, "B": 2.0, "C": 3.0}
    full_metrics = {}
    for label, res in results.items():
        m = _metrics(res.monthly_returns)
        full_metrics[label] = m
        print(
            f"{label:<10} {wt_values[label]:>4.1f}  "
            f"{100*m['ann_ret']:>7.2f}%  "
            f"{100*m['ann_vol']:>7.2f}%  "
            f"{m['sharpe']:>8.3f}  "
            f"{100*m['mdd']:>8.2f}%"
        )
    print()

    # IS / OOS split
    print("IS / OOS:")
    hdr2 = (f"{'Config':<10} {'wT':>4}  {'IS Sharpe':>10} {'OOS Sharpe':>11}"
            f"  {'IS MDD':>8} {'OOS MDD':>8}  {'ΔSharpe(OOS)':>13}")
    print(hdr2)
    print("-" * len(hdr2))

    base_oos_sharpe = None
    for label, res in results.items():
        is_m  = _metrics(_slice_rets(res.monthly_returns, IS_START, IS_END))
        oos_m = _metrics(_slice_rets(res.monthly_returns, OOS_START, OOS_END))
        if label == "A Base":
            base_oos_sharpe = oos_m["sharpe"]
            delta_str = "—"
        else:
            delta = oos_m["sharpe"] - base_oos_sharpe
            delta_str = f"{delta:+.3f}"
        print(
            f"{label:<10} {wt_values[label]:>4.1f}  "
            f"{is_m['sharpe']:>10.3f} {oos_m['sharpe']:>11.3f}  "
            f"{100*is_m['mdd']:>7.2f}%  {100*oos_m['mdd']:>7.2f}%  "
            f"{delta_str:>13}"
        )
    print()

    # Annual returns table
    print("Annual Returns:")
    all_years = sorted(
        set(
            y
            for res in results.values()
            for y in res.monthly_returns.index.year.unique()
        )
    )
    hdr3 = f"{'Year':<6}" + "".join(f"  {label:<10}" for label in results)
    print(hdr3)
    print("-" * len(hdr3))

    for year in all_years:
        row = f"{year:<6}"
        for label, res in results.items():
            yr_rets = res.monthly_returns[res.monthly_returns.index.year == year]
            if yr_rets.empty:
                yr_cum = float("nan")
            else:
                yr_cum = (1 + yr_rets).prod() - 1
            row += f"  {100*yr_cum:>9.2f}%"
        print(row)
    print()

    # Stress periods
    print("Stress Periods:")
    hdr4 = f"{'Period':<20}" + "".join(f"  {label:<10}" for label in results)
    print(hdr4)
    print("-" * len(hdr4))

    for period_name, (s_start, s_end) in STRESS_PERIODS.items():
        row = f"{period_name:<20}"
        for label, res in results.items():
            cum = _stress_return(res.monthly_returns, s_start, s_end)
            row += f"  {100*cum:>9.2f}%"
        print(row)
    print()


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    # Load data
    print(f"Loading data from {DATA_DIR} …", flush=True)
    data_dict = load_validated_data(DATA_DIR)
    print(f"  Loaded {len(data_dict)} tickers.", flush=True)
    print()

    cfg = ModelConfig()

    # Part 1
    print_code_review(cfg, n_main=16)

    # Part 2
    run_empirical_analysis(data_dict, cfg)

    # Part 3
    run_weight_test(data_dict, cfg)

    print("=" * 70)
    print("Analysis complete.")


if __name__ == "__main__":
    main()
