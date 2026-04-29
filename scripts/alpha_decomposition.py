"""
Proper alpha decomposition for AMAAM.

Three approaches, increasing in rigour:

  1. Multi-factor OLS regression
     r_AMAAM = α + β_eq·r_SPY + β_bond·r_IEF + β_cmdty·r_DBC + β_gold·r_GLD + ε
     Controlling for the four main systematic risk premia the model harvests,
     the residual α is the skill contribution that can't be explained by passive
     factor exposure.

  2. Information Ratio vs conventional benchmarks
     IR = (r_AMAAM - r_bench) / σ(r_AMAAM - r_bench)
     Benchmarks: SPY buy-and-hold, 60/40 SPY+AGG, passive 7Twelve.

  3. Information Ratio vs 1/N equal-weight across the same 22-ETF universe
     The strictest test: does dynamic allocation beat holding the entire
     investable universe equally? Zero skill is required to achieve 1/N.

All results reported for Full, IS (2004–2018), OOS (2018–2024), and Holdout
(2024–2026) windows to show how alpha evolves across time.

Usage
-----
    python3.13 scripts/alpha_decomposition.py
    python3.13 scripts/alpha_decomposition.py --data-dir data/processed
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import logging
import numpy as np
import pandas as pd
import statsmodels.api as sm

from config.default_config import ModelConfig
from config.etf_universe import MAIN_SLEEVE_TICKERS, HEDGING_SLEEVE_TICKERS
from src.backtest.benchmarks import (
    _build_close_matrix,
    _monthly_rebalanced_returns,
    compute_sixty_forty,
    compute_seven_twelve,
    compute_spy_benchmark,
)
from src.backtest.engine import run_backtest
from src.data.loader import load_validated_data

logging.basicConfig(level=logging.WARNING,
                    format="%(asctime)s  %(levelname)-8s  %(message)s")

RF = 0.02 / 12   # monthly risk-free rate

# End date is read from cfg.backtest_end inside main(); these are filled at runtime.
WINDOWS_TEMPLATE = [
    ("Full    (2004–2026)", "2004-01-01", None),
    ("IS      (2004–2018)", "2004-01-01", "2018-01-01"),
    ("OOS     (2018–2024)", "2018-01-01", "2024-01-01"),
    ("Holdout (2024–2026)", "2024-01-01", None),
]


# ── Helpers ──────────────────────────────────────────────────────────────────

def _slice(s: pd.Series, start: str, end: str) -> pd.Series:
    return s[(s.index >= start) & (s.index < end)]


def _ann_ret(r: pd.Series) -> float:
    if len(r) < 2:
        return float("nan")
    return (1 + r).prod() ** (12 / len(r)) - 1


def _ann_vol(r: pd.Series) -> float:
    return r.std() * np.sqrt(12)


def _sharpe(r: pd.Series, rf: float = RF * 12) -> float:
    ret = _ann_ret(r)
    vol = _ann_vol(r)
    return (ret - rf) / vol if vol > 0 else float("nan")


def _ir(active: pd.Series) -> float:
    """Annualised Information Ratio from a series of active (excess) returns."""
    if len(active) < 6:
        return float("nan")
    ann_excess = active.mean() * 12
    te = active.std() * np.sqrt(12)
    return ann_excess / te if te > 0 else float("nan")


def _ols_alpha(y: pd.Series, X: pd.DataFrame) -> dict:
    """
    OLS regression of y on X (factors).  Returns annualised alpha, factor
    betas, R², and t/p-stats for alpha.
    """
    aligned = pd.concat([y, X], axis=1).dropna()
    if len(aligned) < 12:
        return dict(alpha=float("nan"), betas={}, r2=float("nan"),
                    t_alpha=float("nan"), p_alpha=float("nan"), n=0)
    y_ = aligned.iloc[:, 0]
    X_ = sm.add_constant(aligned.iloc[:, 1:])
    res = sm.OLS(y_, X_).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
    betas = {col: res.params[col] for col in aligned.columns[1:]}
    return dict(
        alpha=res.params["const"] * 12,    # annualise monthly alpha
        betas=betas,
        r2=res.rsquared,
        t_alpha=res.tvalues["const"],
        p_alpha=res.pvalues["const"],
        n=len(aligned),
    )


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/processed")
    args = parser.parse_args()

    print("Loading data and running backtest…")
    data = load_validated_data(args.data_dir)
    cfg  = ModelConfig()
    result = run_backtest(data, cfg)
    amaam = result.monthly_returns

    start_full = cfg.backtest_start
    end_full   = cfg.backtest_end

    # Resolve WINDOWS_TEMPLATE using live config end date.
    WINDOWS = [(lbl, ws, we if we is not None else end_full)
               for lbl, ws, we in WINDOWS_TEMPLATE]

    # ── Factor returns (monthly, from close prices) ───────────────────────────
    # Use the same one-day execution lag as AMAAM so dates align.
    factor_tickers = ["SPY", "IEF", "DBC", "GLD"]
    closes_factors = _build_close_matrix(data, factor_tickers)
    factors_monthly = {}
    for t in factor_tickers:
        s = _monthly_rebalanced_returns(
            closes_factors[[t]], {t: 1.0}, start_full, end_full
        )
        factors_monthly[t] = s
    factors_df = pd.DataFrame(factors_monthly)

    # ── Benchmark returns ─────────────────────────────────────────────────────
    bench_60_40  = compute_sixty_forty(data, start_full, end_full)
    bench_7twelve = compute_seven_twelve(data, start_full, end_full)
    bench_spy    = compute_spy_benchmark(data, start_full, end_full)

    # 1/N equal-weight across all 22 sleeve ETFs (the strictest baseline —
    # the model's own investable universe at zero-skill allocation).
    all_sleeve = MAIN_SLEEVE_TICKERS + HEDGING_SLEEVE_TICKERS
    n_sleeve   = len(all_sleeve)
    closes_sleeve = _build_close_matrix(data, all_sleeve)
    bench_1n = _monthly_rebalanced_returns(
        closes_sleeve,
        {t: 1.0 / n_sleeve for t in all_sleeve},
        start_full, end_full,
    )

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 1: Multi-factor OLS regression
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 76)
    print("SECTION 1 — MULTI-FACTOR OLS REGRESSION")
    print("  r_AMAAM = α + β_eq·SPY + β_bond·IEF + β_cmdty·DBC + β_gold·GLD + ε")
    print("  HAC standard errors (Newey-West, 3 lags).  α annualised.")
    print("=" * 76)

    for label, ws, we in WINDOWS:
        y = _slice(amaam, ws, we)
        X = factors_df.loc[y.index] if not y.empty else pd.DataFrame()
        r = _ols_alpha(y, X)
        if r["n"] == 0:
            print(f"\n  {label}: insufficient data")
            continue

        print(f"\n  {label}  (n={r['n']} months)")
        print(f"    α (annualised) = {r['alpha'] * 100:+.2f}%   "
              f"t = {r['t_alpha']:+.2f}   p = {r['p_alpha']:.4f}   "
              f"{'***' if r['p_alpha'] < 0.01 else '**' if r['p_alpha'] < 0.05 else '*' if r['p_alpha'] < 0.10 else ''}")
        print(f"    R² = {r['r2']:.3f}")
        for factor, beta in r["betas"].items():
            print(f"    β_{factor:<5} = {beta:+.3f}")

    print("\n  Interpretation: α here is the return unexplained after removing")
    print("  equity, bond, commodity, and gold systematic exposures.")
    print("  A small or zero α means the model earns fair compensation for")
    print("  the risk premia it harvests, not undiscovered edge.")

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 2: Information Ratio vs conventional benchmarks
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 76)
    print("SECTION 2 — INFORMATION RATIO vs CONVENTIONAL BENCHMARKS")
    print("  IR = annualised active return / tracking error")
    print("=" * 76)

    benchmarks = {
        "SPY buy-and-hold": bench_spy,
        "60/40 SPY+AGG    ": bench_60_40,
        "Passive 7Twelve  ": bench_7twelve,
    }

    header = f"  {'Window':<24}" + "".join(f"  {b:<18}" for b in benchmarks)
    print(f"\n  {'Window':<24}" +
          "".join(f"  {'Excess Ret / IR':<18}" for _ in benchmarks))
    sub = f"  {'':<24}" + "".join(f"  {b}" for b in benchmarks)
    print(sub)
    print("  " + "-" * (24 + 20 * len(benchmarks)))

    for label, ws, we in WINDOWS:
        a = _slice(amaam, ws, we)
        row = f"  {label:<24}"
        for bname, brets in benchmarks.items():
            b = _slice(brets, ws, we)
            common = a.index.intersection(b.index)
            if len(common) < 6:
                row += f"  {'N/A':<18}"
                continue
            active = a.loc[common] - b.loc[common]
            exc    = active.mean() * 12
            ir     = _ir(active)
            row += f"  {exc * 100:+.2f}% / {ir:+.2f}    "
        print(row)

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 3: IR vs 1/N equal-weight across the same 22-ETF universe
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 76)
    print("SECTION 3 — IR vs 1/N EQUAL-WEIGHT (same 22-ETF universe)")
    print("  The strictest test: does dynamic TRank allocation beat simply")
    print("  holding all 22 sleeve ETFs at 1/22 weight, rebalanced monthly?")
    print("=" * 76)

    for label, ws, we in WINDOWS:
        a = _slice(amaam, ws, we)
        b = _slice(bench_1n, ws, we)
        common = a.index.intersection(b.index)
        if len(common) < 6:
            print(f"\n  {label}: insufficient data")
            continue
        active = a.loc[common] - b.loc[common]
        exc    = active.mean() * 12
        ir     = _ir(active)
        a_sr   = _sharpe(a.loc[common])
        b_sr   = _sharpe(b.loc[common])
        a_ret  = _ann_ret(a.loc[common])
        b_ret  = _ann_ret(b.loc[common])
        a_dd   = (((1 + a.loc[common]).cumprod()) /
                  ((1 + a.loc[common]).cumprod().cummax()) - 1).min()
        b_dd   = (((1 + b.loc[common]).cumprod()) /
                  ((1 + b.loc[common]).cumprod().cummax()) - 1).min()
        print(f"\n  {label}  (n={len(common)} months)")
        print(f"    AMAAM    — ret={a_ret * 100:+.1f}%  SR={a_sr:.3f}  MaxDD={a_dd * 100:.1f}%")
        print(f"    1/N      — ret={b_ret * 100:+.1f}%  SR={b_sr:.3f}  MaxDD={b_dd * 100:.1f}%")
        print(f"    Active return={exc * 100:+.2f}%/yr   IR={ir:+.2f}   "
              f"ΔSharpe={a_sr - b_sr:+.3f}   ΔMDD={a_dd * 100 - b_dd * 100:+.1f}pp")

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 4: Summary table
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 76)
    print("SECTION 4 — SUMMARY: AMAAM vs ALL BENCHMARKS (full period)")
    print("=" * 76)

    ws, we = start_full, end_full
    a = _slice(amaam, ws, we)

    all_benches = {
        "SPY buy-and-hold": bench_spy,
        "60/40 SPY+AGG":    bench_60_40,
        "Passive 7Twelve":  bench_7twelve,
        f"1/N (22 ETFs)":   bench_1n,
    }

    print(f"\n  {'Benchmark':<22}  {'Bench Ret':>10}  {'Bench SR':>9}  "
          f"{'Excess Ret':>11}  {'Track Err':>10}  {'IR':>6}")
    print("  " + "-" * 72)
    for bname, brets in all_benches.items():
        b = _slice(brets, ws, we)
        common = a.index.intersection(b.index)
        if len(common) < 6:
            continue
        active = a.loc[common] - b.loc[common]
        exc    = active.mean() * 12
        te     = active.std() * np.sqrt(12)
        ir     = exc / te if te > 0 else float("nan")
        b_ret  = _ann_ret(b.loc[common])
        b_sr   = _sharpe(b.loc[common])
        print(f"  {bname:<22}  {b_ret * 100:>9.2f}%  {b_sr:>9.3f}  "
              f"{exc * 100:>+10.2f}%  {te * 100:>9.2f}%  {ir:>6.3f}")

    a_ret = _ann_ret(a)
    a_sr  = _sharpe(a)
    a_dd  = (((1 + a).cumprod()) / ((1 + a).cumprod().cummax()) - 1).min()
    print(f"\n  AMAAM (2004–2026): ret={a_ret * 100:.2f}%  SR={a_sr:.3f}  "
          f"MaxDD={a_dd * 100:.2f}%")

    print("\n  Note: positive IR vs 1/N confirms the dynamic allocation adds")
    print("  value over passive equal-weight within the same universe.")
    print("  This is the cleanest definition of skill for this model type.")


if __name__ == "__main__":
    main()
