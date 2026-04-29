"""
Volatility Estimator Comparison: EWMA vs Garman-Klass vs Yang-Zhang

Tests three volatility estimators using the same IS/OOS framework as previous
wC sweep. Runs the full AMAAM backtest with each estimator and reports:
  - IS  (2007-08 → 2017-12): Annualised Return, Sharpe, Max Drawdown, Calmar
  - OOS (2018-01 → 2026-04): same metrics
  - Full period: same metrics

Yang-Zhang formula:
  σ²_YZ = σ²_overnight + k * σ²_open_close + (1-k) * σ²_rogers_satchell
  where:
    k = 0.34 / (1.34 + (n+1)/(n-1))   (Chou & Wang 2006 optimal k)
    σ²_overnight = Σ(ln(O/C_prev))²  / (n-1)
    σ²_rogers_satchell = Σ[ln(H/C)*ln(H/O) + ln(L/C)*ln(L/O)] / n
    σ²_open_close = Σ(ln(C/O))² / (n-1)
"""

import sys
import os
sys.path.insert(0, "/Users/GualyCr/Desktop/AMAAM/amaam/.claude/worktrees/reverent-mayer")

import numpy as np
import pandas as pd

from config.default_config import ModelConfig
from src.data.loader import load_validated_data
import src.backtest.engine as eng
from src.backtest.engine import run_backtest, _precompute_factors


# ---------------------------------------------------------------------------
# Garman-Klass estimator
# ---------------------------------------------------------------------------

def garman_klass_vol(data_dict: dict, window: int) -> pd.DataFrame:
    """
    Garman-Klass volatility using OHLC data.

    Formula: GK = 0.5*(ln(H/L))^2 - (2*ln(2)-1)*(ln(C/C_prev))^2
    Annualised: sqrt(GK_rolling_mean * 252)
    """
    series = {}
    for ticker, df in data_dict.items():
        h = np.log(df["High"] / df["Low"])
        c = np.log(df["Close"] / df["Close"].shift(1))
        gk = 0.5 * h**2 - (2 * np.log(2) - 1) * c**2
        series[ticker] = np.sqrt(gk.rolling(window, min_periods=window).mean() * 252)
    return pd.DataFrame(series)


# ---------------------------------------------------------------------------
# Yang-Zhang estimator
# ---------------------------------------------------------------------------

def yang_zhang_vol(data_dict: dict, window: int) -> pd.DataFrame:
    """
    Yang-Zhang volatility using OHLC data with overnight component.

    Decomposition:
      σ²_YZ = σ²_overnight + k*σ²_open_close + (1-k)*σ²_rogers_satchell

    where k is the Chou-Wang (2006) optimal weighting coefficient:
      k = 0.34 / (1.34 + (n+1)/(n-1))

    This estimator is:
    - ~8x more efficient than close-to-close
    - Drift-independent (unlike Garman-Klass)
    - Handles overnight gaps explicitly
    """
    k_coeff = 0.34 / (1.34 + (window + 1) / (window - 1))
    series = {}

    for ticker, df in data_dict.items():
        o = np.log(df["Open"])
        h = np.log(df["High"])
        l = np.log(df["Low"])
        c = np.log(df["Close"])
        c_prev = c.shift(1)
        o_prev = o.shift(1)  # not used but kept for clarity

        # Overnight component: ln(O_t / C_{t-1})
        overnight_ret = o - c_prev
        var_overnight = overnight_ret.rolling(window, min_periods=window).var(ddof=1)

        # Open-to-close component: ln(C_t / O_t)
        open_close_ret = c - o
        var_open_close = open_close_ret.rolling(window, min_periods=window).var(ddof=1)

        # Rogers-Satchell component: ln(H/C)*ln(H/O) + ln(L/C)*ln(L/O)
        rs = (h - c) * (h - o) + (l - c) * (l - o)
        var_rs = rs.rolling(window, min_periods=window).mean()

        var_yz = var_overnight + k_coeff * var_open_close + (1 - k_coeff) * var_rs
        series[ticker] = np.sqrt(var_yz.clip(lower=0) * 252)

    return pd.DataFrame(series)


# ---------------------------------------------------------------------------
# Backtest runner with injected vol estimator
# ---------------------------------------------------------------------------

def run_with_vol(data_dict: dict, vol_df: pd.DataFrame, label: str) -> dict:
    """Run AMAAM backtest using a pre-computed volatility DataFrame."""
    cfg = ModelConfig()
    original_precompute = eng._precompute_factors

    def patched_precompute(dd, main_t, hedge_t, config):
        factors = original_precompute(dd, main_t, hedge_t, config)
        # Inject the custom vol — only for tickers present in vol_df
        combined_vol = factors["volatility"].copy()
        for col in vol_df.columns:
            if col in combined_vol.columns:
                combined_vol[col] = vol_df[col]
        factors["volatility"] = combined_vol
        return factors

    eng._precompute_factors = patched_precompute
    try:
        result = run_backtest(data_dict, cfg)
    finally:
        eng._precompute_factors = original_precompute

    return result


# ---------------------------------------------------------------------------
# Metrics helper
# ---------------------------------------------------------------------------

def slice_metrics(result, start: str, end: str, label: str) -> dict:
    """Extract annualised metrics for a date slice."""
    ret = result.monthly_returns
    ret = ret[(ret.index >= start) & (ret.index <= end)]
    if len(ret) < 6:
        return {"label": label, "Return": float("nan"), "Sharpe": float("nan"),
                "MaxDD": float("nan"), "Calmar": float("nan"), "N": 0}

    ann_ret = (1 + ret).prod() ** (12 / len(ret)) - 1
    ann_vol = ret.std() * np.sqrt(12)
    sharpe = (ann_ret - 0.02) / ann_vol
    eq = (1 + ret).cumprod()
    drawdown = (eq / eq.cummax() - 1).min()
    calmar = ann_ret / abs(drawdown) if drawdown != 0 else float("nan")

    return {
        "label": label,
        "Return": round(ann_ret * 100, 2),
        "Sharpe": round(sharpe, 3),
        "MaxDD":  round(drawdown * 100, 2),
        "Calmar": round(calmar, 3),
        "N": len(ret),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading data…")
    data_dict = load_validated_data("data/processed")
    cfg = ModelConfig()

    IS_START  = cfg.backtest_start      # "2007-08-01"
    IS_END    = "2017-12-31"
    OOS_START = cfg.holdout_start       # "2018-01-01"
    OOS_END   = cfg.backtest_end        # "2026-03-31"
    FULL_START = cfg.backtest_start

    # Window matches momentum_lookback for consistency with prior GK-21 test
    VOL_WINDOW = cfg.momentum_lookback  # 84 days

    # Pre-compute all three vol surfaces once
    all_tickers = list(data_dict.keys())
    print(f"Computing Garman-Klass vol (window={VOL_WINDOW})…")
    gk_vol = garman_klass_vol(data_dict, VOL_WINDOW)

    print(f"Computing Yang-Zhang vol (window={VOL_WINDOW})…")
    yz_vol = yang_zhang_vol(data_dict, VOL_WINDOW)

    # Run backtests
    print("\nRunning EWMA baseline…")
    res_ewma = run_backtest(data_dict, cfg)

    print("Running Garman-Klass…")
    res_gk = run_with_vol(data_dict, gk_vol, "GK-84")

    print("Running Yang-Zhang…")
    res_yz = run_with_vol(data_dict, yz_vol, "YZ-84")

    # Collect results
    periods = [
        ("IS",   IS_START,   IS_END),
        ("OOS",  OOS_START,  OOS_END),
        ("FULL", FULL_START, OOS_END),
    ]

    results = {}
    for label, res in [("EWMA-baseline", res_ewma), ("GK-84", res_gk), ("YZ-84", res_yz)]:
        results[label] = {period: slice_metrics(res, s, e, period)
                          for period, s, e in periods}

    # Print table
    print("\n" + "=" * 85)
    print("VOLATILITY ESTIMATOR COMPARISON — AMAAM IS/OOS")
    print("=" * 85)
    print(f"{'Estimator':<18} {'Period':<6} {'Ann.Ret%':>9} {'Sharpe':>8} {'MaxDD%':>9} {'Calmar':>8} {'Months':>7}")
    print("-" * 85)

    for est_label in ["EWMA-baseline", "GK-84", "YZ-84"]:
        for period, _, _ in periods:
            m = results[est_label][period]
            print(f"{est_label:<18} {period:<6} {m['Return']:>9.2f} {m['Sharpe']:>8.3f} "
                  f"{m['MaxDD']:>9.2f} {m['Calmar']:>8.3f} {m['N']:>7}")
        print()

    # Summary: which estimator wins each period by Sharpe?
    print("=" * 85)
    print("SHARPE RANKING BY PERIOD")
    print("-" * 85)
    for period, _, _ in periods:
        ranked = sorted(results.keys(),
                        key=lambda k: results[k][period]["Sharpe"], reverse=True)
        print(f"  {period:<6}: " + " > ".join(
            f"{k} ({results[k][period]['Sharpe']:.3f})" for k in ranked
        ))

    # Year-by-year comparison for OOS
    print("\n" + "=" * 85)
    print("YEAR-BY-YEAR ANNUAL RETURNS (OOS only: 2018–2026)")
    print("-" * 85)
    print(f"{'Year':<6} {'EWMA':>10} {'GK-84':>10} {'YZ-84':>10}")
    print("-" * 85)

    for year in range(2018, 2027):
        row = [str(year)]
        for label, res in [("EWMA-baseline", res_ewma), ("GK-84", res_gk), ("YZ-84", res_yz)]:
            ret = res.monthly_returns
            yr = ret[ret.index.year == year]
            if len(yr) == 0:
                row.append("  —")
            else:
                ann = (1 + yr).prod() ** (12 / len(yr)) - 1
                row.append(f"{ann*100:+.1f}%")
        print(f"{row[0]:<6} {row[1]:>10} {row[2]:>10} {row[3]:>10}")


if __name__ == "__main__":
    main()
