"""
SJB Hedging Sleeve Test.

Tests whether adding SJB (ProShares Short High Yield ETF) to the hedging
sleeve improves model performance.  Runs two configurations over aligned
IS and OOS windows to eliminate period-selection bias:

  A: Baseline (GLD, TLT, IEF, SH, UUP, SHY — current 6-asset sleeve)
  B: +SJB (7-asset sleeve, still top-2 selected each month)

SJB inception: 2011-03-22.  After 126-day warm-up, aligned_start = 2012-01-01.
IS window : 2012-01-01 → 2018-01-01
OOS window: 2018-01-01 → 2026-04-23
"""

import copy
import sys
from math import sqrt
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config.default_config import ModelConfig
import config.etf_universe as universe_mod
from src.data.loader import load_validated_data
from src.backtest.engine import run_backtest, HEDGING_SLEEVE_TICKERS, MAIN_SLEEVE_TICKERS


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_DIR = ROOT / "data" / "processed"

ALIGNED_START = "2012-01-01"
IS_START  = "2012-01-01"
IS_END    = "2018-01-01"
OOS_START = "2018-01-01"
OOS_END   = "2026-04-23"

# Stress periods: (label, start, end)
STRESS_PERIODS = [
    ("Taper Tantrum 2013",  "2013-05-01", "2013-06-30"),
    ("HY Stress 2015-16",   "2015-08-01", "2016-02-29"),
    ("COVID Crash 2020",    "2020-02-01", "2020-03-31"),
    ("2022 Rate Shock",     "2022-01-01", "2022-12-31"),
]


# ---------------------------------------------------------------------------
# Performance metrics
# ---------------------------------------------------------------------------

def compute_metrics(monthly_returns: pd.Series) -> dict:
    """Compute annualised return, vol, Sharpe, and max drawdown from a return series."""
    if monthly_returns.empty:
        return {"ann_ret": float("nan"), "ann_vol": float("nan"),
                "sharpe": float("nan"), "mdd": float("nan")}

    ann_ret = (1 + monthly_returns.mean()) ** 12 - 1
    ann_vol = monthly_returns.std() * sqrt(12)
    sharpe  = ann_ret / ann_vol if ann_vol > 0 else float("nan")

    # Max drawdown from equity curve (cumulative product of 1 + r).
    equity = (1 + monthly_returns).cumprod()
    mdd = (equity / equity.cummax() - 1).min()

    return {"ann_ret": ann_ret, "ann_vol": ann_vol, "sharpe": sharpe, "mdd": mdd}


def stress_cumret(monthly_returns: pd.Series, start: str, end: str) -> float:
    """Cumulative return over a stress window (NaN if no data)."""
    mask = (monthly_returns.index >= pd.Timestamp(start)) & \
           (monthly_returns.index <= pd.Timestamp(end))
    sub = monthly_returns.loc[mask]
    if sub.empty:
        return float("nan")
    return (1 + sub).prod() - 1


# ---------------------------------------------------------------------------
# Run a single backtest and return monthly_returns for the given window
# ---------------------------------------------------------------------------

def run_window(data_dict: dict, config: ModelConfig, start: str, end: str) -> pd.Series:
    """Run backtest with given config and filter to [start, end] window."""
    cfg = copy.copy(config)
    # We set wide start/end so the engine accumulates enough warm-up history,
    # then filter the output to the aligned window.  Setting backtest_start to
    # ALIGNED_START minus warm-up (2011-01-01) lets all factor windows fill in.
    cfg.backtest_start = "2011-01-01"
    cfg.backtest_end   = end

    result = run_backtest(data_dict, cfg)

    # Filter monthly returns to the requested aligned window.
    mr = result.monthly_returns
    mr = mr.loc[(mr.index >= pd.Timestamp(start)) & (mr.index < pd.Timestamp(end))]
    return mr


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # ── Load data ──────────────────────────────────────────────────────────
    print("Loading data...", flush=True)
    data_dict = load_validated_data(DATA_DIR)

    # Verify SJB is present.
    if "SJB" not in data_dict:
        raise FileNotFoundError(
            "SJB.csv not found in data/processed/. "
            "Run: python3.13 -c \"import yfinance as yf; "
            "sjb=yf.download('SJB',start='2010-01-01',auto_adjust=True); "
            "sjb.columns=[c[0] for c in sjb.columns]; "
            "sjb.to_csv('data/processed/SJB.csv')\""
        )

    sjb_first = data_dict["SJB"]["Close"].dropna().index.min()
    print(f"SJB first date: {sjb_first.date()}", flush=True)

    # ── Config A: Baseline (no SJB) ────────────────────────────────────────
    # Universe_mod lists are module-level — patch them in-place for config B;
    # restore for config A.  Safest: always restore to baseline before each run.
    BASELINE_HEDGING = list(universe_mod.HEDGING_SLEEVE_TICKERS)

    def restore_universe():
        universe_mod.HEDGING_SLEEVE_TICKERS[:] = BASELINE_HEDGING

    def inject_sjb():
        if "SJB" not in universe_mod.HEDGING_SLEEVE_TICKERS:
            universe_mod.HEDGING_SLEEVE_TICKERS.append("SJB")

    # Config A — baseline.
    restore_universe()
    # Patch the engine module's imported list too (it imported the list object).
    import src.backtest.engine as engine_mod
    engine_mod.HEDGING_SLEEVE_TICKERS[:] = BASELINE_HEDGING

    config_a = ModelConfig()

    print("Running Config A (Baseline) IS...", flush=True)
    mr_a_is  = run_window(data_dict, config_a, IS_START, IS_END)
    print("Running Config A (Baseline) OOS...", flush=True)
    mr_a_oos = run_window(data_dict, config_a, OOS_START, OOS_END)

    # Config B — +SJB.
    inject_sjb()
    engine_mod.HEDGING_SLEEVE_TICKERS[:] = universe_mod.HEDGING_SLEEVE_TICKERS

    config_b = ModelConfig()

    print("Running Config B (+SJB) IS...", flush=True)
    mr_b_is  = run_window(data_dict, config_b, IS_START, IS_END)
    print("Running Config B (+SJB) OOS...", flush=True)
    mr_b_oos = run_window(data_dict, config_b, OOS_START, OOS_END)

    # ── Compute metrics ────────────────────────────────────────────────────
    ma_is  = compute_metrics(mr_a_is)
    mb_is  = compute_metrics(mr_b_is)
    ma_oos = compute_metrics(mr_a_oos)
    mb_oos = compute_metrics(mr_b_oos)

    # ── Print results ──────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("=== SJB HEDGING SLEEVE TEST ===")
    print("=" * 60)

    def fmt_row(label, m, delta_sharpe=None):
        ds = f"  {delta_sharpe:+.3f}" if delta_sharpe is not None else "    —   "
        return (
            f"{label:<10}  "
            f"{m['ann_ret']:>6.2%}  "
            f"{m['ann_vol']:>6.2%}  "
            f"{m['sharpe']:>6.3f}  "
            f"{m['mdd']:>8.2%}  "
            f"{ds}"
        )

    hdr = f"{'Config':<10}  {'Ann.Ret':>6}  {'Ann.Vol':>6}  {'Sharpe':>6}  {'MDD':>8}  {'ΔSharpe':>8}"

    print(f"\nIS ({IS_START} → {IS_END}, aligned window):")
    print(hdr)
    print(fmt_row("A Base",  ma_is))
    print(fmt_row("B +SJB",  mb_is,  delta_sharpe=mb_is["sharpe"]  - ma_is["sharpe"]))

    print(f"\nOOS ({OOS_START} → {OOS_END}):")
    print(hdr)
    print(fmt_row("A Base",  ma_oos))
    print(fmt_row("B +SJB",  mb_oos, delta_sharpe=mb_oos["sharpe"] - ma_oos["sharpe"]))

    print("\nStress Periods (cumulative return):")
    print(f"{'Period':<24}  {'A Base':>8}  {'B +SJB':>8}  {'Delta':>8}")
    print("-" * 56)
    for label, s_start, s_end in STRESS_PERIODS:
        ra = stress_cumret(mr_a_oos if s_start >= OOS_START else mr_a_is, s_start, s_end)
        rb = stress_cumret(mr_b_oos if s_start >= OOS_START else mr_b_is, s_start, s_end)
        # COVID and 2022 are in OOS; Taper Tantrum and HY Stress are in IS.
        # Determine correct series based on period start.
        if s_start >= OOS_START:
            ra = stress_cumret(mr_a_oos, s_start, s_end)
            rb = stress_cumret(mr_b_oos, s_start, s_end)
        else:
            ra = stress_cumret(mr_a_is, s_start, s_end)
            rb = stress_cumret(mr_b_is, s_start, s_end)

        if ra != ra or rb != rb:   # NaN check
            delta_str = "    N/A"
        else:
            delta_str = f"{rb - ra:>+8.2%}"

        a_str = f"{ra:>8.2%}" if ra == ra else "     N/A"
        b_str = f"{rb:>8.2%}" if rb == rb else "     N/A"
        print(f"{label:<24}  {a_str}  {b_str}  {delta_str}")

    print()
    print("Data points:")
    print(f"  IS  months A={len(mr_a_is)}, B={len(mr_b_is)}")
    print(f"  OOS months A={len(mr_a_oos)}, B={len(mr_b_oos)}")
    print()


if __name__ == "__main__":
    main()
