"""
test_sef_btal_top3.py
---------------------
Test adding SEF (ProShares Short Financials) and BTAL (AGFiQ Market Neutral
Anti-Beta) individually to the 6-asset hedging sleeve, with top-3 selection
instead of the baseline top-2.

Configs
-------
A: 6-asset sleeve (GLD, TLT, IEF, SH, UUP, SHY), top-2  — baseline
B: 7-asset sleeve (6 + SEF),  top-3
C: 7-asset sleeve (6 + BTAL), top-3

Aligned window (conservative, full warm-up after latest inception BTAL ~Sep 2011):
  IS : 2012-06-01 → 2018-01-01
  OOS: 2018-01-01 → 2026-04-23
"""

import sys
import warnings
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

REPO = Path("/Users/GualyCr/Desktop/AMAAM/amaam/.claude/worktrees/reverent-mayer")
sys.path.insert(0, str(REPO))

from config.default_config import ModelConfig
import config.etf_universe as universe_mod
import src.backtest.engine as engine_mod
from src.backtest.engine import run_backtest
from src.data.loader import load_validated_data

DATA_DIR = REPO / "data" / "processed"

# ---------------------------------------------------------------------------
# Step 1: Download SEF and BTAL if not present
# ---------------------------------------------------------------------------

for ticker in ["SEF", "BTAL"]:
    csv_path = DATA_DIR / f"{ticker}.csv"
    if not csv_path.exists():
        print(f"Downloading {ticker}…")
        df = yf.download(ticker, start="2008-01-01", end="2026-04-23", auto_adjust=True,
                         progress=False, multi_level_index=False)
        # Flatten any MultiIndex columns and ensure standard OHLCV layout
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.index.name = "Date"
        df = df[["Open", "High", "Low", "Close", "Volume"]]
        df.to_csv(csv_path)
        print(f"  Saved {len(df)} rows to {csv_path.name}")
    else:
        print(f"{ticker}.csv already present.")

# ---------------------------------------------------------------------------
# Step 2: Load all data
# ---------------------------------------------------------------------------

data_dict = load_validated_data(DATA_DIR)

# Report actual first dates for SEF and BTAL
for t in ["SEF", "BTAL"]:
    first = data_dict[t].index.min().date()
    last  = data_dict[t].index.max().date()
    print(f"{t}: first={first}, last={last}, rows={len(data_dict[t])}")

# ---------------------------------------------------------------------------
# Constants for aligned window
# ---------------------------------------------------------------------------

IS_START  = "2012-06-01"
IS_END    = "2018-01-01"
OOS_START = "2018-01-01"
OOS_END   = "2026-04-23"

BASE_SLEEVE = ["GLD", "TLT", "IEF", "SH", "UUP", "SHY"]

# ---------------------------------------------------------------------------
# Helper: compute performance metrics from monthly returns Series
# ---------------------------------------------------------------------------

def compute_metrics(monthly_rets: pd.Series) -> dict:
    """
    Compute annualised return, volatility, Sharpe ratio and max drawdown
    from a monthly returns Series.
    """
    r = monthly_rets.dropna()
    if len(r) == 0:
        return {"ann_ret": np.nan, "ann_vol": np.nan, "sharpe": np.nan, "mdd": np.nan}

    ann_ret = (1 + r.mean()) ** 12 - 1
    ann_vol = r.std() * sqrt(12)
    sharpe  = ann_ret / ann_vol if ann_vol > 0 else np.nan

    equity  = (1 + r).cumprod()
    mdd     = (equity / equity.cummax() - 1).min()

    return {"ann_ret": ann_ret, "ann_vol": ann_vol, "sharpe": sharpe, "mdd": mdd}


def period_cumret(monthly_rets: pd.Series, start: str, end: str) -> float:
    """Cumulative return over a sub-period from a monthly returns Series."""
    r = monthly_rets.loc[start:end].dropna()
    if len(r) == 0:
        return np.nan
    return (1 + r).prod() - 1


# ---------------------------------------------------------------------------
# Helper: run one backtest config and return filtered monthly returns
# ---------------------------------------------------------------------------

def run_config(
    sleeve_tickers: list,
    top_n: int,
    label: str,
) -> pd.Series:
    """
    Temporarily patch the engine's HEDGING_SLEEVE_TICKERS module-level name
    and universe_mod list, run the backtest, then restore.

    Returns the full monthly_returns Series (all dates produced by the engine).
    """
    # Save originals
    orig_engine_hedge   = engine_mod.HEDGING_SLEEVE_TICKERS
    orig_universe_hedge = universe_mod.HEDGING_SLEEVE_TICKERS

    try:
        # Patch both the engine module's cached reference and the universe list
        engine_mod.HEDGING_SLEEVE_TICKERS          = sleeve_tickers
        # universe_mod list is referenced via the module attribute; reassign
        universe_mod.HEDGING_SLEEVE_TICKERS        = sleeve_tickers

        cfg = ModelConfig()
        cfg.hedging_sleeve_top_n = top_n
        # Use the full backtest window so we get all dates; we will slice later
        cfg.backtest_start = "2004-01-01"
        cfg.backtest_end   = "2026-04-23"

        print(f"\nRunning config {label}: sleeve={sleeve_tickers}, top_n={top_n}…")
        result = run_backtest(data_dict, cfg)
        print(f"  Done. {len(result.monthly_returns)} monthly obs.")
        return result.monthly_returns

    finally:
        engine_mod.HEDGING_SLEEVE_TICKERS   = orig_engine_hedge
        universe_mod.HEDGING_SLEEVE_TICKERS = orig_universe_hedge


# ---------------------------------------------------------------------------
# Step 3: Run all three configs
# ---------------------------------------------------------------------------

rets_A = run_config(BASE_SLEEVE,             top_n=2, label="A (6/2 baseline)")
rets_B = run_config(BASE_SLEEVE + ["SEF"],   top_n=3, label="B (7/3 +SEF)")
rets_C = run_config(BASE_SLEEVE + ["BTAL"],  top_n=3, label="C (7/3 +BTAL)")

# ---------------------------------------------------------------------------
# Step 4: Compute metrics for IS and OOS windows
# ---------------------------------------------------------------------------

periods = {
    "IS  (2012–2018)": (IS_START,  IS_END),
    "OOS (2018–2026)": (OOS_START, OOS_END),
}

configs = [
    ("A", "Baseline (6/2)", "current",      2, rets_A),
    ("B", "+SEF     (7/3)", "current+SEF",  3, rets_B),
    ("C", "+BTAL    (7/3)", "current+BTAL", 3, rets_C),
]

# ---------------------------------------------------------------------------
# Step 5: Stress period definitions
# Dates reference the monthly_returns index (first trading day of each month)
# ---------------------------------------------------------------------------

stress_periods = [
    ("Eurozone stress 2012",  "2012-01-01", "2012-06-30"),
    ("HY Stress 2015-16",     "2015-08-01", "2016-02-29"),
    ("Volmageddon Feb 2018",  "2018-02-01", "2018-02-28"),
    ("COVID 2020",            "2020-02-01", "2020-03-31"),
    ("2022 Rate Shock",       "2022-01-01", "2022-12-31"),
    ("2025 Tariff Shock",     "2025-02-01", "2026-04-23"),
]

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

print("\n")
print("=" * 78)
print("=== SEF & BTAL TEST (7 assets, top-3 selection) ===")
print("Aligned window: IS 2012–2018, OOS 2018–2026")
print("=" * 78)

for period_label, (p_start, p_end) in periods.items():
    print(f"\n{period_label}:")
    header = (
        f"{'Config':<30}  {'Sleeve':<14}  {'TopN':>4}  "
        f"{'Ann.Ret':>8}  {'Ann.Vol':>8}  {'Sharpe':>7}  {'MDD':>8}  {'ΔSharpe':>8}"
    )
    print(header)
    print("-" * len(header))

    sharpe_A = None
    for cfg_label, cfg_name, sleeve_label, top_n, rets in configs:
        r_slice = rets.loc[p_start:p_end]
        m = compute_metrics(r_slice)

        if cfg_label == "A":
            sharpe_A = m["sharpe"]
            delta_s  = "—"
        else:
            diff = m["sharpe"] - sharpe_A if sharpe_A is not None else np.nan
            delta_s = f"{diff:+.3f}"

        row = (
            f"{cfg_label}  {cfg_name:<28}  {sleeve_label:<14}  {top_n:>4}  "
            f"{m['ann_ret']:>7.2%}  {m['ann_vol']:>7.2%}  {m['sharpe']:>7.3f}  "
            f"{m['mdd']:>7.2%}  {delta_s:>8}"
        )
        print(row)

print()
print("Stress Periods (cumulative return):")
stress_header = (
    f"{'Period':<28}  {'A(6/2)':>10}  {'B+SEF(7/3)':>12}  {'C+BTAL(7/3)':>13}"
)
print(stress_header)
print("-" * len(stress_header))

for sp_label, sp_start, sp_end in stress_periods:
    ra = period_cumret(rets_A, sp_start, sp_end)
    rb = period_cumret(rets_B, sp_start, sp_end)
    rc = period_cumret(rets_C, sp_start, sp_end)

    def fmt(v):
        return f"{v:.2%}" if not np.isnan(v) else "n/a"

    print(f"{sp_label:<28}  {fmt(ra):>10}  {fmt(rb):>12}  {fmt(rc):>13}")

print()
print("=" * 78)
