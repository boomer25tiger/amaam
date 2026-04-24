"""
Expanded hedging sleeve + top-3 selection test.

Tests four configurations on an aligned window (IS 2012-2018, OOS 2018-2026):
  A: Current 6-asset hedging sleeve, top-2 (baseline)
  B: Current 6-asset hedging sleeve, top-3 (isolates selection slot effect)
  C: All 14-asset hedging sleeve, top-2 (isolates wider universe effect)
  D: All 14-asset hedging sleeve, top-3 (combination)

Binding data constraint: WTMF (~Jan 2011), SJB (~Mar 2011) → aligned window
starts 2012-01-01 for a clean full year of warm-up after last inception.

All four configs run on identical data slices for apples-to-apples comparison.
"""

import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

# ---------------------------------------------------------------------------
# Path setup — ensure project root is importable regardless of cwd
# ---------------------------------------------------------------------------
ROOT = Path("/Users/GualyCr/Desktop/AMAAM/amaam/.claude/worktrees/reverent-mayer")
sys.path.insert(0, str(ROOT))

from config.default_config import ModelConfig
from config import etf_universe as universe_mod
import src.backtest.engine as engine_mod
from src.backtest.engine import run_backtest
from src.data.loader import load_validated_data

logging.basicConfig(level=logging.WARNING)  # suppress verbose factor logs

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_DIR = ROOT / "data" / "processed"

CURRENT_SLEEVE = ["GLD", "TLT", "IEF", "SH", "UUP", "SHY"]
EXTRA_TICKERS  = ["TIP", "WTMF", "TBF", "PST", "BIL", "EUM", "EFZ", "SJB"]
EXPANDED_SLEEVE = CURRENT_SLEEVE + EXTRA_TICKERS  # 14 assets

IS_START  = "2012-01-01"
IS_END    = "2018-01-01"
OOS_START = "2018-01-01"
OOS_END   = "2026-04-23"

STRESS_PERIODS = {
    "HY Stress 2015-16": ("2015-08-01", "2016-02-29"),
    "Volmageddon Feb 2018": ("2018-02-01", "2018-02-28"),
    "COVID 2020": ("2020-02-01", "2020-03-31"),
    "2022 Rate Shock": ("2022-01-01", "2022-12-31"),
}

# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _load_or_download(ticker: str, data_dir: Path) -> pd.DataFrame:
    """
    Load ticker CSV from data_dir; download via yfinance if not present.

    Returns an OHLCV DataFrame with DatetimeIndex. yfinance auto_adjust=True
    ensures prices are split/dividend adjusted, consistent with existing data.
    """
    csv_path = data_dir / f"{ticker}.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path, index_col="Date", parse_dates=True)
        print(f"  Loaded {ticker} from CSV ({len(df)} rows)")
        return df

    print(f"  Downloading {ticker} via yfinance…", end=" ", flush=True)
    raw = yf.download(
        ticker,
        start="2006-01-01",
        end=OOS_END,
        auto_adjust=True,
        progress=False,
    )
    if raw.empty:
        raise ValueError(f"yfinance returned empty DataFrame for {ticker}")

    # yfinance multi-level columns when single ticker; flatten if needed
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.droplevel(1)

    # Standardise index name to match existing CSVs
    raw.index.name = "Date"
    raw = raw[["Open", "High", "Low", "Close", "Volume"]]
    raw.to_csv(csv_path)
    print(f"saved ({len(raw)} rows)")
    return raw


def _build_data_dict() -> dict:
    """
    Load all 14 hedging sleeve tickers + main sleeve + benchmarks.

    Existing data is loaded via load_validated_data (which picks up every CSV
    in the processed dir). Additional tickers not yet present are downloaded.
    """
    print("Loading base data from processed/...")
    data_dict = load_validated_data(DATA_DIR)

    print("Ensuring expanded sleeve tickers are present:")
    for ticker in EXPANDED_SLEEVE:
        if ticker not in data_dict:
            df = _load_or_download(ticker, DATA_DIR)
            data_dict[ticker] = df
        else:
            print(f"  {ticker} already in base data")

    return data_dict


# ---------------------------------------------------------------------------
# Performance metrics helpers
# ---------------------------------------------------------------------------

def _ann_metrics(monthly_rets: pd.Series) -> dict:
    """
    Compute annualised return, volatility, Sharpe (rf=2%), and max drawdown.

    Uses 12 periods/year (monthly returns). Risk-free rate: 2% annual = ~0.1667%/mo.
    """
    if monthly_rets.empty or len(monthly_rets) < 2:
        return {"ann_ret": np.nan, "ann_vol": np.nan, "sharpe": np.nan, "mdd": np.nan}

    rf_monthly = 0.02 / 12.0
    ann_ret = (1 + monthly_rets).prod() ** (12 / len(monthly_rets)) - 1
    ann_vol = monthly_rets.std() * np.sqrt(12)
    excess  = monthly_rets - rf_monthly
    sharpe  = (excess.mean() / excess.std()) * np.sqrt(12) if excess.std() > 0 else np.nan

    cum = (1 + monthly_rets).cumprod()
    rolling_max = cum.cummax()
    drawdown = (cum - rolling_max) / rolling_max
    mdd = drawdown.min()

    return {"ann_ret": ann_ret, "ann_vol": ann_vol, "sharpe": sharpe, "mdd": mdd}


def _stress_return(monthly_rets: pd.Series, start: str, end: str) -> float:
    """Cumulative return over a stress period; returns NaN if no data."""
    mask = (monthly_rets.index >= pd.Timestamp(start)) & (monthly_rets.index <= pd.Timestamp(end))
    sub = monthly_rets.loc[mask]
    if sub.empty:
        return np.nan
    return float((1 + sub).prod() - 1)


# ---------------------------------------------------------------------------
# Monkey-patch helpers
# ---------------------------------------------------------------------------

def _run_with_sleeve(
    data_dict: dict,
    new_hedging: list,
    top_n: int,
    is_start: str,
    is_end: str,
    oos_start: str,
    oos_end: str,
) -> dict:
    """
    Run the backtest with a patched hedging sleeve and selection count.

    Patches both config.etf_universe (the source of truth) AND
    src.backtest.engine (which binds the list at import time) so that the
    engine's run_backtest picks up the updated tickers.

    Returns dict with IS and OOS BacktestResult objects and their metrics.
    """
    # Snapshot originals for restoration in the finally block
    orig_hedging = universe_mod.HEDGING_SLEEVE_TICKERS[:]
    orig_all     = universe_mod.ALL_TICKERS[:]
    orig_engine_hedge = engine_mod.HEDGING_SLEEVE_TICKERS[:]
    orig_engine_main  = engine_mod.MAIN_SLEEVE_TICKERS[:]

    try:
        # Build new ALL_TICKERS: main + new hedging + benchmark-only
        new_all = sorted(
            set(universe_mod.MAIN_SLEEVE_TICKERS)
            | set(new_hedging)
            | set(["SPY", "AGG", "VV", "TIP", "IGOV"])
        )

        # Patch universe module (for any module that re-reads from it)
        universe_mod.HEDGING_SLEEVE_TICKERS[:] = new_hedging
        universe_mod.ALL_TICKERS[:] = new_all

        # Patch engine module bindings (imported names in engine namespace)
        engine_mod.HEDGING_SLEEVE_TICKERS[:] = new_hedging

        # Build config: full window so factors warm up properly, then we slice
        cfg = ModelConfig()
        cfg.hedging_sleeve_top_n = top_n
        cfg.backtest_start = "2010-01-01"   # warm-up before IS window
        cfg.backtest_end   = oos_end

        result = run_backtest(data_dict, cfg)
        mr = result.monthly_returns

        # Slice to IS and OOS windows
        is_mask  = (mr.index >= pd.Timestamp(is_start))  & (mr.index < pd.Timestamp(is_end))
        oos_mask = (mr.index >= pd.Timestamp(oos_start)) & (mr.index <= pd.Timestamp(oos_end))

        return {
            "is_ret":  mr.loc[is_mask],
            "oos_ret": mr.loc[oos_mask],
            "full_ret": mr,
        }

    finally:
        # Always restore originals
        universe_mod.HEDGING_SLEEVE_TICKERS[:] = orig_hedging
        universe_mod.ALL_TICKERS[:] = orig_all
        engine_mod.HEDGING_SLEEVE_TICKERS[:] = orig_engine_hedge
        engine_mod.MAIN_SLEEVE_TICKERS[:] = orig_engine_main


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt_row(label: str, assets: int, top_n: int, m: dict, delta_sharpe=None) -> str:
    ds = "   —   " if delta_sharpe is None else f"{delta_sharpe:+.3f}"
    return (
        f"  {label:<20s}  {assets:>3d}   {top_n:>1d}   "
        f"{m['ann_ret']:>7.2%}  {m['ann_vol']:>7.2%}  {m['sharpe']:>7.3f}  "
        f"{m['mdd']:>8.2%}    {ds}"
    )


def _fmt_stress_row(label: str, values: list) -> str:
    vals = "  ".join(f"{v:>8.2%}" if not np.isnan(v) else f"{'N/A':>8s}" for v in values)
    return f"  {label:<25s}  {vals}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("\n=== EXPANDED SLEEVE + TOP-3 TEST ===")
    print(f"Aligned window: IS {IS_START[:4]}–{IS_END[:4]}, OOS {OOS_START[:4]}–{OOS_END[:4]}")
    print()

    # ---- Data ---------------------------------------------------------------
    data_dict = _build_data_dict()

    # ---- Align all 14 hedging sleeve tickers + main sleeve to common calendar
    # We run each config with the full available data_dict; the engine filters
    # to whichever tickers are in the patched sleeve lists.
    print("\nRunning four configs (this takes ~1–2 min each)...")

    configs = [
        ("A", "Current/Top-2",  CURRENT_SLEEVE,  2),
        ("B", "Current/Top-3",  CURRENT_SLEEVE,  3),
        ("C", "Expanded/Top-2", EXPANDED_SLEEVE, 2),
        ("D", "Expanded/Top-3", EXPANDED_SLEEVE, 3),
    ]

    results = {}
    for cfg_id, label, sleeve, top_n in configs:
        print(f"  Running config {cfg_id}: {label} ({len(sleeve)} assets, top-{top_n})...")
        res = _run_with_sleeve(
            data_dict,
            new_hedging=sleeve,
            top_n=top_n,
            is_start=IS_START,
            is_end=IS_END,
            oos_start=OOS_START,
            oos_end=OOS_END,
        )
        results[cfg_id] = {"label": label, "assets": len(sleeve), "top_n": top_n, **res}
        is_m  = _ann_metrics(res["is_ret"])
        oos_m = _ann_metrics(res["oos_ret"])
        print(f"    IS  Sharpe={is_m['sharpe']:.3f}  Ann.Ret={is_m['ann_ret']:.2%}")
        print(f"    OOS Sharpe={oos_m['sharpe']:.3f}  Ann.Ret={oos_m['ann_ret']:.2%}")

    # ---- Print output table -------------------------------------------------
    hdr = (
        f"  {'Config':<20s}  {'Ast':>3s}  {'N':>1s}   "
        f"{'Ann.Ret':>7s}  {'Ann.Vol':>7s}  {'Sharpe':>7s}  "
        f"{'MDD':>8s}    {'ΔSharpe':>7s}"
    )
    sep = "  " + "-" * 80

    base_is_sharpe  = _ann_metrics(results["A"]["is_ret"])["sharpe"]
    base_oos_sharpe = _ann_metrics(results["A"]["oos_ret"])["sharpe"]

    print("\n" + "─" * 86)
    print(f"IS ({IS_START[:7]} – {IS_END[:7]}):")
    print(hdr)
    print(sep)
    for cfg_id, label, sleeve, top_n in configs:
        r  = results[cfg_id]
        m  = _ann_metrics(r["is_ret"])
        ds = None if cfg_id == "A" else m["sharpe"] - base_is_sharpe
        print(_fmt_row(f"{cfg_id}  {r['label']}", r["assets"], r["top_n"], m, ds))

    print()
    print(f"OOS ({OOS_START[:7]} – {OOS_END[:7]}):")
    print(hdr)
    print(sep)
    for cfg_id, label, sleeve, top_n in configs:
        r  = results[cfg_id]
        m  = _ann_metrics(r["oos_ret"])
        ds = None if cfg_id == "A" else m["sharpe"] - base_oos_sharpe
        print(_fmt_row(f"{cfg_id}  {r['label']}", r["assets"], r["top_n"], m, ds))

    # ---- Stress periods -----------------------------------------------------
    print()
    print("Stress Periods (cumulative return):")
    stress_hdr = (
        f"  {'Period':<25s}  {'A(6/2)':>8s}  {'B(6/3)':>8s}  "
        f"{'C(14/2)':>8s}  {'D(14/3)':>8s}"
    )
    print(stress_hdr)
    print("  " + "-" * 70)

    for period_label, (s_start, s_end) in STRESS_PERIODS.items():
        vals = [
            _stress_return(results["A"]["full_ret"], s_start, s_end),
            _stress_return(results["B"]["full_ret"], s_start, s_end),
            _stress_return(results["C"]["full_ret"], s_start, s_end),
            _stress_return(results["D"]["full_ret"], s_start, s_end),
        ]
        print(_fmt_stress_row(period_label, vals))

    print("\n" + "─" * 86)


if __name__ == "__main__":
    main()
