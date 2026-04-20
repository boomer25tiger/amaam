"""
XLRE holding audit — scenario C (+XLRE in main sleeve).

For every month XLRE is held, shows:
  - Weight, XLRE's own return, what it displaced (the 6th-ranked main asset
    in the baseline that lost its slot), and whether XLRE beat SHY.

Also reports year-by-year selection count, average return, and
the baseline's displaced asset return for context.
"""

import sys
sys.path.insert(0, "/Users/GualyCr/Desktop/AMAAM/amaam/.claude/worktrees/reverent-mayer")

import numpy as np
import pandas as pd

from config.default_config import ModelConfig
import config.etf_universe as universe_mod
import src.backtest.engine as eng
from src.backtest.engine import run_backtest, _precompute_factors, _month_end_dates, _build_exec_date_map
from src.data.loader import load_validated_data
from src.ranking.trank import rank_assets, compute_trank, select_top_n


COMMON_START = "2007-08-01"


def trim(data_dict):
    return {
        t: df.loc[df.index >= COMMON_START] if df.index[0] < pd.Timestamp(COMMON_START) else df
        for t, df in data_dict.items()
    }


def run_with_xlre(data_dict):
    original = universe_mod.MAIN_SLEEVE_TICKERS[:]
    universe_mod.MAIN_SLEEVE_TICKERS = original + ["XLRE"]
    eng.MAIN_SLEEVE_TICKERS = universe_mod.MAIN_SLEEVE_TICKERS
    try:
        r = run_backtest(data_dict, ModelConfig())
    finally:
        universe_mod.MAIN_SLEEVE_TICKERS = original
        eng.MAIN_SLEEVE_TICKERS = original
    return r


def run_baseline(data_dict):
    return run_backtest(data_dict, ModelConfig())


def main():
    print("Loading and trimming data…")
    raw = load_validated_data("data/processed")
    data_dict = trim(raw)
    cfg = ModelConfig()

    print("Running baseline…")
    base = run_baseline(data_dict)

    print("Running +XLRE…")
    xlre = run_with_xlre(data_dict)

    closes = pd.DataFrame({t: data_dict[t]["Close"] for t in data_dict})
    exec_dates = xlre.monthly_returns.index.tolist()
    signal_dates = xlre.allocations.index.tolist()
    allocs_xlre = xlre.allocations
    allocs_base = base.allocations

    # ── Build month-level records ─────────────────────────────────────────────
    records = []
    for i, sig in enumerate(signal_dates):
        if i >= len(exec_dates) - 1:
            break
        xlre_w = allocs_xlre.at[sig, "XLRE"] if "XLRE" in allocs_xlre.columns else 0.0
        if xlre_w <= 0:
            continue

        e0, e1 = exec_dates[i], exec_dates[i + 1]

        def asset_ret(ticker):
            if ticker in closes.columns and e0 in closes.index and e1 in closes.index:
                p0, p1 = closes.at[e0, ticker], closes.at[e1, ticker]
                return (p1 / p0 - 1.0) if p0 > 0 else float("nan")
            return float("nan")

        xlre_r  = asset_ret("XLRE")
        shy_r   = asset_ret("SHY")

        # What was the 6th main-sleeve asset in the baseline that XLRE displaced?
        # Reconstruct baseline's top-6 for this signal date.
        base_top6 = []
        if sig in allocs_base.index:
            main_tickers = universe_mod.MAIN_SLEEVE_TICKERS  # original 16
            row = allocs_base.loc[sig]
            held = [t for t in main_tickers if row.get(t, 0) > 0]
            base_top6 = held

        displaced = base_top6[-1] if base_top6 else None
        displaced_r = asset_ret(displaced) if displaced else float("nan")

        records.append({
            "sig":        sig,
            "exec_from":  e0,
            "year":       e0.year,
            "weight":     xlre_w,
            "xlre_ret":   xlre_r,
            "shy_ret":    shy_r,
            "excess_shy": xlre_r - shy_r if not (np.isnan(xlre_r) or np.isnan(shy_r)) else float("nan"),
            "displaced":  displaced if displaced else "—",
            "disp_ret":   displaced_r,
            "xlre_vs_disp": xlre_r - displaced_r if not np.isnan(displaced_r) else float("nan"),
        })

    df = pd.DataFrame(records)
    total_months = len(exec_dates) - 1

    # ── Overall stats ─────────────────────────────────────────────────────────
    print(f"\nTotal backtest months: {total_months}")
    print(f"XLRE selected: {len(df)} months ({len(df)/total_months*100:.1f}%)")
    print(f"XLRE avg monthly return when held: {df['xlre_ret'].mean()*100:+.2f}%")
    print(f"XLRE win rate vs SHY: {(df['excess_shy']>0).sum()}/{len(df)} ({(df['excess_shy']>0).mean()*100:.0f}%)")
    print(f"XLRE avg excess vs SHY: {df['excess_shy'].mean()*100:+.2f}%")
    valid_disp = df.dropna(subset=["xlre_vs_disp"])
    print(f"XLRE vs displaced asset ({len(valid_disp)} months): {valid_disp['xlre_vs_disp'].mean()*100:+.2f}% avg")

    # ── Year-by-year ─────────────────────────────────────────────────────────
    print("\n" + "=" * 95)
    print("YEAR-BY-YEAR XLRE SELECTION STATS")
    print(f"{'Year':<6} {'Sel':>4} {'XLRE ret%':>10} {'vs SHY':>8} {'Win%':>6} {'MktCtx':>30}")
    print("-" * 95)

    contexts = {
        2016: "EM rebound, rate rise Dec",
        2017: "Low-vol bull market",
        2018: "Vol spike, rate fears",
        2019: "Rate cuts, partial recovery",
        2020: "COVID crash + recovery",
        2021: "Reopening boom",
        2022: "Rates shock, REIT −26%",
        2023: "Rate peak, tech rally",
        2024: "Soft landing, rate cuts",
        2025: "Tariff shock, recovery",
        2026: "YTD",
    }

    for year in range(2016, 2027):
        yr = df[df["year"] == year]
        if yr.empty:
            print(f"{year:<6} {'—':>4}")
            continue
        n     = len(yr)
        avg_r = yr["xlre_ret"].mean() * 100
        avg_e = yr["excess_shy"].mean() * 100
        wins  = (yr["excess_shy"] > 0).mean() * 100
        ctx   = contexts.get(year, "")
        print(f"{year:<6} {n:>4} {avg_r:>+10.2f}% {avg_e:>+8.2f}% {wins:>5.0f}%  {ctx}")

    # ── Month-by-month detail ─────────────────────────────────────────────────
    print("\n" + "=" * 105)
    print("MONTH-BY-MONTH DETAIL (all months XLRE held)")
    print(f"{'Period':<10} {'Wt':>5} {'XLRE':>8} {'SHY':>7} {'vsSHY':>7} {'Displaced':>10} {'DispRet':>8} {'XLREvDisp':>10} {'Beat?':>6}")
    print("-" * 105)

    for _, r in df.sort_values("exec_from").iterrows():
        period  = r["exec_from"].strftime("%Y-%m")
        beat    = "✓" if r["excess_shy"] > 0 else "✗"
        beat_d  = "↑" if r["xlre_vs_disp"] > 0 else ("↓" if not np.isnan(r["xlre_vs_disp"]) else " ")
        print(f"{period:<10} {r['weight']:>5.1%} {r['xlre_ret']*100:>+8.2f}% "
              f"{r['shy_ret']*100:>+7.2f}% {r['excess_shy']*100:>+7.2f}% "
              f"{r['displaced']:>10} {r['disp_ret']*100:>+8.2f}% "
              f"{r['xlre_vs_disp']*100:>+10.2f}% {beat:>4}{beat_d:>2}")

    # ── Displacement analysis ─────────────────────────────────────────────────
    print("\n" + "=" * 95)
    print("WHICH ASSETS DID XLRE DISPLACE? (from 6th slot in baseline top-6)")
    print("-" * 95)
    disp_counts = df["displaced"].value_counts()
    for asset, count in disp_counts.items():
        sub = df[df["displaced"] == asset].dropna(subset=["xlre_vs_disp"])
        avg_edge = sub["xlre_vs_disp"].mean() * 100 if not sub.empty else float("nan")
        print(f"  {asset:<8} displaced {count:>2}× — XLRE avg {avg_edge:+.2f}% vs that asset when displaced")

    # ── 2022 deep-dive (XLRE down 26%) ───────────────────────────────────────
    print("\n" + "=" * 95)
    print("2022 DETAIL (REIT was the worst sector that year)")
    print("-" * 95)
    df22 = df[df["year"] == 2022]
    if df22.empty:
        print("  XLRE NOT selected at all in 2022 — model correctly avoided it")
    else:
        print(f"  XLRE held {len(df22)} months in 2022:")
        for _, r in df22.sort_values("exec_from").iterrows():
            print(f"    {r['exec_from'].strftime('%Y-%m')}: XLRE {r['xlre_ret']*100:+.2f}%  "
                  f"displaced={r['displaced']} ({r['disp_ret']*100:+.2f}%)")


if __name__ == "__main__":
    main()
