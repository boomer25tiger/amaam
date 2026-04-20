"""
Full main-sleeve asset audit.

For every asset in the main sleeve, reports:
  - Selection frequency and average weight
  - Average monthly return when held
  - Win rate vs SHY (opportunity cost)
  - Average excess return vs SHY
  - Total gross contribution (weight × return, summed)
  - Year-by-year selection counts

Sorted by avg excess vs SHY descending so the best and worst are obvious.
"""

import sys
sys.path.insert(0, "/Users/GualyCr/Desktop/AMAAM/amaam/.claude/worktrees/reverent-mayer")

import numpy as np
import pandas as pd

from config.default_config import ModelConfig
from config.etf_universe import MAIN_SLEEVE_TICKERS, ETF_METADATA
from src.data.loader import load_validated_data
from src.backtest.engine import run_backtest


def main():
    print("Loading data and running backtest…")
    data_dict = load_validated_data("data/processed")
    cfg = ModelConfig()
    res = run_backtest(data_dict, cfg)

    allocs     = res.allocations.copy()
    allocs.index = pd.to_datetime(allocs.index)
    exec_dates = res.monthly_returns.index.tolist()
    signal_dates = allocs.index.tolist()
    total_months = len(exec_dates) - 1

    closes = pd.DataFrame({t: data_dict[t]["Close"] for t in data_dict})

    # ── Build per-month per-asset records ────────────────────────────────────
    records = []
    for i, sig in enumerate(signal_dates):
        if i >= len(exec_dates) - 1:
            break
        e0, e1 = exec_dates[i], exec_dates[i + 1]

        shy_ret = float("nan")
        if "SHY" in closes.columns and e0 in closes.index and e1 in closes.index:
            s0, s1 = closes.at[e0, "SHY"], closes.at[e1, "SHY"]
            if s0 > 0:
                shy_ret = s1 / s0 - 1

        for ticker in MAIN_SLEEVE_TICKERS:
            if ticker not in allocs.columns:
                continue
            weight = allocs.at[sig, ticker] if ticker in allocs.columns else 0.0
            if weight <= 0:
                continue

            asset_ret = float("nan")
            if ticker in closes.columns and e0 in closes.index and e1 in closes.index:
                p0, p1 = closes.at[e0, ticker], closes.at[e1, ticker]
                if p0 > 0:
                    asset_ret = p1 / p0 - 1

            records.append({
                "sig":        sig,
                "exec_from":  e0,
                "year":       e0.year,
                "ticker":     ticker,
                "weight":     weight,
                "asset_ret":  asset_ret,
                "shy_ret":    shy_ret,
                "excess":     asset_ret - shy_ret if not (np.isnan(asset_ret) or np.isnan(shy_ret)) else float("nan"),
                "contrib":    weight * asset_ret if not np.isnan(asset_ret) else float("nan"),
            })

    df = pd.DataFrame(records)

    # ── Aggregate stats per ticker ───────────────────────────────────────────
    stats = []
    for ticker in MAIN_SLEEVE_TICKERS:
        sub = df[df["ticker"] == ticker]
        if sub.empty:
            continue
        n       = len(sub)
        avg_ret = sub["asset_ret"].mean() * 100
        avg_exc = sub["excess"].mean() * 100
        win_pct = (sub["excess"] > 0).mean() * 100
        contrib = sub["contrib"].sum() * 100
        avg_w   = sub["weight"].mean()
        name    = ETF_METADATA.get(ticker, None)
        short   = name.asset_class if name else ""
        stats.append({
            "ticker":   ticker,
            "class":    short,
            "months":   n,
            "pct":      n / total_months * 100,
            "avg_w":    avg_w,
            "avg_ret":  avg_ret,
            "avg_exc":  avg_exc,
            "win_pct":  win_pct,
            "contrib":  contrib,
        })

    sdf = pd.DataFrame(stats).sort_values("avg_exc", ascending=False)

    # ── Print ranked table ───────────────────────────────────────────────────
    print(f"\nTotal backtest months: {total_months}")
    print("\n" + "=" * 100)
    print("MAIN SLEEVE ASSET PERFORMANCE RANKED BY AVG EXCESS vs SHY")
    print(f"{'Ticker':<6} {'Sel':>4} {'Sel%':>5} {'AvgRet%':>8} {'vsSHY%':>8} {'Win%':>6} {'Contrib%':>9}  Asset class")
    print("-" * 100)

    for _, row in sdf.iterrows():
        flag = "  ◀ weak" if row["avg_exc"] < 0 else ""
        print(f"{row['ticker']:<6} {row['months']:>4} {row['pct']:>5.1f}% "
              f"{row['avg_ret']:>+8.2f}% {row['avg_exc']:>+8.2f}% "
              f"{row['win_pct']:>5.0f}% {row['contrib']:>+9.2f}%  "
              f"{row['class']}{flag}")

    # ── Identify underperformers ─────────────────────────────────────────────
    weak = sdf[sdf["avg_exc"] < 0]
    marginal = sdf[(sdf["avg_exc"] >= 0) & (sdf["avg_exc"] < 0.20)]

    print(f"\n{'='*100}")
    print(f"Assets with NEGATIVE avg excess vs SHY: {list(weak['ticker'])}")
    print(f"Assets with marginal excess (0–0.20%):  {list(marginal['ticker'])}")

    # ── Year-by-year selection heatmap (counts only) ─────────────────────────
    print("\n" + "=" * 100)
    print("YEAR-BY-YEAR SELECTION COUNT PER ASSET")
    all_years = sorted(df["year"].unique())
    tickers_ordered = sdf["ticker"].tolist()

    # Header
    header = f"{'Ticker':<6}"
    for y in all_years:
        header += f" {str(y)[2:]:>4}"
    print(header)
    print("-" * (6 + 5 * len(all_years)))

    for ticker in tickers_ordered:
        row_str = f"{ticker:<6}"
        sub = df[df["ticker"] == ticker]
        for y in all_years:
            n = (sub["year"] == y).sum()
            row_str += f" {n if n > 0 else '·':>4}"
        print(row_str)

    # ── Deep-dive: worst performers ──────────────────────────────────────────
    bottom = sdf.tail(5)["ticker"].tolist()
    print(f"\n{'='*100}")
    print(f"BOTTOM-5 DETAIL: avg return per year when held")
    print(f"{'Year':<6}", end="")
    for t in bottom:
        print(f" {t:>10}", end="")
    print()
    print("-" * (6 + 11 * len(bottom)))

    for year in all_years:
        print(f"{year:<6}", end="")
        for t in bottom:
            sub = df[(df["ticker"] == t) & (df["year"] == year)]
            if sub.empty:
                print(f" {'—':>10}", end="")
            else:
                avg = sub["asset_ret"].mean() * 100
                print(f" {avg:>+10.2f}", end="")
        print()

    # ── Portfolio return when each ticker is vs is NOT in the portfolio ───────
    print(f"\n{'='*100}")
    print("PORTFOLIO MONTHLY RETURN: months ticker IS held vs is NOT held")
    print(f"{'Ticker':<6} {'Held avg':>10} {'Not-held avg':>13} {'Difference':>11}")
    print("-" * 45)

    port_rets = res.monthly_returns.copy()
    port_rets.index = pd.to_datetime(port_rets.index)

    for _, row in sdf.iterrows():
        ticker = row["ticker"]
        sub = df[df["ticker"] == ticker]
        held_months = set(sub["exec_from"].dt.to_period("M"))

        held_port = []
        not_held_port = []
        for i, sig in enumerate(signal_dates):
            if i >= len(exec_dates) - 1:
                break
            e0 = exec_dates[i]
            period = pd.Period(e0, freq="M")
            if e0 in port_rets.index:
                r = port_rets.at[e0]
                if period in held_months:
                    held_port.append(r)
                else:
                    not_held_port.append(r)

        avg_held     = np.mean(held_port) * 100     if held_port     else float("nan")
        avg_not_held = np.mean(not_held_port) * 100 if not_held_port else float("nan")
        diff = avg_held - avg_not_held
        flag = " ◀" if diff < 0 else ""
        print(f"{ticker:<6} {avg_held:>+10.2f}% {avg_not_held:>+13.2f}% {diff:>+11.2f}%{flag}")


if __name__ == "__main__":
    main()
