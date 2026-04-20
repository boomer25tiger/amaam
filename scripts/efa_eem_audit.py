"""
EFA and EEM holding audit.

For each month where EFA or EEM was held, shows:
  - Signal date, execution date, weight
  - That month's individual asset return
  - Whether the model would have been better off in SHY instead

Also reports aggregate statistics and year-by-year selection counts.
"""

import sys
sys.path.insert(0, "/Users/GualyCr/Desktop/AMAAM/amaam/.claude/worktrees/reverent-mayer")

import numpy as np
import pandas as pd

from config.default_config import ModelConfig
from src.data.loader import load_validated_data
from src.backtest.engine import run_backtest


def main():
    print("Loading data and running backtest…")
    data_dict = load_validated_data("data/processed")
    cfg = ModelConfig()
    res = run_backtest(data_dict, cfg)

    allocs = res.allocations.copy()
    allocs.index = pd.to_datetime(allocs.index)

    # Build a monthly price-return matrix for all assets
    closes = pd.DataFrame({t: data_dict[t]["Close"] for t in data_dict})

    # Map signal date → execution date (next trading day) using the engine's logic
    # We re-derive exec dates from the equity curve index (those are exec dates)
    exec_dates = res.monthly_returns.index.tolist()

    # allocs is indexed by signal date; monthly_returns by exec date
    # Align: alloc[i] → return[i] (the alloc from signal date i produces the
    # return recorded at exec_date i+1 in the engine loop)
    signal_dates = allocs.index.tolist()

    # Build forward return for each ticker for each holding month
    # Using actual closes between consecutive exec dates
    records = []
    for i, sig_date in enumerate(signal_dates):
        if i >= len(exec_dates) - 1:
            break
        exec0 = exec_dates[i]
        exec1 = exec_dates[i + 1]

        for ticker in ["EFA", "EEM"]:
            weight = allocs.at[sig_date, ticker] if ticker in allocs.columns else 0.0
            if weight <= 0:
                continue

            if ticker in closes.columns and exec0 in closes.index and exec1 in closes.index:
                p0 = closes.at[exec0, ticker]
                p1 = closes.at[exec1, ticker]
                asset_ret = (p1 / p0 - 1.0) if p0 > 0 else float("nan")
            else:
                asset_ret = float("nan")

            # SHY return over the same period (the opportunity cost benchmark)
            if "SHY" in closes.columns and exec0 in closes.index and exec1 in closes.index:
                s0 = closes.at[exec0, "SHY"]
                s1 = closes.at[exec1, "SHY"]
                shy_ret = (s1 / s0 - 1.0) if s0 > 0 else float("nan")
            else:
                shy_ret = float("nan")

            records.append({
                "signal_date": sig_date,
                "exec_from":   exec0,
                "exec_to":     exec1,
                "ticker":      ticker,
                "weight":      round(weight, 4),
                "asset_ret":   asset_ret,
                "shy_ret":     shy_ret,
                "excess_vs_shy": asset_ret - shy_ret if not np.isnan(asset_ret) else float("nan"),
                "contributed": weight * asset_ret if not np.isnan(asset_ret) else float("nan"),
                "year":        exec0.year,
            })

    df = pd.DataFrame(records)
    if df.empty:
        print("Neither EFA nor EEM was ever held.")
        return

    # ── Overall selection summary ────────────────────────────────────────────
    total_months = len(exec_dates) - 1
    print(f"\nTotal backtest months: {total_months}")
    print("=" * 80)
    print("SELECTION FREQUENCY & AVERAGE WEIGHT")
    print("-" * 80)
    for ticker in ["EFA", "EEM"]:
        sub = df[df["ticker"] == ticker]
        if sub.empty:
            print(f"  {ticker}: never selected")
            continue
        n = len(sub)
        avg_w = sub["weight"].mean()
        avg_ret = sub["asset_ret"].mean() * 100
        avg_shy = sub["shy_ret"].mean() * 100
        avg_exc = sub["excess_vs_shy"].mean() * 100
        total_contrib = sub["contributed"].sum() * 100
        pos_months = (sub["asset_ret"] > 0).sum()
        print(f"  {ticker}: selected {n}/{total_months} months ({n/total_months*100:.1f}%)")
        print(f"         avg weight when held : {avg_w:.1%}")
        print(f"         avg monthly return   : {avg_ret:+.2f}%")
        print(f"         avg SHY return (same): {avg_shy:+.2f}%")
        print(f"         avg excess vs SHY    : {avg_exc:+.2f}%")
        print(f"         positive months      : {pos_months}/{n} ({pos_months/n*100:.0f}%)")
        print(f"         total return contrib : {total_contrib:+.2f}% (raw, not portfolio-weighted)")
        print()

    # ── Year-by-year ─────────────────────────────────────────────────────────
    print("=" * 80)
    print("YEAR-BY-YEAR SELECTION COUNT AND AVERAGE RETURN")
    print(f"{'Year':<6} {'EFA sel':>8} {'EFA ret%':>10} {'EEM sel':>8} {'EEM ret%':>10}")
    print("-" * 80)
    for year in range(2007, 2027):
        row = [str(year)]
        for ticker in ["EFA", "EEM"]:
            sub = df[(df["ticker"] == ticker) & (df["year"] == year)]
            if sub.empty:
                row += ["—", "—"]
            else:
                n = len(sub)
                avg_r = sub["asset_ret"].mean() * 100
                row += [str(n), f"{avg_r:+.1f}%"]
        if row[1] == "—" and row[3] == "—":
            continue
        print(f"{row[0]:<6} {row[1]:>8} {row[2]:>10} {row[3]:>8} {row[4]:>10}")

    # ── Month-by-month detail for held months ────────────────────────────────
    print("\n" + "=" * 80)
    print("MONTH-BY-MONTH DETAIL (all months EFA or EEM was held)")
    print(f"{'Period':<12} {'Ticker':<6} {'Wt':>6} {'Asset':>8} {'SHY':>7} {'vs SHY':>8} {'Better?':>8}")
    print("-" * 80)
    for _, row in df.sort_values(["exec_from", "ticker"]).iterrows():
        period = row["exec_from"].strftime("%Y-%m")
        better = "✓" if row["excess_vs_shy"] > 0 else "✗"
        print(f"{period:<12} {row['ticker']:<6} {row['weight']:>6.1%} "
              f"{row['asset_ret']*100:>+8.2f}% {row['shy_ret']*100:>+7.2f}% "
              f"{row['excess_vs_shy']*100:>+8.2f}% {better:>8}")

    # ── Win rate and regime context ───────────────────────────────────────────
    print("\n" + "=" * 80)
    print("WIN RATE vs SHY (combined EFA+EEM held months)")
    combined = df.dropna(subset=["excess_vs_shy"])
    wins = (combined["excess_vs_shy"] > 0).sum()
    total = len(combined)
    avg_win  = combined.loc[combined["excess_vs_shy"] > 0, "excess_vs_shy"].mean() * 100
    avg_loss = combined.loc[combined["excess_vs_shy"] <= 0, "excess_vs_shy"].mean() * 100
    print(f"  Win rate vs SHY  : {wins}/{total} ({wins/total*100:.0f}%)")
    print(f"  Avg win  (excess): {avg_win:+.2f}%")
    print(f"  Avg loss (excess): {avg_loss:+.2f}%")
    print(f"  Expectancy       : {combined['excess_vs_shy'].mean()*100:+.2f}% per held month")


if __name__ == "__main__":
    main()
