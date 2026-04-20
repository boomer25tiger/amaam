"""
2D grid sweep: wM × wC with wV = 1 - wM - wC.

wM: 0.45, 0.50, 0.55, 0.60, 0.65
wC: 0.05 → max where wV = 1 - wM - wC >= 0.05
wT: 1.0 fixed throughout.

Baseline: wM=0.50, wV=0.25, wC=0.25.

Output:
  1. Full grid of OOS Sharpe (heatmap-style table)
  2. Full grid of IS Sharpe
  3. Delta vs baseline (both IS and OOS) — marks "✓" where both are positive
  4. Spearman ρ (all grid points together)
  5. Year-by-year OOS for top candidates
  6. Defensive sector selection for top candidates
"""

import sys
sys.path.insert(0, "/Users/GualyCr/Desktop/AMAAM/amaam/.claude/worktrees/reverent-mayer")

import numpy as np
import pandas as pd
from dataclasses import replace
from itertools import product

from config.default_config import ModelConfig
from src.data.loader import load_validated_data
from src.backtest.engine import run_backtest


WM_VALUES = [round(v, 2) for v in np.arange(0.45, 0.66, 0.05)]
WC_VALUES = [round(v, 2) for v in np.arange(0.05, 0.46, 0.05)]
WV_MIN    = 0.05   # enforce a floor so volatility signal isn't zeroed out


def slice_metrics(result, start, end):
    ret = result.monthly_returns
    ret = ret[(ret.index >= start) & (ret.index <= end)]
    if len(ret) < 3:
        return dict(ret=float("nan"), sharpe=float("nan"),
                    maxdd=float("nan"), calmar=float("nan"), n=0)
    ann_ret = (1 + ret).prod() ** (12 / len(ret)) - 1
    ann_vol = ret.std() * np.sqrt(12)
    sharpe  = (ann_ret - 0.02) / ann_vol
    eq      = (1 + ret).cumprod()
    maxdd   = (eq / eq.cummax() - 1).min()
    calmar  = ann_ret / abs(maxdd) if maxdd != 0 else float("nan")
    return dict(ret=ann_ret*100, sharpe=sharpe, maxdd=maxdd*100, calmar=calmar, n=len(ret))


def spearman_rho(x, y):
    n = len(x)
    rx = pd.Series(x).rank().values
    ry = pd.Series(y).rank().values
    d2 = np.sum((rx - ry) ** 2)
    return 1 - 6 * d2 / (n * (n**2 - 1))


def sel_pct(result, ticker):
    allocs = result.allocations
    if ticker not in allocs.columns:
        return 0.0
    total = len(result.monthly_returns)
    return (allocs[ticker] > 0).sum() / total * 100


def main():
    print("Loading data…")
    data_dict = load_validated_data("data/processed")
    base_cfg  = ModelConfig()

    IS_START, IS_END   = base_cfg.backtest_start, "2017-12-31"
    OOS_START, OOS_END = base_cfg.holdout_start,  base_cfg.backtest_end

    # Build the full grid of valid (wM, wC) combinations
    grid = []
    for wm, wc in product(WM_VALUES, WC_VALUES):
        wv = round(1.0 - wm - wc, 2)
        if wv < WV_MIN:
            continue
        grid.append((wm, wc, wv))

    print(f"Grid size: {len(grid)} configurations\n")

    rows = []
    for wm, wc, wv in grid:
        is_baseline = (wm == 0.50 and wc == 0.25)
        tag = " ← baseline" if is_baseline else ""
        print(f"  wM={wm:.2f}  wV={wv:.2f}  wC={wc:.2f}{tag}")

        cfg = replace(base_cfg, weight_momentum=wm,
                      weight_volatility=wv, weight_correlation=wc)
        res = run_backtest(data_dict, cfg)
        m_is  = slice_metrics(res, IS_START, IS_END)
        m_oos = slice_metrics(res, OOS_START, OOS_END)

        rows.append(dict(
            wm=wm, wv=wv, wc=wc,
            is_ret=m_is["ret"],    is_sr=m_is["sharpe"],
            is_dd=m_is["maxdd"],   is_cal=m_is["calmar"],
            oos_ret=m_oos["ret"],  oos_sr=m_oos["sharpe"],
            oos_dd=m_oos["maxdd"], oos_cal=m_oos["calmar"],
            xlp=sel_pct(res, "XLP"),
            xlu=sel_pct(res, "XLU"),
            xlv=sel_pct(res, "XLV"),
            result=res,
        ))

    df = pd.DataFrame(rows)

    # Baseline values for delta computation
    base_row = df[(df["wm"] == 0.50) & (df["wc"] == 0.25)].iloc[0]
    orig_is  = base_row["is_sr"]
    orig_oos = base_row["oos_sr"]

    # ── OOS Sharpe grid ───────────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print(f"OOS SHARPE GRID  (2018-01 → 2026-04)  —  baseline wM=0.50/wV=0.25/wC=0.25 = {orig_oos:.3f}")
    print(f"{'wM \\ wC':>8}", end="")
    for wc in WC_VALUES:
        print(f"  {wc:.2f}", end="")
    print()
    print("-" * 90)
    for wm in WM_VALUES:
        print(f"wM={wm:.2f} ", end="")
        for wc in WC_VALUES:
            sub = df[(df["wm"] == wm) & (df["wc"] == wc)]
            if sub.empty:
                print(f"     —", end="")
            else:
                sr = sub.iloc[0]["oos_sr"]
                marker = "*" if (sub.iloc[0]["oos_sr"] > orig_oos and
                                 sub.iloc[0]["is_sr"]  > orig_is) else " "
                print(f" {sr:>5.3f}{marker}", end="")
        print()
    print("  (* = beats baseline in BOTH IS and OOS)")

    # ── IS Sharpe grid ────────────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print(f"IS SHARPE GRID  (2007-08 → 2017-12)  —  baseline IS = {orig_is:.3f}")
    print(f"{'wM \\ wC':>8}", end="")
    for wc in WC_VALUES:
        print(f"  {wc:.2f}", end="")
    print()
    print("-" * 90)
    for wm in WM_VALUES:
        print(f"wM={wm:.2f} ", end="")
        for wc in WC_VALUES:
            sub = df[(df["wm"] == wm) & (df["wc"] == wc)]
            if sub.empty:
                print(f"     —", end="")
            else:
                sr = sub.iloc[0]["is_sr"]
                print(f" {sr:>5.3f} ", end="")
        print()

    # ── Both-positive map ─────────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print(f"DELTA vs BASELINE — 'Both positive' map  (✓ = ΔIS>0 AND ΔOOS>0)")
    print(f"{'wM \\ wC':>8}", end="")
    for wc in WC_VALUES:
        print(f"   {wc:.2f}", end="")
    print()
    print("-" * 90)
    for wm in WM_VALUES:
        print(f"wM={wm:.2f} ", end="")
        for wc in WC_VALUES:
            sub = df[(df["wm"] == wm) & (df["wc"] == wc)]
            if sub.empty:
                print(f"     —", end="")
            else:
                r = sub.iloc[0]
                d_is  = r["is_sr"]  - orig_is
                d_oos = r["oos_sr"] - orig_oos
                if d_is > 0 and d_oos > 0:
                    print(f"  ✓{d_oos:>+.2f}", end="")
                else:
                    print(f"  ·{d_oos:>+.2f}", end="")
        print()

    # ── Spearman ρ ────────────────────────────────────────────────────────────
    valid = df.dropna(subset=["is_sr", "oos_sr"])
    rho = spearman_rho(valid["is_sr"].tolist(), valid["oos_sr"].tolist())
    print(f"\nIS/OOS Spearman ρ across all {len(valid)} grid points: {rho:+.3f}")

    best_is  = df.loc[df["is_sr"].idxmax()]
    best_oos = df.loc[df["oos_sr"].idxmax()]
    print(f"Best IS  Sharpe: wM={best_is['wm']:.2f} wV={best_is['wv']:.2f} wC={best_is['wc']:.2f}  SR={best_is['is_sr']:.3f}")
    print(f"Best OOS Sharpe: wM={best_oos['wm']:.2f} wV={best_oos['wv']:.2f} wC={best_oos['wc']:.2f}  SR={best_oos['oos_sr']:.3f}")

    # ── "Both positive" full list ─────────────────────────────────────────────
    both_pos = df[(df["is_sr"] > orig_is) & (df["oos_sr"] > orig_oos)].copy()
    both_pos["d_is"]  = both_pos["is_sr"]  - orig_is
    both_pos["d_oos"] = both_pos["oos_sr"] - orig_oos
    both_pos = both_pos.sort_values("d_oos", ascending=False)

    print(f"\n{'='*90}")
    print(f"ALL CONFIGS BEATING BASELINE IN BOTH IS AND OOS  ({len(both_pos)} found)")
    print(f"{'wM':>5} {'wV':>5} {'wC':>5}  {'IS SR':>7} {'ΔIS':>7}  {'OOS SR':>7} {'ΔOOS':>7}  {'OOS Ret%':>9} {'OOS DD%':>8}")
    print("-" * 90)
    for _, r in both_pos.iterrows():
        print(f"{r['wm']:>5.2f} {r['wv']:>5.2f} {r['wc']:>5.2f}  "
              f"{r['is_sr']:>7.3f} {r['d_is']:>+7.3f}  "
              f"{r['oos_sr']:>7.3f} {r['d_oos']:>+7.3f}  "
              f"{r['oos_ret']:>9.2f}% {r['oos_dd']:>8.2f}%")

    if both_pos.empty:
        print("  (none)")

    # ── Top candidates: year-by-year OOS ──────────────────────────────────────
    # Pick top-5 by OOS Sharpe from "both positive", plus baseline
    top_n = min(5, len(both_pos))
    candidates = both_pos.head(top_n)[["wm","wv","wc"]].values.tolist()

    print(f"\n{'='*90}")
    print("YEAR-BY-YEAR OOS — top 'both positive' candidates + baseline")
    header = f"{'Year':<6}"
    labels = []
    for wm, wv, wc in candidates:
        lbl = f"M{wm:.0%}/V{wv:.0%}/C{wc:.0%}"
        lbl = lbl.replace("%", "")
        header += f"  {lbl:>14}"
        labels.append((wm, wv, wc))
    # always include baseline
    base_label = "Baseline"
    header += f"  {base_label:>14}"
    print(header)
    print("-" * 90)

    # Build monthly return series for each candidate
    cand_rets = {}
    for wm, wv, wc in labels:
        sub = df[(df["wm"] == wm) & (df["wv"] == wv) & (df["wc"] == wc)]
        if not sub.empty:
            cand_rets[(wm, wv, wc)] = sub.iloc[0]["result"].monthly_returns

    base_rets = base_row["result"].monthly_returns

    for year in range(2018, 2027):
        row_str = f"{year:<6}"
        for key in labels:
            if key in cand_rets:
                yr = cand_rets[key]
                yr = yr[yr.index.year == year]
                ann = (1 + yr).prod() ** (12 / len(yr)) - 1 if len(yr) > 0 else float("nan")
                row_str += f"  {ann*100:>+12.1f}%" if not np.isnan(ann) else f"  {'—':>13}"
            else:
                row_str += f"  {'—':>13}"
        yr = base_rets[base_rets.index.year == year]
        ann = (1 + yr).prod() ** (12 / len(yr)) - 1 if len(yr) > 0 else float("nan")
        row_str += f"  {ann*100:>+12.1f}%" if not np.isnan(ann) else f"  {'—':>13}"
        print(row_str)

    # ── Defensive sector selection ────────────────────────────────────────────
    print(f"\n{'='*90}")
    print("DEFENSIVE SECTOR SELECTION  —  top 'both positive' candidates + baseline")
    print(f"{'wM':>5} {'wV':>5} {'wC':>5}  {'XLP%':>6} {'XLU%':>6} {'XLV%':>6}  {'Combined':>9}")
    print("-" * 60)
    for _, r in both_pos.head(top_n).iterrows():
        print(f"{r['wm']:>5.2f} {r['wv']:>5.2f} {r['wc']:>5.2f}  "
              f"{r['xlp']:>6.1f}% {r['xlu']:>6.1f}% {r['xlv']:>6.1f}%  "
              f"{r['xlp']+r['xlu']+r['xlv']:>8.1f}%")
    br = base_row
    print(f"{br['wm']:>5.2f} {br['wv']:>5.2f} {br['wc']:>5.2f}  "
          f"{br['xlp']:>6.1f}% {br['xlu']:>6.1f}% {br['xlv']:>6.1f}%  "
          f"{br['xlp']+br['xlu']+br['xlv']:>8.1f}%  ← baseline")


if __name__ == "__main__":
    main()
