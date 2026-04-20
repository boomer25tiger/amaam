"""
wC sweep at fixed wM=0.50.

wM = 0.50 fixed (original momentum weight).
wC swept from 0.05 to 0.45 in steps of 0.05.
wV = 0.50 - wC so that wM + wV + wC = 1.0 throughout.
wT = 1.0 unchanged.

At wC=0.25: wM=0.50, wV=0.25, wC=0.25 → original baseline.
At wC=0.35: wM=0.50, wV=0.15, wC=0.35 → "both positive" config from prior sweep.

Reports IS / OOS metrics, Spearman ρ, year-by-year OOS, and
defensive sector selection frequency.
"""

import sys
sys.path.insert(0, "/Users/GualyCr/Desktop/AMAAM/amaam/.claude/worktrees/reverent-mayer")

import numpy as np
import pandas as pd
from dataclasses import replace

from config.default_config import ModelConfig
from src.data.loader import load_validated_data
from src.backtest.engine import run_backtest

WM_FIXED = 0.50


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

    wc_values = [round(v, 2) for v in np.arange(0.05, 0.46, 0.05)]

    rows = []
    for wc in wc_values:
        wv = round(0.50 - wc, 2)
        if wv < 0:
            print(f"  wC={wc:.2f} → wV={wv:.2f} (negative, skipping)")
            continue
        cfg = replace(base_cfg, weight_momentum=WM_FIXED,
                      weight_volatility=wv, weight_correlation=wc)
        tag = " ← original baseline" if wc == 0.25 else ""
        print(f"  wM={WM_FIXED:.2f}  wV={wv:.2f}  wC={wc:.2f}{tag}")

        res = run_backtest(data_dict, cfg)
        m_is  = slice_metrics(res, IS_START, IS_END)
        m_oos = slice_metrics(res, OOS_START, OOS_END)

        rows.append(dict(
            wc=wc, wv=wv,
            is_ret=m_is["ret"],    is_sr=m_is["sharpe"],
            is_dd=m_is["maxdd"],   is_cal=m_is["calmar"],
            oos_ret=m_oos["ret"],  oos_sr=m_oos["sharpe"],
            oos_dd=m_oos["maxdd"], oos_cal=m_oos["calmar"],
            xlp=sel_pct(res, "XLP"),
            xlu=sel_pct(res, "XLU"),
            xlv=sel_pct(res, "XLV"),
        ))

    df = pd.DataFrame(rows)

    # ── IS table ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 88)
    print(f"IN-SAMPLE (2007-08 → 2017-12)  |  wM={WM_FIXED:.2f} fixed")
    print(f"{'wV':>5} {'wC':>5}  {'Ret%':>7} {'Sharpe':>8} {'MaxDD%':>9} {'Calmar':>8}")
    print("-" * 88)
    for _, r in df.iterrows():
        tag = " ← baseline" if r["wc"] == 0.25 else ""
        print(f"{r['wv']:>5.2f} {r['wc']:>5.2f}  {r['is_ret']:>7.2f} {r['is_sr']:>8.3f} "
              f"{r['is_dd']:>9.2f} {r['is_cal']:>8.3f}{tag}")

    # ── OOS table ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 88)
    print(f"OUT-OF-SAMPLE (2018-01 → 2026-04)  |  wM={WM_FIXED:.2f} fixed")
    print(f"{'wV':>5} {'wC':>5}  {'Ret%':>7} {'Sharpe':>8} {'MaxDD%':>9} {'Calmar':>8}")
    print("-" * 88)
    for _, r in df.iterrows():
        tag = " ← baseline" if r["wc"] == 0.25 else ""
        print(f"{r['wv']:>5.2f} {r['wc']:>5.2f}  {r['oos_ret']:>7.2f} {r['oos_sr']:>8.3f} "
              f"{r['oos_dd']:>9.2f} {r['oos_cal']:>8.3f}{tag}")

    # ── IS/OOS rank correlation ───────────────────────────────────────────────
    rho = spearman_rho(df["is_sr"].tolist(), df["oos_sr"].tolist())
    print(f"\nIS/OOS Spearman ρ (Sharpe ranks): {rho:+.3f}")
    best_is  = df.loc[df["is_sr"].idxmax()]
    best_oos = df.loc[df["oos_sr"].idxmax()]
    print(f"Best IS  Sharpe: wC={best_is['wc']:.2f}  wV={best_is['wv']:.2f}  SR={best_is['is_sr']:.3f}")
    print(f"Best OOS Sharpe: wC={best_oos['wc']:.2f}  wV={best_oos['wv']:.2f}  SR={best_oos['oos_sr']:.3f}")

    # ── Delta vs original baseline ────────────────────────────────────────────
    baseline_row = df[df["wc"] == 0.25].iloc[0]
    orig_is  = baseline_row["is_sr"]
    orig_oos = baseline_row["oos_sr"]

    print(f"\nOriginal baseline (wM=0.50, wV=0.25, wC=0.25):  IS={orig_is:.3f}  OOS={orig_oos:.3f}")
    print(f"\nDelta vs original baseline:")
    print(f"{'wV':>5} {'wC':>5}  {'ΔIS':>8} {'ΔOOS':>8}  Both positive?")
    print("-" * 55)
    for _, r in df.iterrows():
        d_is  = r["is_sr"]  - orig_is
        d_oos = r["oos_sr"] - orig_oos
        both  = "✓" if d_is > 0 and d_oos > 0 else " "
        tag   = " ← baseline" if r["wc"] == 0.25 else ""
        print(f"{r['wv']:>5.2f} {r['wc']:>5.2f}  {d_is:>+8.3f} {d_oos:>+8.3f}  {both}{tag}")

    # ── Year-by-year OOS ──────────────────────────────────────────────────────
    candidates = [wc for wc in wc_values if wc in [0.15, 0.20, 0.25, 0.30, 0.35, 0.40]]
    candidates = [wc for wc in candidates if (0.50 - wc) >= 0]
    year_rets = {}
    print(f"\n{'='*88}")
    print("YEAR-BY-YEAR OOS — selected wC candidates (wM=0.50 fixed)")
    print(f"{'Year':<6}", end="")
    for wc in candidates:
        print(f"  wC={wc:.2f}", end="")
    print()
    print("-" * 88)

    for wc in candidates:
        wv = round(0.50 - wc, 2)
        cfg = replace(base_cfg, weight_momentum=WM_FIXED,
                      weight_volatility=wv, weight_correlation=wc)
        res = run_backtest(data_dict, cfg)
        year_rets[wc] = res.monthly_returns

    for year in range(2018, 2027):
        print(f"{year:<6}", end="")
        for wc in candidates:
            yr = year_rets[wc]
            yr = yr[yr.index.year == year]
            if len(yr) == 0:
                print(f"     — ", end="")
            else:
                ann = (1 + yr).prod() ** (12 / len(yr)) - 1
                print(f"  {ann*100:>+5.1f}%", end="")
        print()

    # ── Defensive selection ───────────────────────────────────────────────────
    print(f"\n{'='*88}")
    print("DEFENSIVE SECTOR SELECTION  (XLP + XLU + XLV combined %)")
    print(f"{'wV':>5} {'wC':>5}  {'XLP%':>6} {'XLU%':>6} {'XLV%':>6}  {'Combined':>9}")
    print("-" * 55)
    for _, r in df.iterrows():
        tag = " ← baseline" if r["wc"] == 0.25 else ""
        print(f"{r['wv']:>5.2f} {r['wc']:>5.2f}  {r['xlp']:>6.1f}% {r['xlu']:>6.1f}% "
              f"{r['xlv']:>6.1f}%  {r['xlp']+r['xlu']+r['xlv']:>8.1f}%{tag}")


if __name__ == "__main__":
    main()
