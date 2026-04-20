"""
wV (volatility weight) sweep with IS/OOS discipline.

Keeps wC = 0.25 fixed (already validated) and wT = 1.0 fixed.
Varies wV from 0.05 to 0.40 in steps of 0.05.
wM = 0.75 - wV so that wM + wV + wC = 1.0 throughout.

For each (wM, wV) pair:
  - IS  (2007-08 → 2017-12): Sharpe, Return, MaxDD, Calmar
  - OOS (2018-01 → 2026-04): same

IS/OOS Spearman rank correlation on Sharpe reported at the end.
"""

import sys
sys.path.insert(0, "/Users/GualyCr/Desktop/AMAAM/amaam/.claude/worktrees/reverent-mayer")

import numpy as np
import pandas as pd
from dataclasses import replace

from config.default_config import ModelConfig
from src.data.loader import load_validated_data
from src.backtest.engine import run_backtest


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


def main():
    print("Loading data…")
    data_dict = load_validated_data("data/processed")
    base_cfg = ModelConfig()

    IS_START, IS_END   = base_cfg.backtest_start, "2017-12-31"
    OOS_START, OOS_END = base_cfg.holdout_start,  base_cfg.backtest_end

    # wV from 0.05 to 0.40 inclusive; wM = 0.75 - wV; wC = 0.25 fixed
    wv_values = [round(v, 2) for v in np.arange(0.05, 0.41, 0.05)]

    rows = []
    for wv in wv_values:
        wm = round(0.75 - wv, 2)
        cfg = replace(base_cfg, weight_momentum=wm, weight_volatility=wv)
        tag = "← baseline" if wv == 0.25 else ""
        print(f"  wM={wm:.2f}  wV={wv:.2f}  wC=0.25  {tag}")

        res = run_backtest(data_dict, cfg)
        m_is  = slice_metrics(res, IS_START, IS_END)
        m_oos = slice_metrics(res, OOS_START, OOS_END)

        rows.append(dict(
            wv=wv, wm=wm,
            is_ret=m_is["ret"],   is_sr=m_is["sharpe"],
            is_dd=m_is["maxdd"],  is_cal=m_is["calmar"],
            oos_ret=m_oos["ret"], oos_sr=m_oos["sharpe"],
            oos_dd=m_oos["maxdd"],oos_cal=m_oos["calmar"],
        ))

    df = pd.DataFrame(rows)

    # ── IS table ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("IN-SAMPLE (2007-08 → 2017-12)")
    print(f"{'wM':>5} {'wV':>5}  {'Ret%':>7} {'Sharpe':>8} {'MaxDD%':>9} {'Calmar':>8}")
    print("-" * 80)
    for _, r in df.iterrows():
        tag = " ← baseline" if r["wv"] == 0.25 else ""
        print(f"{r['wm']:>5.2f} {r['wv']:>5.2f}  {r['is_ret']:>7.2f} {r['is_sr']:>8.3f} "
              f"{r['is_dd']:>9.2f} {r['is_cal']:>8.3f}{tag}")

    # ── OOS table ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("OUT-OF-SAMPLE (2018-01 → 2026-04)")
    print(f"{'wM':>5} {'wV':>5}  {'Ret%':>7} {'Sharpe':>8} {'MaxDD%':>9} {'Calmar':>8}")
    print("-" * 80)
    for _, r in df.iterrows():
        tag = " ← baseline" if r["wv"] == 0.25 else ""
        print(f"{r['wm']:>5.2f} {r['wv']:>5.2f}  {r['oos_ret']:>7.2f} {r['oos_sr']:>8.3f} "
              f"{r['oos_dd']:>9.2f} {r['oos_cal']:>8.3f}{tag}")

    # ── IS/OOS Spearman ───────────────────────────────────────────────────────
    rho = spearman_rho(df["is_sr"].tolist(), df["oos_sr"].tolist())
    print(f"\nIS/OOS Spearman ρ (Sharpe ranks): {rho:+.3f}")
    if rho > 0.5:
        print("  → Strong positive rank correlation — IS winner likely to generalise")
    elif rho > 0.0:
        print("  → Weak positive rank correlation — modest generalisation signal")
    elif rho > -0.3:
        print("  → Near-zero / mildly negative — no reliable generalisation")
    else:
        print("  → Negative rank correlation — IS winner actively predicts OOS loser")

    # ── Best wV by period ──────────────────────────────────────────────────────
    best_is  = df.loc[df["is_sr"].idxmax()]
    best_oos = df.loc[df["oos_sr"].idxmax()]
    print(f"\nBest IS  Sharpe: wV={best_is['wv']:.2f}  SR={best_is['is_sr']:.3f}")
    print(f"Best OOS Sharpe: wV={best_oos['wv']:.2f}  SR={best_oos['oos_sr']:.3f}")

    # ── Year-by-year OOS for selected candidates ──────────────────────────────
    candidates = sorted(set([0.10, 0.15, 0.20, 0.25, best_oos["wv"]]))
    print("\n" + "=" * 80)
    print(f"YEAR-BY-YEAR OOS RETURNS — selected wV candidates")
    print(f"{'Year':<6}", end="")
    for wv in candidates:
        wm = round(0.75 - wv, 2)
        print(f"  wV={wv:.2f}", end="")
    print()
    print("-" * 80)

    year_rets = {}
    for wv in candidates:
        wm = round(0.75 - wv, 2)
        cfg = replace(base_cfg, weight_momentum=wm, weight_volatility=wv)
        res = run_backtest(data_dict, cfg)
        ret = res.monthly_returns
        year_rets[wv] = ret

    for year in range(2018, 2027):
        print(f"{year:<6}", end="")
        for wv in candidates:
            yr = year_rets[wv]
            yr = yr[yr.index.year == year]
            if len(yr) == 0:
                print(f"     —  ", end="")
            else:
                ann = (1 + yr).prod() ** (12 / len(yr)) - 1
                print(f"  {ann*100:>+5.1f}%", end="")
        print()

    # ── Selection frequency of XLP/XLU/XLV under each wV ─────────────────────
    print("\n" + "=" * 80)
    print("DEFENSIVE SECTOR SELECTION FREQUENCY vs wV")
    print(f"{'wV':>5}  {'XLP%':>6} {'XLU%':>6} {'XLV%':>6}  combined")
    print("-" * 80)

    for wv in [0.10, 0.15, 0.20, 0.25]:
        wm = round(0.75 - wv, 2)
        cfg = replace(base_cfg, weight_momentum=wm, weight_volatility=wv)
        res = run_backtest(data_dict, cfg)
        allocs = res.allocations
        total = len(res.monthly_returns)
        def sel_pct(t):
            if t not in allocs.columns:
                return 0.0
            return (allocs[t] > 0).sum() / total * 100
        xlp = sel_pct("XLP")
        xlu = sel_pct("XLU")
        xlv = sel_pct("XLV")
        tag = " ← baseline" if wv == 0.25 else ""
        print(f"{wv:>5.2f}  {xlp:>6.1f}% {xlu:>6.1f}% {xlv:>6.1f}%  {xlp+xlu+xlv:>7.1f}%{tag}")


if __name__ == "__main__":
    main()
