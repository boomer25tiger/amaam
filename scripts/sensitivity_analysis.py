"""
Sensitivity analysis for candidate weights (wM=0.65, wV=0.25, wC=0.10).

Tests three types of perturbation:
  1. Single-weight ±0.05 nudges (re-normalising to keep sum = 1.0)
  2. All adjacent grid points in the 2D (wM, wC) sweep
  3. Pairwise swaps: shift 0.05 from one weight to another

Reports IS and OOS Sharpe delta vs candidate for every perturbation.
A robust result shows a smooth, gradual decline as you move away from the
candidate — not a sharp spike surrounded by worse configurations.
"""

import sys
sys.path.insert(0, "/Users/GualyCr/Desktop/AMAAM/amaam/.claude/worktrees/reverent-mayer")

import numpy as np
import pandas as pd
from dataclasses import replace

from config.default_config import ModelConfig
from src.data.loader import load_validated_data
from src.backtest.engine import run_backtest


CANDIDATE = (0.65, 0.25, 0.10)   # (wM, wV, wC)


def slice_metrics(rets, start, end, rf=0.02):
    r = rets[(rets.index >= start) & (rets.index <= end)]
    if len(r) < 6:
        return dict(sr=float("nan"), ret=float("nan"),
                    maxdd=float("nan"), calmar=float("nan"))
    ann_ret = (1 + r).prod() ** (12 / len(r)) - 1
    ann_vol = r.std() * np.sqrt(12)
    sr = (ann_ret - rf) / ann_vol
    eq = (1 + r).cumprod()
    maxdd = (eq / eq.cummax() - 1).min()
    calmar = ann_ret / abs(maxdd) if maxdd != 0 else float("nan")
    return dict(sr=sr, ret=ann_ret*100, maxdd=maxdd*100, calmar=calmar)


def run_cfg(data_dict, base_cfg, wm, wv, wc):
    cfg = replace(base_cfg, weight_momentum=round(wm, 2),
                  weight_volatility=round(wv, 2), weight_correlation=round(wc, 2))
    return run_backtest(data_dict, cfg).monthly_returns


def main():
    print("Loading data…")
    data_dict = load_validated_data("data/processed")
    base_cfg  = ModelConfig()

    IS_START,  IS_END  = base_cfg.backtest_start, "2017-12-31"
    OOS_START, OOS_END = base_cfg.holdout_start,  base_cfg.backtest_end

    wm0, wv0, wc0 = CANDIDATE
    print(f"Candidate: wM={wm0}  wV={wv0}  wC={wc0}\n")

    # ── Run candidate ─────────────────────────────────────────────────────────
    cand_rets = run_cfg(data_dict, base_cfg, wm0, wv0, wc0)
    m_is_c  = slice_metrics(cand_rets, IS_START,  IS_END)
    m_oos_c = slice_metrics(cand_rets, OOS_START, OOS_END)
    print(f"Candidate  IS SR={m_is_c['sr']:.3f}  OOS SR={m_oos_c['sr']:.3f}\n")

    # ── Build perturbation set ────────────────────────────────────────────────
    # Single-weight nudges: shift δ into/from each weight, compensating from another
    delta = 0.05
    perturbs = []

    # Format: (label, wm, wv, wc)
    # Move δ from wV to wM (more momentum, less vol)
    perturbs.append(("wM+0.05  wV-0.05", wm0+delta, wv0-delta, wc0))
    # Move δ from wM to wV (less momentum, more vol)
    perturbs.append(("wM-0.05  wV+0.05", wm0-delta, wv0+delta, wc0))
    # Move δ from wC to wM (more momentum, less corr)
    perturbs.append(("wM+0.05  wC-0.05", wm0+delta, wv0, wc0-delta))
    # Move δ from wM to wC (less momentum, more corr)
    perturbs.append(("wM-0.05  wC+0.05", wm0-delta, wv0, wc0+delta))
    # Move δ from wC to wV (more vol, less corr)
    perturbs.append(("wV+0.05  wC-0.05", wm0, wv0+delta, wc0-delta))
    # Move δ from wV to wC (less vol, more corr)
    perturbs.append(("wV-0.05  wC+0.05", wm0, wv0-delta, wc0+delta))

    # 2× nudges
    perturbs.append(("wM+0.10  wV-0.10", wm0+2*delta, wv0-2*delta, wc0))
    perturbs.append(("wM-0.10  wV+0.10", wm0-2*delta, wv0+2*delta, wc0))
    perturbs.append(("wM+0.10  wC-0.05 wV-0.05",
                      wm0+2*delta, wv0-delta, wc0-delta))
    perturbs.append(("wM-0.10  wC+0.05 wV+0.05",
                      wm0-2*delta, wv0+delta, wc0+delta))

    # Filter: all weights must be in [0.05, 0.90] and sum to 1.0
    valid = []
    for label, wm, wv, wc in perturbs:
        wm, wv, wc = round(wm, 2), round(wv, 2), round(wc, 2)
        if wm >= 0.05 and wv >= 0.05 and wc >= 0.05 and abs(wm+wv+wc-1.0) < 0.001:
            valid.append((label, wm, wv, wc))

    # ── Run all perturbations ─────────────────────────────────────────────────
    rows = []
    for label, wm, wv, wc in valid:
        print(f"  {label}  wM={wm:.2f} wV={wv:.2f} wC={wc:.2f}")
        rets = run_cfg(data_dict, base_cfg, wm, wv, wc)
        m_is  = slice_metrics(rets, IS_START,  IS_END)
        m_oos = slice_metrics(rets, OOS_START, OOS_END)
        rows.append(dict(
            label=label, wm=wm, wv=wv, wc=wc,
            is_sr=m_is["sr"],   oos_sr=m_oos["sr"],
            d_is=m_is["sr"]  - m_is_c["sr"],
            d_oos=m_oos["sr"] - m_oos_c["sr"],
        ))

    df = pd.DataFrame(rows)

    # ── Results table ─────────────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print(f"SENSITIVITY ANALYSIS — centred on wM={wm0}/wV={wv0}/wC={wc0}")
    print(f"Candidate IS SR = {m_is_c['sr']:.3f}   OOS SR = {m_oos_c['sr']:.3f}")
    print("=" * 90)
    print(f"{'Perturbation':<30} {'wM':>5} {'wV':>5} {'wC':>5}  "
          f"{'IS SR':>7} {'ΔIS':>7}  {'OOS SR':>7} {'ΔOOS':>7}  Stable?")
    print("-" * 90)

    for _, r in df.sort_values("d_oos").iterrows():
        stable = "✓" if abs(r["d_is"]) < 0.05 and abs(r["d_oos"]) < 0.05 else "!"
        print(f"{r['label']:<30} {r['wm']:>5.2f} {r['wv']:>5.2f} {r['wc']:>5.2f}  "
              f"{r['is_sr']:>7.3f} {r['d_is']:>+7.3f}  "
              f"{r['oos_sr']:>7.3f} {r['d_oos']:>+7.3f}  {stable}")

    print(f"\n  ✓ = perturbation changes IS and OOS Sharpe by <0.05 (stable neighbourhood)")
    print(f"  ! = perturbation causes >0.05 change (sharp spike — fragility signal)")

    stable_count = ((df["d_is"].abs() < 0.05) & (df["d_oos"].abs() < 0.05)).sum()
    print(f"\n  {stable_count}/{len(df)} perturbations within ±0.05 Sharpe of candidate")

    # ── Direction analysis ────────────────────────────────────────────────────
    print(f"\n{'='*90}")
    print("DIRECTION ANALYSIS — which moves improve and which hurt")
    improve_oos = df[df["d_oos"] > 0.01].sort_values("d_oos", ascending=False)
    hurt_oos    = df[df["d_oos"] < -0.01].sort_values("d_oos")

    if not improve_oos.empty:
        print("\nMoves that improve OOS (>+0.01):")
        for _, r in improve_oos.iterrows():
            print(f"  {r['label']:<30} ΔOOS={r['d_oos']:>+.3f}  ΔIS={r['d_is']:>+.3f}")
    else:
        print("\nNo perturbation improves OOS by >0.01 — candidate is a local optimum.")

    if not hurt_oos.empty:
        print("\nMoves that hurt OOS (>-0.01):")
        for _, r in hurt_oos.iterrows():
            print(f"  {r['label']:<30} ΔOOS={r['d_oos']:>+.3f}  ΔIS={r['d_is']:>+.3f}")


if __name__ == "__main__":
    main()
