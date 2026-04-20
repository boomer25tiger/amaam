"""
Walk-forward validation of weight configurations.

Approach: run all 39 grid configs once on full history, then slice
each config's monthly_returns to train/test windows — equivalent to
re-running on just that window since warmup always starts at 2007-08.

Expanding training windows, 2-year test windows:
  Fold 1: train 2007-08 → 2012-12,  test 2013-01 → 2014-12
  Fold 2: train 2007-08 → 2014-12,  test 2015-01 → 2016-12
  Fold 3: train 2007-08 → 2016-12,  test 2017-01 → 2018-12
  Fold 4: train 2007-08 → 2018-12,  test 2019-01 → 2020-12
  Fold 5: train 2007-08 → 2020-12,  test 2021-01 → 2022-12
  Fold 6: train 2007-08 → 2022-12,  test 2023-01 → 2024-12

For each fold:
  1. Select the config with the best Sharpe on the training window.
  2. Evaluate that config on the test window.
  3. Compare against the baseline and the candidate (wM=0.65/wV=0.25/wC=0.10).

Key question: does the wM=0.65 region consistently win training windows,
and does it hold up on test windows — or is the "both positive" result
an artifact of a single IS/OOS split?
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
WV_MIN    = 0.05

FOLDS = [
    ("Fold 1", "2007-08-01", "2012-12-31", "2013-01-01", "2014-12-31"),
    ("Fold 2", "2007-08-01", "2014-12-31", "2015-01-01", "2016-12-31"),
    ("Fold 3", "2007-08-01", "2016-12-31", "2017-01-01", "2018-12-31"),
    ("Fold 4", "2007-08-01", "2018-12-31", "2019-01-01", "2020-12-31"),
    ("Fold 5", "2007-08-01", "2020-12-31", "2021-01-01", "2022-12-31"),
    ("Fold 6", "2007-08-01", "2022-12-31", "2023-01-01", "2024-12-31"),
]


def sharpe(ret_series, start, end, rf=0.02):
    r = ret_series[(ret_series.index >= start) & (ret_series.index <= end)]
    if len(r) < 6:
        return float("nan")
    ann_ret = (1 + r).prod() ** (12 / len(r)) - 1
    ann_vol = r.std() * np.sqrt(12)
    return (ann_ret - rf) / ann_vol if ann_vol > 0 else float("nan")


def full_metrics(ret_series, start, end, rf=0.02):
    r = ret_series[(ret_series.index >= start) & (ret_series.index <= end)]
    if len(r) < 6:
        return dict(ret=float("nan"), sr=float("nan"),
                    maxdd=float("nan"), calmar=float("nan"), n=0)
    ann_ret = (1 + r).prod() ** (12 / len(r)) - 1
    ann_vol = r.std() * np.sqrt(12)
    sr      = (ann_ret - rf) / ann_vol
    eq      = (1 + r).cumprod()
    maxdd   = (eq / eq.cummax() - 1).min()
    calmar  = ann_ret / abs(maxdd) if maxdd != 0 else float("nan")
    return dict(ret=ann_ret*100, sr=sr, maxdd=maxdd*100, calmar=calmar, n=len(r))


def main():
    print("Loading data…")
    data_dict = load_validated_data("data/processed")
    base_cfg  = ModelConfig()

    # ── Run all grid configs once ─────────────────────────────────────────────
    grid = []
    for wm, wc in product(WM_VALUES, WC_VALUES):
        wv = round(1.0 - wm - wc, 2)
        if wv < WV_MIN:
            continue
        grid.append((wm, wc, wv))

    print(f"Running {len(grid)} grid configs once on full history…")
    configs = []
    for wm, wc, wv in grid:
        cfg = replace(base_cfg, weight_momentum=wm,
                      weight_volatility=wv, weight_correlation=wc)
        res = run_backtest(data_dict, cfg)
        configs.append(dict(wm=wm, wv=wv, wc=wc, rets=res.monthly_returns))
        is_base = (wm == 0.50 and wc == 0.25)
        is_cand = (wm == 0.65 and wc == 0.10)
        tag = " ← baseline" if is_base else (" ← candidate" if is_cand else "")
        print(f"  wM={wm:.2f} wV={wv:.2f} wC={wc:.2f}{tag}")

    df_all = pd.DataFrame(configs)

    baseline_rets  = df_all[(df_all["wm"]==0.50) & (df_all["wc"]==0.25)].iloc[0]["rets"]
    candidate_rets = df_all[(df_all["wm"]==0.65) & (df_all["wc"]==0.10)].iloc[0]["rets"]

    # ── Walk-forward ──────────────────────────────────────────────────────────
    print("\n" + "=" * 100)
    print("WALK-FORWARD RESULTS")
    print("=" * 100)

    fold_results = []

    for fold_name, tr_s, tr_e, te_s, te_e in FOLDS:
        # Find best config by training Sharpe
        best_sr   = -np.inf
        best_cfg  = None
        for _, row in df_all.iterrows():
            sr = sharpe(row["rets"], tr_s, tr_e)
            if not np.isnan(sr) and sr > best_sr:
                best_sr  = sr
                best_cfg = row

        # Evaluate winner on test window
        winner_test_sr   = sharpe(best_cfg["rets"],   te_s, te_e)
        baseline_test_sr = sharpe(baseline_rets,       te_s, te_e)
        cand_test_sr     = sharpe(candidate_rets,      te_s, te_e)
        winner_train_sr  = best_sr

        fold_results.append(dict(
            fold=fold_name, tr_e=tr_e, te_s=te_s, te_e=te_e,
            winner_wm=best_cfg["wm"], winner_wv=best_cfg["wv"], winner_wc=best_cfg["wc"],
            train_sr=winner_train_sr,
            winner_test_sr=winner_test_sr,
            baseline_test_sr=baseline_test_sr,
            candidate_test_sr=cand_test_sr,
            winner_beats_base=winner_test_sr > baseline_test_sr,
            cand_beats_base=cand_test_sr > baseline_test_sr,
        ))

        print(f"\n{fold_name}  (train → {tr_e[:7]}  |  test {te_s[:7]} → {te_e[:7]})")
        print(f"  Best train config: wM={best_cfg['wm']:.2f} wV={best_cfg['wv']:.2f} "
              f"wC={best_cfg['wc']:.2f}  train SR={winner_train_sr:.3f}")
        print(f"  Winner on test :  SR={winner_test_sr:.3f}")
        print(f"  Baseline on test: SR={baseline_test_sr:.3f}  "
              f"({'winner beats' if winner_test_sr > baseline_test_sr else 'baseline beats'})")
        print(f"  Candidate on test (wM=0.65/wV=0.25/wC=0.10): SR={cand_test_sr:.3f}  "
              f"({'candidate beats' if cand_test_sr > baseline_test_sr else 'baseline beats'})")

    # ── Summary ───────────────────────────────────────────────────────────────
    fr = pd.DataFrame(fold_results)

    print("\n" + "=" * 100)
    print("SUMMARY")
    print(f"{'Fold':<8} {'Train end':<12} {'Test window':<24} {'Winner config':<24} "
          f"{'Winner SR':>9} {'Base SR':>8} {'Cand SR':>8}")
    print("-" * 100)
    for _, r in fr.iterrows():
        print(f"{r['fold']:<8} {r['tr_e'][:7]:<12} "
              f"{r['te_s'][:7]}→{r['te_e'][:7]:<14} "
              f"wM={r['winner_wm']:.2f}/wV={r['winner_wv']:.2f}/wC={r['winner_wc']:.2f}  "
              f"{r['winner_test_sr']:>9.3f} {r['baseline_test_sr']:>8.3f} "
              f"{r['candidate_test_sr']:>8.3f}")

    n_winner_beats = fr["winner_beats_base"].sum()
    n_cand_beats   = fr["cand_beats_base"].sum()
    print(f"\nWalk-forward winner beats baseline: {n_winner_beats}/{len(fr)} folds")
    print(f"Fixed candidate (wM=0.65) beats baseline: {n_cand_beats}/{len(fr)} folds")

    # ── wM of walk-forward winners ────────────────────────────────────────────
    print(f"\nwM of walk-forward winner per fold:")
    for _, r in fr.iterrows():
        arrow = "←" if r["winner_wm"] >= 0.60 else ""
        print(f"  {r['fold']}: wM={r['winner_wm']:.2f} wC={r['winner_wc']:.2f}  {arrow}")

    wm_vals = fr["winner_wm"].tolist()
    print(f"\n  Mean winner wM: {np.mean(wm_vals):.3f}")
    print(f"  Min  winner wM: {np.min(wm_vals):.2f}")
    print(f"  Max  winner wM: {np.max(wm_vals):.2f}")
    print(f"  Folds where winner wM >= 0.60: {sum(w >= 0.60 for w in wm_vals)}/{len(wm_vals)}")

    # ── Stacked OOS: concatenate test windows for winner and baseline ─────────
    print(f"\n{'='*100}")
    print("STACKED OOS — concatenate all test windows")
    print("(each fold's test period uses the config that won that fold's training window)\n")

    winner_rets_stacked   = []
    baseline_rets_stacked = []
    cand_rets_stacked     = []

    for fdata, (fold_name, tr_s, tr_e, te_s, te_e) in zip(fold_results, FOLDS):
        # winner config for this fold
        wmatch = df_all[
            (df_all["wm"] == fdata["winner_wm"]) &
            (df_all["wv"] == fdata["winner_wv"]) &
            (df_all["wc"] == fdata["winner_wc"])
        ].iloc[0]["rets"]
        w_slice = wmatch[(wmatch.index >= te_s) & (wmatch.index <= te_e)]
        b_slice = baseline_rets[(baseline_rets.index >= te_s) & (baseline_rets.index <= te_e)]
        c_slice = candidate_rets[(candidate_rets.index >= te_s) & (candidate_rets.index <= te_e)]
        winner_rets_stacked.append(w_slice)
        baseline_rets_stacked.append(b_slice)
        cand_rets_stacked.append(c_slice)

    for label, stacked in [("Walk-forward winner (adaptive)", winner_rets_stacked),
                             ("Fixed candidate  wM=0.65/wV=0.25/wC=0.10", cand_rets_stacked),
                             ("Baseline         wM=0.50/wV=0.25/wC=0.25", baseline_rets_stacked)]:
        combined = pd.concat(stacked).sort_index()
        m = full_metrics(combined, combined.index.min().strftime("%Y-%m-%d"),
                         combined.index.max().strftime("%Y-%m-%d"))
        print(f"  {label}")
        print(f"    Ret={m['ret']:>+6.2f}%  SR={m['sr']:.3f}  MaxDD={m['maxdd']:.2f}%  "
              f"Calmar={m['calmar']:.3f}  n={m['n']} months\n")


if __name__ == "__main__":
    main()
