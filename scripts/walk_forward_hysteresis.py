"""
Walk-forward validation of selection-hysteresis configurations.

Tests three exit-buffer settings against the same six expanding-window folds
used by the canonical weight walk-forward (walk_forward.py):

  Fold 1: train 2007-08 → 2012-12,  test 2013-01 → 2014-12
  Fold 2: train 2007-08 → 2014-12,  test 2015-01 → 2016-12
  Fold 3: train 2007-08 → 2016-12,  test 2017-01 → 2018-12
  Fold 4: train 2007-08 → 2018-12,  test 2019-01 → 2020-12
  Fold 5: train 2007-08 → 2020-12,  test 2021-01 → 2022-12
  Fold 6: train 2007-08 → 2022-12,  test 2023-01 → 2024-12

For each fold the script:
  1. Picks the buffer with the best Sharpe on the training window.
  2. Evaluates every buffer on the test window.
  3. Reports which buffer wins each test window.

Final section: stacked OOS — concatenate all six test windows and compute
Sharpe, return, max-drawdown, and Calmar for each buffer and for the
adaptive picker (the buffer that won each fold's training window).

Decision rule: adopt a non-zero buffer only if it wins ≥ 4/6 test folds
AND the stacked OOS Sharpe improves over Buffer-0.

Usage
-----
    python3.13 scripts/walk_forward_hysteresis.py
    python3.13 scripts/walk_forward_hysteresis.py --data-dir data/processed
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from dataclasses import replace

from config.default_config import ModelConfig
from src.data.loader import load_validated_data
from src.backtest.engine import run_backtest

BUFFERS = [0, 1, 2]

FOLDS = [
    ("Fold 1", "2007-08-01", "2012-12-31", "2013-01-01", "2014-12-31"),
    ("Fold 2", "2007-08-01", "2014-12-31", "2015-01-01", "2016-12-31"),
    ("Fold 3", "2007-08-01", "2016-12-31", "2017-01-01", "2018-12-31"),
    ("Fold 4", "2007-08-01", "2018-12-31", "2019-01-01", "2020-12-31"),
    ("Fold 5", "2007-08-01", "2020-12-31", "2021-01-01", "2022-12-31"),
    ("Fold 6", "2007-08-01", "2022-12-31", "2023-01-01", "2024-12-31"),
]


def _sharpe(rets: pd.Series, start: str, end: str, rf: float = 0.02) -> float:
    r = rets[(rets.index >= start) & (rets.index <= end)]
    if len(r) < 6:
        return float("nan")
    ann_ret = (1 + r).prod() ** (12 / len(r)) - 1
    ann_vol = r.std() * np.sqrt(12)
    return (ann_ret - rf) / ann_vol if ann_vol > 0 else float("nan")


def _full_metrics(rets: pd.Series, start: str, end: str, rf: float = 0.02) -> dict:
    r = rets[(rets.index >= start) & (rets.index <= end)]
    if len(r) < 6:
        return dict(ret=float("nan"), sr=float("nan"),
                    maxdd=float("nan"), calmar=float("nan"), n=0)
    ann_ret = (1 + r).prod() ** (12 / len(r)) - 1
    ann_vol = r.std() * np.sqrt(12)
    sr      = (ann_ret - rf) / ann_vol if ann_vol > 0 else float("nan")
    eq      = (1 + r).cumprod()
    maxdd   = (eq / eq.cummax() - 1).min()
    calmar  = ann_ret / abs(maxdd) if maxdd != 0 else float("nan")
    return dict(ret=ann_ret * 100, sr=sr, maxdd=maxdd * 100, calmar=calmar, n=len(r))


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/processed")
    args = parser.parse_args()

    print("Loading data…")
    data = load_validated_data(args.data_dir)
    base_cfg = ModelConfig()

    # ── Run each buffer config once on full history ───────────────────────────
    print(f"Running {len(BUFFERS)} buffer configs on full history…")
    results: dict[int, pd.Series] = {}
    for buf in BUFFERS:
        cfg = replace(base_cfg, selection_exit_buffer=buf)
        res = run_backtest(data, cfg)
        results[buf] = res.monthly_returns
        full = _full_metrics(
            res.monthly_returns,
            res.monthly_returns.index.min().strftime("%Y-%m-%d"),
            res.monthly_returns.index.max().strftime("%Y-%m-%d"),
        )
        print(f"  Buffer-{buf}: SR={full['sr']:.3f}  Ret={full['ret']:+.2f}%  "
              f"MaxDD={full['maxdd']:.2f}%  Calmar={full['calmar']:.3f}")

    # ── Per-fold walk-forward ─────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("WALK-FORWARD FOLDS")
    print("=" * 90)

    fold_records = []

    for fold_name, tr_s, tr_e, te_s, te_e in FOLDS:
        train_srs = {buf: _sharpe(results[buf], tr_s, tr_e) for buf in BUFFERS}
        test_srs  = {buf: _sharpe(results[buf], te_s, te_e) for buf in BUFFERS}

        # Best buffer on training window (break ties in favour of lower buffer).
        best_buf = min(
            (buf for buf in BUFFERS if not np.isnan(train_srs[buf])),
            key=lambda b: (-train_srs[b], b),
        )

        print(f"\n{fold_name}  (train → {tr_e[:7]}  |  test {te_s[:7]} → {te_e[:7]})")
        print(f"  Train Sharpe:  " +
              "  ".join(f"Buf-{b}={train_srs[b]:.3f}" for b in BUFFERS) +
              f"  → best: Buffer-{best_buf}")
        print(f"  Test  Sharpe:  " +
              "  ".join(f"Buf-{b}={test_srs[b]:.3f}" for b in BUFFERS))

        # How many buffers beat Buffer-0 on the test window?
        for buf in BUFFERS[1:]:
            beat = test_srs[buf] > test_srs[0]
            print(f"  Buffer-{buf} vs Buffer-0 on test: "
                  f"{'BEATS' if beat else 'LOSES'}  "
                  f"(Δ SR = {test_srs[buf] - test_srs[0]:+.3f})")

        fold_records.append(dict(
            fold=fold_name, tr_e=tr_e, te_s=te_s, te_e=te_e,
            best_buf=best_buf,
            **{f"train_{b}": train_srs[b] for b in BUFFERS},
            **{f"test_{b}": test_srs[b] for b in BUFFERS},
        ))

    # ── Summary table ─────────────────────────────────────────────────────────
    fr = pd.DataFrame(fold_records)

    print("\n" + "=" * 90)
    print("FOLD SUMMARY — Test Sharpe")
    header = f"{'Fold':<8}  {'Test window':<20}  {'Buf-0':>7}  {'Buf-1':>7}  {'Buf-2':>7}  {'Train winner':>13}"
    print(header)
    print("-" * len(header))
    for _, r in fr.iterrows():
        print(f"{r['fold']:<8}  {r['te_s'][:7]}→{r['te_e'][:7]:<10}  "
              f"{r['test_0']:>7.3f}  {r['test_1']:>7.3f}  {r['test_2']:>7.3f}  "
              f"     Buffer-{int(r['best_buf'])}")

    # Win counts: how often does each non-zero buffer beat Buffer-0 on test?
    print()
    for buf in BUFFERS[1:]:
        wins = (fr[f"test_{buf}"] > fr["test_0"]).sum()
        avg_delta = (fr[f"test_{buf}"] - fr["test_0"]).mean()
        print(f"  Buffer-{buf} beats Buffer-0 on test: {wins}/{len(fr)} folds  "
              f"(avg Δ SR = {avg_delta:+.3f})")

    # How consistent is the adaptive picker?
    adaptive_wins = sum(
        fr.loc[i, f"test_{int(fr.loc[i, 'best_buf'])}"] > fr.loc[i, "test_0"]
        for i in fr.index
    )
    print(f"\n  Adaptive picker (best train buffer per fold) beats Buffer-0: "
          f"{adaptive_wins}/{len(fr)} folds")

    # ── Stacked OOS ───────────────────────────────────────────────────────────
    print(f"\n{'='*90}")
    print("STACKED OOS — all six test windows concatenated")
    print()

    stacked: dict[str, list] = {f"buf_{b}": [] for b in BUFFERS}
    stacked["adaptive"] = []

    for fdata, (_, tr_s, tr_e, te_s, te_e) in zip(fold_records, FOLDS):
        for buf in BUFFERS:
            r = results[buf]
            stacked[f"buf_{buf}"].append(r[(r.index >= te_s) & (r.index <= te_e)])
        # Adaptive: the buffer that won this fold's training window.
        best = fdata["best_buf"]
        r = results[best]
        stacked["adaptive"].append(r[(r.index >= te_s) & (r.index <= te_e)])

    labels = {
        "buf_0":    "Buffer-0  (baseline, no hysteresis)",
        "buf_1":    "Buffer-1  (exit at rank > N+1)     ",
        "buf_2":    "Buffer-2  (exit at rank > N+2)     ",
        "adaptive": "Adaptive  (best-train buffer/fold) ",
    }

    stacked_metrics = {}
    for key, slices in stacked.items():
        combined = pd.concat(slices).sort_index()
        m = _full_metrics(
            combined,
            combined.index.min().strftime("%Y-%m-%d"),
            combined.index.max().strftime("%Y-%m-%d"),
        )
        stacked_metrics[key] = m
        print(f"  {labels[key]}  "
              f"SR={m['sr']:.3f}  Ret={m['ret']:>+6.2f}%  "
              f"MaxDD={m['maxdd']:>7.2f}%  Calmar={m['calmar']:.3f}  n={m['n']}mo")

    # ── Verdict ───────────────────────────────────────────────────────────────
    print(f"\n{'='*90}")
    print("VERDICT")
    buf0_oos_sr = stacked_metrics["buf_0"]["sr"]
    for buf in BUFFERS[1:]:
        wins    = int((fr[f"test_{buf}"] > fr["test_0"]).sum())
        oos_sr  = stacked_metrics[f"buf_{buf}"]["sr"]
        beats   = oos_sr > buf0_oos_sr
        adopt   = wins >= 4 and beats
        print(f"  Buffer-{buf}: wins {wins}/6 folds  |  stacked OOS SR {oos_sr:.3f} vs "
              f"{buf0_oos_sr:.3f} (Buffer-0)  |  "
              f"{'ADOPT ✓' if adopt else 'REJECT ✗'}")
    print(f"\n  (Adoption threshold: ≥ 4/6 fold wins AND stacked OOS SR > Buffer-0)")


if __name__ == "__main__":
    main()
