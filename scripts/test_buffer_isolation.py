"""
Buffer isolation test: sma_ratio vs sma_carry vs sma200.

What we're testing:
  sma_ratio:  buffer=±1%, carry=YES  (baseline)
  sma_carry:  buffer=0%,  carry=YES  (isolates buffer effect)
  sma200:     buffer=0%,  carry=NO   (plain comparison)

IS: 2004-01-01 → 2017-12-31
Walk-forward windows: 7 x 2-year OOS windows spanning 2010-2023.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import logging
import numpy as np
import pandas as pd
from dataclasses import replace

from config.default_config import ModelConfig
from src.backtest.engine import run_backtest
from src.data.loader import load_validated_data

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)

IS_START   = "2004-01-01"
IS_END     = "2017-12-31"
FULL_START = "2004-01-01"
FULL_END   = "2026-04-10"

WF_WINDOWS = [
    ("WF1", "2010-01", "2011-12"),
    ("WF2", "2012-01", "2013-12"),
    ("WF3", "2014-01", "2015-12"),
    ("WF4", "2016-01", "2017-12"),
    ("WF5", "2018-01", "2019-12"),
    ("WF6", "2020-01", "2021-12"),
    ("WF7", "2022-01", "2023-12"),
]

METHODS = ["sma_ratio", "sma_carry", "sma200"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_config(trend_method: str, start: str, end: str) -> ModelConfig:
    """Build a ModelConfig with the given trend method and date range."""
    base = ModelConfig()
    return replace(
        base,
        trend_method=trend_method,
        backtest_start=start,
        backtest_end=end,
        momentum_blend=True,
        momentum_blend_lookbacks=[21, 63, 126],
        transaction_cost=0.0005,
    )


def _slice_metrics(rets: pd.Series, start: str, end: str) -> dict:
    """Compute Sharpe, Ann.Ret, Ann.Vol, and MDD on a slice of monthly returns."""
    r = rets[(rets.index >= start) & (rets.index <= end)]
    if len(r) < 3:
        nan = float("nan")
        return {"sharpe": nan, "ann_ret": nan, "ann_vol": nan, "mdd": nan}

    ann_factor = 12.0
    ann_ret = (1 + r).prod() ** (ann_factor / len(r)) - 1
    ann_vol = r.std() * np.sqrt(ann_factor)
    sharpe  = (ann_ret - 0.02) / ann_vol if ann_vol > 0 else float("nan")

    eq  = (1 + r).cumprod()
    mdd = (eq / eq.cummax() - 1).min()

    return {"sharpe": sharpe, "ann_ret": ann_ret, "ann_vol": ann_vol, "mdd": mdd}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Loading data…")
    data_dict = load_validated_data("data/processed")

    # ── Run full-history backtests for all methods ─────────────────────────
    print("Running backtests…")
    full_returns: dict[str, pd.Series] = {}
    for method in METHODS:
        print(f"  [{method}] …", end="", flush=True)
        cfg = _build_config(method, FULL_START, FULL_END)
        result = run_backtest(data_dict, cfg)
        full_returns[method] = result.monthly_returns
        print(" done")

    # ── IS metrics ────────────────────────────────────────────────────────
    is_metrics: dict[str, dict] = {}
    for method in METHODS:
        is_metrics[method] = _slice_metrics(full_returns[method], IS_START, IS_END)

    base_sharpe = is_metrics["sma_ratio"]["sharpe"]

    # ── Walk-forward OOS Sharpe and MDD ───────────────────────────────────
    wf_sharpe: dict[str, list] = {m: [] for m in METHODS}
    wf_mdd:    dict[str, list] = {m: [] for m in METHODS}

    for wname, wf_start, wf_end in WF_WINDOWS:
        for method in METHODS:
            m = _slice_metrics(full_returns[method], wf_start, wf_end)
            wf_sharpe[method].append(m["sharpe"])
            wf_mdd[method].append(m["mdd"])

    # ── Summary statistics ────────────────────────────────────────────────
    summary: dict[str, dict] = {}
    for method in METHODS:
        sharpes = [s for s in wf_sharpe[method] if not np.isnan(s)]
        n_valid = len(sharpes)
        base_sharpes = [s for s in wf_sharpe["sma_ratio"] if not np.isnan(s)]

        mean_s   = np.mean(sharpes)   if n_valid else float("nan")
        median_s = np.median(sharpes) if n_valid else float("nan")
        std_s    = np.std(sharpes)    if n_valid > 1 else float("nan")

        # Win rate vs sma_ratio (only defined for non-baseline methods)
        if method != "sma_ratio":
            paired = [
                (wf_sharpe[method][i], wf_sharpe["sma_ratio"][i])
                for i in range(len(WF_WINDOWS))
                if not np.isnan(wf_sharpe[method][i])
                   and not np.isnan(wf_sharpe["sma_ratio"][i])
            ]
            win_rate = np.mean([a > b for a, b in paired]) if paired else float("nan")
        else:
            win_rate = float("nan")

        # Average rank across WF windows (1 = best Sharpe)
        ranks = []
        for i in range(len(WF_WINDOWS)):
            window_sharpes = {m: wf_sharpe[m][i] for m in METHODS}
            valid_ws = {m: s for m, s in window_sharpes.items() if not np.isnan(s)}
            if len(valid_ws) < 2:
                ranks.append(float("nan"))
                continue
            sorted_methods = sorted(valid_ws, key=lambda m: valid_ws[m], reverse=True)
            rank = sorted_methods.index(method) + 1 if method in sorted_methods else float("nan")
            ranks.append(rank)
        avg_rank = np.nanmean(ranks) if ranks else float("nan")

        summary[method] = {
            "mean_sharpe":   mean_s,
            "median_sharpe": median_s,
            "sharpe_std":    std_s,
            "win_rate":      win_rate,
            "avg_rank":      avg_rank,
        }

    # ══════════════════════════════════════════════════════════════════════
    # OUTPUT
    # ══════════════════════════════════════════════════════════════════════

    print()
    print("=" * 72)
    print("=== BUFFER ISOLATION TEST: sma_ratio vs sma_carry vs sma200 ===")
    print("=" * 72)
    print()
    print("What we're testing:")
    print("  sma_ratio:  buffer=±1%, carry=YES  (baseline)")
    print("  sma_carry:  buffer=0%,  carry=YES  (isolates buffer effect)")
    print("  sma200:     buffer=0%,  carry=NO   (plain comparison)")

    # ── IS Performance ────────────────────────────────────────────────────
    print()
    print(f"IS PERFORMANCE ({IS_START[:4]}–{IS_END[:4]}):")
    print(f"{'Method':<12}{'Sharpe':>8}{'Ann.Ret':>10}{'Ann.Vol':>10}{'MDD':>10}{'ΔSharpe':>10}")
    print("-" * 60)
    for method in METHODS:
        m = is_metrics[method]
        sharpe  = m["sharpe"]
        ann_ret = m["ann_ret"]
        ann_vol = m["ann_vol"]
        mdd     = m["mdd"]
        if method == "sma_ratio":
            delta_str = "      —"
        else:
            delta = sharpe - base_sharpe
            delta_str = f"{delta:+.3f}" if not np.isnan(delta) else "    N/A"
        s_str  = f"{sharpe:.3f}"  if not np.isnan(sharpe)  else "  N/A"
        r_str  = f"{ann_ret*100:.2f}%" if not np.isnan(ann_ret) else "  N/A"
        v_str  = f"{ann_vol*100:.2f}%" if not np.isnan(ann_vol) else "  N/A"
        dd_str = f"{mdd*100:.2f}%"    if not np.isnan(mdd)     else "  N/A"
        print(f"{method:<12}{s_str:>8}{r_str:>10}{v_str:>10}{dd_str:>10}{delta_str:>10}")

    # ── Walk-forward OOS Sharpe ───────────────────────────────────────────
    print()
    print("WALK-FORWARD OOS SHARPE:")
    header = f"{'Window':<8}{'Period':<14}{'sma_ratio':>11}{'sma_carry':>11}{'sma200':>9}{'Best':>10}"
    print(header)
    print("-" * 65)
    for i, (wname, wf_start, wf_end) in enumerate(WF_WINDOWS):
        period = f"{wf_start}–{wf_end}"
        row_sharpes = {m: wf_sharpe[m][i] for m in METHODS}
        valid_ws = {m: s for m, s in row_sharpes.items() if not np.isnan(s)}
        best = max(valid_ws, key=lambda m: valid_ws[m]) if valid_ws else "N/A"

        def _fmt_s(s: float) -> str:
            return f"{s:.3f}" if not np.isnan(s) else "  N/A"

        print(
            f"{wname:<8}{period:<14}"
            f"{_fmt_s(row_sharpes['sma_ratio']):>11}"
            f"{_fmt_s(row_sharpes['sma_carry']):>11}"
            f"{_fmt_s(row_sharpes['sma200']):>9}"
            f"{best:>10}"
        )

    # ── Walk-forward OOS MDD ──────────────────────────────────────────────
    print()
    print("WALK-FORWARD OOS MDD:")
    print(f"{'Window':<8}{'Period':<14}{'sma_ratio':>11}{'sma_carry':>11}{'sma200':>9}")
    print("-" * 55)
    for i, (wname, wf_start, wf_end) in enumerate(WF_WINDOWS):
        period = f"{wf_start}–{wf_end}"

        def _fmt_dd(d: float) -> str:
            return f"{d*100:.2f}%" if not np.isnan(d) else "  N/A"

        print(
            f"{wname:<8}{period:<14}"
            f"{_fmt_dd(wf_mdd['sma_ratio'][i]):>11}"
            f"{_fmt_dd(wf_mdd['sma_carry'][i]):>11}"
            f"{_fmt_dd(wf_mdd['sma200'][i]):>9}"
        )

    # ── Summary ────────────────────────────────────────────────────────────
    print()
    print("SUMMARY:")
    print(f"{'Metric':<22}{'sma_ratio':>11}{'sma_carry':>11}{'sma200':>9}")
    print("-" * 55)

    def _fmt_val(v: float, fmt: str = ".3f") -> str:
        return format(v, fmt) if not np.isnan(v) else "  N/A"

    metrics_rows = [
        ("Mean OOS Sharpe",   "mean_sharpe",   ".3f"),
        ("Median OOS Sharpe", "median_sharpe", ".3f"),
        ("Sharpe Std Dev",    "sharpe_std",    ".3f"),
    ]
    for label, key, fmt in metrics_rows:
        row = f"{label:<22}"
        for method in METHODS:
            v = summary[method][key]
            row += f"{_fmt_val(v, fmt):>11}" if method != "sma200" else f"{_fmt_val(v, fmt):>9}"
        print(row)

    # Win rate row (—  for baseline)
    row = f"{'Win rate vs base':<22}{'—':>11}"
    for method in ["sma_carry", "sma200"]:
        v = summary[method]["win_rate"]
        cell = f"{v*100:.1f}%" if not np.isnan(v) else "  N/A"
        width = 11 if method == "sma_carry" else 9
        row += f"{cell:>{width}}"
    print(row)

    # Avg rank row
    row = f"{'Avg rank':<22}"
    for i, method in enumerate(METHODS):
        v = summary[method]["avg_rank"]
        cell = _fmt_val(v, ".1f")
        width = 11 if i < 2 else 9
        row += f"{cell:>{width}}"
    print(row)

    # ── Verdict ────────────────────────────────────────────────────────────
    print()
    # Determine winner by mean OOS Sharpe
    mean_sharpes = {m: summary[m]["mean_sharpe"] for m in METHODS}
    valid_means = {m: s for m, s in mean_sharpes.items() if not np.isnan(s)}
    winner = max(valid_means, key=lambda m: valid_means[m]) if valid_means else "N/A"

    sc_vs_sr = mean_sharpes.get("sma_carry", float("nan")) - mean_sharpes.get("sma_ratio", float("nan"))
    s2_vs_sr = mean_sharpes.get("sma200",    float("nan")) - mean_sharpes.get("sma_ratio", float("nan"))
    sc_vs_s2 = mean_sharpes.get("sma_carry", float("nan")) - mean_sharpes.get("sma200",   float("nan"))

    # Infer what the results tell us
    if not np.isnan(sc_vs_sr) and not np.isnan(s2_vs_sr):
        if abs(sc_vs_sr) < 0.02:
            buffer_verdict = "buffer has negligible effect on OOS Sharpe"
        elif sc_vs_sr > 0:
            buffer_verdict = "removing the buffer improves OOS Sharpe (buffer hurts)"
        else:
            buffer_verdict = "the ±1% buffer is beneficial OOS (reduces whipsawing)"

        if abs(sc_vs_s2) < 0.02:
            carry_verdict = "carry vs no-carry produces identical results (as expected for zero-buffer binary signal)"
        elif sc_vs_s2 > 0:
            carry_verdict = "carry logic adds value even at zero buffer"
        else:
            carry_verdict = "carry logic hurts at zero buffer — pure daily recompute is cleaner"
    else:
        buffer_verdict = "insufficient data"
        carry_verdict  = "insufficient data"

    sc_wr = summary["sma_carry"]["win_rate"]
    s2_wr = summary["sma200"]["win_rate"]

    print(f"VERDICT: Winner by mean OOS Sharpe = {winner}")
    print(f"  sma_carry vs sma_ratio: ΔSharpe = {sc_vs_sr:+.3f} — {buffer_verdict}.")
    print(f"  sma_carry vs sma200:    ΔSharpe = {sc_vs_s2:+.3f} — {carry_verdict}.")
    if not np.isnan(sc_wr) and not np.isnan(s2_wr):
        print(f"  Win rates vs baseline: sma_carry={sc_wr*100:.1f}%, sma200={s2_wr*100:.1f}%.")
    print()


if __name__ == "__main__":
    main()
