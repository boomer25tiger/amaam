"""
Blended lookback comparison: single vs blended correlation and vol windows.

Runs three configurations in IS (2004-01-01 to 2018-01-01) and then
IS + OOS + walk-forward for the two non-baseline configs.

Configurations:
  1. Current  84d single  — corr_lb=84, corr_blend=False, yz_win=84, vol_blend=False
  2. Single  126d         — corr_lb=126, corr_blend=False, yz_win=126, vol_blend=False
  3. Blended [63,126,252] — corr_blend=True, vol_blend=True (lb param unused)

All use: sma_ratio trend, blended momentum [21,63,126], 5bps costs.
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

IS_START  = "2004-01-01"
IS_END    = "2018-01-01"
OOS_START = "2018-01-01"
OOS_END   = ModelConfig().backtest_end
FULL_END  = ModelConfig().backtest_end

# Walk-forward folds: (name, train_start, train_end, test_start, test_end)
FOLDS = [
    ("Fold 1", IS_START, "2012-12-31", "2013-01-01", "2014-12-31"),
    ("Fold 2", IS_START, "2014-12-31", "2015-01-01", "2016-12-31"),
    ("Fold 3", IS_START, "2016-12-31", "2017-01-01", "2018-12-31"),
    ("Fold 4", IS_START, "2018-12-31", "2019-01-01", "2020-12-31"),
    ("Fold 5", IS_START, "2020-12-31", "2021-01-01", "2022-12-31"),
    ("Fold 6", IS_START, "2022-12-31", "2023-01-01", "OOS_END"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _slice_metrics(rets: pd.Series, start: str, end: str) -> dict:
    """Compute performance metrics on a slice of monthly returns."""
    if end == "OOS_END":
        end = OOS_END
    r = rets[(rets.index >= start) & (rets.index <= end)]
    if len(r) < 6:
        return {k: float("nan") for k in
                ["Ann Return", "Vol", "Sharpe", "Sortino", "Calmar",
                 "Max DD", "Worst Year", "% Pos Months"]}

    ann_factor = 12.0
    ann_ret  = (1 + r).prod() ** (ann_factor / len(r)) - 1
    ann_vol  = r.std() * np.sqrt(ann_factor)
    sharpe   = (ann_ret - 0.02) / ann_vol if ann_vol > 0 else float("nan")

    # Sortino: downside deviation
    downside = r[r < 0]
    if len(downside) > 1:
        ann_downside = downside.std() * np.sqrt(ann_factor)
        sortino = (ann_ret - 0.02) / ann_downside if ann_downside > 0 else float("nan")
    else:
        sortino = float("nan")

    eq    = (1 + r).cumprod()
    maxdd = (eq / eq.cummax() - 1).min()
    calmar = ann_ret / abs(maxdd) if maxdd != 0 else float("nan")

    # Worst year
    yearly = r.resample("YE").apply(lambda x: (1 + x).prod() - 1)
    worst_year = yearly.min() if len(yearly) > 0 else float("nan")

    pct_pos = (r > 0).mean()

    return {
        "Ann Return":   ann_ret,
        "Vol":          ann_vol,
        "Sharpe":       sharpe,
        "Sortino":      sortino,
        "Calmar":       calmar,
        "Max DD":       maxdd,
        "Worst Year":   worst_year,
        "% Pos Months": pct_pos,
    }


def _build_config(
    corr_lb: int,
    corr_blend: bool,
    yz_win: int,
    vol_blend: bool,
    start: str = IS_START,
    end: str = FULL_END,
) -> ModelConfig:
    base = ModelConfig()
    return replace(
        base,
        backtest_start=start,
        backtest_end=end,
        trend_method="sma_ratio",
        momentum_blend=True,
        momentum_blend_lookbacks=[21, 63, 126],
        transaction_cost=0.0005,
        correlation_lookback=corr_lb,
        correlation_blend=corr_blend,
        yang_zhang_window=yz_win,
        vol_blend=vol_blend,
    )


def _fmt(val: float, key: str) -> str:
    if np.isnan(val):
        return "   N/A "
    pct_keys = {"Ann Return", "Vol", "Max DD", "Worst Year", "% Pos Months"}
    if key in pct_keys:
        return f"{val * 100:+7.2f}%"
    return f"{val:8.3f} "


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Loading data…")
    data_dict = load_validated_data("data/processed")

    CONFIGS = [
        # label, corr_lb, corr_blend, yz_win, vol_blend
        ("Current  84d single",     84,   False, 84,   False),
        ("Single  126d",            126,  False, 126,  False),
        ("Blended [63,126,252]",    84,   True,  84,   True),
    ]

    METRICS_ORDER = ["Ann Return", "Vol", "Sharpe", "Sortino", "Calmar",
                     "Max DD", "Worst Year", "% Pos Months"]

    # ── Step 1: IS comparison ─────────────────────────────────────────────────
    print("\nRunning IS (2004-01-01 → 2018-01-01) for all 3 configs…")
    is_results: dict[str, dict] = {}
    full_returns: dict[str, pd.Series] = {}

    for label, corr_lb, corr_blend, yz_win, vol_blend in CONFIGS:
        print(f"  [{label}] …", end="", flush=True)
        cfg = _build_config(corr_lb, corr_blend, yz_win, vol_blend,
                            start=IS_START, end=IS_END)
        result = run_backtest(data_dict, cfg)
        is_results[label] = _slice_metrics(result.monthly_returns, IS_START, IS_END)
        print(" done")

    # Also run full-history for OOS / walk-forward (non-baseline only)
    print("\nRunning full history (2004 → 2026) for non-baseline configs…")
    for label, corr_lb, corr_blend, yz_win, vol_blend in CONFIGS[1:]:
        print(f"  [{label}] …", end="", flush=True)
        cfg = _build_config(corr_lb, corr_blend, yz_win, vol_blend,
                            start=IS_START, end=FULL_END)
        result = run_backtest(data_dict, cfg)
        full_returns[label] = result.monthly_returns
        print(" done")

    # Baseline full history (needed for walk-forward reference)
    label0 = CONFIGS[0][0]
    print(f"  [{label0}] …", end="", flush=True)
    cfg0 = _build_config(*CONFIGS[0][1:], start=IS_START, end=FULL_END)
    r0 = run_backtest(data_dict, cfg0)
    full_returns[label0] = r0.monthly_returns
    print(" done")

    # ── Print IS table ────────────────────────────────────────────────────────
    _print_comparison_table(
        "IS COMPARISON (2004-01-01 → 2018-01-01)",
        is_results, METRICS_ORDER, list(is_results.keys()),
    )

    # ── Step 2: IS + OOS for non-baseline configs ─────────────────────────────
    for label, *_ in CONFIGS[1:]:
        rets = full_returns[label]
        is_m  = _slice_metrics(rets, IS_START,  IS_END)
        oos_m = _slice_metrics(rets, OOS_START, OOS_END)

        _print_comparison_table(
            f"IS vs OOS — {label}",
            {"IS  (2004→2018)": is_m, "OOS (2018→2026)": oos_m},
            METRICS_ORDER,
            ["IS  (2004→2018)", "OOS (2018→2026)"],
        )

    # ── Step 3: Walk-forward for non-baseline configs ─────────────────────────
    for label, *_ in CONFIGS[1:]:
        rets = full_returns[label]
        base_rets = full_returns[label0]

        _print_walk_forward(label, rets, base_rets)

    print("\n[Done]")


def _print_comparison_table(
    title: str,
    results: dict,
    metrics: list,
    col_order: list,
) -> None:
    col_w = 20
    label_w = 16

    print(f"\n{'=' * (label_w + col_w * len(col_order) + 4)}")
    print(f"  {title}")
    print(f"{'=' * (label_w + col_w * len(col_order) + 4)}")

    # Header
    hdr = f"{'Metric':<{label_w}}"
    for col in col_order:
        hdr += f"{col:>{col_w}}"
    print(hdr)
    print("-" * (label_w + col_w * len(col_order) + 4))

    pct_keys = {"Ann Return", "Vol", "Max DD", "Worst Year", "% Pos Months"}

    for m in metrics:
        row_vals = {col: results[col][m] for col in col_order}
        valid = {k: v for k, v in row_vals.items() if not np.isnan(v)}

        # Determine best: for Max DD and Worst Year, higher (less negative) is better
        best_key = None
        if valid:
            if m in {"Max DD", "Worst Year"}:
                best_key = max(valid, key=lambda k: valid[k])
            else:
                best_key = max(valid, key=lambda k: valid[k])

        row = f"{m:<{label_w}}"
        for col in col_order:
            val = row_vals[col]
            marker = " ◀" if (col == best_key and len(valid) > 1) else "  "
            if np.isnan(val):
                cell = "N/A"
            elif m in pct_keys:
                cell = f"{val * 100:+.2f}%"
            else:
                cell = f"{val:.3f}"
            row += f"{cell + marker:>{col_w}}"
        print(row)

    print()


def _print_walk_forward(
    label: str,
    rets: pd.Series,
    base_rets: pd.Series,
) -> None:
    print(f"\n{'=' * 90}")
    print(f"  WALK-FORWARD — {label}")
    print(f"{'=' * 90}")
    print(f"{'Fold':<10}{'Period':<28}{'Sharpe':>10}{'Ann Ret':>12}{'Max DD':>12}{'vs Baseline':>14}")
    print("-" * 90)

    for fold_name, tr_s, tr_e, te_s, te_e in FOLDS:
        if te_e == "OOS_END":
            te_e = OOS_END
        m   = _slice_metrics(rets,      te_s, te_e)
        mb  = _slice_metrics(base_rets, te_s, te_e)

        sr_str  = f"{m['Sharpe']:+.3f}" if not np.isnan(m['Sharpe']) else "N/A"
        ret_str = f"{m['Ann Return'] * 100:+.2f}%" if not np.isnan(m['Ann Return']) else "N/A"
        dd_str  = f"{m['Max DD'] * 100:.2f}%"      if not np.isnan(m['Max DD'])     else "N/A"

        if not np.isnan(m['Sharpe']) and not np.isnan(mb['Sharpe']):
            diff = m['Sharpe'] - mb['Sharpe']
            diff_str = f"{diff:+.3f}"
        else:
            diff_str = "N/A"

        period = f"{te_s[:7]} → {te_e[:7]}"
        print(f"{fold_name:<10}{period:<28}{sr_str:>10}{ret_str:>12}{dd_str:>12}{diff_str:>14}")

    print()


if __name__ == "__main__":
    main()
