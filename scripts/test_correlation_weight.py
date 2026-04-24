"""
Correlation weight sweep: wC from 0.00 to 0.20 in steps of 0.05.

Keeps wM + wV + wC = 1.0 by redistributing the remaining weight between
wM and wV in the baseline 2.6:1 ratio (wM=0.65, wV=0.25 at baseline wC=0.10).
Reports IS (2004-2017) and walk-forward OOS Sharpe/Ann.Ret/MDD.
"""

import sys
sys.path.insert(0, '/Users/GualyCr/Desktop/AMAAM/amaam/.claude/worktrees/reverent-mayer')

from config.default_config import ModelConfig
from src.backtest.engine import run_backtest
from src.data.loader import load_validated_data
from pathlib import Path
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ann_ret(r: pd.Series) -> float:
    """Annualised return from monthly return series."""
    r = r.dropna()
    return (1 + r.mean()) ** 12 - 1


def ann_vol(r: pd.Series) -> float:
    """Annualised volatility from monthly return series."""
    return r.std() * np.sqrt(12)


def sharpe(r: pd.Series) -> float:
    """Annualised Sharpe ratio (no risk-free rate adjustment for comparison)."""
    r = r.dropna()
    if len(r) < 3:
        return float("nan")
    v = ann_vol(r)
    return ann_ret(r) / v if v > 0 else float("nan")


def mdd(r: pd.Series) -> float:
    """Maximum drawdown from monthly return series."""
    r = r.dropna()
    if len(r) == 0:
        return float("nan")
    eq = (1 + r).cumprod()
    return (eq / eq.cummax() - 1).min()


def slice_r(result, start: str, end: str) -> pd.Series:
    """Slice monthly returns to a Period range [start, end] inclusive."""
    mr = result.monthly_returns.copy()
    mr.index = mr.index.to_period("M")
    return mr[(mr.index >= pd.Period(start, "M")) & (mr.index <= pd.Period(end, "M"))]


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

data_dir = Path('/Users/GualyCr/Desktop/AMAAM/amaam/.claude/worktrees/reverent-mayer/data/processed')
data_dict = load_validated_data(data_dir)

# ---------------------------------------------------------------------------
# Build weight configurations (wM:wV = 2.6:1 maintained)
# ---------------------------------------------------------------------------

wc_values = [0.00, 0.05, 0.10, 0.15, 0.20]
configs = []
for wc in wc_values:
    wm = round((1 - wc) * 0.65 / 0.90, 4)
    wv = round((1 - wc) * 0.25 / 0.90, 4)
    configs.append((wc, wm, wv))

print("=== CORRELATION WEIGHT SWEEP: wC = 0.00 → 0.20 ===")
print()
print("Weight combinations (wM + wV + wC = 1.0, maintaining wM:wV = 2.6:1):")
for wc, wm, wv in configs:
    marker = "  ← baseline" if abs(wc - 0.10) < 1e-9 else ""
    total = round(wm + wv + wc, 4)
    print(f"  wC={wc:.2f}:  wM={wm:.4f}  wV={wv:.4f}  wC={wc:.4f}  [sum={total}]{marker}")
print()

# ---------------------------------------------------------------------------
# Run all 5 backtests
# ---------------------------------------------------------------------------

results = {}
for wc, wm, wv in configs:
    cfg = ModelConfig()
    cfg.weight_correlation = wc
    cfg.weight_momentum = wm
    cfg.weight_volatility = wv
    key = f"wC={wc:.2f}"
    print(f"Running {key} (wM={wm}, wV={wv}, wC={wc})...")
    results[key] = run_backtest(data_dict, cfg)

print()

# ---------------------------------------------------------------------------
# IS performance table (2004-2017)
# ---------------------------------------------------------------------------

IS_START = "2004-01"
IS_END   = "2017-12"

baseline_key = "wC=0.10"

print("IS PERFORMANCE (2004–2017):")
header = f"{'Config':<10}{'wM':>8}{'wV':>8}{'wC':>8}{'Sharpe':>8}{'Ann.Ret':>9}{'Ann.Vol':>9}{'MDD':>9}{'ΔSharpe':>10}"
print(header)
print("-" * len(header))

is_metrics = {}
for wc, wm, wv in configs:
    key = f"wC={wc:.2f}"
    r   = slice_r(results[key], IS_START, IS_END)
    sh  = sharpe(r)
    ar  = ann_ret(r)
    av  = ann_vol(r)
    md  = mdd(r)
    is_metrics[key] = {"sharpe": sh, "ann_ret": ar, "ann_vol": av, "mdd": md}

base_sh = is_metrics[baseline_key]["sharpe"]

for wc, wm, wv in configs:
    key = f"wC={wc:.2f}"
    m   = is_metrics[key]
    delta = m["sharpe"] - base_sh
    delta_str = f"{delta:+.3f}" if key != baseline_key else "—"
    print(
        f"{key:<10}{wm:>8.4f}{wv:>8.4f}{wc:>8.3f}"
        f"{m['sharpe']:>8.3f}{m['ann_ret']*100:>8.2f}%"
        f"{m['ann_vol']*100:>8.2f}%{m['mdd']*100:>8.2f}%"
        f"{delta_str:>10}"
    )

print()

# ---------------------------------------------------------------------------
# Walk-forward windows
# ---------------------------------------------------------------------------

wf_windows = [
    ("WF1", "2010-01", "2011-12"),
    ("WF2", "2012-01", "2013-12"),
    ("WF3", "2014-01", "2015-12"),
    ("WF4", "2016-01", "2017-12"),
    ("WF5", "2018-01", "2019-12"),
    ("WF6", "2020-01", "2021-12"),
    ("WF7", "2022-01", "2023-12"),
]

keys = [f"wC={wc:.2f}" for wc, _, _ in configs]

# --- Sharpe table ---
print("WALK-FORWARD OOS SHARPE:")
col_w = 9
hdr_parts = ["Window", "Period   "] + [f"{k:>{col_w}}" for k in keys] + ["  Best"]
print("  ".join(hdr_parts))
print("-" * (len("  ".join(hdr_parts)) + 5))

wf_sharpes: dict[str, list] = {k: [] for k in keys}  # collect for summary

for wfname, wf_start, wf_end in wf_windows:
    period_label = f"{wf_start[:4]}-{wf_end[2:4]}"
    sharpes = {}
    for k in keys:
        r = slice_r(results[k], wf_start, wf_end)
        sh = sharpe(r)
        sharpes[k] = sh
        wf_sharpes[k].append(sh)

    best_key = max(sharpes, key=lambda x: sharpes[x] if not np.isnan(sharpes[x]) else -999)
    vals = "  ".join(f"{sharpes[k]:>{col_w}.3f}" for k in keys)
    print(f"{wfname:<6}  {period_label:<9}  {vals}  {best_key}")

print()

# --- MDD table ---
print("WALK-FORWARD OOS MDD:")
hdr_parts2 = ["Window", "Period   "] + [f"{k:>{col_w}}" for k in keys]
print("  ".join(hdr_parts2))
print("-" * (len("  ".join(hdr_parts2)) + 5))

for wfname, wf_start, wf_end in wf_windows:
    period_label = f"{wf_start[:4]}-{wf_end[2:4]}"
    mdds = {}
    for k in keys:
        r = slice_r(results[k], wf_start, wf_end)
        mdds[k] = mdd(r)
    vals = "  ".join(f"{mdds[k]*100:>{col_w}.2f}%" for k in keys)
    print(f"{wfname:<6}  {period_label:<9}  {vals}")

print()

# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

print("SUMMARY:")
metrics_rows = [
    "Mean OOS Sharpe",
    "Median OOS Sharpe",
    "Sharpe Std Dev",
    "Win rate vs base",
    "Avg rank",
]

label_w = 20
col_w2  = 9

# Column header
header_line = f"{'Metric':<{label_w}}" + "  ".join(f"{k:>{col_w2}}" for k in keys)
print(header_line)
print("-" * len(header_line))

# Pre-compute per-config summary values
summary: dict[str, dict] = {}
base_sharpes = wf_sharpes[baseline_key]

for k in keys:
    sh_list  = [s for s in wf_sharpes[k] if not np.isnan(s)]
    base_list = [s for s in base_sharpes if not np.isnan(s)]
    n = min(len(sh_list), len(base_list))
    wins = sum(sh_list[i] > base_list[i] for i in range(n)) if n > 0 else 0
    win_rate = wins / n if n > 0 else float("nan")

    # Average rank across windows (1 = best)
    ranks = []
    for idx, (_, wf_start, wf_end) in enumerate(wf_windows):
        window_sharpes = {}
        for k2 in keys:
            sh_list2 = wf_sharpes[k2]
            window_sharpes[k2] = sh_list2[idx] if idx < len(sh_list2) else float("nan")
        # rank 1 = highest Sharpe
        sorted_keys = sorted(
            window_sharpes.keys(),
            key=lambda x: window_sharpes[x] if not np.isnan(window_sharpes[x]) else -999,
            reverse=True,
        )
        rank = sorted_keys.index(k) + 1
        ranks.append(rank)

    summary[k] = {
        "mean":     float(np.mean(sh_list)) if sh_list else float("nan"),
        "median":   float(np.median(sh_list)) if sh_list else float("nan"),
        "std":      float(np.std(sh_list, ddof=1)) if len(sh_list) > 1 else float("nan"),
        "win_rate": win_rate,
        "avg_rank": float(np.mean(ranks)) if ranks else float("nan"),
    }

# Print rows
for metric_name in metrics_rows:
    if metric_name == "Mean OOS Sharpe":
        vals = "  ".join(f"{summary[k]['mean']:>{col_w2}.3f}" for k in keys)
    elif metric_name == "Median OOS Sharpe":
        vals = "  ".join(f"{summary[k]['median']:>{col_w2}.3f}" for k in keys)
    elif metric_name == "Sharpe Std Dev":
        vals = "  ".join(f"{summary[k]['std']:>{col_w2}.3f}" for k in keys)
    elif metric_name == "Win rate vs base":
        parts = []
        for k in keys:
            if k == baseline_key:
                parts.append(f"{'—':>{col_w2}}")
            else:
                parts.append(f"{summary[k]['win_rate']*100:>{col_w2-1}.1f}%")
        vals = "  ".join(parts)
    elif metric_name == "Avg rank":
        vals = "  ".join(f"{summary[k]['avg_rank']:>{col_w2}.1f}" for k in keys)
    else:
        vals = ""
    print(f"{metric_name:<{label_w}}{vals}")

print()

# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------

best_key_oos = max(keys, key=lambda k: summary[k]["mean"])
best_mean    = summary[best_key_oos]["mean"]
base_mean    = summary[baseline_key]["mean"]
delta_mean   = best_mean - base_mean

# Which wC values beat baseline OOS?
beaters = [k for k in keys if k != baseline_key and summary[k]["mean"] > base_mean]

print("VERDICT:")
if best_key_oos == baseline_key:
    verdict = (
        f"The baseline wC=0.10 achieves the highest mean OOS Sharpe ({base_mean:.3f}), "
        f"suggesting the correlation factor at its current weight is already well-calibrated. "
        f"Reducing or increasing correlation weight degrades average walk-forward performance."
    )
elif best_key_oos in beaters:
    wc_best_val = float(best_key_oos.split("=")[1])
    direction   = "higher" if wc_best_val > 0.10 else "lower"
    verdict = (
        f"{best_key_oos} achieves the highest mean OOS Sharpe ({best_mean:.3f}, "
        f"+{delta_mean:.3f} vs baseline {base_mean:.3f}). "
        f"A {direction} correlation weight improves walk-forward performance, "
        f"implying the correlation factor adds {'more' if direction == 'higher' else 'less'} "
        f"discriminating power than the baseline assumes. "
        f"{'All tested wC > 0.10 values beat baseline, confirming more correlation weight is better.' if all(summary[k]['mean'] > base_mean for k in keys if float(k.split('=')[1]) > 0.10) else ''}"
        f"{'All tested wC < 0.10 values beat baseline, confirming less correlation weight is better.' if all(summary[k]['mean'] > base_mean for k in keys if float(k.split('=')[1]) < 0.10) else ''}"
    )
else:
    verdict = (
        f"{best_key_oos} leads on mean OOS Sharpe ({best_mean:.3f}). "
        f"No clear monotonic trend: adjusting wC does not consistently improve or hurt performance."
    )

print(verdict)
