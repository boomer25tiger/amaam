"""
Test inverse-volatility weighting vs equal weighting (IS only: 2004–2017).

The engine natively supports weighting_scheme = "inverse_volatility" via
src/portfolio/weighting.py, so no post-processing is required.  Config B
sets that field and reruns the backtest; all other parameters are held
constant.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from config.default_config import ModelConfig
from src.backtest.engine import run_backtest
from src.data.loader import load_validated_data

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
IS_START = "2004-01-01"
IS_END = "2017-12-31"

data_dict = load_validated_data(DATA_DIR)

# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------
cfg_a = ModelConfig(backtest_start=IS_START, backtest_end=IS_END, weighting_scheme="equal")
cfg_b = ModelConfig(
    backtest_start=IS_START,
    backtest_end=IS_END,
    weighting_scheme="inverse_volatility",
)

print("Running Config A (equal weight)…")
result_a = run_backtest(data_dict, cfg_a)

print("Running Config B (inverse_volatility)…")
result_b = run_backtest(data_dict, cfg_b)


# ---------------------------------------------------------------------------
# Helper: slice to IS period and re-compute metrics fresh
# ---------------------------------------------------------------------------
from src.backtest.metrics import compute_all_metrics


def is_metrics(result):
    """Return metrics computed on IS monthly returns only."""
    r = result.monthly_returns
    r = r[(r.index >= IS_START) & (r.index <= IS_END)]
    n = len(r)
    if n > 1:
        span = (r.index.max() - r.index.min()).days
        ppy = n / (span / 365.25) if span > 0 else 12.0
    else:
        ppy = 12.0
    return compute_all_metrics(r, risk_free_rate=0.02, periods_per_year=ppy), r


def annual_returns(monthly_r: pd.Series) -> pd.Series:
    """Compound monthly returns to calendar-year totals."""
    return monthly_r.groupby(monthly_r.index.to_period("Y")).apply(
        lambda r: float((1.0 + r).prod() - 1.0)
    )


def stress_return(monthly_r: pd.Series, start: str, end: str) -> float:
    """Cumulative return over a stress window (inclusive)."""
    sub = monthly_r[(monthly_r.index >= start) & (monthly_r.index <= end)]
    if sub.empty:
        return float("nan")
    return float((1.0 + sub).prod() - 1.0)


# ---------------------------------------------------------------------------
# Compute
# ---------------------------------------------------------------------------
m_a, r_a = is_metrics(result_a)
m_b, r_b = is_metrics(result_b)

ann_a = annual_returns(r_a)
ann_b = annual_returns(r_b)

STRESS = {
    "GFC":           ("2007-10-01", "2009-03-31"),
    "Euro Crisis":   ("2010-04-01", "2011-09-30"),
    "Taper Tantrum": ("2013-05-01", "2013-06-30"),
    "China Selloff": ("2015-08-01", "2016-02-29"),
}

# ---------------------------------------------------------------------------
# Print output
# ---------------------------------------------------------------------------
print()
print("=" * 68)
print("=== INVERSE VOL vs EQUAL WEIGHTING (IS ONLY: 2004–2017) ===")
print("=" * 68)
print()
print("Weighting method used: equal (A) / inverse_vol native (B)")
print()

# ── IS Summary ──────────────────────────────────────────────────────────────
def pct(v):
    return f"{v * 100:+.2f}%" if not np.isnan(v) else "   nan"


def pct_plain(v):
    return f"{v * 100:.2f}%" if not np.isnan(v) else "nan"


def ratio(v):
    return f"{v:.3f}" if not np.isnan(v) else " nan"


header = f"{'Metric':<26} {'A Equal':>10}  {'B Inv-Vol':>10}  {'Delta':>10}"
sep = "-" * len(header)

print("IS SUMMARY:")
print(header)
print(sep)

rows = [
    ("Ann. Return",
     pct_plain(m_a["Annualized Return"]),
     pct_plain(m_b["Annualized Return"]),
     pct(m_b["Annualized Return"] - m_a["Annualized Return"])),
    ("Ann. Volatility",
     pct_plain(m_a["Annualized Volatility"]),
     pct_plain(m_b["Annualized Volatility"]),
     pct(m_b["Annualized Volatility"] - m_a["Annualized Volatility"])),
    ("Sharpe",
     ratio(m_a["Sharpe Ratio"]),
     ratio(m_b["Sharpe Ratio"]),
     f"{m_b['Sharpe Ratio'] - m_a['Sharpe Ratio']:+.3f}"),
    ("Sortino",
     ratio(m_a["Sortino Ratio"]),
     ratio(m_b["Sortino Ratio"]),
     f"{m_b['Sortino Ratio'] - m_a['Sortino Ratio']:+.3f}"),
    ("Max Drawdown",
     pct_plain(m_a["Max Drawdown"]),
     pct_plain(m_b["Max Drawdown"]),
     pct(m_b["Max Drawdown"] - m_a["Max Drawdown"])),
    ("Calmar",
     ratio(m_a["Calmar Ratio"]),
     ratio(m_b["Calmar Ratio"]),
     f"{m_b['Calmar Ratio'] - m_a['Calmar Ratio']:+.3f}"),
    ("% Months Positive",
     pct_plain(m_a["% Positive Months"]),
     pct_plain(m_b["% Positive Months"]),
     pct(m_b["% Positive Months"] - m_a["% Positive Months"])),
    ("Best Month",
     pct_plain(m_a["Best Month"]),
     pct_plain(m_b["Best Month"]),
     pct(m_b["Best Month"] - m_a["Best Month"])),
    ("Worst Month",
     pct_plain(m_a["Worst Month"]),
     pct_plain(m_b["Worst Month"]),
     pct(m_b["Worst Month"] - m_a["Worst Month"])),
]

for name, va, vb, delta in rows:
    print(f"{name:<26} {va:>10}  {vb:>10}  {delta:>10}")

print()

# ── Annual Returns ───────────────────────────────────────────────────────────
print("ANNUAL RETURNS (IS):")
ann_header = f"{'Year':<6} {'A Equal':>10}  {'B Inv-Vol':>10}  {'Delta':>10}"
print(ann_header)
print("-" * len(ann_header))

all_years = sorted(set(list(ann_a.index) + list(ann_b.index)))
for yr in all_years:
    yr_str = str(yr)
    if int(yr_str) < 2005 or int(yr_str) > 2017:
        continue
    va = ann_a.get(yr, float("nan"))
    vb = ann_b.get(yr, float("nan"))
    delta = vb - va if not (np.isnan(va) or np.isnan(vb)) else float("nan")
    print(
        f"{yr_str:<6} {pct_plain(va):>10}  {pct_plain(vb):>10}  {pct(delta):>10}"
    )

print()

# ── Stress Periods ───────────────────────────────────────────────────────────
print("STRESS PERIODS (IS):")
stress_header = f"{'Period':<20} {'A Equal':>10}  {'B Inv-Vol':>10}  {'Delta':>10}"
print(stress_header)
print("-" * len(stress_header))

for label, (s, e) in STRESS.items():
    va = stress_return(r_a, s, e)
    vb = stress_return(r_b, s, e)
    delta = vb - va if not (np.isnan(va) or np.isnan(vb)) else float("nan")
    print(
        f"{label:<20} {pct_plain(va):>10}  {pct_plain(vb):>10}  {pct(delta):>10}"
    )

print()
