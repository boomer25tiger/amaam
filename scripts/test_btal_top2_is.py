"""
test_btal_top2_is.py
--------------------
Test BTAL as a 7th candidate in the hedging sleeve, keeping top-2 selection.
IS window only: 2012-06-01 → 2018-01-01.

Configs
-------
A: GLD, TLT, IEF, SH, UUP, SHY          — top-2  (baseline)
B: GLD, TLT, IEF, SH, UUP, SHY, BTAL   — top-2  (+BTAL)
"""

import sys
import warnings
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

REPO = Path("/Users/GualyCr/Desktop/AMAAM/amaam/.claude/worktrees/reverent-mayer")
sys.path.insert(0, str(REPO))

from config.default_config import ModelConfig
import config.etf_universe as universe_mod
import src.backtest.engine as engine_mod
from src.backtest.engine import run_backtest
from src.data.loader import load_validated_data

DATA_DIR = REPO / "data" / "processed"

# ---------------------------------------------------------------------------
# IS window constants
# ---------------------------------------------------------------------------

IS_START = "2012-06-01"
IS_END   = "2018-01-01"

BASE_SLEEVE   = ["GLD", "TLT", "IEF", "SH", "UUP", "SHY"]
BTAL_SLEEVE   = ["GLD", "TLT", "IEF", "SH", "UUP", "SHY", "BTAL"]

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

data_dict = load_validated_data(DATA_DIR)

# BTAL — handle potential MultiIndex columns from yfinance format
btal = pd.read_csv(DATA_DIR / "BTAL.csv", index_col=0, parse_dates=True)
if isinstance(btal.columns, pd.MultiIndex):
    btal.columns = btal.columns.get_level_values(0)
data_dict["BTAL"] = btal

print(f"BTAL loaded: first={btal.index.min().date()}, last={btal.index.max().date()}, rows={len(btal)}")

# ---------------------------------------------------------------------------
# Helper: performance metrics from monthly returns
# ---------------------------------------------------------------------------

def compute_metrics(monthly_rets: pd.Series) -> dict:
    """Compute ann. return, ann. vol, Sharpe, and MDD from a monthly returns Series.

    Parameters
    ----------
    monthly_rets : pd.Series
        Monthly net returns (not cumulative).

    Returns
    -------
    dict
        Keys: ann_ret, ann_vol, sharpe, mdd.
    """
    r = monthly_rets.dropna()
    if len(r) == 0:
        return {"ann_ret": np.nan, "ann_vol": np.nan, "sharpe": np.nan, "mdd": np.nan}

    ann_ret = (1 + r.mean()) ** 12 - 1
    ann_vol = r.std() * sqrt(12)
    sharpe  = ann_ret / ann_vol if ann_vol > 0 else np.nan

    equity = (1 + r).cumprod()
    mdd    = (equity / equity.cummax() - 1).min()

    return {"ann_ret": ann_ret, "ann_vol": ann_vol, "sharpe": sharpe, "mdd": mdd}


def period_cumret(monthly_rets: pd.Series, start: str, end: str) -> float:
    """Cumulative return over a sub-period.

    Parameters
    ----------
    monthly_rets : pd.Series
        Full monthly returns series.
    start : str
        ISO start date (inclusive).
    end : str
        ISO end date (inclusive).

    Returns
    -------
    float
        Cumulative return, or NaN if no data.
    """
    r = monthly_rets.loc[start:end].dropna()
    if len(r) == 0:
        return np.nan
    return (1 + r).prod() - 1


# ---------------------------------------------------------------------------
# Helper: run one backtest config
# ---------------------------------------------------------------------------

def run_config(sleeve_tickers: list, top_n: int, label: str) -> tuple:
    """Run a backtest with the given hedging sleeve and return (monthly_returns, allocations).

    Parameters
    ----------
    sleeve_tickers : list
        Tickers for the hedging sleeve.
    top_n : int
        Number of top hedging assets to select each period.
    label : str
        Human-readable label for progress output.

    Returns
    -------
    tuple[pd.Series, pd.DataFrame]
        (monthly_returns, allocations)
    """
    orig_engine_hedge   = engine_mod.HEDGING_SLEEVE_TICKERS
    orig_universe_hedge = universe_mod.HEDGING_SLEEVE_TICKERS[:]

    try:
        engine_mod.HEDGING_SLEEVE_TICKERS    = sleeve_tickers
        universe_mod.HEDGING_SLEEVE_TICKERS[:] = sleeve_tickers

        # Extend ALL_TICKERS if BTAL is new
        orig_all = universe_mod.ALL_TICKERS[:]
        new_all = sorted(
            set(universe_mod.ALL_TICKERS)
            | set(sleeve_tickers)
            | {"SPY", "AGG", "VV", "TIP", "IGOV"}
        )
        universe_mod.ALL_TICKERS[:] = new_all

        cfg = ModelConfig()
        cfg.hedging_sleeve_top_n = top_n
        cfg.backtest_start = "2004-01-01"
        cfg.backtest_end   = "2026-04-23"

        print(f"\nRunning config {label}: sleeve={sleeve_tickers}, top_n={top_n}…")
        result = run_backtest(data_dict, cfg)
        print(f"  Done. {len(result.monthly_returns)} monthly obs.")
        return result.monthly_returns, result.allocations

    finally:
        engine_mod.HEDGING_SLEEVE_TICKERS      = orig_engine_hedge
        universe_mod.HEDGING_SLEEVE_TICKERS[:] = orig_universe_hedge
        universe_mod.ALL_TICKERS[:]            = orig_all


# ---------------------------------------------------------------------------
# Run configs
# ---------------------------------------------------------------------------

rets_A, allocs_A = run_config(BASE_SLEEVE,  top_n=2, label="A (6/2 baseline)")
rets_B, allocs_B = run_config(BTAL_SLEEVE,  top_n=2, label="B (7/2 +BTAL)")

# ---------------------------------------------------------------------------
# IS slice
# ---------------------------------------------------------------------------

rets_A_is = rets_A.loc[IS_START:IS_END]
rets_B_is = rets_B.loc[IS_START:IS_END]

m_A = compute_metrics(rets_A_is)
m_B = compute_metrics(rets_B_is)

delta_sharpe = m_B["sharpe"] - m_A["sharpe"]
is_months    = len(rets_A_is.dropna())

# ---------------------------------------------------------------------------
# BTAL selection frequency analysis (IS window)
# ---------------------------------------------------------------------------

# allocs_B index is the signal date; filter to IS window
allocs_B_is = allocs_B.loc[IS_START:IS_END]

btal_selected_mask = allocs_B_is.get("BTAL", pd.Series(0.0, index=allocs_B_is.index)).fillna(0) > 0

btal_selected_months = allocs_B_is.index[btal_selected_mask]
n_selected = btal_selected_mask.sum()
n_total    = len(btal_selected_mask)

# For each BTAL-selected month, find the partner hedge
HEDGE_TICKERS_B = BTAL_SLEEVE

partner_list = []
for dt in btal_selected_months:
    row = allocs_B_is.loc[dt]
    # hedges selected are those hedge-sleeve tickers with non-zero weight
    selected_hedges = [
        t for t in HEDGE_TICKERS_B
        if t in row.index and pd.notna(row[t]) and row[t] > 0
    ]
    partners = [t for t in selected_hedges if t != "BTAL"]
    partner_list.append({"date": dt, "partners": partners})

# BTAL returns in IS months
btal_prices = data_dict["BTAL"]["Close"] if "Close" in data_dict["BTAL"].columns else data_dict["BTAL"].iloc[:, 0]
btal_monthly = btal_prices.resample("ME").last().pct_change().dropna()
# align index to IS window
btal_is = btal_monthly.loc[IS_START:IS_END]

# Map selection months to BTAL monthly returns.
# allocs signal dates are month-end; btal_monthly index is also month-end.
# We use the month period to match.
selected_periods = set(pd.Period(d, "M") for d in btal_selected_months)
btal_selected_rets = []
btal_not_selected_rets = []

for dt, ret in btal_is.items():
    p = pd.Period(dt, "M")
    if p in selected_periods:
        btal_selected_rets.append(ret)
    else:
        btal_not_selected_rets.append(ret)

btal_avg_selected     = np.mean(btal_selected_rets)     if btal_selected_rets     else np.nan
btal_avg_not_selected = np.mean(btal_not_selected_rets) if btal_not_selected_rets else np.nan

# Build partner hedge summary string
if partner_list:
    from collections import Counter
    all_partners = [p for entry in partner_list for p in entry["partners"]]
    partner_counts = Counter(all_partners)
    partner_summary = ", ".join(f"{t}({c}x)" for t, c in partner_counts.most_common())
else:
    partner_summary = "—"

# Full list of selected months with partners
selected_detail_lines = []
for entry in partner_list:
    dt_str   = entry["date"].strftime("%Y-%m")
    partners = ", ".join(entry["partners"]) if entry["partners"] else "—"
    selected_detail_lines.append(f"    {dt_str}  partner: {partners}")

# ---------------------------------------------------------------------------
# Stress periods (IS window only, so only periods within 2012–2018 matter)
# ---------------------------------------------------------------------------

stress_periods = [
    ("Eurozone 2012",     "2012-01-01", "2012-06-30"),
    ("Taper 2013",        "2013-05-01", "2013-06-30"),
    ("HY Stress 2015-16", "2015-08-01", "2016-02-29"),
    ("Volmageddon 2018",  "2018-01-01", "2018-02-28"),
]

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

print()
print("=" * 75)
print("=== BTAL AS 7TH CANDIDATE, TOP-2 SELECTION — IS ONLY ===")
print(f"IS window: {IS_START[:7]} → {IS_END[:7]}")
print("=" * 75)
print()

header = (
    f"{'Config':<20}  {'Sleeve':>6}  {'TopN':>4}  "
    f"{'Ann.Ret':>8}  {'Ann.Vol':>8}  {'Sharpe':>7}  {'MDD':>8}  {'ΔSharpe':>8}"
)
print(header)
print("-" * len(header))

configs_display = [
    ("A  Baseline",   6, 2, m_A, "—"),
    ("B  +BTAL",      7, 2, m_B, f"{delta_sharpe:+.3f}"),
]
for label, n_sleeve, n_top, m, ds in configs_display:
    print(
        f"{label:<20}  {n_sleeve:>6}  {n_top:>4}  "
        f"{m['ann_ret']:>7.2%}  {m['ann_vol']:>7.2%}  {m['sharpe']:>7.3f}  "
        f"{m['mdd']:>7.2%}  {ds:>8}"
    )

print()
print("BTAL selection frequency (IS):")
pct = 100 * n_selected / n_total if n_total > 0 else 0.0
print(f"  Selected: {n_selected} months of {n_total} total ({pct:.1f}%)")
print(f"  When selected, partner hedge(s): {partner_summary}")
print(f"  BTAL avg return when selected:      {btal_avg_selected:>6.2%}/mo")
print(f"  BTAL avg return when not selected:  {btal_avg_not_selected:>6.2%}/mo")

if selected_detail_lines:
    print()
    print("  Month-by-month (BTAL selected):")
    for line in selected_detail_lines:
        print(line)

print()
print("Stress Periods (IS):")
stress_header = (
    f"{'Period':<20}  {'A(6/2)':>9}  {'B+BTAL(7/2)':>12}  {'Delta':>8}"
)
print(stress_header)
print("-" * len(stress_header))

for sp_label, sp_start, sp_end in stress_periods:
    ra = period_cumret(rets_A_is, sp_start, sp_end)
    rb = period_cumret(rets_B_is, sp_start, sp_end)

    def fmt(v: float) -> str:
        return f"{v:.2%}" if not np.isnan(v) else "n/a"

    delta = rb - ra if (not np.isnan(ra) and not np.isnan(rb)) else np.nan
    print(
        f"{sp_label:<20}  {fmt(ra):>9}  {fmt(rb):>12}  {fmt(delta):>8}"
    )

print()
print("=" * 75)
