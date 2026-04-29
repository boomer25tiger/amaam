"""
Top-7 vs Top-6 main sleeve comparison for AMAAM.

Runs two full backtests:
  A: main_sleeve_top_n = 6  (baseline, 37.5% of 16)
  B: main_sleeve_top_n = 7  (43.75% of 16, closest to FAA ~40% rule)

Reports full period summary, IS/OOS split, annual returns, stress periods,
drawdown stats, and 7th-asset analysis.
"""

import sys
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, "/Users/GualyCr/Desktop/AMAAM/amaam/.claude/worktrees/reverent-mayer")

import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from config.default_config import ModelConfig
from src.backtest.engine import run_backtest, BacktestResult
from src.backtest.metrics import compute_drawdown_series
from src.data.loader import load_validated_data

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ann_ret(rets: pd.Series, periods_per_year: float = 12.0) -> float:
    n = len(rets)
    if n == 0:
        return float("nan")
    total = float((1.0 + rets).prod() - 1.0)
    return float((1.0 + total) ** (periods_per_year / n) - 1.0)


def _ann_vol(rets: pd.Series, periods_per_year: float = 12.0) -> float:
    if len(rets) < 2:
        return float("nan")
    return float(rets.std() * math.sqrt(periods_per_year))


def _sharpe(rets: pd.Series, rf: float = 0.02, periods_per_year: float = 12.0) -> float:
    if len(rets) < 2:
        return float("nan")
    period_rf = (1.0 + rf) ** (1.0 / periods_per_year) - 1.0
    excess = rets - period_rf
    vol = rets.std()
    if vol == 0:
        return float("nan")
    return float(excess.mean() * periods_per_year / (vol * math.sqrt(periods_per_year)))


def _sortino(rets: pd.Series, rf: float = 0.02, periods_per_year: float = 12.0) -> float:
    if len(rets) < 2:
        return float("nan")
    period_rf = (1.0 + rf) ** (1.0 / periods_per_year) - 1.0
    excess = rets - period_rf
    downside = excess[excess < 0]
    if len(downside) == 0:
        return float("nan")
    dd_dev = float(math.sqrt((downside**2).mean()) * math.sqrt(periods_per_year))
    if dd_dev == 0:
        return float("nan")
    ann_r = _ann_ret(rets, periods_per_year)
    return float(ann_r - rf) / dd_dev


def _max_dd(rets: pd.Series) -> float:
    if len(rets) == 0:
        return float("nan")
    dd = compute_drawdown_series(rets)
    return float(dd.min())


def _calmar(rets: pd.Series, periods_per_year: float = 12.0) -> float:
    mdd = _max_dd(rets)
    if mdd >= 0:
        return float("nan")
    return _ann_ret(rets, periods_per_year) / abs(mdd)


def _pct_pos(rets: pd.Series) -> float:
    if len(rets) == 0:
        return float("nan")
    return float((rets > 0).mean())


def _slice(rets: pd.Series, start: str, end: str) -> pd.Series:
    return rets.loc[start:end]


def _stress_return(rets: pd.Series, start: str, end: str) -> float:
    """Total return (not annualised) for a stress period."""
    s = _slice(rets, start, end)
    if len(s) == 0:
        return float("nan")
    return float((1.0 + s).prod() - 1.0)


def _time_in_dd(rets: pd.Series) -> float:
    """Fraction of months spent below a prior high-water mark."""
    if len(rets) == 0:
        return float("nan")
    dd = compute_drawdown_series(rets)
    return float((dd < 0).mean())


def _top5_drawdowns(rets: pd.Series) -> List[Tuple[str, str, float, int]]:
    """Return top-5 drawdown episodes (start, trough, dd, duration_months)."""
    if len(rets) == 0:
        return []
    equity = (1.0 + rets).cumprod()
    peak = equity.cummax()
    dd = (equity - peak) / peak

    episodes: List[Tuple[str, str, float, int]] = []
    in_dd = False
    ep_start = None
    ep_peak_val = None

    for idx, val in dd.items():
        if val < 0:
            if not in_dd:
                in_dd = True
                ep_start = idx
                ep_min = val
            else:
                ep_min = min(ep_min, val)
        else:
            if in_dd:
                # Episode ended
                ep_slice = dd.loc[ep_start:idx]
                dur = len(ep_slice)
                episodes.append((str(ep_start.date()), str(idx.date()), ep_min, dur))
                in_dd = False

    if in_dd:
        ep_slice = dd.loc[ep_start:]
        dur = len(ep_slice)
        episodes.append((str(ep_start.date()), "ongoing", ep_min, dur))

    episodes.sort(key=lambda x: x[2])
    return episodes[:5]


def _annual_rets(rets: pd.Series) -> pd.Series:
    """Calendar-year returns as a Series indexed by integer year."""
    if len(rets) == 0:
        return pd.Series(dtype=float)
    annual = rets.groupby(rets.index.year).apply(lambda r: float((1.0 + r).prod() - 1.0))
    return annual


def _fmt_pct(v: float, decimals: int = 2) -> str:
    if math.isnan(v):
        return "   N/A  "
    sign = "+" if v >= 0 else ""
    return f"{sign}{v*100:.{decimals}f}%"


def _fmt_float(v: float, decimals: int = 3) -> str:
    if math.isnan(v):
        return "  N/A  "
    sign = "+" if v >= 0 else ""
    return f"{sign}{v:.{decimals}f}"


# ---------------------------------------------------------------------------
# 7th-asset analysis
# ---------------------------------------------------------------------------

def analyze_seventh_asset(
    result_a: BacktestResult,
    result_b: BacktestResult,
    data_dict: Dict[str, pd.DataFrame],
) -> None:
    """Identify, tally, and assess the asset that B adds vs A."""
    alloc_a = result_a.allocations
    alloc_b = result_b.allocations

    if alloc_a.empty or alloc_b.empty:
        print("  (allocation data unavailable)")
        return

    # Shared signal dates
    common_dates = alloc_a.index.intersection(alloc_b.index)

    seventh_counts: Dict[str, int] = {}
    seventh_returns: List[float] = []
    top6_avg_returns: List[float] = []

    # Build close-price lookup for monthly returns
    close_dict: Dict[str, pd.Series] = {
        t: data_dict[t]["Close"] for t in data_dict
    }

    for sig_date in common_dates:
        row_a = alloc_a.loc[sig_date]
        row_b = alloc_b.loc[sig_date]

        # Assets held in B but not A (non-zero weight in B, zero/absent in A)
        held_a = set(row_a[row_a > 1e-8].index.tolist())
        held_b = set(row_b[row_b > 1e-8].index.tolist())
        added = held_b - held_a

        # Also find top-6 of A
        top6_assets = list(held_a)

        if not added:
            continue

        # Find the execution date pair for this signal date
        # Use the monthly return series to get the return for this period
        # signal_date → next signal_date is the holding period
        signal_dates_b = sorted(alloc_b.index.tolist())
        sig_pos = signal_dates_b.index(sig_date) if sig_date in signal_dates_b else -1
        if sig_pos < 0 or sig_pos + 1 >= len(signal_dates_b):
            continue

        next_sig = signal_dates_b[sig_pos + 1]

        # Find exec dates (first trading day after signal)
        closes_b_ref = next(iter(close_dict.values()))
        all_dates = closes_b_ref.index.tolist()
        date_to_idx = {d: i for i, d in enumerate(all_dates)}
        idx0 = date_to_idx.get(sig_date)
        if idx0 is None or idx0 + 1 >= len(all_dates):
            continue
        exec0 = all_dates[idx0 + 1]

        idx1 = date_to_idx.get(next_sig)
        if idx1 is None or idx1 + 1 >= len(all_dates):
            continue
        exec1 = all_dates[idx1 + 1]

        for ticker in added:
            seventh_counts[ticker] = seventh_counts.get(ticker, 0) + 1
            if ticker in close_dict:
                c0 = close_dict[ticker].get(exec0)
                c1 = close_dict[ticker].get(exec1)
                if c0 and c1 and c0 > 0 and not (np.isnan(c0) or np.isnan(c1)):
                    ret = c1 / c0 - 1.0
                    seventh_returns.append(ret)

        # Average return of top-6 assets that same period
        period_top6_rets = []
        for t in top6_assets:
            if t in close_dict:
                c0 = close_dict[t].get(exec0)
                c1 = close_dict[t].get(exec1)
                if c0 and c1 and c0 > 0 and not (np.isnan(c0) or np.isnan(c1)):
                    period_top6_rets.append(c1 / c0 - 1.0)
        if period_top6_rets:
            top6_avg_returns.append(float(np.mean(period_top6_rets)))

    if not seventh_counts:
        print("  No months where B adds a distinct 7th asset vs A.")
        return

    sorted_7th = sorted(seventh_counts.items(), key=lambda x: -x[1])
    top_str = ", ".join(f"{t} ({n} months)" for t, n in sorted_7th[:5])
    avg_7th_ret = float(np.mean(seventh_returns)) if seventh_returns else float("nan")
    avg_top6_ret = float(np.mean(top6_avg_returns)) if top6_avg_returns else float("nan")

    print(f"  Most common 7th assets: {top_str}")
    print(f"  Months where B adds a distinct asset:  {sum(seventh_counts.values())}")
    print(f"  Avg return of 7th asset when selected: {_fmt_pct(avg_7th_ret, 2)}/mo")
    print(f"  Avg return of top-6 assets (same months): {_fmt_pct(avg_top6_ret, 2)}/mo")
    contribution = "POSITIVE" if avg_7th_ret >= 0 else "NEGATIVE"
    rel = "above" if avg_7th_ret >= avg_top6_ret else "below"
    print(f"  7th asset contributed {contribution} on avg; {rel} the top-6 avg")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    data_dir = Path(
        "/Users/GualyCr/Desktop/AMAAM/amaam/.claude/worktrees/reverent-mayer/data/processed"
    )
    print("Loading data…")
    data_dict = load_validated_data(data_dir)

    # Config A: top-6 (baseline)
    cfg_a = ModelConfig()
    # Config B: top-7
    cfg_b = ModelConfig()
    cfg_b.main_sleeve_top_n = 7

    print("Running backtest A (top-6)…")
    result_a = run_backtest(data_dict, cfg_a)

    print("Running backtest B (top-7)…")
    result_b = run_backtest(data_dict, cfg_b)

    rets_a = result_a.monthly_returns
    rets_b = result_b.monthly_returns

    PPY = 12.0  # periods per year

    # ── Full period ──────────────────────────────────────────────────────────
    metrics = [
        ("Ann. Return",       _ann_ret,  True,  False),
        ("Ann. Volatility",   _ann_vol,  True,  False),
        ("Sharpe",            _sharpe,   False, False),
        ("Sortino",           _sortino,  False, False),
        ("Max Drawdown",      _max_dd,   True,  False),
        ("Calmar",            _calmar,   False, False),
        ("% Months Positive", _pct_pos,  True,  False),
        ("Best Month",        lambda r: float(r.max()), True, False),
        ("Worst Month",       lambda r: float(r.min()), True, False),
    ]

    print()
    print("=" * 65)
    print("=== MAIN SLEEVE: TOP-7 vs TOP-6 ===")
    print("=" * 65)
    print()
    print("FULL PERIOD:")
    print(f"{'Metric':<25} {'A (Top-6)':>12} {'B (Top-7)':>12} {'Delta':>10}")
    print("-" * 62)

    val_a: Dict[str, float] = {}
    val_b: Dict[str, float] = {}

    for label, fn, is_pct, _ in metrics:
        va = fn(rets_a) if fn != _ann_vol and fn != _ann_ret and fn != _sharpe and fn != _sortino and fn != _calmar else fn(rets_a, PPY) if fn in (_ann_ret, _ann_vol, _calmar) else fn(rets_a)
        vb = fn(rets_b) if fn != _ann_vol and fn != _ann_ret and fn != _sharpe and fn != _sortino and fn != _calmar else fn(rets_b, PPY) if fn in (_ann_ret, _ann_vol, _calmar) else fn(rets_b)

        # Simpler dispatch
        if fn is _ann_ret:
            va, vb = _ann_ret(rets_a, PPY), _ann_ret(rets_b, PPY)
        elif fn is _ann_vol:
            va, vb = _ann_vol(rets_a, PPY), _ann_vol(rets_b, PPY)
        elif fn is _sharpe:
            va, vb = _sharpe(rets_a), _sharpe(rets_b)
        elif fn is _sortino:
            va, vb = _sortino(rets_a), _sortino(rets_b)
        elif fn is _max_dd:
            va, vb = _max_dd(rets_a), _max_dd(rets_b)
        elif fn is _calmar:
            va, vb = _calmar(rets_a, PPY), _calmar(rets_b, PPY)
        elif fn is _pct_pos:
            va, vb = _pct_pos(rets_a), _pct_pos(rets_b)
        else:
            va, vb = fn(rets_a), fn(rets_b)

        val_a[label] = va
        val_b[label] = vb
        delta = vb - va

        if is_pct:
            sa = _fmt_pct(va)
            sb = _fmt_pct(vb)
            sd = _fmt_pct(delta)
        else:
            sa = _fmt_float(va)
            sb = _fmt_float(vb)
            sd = _fmt_float(delta)

        print(f"{label:<25} {sa:>12} {sb:>12} {sd:>10}")

    # ── IS / OOS ─────────────────────────────────────────────────────────────
    IS_START = "2004-01-01"
    IS_END   = "2017-12-31"
    OOS_START = "2018-01-01"
    OOS_END   = cfg.backtest_end

    rets_a_is  = _slice(rets_a, IS_START, IS_END)
    rets_b_is  = _slice(rets_b, IS_START, IS_END)
    rets_a_oos = _slice(rets_a, OOS_START, OOS_END)
    rets_b_oos = _slice(rets_b, OOS_START, OOS_END)

    sr_a_is  = _sharpe(rets_a_is)
    sr_b_is  = _sharpe(rets_b_is)
    sr_a_oos = _sharpe(rets_a_oos)
    sr_b_oos = _sharpe(rets_b_oos)

    # IS→OOS degradation: (IS_SR - OOS_SR) / IS_SR * 100
    deg_a = (sr_a_is - sr_a_oos) / abs(sr_a_is) * 100 if sr_a_is != 0 else float("nan")
    deg_b = (sr_b_is - sr_b_oos) / abs(sr_b_is) * 100 if sr_b_is != 0 else float("nan")

    print()
    print("IS (2004–2017):")
    print(f"{'':30} {'A (Top-6)':>10} {'B (Top-7)':>10} {'ΔSharpe':>8}")
    print(f"  {'Sharpe':<28} {_fmt_float(sr_a_is):>10} {_fmt_float(sr_b_is):>10} {_fmt_float(sr_b_is - sr_a_is):>8}")
    print(f"  {'Ann. Return':<28} {_fmt_pct(_ann_ret(rets_a_is, PPY)):>10} {_fmt_pct(_ann_ret(rets_b_is, PPY)):>10}")
    print(f"  {'MDD':<28} {_fmt_pct(_max_dd(rets_a_is)):>10} {_fmt_pct(_max_dd(rets_b_is)):>10}")

    print()
    print("OOS (2018–2026):")
    print(f"{'':30} {'A (Top-6)':>10} {'B (Top-7)':>10} {'ΔSharpe':>8}")
    print(f"  {'Sharpe':<28} {_fmt_float(sr_a_oos):>10} {_fmt_float(sr_b_oos):>10} {_fmt_float(sr_b_oos - sr_a_oos):>8}")
    print(f"  {'Ann. Return':<28} {_fmt_pct(_ann_ret(rets_a_oos, PPY)):>10} {_fmt_pct(_ann_ret(rets_b_oos, PPY)):>10}")
    print(f"  {'MDD':<28} {_fmt_pct(_max_dd(rets_a_oos)):>10} {_fmt_pct(_max_dd(rets_b_oos)):>10}")

    print()
    dstr_a = f"{deg_a:+.1f}%" if not math.isnan(deg_a) else "N/A"
    dstr_b = f"{deg_b:+.1f}%" if not math.isnan(deg_b) else "N/A"
    print(f"IS→OOS degradation:   A: {dstr_a}    B: {dstr_b}")

    # ── Annual returns ────────────────────────────────────────────────────────
    ann_a = _annual_rets(rets_a)
    ann_b = _annual_rets(rets_b)

    all_years = sorted(set(ann_a.index) | set(ann_b.index))

    print()
    print("ANNUAL RETURNS:")
    print(f"{'Year':<6} {'A(Top-6)':>10} {'B(Top-7)':>10} {'Delta':>8}  Note")
    print("-" * 52)
    for yr in all_years:
        if yr < 2005:
            continue
        va = ann_a.get(yr, float("nan"))
        vb = ann_b.get(yr, float("nan"))
        delta = vb - va
        note = ""
        if not math.isnan(delta):
            if delta > 0.02:
                note = "<-- B beats A (>2pp)"
            elif delta < -0.02:
                note = "<-- B lags A (>2pp)"
        print(f"{yr:<6} {_fmt_pct(va):>10} {_fmt_pct(vb):>10} {_fmt_pct(delta):>8}  {note}")

    # ── Stress periods ────────────────────────────────────────────────────────
    stress_periods = [
        ("GFC",              "2007-10-01", "2009-03-31"),
        ("Euro Crisis",      "2010-04-01", "2011-09-30"),
        ("Taper Tantrum",    "2013-05-01", "2013-06-30"),
        ("China Selloff",    "2015-08-01", "2016-02-29"),
        ("COVID",            "2020-02-01", "2020-03-31"),
        ("2022 Rate Shock",  "2022-01-01", "2022-12-31"),
        ("2025 Tariff Shock","2025-02-01", cfg.backtest_end),
    ]

    print()
    print("STRESS PERIODS:")
    print(f"{'Period':<22} {'A(Top-6)':>10} {'B(Top-7)':>10} {'Delta':>8}")
    print("-" * 55)
    for name, s, e in stress_periods:
        va = _stress_return(rets_a, s, e)
        vb = _stress_return(rets_b, s, e)
        delta = vb - va if not (math.isnan(va) or math.isnan(vb)) else float("nan")
        print(f"{name:<22} {_fmt_pct(va):>10} {_fmt_pct(vb):>10} {_fmt_pct(delta):>8}")

    # ── Drawdown comparison ───────────────────────────────────────────────────
    tdd_a = _time_in_dd(rets_a)
    tdd_b = _time_in_dd(rets_b)

    print()
    print("DRAWDOWNS:")
    print(f"  Time in drawdown:   A: {tdd_a*100:.1f}%    B: {tdd_b*100:.1f}%")

    print()
    print("  Top 5 drawdowns — B (Top-7):")
    top5 = _top5_drawdowns(rets_b)
    print(f"  {'Rank':<5} {'Start':<13} {'End':<13} {'DD':>8} {'Dur(mo)':>8}")
    print("  " + "-" * 50)
    for i, (start, end, dd, dur) in enumerate(top5, 1):
        print(f"  {i:<5} {start:<13} {end:<13} {_fmt_pct(dd):>8} {dur:>8}")

    # ── 7th asset analysis ────────────────────────────────────────────────────
    print()
    print("7TH ASSET ANALYSIS:")
    analyze_seventh_asset(result_a, result_b, data_dict)

    print()
    print("=" * 65)
    print("Done.")


if __name__ == "__main__":
    main()
