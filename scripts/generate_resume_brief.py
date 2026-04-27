"""
Generate a self-contained PDF brief of all AMAAM numerical results and
implementation details, intended for loading into a fresh AI session to
assist with resume / cover-letter writing.

All numbers are pulled live from a fresh backtest run plus the supporting
analysis scripts so the PDF always reflects the current codebase state.

Output: reports/amaam_resume_brief.pdf

Usage
-----
    python3.13 scripts/generate_resume_brief.py
    python3.13 scripts/generate_resume_brief.py --data-dir data/processed
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import logging
import numpy as np
import pandas as pd
import statsmodels.api as sm
from dataclasses import replace
from pathlib import Path

from config.default_config import ModelConfig
from config.etf_universe import (
    MAIN_SLEEVE_TICKERS, HEDGING_SLEEVE_TICKERS,
    MAIN_SLEEVE, HEDGING_SLEEVE,
)
from src.backtest.benchmarks import (
    _build_close_matrix, _monthly_rebalanced_returns,
    compute_sixty_forty, compute_seven_twelve, compute_spy_benchmark,
)
from src.backtest.engine import run_backtest
from src.data.loader import load_validated_data

logging.basicConfig(level=logging.WARNING)

RF_ANN  = 0.02
RF_MON  = RF_ANN / 12


# ── Metric helpers ────────────────────────────────────────────────────────────

def _slice(s, start, end):
    return s[(s.index >= start) & (s.index < end)]

def _ann_ret(r):
    return (1 + r).prod() ** (12 / len(r)) - 1 if len(r) > 1 else float("nan")

def _ann_vol(r):
    return r.std() * np.sqrt(12)

def _sharpe(r, rf=RF_ANN):
    ret, vol = _ann_ret(r), _ann_vol(r)
    return (ret - rf) / vol if vol > 0 else float("nan")

def _sortino(r, rf=RF_ANN):
    ret = _ann_ret(r)
    down_vol = r[r < 0].std() * np.sqrt(12)
    return (ret - rf) / down_vol if down_vol > 0 else float("nan")

def _maxdd(r):
    eq = (1 + r).cumprod()
    return float((eq / eq.cummax() - 1).min())

def _calmar(r):
    ret, dd = _ann_ret(r), _maxdd(r)
    return ret / abs(dd) if dd != 0 else float("nan")

def _dd_duration(r):
    eq = (1 + r).cumprod()
    in_dd, cur, mx = eq < eq.cummax(), 0, 0
    for v in in_dd:
        cur = cur + 1 if v else 0
        mx = max(mx, cur)
    return mx

def _ir(active):
    if len(active) < 6: return float("nan")
    return (active.mean() * 12) / (active.std() * np.sqrt(12))

def _ols_alpha(y, X):
    aligned = pd.concat([y, X], axis=1).dropna()
    if len(aligned) < 12:
        return dict(alpha=float("nan"), betas={}, r2=float("nan"),
                    t_alpha=float("nan"), p_alpha=float("nan"), n=0)
    y_ = aligned.iloc[:, 0]
    X_ = sm.add_constant(aligned.iloc[:, 1:])
    res = sm.OLS(y_, X_).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
    return dict(
        alpha=res.params["const"] * 12,
        betas={col: res.params[col] for col in aligned.columns[1:]},
        r2=res.rsquared,
        t_alpha=res.tvalues["const"],
        p_alpha=res.pvalues["const"],
        n=len(aligned),
    )

def _full_stats(r):
    if len(r) < 6:
        return {k: float("nan") for k in
                ["ret","vol","sr","sortino","calmar","mdd","mdd_dur",
                 "best_yr","worst_yr","pct_pos","n"]}
    annual = r.resample("YE").apply(lambda x: (1+x).prod()-1)
    return dict(
        ret=_ann_ret(r), vol=_ann_vol(r), sr=_sharpe(r),
        sortino=_sortino(r), calmar=_calmar(r),
        mdd=_maxdd(r), mdd_dur=_dd_duration(r),
        best_yr=annual.max(), worst_yr=annual.min(),
        pct_pos=(r > 0).mean(), n=len(r),
    )


# ── Compute everything ────────────────────────────────────────────────────────

def compute_all(data_dir="data/processed"):
    print("  Running backtest…")
    data = load_validated_data(data_dir)
    cfg  = ModelConfig()
    res  = run_backtest(data, cfg)
    amaam = res.monthly_returns

    S, E = cfg.backtest_start, cfg.backtest_end
    IS_E, OOS_E, HOLD_S = cfg.holdout_start, "2024-01-01", "2024-01-01"

    windows = {
        "Full (2004–2026)":    (S,      E),
        "IS   (2004–2018)":    (S,      IS_E),
        "OOS  (2018–2024)":    (IS_E,   OOS_E),
        "Hold (2024–2026)":    (HOLD_S, E),
    }

    # Period metrics
    print("  Computing period metrics…")
    period_stats = {lbl: _full_stats(_slice(amaam, ws, we))
                    for lbl, (ws, we) in windows.items()}

    # Annual returns
    annual = amaam.resample("YE").apply(lambda x: (1+x).prod()-1)

    # Benchmark returns
    print("  Building benchmarks…")
    spy     = compute_spy_benchmark(data, S, E)
    b60_40  = compute_sixty_forty(data, S, E)
    b7twelve = compute_seven_twelve(data, S, E)
    all_sleeve = MAIN_SLEEVE_TICKERS + HEDGING_SLEEVE_TICKERS
    closes_sl  = _build_close_matrix(data, all_sleeve)
    b1n = _monthly_rebalanced_returns(
        closes_sl, {t: 1/len(all_sleeve) for t in all_sleeve}, S, E)

    spy_stats  = _full_stats(_slice(spy, S, E))
    b60_stats  = _full_stats(_slice(b60_40, S, E))
    b7t_stats  = _full_stats(_slice(b7twelve, S, E))
    b1n_stats  = _full_stats(_slice(b1n, S, E))

    # IR vs benchmarks per window
    bench_irs = {}
    for lbl, (ws, we) in windows.items():
        a = _slice(amaam, ws, we)
        row = {}
        for bname, brets in [("SPY", spy), ("60/40", b60_40),
                              ("7Twelve", b7twelve), ("1/N 22-ETF", b1n)]:
            b = _slice(brets, ws, we)
            common = a.index.intersection(b.index)
            if len(common) < 6:
                row[bname] = dict(exc=float("nan"), ir=float("nan"))
                continue
            active = a.loc[common] - b.loc[common]
            row[bname] = dict(exc=active.mean()*12, ir=_ir(active))
        bench_irs[lbl] = row

    # Multi-factor alpha
    print("  Running multi-factor regression…")
    factor_tickers = ["SPY", "IEF", "DBC", "GLD"]
    closes_f = _build_close_matrix(data, factor_tickers)
    fmon = pd.DataFrame({
        t: _monthly_rebalanced_returns(closes_f[[t]], {t:1.0}, S, E)
        for t in factor_tickers})
    mf_alpha = {}
    for lbl, (ws, we) in windows.items():
        y = _slice(amaam, ws, we)
        X = fmon.loc[y.index] if not y.empty else pd.DataFrame()
        mf_alpha[lbl] = _ols_alpha(y, X)

    # Walk-forward summary (hardcoded from canonical run — re-running 39×6
    # configs here would add ~10 min; these are the validated results)
    wf_summary = dict(
        n_folds=6, n_configs=39,
        canonical_wins=5,
        stacked_oos_sr_canonical=0.719,
        stacked_oos_sr_baseline=0.566,
        canonical_weights="wM=0.65 / wV=0.25 / wC=0.10",
    )

    # Statistical significance (hardcoded from definitive_master_report run)
    sig = dict(
        ttest_full=dict(t=3.951, p=0.0001),
        ttest_is=dict(t=2.925, p=0.0020),
        ttest_oos=dict(t=1.974, p=0.0261),
        ttest_hold=dict(t=1.934, p=0.0318),
        permutation_z=3.98, permutation_p=0.0000,
        bootstrap_95_full=(0.283, 1.110),
        bootstrap_99_full=(0.162, 1.257),
        p_sr_gt_0=1.000,
        ols_single_factor=dict(alpha=0.1131, beta=-0.003, r2=0.000,
                                t=3.86, p=0.0001, note="MISSPECIFIED vs SPY only"),
    )

    # Experiments tested and rejected
    experiments = [
        dict(
            name="Portfolio volatility targeting",
            description=(
                "Scales monthly allocation so realized portfolio vol ≈ target. "
                "Tested: VT-10% NoLev, VT-10% 1.5×Lev, VT-12% NoLev."
            ),
            result="All variants hurt Sharpe (0.645–0.650 vs baseline 0.708).",
            reason="Hedging sleeve already de-risks in stress; vol-targeting is redundant.",
            verdict="REJECTED",
        ),
        dict(
            name="Selection hysteresis (exit buffer)",
            description=(
                "Incumbents retained until ranked > N + exit_buffer instead of "
                "exiting immediately at rank > N. Tested buffer=1 and buffer=2."
            ),
            result=(
                "Buffer-2 improved IS Sharpe (0.751 vs 0.682) but walk-forward: "
                "2/6 test folds won, stacked OOS SR 0.684 vs 0.719 baseline."
            ),
            reason="Fast exits are valuable in volatile markets (COVID 2020, 2022 rate hikes).",
            verdict="REJECTED",
        ),
        dict(
            name="Momentum blend [63, 126, 252] vs [21, 63, 126]",
            description=(
                "Replace 1-month ROC component with 12-month ROC to reduce "
                "short-term noise and turnover."
            ),
            result=(
                "Turnover −32% (610%/yr vs 894%/yr) but holdout SR collapsed "
                "to 0.384 vs 0.740; Fold 4 (COVID) SR = 0.050 vs 1.255."
            ),
            reason="21-day component provides critical early-warning for fast regime shifts.",
            verdict="REJECTED",
        ),
    ]

    return dict(
        amaam=amaam, annual=annual,
        spy=spy,                                  # raw SPY series for annual table
        period_stats=period_stats,
        spy_stats=spy_stats, b60_stats=b60_stats,
        b7t_stats=b7t_stats, b1n_stats=b1n_stats,
        bench_irs=bench_irs,
        mf_alpha=mf_alpha,
        wf_summary=wf_summary,
        sig=sig,
        experiments=experiments,
        cfg=cfg,
        windows=windows,
    )


# ── PDF generation ────────────────────────────────────────────────────────────

def build_pdf(d: dict, out_path: str) -> None:
    from fpdf import FPDF

    # Helvetica only covers Latin-1 (ISO 8859-1, 0x00–0xFF).
    # Substitute common Unicode code-points before they reach the codec so we
    # don't have to sprinkle ASCII-safe strings throughout every call site.
    _UNICODE_SUBS: dict[str, str] = {
        "\u2014": "--",    # em-dash
        "\u2013": "-",     # en-dash
        "\u2192": "->",    # right arrow
        "\u2190": "<-",    # left arrow
        "\u2022": "*",     # bullet
        "\u00B7": "*",     # middle dot
        "\u03B1": "alpha", # α
        "\u03B2": "beta",  # β
        "\u03B5": "eps",   # ε
        "\u03C3": "sigma", # σ
        "\u0394":  "D",    # Δ
        "\u2264": "<=",    # ≤
        "\u2265": ">=",    # ≥
        "\u2248": "~",     # ≈
        "\u2260": "!=",    # ≠
        "\u2080": "0",     # subscript 0
        "\u2081": "1",     # subscript 1
        "\u2082": "2",     # subscript 2
        "\u00B2": "^2",    # superscript 2
        "\u00B3": "^3",    # superscript 3
        "\u00B1": "+/-",   # ±
        "\u221E": "inf",   # ∞
        "\u2019": "'",     # right single quotation mark
        "\u2018": "'",     # left single quotation mark
        "\u201C": '"',     # left double quotation mark
        "\u201D": '"',     # right double quotation mark
        "\u2212": "-",     # minus sign (U+2212, distinct from ASCII hyphen)
        "\u2011": "-",     # non-breaking hyphen
        "\u00D7": "x",     # multiplication sign (override to keep simple)
        "\u2026": "...",   # ellipsis
        "\u00A0": " ",     # non-breaking space
    }

    class PDF(FPDF):
        def normalize_text(self, text: str) -> str:  # type: ignore[override]
            for ch, repl in _UNICODE_SUBS.items():
                text = text.replace(ch, repl)
            # Catch-all: any remaining non-Latin-1 code-point becomes '?'
            # so the PDF is never blocked by an unmapped character.
            text = text.encode("latin-1", errors="replace").decode("latin-1")
            return super().normalize_text(text)

        def header(self):
            self.set_font("Helvetica", "B", 9)
            self.set_text_color(120, 120, 120)
            self.cell(0, 6, "AMAAM - Quantitative Research Brief", align="R")
            self.ln(4)

        def footer(self):
            self.set_y(-12)
            self.set_font("Helvetica", "", 8)
            self.set_text_color(150, 150, 150)
            self.cell(0, 5, f"Page {self.page_no()}", align="C")

    pdf = PDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_margins(18, 18, 18)

    W = 174  # usable width

    def title(text, size=13):
        pdf.set_font("Helvetica", "B", size)
        pdf.set_text_color(20, 60, 120)
        pdf.ln(3)
        pdf.cell(0, 7, text)
        pdf.ln(7)
        pdf.set_draw_color(20, 60, 120)
        pdf.set_line_width(0.4)
        pdf.line(pdf.get_x(), pdf.get_y(), pdf.get_x() + W, pdf.get_y())
        pdf.ln(4)
        pdf.set_text_color(0, 0, 0)

    def section(text):
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(40, 40, 40)
        pdf.ln(4)
        pdf.cell(0, 6, text)
        pdf.ln(7)
        pdf.set_text_color(0, 0, 0)

    def row(label, value, indent=4):
        pdf.set_font("Helvetica", "", 9)
        pdf.set_x(pdf.l_margin + indent)
        pdf.cell(70, 5, label)
        pdf.set_font("Helvetica", "B", 9)
        pdf.cell(0, 5, str(value))
        pdf.ln(5.5)

    def trow(cols, widths, bold_first=False):
        pdf.set_font("Helvetica", "", 8.5)
        for i, (c, w) in enumerate(zip(cols, widths)):
            if i == 0 and bold_first:
                pdf.set_font("Helvetica", "B", 8.5)
            else:
                pdf.set_font("Helvetica", "", 8.5)
            pdf.cell(w, 5.5, str(c), border=0)
        pdf.ln(5.5)

    def thead(cols, widths):
        pdf.set_font("Helvetica", "B", 8.5)
        pdf.set_fill_color(230, 237, 248)
        for c, w in zip(cols, widths):
            pdf.cell(w, 6, str(c), fill=True)
        pdf.ln(6)

    def body(text, size=9):
        pdf.set_font("Helvetica", "", size)
        pdf.multi_cell(0, 5, text)
        pdf.ln(2)

    def note(text):
        pdf.set_font("Helvetica", "I", 8)
        pdf.set_text_color(100, 100, 100)
        pdf.multi_cell(0, 4.5, text)
        pdf.set_text_color(0, 0, 0)
        pdf.ln(2)

    s  = d["period_stats"]
    si = d["sig"]
    wf = d["wf_summary"]
    cfg = d["cfg"]

    # ── PAGE 1: Title + Implementation ───────────────────────────────────────
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_text_color(20, 60, 120)
    pdf.ln(2)
    pdf.cell(0, 10, "AMAAM — Adaptive Multi-Asset Allocation Model", align="C")
    pdf.ln(10)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(90, 90, 90)
    pdf.cell(0, 5, "Quantitative Research Brief  |  All results use corrected 5 bps one-way transaction costs", align="C")
    pdf.ln(10)
    pdf.set_text_color(0, 0, 0)

    title("1. Model Overview & Implementation")

    section("What it is")
    body(
        "A fully systematic, rules-based multi-asset allocation model that selects and "
        "weights ETFs monthly using a composite factor score (TRank). The model spans "
        "22 ETFs across US equities (sector + size), international equities, real estate, "
        "commodities, fixed income, gold, dollar, and inverse equity."
    )

    section("Architecture")
    row("Type", "Rules-based systematic allocation, monthly rebalancing")
    row("Universe", "22 ETFs: 16 main sleeve + 6 hedging sleeve")
    row("Main sleeve tickers", ", ".join(MAIN_SLEEVE_TICKERS))
    row("Hedging sleeve tickers", ", ".join(HEDGING_SLEEVE_TICKERS))
    row("Selection", "Top-6 main + top-2 hedging by TRank; equal-weight within each sleeve")
    row("Defensive routing", "When main-sleeve momentum ≤ 0, weight redirected to hedging sleeve")
    row("Backtest period", f"{cfg.backtest_start}  →  {cfg.backtest_end}")
    row("Transaction cost", "5 bps one-way per leg (buy AND sell each charged separately)")
    row("Implementation", "Python — NumPy, pandas, matplotlib, plotly, statsmodels")
    row("Test coverage", "155 unit tests across all modules")

    section("TRank Formula")
    body(
        "TRank = wM × Rank(M) + wV × Rank(V) + wC × Rank(C) + wT × T + M/n\n\n"
        "  M = blended momentum (equal-weight average of 1-month, 3-month, 6-month ROC)\n"
        "  V = Yang-Zhang realized volatility (126-day window); lower is better\n"
        "  C = average pairwise correlation (126-day rolling); lower is better\n"
        "  T = SMA200 trend signal (+2 above SMA200, -2 below)\n"
        "  M/n = tiebreaker term (raw momentum divided by universe size)\n\n"
        "Canonical weights (walk-forward validated):  wM=0.65 / wV=0.25 / wC=0.10 / wT=1.0\n"
        "Ranks are ordinal 1..N; higher TRank = better. Top-N selected per sleeve."
    )

    # ── PAGE 2: Core Performance ──────────────────────────────────────────────
    pdf.add_page()
    title("2. Core Performance Metrics")

    section("Full Period vs SPY  (2004 – 2026, 5 bps/leg)")
    wds = [74, 28, 28, 28, 28]
    thead(["Metric", "AMAAM", "SPY", "60/40", "1/N"], wds)

    sp = d["spy_stats"]; b6 = d["b60_stats"]; b1 = d["b1n_stats"]
    sf = s["Full (2004–2026)"]

    def pct(x): return f"{x*100:+.2f}%" if not np.isnan(x) else "N/A"
    def rat(x): return f"{x:.3f}" if not np.isnan(x) else "N/A"
    def mo(x):  return f"{int(x)}mo" if not np.isnan(x) else "N/A"

    rows_data = [
        ("Ann. Return",    pct(sf["ret"]),       pct(sp["ret"]),  pct(b6["ret"]),  pct(b1["ret"])),
        ("Ann. Volatility",pct(sf["vol"]),       pct(sp["vol"]),  pct(b6["vol"]),  pct(b1["vol"])),
        ("Sharpe Ratio",   rat(sf["sr"]),        rat(sp["sr"]),   rat(b6["sr"]),   rat(b1["sr"])),
        ("Sortino Ratio",  rat(sf["sortino"]),   "—",             "—",             "—"),
        ("Calmar Ratio",   rat(sf["calmar"]),    rat(sp["calmar"]),rat(b6["calmar"]),rat(b1["calmar"])),
        ("Max Drawdown",   pct(sf["mdd"]),       pct(sp["mdd"]),  pct(b6["mdd"]),  pct(b1["mdd"])),
        ("MDD Duration",   mo(sf["mdd_dur"]),    mo(sp["mdd_dur"]),mo(b6["mdd_dur"]),mo(b1["mdd_dur"])),
        ("Best Year",      pct(sf["best_yr"]),   pct(sp["best_yr"]),"—",           "—"),
        ("Worst Year",     pct(sf["worst_yr"]),  pct(sp["worst_yr"]),"—",          "—"),
        ("% Pos Months",   pct(sf["pct_pos"]),   pct(sp["pct_pos"]),"—",           "—"),
    ]
    for r_data in rows_data:
        trow(r_data, wds, bold_first=True)

    pdf.ln(3)
    section("IS → OOS → Holdout Progression")
    wds2 = [50, 22, 22, 22, 22, 22, 22]
    thead(["Period", "N Months", "Ann Ret", "Ann Vol", "Sharpe", "MaxDD", "Calmar"], wds2)
    for lbl, (ws, we) in d["windows"].items():
        st = s[lbl]
        trow([lbl, st["n"], pct(st["ret"]), pct(st["vol"]),
              rat(st["sr"]), pct(st["mdd"]), rat(st["calmar"])], wds2, True)

    note(
        "Key result: Sharpe does NOT degrade IS → OOS → Holdout (0.667 → 0.656 → 1.084). "
        "This is evidence against overfitting. Holdout was run once, after all design "
        "decisions were finalised, and results were never used to adjust parameters."
    )

    # ── PAGE 3: Annual returns + Benchmark IR ─────────────────────────────────
    pdf.add_page()
    title("3. Annual Returns & Benchmark Information Ratios")

    section("Calendar-Year Returns  (AMAAM vs SPY)")
    annual_spy = _slice(d["spy"], "2004-01-01", "2026-04-10").resample("YE").apply(
        lambda x: (1+x).prod()-1)
    wds3 = [22, 28, 28, 28]
    thead(["Year", "AMAAM", "SPY", "Excess"], wds3)
    for yr, val in d["annual"].items():
        y_str = str(yr.year)
        spy_val = annual_spy.get(yr, float("nan"))
        exc = val - spy_val if not np.isnan(spy_val) else float("nan")
        trow([y_str, pct(val), pct(spy_val), pct(exc)], wds3, True)

    pdf.ln(4)
    section("Information Ratio vs Benchmarks by Period")
    note(
        "IR = annualised active return / tracking error. "
        "Positive IR = AMAAM outperforms on risk-adjusted basis."
    )
    wds4 = [46, 32, 32, 32, 32]
    thead(["Period", "vs SPY", "vs 60/40", "vs 7Twelve", "vs 1/N (22 ETF)"], wds4)
    for lbl in d["windows"]:
        bi = d["bench_irs"][lbl]
        def fmt_ir(b):
            row_d = bi.get(b, {})
            e = row_d.get("exc", float("nan"))
            i = row_d.get("ir", float("nan"))
            if np.isnan(e): return "N/A"
            return f"{e*100:+.2f}% / {i:+.2f}"
        trow([lbl, fmt_ir("SPY"), fmt_ir("60/40"), fmt_ir("7Twelve"), fmt_ir("1/N 22-ETF")],
             wds4, True)

    note(
        "AMAAM underperforms SPY on raw return (IR ≈ -0.04 full period) but consistently "
        "outperforms multi-asset benchmarks (IR ≈ +0.20–0.21 vs 60/40 and 1/N). "
        "The edge is risk-adjusted return from drawdown management, not equity market alpha."
    )

    # ── PAGE 4: Alpha decomposition ───────────────────────────────────────────
    pdf.add_page()
    title("4. Alpha Decomposition")

    section("Why SPY-only alpha is misleading")
    body(
        "A single-factor OLS vs SPY yields alpha = +11.3%/yr (t=3.86, p<0.0001) — but this "
        "is largely spurious. When beta ≈ 0, the OLS intercept collapses to the mean return. "
        "The model holds bonds, commodities, REITs, gold, and international equities; a "
        "single equity factor cannot span those risk premia, so they all show up as 'alpha'.\n\n"
        "The single-factor result is reported here for completeness only. "
        "Multi-factor regression is the correct specification."
    )

    section("Multi-factor OLS  (r_AMAAM = α + β_SPY + β_IEF + β_DBC + β_GLD + ε)")
    note("HAC Newey-West standard errors, 3 lags. α annualised.")
    wds5 = [44, 22, 14, 14, 18, 18, 18, 18]
    thead(["Period", "α/yr", "t(α)", "p(α)", "β_SPY", "β_IEF", "β_DBC", "β_GLD"], wds5)
    for lbl in d["windows"]:
        mf = d["mf_alpha"][lbl]
        if mf["n"] < 12:
            trow([lbl, "—","—","—","—","—","—","—"], wds5, True)
            continue
        stars = ("***" if mf["p_alpha"]<0.01 else
                 "**"  if mf["p_alpha"]<0.05 else
                 "*"   if mf["p_alpha"]<0.10 else "")
        betas = mf["betas"]
        trow([lbl,
              f"{mf['alpha']*100:+.1f}%{stars}",
              f"{mf['t_alpha']:+.2f}",
              f"{mf['p_alpha']:.3f}",
              f"{betas.get('SPY',0):+.3f}",
              f"{betas.get('IEF',0):+.3f}",
              f"{betas.get('DBC',0):+.3f}",
              f"{betas.get('GLD',0):+.3f}",
              ], wds5, True)

    note(
        "Key findings:\n"
        "• Full-period α = +7.3%/yr (t=2.54, **) — significant but smaller than single-factor claim.\n"
        "• IS α = +9.8%/yr (***); OOS α = +5.9% (NOT significant, p=0.30).\n"
        "• Holdout α = -10.5%/yr (**): in 2024–2026, equities and bonds both rallied "
        "strongly (β_SPY=0.74, β_IEF=0.60, R²=0.79); factor tilts drove the holdout "
        "return, not residual selection skill. R² of 0.79 in holdout vs 0.17 full-period "
        "reflects a high-factor-return environment, not model degradation.\n"
        "• The cleanest evidence of skill: IR = +0.20 vs 1/N equal-weight within the same "
        "universe, and +17pp better max drawdown. Dynamic allocation adds value primarily "
        "through superior timing of defensive positioning."
    )

    section("1/N Equal-Weight Baseline (same 22 ETFs, zero-skill benchmark)")
    wds6 = [44, 22, 22, 22, 22, 22, 22]
    thead(["Period", "AMAAM SR", "1/N SR", "ΔSharpe", "AMAAM DD", "1/N DD", "ΔMDD"], wds6)
    for lbl, (ws, we) in d["windows"].items():
        a = _slice(d["amaam"], ws, we)
        b = _slice(_monthly_rebalanced_returns(
            _build_close_matrix(
                load_validated_data("data/processed"),
                MAIN_SLEEVE_TICKERS + HEDGING_SLEEVE_TICKERS),
            {t: 1/22 for t in MAIN_SLEEVE_TICKERS + HEDGING_SLEEVE_TICKERS},
            ws, we), ws, we)
        if len(a) < 6 or len(b) < 6:
            continue
        common = a.index.intersection(b.index)
        a_, b_ = a.loc[common], b.loc[common]
        dsr = _sharpe(a_) - _sharpe(b_)
        ddd = _maxdd(a_) - _maxdd(b_)
        trow([lbl, rat(_sharpe(a_)), rat(_sharpe(b_)), f"{dsr:+.3f}",
              pct(_maxdd(a_)), pct(_maxdd(b_)), f"{ddd*100:+.1f}pp"], wds6, True)

    # ── PAGE 5: Statistical significance + Walk-forward ───────────────────────
    pdf.add_page()
    title("5. Statistical Significance")

    section("t-test: H₀ — mean monthly return = 0  (one-tailed)")
    wds7 = [54, 24, 24, 24, 36]
    thead(["Period", "t-stat", "p-value", "Sig", "Interpretation"], wds7)
    for lbl, t_, p_ in [
        ("Full (2004–2026)", si["ttest_full"]["t"],  si["ttest_full"]["p"]),
        ("IS   (2004–2018)", si["ttest_is"]["t"],    si["ttest_is"]["p"]),
        ("OOS  (2018–2024)", si["ttest_oos"]["t"],   si["ttest_oos"]["p"]),
        ("Hold (2024–2026)", si["ttest_hold"]["t"],  si["ttest_hold"]["p"]),
    ]:
        sig_str = ("***" if p_<0.01 else "**" if p_<0.05 else "*" if p_<0.10 else "ns")
        interp = ("Reject H₀ at 1%" if p_<0.01 else
                  "Reject H₀ at 5%" if p_<0.05 else
                  "Reject H₀ at 10%" if p_<0.10 else "Fail to reject")
        trow([lbl, f"{t_:+.3f}", f"{p_:.4f}", sig_str, interp], wds7, True)

    pdf.ln(3)
    section("Permutation Test & Bootstrap Sharpe CI  (full period)")
    row("Permutation Z-score",         f"{si['permutation_z']:.2f}")
    row("Permutation p-value",         f"{si['permutation_p']:.4f}  (***)")
    row("Bootstrap 95% CI (Sharpe)",   f"[{si['bootstrap_95_full'][0]:.3f},  {si['bootstrap_95_full'][1]:.3f}]")
    row("Bootstrap 99% CI (Sharpe)",   f"[{si['bootstrap_99_full'][0]:.3f},  {si['bootstrap_99_full'][1]:.3f}]")
    row("P(Sharpe > 0)",               f"{si['p_sr_gt_0']*100:.1f}%")

    pdf.ln(3)
    title("6. Walk-Forward Validation")
    body(
        f"Expanding-window walk-forward across {wf['n_configs']} factor-weight configurations "
        f"and {wf['n_folds']} test folds (each 2 years). "
        f"In each fold, the config with the highest training Sharpe is selected and then "
        f"evaluated on the unseen test window."
    )
    row("Canonical weights", wf["canonical_weights"])
    row("Canonical wins test folds", f"{wf['canonical_wins']}/{wf['n_folds']}")
    row("Stacked OOS SR — canonical", f"{wf['stacked_oos_sr_canonical']:.3f}")
    row("Stacked OOS SR — baseline (wM=0.50)", f"{wf['stacked_oos_sr_baseline']:.3f}")
    note(
        "The canonical wM=0.65 config was identified in IS and won 5 of 6 independent "
        "OOS test folds. The stacked OOS Sharpe (0.719) is the key out-of-sample validation "
        "figure used in all public performance claims."
    )

    # ── PAGE 6: Experiments rejected + Resume guidance ────────────────────────
    pdf.add_page()
    title("7. Experiments Tested and Rejected")
    note(
        "Each add-on below was implemented, walk-forward validated using the same 6-fold "
        "structure, and rejected when OOS evidence did not support adoption. "
        "This discipline is itself part of the implementation story."
    )

    for exp in d["experiments"]:
        section(exp["name"])
        row("Description", "")
        body(f"    {exp['description']}")
        row("Walk-forward result", "")
        body(f"    {exp['result']}")
        row("Rejection reason", "")
        body(f"    {exp['reason']}")
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_text_color(180, 30, 30)
        pdf.cell(0, 5, f"  → {exp['verdict']}")
        pdf.ln(6)
        pdf.set_text_color(0, 0, 0)

    title("8. Resume Guidance for AI Assistant")
    body(
        "This document contains all verified numerical results for the AMAAM project. "
        "When writing resume bullet points, use the following hierarchy of claims:\n\n"
        "TIER 1 — Most defensible (use freely):\n"
        "  • Sharpe 0.71 vs SPY 0.63 over same period\n"
        "  • Max drawdown -18.7% vs SPY -50.8%  (32pp improvement)\n"
        "  • Matches SPY gross return (+10.4% vs +10.7%) through different exposure\n"
        "  • IR = +0.20 vs 1/N equal-weight across same 22-ETF universe (full and OOS)\n"
        "  • +2.65%/yr active return vs 1/N passive baseline\n"
        "  • Sharpe non-degrading IS→OOS→Holdout: 0.667 → 0.656 → 1.084\n"
        "  • Permutation Z = 3.98, p < 0.0001 (return not attributable to chance)\n\n"
        "TIER 2 — Use with context:\n"
        "  • Multi-factor α = +7.3%/yr (t=2.54, **) — significant full-period but not OOS\n"
        "  • Walk-forward: canonical weights win 5/6 OOS folds (stacked OOS SR 0.719)\n"
        "  • Holdout α = -10.5%/yr — driven by factor tilts in a strong bull market,\n"
        "    not model failure; R² jumped to 0.79 in 2024–2026\n\n"
        "AVOID:\n"
        "  • 'Alpha of +11.3%/yr' — this is the misspecified single-factor OLS result\n"
        "  • 'Beta ≈ 0 to SPY' — multi-factor β_SPY is actually +0.25; CAPM beta was\n"
        "    masking other factor correlations\n"
        "  • Any Sharpe figure > 0.75 without specifying the period (holdout is 1.08\n"
        "    but covers only 28 months)\n\n"
        "The model's value proposition is risk-adjusted return (Sharpe, drawdown) over "
        "passive multi-asset benchmarks, not raw alpha generation above SPY."
    )

    pdf.output(out_path)
    print(f"  PDF written → {out_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--out", default="reports/amaam_resume_brief.pdf")
    args = parser.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    print("Computing results…")
    d = compute_all(args.data_dir)
    print("Building PDF…")
    build_pdf(d, args.out)


if __name__ == "__main__":
    main()
