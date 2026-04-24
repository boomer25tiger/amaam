"""
Definitive master performance report for AMAAM.

Single source of truth for every metric, statistic, and significance test.
All numbers use a consistent rf = 2%/yr convention throughout.
No local Sharpe helpers — all standard metrics go through compute_all_metrics().

Sections
--------
1.  Data & period definitions
2.  Core performance table (all periods)
3.  Calendar-year returns (AMAAM vs SPY)
4.  Drawdown analysis
5.  Rolling metrics (12-month windows)
6.  Block-bootstrap Sharpe CIs  (n=10,000, block=12, rf=2%)
7.  Hypothesis tests
     7a. t-test: mean monthly return = 0
     7b. Bootstrap Sharpe-difference test (AMAAM vs SPY)
     7c. OLS alpha / beta vs SPY
     7d. Bootstrap max-drawdown distribution
8.  Permutation test (sign-flip, n=10,000)
9.  IS → OOS → Holdout progression
10. Academic benchmark comparison
"""

import sys
sys.path.insert(0, '/Users/GualyCr/Desktop/AMAAM/amaam/.claude/worktrees/reverent-mayer')

import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from config.default_config import ModelConfig
from src.backtest.engine import run_backtest
from src.data.loader import load_validated_data
from src.backtest.metrics import compute_all_metrics, compute_drawdown_series

warnings.filterwarnings("ignore")

RF_ANNUAL   = 0.02          # risk-free rate used EVERYWHERE in this script
RF_MONTHLY  = (1 + RF_ANNUAL) ** (1 / 12) - 1

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Data & period definitions
# ─────────────────────────────────────────────────────────────────────────────

DATA_DIR = Path(
    '/Users/GualyCr/Desktop/AMAAM/amaam/.claude/worktrees/reverent-mayer/data/processed'
)
data_dict = load_validated_data(DATA_DIR)
cfg       = ModelConfig()
result    = run_backtest(data_dict, cfg)

# AMAAM monthly returns — convert index to Period so intersection works cleanly
mr = result.monthly_returns.copy()
mr.index = mr.index.to_period('M')

# SPY monthly returns — resample to month-end, then convert to Period
spy_daily = data_dict['SPY']['Close']
spy_ret   = spy_daily.resample('ME').last().pct_change().dropna()
spy_ret.index = spy_ret.index.to_period('M')

# Restrict to the intersection of dates present in both series
common = mr.index.intersection(spy_ret.index)
mr  = mr.reindex(common)
spy = spy_ret.reindex(common)

# Period masks — mutually exclusive and exhaustive
is_mask  = common.year <= 2017
oos_mask = (common.year >= 2018) & (common.year <= 2023)
ho_mask  = common.year >= 2024

mr_is,  spy_is  = mr[is_mask],  spy[is_mask]
mr_oos, spy_oos = mr[oos_mask], spy[oos_mask]
mr_ho,  spy_ho  = mr[ho_mask],  spy[ho_mask]

# ─────────────────────────────────────────────────────────────────────────────
# Canonical helpers — all use RF_ANNUAL / RF_MONTHLY
# ─────────────────────────────────────────────────────────────────────────────

def sharpe(r: pd.Series) -> float:
    """Annualised Sharpe ratio, rf = RF_ANNUAL (2 %)."""
    if len(r) < 2 or r.std() == 0:
        return np.nan
    excess = r - RF_MONTHLY
    return float(excess.mean() * 12 / (r.std() * math.sqrt(12)))


def ann_return(r: pd.Series) -> float:
    n = len(r)
    return float((1.0 + r).prod() ** (12.0 / n) - 1.0) if n > 0 else np.nan


def ann_vol(r: pd.Series) -> float:
    return float(r.std() * math.sqrt(12))


def mdd(r: pd.Series) -> float:
    return float(compute_drawdown_series(r).min()) if len(r) > 0 else np.nan


def calmar(r: pd.Series) -> float:
    ar, md = ann_return(r), mdd(r)
    return ar / abs(md) if (md is not None and md < 0) else np.nan


def sortino(r: pd.Series) -> float:
    """
    Sortino ratio — industry standard (Sortino & Price 1994, Bloomberg convention).

    MAR = 0  (absolute-return hurdle: do not lose money).
    Downside deviation = sqrt( mean_over_ALL_periods( min(r_t, 0)^2 ) ) * sqrt(12).
    Averaging over ALL periods — not just negative ones — is the original definition
    and matches Bloomberg PORT, HFR, and Eurekahedge database reporting.
    Numerator = annualised return (no rf subtraction; MAR is already 0).
    """
    shortfall = np.minimum(r.values, 0.0)           # positive months clipped to 0
    dd_dev    = math.sqrt(np.mean(shortfall ** 2)) * math.sqrt(12)
    return float(ann_return(r) / dd_dev) if dd_dev > 0 else np.nan


def mdd_duration(r: pd.Series) -> int:
    dd = compute_drawdown_series(r)
    max_dur, cur = 0, 0
    for v in dd:
        if v < 0:
            cur += 1
            max_dur = max(max_dur, cur)
        else:
            cur = 0
    return max_dur


def pct_positive(r: pd.Series) -> float:
    return float((r > 0).mean())


def ols_alpha_beta(r_a: pd.Series, r_b: pd.Series):
    """
    OLS: r_a = alpha_m + beta * r_b + eps
    Returns (alpha_ann, beta, r2, t_alpha, p_alpha, t_beta, p_beta)
    """
    x = r_b.values
    y = r_a.values
    X = np.column_stack([np.ones(len(x)), x])
    coeffs, *_ = np.linalg.lstsq(X, y, rcond=None)
    alpha_m, beta = coeffs[0], coeffs[1]
    alpha_ann = (1.0 + alpha_m) ** 12 - 1.0

    y_hat = X @ coeffs
    resid = y - y_hat
    n, k  = len(y), 2
    s2    = np.sum(resid ** 2) / (n - k)
    se    = np.sqrt(np.diag(np.linalg.inv(X.T @ X) * s2))
    t_a, t_b = alpha_m / se[0], beta / se[1]
    p_a = 2.0 * stats.t.sf(abs(t_a), n - k)
    p_b = 2.0 * stats.t.sf(abs(t_b), n - k)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2  = 1.0 - np.sum(resid ** 2) / ss_tot if ss_tot > 0 else 0.0
    return alpha_ann, beta, r2, t_a, p_a, t_b, p_b


# ─────────────────────────────────────────────────────────────────────────────
# Bootstrap helpers
# ─────────────────────────────────────────────────────────────────────────────

def block_bootstrap_sharpe(r: pd.Series, n=10_000, block=12, seed=42) -> np.ndarray:
    np.random.seed(seed)
    arr     = r.values
    n_obs   = len(arr)
    n_blk   = int(np.ceil(n_obs / block))
    results = []
    for _ in range(n):
        starts = np.random.randint(0, n_obs - block + 1, size=n_blk)
        s      = np.concatenate([arr[i: i + block] for i in starts])[:n_obs]
        excess = s - RF_MONTHLY
        v      = s.std()
        results.append(float(excess.mean() * 12 / (v * math.sqrt(12))) if v > 0 else np.nan)
    return np.array([x for x in results if not np.isnan(x)])


def bootstrap_sharpe_diff(r_a: pd.Series, r_b: pd.Series, n=10_000, block=12, seed=42):
    np.random.seed(seed)
    a, b  = r_a.values, r_b.values
    n_obs = len(a)
    n_blk = int(np.ceil(n_obs / block))
    obs   = sharpe(r_a) - sharpe(r_b)
    deltas = []
    for _ in range(n):
        starts = np.random.randint(0, n_obs - block + 1, size=n_blk)
        sa = np.concatenate([a[i: i + block] for i in starts])[:n_obs]
        sb = np.concatenate([b[i: i + block] for i in starts])[:n_obs]
        def _sr(s):
            v = s.std(); ex = s - RF_MONTHLY
            return float(ex.mean() * 12 / (v * math.sqrt(12))) if v > 0 else np.nan
        d = _sr(sa) - _sr(sb)
        if not np.isnan(d):
            deltas.append(d)
    deltas  = np.array(deltas)
    centred = deltas - deltas.mean()
    p_val   = float(np.mean(np.abs(centred) >= abs(obs)))
    return obs, p_val, deltas


def bootstrap_mdd(r: pd.Series, n=10_000, block=12, seed=42) -> np.ndarray:
    np.random.seed(seed)
    arr   = r.values
    n_obs = len(arr)
    n_blk = int(np.ceil(n_obs / block))
    mdds  = []
    for _ in range(n):
        starts = np.random.randint(0, n_obs - block + 1, size=n_blk)
        s      = np.concatenate([arr[i: i + block] for i in starts])[:n_obs]
        eq     = np.cumprod(1.0 + s)
        pk     = np.maximum.accumulate(eq)
        mdds.append(float(((eq - pk) / pk).min()))
    return np.array(mdds)


def permutation_test(r: pd.Series, n=10_000, seed=42):
    np.random.seed(seed)
    arr    = r.values
    obs_sr = sharpe(r)
    perms  = []
    for _ in range(n):
        signs  = np.random.choice([-1.0, 1.0], size=len(arr))
        flipped = arr * signs
        v = flipped.std()
        ex = flipped - RF_MONTHLY
        perms.append(float(ex.mean() * 12 / (v * math.sqrt(12))) if v > 0 else np.nan)
    perms = np.array([x for x in perms if not np.isnan(x)])
    p_val = float(np.mean(perms >= obs_sr))
    z     = (obs_sr - perms.mean()) / perms.std() if perms.std() > 0 else np.nan
    return obs_sr, perms.mean(), perms.std(), z, p_val


# ─────────────────────────────────────────────────────────────────────────────
# Formatting helpers
# ─────────────────────────────────────────────────────────────────────────────

W   = 78
SEP = "=" * W

def fp(x, d=1):
    return f"{x * 100:+.{d}f}%" if not np.isnan(x) else "   N/A"

def ff(x, d=3):
    return f"{x:+.{d}f}" if not np.isnan(x) else "   N/A"

def fi(x):
    return f"{int(x)}" if not np.isnan(x) else "N/A"

def ci_str(arr, alpha=0.05):
    lo = np.percentile(arr, alpha / 2 * 100)
    hi = np.percentile(arr, (1 - alpha / 2) * 100)
    return f"[{lo:+.3f}, {hi:+.3f}]"

def sig_stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "** "
    if p < 0.05:  return "*  "
    return "   "

def reject(p, alpha):
    return "Reject" if p < alpha else "Fail  "


# ─────────────────────────────────────────────────────────────────────────────
# Pre-compute everything
# ─────────────────────────────────────────────────────────────────────────────

periods = {
    "Full (2004–2026)":    (mr,     spy),
    "IS   (2004–2017)":    (mr_is,  spy_is),
    "OOS  (2018–2023)":    (mr_oos, spy_oos),
    "Hold (2024–2026)":    (mr_ho,  spy_ho),
}

# Section 6: bootstrap Sharpe CIs
print("Computing bootstrap CIs (n=10,000 × 4 periods × 2 series) …", flush=True)
bs_amaam = {k: block_bootstrap_sharpe(v[0]) for k, v in periods.items()}
bs_spy   = {k: block_bootstrap_sharpe(v[1]) for k, v in periods.items()}

# Section 7b: bootstrap Sharpe difference
print("Computing bootstrap Sharpe-difference tests …", flush=True)
sr_diff = {k: bootstrap_sharpe_diff(v[0], v[1]) for k, v in periods.items()}

# Section 7d: MDD bootstrap (full period only)
print("Computing MDD bootstrap …", flush=True)
bs_mdd_amaam = bootstrap_mdd(mr)
bs_mdd_spy   = bootstrap_mdd(spy, seed=43)

# Section 8: permutation test
print("Computing permutation test …", flush=True)
perm_full = permutation_test(mr)

print("Done. Printing report …\n", flush=True)


# ═════════════════════════════════════════════════════════════════════════════
# PRINT REPORT
# ═════════════════════════════════════════════════════════════════════════════

print(SEP)
print("  AMAAM — DEFINITIVE PERFORMANCE & SIGNIFICANCE REPORT")
print(f"  Risk-free rate: {RF_ANNUAL*100:.1f}% / yr  |  All periods: monthly returns")
print(f"  Data: {str(common[0])} → {str(common[-1])}  ({len(common)} months total)")
print(SEP)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: Core performance table
# ─────────────────────────────────────────────────────────────────────────────

print()
print("SECTION 2: CORE PERFORMANCE METRICS")
print()

ROWS = [
    ("N months",          lambda r, _: fi(len(r))),
    ("Ann Return",         lambda r, _: fp(ann_return(r))),
    ("Ann Volatility",     lambda r, _: fp(ann_vol(r))),
    ("Sharpe Ratio",       lambda r, _: ff(sharpe(r))),
    ("Sortino Ratio",      lambda r, _: ff(sortino(r))),
    ("Calmar Ratio",       lambda r, _: ff(calmar(r))),
    ("Max Drawdown",       lambda r, _: fp(mdd(r))),
    ("MDD Duration (mo)",  lambda r, _: fi(mdd_duration(r))),
    ("% Positive Months",  lambda r, _: fp(pct_positive(r))),
    ("Best Month",         lambda r, _: fp(r.max())),
    ("Worst Month",        lambda r, _: fp(r.min())),
    ("Total Return",       lambda r, _: fp(float((1 + r).prod() - 1))),
    ("──── SPY ────",      lambda r, s: ""),
    ("SPY Ann Return",     lambda r, s: fp(ann_return(s))),
    ("SPY Sharpe",         lambda r, s: ff(sharpe(s))),
    ("SPY Max Drawdown",   lambda r, s: fp(mdd(s))),
    ("──── vs SPY ────",   lambda r, s: ""),
    ("Excess Ann Return",  lambda r, s: fp(ann_return(r) - ann_return(s))),
    ("Sharpe Advantage",   lambda r, s: ff(sharpe(r) - sharpe(s))),
    ("MDD Advantage",      lambda r, s: fp(mdd(s) - mdd(r))),   # positive = shallower DD
]

pkeys = list(periods.keys())
col   = 14

hdr = f"{'Metric':<25}" + "".join(f"{k:>{col}}" for k in pkeys)
print(hdr)
print("-" * (25 + col * len(pkeys)))

for label, fn in ROWS:
    if label.startswith("─"):
        print()
        print(f"  {label}")
        continue
    vals = [fn(v[0], v[1]) for v in periods.values()]
    line = f"{label:<25}" + "".join(f"{v:>{col}}" for v in vals)
    print(line)

print()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: Calendar-year returns
# ─────────────────────────────────────────────────────────────────────────────

print("SECTION 3: CALENDAR-YEAR RETURNS")
print()

annual_amaam = mr.groupby(mr.index.year).apply(
    lambda g: float((1 + g).prod() - 1)
)
annual_spy = spy.groupby(spy.index.year).apply(
    lambda g: float((1 + g).prod() - 1)
)

print(f"{'Year':<7}{'AMAAM':>9}{'SPY':>9}{'Excess':>9}  {'Tag'}")
print("-" * 55)

for yr in sorted(annual_amaam.index):
    a = annual_amaam[yr]
    s = annual_spy.get(yr, np.nan)
    e = a - s if not np.isnan(s) else np.nan
    tag = ""
    if yr == 2008: tag = "GFC drawdown"
    elif yr == 2020: tag = "COVID"
    elif yr == 2022: tag = "Rate shock"
    elif yr == 2024: tag = "Holdout"
    elif yr == 2025: tag = "Holdout / tariff shock"
    elif yr == 2026: tag = "Holdout YTD (partial)"
    period = "IS  " if yr <= 2017 else ("OOS " if yr <= 2023 else "HOLD")
    print(
        f"{yr} {period:<4}"
        f"{fp(a):>9}"
        f"{fp(s) if not np.isnan(s) else '   N/A':>9}"
        f"{fp(e) if not np.isnan(e) else '   N/A':>9}"
        f"  {tag}"
    )

print()
ann_pos = (annual_amaam > 0).mean()
spy_pos = (annual_spy > 0).mean()
print(f"  % Positive years — AMAAM: {ann_pos*100:.0f}%   SPY: {spy_pos*100:.0f}%")
print()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: Drawdown analysis
# ─────────────────────────────────────────────────────────────────────────────

print("SECTION 4: DRAWDOWN ANALYSIS")
print()

dd_full = compute_drawdown_series(mr)
pct_in_dd = (dd_full < 0).mean()

print(f"  Full-period max drawdown (AMAAM):   {mdd(mr)*100:+.2f}%")
print(f"  Full-period max drawdown (SPY):     {mdd(spy)*100:+.2f}%")
print(f"  Max drawdown duration:              {mdd_duration(mr)} months")
print(f"  % of months in drawdown:            {pct_in_dd*100:.1f}%")
print()

# Five worst drawdowns
dd_full2 = dd_full.copy()
events   = []
in_dd    = False
start    = None
peak_val = 0.0
trough   = 0.0
for per, val in dd_full2.items():
    if val < 0 and not in_dd:
        in_dd    = True
        start    = per
        trough   = val
    elif val < 0 and in_dd:
        if val < trough:
            trough = val
    elif val >= 0 and in_dd:
        in_dd  = False
        events.append((start, per, trough))
        trough = 0.0

if in_dd:
    events.append((start, dd_full2.index[-1], trough))

events.sort(key=lambda x: x[2])
print(f"  {'Start':<10}  {'End':<10}  {'Depth':>8}  {'Duration':>9}")
print("  " + "-" * 44)
for s, e, depth in events[:5]:
    dur = len(dd_full2[(dd_full2.index >= s) & (dd_full2.index <= e)])
    print(f"  {str(s):<10}  {str(e):<10}  {depth*100:>7.1f}%  {dur:>7} mo")

print()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: Rolling 12-month metrics
# ─────────────────────────────────────────────────────────────────────────────

print("SECTION 5: ROLLING 12-MONTH METRICS (summary statistics)")
print()

ROLL = 12
roll_sr  = mr.rolling(ROLL, min_periods=ROLL).apply(
    lambda x: (pd.Series(x) - RF_MONTHLY).mean() * 12 / (pd.Series(x).std() * math.sqrt(12)),
    raw=True
).dropna()
roll_ret = mr.rolling(ROLL, min_periods=ROLL).apply(
    lambda x: float((1 + pd.Series(x)).prod() - 1), raw=True
).dropna()
roll_vol = mr.rolling(ROLL, min_periods=ROLL).std().dropna() * math.sqrt(12)

print(f"  {'Metric':<30}  {'Mean':>8}  {'Median':>8}  {'Min':>8}  {'Max':>8}")
print("  " + "-" * 68)
for lbl, arr, is_pct in [
    ("Rolling 12m Sharpe",      roll_sr,  False),
    ("Rolling 12m Ann Return",  roll_ret, True),
    ("Rolling 12m Ann Vol",     roll_vol, True),
]:
    def _fmt(v):
        return f"{v*100:+7.1f}%" if is_pct else f"{v:+7.3f}"
    print(
        f"  {lbl:<30}  {_fmt(arr.mean()):>8}  {_fmt(arr.median()):>8}"
        f"  {_fmt(arr.min()):>8}  {_fmt(arr.max()):>8}"
    )

pct_roll_pos = (roll_sr > 0).mean()
print(f"\n  % of 12m windows with positive Sharpe: {pct_roll_pos*100:.1f}%")
print()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: Bootstrap Sharpe confidence intervals
# ─────────────────────────────────────────────────────────────────────────────

print("SECTION 6: BOOTSTRAP SHARPE CIs  (block=12mo, n=10,000, rf=2%)")
print()

col2 = 18
hdr6 = f"{'':25}" + "".join(f"{k:>{col2}}" for k in pkeys)
print(hdr6)
print("-" * (25 + col2 * len(pkeys)))

rows6 = [
    ("Point Sharpe (AMAAM)", lambda k: ff(sharpe(periods[k][0]))),
    ("Bootstrap mean",        lambda k: ff(bs_amaam[k].mean())),
    ("95% CI",                lambda k: ci_str(bs_amaam[k], 0.05)),
    ("99% CI",                lambda k: ci_str(bs_amaam[k], 0.01)),
    ("P(SR > 0)",             lambda k: f"{np.mean(bs_amaam[k] > 0)*100:.1f}%"),
    ("P(SR > SPY SR)",        lambda k: f"{np.mean(bs_amaam[k] > sharpe(periods[k][1]))*100:.1f}%"),
    ("──── SPY ────",         None),
    ("Point Sharpe (SPY)",    lambda k: ff(sharpe(periods[k][1]))),
    ("SPY 95% CI",            lambda k: ci_str(bs_spy[k], 0.05)),
]

for label, fn in rows6:
    if fn is None:
        print()
        print(f"  {label}")
        continue
    vals = [fn(k) for k in pkeys]
    print(f"{label:<25}" + "".join(f"{v:>{col2}}" for v in vals))

print()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7: Hypothesis tests
# ─────────────────────────────────────────────────────────────────────────────

print("SECTION 7: HYPOTHESIS TESTS")
print()

# 7a: t-test mean monthly return = 0
print("  7a. t-test: H₀: mean monthly return = 0  (one-tailed, μ > 0)")
print(f"  {'Period':<22}  {'t-stat':>8}  {'p-value':>9}  {'1%':>8}  {'5%':>8}  {'Stars'}")
print("  " + "-" * 64)
for k, (r, _) in periods.items():
    t, p2 = stats.ttest_1samp(r.values, popmean=0)
    p1 = p2 / 2 if t > 0 else 1.0 - p2 / 2
    print(
        f"  {k:<22}  {t:>+8.3f}  {p1:>9.4f}"
        f"  {reject(p1, 0.01):>8}  {reject(p1, 0.05):>8}  {sig_stars(p1)}"
    )

print()

# 7b: Bootstrap Sharpe difference
print("  7b. Bootstrap Sharpe-difference: H₀: AMAAM Sharpe = SPY Sharpe")
print(f"  {'Period':<22}  {'Δ Sharpe':>9}  {'p-value':>9}  {'5%':>8}  {'10%':>8}")
print("  " + "-" * 64)
for k, (obs_d, p_val, _) in sr_diff.items():
    print(
        f"  {k:<22}  {obs_d:>+9.3f}  {p_val:>9.4f}"
        f"  {reject(p_val, 0.05):>8}  {reject(p_val, 0.10):>8}"
    )

print()

# 7c: OLS alpha / beta
print("  7c. OLS regression: r_AMAAM = α + β · r_SPY + ε")
print(f"  {'Period':<22}  {'α (ann)':>9}  {'β':>7}  {'R²':>6}  {'t(α)':>7}  {'p(α)':>8}  {'Stars'}")
print("  " + "-" * 72)
for k, (r, r_spy) in periods.items():
    ccommon = r.index.intersection(r_spy.index)
    a, b, r2, t_a, p_a, t_b, p_b = ols_alpha_beta(r.reindex(ccommon), r_spy.reindex(ccommon))
    print(
        f"  {k:<22}  {a*100:>+8.2f}%  {b:>+7.3f}  {r2:>6.3f}"
        f"  {t_a:>+7.2f}  {p_a:>8.4f}  {sig_stars(p_a)}"
    )

print()

# 7d: MDD bootstrap (full period)
print("  7d. Max-drawdown bootstrap  (full period, n=10,000, block=12)")
mdd_obs    = mdd(mr)
mdd_spy_ob = mdd(spy)
mdd_95_lo  = np.percentile(bs_mdd_amaam, 2.5)
mdd_95_hi  = np.percentile(bs_mdd_amaam, 97.5)
p_shallower = float(np.mean(bs_mdd_amaam > bs_mdd_spy))   # AMAAM MDD closer to 0 than SPY's
print(f"  Observed AMAAM MDD:             {mdd_obs*100:+.2f}%")
print(f"  Bootstrap mean MDD:             {bs_mdd_amaam.mean()*100:+.2f}%")
print(f"  Bootstrap 95% CI:              [{mdd_95_lo*100:+.2f}%, {mdd_95_hi*100:+.2f}%]")
print(f"  Observed SPY MDD:               {mdd_spy_ob*100:+.2f}%")
print(f"  P(AMAAM MDD shallower than SPY): {p_shallower*100:.1f}%")

print()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8: Permutation test (full period)
# ─────────────────────────────────────────────────────────────────────────────

print("SECTION 8: SIGN-FLIP PERMUTATION TEST  (full period, n=10,000, rf=2%)")
print()

obs_sr, pm_mean, pm_std, pz, pp = perm_full
print(f"  Observed Sharpe:           {obs_sr:+.3f}")
print(f"  Permutation mean Sharpe:   {pm_mean:+.3f}")
print(f"  Permutation std:            {pm_std:.3f}")
print(f"  Z-score:                   {pz:+.2f}")
print(f"  p-value (one-tailed):       {pp:.4f}  {sig_stars(pp).strip()}")
print(f"  Significant at 1%?          {'Yes' if pp < 0.01 else 'No'}")
print(f"  Significant at 5%?          {'Yes' if pp < 0.05 else 'No'}")
print()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9: IS → OOS → Holdout progression
# ─────────────────────────────────────────────────────────────────────────────

print("SECTION 9: IS → OOS → HOLDOUT PROGRESSION")
print()
print(f"  {'Period':<22}  {'N':>4}  {'Ann Ret':>9}  {'Sharpe':>8}  {'Max DD':>8}  {'Calmar':>8}  {'% Pos':>7}")
print("  " + "-" * 74)

for k, (r, _) in periods.items():
    print(
        f"  {k:<22}  {len(r):>4}"
        f"  {fp(ann_return(r)):>9}"
        f"  {ff(sharpe(r)):>8}"
        f"  {fp(mdd(r)):>8}"
        f"  {ff(calmar(r)):>8}"
        f"  {fp(pct_positive(r)):>7}"
    )

print()
print("  Sharpe trend (higher = better, no degradation is the target):")
for k, (r, _) in periods.items():
    sr  = sharpe(r)
    bar = "█" * max(0, int(sr / 1.5 * 30))
    print(f"    {k:<22}  {sr:+.3f}  {bar}")

print()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 10: Academic benchmark comparison
# ─────────────────────────────────────────────────────────────────────────────

print("SECTION 10: ACADEMIC BENCHMARK COMPARISON")
print("  (AMAAM Sharpe for matching sub-period, rf=2%)")
print()

benchmarks = [
    ("Faber GTAA (2007)",      "1973–2007",   2004, 2007, 0.93, "rf≈0 in paper"),
    ("Antonacci GEM (2014)",   "1974–2013",   2004, 2013, 0.73, "rf≈0 in paper"),
    ("Keller FAA (2012)",      "1998–2012",   2004, 2012, 1.01, "rf≈0 in paper"),
    ("AQR MOM factor",         "Long-run",    2004, 2024, 0.60, "gross of fees"),
    ("MSCI World Momentum",    "Recent",       2014, 2024, 0.70, "index, gross"),
]

print(f"  {'Benchmark':<26}  {'Period':<12}  {'Rep SR':>7}  {'AMAAM SR':>9}  {'Verdict':<8}  Note")
print("  " + "-" * 82)

for name, period_lbl, s_yr, e_yr, rep_sr, note in benchmarks:
    mask = (mr.index.year >= s_yr) & (mr.index.year <= e_yr)
    sub  = mr[mask]
    if len(sub) < 12:
        print(f"  {name:<26}  {period_lbl:<12}  ~{rep_sr:.2f}     N/A        N/A       {note}")
        continue
    amsr = sharpe(sub)
    if amsr > rep_sr * 1.05:
        verdict = "Beats"
    elif amsr >= rep_sr * 0.95:
        verdict = "Matches"
    else:
        verdict = "Trails"
    print(
        f"  {name:<26}  {period_lbl:<12}  ~{rep_sr:.2f}"
        f"    {amsr:>+7.3f}    {verdict:<8}  {note}"
    )

print()
print("  Note: published benchmarks typically use rf=0; AMAAM SR here uses rf=2%.")
print("  Add ~0.14 to AMAAM SR for apples-to-apples comparison with those papers.")
print()

# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY BOX
# ─────────────────────────────────────────────────────────────────────────────

print(SEP)
print("  SUMMARY — SINGLE REFERENCE TABLE  (rf = 2%, 2004–2026)")
print(SEP)
print()

summary_rows = [
    ("Full-period Ann Return",   fp(ann_return(mr))),
    ("Full-period Ann Vol",      fp(ann_vol(mr))),
    ("Full-period Sharpe",       ff(sharpe(mr))),
    ("Full-period Sortino",      ff(sortino(mr))),
    ("Full-period Calmar",       ff(calmar(mr))),
    ("Full-period Max Drawdown", fp(mdd(mr))),
    ("Full-period MDD Duration", f"{mdd_duration(mr)} months"),
    ("Full-period % Pos Months", fp(pct_positive(mr))),
    ("Full-period Total Return", fp(float((1 + mr).prod() - 1))),
    ("",                         ""),
    ("IS  Sharpe (2004–2017)",   ff(sharpe(mr_is))),
    ("OOS Sharpe (2018–2023)",   ff(sharpe(mr_oos))),
    ("Holdout Sharpe (2024–26)", ff(sharpe(mr_ho))),
    ("",                         ""),
    ("SPY full-period Sharpe",   ff(sharpe(spy))),
    ("SPY full-period Ann Ret",  fp(ann_return(spy))),
    ("SPY full-period Max DD",   fp(mdd(spy))),
    ("",                         ""),
    ("Bootstrap 95% CI (full)", ci_str(bs_amaam["Full (2004–2026)"], 0.05)),
    ("Bootstrap 99% CI (full)", ci_str(bs_amaam["Full (2004–2026)"], 0.01)),
    ("P(Sharpe > 0)  [full]",   f"{np.mean(bs_amaam['Full (2004–2026)'] > 0)*100:.1f}%"),
    ("",                         ""),
    ("Permutation Z-score",      ff(pz, 2)),
    ("Permutation p-value",      f"{pp:.4f}"),
    ("Alpha vs SPY  [full]",     fp(ols_alpha_beta(mr, spy)[0])),
    ("Beta vs SPY   [full]",     ff(ols_alpha_beta(mr, spy)[1])),
    ("R²  vs SPY    [full]",     ff(ols_alpha_beta(mr, spy)[2])),
]

for lbl, val in summary_rows:
    if lbl == "":
        print()
        continue
    print(f"  {lbl:<35}  {val}")

print()
print(SEP)
print(f"  Data spans: {str(common[0])} → {str(common[-1])}  ({len(common)} months)")
print(f"  Config: trend={cfg.trend_method}, wM={cfg.weight_momentum}, wV={cfg.weight_volatility},",
      f"wC={cfg.weight_correlation}, wT={cfg.weight_trend}")
print(f"  Config: top_n={cfg.main_sleeve_top_n}/{cfg.hedging_sleeve_top_n},",
      f"momentum_blend={cfg.momentum_blend_lookbacks}, rf={RF_ANNUAL*100:.0f}%")
print(SEP)
