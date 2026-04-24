"""
Statistical significance analysis for AMAAM.

Runs block-bootstrap Sharpe CIs, hypothesis tests (t-test, JK, alpha/beta OLS,
MDD bootstrap), permutation test, and academic benchmark comparison.  Results
are printed in the formatted table layout defined in the project brief.

Sections
--------
1. Block-bootstrap Sharpe confidence intervals (n=10,000, block=12 months)
2. Hypothesis tests (return t-test, Sharpe diff bootstrap, alpha/beta OLS,
   MDD bootstrap)
3. Permutation test on Sharpe ratio
4. Academic context vs published TAA benchmarks
"""

import sys
sys.path.insert(0, '/Users/GualyCr/Desktop/AMAAM/amaam/.claude/worktrees/reverent-mayer')

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from config.default_config import ModelConfig
from src.backtest.engine import run_backtest
from src.data.loader import load_validated_data


# ---------------------------------------------------------------------------
# Data setup  (identical to the brief's setup block)
# ---------------------------------------------------------------------------

data_dir = Path(
    '/Users/GualyCr/Desktop/AMAAM/amaam/.claude/worktrees/reverent-mayer/data/processed'
)
data_dict = load_validated_data(data_dir)
cfg = ModelConfig()
result = run_backtest(data_dict, cfg)

mr = result.monthly_returns.copy()
mr.index = mr.index.to_period('M')

spy_daily = data_dict['SPY']['Close']
spy_me = spy_daily.resample('ME').last()
spy_ret = spy_me.pct_change().dropna()
spy_ret.index = spy_ret.index.to_period('M')
spy_aligned = spy_ret.reindex(mr.index).dropna()
mr_aligned = mr.reindex(spy_aligned.index).dropna()

# IS / OOS masks
is_mask  = mr_aligned.index.year <= 2017
oos_mask = mr_aligned.index.year >= 2018

mr_is   = mr_aligned[is_mask]
mr_oos  = mr_aligned[oos_mask]
spy_is  = spy_aligned[is_mask]
spy_oos = spy_aligned[oos_mask]


# ---------------------------------------------------------------------------
# Helper: annualised Sharpe from a monthly return array
# ---------------------------------------------------------------------------

def sharpe(r: np.ndarray) -> float:
    """Annualised Sharpe ratio (risk-free = 0) from monthly returns."""
    if len(r) == 0 or r.std() == 0:
        return np.nan
    return (r.mean() / r.std()) * np.sqrt(12)


def max_drawdown(r: np.ndarray) -> float:
    """Maximum peak-to-trough drawdown from a monthly return array."""
    equity = np.cumprod(1.0 + r)
    peak   = np.maximum.accumulate(equity)
    dd     = (equity - peak) / peak
    return float(dd.min())


# ---------------------------------------------------------------------------
# Section 1: Block-bootstrap Sharpe confidence intervals
# ---------------------------------------------------------------------------

def block_bootstrap_sharpe(
    returns: pd.Series,
    n_bootstrap: int = 10_000,
    block_size: int = 12,
    seed: int = 42,
) -> np.ndarray:
    """
    Block bootstrap for Sharpe ratio CI.

    Sampling complete, overlapping blocks of length *block_size* preserves the
    serial-correlation structure present in momentum strategies (Politis &
    Romano 1994).  Each resample draws enough blocks to cover *n* observations.

    Parameters
    ----------
    returns : pd.Series
        Monthly return series.
    n_bootstrap : int
        Number of bootstrap resamples.
    block_size : int
        Block length in months.
    seed : int
        NumPy random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Array of bootstrap Sharpe ratios with NaN samples removed.
    """
    np.random.seed(seed)
    r = returns.values
    n = len(r)
    n_blocks = int(np.ceil(n / block_size))
    sharpes = []
    for _ in range(n_bootstrap):
        starts = np.random.randint(0, n - block_size + 1, size=n_blocks)
        sample = np.concatenate([r[s: s + block_size] for s in starts])[:n]
        ann_r  = (1 + sample.mean()) ** 12 - 1
        ann_v  = sample.std() * np.sqrt(12)
        sharpes.append(ann_r / ann_v if ann_v > 0 else np.nan)
    return np.array([s for s in sharpes if not np.isnan(s)])


def bootstrap_mdd(
    returns: pd.Series,
    n_bootstrap: int = 10_000,
    block_size: int = 12,
    seed: int = 42,
) -> np.ndarray:
    """
    Block bootstrap distribution of maximum drawdown.

    Parameters
    ----------
    returns : pd.Series
        Monthly return series.
    n_bootstrap : int
        Number of resamples.
    block_size : int
        Block length in months.
    seed : int
        Random seed.

    Returns
    -------
    np.ndarray
        Array of bootstrap MDD values (all <= 0).
    """
    np.random.seed(seed)
    r = returns.values
    n = len(r)
    n_blocks = int(np.ceil(n / block_size))
    mdds = []
    for _ in range(n_bootstrap):
        starts = np.random.randint(0, n - block_size + 1, size=n_blocks)
        sample = np.concatenate([r[s: s + block_size] for s in starts])[:n]
        mdds.append(max_drawdown(sample))
    return np.array(mdds)


# --- Run bootstraps for all periods ---

periods = {
    'Full': (mr_aligned, spy_aligned),
    'IS':   (mr_is,      spy_is),
    'OOS':  (mr_oos,     spy_oos),
}

bs_amaam = {}
bs_spy   = {}
for label, (amr, spr) in periods.items():
    bs_amaam[label] = block_bootstrap_sharpe(amr)
    bs_spy[label]   = block_bootstrap_sharpe(spr)

spy_sharpes = {k: sharpe(v[0].values) for k, v in periods.items()}


# ---------------------------------------------------------------------------
# Section 2: Hypothesis tests
# ---------------------------------------------------------------------------

# Test 1: t-test mean monthly return = 0
def ttest_mean(r: pd.Series):
    """One-sample t-test: H0 mean = 0, one-tailed (mean > 0)."""
    t, p_two = stats.ttest_1samp(r.values, popmean=0)
    p_one = p_two / 2 if t > 0 else 1.0 - p_two / 2
    return t, p_one


# Test 2: Bootstrap Sharpe difference (AMAAM - SPY)
def bootstrap_sharpe_diff(
    r_a: pd.Series,
    r_b: pd.Series,
    n_bootstrap: int = 10_000,
    block_size: int = 12,
    seed: int = 42,
) -> tuple:
    """
    Block bootstrap for the Sharpe ratio difference (A - B).

    Builds paired resamples (same block start indices for both series) so
    correlation between the strategies is preserved.  Returns the observed
    delta and the p-value for H0: delta = 0 (two-tailed).

    Parameters
    ----------
    r_a, r_b : pd.Series
        Monthly return series to compare.
    n_bootstrap : int
        Number of resamples.
    block_size : int
        Block length in months.
    seed : int
        Random seed.

    Returns
    -------
    tuple
        (observed_delta, p_value, bootstrap_deltas_array)
    """
    np.random.seed(seed)
    ra = r_a.values
    rb = r_b.values
    n  = len(ra)
    n_blocks = int(np.ceil(n / block_size))
    obs_delta = sharpe(ra) - sharpe(rb)
    deltas = []
    for _ in range(n_bootstrap):
        starts  = np.random.randint(0, n - block_size + 1, size=n_blocks)
        samp_a  = np.concatenate([ra[s: s + block_size] for s in starts])[:n]
        samp_b  = np.concatenate([rb[s: s + block_size] for s in starts])[:n]
        deltas.append(sharpe(samp_a) - sharpe(samp_b))
    deltas = np.array(deltas)
    # Two-tailed: centre bootstrap deltas around 0 then check extremes
    centred = deltas - deltas.mean()
    p_val   = float(np.mean(np.abs(centred) >= abs(obs_delta)))
    return obs_delta, p_val, deltas


# Test 3 / 5: OLS alpha and beta
def ols_alpha_beta(r_a: pd.Series, r_b: pd.Series):
    """
    OLS regression r_A = alpha + beta * r_B + eps.

    Monthly alpha is annualised to a yearly figure.  Returns (alpha_ann,
    beta, r_squared, t_alpha, p_alpha, t_beta, p_beta).
    """
    x = r_b.values
    y = r_a.values
    X = np.column_stack([np.ones(len(x)), x])
    res = np.linalg.lstsq(X, y, rcond=None)
    coeffs = res[0]
    alpha_m, beta = coeffs[0], coeffs[1]
    alpha_ann = (1 + alpha_m) ** 12 - 1

    y_hat  = X @ coeffs
    resid  = y - y_hat
    n      = len(y)
    k      = 2
    s2     = np.sum(resid ** 2) / (n - k)
    XtX_inv = np.linalg.inv(X.T @ X)
    se      = np.sqrt(np.diag(XtX_inv) * s2)

    t_a = alpha_m / se[0]
    t_b = beta    / se[1]
    df  = n - k
    p_a = 2 * stats.t.sf(abs(t_a), df)
    p_b = 2 * stats.t.sf(abs(t_b), df)

    ss_tot = np.sum((y - y.mean()) ** 2)
    ss_res = np.sum(resid ** 2)
    r2     = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return alpha_ann, beta, r2, t_a, p_a, t_b, p_b


# Test 4: MDD bootstrap
def reject_str(p_val: float, alpha: float) -> str:
    return "Reject" if p_val < alpha else "Fail"


# ---------------------------------------------------------------------------
# Section 3: Permutation test
# ---------------------------------------------------------------------------

def permutation_sharpe(
    returns: pd.Series,
    n_perm: int = 10_000,
    seed: int = 42,
) -> tuple:
    """
    Sign-flip permutation test for the Sharpe ratio.

    Randomly flips the sign of each monthly return with probability 0.5.
    This is the canonical "randomisation test" for momentum/TAA strategies
    (Harvey & Liu 2015; Romano & Wolf 2005): it destroys the directional
    *signal* (whether each period's bet was correct) while exactly preserving
    the magnitude distribution, giving a tight null that is far more
    conservative than naive shuffling.  A naive shuffle of monthly returns
    yields a constant Sharpe because mean and std are order-invariant
    statistics; sign-flip permutation breaks that invariance by converting
    some positive months to negative.

    The p-value answers: "If every monthly allocation were a fair coin flip
    (50 % chance of correct direction), what fraction of 10,000 trials would
    produce a Sharpe ratio at least as large as observed?"

    Parameters
    ----------
    returns : pd.Series
        Monthly return series.
    n_perm : int
        Number of sign-flip permutations.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    tuple
        (observed_sharpe, perm_mean, perm_std, z_score, p_value, perm_array)
    """
    np.random.seed(seed)
    r = returns.values
    obs_sr = sharpe(r)
    n = len(r)
    perm_srs = []
    for _ in range(n_perm):
        # Each return gets a ±1 sign flip with equal probability.
        # |r| is preserved; only direction is randomised.
        signs = np.random.choice([-1.0, 1.0], size=n)
        flipped = r * signs
        perm_srs.append(sharpe(flipped))
    perm_srs = np.array([s for s in perm_srs if not np.isnan(s)])
    p_val  = float(np.mean(perm_srs >= obs_sr))
    z_sc   = (obs_sr - perm_srs.mean()) / perm_srs.std() if perm_srs.std() > 0 else np.nan
    return obs_sr, perm_srs.mean(), perm_srs.std(), z_sc, p_val, perm_srs


# ---------------------------------------------------------------------------
# Section 4: Academic benchmark comparison
# ---------------------------------------------------------------------------

def amaam_sharpe_for_period(start_year: int, end_year: int) -> float:
    """
    Compute AMAAM annualised Sharpe over a sub-period defined by calendar year.

    Uses the period-indexed mr_aligned series populated at module level.
    Returns NaN when fewer than 12 months of data are available.

    Parameters
    ----------
    start_year : int
        First calendar year to include (inclusive).
    end_year : int
        Last calendar year to include (inclusive).

    Returns
    -------
    float
        Annualised Sharpe ratio for the requested sub-period.
    """
    mask = (mr_aligned.index.year >= start_year) & (mr_aligned.index.year <= end_year)
    sub  = mr_aligned[mask]
    if len(sub) < 12:
        return np.nan
    return sharpe(sub.values)


# ---------------------------------------------------------------------------
# Compute all statistics
# ---------------------------------------------------------------------------

# --- Section 1 point estimates ---
pt_sharpe = {k: sharpe(v[0].values) for k, v in periods.items()}
pt_spy    = {k: sharpe(v[1].values) for k, v in periods.items()}

# --- Section 2 hypothesis tests ---
ttest_results = {k: ttest_mean(v[0]) for k, v in periods.items()}
sr_diff       = {k: bootstrap_sharpe_diff(v[0], v[1]) for k, v in periods.items()}
ols_results   = {k: ols_alpha_beta(v[0], v[1]) for k, v in periods.items()}

# MDD on full aligned period
mdd_obs     = max_drawdown(mr_aligned.values)
mdd_spy_obs = max_drawdown(spy_aligned.values)
bs_mdd_amaam = bootstrap_mdd(mr_aligned, seed=42)
bs_mdd_spy   = bootstrap_mdd(spy_aligned, seed=43)
p_mdd_lt_spy = float(np.mean(bs_mdd_amaam > bs_mdd_spy))   # P(AMAAM MDD < SPY MDD)
# Note: MDD values are negative; AMAAM MDD < SPY MDD means AMAAM drawdown is
# *shallower* (closer to zero), i.e. bs_mdd_amaam > bs_mdd_spy in numeric terms.

# --- Section 3 permutation test ---
perm_obs, perm_mean, perm_std, perm_z, perm_p, _ = permutation_sharpe(mr_aligned)

# --- Section 4 benchmarks ---
benchmarks = [
    # (name, period_label, pub_start, pub_end, reported_sr)
    ("Faber GTAA (2007)",    "1973-2007", 2004, 2007, 0.93),
    ("Antonacci GEM (2014)", "1974-2013", 2004, 2013, 0.73),
    ("Keller FAA (2012)",    "1998-2012", 2004, 2012, 1.01),
    ("AQR MOM factor",       "Long-run",  2004, 2024, 0.60),
    ("MSCI World Momentum",  "Recent",    2014, 2024, 0.70),
]


# ---------------------------------------------------------------------------
# Formatted output
# ---------------------------------------------------------------------------

W = 72
SEP = "=" * W


def fmt_ci(arr: np.ndarray, alpha: float) -> str:
    lo = np.percentile(arr, alpha / 2 * 100)
    hi = np.percentile(arr, (1 - alpha / 2) * 100)
    return f"[{lo:+.3f}, {hi:+.3f}]"


def fmt_pct(p: float) -> str:
    return f"{p * 100:.1f}%"


print(SEP)
print("AMAAM STATISTICAL SIGNIFICANCE ANALYSIS")
print(SEP)
print()

# ===========================================================================
# Section 1
# ===========================================================================
print("SECTION 1: BOOTSTRAP SHARPE CONFIDENCE INTERVALS")
print("(Block bootstrap, block=12mo, n=10,000 resamples)")
print()

labels      = ["Full Period", "IS (2004-17)", "OOS (2018-26)"]
period_keys = ["Full", "IS", "OOS"]

col_w = 15

def row(*cells):
    return (
        f"{cells[0]:<25}"
        + "".join(f"{str(c):<{col_w}}" for c in cells[1:])
    )

print(row("", *labels))
print("-" * W)

# AMAAM point SR
print(row(
    "AMAAM Point SR",
    *[f"{pt_sharpe[k]:+.3f}" for k in period_keys],
))

# Bootstrap mean SR
print(row(
    "Bootstrap Mean SR",
    *[f"{bs_amaam[k].mean():+.3f}" for k in period_keys],
))

# 95% CI
print(row(
    "95% CI",
    *[fmt_ci(bs_amaam[k], 0.05) for k in period_keys],
))

# 99% CI
print(row(
    "99% CI",
    *[fmt_ci(bs_amaam[k], 0.01) for k in period_keys],
))

# P(SR > 0)
print(row(
    "P(SR > 0)",
    *[fmt_pct(np.mean(bs_amaam[k] > 0)) for k in period_keys],
))

# P(SR > SPY SR)
print(row(
    "P(SR > SPY SR)",
    *[fmt_pct(np.mean(bs_amaam[k] > pt_spy[k])) for k in period_keys],
))

print()
# SPY rows
print(row(
    "SPY Point SR",
    *[f"{pt_spy[k]:+.3f}" for k in period_keys],
))
print(row(
    "SPY 95% CI",
    *[fmt_ci(bs_spy[k], 0.05) for k in period_keys],
))

print()

# ===========================================================================
# Section 2
# ===========================================================================
print("SECTION 2: HYPOTHESIS TESTS")
print()

# Test 1
print("Test 1: H0: AMAAM monthly mean return = 0")
for k in period_keys:
    t, p = ttest_results[k]
    r1 = reject_str(p, 0.01)
    r5 = reject_str(p, 0.05)
    lbl = {"Full": "Full", "IS": "IS  ", "OOS": "OOS "}[k]
    print(
        f"  {lbl}:  t={t:+.2f}, p={p:.4f}"
        f"  →  {r1} at 1% | {r5} at 5%"
    )

print()

# Test 2
print("Test 2: H0: AMAAM Sharpe = SPY Sharpe (bootstrap difference test)")
for k in period_keys:
    delta, pval, _ = sr_diff[k]
    r5  = reject_str(pval, 0.05)
    r10 = reject_str(pval, 0.10)
    lbl = {"Full": "Full", "IS": "IS  ", "OOS": "OOS "}[k]
    print(
        f"  {lbl}:  ΔSR={delta:+.3f}, p={pval:.4f}"
        f"  →  {r5} at 5% | {r10} at 10%"
    )

print()

# Test 3: OLS alpha / beta
print("Test 3: Alpha vs SPY (OLS regression)")
for k in period_keys:
    alpha_ann, beta, r2, t_a, p_a, t_b, p_b = ols_results[k]
    lbl = {"Full": "Full", "IS": "IS  ", "OOS": "OOS "}[k]
    print(
        f"  {lbl}:  α={alpha_ann*100:+.2f}%/yr, β={beta:.2f}, R²={r2:.2f},"
        f"  t(α)={t_a:+.2f}, p(α)={p_a:.4f}"
    )

print()

# Test 4: MDD significance
print("Test 4: MDD significance (bootstrap, n=10,000)")
mdd_95_lo = np.percentile(bs_mdd_amaam, 2.5)
mdd_95_hi = np.percentile(bs_mdd_amaam, 97.5)
print(f"  Observed MDD:       {mdd_obs*100:+.2f}%")
print(f"  Bootstrap mean:     {bs_mdd_amaam.mean()*100:+.2f}%")
print(f"  95% CI:            [{mdd_95_lo*100:+.2f}%, {mdd_95_hi*100:+.2f}%]")
print(f"  SPY observed MDD:   {mdd_spy_obs*100:+.2f}%")
print(f"  P(AMAAM MDD < SPY MDD, i.e. shallower): {fmt_pct(p_mdd_lt_spy)}")

print()

# ===========================================================================
# Section 3
# ===========================================================================
print("SECTION 3: PERMUTATION TEST")
sig1  = "statistically significant" if perm_p < 0.01 else "not significant"
sig5  = "statistically significant" if perm_p < 0.05 else "not significant"
print(f"  Observed Sharpe:          {perm_obs:+.3f}")
print(f"  Permutation mean:         {perm_mean:+.3f}")
print(f"  Permutation std:          {perm_std:.3f}")
print(f"  Z-score:                  {perm_z:+.2f}")
print(f"  P-value (SR >= observed): {perm_p:.4f}")
print(f"  → Signal is {sig1} at 1%")
print(f"  → Signal is {sig5} at 5%")

print()

# ===========================================================================
# Section 4: Academic benchmarks
# ===========================================================================
print("SECTION 4: ACADEMIC CONTEXT")

hdr = (
    f"{'Benchmark':<26}"
    f"{'Period':<12}"
    f"{'Reported SR':<13}"
    f"{'AMAAM SR':<12}"
    f"{'Verdict'}"
)
print(hdr)
print("-" * W)

for name, period_lbl, s_yr, e_yr, rep_sr in benchmarks:
    amsr = amaam_sharpe_for_period(s_yr, e_yr)
    if np.isnan(amsr):
        amsr_str = "  N/A  "
        verdict  = "Insufficient data"
    else:
        amsr_str = f"{amsr:+.3f}"
        if amsr > rep_sr * 1.05:
            verdict = "Beats"
        elif amsr >= rep_sr * 0.95:
            verdict = "Matches"
        else:
            verdict = "Trails"
    print(
        f"{name:<26}"
        f"{period_lbl:<12}"
        f"~{rep_sr:.2f}"
        f"{'':>8}"
        f"{amsr_str:<12}"
        f"{verdict}"
    )

print()
print(SEP)

# ---------------------------------------------------------------------------
# Extra: full-period OLS summary (beta, R²)
# ---------------------------------------------------------------------------
print()
print("SUPPLEMENTARY: Full-Period Regression Summary")
print("-" * W)
for k in period_keys:
    alpha_ann, beta, r2, t_a, p_a, t_b, p_b = ols_results[k]
    lbl = {"Full": "Full", "IS": "IS  ", "OOS": "OOS "}[k]
    print(
        f"  {lbl}:  β={beta:.3f}, R²={r2:.3f},"
        f"  t(β)={t_b:+.2f}, p(β)={p_b:.4f}"
    )

print()
print("Data spans (aligned periods):")
print(f"  Full: {mr_aligned.index[0]}  →  {mr_aligned.index[-1]}"
      f"  ({len(mr_aligned)} months)")
print(f"  IS:   {mr_is.index[0]}  →  {mr_is.index[-1]}"
      f"  ({len(mr_is)} months)")
print(f"  OOS:  {mr_oos.index[0]}  →  {mr_oos.index[-1]}"
      f"  ({len(mr_oos)} months)")
print(SEP)
