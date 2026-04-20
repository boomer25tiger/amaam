"""
Deflated Sharpe Ratio (DSR) analysis — Bailey, Borwein, López de Prado & Zhu (2014).

The DSR answers: after accounting for the number of configurations tested,
is the selected strategy's OOS Sharpe statistically significant, or could
it plausibly arise from luck alone?

Steps:
  1. Compute the Probabilistic Sharpe Ratio (PSR) for the candidate against
     the baseline: P(true SR > baseline SR | observed OOS returns).
  2. Compute the expected maximum Sharpe under the null (no skill) given
     N=39 trials — this is the DSR hurdle.
  3. Report whether the candidate clears the DSR hurdle at 95% confidence.
  4. Show sensitivity of DSR to assumed number of independent trials
     (39 total; effective N likely lower due to correlation between configs).

Formulae (Bailey et al. 2014):
  PSR(SR*) = Φ{(SR̂ - SR*) · √(T-1) / √(1 - γ₃·SR̂ + (γ₄-1)/4 · SR̂²)}

  E[max SR | N trials] ≈ √(V̂[SR]) ·
      [(1-γ)·Φ⁻¹(1 - 1/N) + γ·Φ⁻¹(1 - 1/(N·e·ln(N)))]

  where γ = 0.5772… (Euler-Mascheroni), V̂[SR] ≈ 1/T for SR near zero.

  DSR = PSR(E[max SR | N])
"""

import sys
sys.path.insert(0, "/Users/GualyCr/Desktop/AMAAM/amaam/.claude/worktrees/reverent-mayer")

import numpy as np
import pandas as pd
from dataclasses import replace
from scipy.stats import norm
from scipy.special import digamma

from config.default_config import ModelConfig
from src.data.loader import load_validated_data
from src.backtest.engine import run_backtest

EULER_MASCHERONI = 0.5772156649
CANDIDATE = dict(weight_momentum=0.65, weight_volatility=0.25, weight_correlation=0.10)
BASELINE  = dict(weight_momentum=0.50, weight_volatility=0.25, weight_correlation=0.25)


def oos_returns(result, oos_start, oos_end):
    r = result.monthly_returns
    return r[(r.index >= oos_start) & (r.index <= oos_end)]


def annualised_sr(r, rf=0.02):
    ann_ret = (1 + r).prod() ** (12 / len(r)) - 1
    ann_vol = r.std() * np.sqrt(12)
    return (ann_ret - rf) / ann_vol


def psr(r_series, sr_star, rf=0.02):
    """
    Probabilistic Sharpe Ratio: P(true SR > sr_star | observed r_series).
    Uses the non-iid correction for skewness and excess kurtosis.

    Parameters
    ----------
    r_series : pd.Series   Monthly returns (not annualised)
    sr_star  : float       Benchmark Sharpe (annualised)
    rf       : float       Annual risk-free rate

    Returns
    -------
    float  Probability in [0, 1]
    """
    T   = len(r_series)
    sr_hat = annualised_sr(r_series, rf)

    # Moments of monthly returns
    mu    = r_series.mean()
    sigma = r_series.std(ddof=1)
    skew  = r_series.skew()
    kurt  = r_series.kurtosis() + 3   # pandas kurtosis is excess; convert to raw

    # Monthly SR hat (for the variance formula which uses un-annualised SR)
    sr_m = mu / sigma  # monthly Sharpe (rf absorbed into mu is conservative; keep simple)

    # Variance of the SR estimator (Mertens 2002 / Bailey-López de Prado 2012)
    var_sr = (1 / (T - 1)) * (1 - skew * sr_m + ((kurt - 1) / 4) * sr_m**2)
    if var_sr <= 0:
        var_sr = 1 / T   # fallback

    # Annualise: SR_annual = SR_monthly * sqrt(12), so Var_annual = Var_monthly * 12
    var_sr_ann = var_sr * 12

    z = (sr_hat - sr_star) / np.sqrt(var_sr_ann)
    return float(norm.cdf(z)), sr_hat, np.sqrt(var_sr_ann)


def expected_max_sr(n_trials, T, sr_hat=0.0, skew=0.0, kurt=3.0, rf=0.0):
    """
    Expected maximum Sharpe ratio over n_trials independent strategies
    under the null of zero true SR (Bailey et al. 2014, eq. 8).

    Uses the asymptotic extreme-value approximation.
    """
    # Variance of SR estimator under null
    var_sr_m = (1 / (T - 1)) * (1 + (kurt - 1) / 4 * 0)  # SR=0 under null
    var_sr_ann = var_sr_m * 12
    sigma_sr = np.sqrt(var_sr_ann)

    gamma = EULER_MASCHERONI
    e     = np.e

    # Avoid log(1) = 0 for n_trials = 1
    if n_trials <= 1:
        return 0.0

    term1 = (1 - gamma) * norm.ppf(1 - 1 / n_trials)
    inner = n_trials * e * np.log(n_trials)
    term2 = gamma * norm.ppf(1 - 1 / inner) if inner > 1 else 0.0

    return sigma_sr * (term1 + term2)


def main():
    print("Loading data and running backtests…")
    data_dict = load_validated_data("data/processed")
    base_cfg  = ModelConfig()

    OOS_START, OOS_END = base_cfg.holdout_start, base_cfg.backtest_end

    cand_res = run_backtest(data_dict, replace(base_cfg, **CANDIDATE))
    base_res = run_backtest(data_dict, replace(base_cfg, **BASELINE))

    r_cand = oos_returns(cand_res, OOS_START, OOS_END)
    r_base = oos_returns(base_res, OOS_START, OOS_END)

    T         = len(r_cand)
    N_TOTAL   = 39    # configurations tested in grid sweep
    # Effective N is likely lower; configs are correlated (same momentum
    # signal, overlapping assets). We test a range.

    sr_baseline = annualised_sr(r_base)

    print(f"\nOOS period: {OOS_START} → {OOS_END}  ({T} months)")
    print(f"Baseline OOS Sharpe  (wM=0.50/wV=0.25/wC=0.25): {sr_baseline:.4f}")

    # ── PSR: candidate vs baseline ────────────────────────────────────────────
    prob, sr_cand, se_sr = psr(r_cand, sr_baseline)
    print(f"\nCandidate OOS Sharpe (wM=0.65/wV=0.25/wC=0.10): {sr_cand:.4f}  (SE={se_sr:.4f})")
    print(f"PSR — P(true SR_candidate > SR_baseline): {prob:.4f}  ({prob*100:.1f}%)")

    # ── DSR: correct for N trials ─────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("DEFLATED SHARPE RATIO — sensitivity to assumed number of trials")
    print(f"{'N trials':>10} {'E[max SR]':>10} {'DSR':>8} {'Clears 95%?':>12}")
    print("-" * 70)

    for n in [5, 10, 15, 20, 25, 30, 39]:
        e_max = expected_max_sr(n, T)
        dsr_val, _, _ = psr(r_cand, e_max)
        clears = "✓  YES" if dsr_val >= 0.95 else "✗  no"
        marker = " ← (actual N)" if n == N_TOTAL else ""
        print(f"{n:>10}  {e_max:>10.4f}  {dsr_val:>8.4f}  {clears}{marker}")

    # ── Full detail at N=39 ───────────────────────────────────────────────────
    e_max_full = expected_max_sr(N_TOTAL, T)
    dsr_full, _, _ = psr(r_cand, e_max_full)

    print(f"\n{'='*70}")
    print(f"SUMMARY (N={N_TOTAL} trials, T={T} OOS months)")
    print(f"  Candidate SR             : {sr_cand:.4f}")
    print(f"  Baseline SR (SR*)        : {sr_baseline:.4f}")
    print(f"  PSR vs baseline          : {prob*100:.1f}%")
    print(f"  E[max SR | N={N_TOTAL}]       : {e_max_full:.4f}")
    print(f"  DSR (vs E[max SR])       : {dsr_full*100:.1f}%")
    print(f"  Clears 95% DSR hurdle?   : {'YES ✓' if dsr_full >= 0.95 else 'NO ✗'}")
    print(f"  Clears 90% DSR hurdle?   : {'YES ✓' if dsr_full >= 0.90 else 'NO ✗'}")

    # ── Skewness / kurtosis of candidate OOS returns ──────────────────────────
    skew = r_cand.skew()
    kurt = r_cand.kurtosis() + 3   # raw kurtosis
    print(f"\n  OOS return skewness : {skew:+.3f}")
    print(f"  OOS return kurtosis : {kurt:.3f}  (normal = 3.0)")

    if abs(skew) > 0.5 or abs(kurt - 3) > 1.0:
        print("  ⚠  Non-normal returns — PSR non-normality correction is active")
    else:
        print("  Returns roughly normal — standard SR inference is adequate")

    # ── How high would SR need to be to clear DSR at 95%? ────────────────────
    # Find SR* such that PSR(SR*) = 0.95 at N=39
    # Binary search
    lo, hi = 0.0, 3.0
    for _ in range(50):
        mid = (lo + hi) / 2
        p, _, _ = psr(r_cand, mid)
        if p > 0.95:
            lo = mid
        else:
            hi = mid
    sr_needed = (lo + hi) / 2
    print(f"\n  SR needed to clear DSR@95% at N=39: {sr_needed:.4f}")
    print(f"  Candidate SR shortfall            : {sr_cand - sr_needed:+.4f}")

    # ── Interpretation ────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("INTERPRETATION")
    if dsr_full >= 0.95:
        print("  The candidate clears the 95% DSR hurdle. Even after penalising")
        print("  for 39 configurations tested, the OOS Sharpe improvement is")
        print("  statistically distinguishable from luck at the 5% level.")
    elif dsr_full >= 0.90:
        print("  The candidate clears the 90% DSR hurdle but not 95%. The result")
        print("  is suggestive but not conclusive. Use alongside walk-forward and")
        print("  theoretical justification rather than treating it as proof.")
    else:
        print("  The candidate does NOT clear the 90% DSR hurdle. The OOS gain")
        print("  is within the range expected from random search over 39 configs.")
        print("  Rely on theoretical motivation and walk-forward consistency,")
        print("  not the magnitude of the OOS Sharpe improvement, to justify the change.")


if __name__ == "__main__":
    main()
