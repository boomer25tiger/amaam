# AMAAM — Adaptive Multi-Asset Allocation Model

A quantitative, rules-based asset allocation system that selects and weights
ETFs across equities, fixed income, commodities, and real assets using a
composite rank score built from momentum, volatility, and cross-asset
correlation signals.

---

## Table of Contents

1. [Model Overview](#model-overview)
2. [Academic Foundation](#academic-foundation)
3. [Environment Setup](#environment-setup)
4. [Data Download](#data-download)
5. [Running the Backtest](#running-the-backtest)
6. [Generating Charts and Reports](#generating-charts-and-reports)
7. [Live Signal Generation](#live-signal-generation)
8. [Key Results](#key-results)
9. [Additional Statistics](#additional-statistics)
10. [Known Limitations](#known-limitations)
11. [References](#references)
12. [Module Structure and Refactoring Notes](#module-structure-and-refactoring-notes)

---

## Model Overview

AMAAM allocates capital across a **two-sleeve universe** of 22 ETFs every
calendar month using a Total Rank (TRank) composite score:

```
TRank(i) = wM × rank(M_i) + wV × rank(V_i) + wC × rank(C_i) + wT × T_i + M_i / n
```

| Symbol | Component | Default weight |
|--------|-----------|---------------|
| M      | Momentum (blend of 1-, 3-, 6-month lookbacks) | wM = 0.65 |
| V      | Volatility (Yang-Zhang estimator, 126-day window; lower = better rank) | wV = 0.25 |
| C      | Cross-asset correlation (lower avg correlation = better rank) | wC = 0.10 |
| T      | Trend filter (SMA-200, binary 0/1) | wT = 1.0 |

The tiebreaker term `M_i / n` (where n is universe size) breaks rank ties
deterministically using raw momentum without inflating the composite score.

### Two-Sleeve Architecture

**Main sleeve** (16 ETFs — SPY, QQQ, IWM, EFA, EEM, VNQ, TLT, IEF, SHY,
LQD, HYG, GLD, SLV, DBC, XLE, XLU):
- Select the **top-6 ETFs** by TRank each month.
- Equal-weight within selected set (1/6 each).

**Hedging sleeve** (6 ETFs — BIL, SHV, SGOV, JAAA, JPST, FLOT):
- Select the **top-2 ETFs** by TRank each month.
- Equal-weight within selected set (1/2 each).

Sleeve allocation is dynamic: cash equivalents (SHY, BIL, SHV, SGOV) act as
defensive positions during weak trend environments; no hard-coded sleeve
split ratio.

### Rebalancing

Monthly, on the last trading day of each calendar month. Transaction costs
are modelled as proportional to turnover (default: 5 bps one-way).

---

## Academic Foundation

AMAAM extends two prize-winning frameworks developed by Giordano:

1. **Ranked Asset Allocation Model (RAAM)**
   Giordano, N. (2018). *Ranked Asset Allocation Model*.
   **2018 Charles H. Dow Award**, Market Technicians Association.
   Introduces the TRank composite score combining momentum, volatility, and
   correlation ranks for cross-sectional ETF selection.

2. **Antifragile Asset Allocation Model (AAAM)**
   Giordano, N. (2021). *Antifragile Asset Allocation Model*.
   **1st Place, SIAT Technical Analyst Award** (Italian Society of Technical
   Analysts).
   Extends RAAM with a dynamic hedging sleeve and antifragile regime-switching
   properties, allowing the model to benefit from volatility spikes rather than
   simply surviving them.

---

## Environment Setup

**Requirements:** Python 3.13 or later.

```bash
# 1. Clone the repository
git clone <repo-url>
cd amaam

# 2. Create and activate a virtual environment
python3.13 -m venv .venv
source .venv/bin/activate          # macOS/Linux
# .venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

**Core dependencies:**

| Package | Version | Purpose |
|---------|---------|---------|
| numpy | ≥ 1.24 | Numerical arrays, rolling calculations |
| pandas | ≥ 2.0 | Time-series data management |
| scipy | ≥ 1.10 | Statistical tests (bootstrap, permutation) |
| yfinance | ≥ 0.2.30 | Historical OHLCV data download |
| exchange-calendars | ≥ 4.2 | Trading-day calendar for rebalancing dates |
| matplotlib | ≥ 3.7 | Static chart generation (PNG) |
| plotly | ≥ 5.15 | Interactive chart generation (HTML) |
| tqdm | ≥ 4.65 | Progress bars for long grid searches |
| pyyaml | ≥ 6.0 | Configuration file parsing |
| python-dotenv | ≥ 1.0 | Environment variable management |
| pytest / pytest-cov | ≥ 7.0 / 4.0 | Unit testing and coverage |

> **Live trading (Phase 9):** `schwab-py >= 1.0` must be installed separately
> when the Schwab brokerage integration is needed. It is intentionally excluded
> from `requirements.txt` to avoid a mandatory dependency during research.

---

## Data Download

All price data is fetched from Yahoo Finance via `yfinance` and cached locally
in `data/raw/` (git-ignored). A validator then checks for gaps, stale prices,
and corporate-action anomalies before writing cleaned data to `data/processed/`.

```bash
python3.13 scripts/download_data.py
```

This downloads adjusted daily OHLCV prices for all 22 ETFs plus benchmark
tickers (SPY, AGG) from 2004-01-01 to the current date. The script is
idempotent — existing files are not re-downloaded unless `--force` is passed.

**Estimated download time:** ~2 minutes on a typical broadband connection.

---

## Running the Backtest

```bash
python3.13 scripts/run_backtest.py
```

This runs the full backtest from 2004-01-01 to the most recent data available,
prints a summary metrics table to stdout, and writes results to
`results/backtest_results.pkl` for downstream use.

**Configuration** is controlled entirely through `config/default_config.py`.
Key parameters:

```python
ModelConfig(
    weight_momentum     = 0.65,   # TRank momentum weight
    weight_volatility   = 0.25,   # TRank volatility weight (lower vol = better)
    weight_correlation  = 0.10,   # TRank correlation weight (lower corr = better)
    weight_trend        = 1.0,    # Trend filter multiplier
    top_n_main          = 6,      # ETFs selected from main sleeve
    top_n_hedge         = 2,      # ETFs selected from hedging sleeve
    momentum_blend      = True,   # Blend 1-/3-/6-month lookbacks
    trend_method        = "sma200",
    yang_zhang_window   = 126,    # ~6 months of trading days
    transaction_cost    = 0.0005, # 5 bps one-way
    start_date          = "2004-01-01",
    end_date            = "2026-04-10",
)
```

Do **not** hard-code parameter values in logic code; always import from
`ModelConfig`.

### Walk-Forward Validation

```bash
python3.13 scripts/walk_forward.py
```

Runs 39 weight configurations (wM ∈ {0.45, 0.50, 0.55, 0.60, 0.65} ×
wC ∈ {0.05 … 0.45, step 0.05}, subject to wV ≥ 0.05) across 6 expanding
training / 2-year test folds. Prints per-fold results and a stacked OOS
summary table.

---

## Generating Charts and Reports

```bash
python3.13 scripts/generate_reports.py
```

Produces **32 PNG charts** and **23 interactive HTML charts** in
`reports/figures/`. The script also runs the walk-forward grid search
internally so chart 32 (walk-forward validation) is always up to date.

**Estimated runtime:** 10–20 minutes (dominated by the 39 × full-history
backtests inside the walk-forward grid search).

### Chart Inventory

| # | Title | Type |
|---|-------|------|
| 01 | Equity Curves (Growth of $100) | Performance |
| 02 | Annual Returns Comparison | Performance |
| 03 | Rolling 12-Month Returns | Performance |
| 04 | Drawdown Comparison | Risk |
| 05 | IS vs OOS Performance Metrics | Validation |
| 06 | Monthly Returns Heatmap | Performance |
| 07 | Rolling Sharpe Ratio (36-month) | Risk-adjusted |
| 08 | Rolling Volatility (36-month) | Risk |
| 09 | Correlation Matrix | Risk |
| 10 | Asset Allocation Over Time | Allocation |
| 11 | Allocation Heatmap | Allocation |
| 12 | Sleeve Composition | Allocation |
| 13 | Monthly Turnover | Allocation |
| 14 | Regime Performance | Analysis |
| 15 | Selection Sensitivity | Analysis |
| 16 | Weighting Scheme Comparison | Analysis |
| 17 | Cost Scenarios — Equity Curves | Analysis |
| 18 | Cost Scenarios — Annual Returns | Analysis |
| 19 | Benchmark Comparison | Performance |
| 20 | Statistical Significance | Analysis |
| 21 | Bootstrap Distribution | Analysis |
| 22 | Performance Summary Table | Summary |
| 23 | Risk Summary Table | Summary |
| 24 | Risk-Return Scatter | Supplementary |
| 25 | Beta Scatter vs SPY | Supplementary |
| 26 | Rolling SPY Correlation | Supplementary |
| 27 | Drawdown Duration | Supplementary |
| 28 | Win Rate & Payoff Statistics | Supplementary |
| 29 | VaR / CVaR | Supplementary |
| 30 | Rolling Calmar Ratio (36-month) | Supplementary |
| 31 | Return Autocorrelation (ACF) | Supplementary |
| 32 | Walk-Forward Validation | Validation |

---

## Live Signal Generation

> **Status:** Phase 9 implementation is in progress. The Schwab brokerage
> integration (`schwab-py`) is not yet complete. The script below runs but
> outputs a dry-run signal only — no orders are placed.

```bash
python3.13 scripts/run_live_signal.py
```

This script:
1. Downloads the latest prices for all 22 ETFs.
2. Computes TRank scores as-of today using the same engine as the backtest.
3. Prints the target allocation (ETF tickers and weights).
4. (When Phase 9 is complete) submits rebalancing orders to Schwab via OAuth.

---

## Key Results

Full backtest: **2004-01-01 → 2026-04-10** (250 months).

### Performance Summary

| Metric | AMAAM | SPY | 60/40 | 7Twelve |
|--------|------:|----:|------:|--------:|
| Ann. Return | **10.68%** | 10.23% | 7.67% | 7.40% |
| Ann. Volatility | **12.40%** | 16.24% | 10.18% | 12.00% |
| Sharpe Ratio (rf=2%) | **0.723** | 0.562 | 0.584 | 0.493 |
| Sortino Ratio | **1.437** | 0.934 | 1.142 | 0.910 |
| Calmar Ratio | **0.578** | 0.193 | 0.231 | 0.197 |
| Max Drawdown | **−18.47%** | −52.90% | −33.21% | −37.54% |
| MDD Duration (months) | **27** | 53 | 38 | 27 |

### Return Distribution

| Metric | AMAAM | SPY | 60/40 | 7Twelve |
|--------|------:|----:|------:|--------:|
| Best Month | +9.73% | +15.64% | +10.29% | +15.31% |
| Worst Month | −11.01% | −19.89% | −12.27% | −17.31% |
| Best Calendar Year | +37.36% | +39.06% | +27.93% | +43.22% |
| Worst Calendar Year | −13.82% | −43.22% | −27.44% | −31.92% |
| % Positive Months | 62.8% | 65.0% | 66.2% | 63.5% |
| % Positive Years | 86.4% | 87.0% | 87.0% | 82.6% |
| Win/Loss Payoff Ratio | **1.108×** | 0.912× | 0.940× | 0.988× |

> AMAAM is the only strategy with a payoff ratio above 1.0, meaning average
> winning months exceed average losing months in magnitude — asymmetric return
> profile.

### Higher Moments

| Metric | AMAAM | SPY | 60/40 | 7Twelve |
|--------|------:|----:|------:|--------:|
| Skewness | −0.222 | −0.705 | −0.615 | −0.805 |
| Excess Kurtosis | +0.032 | +2.805 | +2.969 | +5.658 |

AMAAM exhibits near-zero kurtosis (close to Gaussian tail behaviour), while
all benchmarks show significant leptokurtosis — fat tails on the downside.

### Turnover

| Metric | Value |
|--------|-------|
| Average Monthly Turnover | 74.5% |
| Annualised Turnover | ~893% |
| Total Return (gross) | +705.36% over 250 months |

High turnover is an inherent cost of monthly ETF rotation. At 5 bps one-way
transaction cost the impact on the Sharpe ratio is ~0.04 (shown in chart 17).

---

## Additional Statistics

### Statistical Significance

| Test | Result |
|------|--------|
| Bootstrap 95% CI (Sharpe) | [0.302, 1.127] — excludes zero |
| Permutation test Z-score vs SPY | Z = +4.06, p ≈ 0 (one-tailed) |
| Alpha vs SPY (OLS) | +11.55% / yr, p = 0.0001 |
| Beta vs SPY | −0.003 (statistically indistinguishable from zero) |
| R² vs SPY | ≈ 0 — returns are orthogonal to SPY |

### Sub-period Robustness

| Period | Sharpe |
|--------|--------|
| In-Sample (2004–2018, 151 months) | 0.761 |
| Out-of-Sample (2018–2026, 98 months) | 0.671 |
| Holdout (2024–2026, 28 months — run-once) | 1.103 |

IS and OOS Sharpes are both positive and within a normal sampling range of
each other — no evidence of in-sample overfitting.

### Walk-Forward Validation (Stacked OOS)

6-fold expanding-window validation, 2-year test windows each:

| Strategy | Stacked OOS Sharpe |
|----------|--------------------|
| Walk-forward winner (adaptive config) | 0.739 |
| Fixed candidate wM=0.65/wV=0.25/wC=0.10 | 0.739 |
| Baseline wM=0.50/wV=0.25/wC=0.25 | 0.586 |

The canonical wM=0.65 configuration wins 5/6 training folds and produces a
stacked OOS Sharpe 26% higher than the equal-weight baseline.

### Return Autocorrelation (ACF, 24 lags)

Confidence interval (95%, N=250): ±0.124

| Lag | ACF | Significant? |
|-----|-----|-------------|
| 1 | −0.187 | Yes (negative) |
| 10 | −0.143 | Yes (negative) |
| 12 | −0.126 | Yes (negative) |
| 13 | +0.135 | Yes (positive) |
| 18 | −0.133 | Yes (negative) |

The lag-1 negative autocorrelation (−0.187) is a mechanical artefact of
monthly rebalancing: by construction, the portfolio sells recent winners and
buys recent laggards each period, creating mean-reversion in the return series
at a 1-month horizon. It does not indicate negative serial dependence in the
underlying signal.

---

## Known Limitations

The following limitations are documented in the model specification (Section 13)
and supplemented by findings from this implementation and validation.

### From the Specification (Section 13)

1. **Survivorship bias** — The ETF universe is fixed at inception; delisted
   or merged ETFs are not modelled. Results may overstate returns achievable
   in real time if universe selection was informed by hindsight.

2. **Look-ahead bias** — All signals use data available strictly before each
   rebalancing date. However, adjusted price data downloaded today may reflect
   retroactive corporate-action adjustments not available at the original trade
   date.

3. **Transaction cost model** — The model uses a flat proportional cost (5 bps
   one-way). Real-world costs include bid-ask spreads, market impact (for large
   AUM), and broker commissions that are non-proportional for small lots.

4. **Liquidity assumptions** — All ETFs are assumed infinitely liquid; fills are
   modelled at closing prices on the rebalancing day. Slippage from intraday
   execution is not captured.

5. **Fixed risk-free rate** — rf = 2%/yr throughout the backtest. The actual
   short rate ranged from ≈0% (2009–2015, 2020–2022) to ≈5.3% (2023–2024),
   which means Sharpe ratios in low-rate periods are upward-biased and in
   high-rate periods are downward-biased relative to a time-varying rf.

6. **No leverage or short-selling** — The model is long-only. Periods of broad
   market weakness reduce performance rather than producing gains.

7. **ETF inception dates** — Several ETFs in the universe launched after 2004.
   The loader splices proxy returns (related ETFs or indices) for pre-inception
   periods. Proxy quality varies and may not perfectly replicate what the ETF
   would have returned.

8. **Tax treatment** — No tax lot accounting, wash-sale rules, or capital gains
   management. After-tax returns for a taxable account would be materially lower
   given the ~893% annualised turnover.

9. **Regime sensitivity** — The model's momentum-based ranking performs well in
   trending markets and underperforms in rapid mean-reverting regimes (e.g.,
   the V-shaped 2020 COVID recovery).

10. **Parameter sensitivity** — The wM=0.65 configuration was selected based on
    walk-forward evidence but the surface around it is relatively flat (see
    chart 15 — Selection Sensitivity). Small parameter perturbations do not
    materially change results, but the optimal configuration may shift in future
    regimes.

11. **Correlation signal lag** — The cross-asset correlation score uses a
    trailing window. During fast-moving crises, correlations spike (everything
    sells off together) before the signal can respond, limiting the
    diversification benefit exactly when it is most needed.

12. **Single-asset selection per sleeve** — Top-N selection means the portfolio
    holds at most 8 ETFs simultaneously (6 main + 2 hedge). Concentration risk
    is higher than in broad index strategies.

### Session-Discovered Limitations

13. **Lag-1 ACF artefact (−0.187)** — The statistically significant negative
    first-order autocorrelation in monthly returns is a mechanical consequence
    of the rebalancing rule, not a genuine predictive signal. The portfolio
    systematically sells recent outperformers and buys recent underperformers
    each month (within the selected set), creating short-horizon mean reversion
    in the realised return series. This is not an exploitable inefficiency.

14. **Walk-forward Fold 3 underperformance (2017–2018 test window)** — The
    walk-forward winner for Fold 3 underperforms the baseline on the test window
    (SR = 0.287 vs 0.460 for baseline). The 2017–2018 period saw unusually low
    volatility (the XIV/VIX shock in Feb 2018), which may have punished the
    volatility-sensitive ranking. This is a real regime risk.

15. **Holdout negative beta during April 2025 tariff shock (β = −0.355)** —
    The 28-month holdout period (2024-01 to 2026-04) includes a rapid
    equity-market selloff driven by US trade-policy uncertainty. AMAAM rotated
    defensively (TLT, SHY, BIL) during this period, producing a large negative
    rolling beta vs SPY. This is consistent with the model's design intent but
    inflates the holdout Sharpe (1.103) relative to a more benign environment.
    Forward-looking investors should not extrapolate the holdout SR.

16. **Risk-free rate misapplication in diagnostic scripts** — Early diagnostic
    runs inadvertently used rf = 0 (instead of rf = 2%) in standalone Sharpe
    calculations. All numbers published in this README and in the charts use
    the correct rf = 2%/yr; the discrepancy only affected transient console
    output during development.

---

## References

1. Giordano, N. (2018). *Ranked Asset Allocation Model*. Charles H. Dow Award,
   Market Technicians Association.

2. Giordano, N. (2021). *Antifragile Asset Allocation Model*. 1st Place,
   SIAT Technical Analyst Award.

3. Keller, W. J., & van Putten, H. (2012). *Generalized Momentum and
   Flexible Asset Allocation*. SSRN Working Paper 2193735.

4. Keller, W. J., & Butler, A. (2014). *Momentum and Markowitz: A Golden
   Combination*. SSRN Working Paper 2404203.

5. DeMiguel, V., Garlappi, L., & Uppal, R. (2009). Optimal versus naive
   diversification: How inefficient is the 1/N portfolio strategy?
   *Review of Financial Studies*, 22(5), 1915–1953.

6. Dichtl, H., Drobetz, W., Lohre, H., Rother, C., & Vosskamp, P. (2021).
   Optimal Timing and Tilting of Equity Factors. *Financial Analysts Journal*,
   77(2), 81–99.

7. Fama, E. F., & MacBeth, J. D. (1973). Risk, Return, and Equilibrium:
   Empirical Tests. *Journal of Political Economy*, 81(3), 607–636.

8. Bollerslev, T. (1986). Generalized Autoregressive Conditional
   Heteroskedasticity. *Journal of Econometrics*, 31(3), 307–327.

9. Zangari, P. (1996). A VaR Methodology for Portfolios That Include Options.
   *RiskMetrics Monitor*, Q1, 4–12.

10. Sortino, F. A., & Price, L. N. (1994). Performance Measurement in a
    Downside Risk Framework. *Journal of Investing*, 3(3), 59–64.

11. Yang, D., & Zhang, Q. (2000). Drift-Independent Volatility Estimation
    Based on High, Low, Open, and Close Prices. *Journal of Business*,
    73(3), 477–491.

---

## Project Layout

```
amaam/
├── config/
│   ├── default_config.py       # ModelConfig dataclass — single source of truth
│   └── etf_universe.py         # ETF and benchmark metadata
├── data/
│   ├── raw/                    # Downloaded OHLCV (git-ignored)
│   └── processed/              # Validated, cleaned data (git-ignored)
├── reports/
│   └── figures/                # Generated PNG and HTML charts
├── results/                    # Backtest output files (git-ignored)
├── scripts/
│   ├── download_data.py        # Fetch and cache raw price data
│   ├── run_backtest.py         # Run full backtest, print metrics
│   ├── generate_reports.py     # Generate all 32 PNG + 23 HTML charts
│   ├── walk_forward.py         # Standalone walk-forward validation
│   └── run_live_signal.py      # Live signal generation (Phase 9 stub)
├── src/
│   ├── backtest/
│   │   ├── engine.py           # Backtest loop and position management
│   │   └── metrics.py          # Performance metric computation
│   ├── data/
│   │   ├── downloader.py       # yfinance wrapper
│   │   ├── loader.py           # Load and merge processed data
│   │   ├── proxy.py            # Pre-inception proxy returns
│   │   └── validator.py        # Data quality checks (Spec §4.3)
│   ├── factors/
│   │   ├── momentum.py         # Momentum signal (blend + single lookbacks)
│   │   ├── volatility.py       # Yang-Zhang volatility estimator
│   │   ├── correlation.py      # Cross-asset correlation scoring
│   │   └── trend.py            # Trend filter ensemble
│   ├── portfolio/
│   │   └── constructor.py      # Sleeve selection and weight assignment
│   └── visualization/
│       ├── matplotlib_charts.py # Static PNG chart library
│       └── plotly_charts.py     # Interactive HTML chart library
├── tests/                      # pytest test suite
├── AMAAM_SPECIFICATION.md      # Authoritative model specification
├── requirements.txt
└── README.md
```

---

*Built with Python 3.13. All backtest results are hypothetical; past
performance does not guarantee future results.*
