# AMAAM: Adaptive Multi-Asset Allocation Model
## Complete Implementation Specification

---

## 1. PROJECT OVERVIEW

### 1.1 Background

This project implements a novel quantitative asset allocation model called the Adaptive Multi-Asset Allocation Model (AMAAM). The model synthesizes and extends two published papers by Gioele Giordano:

1. **"Ranked Asset Allocation Model" (RAAM)** — 2018 Charles H. Dow Award winner. Applies a multi-factor ranking algorithm to the 7Twelve Portfolio (12 ETFs across 7 asset classes) to actively select and weight assets monthly.

2. **"Antifragile Asset Allocation Model" (AAAM)** — 1st Place, SIAT Technical Analyst Award. Uses a two-sleeve architecture: a Sector Rotation Model (11 S&P 500 sector ETFs) for offense, and a Black Swan Hedging Model (7 defensive ETFs) for tail risk protection. Weight flows from the offensive sleeve to the defensive sleeve when momentum turns negative.

Both models share an identical ranking engine (the TRank formula) derived from Keller and van Putten's Flexible Asset Allocation (FAA) framework, which ranks assets on four factors: Momentum, Volatility, Correlation, and Trend.

### 1.2 The AMAAM Innovation

The AMAAM combines the two papers' architectures:

- Replaces the AAAM's sector-only offensive sleeve with a diversified universe that includes both S&P 500 sector ETFs and multi-asset class ETFs (mid-cap, small-cap, international developed, international emerging, real estate, commodities) from the RAAM's 7Twelve base.
- Retains the AAAM's two-sleeve architecture with a modified Black Swan Hedging Model.
- Removes bond and cash ETFs from the offensive sleeve to avoid overlap with the hedging sleeve.
- Replaces single-currency hedging ETFs (FXY, FXF) with a dollar index ETF (UUP) for more robust currency hedging.

The theoretical motivation: the RAAM's diversified asset class exposure provides a first line of defense during the lag between market deterioration and the momentum filter triggering the shift to the hedging sleeve. The AAAM's sector-only sleeve has no such buffer because US sector correlations spike during equity selloffs.

### 1.3 Project Goals

1. **GitHub Portfolio Project**: A professional, well-documented codebase demonstrating quantitative research methodology, suitable for review by quant recruiters.
2. **Live Trading Tool**: A production-ready signal generator that outputs monthly allocation decisions for a ~$30k personal portfolio.

### 1.4 Key References

- Keller, W.J. and van Putten, H.S. (2012). "Generalized Momentum and Flexible Asset Allocation (FAA): An Heuristic Approach." SSRN Working Paper.
- Keller, W.J. and Butler, A. (2014). "A Century of Generalized Momentum; From Flexible Asset Allocations (FAA) to Elastic Asset Allocation (EAA)." SSRN Working Paper.
- DeMiguel, V., Garlappi, L., and Uppal, R. (2009). "Optimal Versus Naive Diversification: How Inefficient is the 1/N Portfolio Strategy?" The Review of Financial Studies, 22(5), 1915-1953.
- Dichtl, H., Drobetz, W., Lohre, H., Rother, C., and Vosskamp, P. (2021). "How to build a factor portfolio: Does the allocation strategy matter?" European Financial Management, 27(1), 20-58.
- Fama, E.F. and MacBeth, J.D. (1973). "Risk, Return, and Equilibrium: Empirical Tests." Journal of Political Economy, 81(3), 607-636.
- Bollerslev, T. (1986). "Generalized Autoregressive Conditional Heteroskedasticity." Journal of Econometrics, 31, 307-327.
- Zangari, P. (1996). "RiskMetrics Technical Document." J.P. Morgan.

---

## 2. MODEL ARCHITECTURE

### 2.1 Two-Sleeve Design

The AMAAM consists of two independent sleeves, each managed by the same TRank ranking engine:

**Multi-Asset Rotation Model (Main Sleeve)**
- Universe: 16 ETFs (see Section 2.2)
- Selection: Top 6 by TRank each month
- Momentum filter: Each selected ETF must have positive 4-month absolute momentum
- If an ETF fails the momentum filter, its weight (1/6 ≈ 16.67%) is redirected to the hedging sleeve
- If ALL 6 selected ETFs have negative momentum, 100% of the portfolio goes to the hedging sleeve

**Black Swan Hedging Model (Hedging Sleeve)**
- Universe: 6 ETFs (see Section 2.3)
- Selection: Top 2 by TRank
- Momentum filter: Each selected ETF must have positive 4-month absolute momentum
- If a hedging ETF fails the momentum filter, its weight is replaced with SHY (cash proxy)
- If ALL hedging ETFs have negative momentum, 100% of the hedging allocation goes to SHY

### 2.2 Main Sleeve ETF Universe (16 assets)

| # | Ticker | Name | Asset Class |
|---|--------|------|-------------|
| 1 | IJH | iShares Core S&P Mid-Cap ETF | US Mid-Cap Equity |
| 2 | IJR | iShares Core S&P Small-Cap ETF | US Small-Cap Equity |
| 3 | EFA | iShares MSCI EAFE ETF | International Developed Equity |
| 4 | EEM | iShares MSCI Emerging Markets ETF | International Emerging Equity |
| 5 | RWR | SPDR Dow Jones REIT ETF | Real Estate |
| 6 | DBC | Invesco DB Commodity Tracking ETF | Commodities |
| 7 | XLY | SPDR Consumer Discretionary ETF | US Sector |
| 8 | XLV | SPDR Health Care ETF | US Sector |
| 9 | XLU | SPDR Utilities ETF | US Sector |
| 10 | XLP | SPDR Consumer Staples ETF | US Sector |
| 11 | XLK | SPDR Technology ETF | US Sector |
| 12 | XLI | SPDR Industrial ETF | US Sector |
| 13 | XLF | SPDR Financial ETF | US Sector |
| 14 | XLE | SPDR Energy ETF | US Sector |
| 15 | XLB | SPDR Materials ETF | US Sector |
| 16 | VOX | Vanguard Communication Services ETF | US Sector |

**Design rationale for exclusions:**
- VV (Vanguard Large Cap) excluded because the 10 sector ETFs collectively reconstruct S&P 500 large-cap exposure, creating redundancy.
- VAW (Vanguard Materials) excluded in favor of XLB (SPDR Materials) to keep the sector family consistent.
- AGG, TIP, IGOV, SHY (bond/cash ETFs from the original RAAM) excluded to prevent overlap with the hedging sleeve.

### 2.3 Hedging Sleeve ETF Universe (6 assets)

| # | Ticker | Name | Hedging Mechanism |
|---|--------|------|-------------------|
| 1 | GLD | SPDR Gold Shares ETF | Real asset / inflation hedge |
| 2 | TLT | iShares 20+ Year Treasury Bond ETF | Long duration / rate decline |
| 3 | IEF | iShares 7-10 Year Treasury Bond ETF | Moderate duration / rate decline |
| 4 | SH | ProShares Short S&P500 ETF | Direct inverse equity exposure |
| 5 | UUP | Invesco DB US Dollar Index Bullish Fund | Dollar strength during risk-off |
| 6 | SHY | iShares 1-3 Year Treasury Bond ETF | Cash proxy / capital preservation |

**Design rationale:**
- FXY (Japanese Yen) and FXF (Swiss Franc) from the original AAAM removed due to weakened safe-haven characteristics since 2012 (BOJ/SNB policy distortions).
- UUP added as a more diversified currency hedge (basket of 6 currencies vs single-currency bet). Dollar has historically strengthened during global risk-off events.
- SH retained despite daily rebalancing decay because the momentum filter limits holding duration, and no other instrument provides direct inverse equity exposure for the full backtest window.
- SHY serves dual role: rankable hedging asset AND cash substitute when other hedging ETFs fail the momentum filter.

### 2.4 Benchmark Universe

| Ticker(s) | Benchmark | Description |
|-----------|-----------|-------------|
| SPY | S&P 500 Buy-and-Hold | Equity market benchmark |
| SPY + AGG | 60/40 Portfolio | 60% SPY / 40% AGG, rebalanced monthly |
| 12 ETFs | Passive 7Twelve | Equal-weighted 7Twelve portfolio, rebalanced monthly |

---

## 3. RANKING ENGINE: THE TRANK FORMULA

### 3.1 Formula

```
TRank = (wM * Rank(M) + wV * Rank(V) + wC * Rank(C) + wT * T) + M/n
```

Where:
- `Rank(M)` = ordinal rank (1 to N) of the asset based on Absolute Momentum (ascending: higher momentum = higher rank = better)
- `Rank(V)` = ordinal rank (1 to N) of the asset based on Volatility Model (descending: lower volatility = higher rank = better)
- `Rank(C)` = ordinal rank (1 to N) of the asset based on Average Relative Correlation (descending: lower correlation = higher rank = better)
- `T` = ATR Trend/Breakout System value (+2 or -2), enters as raw value, NOT ranked
- `M` = raw Absolute Momentum value for the asset
- `n` = number of assets in the sleeve (16 for main, 6 for hedging)
- `M/n` is a tiebreaker term to prevent equal TRank values
- `wM, wV, wC, wT` = factor weights (see Section 3.6)

### 3.2 Factor 1: Absolute Momentum (M)

**Purpose**: Determine asset profitability / trend strength.

**Calculation**: 4-month Rate of Change (ROC) on daily closing prices.

```
M = (Price_today / Price_{today - 84 trading days}) - 1
```

- Lookback: 84 trading days (approximately 4 calendar months)
- Computed daily, but only the month-end value is used for ranking
- The same M value serves dual purpose: (1) input to Rank(M) for TRank, and (2) the momentum filter that determines whether a selected asset stays in the portfolio or gets replaced by cash/hedging

**Ranking**: Assets ranked 1 to N in ascending order. Highest momentum = rank N = best.

### 3.3 Factor 2: Volatility Model (V)

**Purpose**: Determine asset risk through realized volatility estimation.

**Calculation**: Yang-Zhang (2000) variance estimator. Yang-Zhang is an
OHLC-based estimator that is simultaneously drift-independent, unbiased,
and minimum-variance. It achieves approximately 8× the statistical
efficiency of a standard close-to-close estimator, requiring a shorter
lookback window for the same precision.

```
σ²_YZ = σ²_overnight + k · σ²_open-to-close + (1 − k) · σ²_RS
```

Where:
- `σ²_overnight` = rolling variance of log(Open_t / Close_{t-1})
- `σ²_open-to-close` = rolling variance of log(Close_t / Open_t)
- `σ²_RS` = Rogers-Satchell (1991) term: mean of (H−C)(H−O) + (L−C)(L−O)
  in log-price space; captures intra-day drift
- `k = 0.34 / (1.34 + (n+1)/(n−1))` — Chou-Wang optimal mixing coefficient
- `n = 84` trading days (matched to momentum_lookback; see `yang_zhang_window`
  in ModelConfig)

All three component variances use the same rolling window `n`.
Convert the final variance estimate to annualized volatility:

```
V = sqrt(σ²_YZ.clip(lower=0) * 252)
```

**Legacy**: An EWMA estimator (`compute_ewma_variance`, `compute_volatility_model`)
is retained in `volatility.py` for reference and backward compatibility but
is no longer used in the main execution path.

**Ranking**: Assets ranked 1 to N in descending order. Lowest volatility = rank N = best.

### 3.4 Factor 3: Average Relative Correlations (C)

**Purpose**: Achieve portfolio diversification by favoring assets with low co-movement.

**Calculation**: 4-month (84 trading day) average pairwise Pearson correlation of daily returns.

For each asset `i` in the sleeve:
1. Compute the correlation matrix of daily returns over the trailing 84 trading days for ALL assets in the sleeve.
2. Asset `i`'s correlation score `C_i` = average of all pairwise correlations between asset `i` and every other asset in the sleeve.

```
C_i = (1 / (N-1)) * Σ_{j≠i} corr(r_i, r_j)
```

Where `r_i` and `r_j` are vectors of daily returns over the lookback window.

**Ranking**: Assets ranked 1 to N in descending order. Lowest average correlation = rank N = best.

**Important**: Correlations are computed within each sleeve independently. The main sleeve's C values reflect correlations among the 16 main sleeve ETFs. The hedging sleeve's C values reflect correlations among the 6 hedging ETFs. There is no cross-sleeve correlation computation.

### 3.5 Factor 4: Keltner Channel Trend Direction (T)

**Purpose**: Determine asset trend direction using an asymmetric Keltner
Channel that adjusts dynamically to volatility.

**Band formulas** (daily OHLC data):

```
ATR_t  = (1/42) × Σ_{i=0}^{41} TR_{t-i}       (SMA of True Range, NOT Wilder)
UB_t   = EMA(Close, 63)_t  + 1.0 × ATR_t       (upper / bullish band)
LB_t   = EMA(Close, 105)_t − 1.0 × ATR_t       (lower / bearish band)
```

Where:
- `ATR` = 42-period **simple** moving average of True Range.
  `TR_t = max(H_t − L_t, |H_t − C_{t-1}|, |L_t − C_{t-1}|)`
- `EMA(Close, 63)` = exponential moving average of closing price with span 63
  (α = 2/64); the faster span confirms uptrends more readily.
- `EMA(Close, 105)` = EMA with span 105 (α = 2/106); the slower span requires
  a sustained price decline before the downtrend signal fires.
- Multiplier k = 1.0 (Keltner multiplier per Variant A).

**Asymmetric design rationale**: The upper band uses a faster EMA (63 days ≈
3 months) so the model captures emerging uptrends without excessive lag.  The
lower band uses a slower EMA (105 days ≈ 5 months) so a brief correction does
not immediately flip the signal to −2; a sustained downtrend is required.
Both bands widen when ATR rises, demanding a more decisive move to change state
in turbulent markets.

**Signal logic** (persistent carry, evaluated at each daily bar):

```
T_t = +2   if High_t  > UB_t    (uptrend confirmed)
T_t = −2   if Low_t   < LB_t    (downtrend confirmed)
T_t = T_{t-1}  otherwise        (no change — state persists)
```

If both conditions fire simultaneously (extremely rare), the uptrend rule
takes precedence.  The initial state before the first valid band date is −2
(conservative: assume no confirmed trend).

**Key behavioural property**: Because the lower band sits *below* the EMA
centre line (LB = EMA − ATR), a full ATR of distance below the 105-day moving
average is required to trigger T = −2.  Conversely, a full ATR above the
63-day EMA triggers T = +2.  Once flipped, the signal persists until the
opposite band is breached — T = +2 can remain in effect for many consecutive
months if prices hold above the lower band.

**Important**: T = −2 simply worsens the asset's TRank, making it less likely
to be selected.  The entire model is long-only; T = −2 is NOT a short-sell
instruction.

**Usage in TRank**: T enters the formula as a raw ±2 value (not ranked) and
is **added** (not subtracted).  Because `select_top_n` picks the *highest*
TRank and rank N = best throughout:
- T = +2 (uptrend): adds 2·wT → TRank rises → asset more likely selected. ✓
- T = −2 (downtrend): subtracts 2·wT → TRank falls → asset less likely selected. ✓

See Section 3.1 for the full formula.

### 3.6 Factor Weights

**Calibrated weights**:
- wM = 0.65 (Momentum — primary return signal, receives dominant weight)
- wV = 0.25 (Volatility — secondary risk-control signal)
- wC = 0.10 (Correlation — tiebreaker role; most regime-dependent factor)
- wT = 1.0 (Scale factor for trend; T operates differently from ranked factors)

**Rationale**: Keller and van Putten (2012) set wR=1, wV=0.5, wC=0.5 in
the original FAA paper, explicitly stating these values were chosen
"arbitrarily." Their intuition was that momentum should receive
approximately double the weight of either risk factor. Normalizing to sum
to 1.0 yields 0.50/0.25/0.25 as a starting point.

A structural audit of the backtest revealed that at wV=0.25, the
volatility factor chronically over-rewarded low-vol defensive sectors
(XLP, XLU, XLV), which collectively occupied the top-6 sleeve 143% of
months at near-cash returns — creating a portfolio drag. The correlation
factor is the most regime-dependent and least theoretically grounded of
the three ranked signals; at wC=0.25 it amplified defensive crowding
rather than improving diversification.

Raising wM to 0.65 and reducing wC to 0.10 (with wV retained at 0.25)
reduced combined defensive-sector selection by 21.5 percentage points,
improved full-period CAGR from 8.53% to 9.47%, OOS Sharpe from 0.537 to
0.593, and OOS Calmar from 0.566 to 0.664. Walk-forward validation across
six independent 2-year windows confirmed directional improvement in 4/6
periods. The weights reflect a design decision consistent with the
strategy's momentum-first purpose.

**Sensitivity analysis**: Factor weights were varied systematically
(scripts/wv_sweep.py, scripts/wc_sweep_wv15.py, scripts/wm_wc_grid_sweep.py,
scripts/walk_forward.py). Walk-forward, sensitivity neighbourhood, and
Deflated Sharpe Ratio tests were applied to candidate configurations before
the final weights were adopted.

### 3.7 Asset Selection and Allocation Logic

**Monthly process (executed on last trading day of each month):**

1. Compute M, V, C, T for all 16 main sleeve ETFs and all 6 hedging sleeve ETFs using daily data through the current date.
2. Compute TRank for all 16 main sleeve ETFs. Select top 6 by TRank.
3. For each of the 6 selected main sleeve ETFs:
   - If M > 0: assign 1/6 weight (≈16.67%) in the portfolio
   - If M ≤ 0: redirect 1/6 weight to the hedging sleeve
4. Compute TRank for all 6 hedging sleeve ETFs. Select top 2 by TRank.
5. Distribute the total redirected weight equally among the top 2 hedging ETFs that have M > 0.
6. Any hedging ETF with M ≤ 0 has its share replaced by SHY.
7. The resulting allocation is implemented at the NEXT trading day's close (one-day implementation lag).
8. Hold the allocation for the entire following month until the next rebalancing.

**Edge cases:**
- If all 6 main sleeve ETFs have M ≤ 0: 100% goes to hedging sleeve
- If all hedging ETFs have M ≤ 0: 100% goes to SHY
- Rank ties: include all tied assets (following Keller's convention). If more than 6 assets tie for the 6th rank, include all of them with proportionally reduced weights.

### 3.8 Weighting Schemes

Two weighting schemes will be implemented and compared:

**Equal Weight (base case)**: Each selected asset receives 1/N_selected weight within its sleeve.

**Inverse-Volatility Weight (alternative)**: Each selected asset's weight is proportional to the inverse of its current Volatility Model value.

```
w_i = (1/V_i) / Σ_j(1/V_j)
```

Where the sum is over all selected assets in the sleeve (after momentum filter).

---

## 4. DATA

### 4.1 Sources

**Historical backtest data**: Downloaded via `yfinance` Python library. All data saved locally as CSV files for reproducibility.

**Live signal data**: Pulled via Schwab API (`schwab-py` library) for production use. Requires active Schwab brokerage account and API credentials.

### 4.2 Data Requirements

- **Frequency**: Daily
- **Fields**: Open, High, Low, Close (adjusted for splits and dividends), Volume
- **Period**: From earliest common inception date through present
- **Binding constraint**: UUP (Invesco DB US Dollar Index Bullish Fund) launched February 2007. With a 4-month (84 trading day) lookback for initialization, the first usable signal date is approximately June 2007. Backtest live trading begins August 2007 to allow additional buffer.

### 4.3 Data Validation

During Phase 1, the following checks must be performed on all downloaded data:

1. No missing trading days (compare against NYSE trading calendar)
2. No NaN values in OHLC fields
3. OHLC consistency: High ≥ max(Open, Close) and Low ≤ min(Open, Close) for every row
4. No negative or zero volume on trading days
5. No duplicate dates
6. Adjusted close accounts for splits and dividends (verify a known split date)
7. Price continuity: flag any single-day returns exceeding ±25% for manual review
8. All ticker time series aligned to common trading calendar
9. VOX reconstitution in September 2018 noted in documentation (economic exposure changed)

---

## 5. BACKTESTING

### 5.1 Backtest Period

- **Start**: August 2007 (first month with valid signals for all ETFs)
- **End**: April 2026 (present)
- **Total**: approximately 225 months (18.75 years)

### 5.2 Train/Test Split

- **Development period**: August 2007 — December 2017 (125 months, 10.4 years)
- **Holdout period**: January 2018 — April 2026 (100 months, 8.3 years)
- All design decisions are finalized using only the development period. The holdout is run ONCE at the end.

### 5.3 Execution Assumptions

- Signals computed on the last trading day of month M using closing prices.
- Allocation applied on the first trading day of month M+1 using closing prices (one-day implementation lag).
- Full rebalancing assumed (all positions can be liquidated and re-established in a single session).
- No partial fills, no slippage beyond the transaction cost assumption.
- No market impact (appropriate for ~$30k portfolio size; documented as a limitation for larger portfolios).
- No short selling. All positions are long-only.
- No leverage.

### 5.4 Rebalancing Frequency

Two frequencies will be tested:

- **Monthly** (base case): Rebalance on the last trading day of each month. 12 events per year.
- **Bi-weekly** (alternative): Rebalance every 2 weeks. Approximately 26 events per year. Tests whether faster signal response improves risk-adjusted returns enough to offset higher transaction costs.

The backtesting engine should accept rebalancing frequency as a configurable parameter.

### 5.5 Transaction Costs

Three scenarios tested:

- **0 bps**: Gross returns (for comparison with original papers)
- **10 bps round trip**: Base case for liquid ETFs (consistent with Frazzini, Israel, and Moskowitz 2015 findings for institutional-size trades in liquid instruments)
- **15 bps round trip**: Stress test accounting for wider spreads on less liquid ETFs (DBC, UUP)

Transaction cost is applied to the dollar value of each trade (buy or sell). Turnover is calculated as the sum of absolute changes in portfolio weights from one rebalancing to the next.

```
Turnover_t = Σ_i |w_{i,t} - w_{i,t-1}|
Cost_t = Turnover_t * cost_per_trade / 2
```

(Divided by 2 because turnover double-counts: selling one asset and buying another each count toward the sum.)

### 5.6 Performance Metrics

All metrics computed for the AMAAM and each benchmark, across all transaction cost scenarios:

| Metric | Description |
|--------|-------------|
| Annualized Return | Geometric mean annual return |
| Annualized Volatility | Standard deviation of returns, annualized (√252 for daily, √12 for monthly) |
| Sharpe Ratio | (Annualized Return - Risk Free Rate) / Annualized Volatility. Use 3-month T-bill rate or fixed 2% as risk-free. |
| Calmar Ratio | Annualized Return / abs(Max Drawdown) |
| Max Drawdown | Largest peak-to-trough decline |
| Max Drawdown Duration | Longest time (in months) to recover from a drawdown |
| Best Month | Highest single-month return |
| Worst Month | Lowest single-month return |
| Best Year | Highest calendar-year return |
| Worst Year | Lowest calendar-year return |
| % Positive Months | Percentage of months with positive returns |
| % Positive Years | Percentage of calendar years with positive returns |
| Average Monthly Turnover | Mean of monthly turnover values |
| Annual Turnover | Sum of monthly turnovers, averaged across years |

---

## 6. SENSITIVITY ANALYSIS

### 6.1 Factor Weight Sensitivity

Vary wM from 0.20 to 0.60 in increments of 0.05. For each wM value, set wV = wC = (1 - wM) / 2 (equal split of remaining weight between volatility and correlation). Record Sharpe, max drawdown, annualized return, and annualized volatility for each combination. Produce heatmap visualization.

Also test: wM = wV = wC = 0.333 (equal weights, excluding T).

### 6.2 Selection Count Sensitivity

Test top 4, top 5, top 6, and top 7 from the main sleeve, holding all other parameters at base case values. Report full performance metrics for each.

### 6.3 Weighting Scheme Comparison

Run full backtest with equal weighting and inverse-volatility weighting. Report side-by-side performance metrics.

### 6.4 Rebalancing Frequency Comparison

Run full backtest with monthly and bi-weekly rebalancing. Report side-by-side performance metrics across all transaction cost scenarios.

---

## 7. REPORTING AND VISUALIZATION

### 7.1 Charts (Matplotlib for GitHub, Plotly for personal use)

All charts should be produced in both formats. Matplotlib PNGs saved to a `reports/figures/` directory for the GitHub README. Plotly HTML files saved to a `reports/interactive/` directory for personal analysis.

**Performance Overview:**
1. Cumulative equity curve (log scale) for AMAAM and all benchmarks on same chart
2. Drawdown chart (peak-to-trough decline over time) for AMAAM and benchmarks
3. Monthly return heatmap (year × month grid with color coding)
4. Annual return bar chart comparing AMAAM to each benchmark
5. Rolling 12-month return chart

**Risk Analysis:**
6. Rolling 12-month Sharpe ratio over time
7. Rolling 12-month volatility over time
8. Rolling max drawdown (trailing 12-month worst drawdown)
9. Distribution histogram of monthly returns with normal distribution overlay

**Allocation Analysis:**
10. Stacked area chart: main sleeve holdings over time
11. Stacked area chart: hedging sleeve holdings over time
12. Line chart: percentage of portfolio allocated to hedging sleeve over time
13. Turnover analysis: monthly turnover percentage over time

**Factor Analysis:**
14. Time series of TRank factor weights (if walk-forward optimization implemented)
15. Return decomposition: contribution from main sleeve vs hedging sleeve
16. Correlation matrix of the 16 main sleeve ETFs (full period)

**Regime Analysis:**
17. Performance statistics table broken out by: 2008 GFC, 2011 Euro crisis, 2015-16 commodity crash, 2018 vol spike, 2020 COVID, 2022 rate shock

**Sensitivity Analysis:**
18. Heatmap: Sharpe ratio across factor weight combinations
19. Bar chart: performance metrics across selection count variants
20. Bar chart: equal weight vs inverse-vol weight comparison

**Transaction Cost Impact:**
21. Equity curves overlaid at 0, 10, and 15 bps
22. Table: all summary statistics across cost scenarios

**Out-of-Sample Validation:**
23. Separate equity curves for development and holdout periods
24. Side-by-side statistics tables for each period

### 7.2 Summary Statistics Table

A comprehensive table in the README showing all metrics (Section 5.6) for AMAAM and each benchmark, across all transaction cost scenarios.

---

## 8. REPOSITORY STRUCTURE

```
amaam/
├── README.md                          # Project overview, methodology, key findings
├── requirements.txt                   # Python dependencies
├── config/
│   ├── default_config.py              # Default model configuration (dataclass)
│   └── etf_universe.py                # ETF ticker definitions and metadata
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── downloader.py              # Data acquisition (yfinance + Schwab API)
│   │   ├── validator.py               # Data quality checks
│   │   └── loader.py                  # Load validated data from local CSV
│   ├── factors/
│   │   ├── __init__.py
│   │   ├── momentum.py                # Absolute Momentum (M) calculation
│   │   ├── volatility.py              # EWMA Volatility Model (V) calculation
│   │   ├── correlation.py             # Average Relative Correlation (C) calculation
│   │   └── trend.py                   # ATR Trend/Breakout System (T) calculation
│   ├── ranking/
│   │   ├── __init__.py
│   │   └── trank.py                   # TRank computation and asset ranking
│   ├── portfolio/
│   │   ├── __init__.py
│   │   ├── allocation.py              # Sleeve allocation logic and momentum filter
│   │   └── weighting.py               # Equal weight and inverse-vol weight schemes
│   ├── backtest/
│   │   ├── __init__.py
│   │   ├── engine.py                  # Core backtesting loop
│   │   ├── metrics.py                 # Performance metric calculations
│   │   └── benchmarks.py              # Benchmark portfolio construction
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── sensitivity.py             # Factor weight and parameter sensitivity
│   │   └── regime.py                  # Regime-specific performance analysis
│   ├── visualization/
│   │   ├── __init__.py
│   │   ├── matplotlib_charts.py       # Static charts for GitHub
│   │   └── plotly_charts.py           # Interactive charts for personal use
│   └── live/
│       ├── __init__.py
│       ├── signal_generator.py        # Live monthly signal computation
│       └── allocation_logger.py       # Log each month's decision and factor values
├── tests/
│   ├── __init__.py
│   ├── test_momentum.py
│   ├── test_volatility.py
│   ├── test_correlation.py
│   ├── test_trend.py
│   ├── test_trank.py
│   ├── test_allocation.py
│   ├── test_weighting.py
│   ├── test_backtest.py
│   └── test_metrics.py
├── data/
│   ├── raw/                           # Downloaded CSV files (gitignored or LFS)
│   └── processed/                     # Validated, aligned data ready for backtest
├── reports/
│   ├── figures/                       # Matplotlib PNG outputs
│   ├── interactive/                   # Plotly HTML outputs
│   └── summary/                       # PDF/markdown research writeup
├── notebooks/                         # Optional Jupyter notebooks for exploration
│   └── exploration.ipynb
└── scripts/
    ├── download_data.py               # One-time data download script
    ├── run_backtest.py                # Main backtest execution script
    ├── run_sensitivity.py             # Sensitivity analysis execution
    ├── run_live_signal.py             # Monthly signal generation
    └── generate_reports.py            # Generate all charts and tables
```

---

## 9. MODULE SPECIFICATIONS

### 9.1 config/default_config.py

Python dataclass containing all configurable model parameters. Every adjustable value in the model should be defined here and nowhere else. Code should never contain hard-coded parameter values.

```python
from dataclasses import dataclass, field
from typing import List, Tuple

@dataclass
class ModelConfig:
    # Factor lookback periods (trading days)
    momentum_lookback: int = 84  # ~4 months
    volatility_lambda: float = 0.94  # EWMA decay factor
    volatility_smoothing: int = 10  # SMA smoothing window for vol
    correlation_lookback: int = 84  # ~4 months
    atr_period: int = 42
    atr_upper_lookback: int = 63  # Highest close lookback
    atr_lower_lookback: int = 105  # Highest low lookback

    # Factor weights (see Section 3.6 for calibration rationale)
    weight_momentum: float = 0.65
    weight_volatility: float = 0.25
    weight_correlation: float = 0.10
    weight_trend: float = 1.0  # Scale factor for T in TRank

    # Selection parameters
    main_sleeve_top_n: int = 6
    hedging_sleeve_top_n: int = 2

    # Weighting scheme: "equal" or "inverse_volatility"
    weighting_scheme: str = "equal"

    # Rebalancing
    rebalancing_frequency: str = "monthly"  # "monthly" or "biweekly"

    # Transaction costs (round trip, in decimal: 0.0010 = 10 bps)
    transaction_cost: float = 0.0010

    # Backtest period
    backtest_start: str = "2007-08-01"
    backtest_end: str = "2026-04-10"
    holdout_start: str = "2018-01-01"

    # Yang-Zhang volatility window (trading days); matched to momentum_lookback
    yang_zhang_window: int = 84
```

### 9.2 config/etf_universe.py

Definitions of the ETF universes for each sleeve, plus benchmarks. Includes ticker, full name, asset class, and inception date for reference.

### 9.3 src/data/downloader.py

**Functions:**
- `download_historical_data(tickers, start_date, end_date, source="yfinance") -> Dict[str, pd.DataFrame]`: Downloads OHLC data for all tickers. Returns dict mapping ticker to DataFrame.
- `download_schwab_data(tickers, start_date, end_date, credentials) -> Dict[str, pd.DataFrame]`: Downloads from Schwab API for live use. Same return format.
- `save_raw_data(data_dict, output_dir)`: Saves each ticker's DataFrame as CSV.
- `load_raw_data(data_dir) -> Dict[str, pd.DataFrame]`: Loads previously saved CSVs.

### 9.4 src/data/validator.py

**Functions:**
- `validate_ohlc(df, ticker) -> List[str]`: Runs all data quality checks from Section 4.3. Returns list of issues found (empty list = clean).
- `validate_universe(data_dict) -> Dict[str, List[str]]`: Validates all tickers. Returns dict of issues per ticker.
- `align_trading_calendar(data_dict) -> Dict[str, pd.DataFrame]`: Aligns all series to common trading dates, forward-fills up to 3 days for minor gaps, flags larger gaps.

### 9.5 src/data/loader.py

**Functions:**
- `load_validated_data(data_dir) -> Dict[str, pd.DataFrame]`: Loads validated, aligned data. Single entry point for all downstream code.
- `get_returns(data_dict, return_type="log") -> Dict[str, pd.Series]`: Computes daily returns (log or simple) for all tickers.
- `get_monthly_dates(data_dict) -> List[pd.Timestamp]`: Returns list of last trading day of each month in the dataset.

### 9.6 src/factors/momentum.py

**Functions:**
- `compute_absolute_momentum(prices, lookback) -> pd.Series`: Takes a Series of daily closing prices and lookback in trading days. Returns Series of momentum values (ROC).
- `compute_momentum_all_assets(data_dict, lookback) -> pd.DataFrame`: Computes momentum for all assets. Returns DataFrame with dates as index, tickers as columns.

### 9.7 src/factors/volatility.py

**Primary functions (Yang-Zhang estimator):**
- `compute_yang_zhang_vol(ohlc, window) -> pd.Series`: Takes a per-ticker
  OHLC DataFrame and rolling window (trading days). Returns annualized
  Yang-Zhang volatility Series.
- `compute_volatility_all_assets(data_dict, config) -> pd.DataFrame`:
  Dispatches to `compute_yang_zhang_vol` for every ticker using
  `config.yang_zhang_window`. Returns a DataFrame of annualized volatility
  with dates as index and tickers as columns.

**Legacy functions (EWMA — retained for reference, not used in main path):**
- `compute_ewma_variance(returns, lambda_param, init_window) -> pd.Series`
- `compute_volatility_model(prices, lambda_param, init_window, smoothing_window) -> pd.Series`

### 9.8 src/factors/correlation.py

**Functions:**
- `compute_average_relative_correlation(returns_dict, tickers, lookback, date) -> Dict[str, float]`: For a given date, computes the trailing correlation matrix and returns each asset's average pairwise correlation.
- `compute_correlation_all_assets(data_dict, tickers, lookback) -> pd.DataFrame`: Computes correlation score for all assets across all dates. Returns DataFrame.

### 9.9 src/factors/trend.py

**Functions:**
- `compute_atr(high, low, close, period) -> pd.Series`: 42-period SMA of True Range.
- `compute_atr_bands(high, low, close, atr_period, upper_ema_span, lower_ema_span) -> Tuple[pd.Series, pd.Series]`: Returns asymmetric Keltner Channel upper and lower bands.
- `compute_trend_signal(high, low, close, atr_period, upper_ema_span, lower_ema_span) -> pd.Series`: Returns Series of +2/−2 persistent trend signals.
- `compute_trend_all_assets(data_dict, config) -> pd.DataFrame`: Computes Keltner Channel trend for all assets.

### 9.10 src/ranking/trank.py

**Functions:**
- `rank_assets(values, ascending=True) -> pd.Series`: Rank a Series of values. Returns ordinal ranks (1 to N).
- `compute_trank(momentum_ranks, volatility_ranks, correlation_ranks, trend_values, raw_momentum, config) -> pd.Series`: Applies the TRank formula. Returns TRank score for each asset.
- `select_top_n(tranks, n) -> List[str]`: Returns the top N tickers by TRank.
- `compute_monthly_rankings(factor_data, config, sleeve_tickers) -> pd.DataFrame`: For each month-end date, computes TRank and selects top N. Returns DataFrame of monthly selections.

### 9.11 src/portfolio/allocation.py

**Functions:**
- `apply_momentum_filter(selected_tickers, momentum_values) -> Tuple[List[str], List[str]]`: Separates selected tickers into those with positive momentum (stay in main sleeve) and those with negative momentum (weight redirected). Returns (active_tickers, redirected_tickers).
- `compute_hedging_allocation(hedging_rankings, momentum_values, redirected_weight, config) -> Dict[str, float]`: Determines hedging sleeve allocation given the total weight redirected from main sleeve.
- `compute_monthly_allocation(main_rankings, hedging_rankings, main_momentum, hedging_momentum, config) -> Dict[str, float]`: Full allocation pipeline for one month. Returns dict of ticker -> weight, summing to 1.0.

### 9.12 src/portfolio/weighting.py

**Functions:**
- `equal_weight(tickers) -> Dict[str, float]`: Returns equal weight for each ticker.
- `inverse_volatility_weight(tickers, volatility_values) -> Dict[str, float]`: Returns inverse-vol weights normalized to sum to 1.0.
- `apply_weighting(tickers, scheme, volatility_values=None) -> Dict[str, float]`: Dispatcher that calls the appropriate weighting function based on the scheme string.

### 9.13 src/backtest/engine.py

**Functions:**
- `run_backtest(data_dict, config) -> BacktestResult`: Main backtest loop. Iterates over monthly (or bi-weekly) rebalancing dates, computes allocation, applies to next period's returns, deducts transaction costs. Returns a BacktestResult object containing the equity curve, monthly returns, allocations over time, turnover, and factor values.
- `BacktestResult` (dataclass): Contains all backtest outputs.

### 9.14 src/backtest/metrics.py

**Functions:**
- `compute_all_metrics(returns, risk_free_rate=0.02) -> Dict[str, float]`: Takes a Series of returns and computes all metrics from Section 5.6.
- `compute_rolling_metrics(returns, window=12) -> pd.DataFrame`: Rolling Sharpe, rolling vol, rolling max drawdown.
- `compute_drawdown_series(returns) -> pd.Series`: Peak-to-trough drawdown at each point in time.

### 9.15 src/backtest/benchmarks.py

**Functions:**
- `compute_spy_benchmark(data_dict, start, end) -> pd.Series`: Buy-and-hold SPY returns.
- `compute_sixty_forty(data_dict, start, end) -> pd.Series`: 60/40 SPY/AGG monthly rebalanced.
- `compute_seven_twelve(data_dict, start, end) -> pd.Series`: Equal-weighted 7Twelve monthly rebalanced.

### 9.16 src/analysis/sensitivity.py

**Functions:**
- `run_weight_sensitivity(data_dict, config, wm_range, step) -> pd.DataFrame`: Runs backtest across a grid of wM values. Returns DataFrame of metrics per weight combination.
- `run_selection_sensitivity(data_dict, config, n_range) -> pd.DataFrame`: Tests different selection counts.
- `run_rebalancing_sensitivity(data_dict, config) -> pd.DataFrame`: Compares monthly vs bi-weekly.

### 9.17 src/analysis/regime.py

**Functions:**
- `define_regimes() -> Dict[str, Tuple[str, str]]`: Returns dict mapping regime name to (start_date, end_date).
- `compute_regime_metrics(returns, regimes) -> pd.DataFrame`: Performance metrics for each regime.

### 9.18 src/visualization/matplotlib_charts.py

One function per chart listed in Section 7.1. Each function takes the relevant data and saves a PNG to the specified output directory. Consistent styling across all charts (font sizes, colors, line widths defined in a style constants section at the top of the file).

### 9.19 src/visualization/plotly_charts.py

Mirror of matplotlib_charts.py but producing interactive Plotly HTML files.

### 9.20 src/live/signal_generator.py

**Functions:**
- `generate_current_signal(config, data_source="schwab", credentials=None) -> Dict[str, float]`: Fetches current data, computes all factors, runs ranking, applies allocation logic, returns current month's target allocation.
- `format_signal_report(allocation, factor_values) -> str`: Formats the allocation and factor values into a readable report.

### 9.21 src/live/allocation_logger.py

**Functions:**
- `log_allocation(allocation, factor_values, date, log_dir) -> None`: Saves the current allocation decision to a JSON log file with timestamp.
- `load_allocation_history(log_dir) -> pd.DataFrame`: Loads all historical allocation logs into a DataFrame.

---

## 10. TESTING STRATEGY

### 10.1 Unit Tests

Each factor module gets its own test file. Tests should cover:

**test_momentum.py:**
- Known input/output: feed a specific price series, verify ROC matches hand calculation
- Edge case: constant prices (momentum = 0)
- Edge case: insufficient data (fewer than lookback days)

**test_volatility.py:**
- Verify EWMA recursion against hand calculation for 5-10 steps
- Verify smoothing produces correct SMA of variance
- Verify annualization factor
- Edge case: zero returns (variance should be near zero)

**test_correlation.py:**
- Verify against manually computed correlation matrix for 3 assets × 10 days
- Verify average pairwise correlation calculation
- Edge case: perfectly correlated assets (C should be 1.0)
- Edge case: perfectly uncorrelated assets (C should be near 0.0)

**test_trend.py:**
- Verify ATR calculation against hand computation
- Verify band calculation
- Verify signal flip: construct a price series that breaks above upper band, confirm T = +2
- Verify signal persistence: no breakout, T should retain previous value

**test_trank.py:**
- Verify ranking direction (higher momentum = higher rank)
- Verify TRank formula produces expected composite score for known inputs
- Verify top-N selection picks the correct assets
- Verify tiebreaker (M/n) resolves ties correctly

**test_allocation.py:**
- Verify momentum filter correctly separates positive/negative momentum assets
- Verify weight redirection: if 2 of 6 main sleeve assets have negative M, verify 2/6 weight goes to hedging sleeve
- Verify edge case: all negative momentum → 100% hedging
- Verify edge case: all hedging negative momentum → 100% SHY
- Verify weights sum to 1.0 in all cases

**test_weighting.py:**
- Verify equal weight produces 1/N for each asset
- Verify inverse-vol weights are proportional to 1/V
- Verify weights sum to 1.0

**test_backtest.py:**
- Run a short backtest (12 months) on synthetic data with known properties
- Verify equity curve matches manual calculation
- Verify transaction cost deduction

**test_metrics.py:**
- Verify Sharpe ratio calculation against known values
- Verify max drawdown calculation against known equity curve
- Verify annualization

### 10.2 Integration Tests

- Run full backtest on the first 24 months of real data. Manually verify allocations for 3-4 months by computing TRank by hand.
- Verify that the live signal generator produces the same allocation as the backtest engine for the most recent month.

---

## 11. LOGGING

Use Python's `logging` module with the following configuration:

- **DEBUG**: Factor values for each asset at each rebalancing date, individual rank values, TRank scores
- **INFO**: Monthly allocation decisions, monthly returns, rebalancing events, factor weight updates
- **WARNING**: Data quality issues, rank ties, edge cases triggered
- **ERROR**: Missing data, computation failures, invalid allocations

Log output should go to both console (INFO level) and a rotating log file (DEBUG level) in a `logs/` directory.

Each log message for a rebalancing event should include:
- Date
- Factor values for all assets
- Rankings for all assets
- TRank scores
- Selected assets
- Momentum filter results
- Final allocation with weights
- Turnover from previous allocation
- Transaction cost incurred

---

## 12. IMPLEMENTATION PHASES

### Phase 1: Data Acquisition and Validation
- Implement downloader.py, validator.py, loader.py
- Download all data, validate, save locally
- Verify inception dates, identify common start date
- **Deliverable**: Clean, validated CSV files for all tickers

### Phase 2: Core Factor Computation
- Implement momentum.py, volatility.py, correlation.py, trend.py
- Unit tests for each factor module
- Visual validation: plot factors for 2-3 ETFs, cross-reference with Giordano's charts for overlapping period
- **Deliverable**: All four factor modules with passing tests

### Phase 3: Ranking System
- Implement trank.py
- Unit tests for ranking and selection
- Validate T factor sign convention
- **Deliverable**: Working TRank computation with passing tests

### Phase 4: Portfolio Construction
- Implement allocation.py, weighting.py
- Unit tests for allocation logic and edge cases
- Manual validation of allocations for critical months (Sep 2008, Mar 2009, Mar 2020, Jan 2022)
- **Deliverable**: Complete allocation pipeline with passing tests

### Phase 5: Backtesting Engine
- Implement engine.py, metrics.py, benchmarks.py
- Integration test on short period
- Run full backtest, verify equity curves are reasonable
- **Deliverable**: Full backtest results for base case

### Phase 6: Sensitivity Analysis
- Implement sensitivity.py
- Run factor weight sensitivity
- Run selection count sensitivity
- Run weighting scheme comparison
- Run rebalancing frequency comparison
- **Deliverable**: Sensitivity analysis results

### Phase 7: Visualization and Reporting
- Implement matplotlib_charts.py and plotly_charts.py
- Implement regime.py
- Generate all charts and tables
- **Deliverable**: Complete visual output

### Phase 8: Out-of-Sample Validation
- Run holdout period backtest
- Compare development vs holdout statistics
- **Deliverable**: Out-of-sample results

### Phase 9: Live Signal Module
- Implement signal_generator.py, allocation_logger.py
- Test with current market data
- Verify output matches most recent backtest month
- **Deliverable**: Working live signal tool

### Phase 10: Documentation and Repository Assembly
- Write README
- Write research summary (5-10 pages)
- Clean and organize all code
- Final review of all docstrings, type hints, and comments
- **Deliverable**: Publication-ready repository

---

## 13. KNOWN LIMITATIONS

Document these in the README:

1. **Single backtest path dependency**: All performance metrics are derived from one historical realization. Rolling window analysis provides sub-period variability but does not substitute for simulation-based confidence intervals.

2. **Monthly rebalancing lag**: The model cannot react to intra-month developments. The 4-month momentum lookback introduces additional delay in detecting regime changes. The hedging sleeve only activates at month-end rebalancing.

3. **Bi-weekly rebalancing partially mitigates the lag but doubles transaction frequency.

4. **Survivorship bias**: All ETFs in the universe exist today. Delisted or merged ETFs that were available during the backtest period are not included.

5. **VOX reconstitution**: The Communication Services sector ETF was fundamentally reconstituted in September 2018. Pre-2018 VOX tracked a different set of companies than post-2018 VOX. The backtest treats the series as continuous.

6. **No market impact modeling**: Appropriate for portfolio sizes under ~$500k. Larger portfolios would need to account for the market impact of rebalancing 8-10 ETF positions simultaneously.

7. **Tax implications not modeled**: Monthly (or bi-weekly) rebalancing in a taxable account generates short-term capital gains. The backtest reports pre-tax returns. For taxable accounts, the strategy is more efficient in a tax-advantaged wrapper (IRA, 401k).

8. **Factor weight selection**: The base case factor weights (50/25/25) are heuristic, following Keller and van Putten (2012). No closed-form derivation exists for optimal weights in this ranking framework. The sensitivity analysis characterizes how performance varies across the weight space.

9. **SH (ProShares Short S&P 500) structural decay**: Daily rebalancing of the inverse ETF causes performance to deviate from the theoretical inverse of the S&P 500 over multi-day holding periods. The momentum filter limits exposure duration, but some decay drag is unavoidable during holding periods.

10. **Correlation factor degradation during crises**: When correlations spike across all assets during market stress, the C factor loses discriminating power. All assets receive similar correlation scores, reducing the factor's contribution to the ranking.

11. **No transaction costs were included in the original Giordano papers**. Our inclusion of 10 and 15 bps scenarios is an improvement but still assumes uniform costs across all ETFs, when in reality spreads vary by instrument and market conditions.

12. **Data source limitations**: yfinance is an unofficial API subject to breaking changes. Historical adjusted close calculations may differ between data providers. Schwab API provides more reliable data for live use.

---

## 14. PYTHON DEPENDENCIES

```
# Core
numpy>=1.24
pandas>=2.0
scipy>=1.10

# Data
yfinance>=0.2.30
schwab-py>=1.0  # Optional, for live signals

# Visualization
matplotlib>=3.7
plotly>=5.15

# Testing
pytest>=7.0
pytest-cov>=4.0

# Utilities
pyyaml>=6.0  # If config files used alongside dataclasses
tqdm>=4.65  # Progress bars for long backtests
```

---

## 15. DOCUMENTATION STANDARDS

- **Docstrings**: NumPy style for all public functions
- **Type hints**: All function signatures must include type hints
- **Comments**: Explain WHY, not WHAT. The code should be readable without comments explaining what it does. Comments should explain design decisions, non-obvious algorithmic choices, and references to the papers.
- **Constants**: No magic numbers in code. All numerical parameters must reference the config dataclass.
- **Logging**: Structured logging at appropriate levels throughout (see Section 11).

---

## 16. STYLE AND CONVENTIONS

- **Naming**: snake_case for functions and variables, PascalCase for classes, UPPER_SNAKE_CASE for constants
- **Line length**: 100 characters max
- **Imports**: stdlib, then third-party, then local, separated by blank lines
- **File length**: No module should exceed ~400 lines. If it does, split it.
- **Error handling**: Use explicit exceptions with informative messages. Never silently swallow errors.
- **Data types**: Use pandas DataFrames and Series as the primary data containers. Use dicts for allocation outputs. Use dataclasses for structured configuration and results.
