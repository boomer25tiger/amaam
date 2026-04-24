
from dataclasses import dataclass, field
from typing import List


@dataclass
class ModelConfig:
    """
    Central parameter store for all AMAAM model settings.

    No logic module should hard-code values; always import from this class.
    See Section 9.1 of the specification for full rationale behind each default.
    """

    # -------------------------------------------------------------------------
    # Factor lookback periods (trading days)
    # -------------------------------------------------------------------------

    # ~4 calendar months; used for both the Absolute Momentum ROC and the
    # Average Relative Correlation window (Sections 3.2 and 3.4).
    # Only used when momentum_blend is False.
    momentum_lookback: int = 84

    # When True, replace the single 4-month ROC with an equal-weight average
    # of ROC across the horizons in momentum_blend_lookbacks.  Averaging
    # across multiple windows reduces single-lookback regime sensitivity and
    # is more robust out-of-sample (Asness et al. 2013, Jegadeesh & Titman 1993).
    momentum_blend: bool = True

    # Trading-day lookbacks used when momentum_blend = True.
    # 21 ≈ 1 month, 63 ≈ 3 months, 126 ≈ 6 months.
    momentum_blend_lookbacks: List[int] = field(default_factory=lambda: [21, 63, 126])

    # Trading days to skip at the near end of every momentum lookback window.
    # 0  = no skip (default).
    # 21 = Jegadeesh-Titman (1993) one-month skip: each ROC is measured ending
    #      one month before the signal date, sidestepping the short-term reversal
    #      that contaminates the most-recent-month component.  The incremental
    #      benefit over blending alone is an empirical question tested separately.
    momentum_skip_days: int = 0

    # RiskMetrics daily exponential decay factor (Zangari 1996).
    # Controls how quickly old observations lose influence in the EWMA variance.
    volatility_lambda: float = 0.94

    # Number of days of SMA smoothing applied to the raw EWMA variance series,
    # as specified in the RAAM paper (Section 3.3).
    volatility_smoothing: int = 10

    # Trailing window for the correlation matrix (Section 3.4).
    # 126 days ≈ 6 calendar months — aligned with the Jegadeesh & Titman (1993)
    # 6-month standard that anchors the momentum and volatility lookbacks.
    # Provides enough observations for reliable pairwise Pearson estimates
    # (recommended minimum: ~60 return observations) while remaining responsive
    # to regime shifts in cross-asset relationships.
    correlation_lookback: int = 126

    # When True, correlation is estimated as an equal-weight average across
    # correlation_blend_lookbacks windows instead of the single correlation_lookback.
    # Mirrors the blended-momentum philosophy; reduces regime sensitivity.
    correlation_blend: bool = False

    # Lookback windows (trading days) for blended correlation.
    # 21 ≈ 1mo, 63 ≈ 3mo, 126 ≈ 6mo, 252 ≈ 12mo.
    # Note: the 21-day window produces correlation estimates from only ~21
    # return observations; Pearson is unreliable below ~60 obs but the
    # averaging with longer windows dilutes noise from this component.
    correlation_blend_lookbacks: List[int] = field(default_factory=lambda: [21, 63, 126, 252])

    # Correlation estimation method (Section 3.4).
    # Options: "pairwise", "portfolio", "portfolio_all", "market", "beta",
    # "ewm", "stress_vol", "stress_drawdown", "stress_blend", "cross_sleeve".
    # Full academic rationale for each method lives in src/factors/correlation.py.
    correlation_method: str = "pairwise"

    # Penalty weight for the cross-sleeve correlation term when
    # correlation_method == "cross_sleeve".
    # C_adj_i = C_within_i + cross_sleeve_lambda * C_cross_i
    # 0.0  = no penalty (equivalent to "pairwise")
    # 0.5  = moderate penalty; test value from specification audit
    # 1.0  = full penalty; C_within and C_cross weighted equally
    # Values above 1.0 risk over-penalising defensives in bull markets.
    cross_sleeve_lambda: float = 0.5

    # Volatility multiplier threshold for the "stress_vol" correlation method.
    # Days where SPY 21-day realized vol exceeds (stress_vol_multiplier × rolling median vol)
    # are classified as high-stress.  1.5 captures approximately the top 25% of
    # volatility days over a typical market cycle; higher values restrict to
    # more extreme crises (fewer observations, noisier estimates).
    stress_vol_multiplier: float = 1.5

    # EWM span (trading days) when correlation_method == "ewm".
    # span=126 matches the effective half-life of the default 126-day rolling
    # window (half-life ≈ span / 2 for the EWM formula used by pandas).
    # span=63 gives faster decay — similar responsiveness to a 3-month window —
    # useful for strategies that need to react more quickly to correlation regime
    # shifts such as 2008-09, 2020-03, and 2022.
    correlation_ewm_span: int = 126

    # Trend signal estimation method (Section 3.5):
    #   "keltner"   — asymmetric Keltner Channel; UB=EMA(Close,63)+ATR, LB=EMA(Close,105)-ATR
    #   "paper_atr" — original paper formula; UB=EMA(Close,63)+ATR, LB=EMA(High,105)+ATR
    #   "sma200"    — Close vs 200-day SMA; Faber (2007)
    #   "sma_ratio" — rolling Close/SMA(200) ratio with ±1% buffer zone
    #   "dual_sma"  — SMA(50) vs SMA(200) golden/death cross
    #   "donchian"  — 200-day price-channel breakout; Turtle Trading / CTA
    #   "tsmom"          — 12-month return sign; Moskowitz, Ooi & Pedersen (2012)
    #   "rolling_sharpe" — rolling 6-month Sharpe > 0; Baz et al. (2015) / AQR
    #   "r2_trend"       — OLS slope > 0 AND R² ≥ 0.65 over 126 days
    #   "macd"           — MACD(12,26) zero-line cross; Appel (1979)
    trend_method: str = "sma200"

    # Simple-moving-average window for True Range used in the Keltner Channel
    # bands (Section 3.5).  Spec formula: ATR_t = (1/42) Σ_{i=0}^{41} TR_{t-i}.
    atr_period: int = 42

    # EMA span for the upper (bullish) Keltner band: UB = EMA(Close, 63) + ATR.
    # A faster EMA (63 days ≈ 3 months) makes the system confirm uptrends
    # readily (Section 3.5).
    atr_upper_lookback: int = 63

    # EMA span for the lower (bearish) Keltner band: LB = EMA(Close, 105) − ATR.
    # A slower EMA (105 days ≈ 5 months) requires a sustained decline before
    # the downtrend signal fires (Section 3.5).
    atr_lower_lookback: int = 105

    # -------------------------------------------------------------------------
    # Factor weights (Keller heuristic, normalized — Section 3.6)
    # -------------------------------------------------------------------------

    # Raised from 0.50 to 0.65 after walk-forward audit identified the volatility
    # factor chronically over-rewarding low-vol defensives; see spec §9.1.
    weight_momentum: float = 0.65

    # Retained at 0.25; the non-momentum budget shift flows to momentum reduction
    # of the correlation weight rather than here.
    weight_volatility: float = 0.25

    # Demoted to tiebreaker role (0.10) from the original 0.25; the correlation
    # signal is the most regime-dependent of the three factors.
    weight_correlation: float = 0.10

    # T weight.  Interpretation depends on trend_rank_scale below:
    #   trend_rank_scale=False → weight applied to raw ±2 values (legacy)
    #   trend_rank_scale=True  → weight applied to ordinal rank 1..N, same
    #                            scale as wM/wV/wC; test grid: 0.10/0.15/0.20/0.25
    weight_trend: float = 1.0

    # When True, T is ordinal-ranked across assets before entering TRank,
    # placing it on the same 1..N scale as M, V, and C.  This corrects the
    # scale mismatch where raw ±2 T values had negligible influence relative
    # to ranked factors spanning 1..15.  Set False only for legacy comparison.
    trend_rank_scale: bool = False

    # -------------------------------------------------------------------------
    # Selection parameters
    # -------------------------------------------------------------------------

    main_sleeve_top_n: int = 6

    hedging_sleeve_top_n: int = 2

    # -------------------------------------------------------------------------
    # Weighting scheme
    # -------------------------------------------------------------------------

    # "equal"              — 1/N weight per selected asset (base case, Section 3.8)
    # "inverse_volatility" — weights proportional to 1/V, normalised to 1.0
    weighting_scheme: str = "equal"

    # -------------------------------------------------------------------------
    # Rebalancing
    # -------------------------------------------------------------------------

    # "monthly"  — last trading day of each month (base case, Section 5.4)
    # "biweekly" — every two weeks (~26 events per year)
    rebalancing_frequency: str = "monthly"

    # -------------------------------------------------------------------------
    # Transaction costs (one-way, decimal: 0.0005 = 5 bps)
    # -------------------------------------------------------------------------

    # 5 bps one-way reflects current ETF bid-ask spreads for liquid large-cap
    # ETFs (XLK, XLF, SPY etc. trade at 1-2 bps; less liquid names like EEM,
    # RWR at 3-5 bps).  Previous 10 bps round-trip was conservative; 5 bps
    # one-way is consistent with retail/small-institutional execution today.
    transaction_cost: float = 0.0005

    # -------------------------------------------------------------------------
    # Backtest period
    # -------------------------------------------------------------------------

    # First valid signal month for all ETFs.  With proxy series extending SH,
    # DBC, and UUP back to 2004-01-01, the binding constraint becomes EEM
    # (inception 2003-04-07) plus the 84-day initialisation buffer (~2004-05).
    # The first live-trading signal is therefore around 2004-06-01 (Section 4.2).
    backtest_start: str = "2004-01-01"

    # Present date at time of implementation (Section 5.1).
    backtest_end: str = "2026-04-10"

    # All design decisions must be finalised using only data before this date;
    # the holdout period is run exactly once at the end (Section 5.2).
    holdout_start: str = "2018-01-01"

    # -------------------------------------------------------------------------
    # EWMA initialisation
    # -------------------------------------------------------------------------

    # Number of daily returns used to seed the EWMA variance before the
    # recursive formula takes over (Section 3.3).
    volatility_init_window: int = 20

    # -------------------------------------------------------------------------
    # Yang-Zhang volatility estimator
    # -------------------------------------------------------------------------

    # 126 days (~6 months) produces more stable cross-sectional vol rankings
    # than the legacy 84-day window; confirmed by IS/OOS/walk-forward testing.
    yang_zhang_window: int = 126

    # When True, Yang-Zhang vol is averaged across vol_blend_lookbacks windows.
    vol_blend: bool = False

    # Lookback windows for blended Yang-Zhang volatility.
    # 21 ≈ 1mo, 63 ≈ 3mo, 126 ≈ 6mo, 252 ≈ 12mo.
    # Unlike correlation, 21-day Yang-Zhang vol is well-behaved (realized
    # variance sums cleanly over short windows); shorter windows capture
    # recent volatility clustering, longer windows provide stability.
    vol_blend_lookbacks: List[int] = field(default_factory=lambda: [21, 63, 126, 252])
