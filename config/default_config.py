"""
Default model configuration for AMAAM.

All numerical parameters and tunable settings for the model live here as a
dataclass. No logic module should hard-code parameter values; they must import
from this file. See Section 9.1 of the specification for the full parameter list
and the rationale behind each default value.
"""

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    # -------------------------------------------------------------------------
    # Factor lookback periods (trading days)
    # -------------------------------------------------------------------------

    # ~4 calendar months; used for both the Absolute Momentum ROC and the
    # Average Relative Correlation window (Sections 3.2 and 3.4).
    momentum_lookback: int = 84

    # RiskMetrics daily exponential decay factor (Zangari 1996).
    # Controls how quickly old observations lose influence in the EWMA variance.
    volatility_lambda: float = 0.94

    # Number of days of SMA smoothing applied to the raw EWMA variance series,
    # as specified in the RAAM paper (Section 3.3).
    volatility_smoothing: int = 10

    # Trailing window for the correlation matrix (Section 3.4).
    # Kept equal to momentum_lookback so both factors share the same horizon.
    correlation_lookback: int = 84

    # ATR period for the Trend/Breakout system upper and lower bands (Section 3.5).
    atr_period: int = 42

    # Lookback for the highest closing price used in the upper band (Section 3.5).
    atr_upper_lookback: int = 63

    # Lookback for the highest low used in the lower band (Section 3.5).
    atr_lower_lookback: int = 105

    # -------------------------------------------------------------------------
    # Factor weights (Keller heuristic, normalized — Section 3.6)
    # -------------------------------------------------------------------------

    # Momentum receives dominant weight because it is the primary return-
    # predictive signal (Keller & van Putten 2012 normalized to sum to 1.0).
    weight_momentum: float = 0.50

    # Volatility and Correlation each receive half the remaining weight,
    # consistent with the 2:1:1 ratio from the original FAA paper.
    weight_volatility: float = 0.25
    weight_correlation: float = 0.25

    # T is a raw ±2 value, not a ranked factor, so its weight is a separate
    # scale factor calibrated during implementation (Section 3.6).
    weight_trend: float = 1.0

    # -------------------------------------------------------------------------
    # Selection parameters
    # -------------------------------------------------------------------------

    # Top-N assets selected from the main sleeve each month (Section 3.7).
    main_sleeve_top_n: int = 6

    # Top-N assets selected from the hedging sleeve each month (Section 3.7).
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
    # Transaction costs (round trip, decimal: 0.0010 = 10 bps)
    # -------------------------------------------------------------------------

    # Base case for liquid ETFs, consistent with Frazzini, Israel & Moskowitz
    # (2015) estimates for institutional-size trades (Section 5.5).
    transaction_cost: float = 0.0010

    # -------------------------------------------------------------------------
    # Backtest period
    # -------------------------------------------------------------------------

    # First valid signal month for all ETFs; binding constraint is UUP
    # inception (February 2007) plus the 84-day initialization buffer (Section 4.2).
    backtest_start: str = "2007-08-01"

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
