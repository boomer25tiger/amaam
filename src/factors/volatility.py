"""
Volatility Model factor (V) for AMAAM.

Estimates realized volatility using the J.P. Morgan RiskMetrics EWMA variance
model (lambda=0.94), followed by a 10-day SMA smoothing step and annualization
by sqrt(252). Lower volatility assets receive higher TRank scores, implementing
the risk-management dimension of the ranking engine. See Section 3.3 of the
specification.
"""
