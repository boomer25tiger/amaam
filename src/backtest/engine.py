"""
Backtesting engine for AMAAM.

Core event loop that iterates over monthly (or bi-weekly) rebalancing dates,
computes the full factor stack and allocation for each period, applies the
resulting weights to the following period's returns, and deducts transaction
costs based on portfolio turnover. Returns a BacktestResult dataclass containing
the equity curve, monthly returns, weight history, turnover series, and per-period
factor values. Rebalancing frequency is a configurable parameter. See Sections
5.3, 5.4, 5.5, and 9.13 of the specification.
"""
