"""
Live signal generator for AMAAM.

Fetches current market data via the Schwab API (or yfinance as a fallback),
runs the complete factor computation and TRank ranking pipeline, applies the
two-sleeve allocation logic, and returns the current month's target portfolio
weights. Also formats a human-readable report of the allocation decision and
the underlying factor values for each ETF. The output should match the most
recent month produced by the backtest engine. See Section 9.20 of the
specification.
"""
