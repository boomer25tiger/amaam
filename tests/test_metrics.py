"""
Unit tests for src/backtest/metrics.py.

Covers: Sharpe ratio against known analytical values, max drawdown against a
hand-constructed equity curve with a known peak-to-trough decline, annualization
scaling (sqrt(252) for daily, sqrt(12) for monthly), and rolling metric output
shapes. See Section 10.1 of the specification.
"""
