"""
Unit tests for src/factors/trend.py.

Covers: ATR calculation against hand computation, band construction, uptrend
signal flip (high > upper band → T = +2), downtrend signal flip (low < lower
band → T = -2), and signal persistence (no breakout → T retains prior value).
See Section 10.1 of the specification.
"""
