"""
Unit tests for src/portfolio/weighting.py.

Covers: equal-weight output (1/N for each asset), inverse-volatility weights
proportional to 1/V, and normalization invariant (weights sum to 1.0 in both
schemes). See Section 10.1 of the specification.
"""
