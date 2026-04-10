"""
Unit tests for src/factors/momentum.py.

Covers: known input/output ROC calculation, constant-price edge case (M=0),
and insufficient-data edge case (fewer than lookback days available). See
Section 10.1 of the specification.
"""
