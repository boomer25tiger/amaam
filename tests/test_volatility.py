"""
Unit tests for src/factors/volatility.py.

Covers: EWMA recursion correctness against hand-computed values (5–10 steps),
SMA smoothing output, annualization factor (sqrt(252)), and the zero-return
edge case (variance near zero). See Section 10.1 of the specification.
"""
