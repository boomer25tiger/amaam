"""
Unit tests for src/portfolio/allocation.py.

Covers: momentum filter separation (positive vs negative M), weight redirection
(2 of 6 negative → 2/6 redirected), all-negative main sleeve (100% to hedging),
all-negative hedging sleeve (100% to SHY), and weight-sum invariant (always 1.0).
See Section 10.1 of the specification.
"""
