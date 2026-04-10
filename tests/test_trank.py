"""
Unit tests for src/ranking/trank.py.

Covers: ranking direction (higher momentum → higher rank number), TRank formula
output for a fully-specified set of known inputs, top-N selection correctness,
and tiebreaker behavior (M/n term resolves equal TRank scores). See Section
10.1 of the specification.
"""
