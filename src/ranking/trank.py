"""
TRank ranking engine for AMAAM.

Combines the four factor scores (M, V, C, T) into a composite TRank score using
the formula from Section 3.1 of the specification. Handles ordinal ranking of
the M, V, and C factors, applies the raw T value, and adds the M/n tiebreaker
term. Selects the top-N assets per sleeve each month, respecting the tie-inclusion
convention from Keller (2012). Used identically by the main sleeve and the
hedging sleeve. See Section 9.10 of the specification.
"""
