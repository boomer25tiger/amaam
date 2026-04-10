"""
Average Relative Correlation factor (C) for AMAAM.

Computes each asset's mean pairwise Pearson correlation with all other assets
in the same sleeve over a trailing 84-trading-day window. Assets with lower
average correlation receive higher TRank scores, promoting portfolio
diversification. Correlations are computed independently within each sleeve;
there is no cross-sleeve correlation. See Section 3.4 of the specification.
"""
