"""
Weighting schemes for AMAAM portfolio construction.

Provides two weighting methods that operate after asset selection and the
momentum filter: equal-weight (1/N) and inverse-volatility weight (weight
proportional to 1/V, normalized to sum to 1.0). Both schemes are applied
within each sleeve independently before the sleeves are combined. See Section
3.8 and Section 9.12 of the specification.
"""
