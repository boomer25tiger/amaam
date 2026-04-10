"""
Absolute Momentum factor (M) for AMAAM.

Computes the 4-month (84 trading day) Rate of Change on daily closing prices.
The momentum value serves two roles: (1) an input to the TRank ranking formula
and (2) the binary filter that determines whether a selected asset retains its
portfolio weight or redirects it to the hedging sleeve. See Section 3.2 of the
specification.
"""
