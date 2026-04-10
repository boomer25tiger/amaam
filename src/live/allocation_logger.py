"""
Allocation decision logger for AMAAM.

Persists each monthly allocation decision — including target weights, all factor
values (M, V, C, T, TRank), selected assets, momentum filter outcomes, and
turnover from the prior month — to a timestamped JSON file in the logs/
directory. Provides a loader that reconstructs the full decision history as a
DataFrame for post-hoc analysis. See Section 9.21 of the specification.
"""
