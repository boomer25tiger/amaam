"""
Benchmark portfolio construction for AMAAM.

Builds the three comparison benchmarks used throughout the backtest analysis:
(1) SPY buy-and-hold, (2) 60/40 SPY/AGG portfolio rebalanced monthly, and
(3) equal-weighted passive 7Twelve portfolio (12 ETFs) rebalanced monthly.
All benchmarks use the same execution assumptions as AMAAM (signal on last
trading day, implemented next close). See Sections 2.4 and 9.15 of the
specification.
"""
