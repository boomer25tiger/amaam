"""
Unit and integration tests for src/backtest/engine.py.

Covers: 12-month backtest on synthetic data with analytically known returns
(equity curve matches manual calculation), transaction cost deduction at each
rebalancing, and consistency between monthly and bi-weekly execution paths.
See Section 10.1 and 10.2 of the specification.
"""
