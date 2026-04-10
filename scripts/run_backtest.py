"""
Main backtest execution script for AMAAM.

Loads validated data, instantiates ModelConfig with the desired parameters,
runs the full backtest via src/backtest/engine.py, computes all performance
metrics, and prints a summary table. Accepts command-line arguments for
transaction cost scenario, weighting scheme, and rebalancing frequency so that
multiple configurations can be run without editing source files.
"""
