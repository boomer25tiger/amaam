"""
Sensitivity analysis execution script for AMAAM.

Runs all four sensitivity sweeps defined in Section 6 of the specification:
factor weight grid, selection count variants, weighting scheme comparison, and
rebalancing frequency comparison. Saves results to reports/summary/ as CSV
files for inspection and as inputs to the visualization pipeline. All sweeps
use the development period only (through December 2017).
"""
