"""
Report generation script for AMAAM.

Loads backtest results, sensitivity analysis outputs, and benchmark data, then
calls every chart function in src/visualization/matplotlib_charts.py (saving
PNGs to reports/figures/) and src/visualization/plotly_charts.py (saving HTML
to reports/interactive/). Also generates the summary statistics table for the
README. Requires a completed backtest run before it can be executed.
"""
