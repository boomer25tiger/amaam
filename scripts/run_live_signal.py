"""
Live monthly signal generation script for AMAAM.

Fetches current market data via the Schwab API, runs the full AMAAM pipeline
through src/live/signal_generator.py, prints the current target allocation and
factor report to stdout, and logs the decision to logs/ via
src/live/allocation_logger.py. Intended to be run on the last trading day of
each month after market close.
"""
