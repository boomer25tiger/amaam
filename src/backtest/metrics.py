"""
Performance metrics for AMAAM backtesting.

Computes the full set of risk-adjusted performance statistics defined in Section
5.6 of the specification: annualized return, annualized volatility, Sharpe ratio,
Calmar ratio, max drawdown, max drawdown duration, best/worst month and year,
percentage of positive periods, and turnover statistics. Also provides rolling
versions of key metrics (rolling Sharpe, rolling volatility, rolling max drawdown)
and a drawdown series for charting. See Section 9.14 of the specification.
"""
