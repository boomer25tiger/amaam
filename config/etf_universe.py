"""
ETF universe definitions for AMAAM.

Declares the main sleeve (16 ETFs), hedging sleeve (6 ETFs), benchmark tickers,
and associated metadata (full name, asset class, inception date). All ticker
lists used by downstream modules should be imported from here rather than
re-defined inline. See Sections 2.2, 2.3, and 2.4 of the specification.
"""

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class ETFInfo:
    """
    Metadata for a single ETF in the universe.

    Attributes
    ----------
    ticker : str
        Yahoo Finance ticker symbol.
    name : str
        Full fund name.
    asset_class : str
        Broad asset-class label used for grouping in charts and reports.
    inception_date : str
        Fund inception date, ISO format ``"YYYY-MM-DD"``. Dates before this
        use proxy returns; the proxy construction is documented in
        ``src/data/proxy.py``.
    """

    ticker: str
    name: str
    asset_class: str
    inception_date: str


# -----------------------------------------------------------------------------
# Main Sleeve — 16 ETFs (Section 2.2)
# -----------------------------------------------------------------------------
# Design rationale for exclusions:
#   - VV excluded: the 10 sector ETFs collectively reconstruct S&P 500 large-cap
#     exposure, making a dedicated large-cap ETF redundant.
#   - VAW excluded in favour of XLB to keep the sector family consistent
#     (all sectors from the SPDR suite).
#   - AGG, TIP, IGOV, SHY excluded to prevent overlap with the hedging sleeve.

MAIN_SLEEVE: List[ETFInfo] = [
    ETFInfo("IJH", "iShares Core S&P Mid-Cap ETF",             "US Mid-Cap Equity",              "2000-05-22"),
    ETFInfo("IJR", "iShares Core S&P Small-Cap ETF",           "US Small-Cap Equity",            "2000-05-22"),
    ETFInfo("EFA", "iShares MSCI EAFE ETF",                    "International Developed Equity", "2001-08-14"),
    ETFInfo("EEM", "iShares MSCI Emerging Markets ETF",        "International Emerging Equity",  "2003-04-07"),
    ETFInfo("RWR", "SPDR Dow Jones REIT ETF",                  "Real Estate",                    "2001-04-23"),
    ETFInfo("DBC", "Invesco DB Commodity Tracking ETF",        "Commodities",                    "2006-02-03"),
    ETFInfo("XLY", "SPDR Consumer Discretionary Select ETF",   "US Sector — Consumer Discret.",  "1998-12-16"),
    ETFInfo("XLV", "SPDR Health Care Select Sector ETF",       "US Sector — Health Care",        "1998-12-16"),
    ETFInfo("XLU", "SPDR Utilities Select Sector ETF",         "US Sector — Utilities",          "1998-12-16"),
    ETFInfo("XLP", "SPDR Consumer Staples Select Sector ETF",  "US Sector — Consumer Staples",   "1998-12-16"),
    ETFInfo("XLK", "SPDR Technology Select Sector ETF",        "US Sector — Technology",         "1998-12-16"),
    ETFInfo("XLI", "SPDR Industrial Select Sector ETF",        "US Sector — Industrials",        "1998-12-16"),
    ETFInfo("XLF", "SPDR Financial Select Sector ETF",         "US Sector — Financials",         "1998-12-16"),
    ETFInfo("XLE", "SPDR Energy Select Sector ETF",            "US Sector — Energy",             "1998-12-16"),
    ETFInfo("XLB", "SPDR Materials Select Sector ETF",         "US Sector — Materials",          "1998-12-16"),
    # VOX was fundamentally reconstituted in September 2018; the backtest treats
    # the series as continuous but this structural break is documented (Section 4.3).
    ETFInfo("VOX", "Vanguard Communication Services ETF",      "US Sector — Communication Svcs", "2004-09-23"),
]

# -----------------------------------------------------------------------------
# Hedging Sleeve — 6 ETFs (Section 2.3)
# -----------------------------------------------------------------------------
# Design rationale:
#   - FXY and FXF removed; BOJ/SNB policy distortions since 2012 have weakened
#     their safe-haven characteristics.
#   - UUP added: dollar index (basket of 6 currencies) provides more robust
#     currency hedge during global risk-off events than any single-currency ETF.
#   - SH retained despite daily-rebalancing decay; the momentum filter limits
#     holding duration, and no other instrument provides direct inverse equity
#     exposure for the full backtest window.
#   - SHY serves a dual role: rankable hedging asset AND cash substitute when
#     other hedging ETFs fail the momentum filter.

HEDGING_SLEEVE: List[ETFInfo] = [
    ETFInfo("GLD", "SPDR Gold Shares ETF",                          "Real Asset / Inflation Hedge",  "2004-11-18"),
    ETFInfo("TLT", "iShares 20+ Year Treasury Bond ETF",            "Long Duration / Rate Decline",  "2002-07-22"),
    ETFInfo("IEF", "iShares 7-10 Year Treasury Bond ETF",           "Moderate Duration / Rate Decl.","2002-07-22"),
    ETFInfo("SH",  "ProShares Short S&P500 ETF",                    "Direct Inverse Equity",         "2006-06-19"),
    ETFInfo("UUP", "Invesco DB US Dollar Index Bullish Fund",        "Dollar Strength / Risk-Off",    "2007-02-20"),
    # SHY is both a rankable hedging asset and the fallback cash proxy.
    ETFInfo("SHY", "iShares 1-3 Year Treasury Bond ETF",            "Cash Proxy / Capital Preserv.", "2002-07-22"),
]

# With 2004 data extension, SH proxy covers 2004-01-01→2006-06-18 (-1×SPY),
# DBC proxy covers 2004-01-01→2006-02-05 (^BCOM rebased), and
# UUP proxy covers 2004-01-01→2007-02-28 (DX-Y.NYB rebased).
# The backtest warm-up buffer (84 trading days) means the first valid signal
# is around 2004-05-01 and the live-trading window opens ~2004-06-01.
BINDING_INCEPTION_TICKER: str = "EEM"   # earliest true binding constraint: 2003-04-14
BINDING_INCEPTION_DATE: str = "2004-01-01"

# Ticker used as the cash substitute whenever a hedging ETF fails the
# momentum filter (Section 3.7).
CASH_PROXY: str = "SHY"

# -----------------------------------------------------------------------------
# Benchmarks (Section 2.4)
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class BenchmarkInfo:
    """
    Metadata for a benchmark portfolio.

    Attributes
    ----------
    label : str
        Short human-readable name used in chart legends and report tables.
    tickers : List[str]
        Ticker symbols that make up the benchmark.
    weights : Dict[str, float]
        Fixed allocation weights keyed by ticker, summing to 1.0.
        An empty dict signals equal-weight allocation, rebalanced monthly.
    description : str
        Longer description of the benchmark's construction methodology.
    """

    label: str
    tickers: List[str]
    weights: Dict[str, float]
    description: str


BENCHMARKS: List[BenchmarkInfo] = [
    BenchmarkInfo(
        label="SPY Buy-and-Hold",
        tickers=["SPY"],
        weights={"SPY": 1.0},
        description="S&P 500 buy-and-hold equity benchmark.",
    ),
    BenchmarkInfo(
        label="60/40 SPY+AGG",
        tickers=["SPY", "AGG"],
        weights={"SPY": 0.60, "AGG": 0.40},
        description="60% SPY / 40% AGG rebalanced monthly.",
    ),
    BenchmarkInfo(
        label="Passive 7Twelve",
        tickers=["VV", "IJH", "IJR", "EFA", "EEM", "RWR", "DBC",
                 "AGG", "TIP", "IGOV", "SHY", "GLD"],
        weights={},  # equal-weight across all 12 tickers, rebalanced monthly
        description="Equal-weighted passive 7Twelve portfolio (12 ETFs) rebalanced monthly.",
    ),
]

# Additional tickers required only for benchmarks (not in either sleeve).
BENCHMARK_ONLY_TICKERS: List[str] = ["SPY", "AGG", "VV", "TIP", "IGOV"]

# -----------------------------------------------------------------------------
# Convenience accessors
# -----------------------------------------------------------------------------

MAIN_SLEEVE_TICKERS: List[str] = [e.ticker for e in MAIN_SLEEVE]
HEDGING_SLEEVE_TICKERS: List[str] = [e.ticker for e in HEDGING_SLEEVE]

# Complete set of tickers that must be downloaded (union of all universes).
ALL_TICKERS: List[str] = sorted(
    set(MAIN_SLEEVE_TICKERS)
    | set(HEDGING_SLEEVE_TICKERS)
    | set(BENCHMARK_ONLY_TICKERS)
)

ETF_METADATA: Dict[str, ETFInfo] = {
    e.ticker: e for e in MAIN_SLEEVE + HEDGING_SLEEVE
}
