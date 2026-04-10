"""
Data validation for AMAAM.

Runs the nine data quality checks defined in Section 4.3 of the specification
on each ticker's OHLC DataFrame: trading-day completeness, NaN detection,
OHLC consistency, volume checks, duplicate-date detection, split/dividend
adjustment verification, price-continuity flags, calendar alignment, and
the VOX reconstitution annotation. Returns structured issue reports so that
downstream code can decide whether to abort or proceed with a warning.
"""

import logging
from typing import Dict, List, Optional, Set

import exchange_calendars as xcals
import pandas as pd

logger = logging.getLogger(__name__)

# VOX was reconstituted in September 2018, changing from a telecom-heavy index
# to a FAANG/media-heavy one.  The backtest treats the series as continuous but
# we must annotate this structural break (Section 4.3, check 9).
_VOX_RECONSTITUTION_DATE: pd.Timestamp = pd.Timestamp("2018-09-01")

# Ad-hoc NYSE closures that exchange_calendars has not yet recorded.  Each
# entry is a date on which the NYSE was closed for a non-recurring reason.
# Add new entries here whenever a surprise closure occurs before the library
# is updated.
#   2025-01-09: NYSE closed for President Jimmy Carter's state funeral.
_EXTRA_NYSE_HOLIDAYS: set = {
    pd.Timestamp("2025-01-09").date(),
}

# Single-day return threshold above which we flag for manual review (check 7).
# 25 % is aggressive for a diversified ETF; legitimate spikes exist (e.g.
# SH during the 2020 COVID crash) but all occurrences warrant human eyes.
_CONTINUITY_FLAG_THRESHOLD: float = 0.25

# Single-day return threshold above which we suspect an unadjusted corporate
# action (check 6).  ETFs do not split frequently; a ±40 % overnight move is
# almost certainly a data artifact rather than a real price change.
_SPLIT_ARTIFACT_THRESHOLD: float = 0.40

# NYSE calendar singleton — created once and reused across all calls to avoid
# the overhead of re-parsing the full calendar definition.
_NYSE_CALENDAR: Optional[xcals.ExchangeCalendar] = None


def _get_nyse() -> xcals.ExchangeCalendar:
    """Return the cached NYSE (XNYS) exchange calendar, initialising on first call."""
    global _NYSE_CALENDAR
    if _NYSE_CALENDAR is None:
        _NYSE_CALENDAR = xcals.get_calendar("XNYS")
    return _NYSE_CALENDAR


def _nyse_session_dates(start: pd.Timestamp, end: pd.Timestamp) -> Set:
    """
    Return the set of NYSE trading *dates* (as ``datetime.date``) in [start, end].

    Using ``.date()`` objects makes the comparison timezone-agnostic, which
    is important because exchange_calendars returns UTC-midnight sessions while
    our DataFrames carry naive timestamps.  Ad-hoc NYSE closures not yet in
    the exchange_calendars database are subtracted via ``_EXTRA_NYSE_HOLIDAYS``.
    """
    cal = _get_nyse()
    sessions = cal.sessions_in_range(start, end)
    return {pd.Timestamp(s).date() for s in sessions} - _EXTRA_NYSE_HOLIDAYS


# ---------------------------------------------------------------------------
# Individual check functions — each returns a List[str] of issue descriptions.
# Empty list means the check passed.
# ---------------------------------------------------------------------------

def _check_duplicate_dates(df: pd.DataFrame) -> List[str]:
    """Check 5: no duplicate index entries."""
    dupes = df.index[df.index.duplicated()].tolist()
    if dupes:
        return [f"Duplicate dates ({len(dupes)}): {dupes[:10]}"]
    return []


def _check_no_nans(df: pd.DataFrame) -> List[str]:
    """Check 2: no NaN values in any OHLCV column."""
    nan_counts = df.isna().sum()
    bad = nan_counts[nan_counts > 0]
    if not bad.empty:
        return [f"NaN values — {bad.to_dict()}"]
    return []


def _check_ohlc_consistency(df: pd.DataFrame) -> List[str]:
    """
    Check 3: High >= max(Open, Close) and Low <= min(Open, Close) on every row.

    Emitted as ``[MANUAL REVIEW]`` rather than a hard failure because yfinance
    ``auto_adjust=True`` applies a precise adjustment ratio to Close but rounds
    Open/High/Low differently, occasionally producing penny-level violations
    (e.g. High < Close by $0.01).  These are adjustment artifacts, not real
    data errors.  A hard failure is only warranted if violations are large or
    numerous — flag those here and let the operator decide.
    """
    issues: List[str] = []

    high_breach = df["High"] < df[["Open", "Close"]].max(axis=1)
    if high_breach.any():
        n = int(high_breach.sum())
        sample = df.index[high_breach][:5].tolist()
        issues.append(
            f"[MANUAL REVIEW] High < max(Open,Close) on {n} row(s) "
            f"(likely auto_adjust rounding); first dates: {sample}"
        )

    low_breach = df["Low"] > df[["Open", "Close"]].min(axis=1)
    if low_breach.any():
        n = int(low_breach.sum())
        sample = df.index[low_breach][:5].tolist()
        issues.append(
            f"[MANUAL REVIEW] Low > min(Open,Close) on {n} row(s) "
            f"(likely auto_adjust rounding); first dates: {sample}"
        )

    return issues


def _check_volume(df: pd.DataFrame) -> List[str]:
    """Check 4: no zero or negative volume on trading days."""
    bad = df["Volume"] <= 0
    if bad.any():
        n = int(bad.sum())
        sample = df.index[bad][:5].tolist()
        return [f"Non-positive volume on {n} row(s); first dates: {sample}"]
    return []


def _check_missing_trading_days(df: pd.DataFrame, ticker: str) -> List[str]:
    """Check 1: no NYSE trading days absent from the DataFrame's index."""
    if df.empty:
        return ["DataFrame is empty."]

    expected = _nyse_session_dates(df.index.min(), df.index.max())
    actual = {pd.Timestamp(d).date() for d in df.index}
    missing = sorted(expected - actual)

    if missing:
        # Report total count and a sample to keep log messages manageable.
        sample = missing[:10]
        return [
            f"Missing {len(missing)} NYSE trading day(s); first 10: {sample}"
        ]
    return []


def _check_adjusted_prices(df: pd.DataFrame, ticker: str) -> List[str]:
    """
    Check 6: verify split/dividend adjustment via ``auto_adjust=True``.

    Since the downloader calls ``yf.Ticker.history(auto_adjust=True)``, all
    OHLC prices are adjusted by construction.  This check detects residual
    artifacts: a single-day return exceeding ±40 % on an ETF is almost
    certainly an unadjusted corporate action rather than a real price move.
    The tighter ±25 % continuity check (check 7) independently flags anything
    this check misses at a lower threshold.
    """
    returns = df["Close"].pct_change(fill_method=None).dropna()
    artifacts = returns.abs() > _SPLIT_ARTIFACT_THRESHOLD
    if artifacts.any():
        n = int(artifacts.sum())
        sample = returns.index[artifacts][:5].tolist()
        return [
            f"Possible unadjusted split/dividend: {n} day(s) with |return| > "
            f"{_SPLIT_ARTIFACT_THRESHOLD:.0%}; first dates: {sample}. "
            "Verify auto_adjust=True was used."
        ]
    return []


def _check_price_continuity(df: pd.DataFrame, ticker: str) -> List[str]:
    """
    Check 7: flag single-day returns exceeding ±25 % for manual review.

    These are not necessarily errors (SH, inverse ETFs can spike sharply in
    a crash) but every occurrence should be verified before trusting the data.
    Items are prefixed with ``[MANUAL REVIEW]`` so the script can distinguish
    them from hard failures.
    """
    returns = df["Close"].pct_change(fill_method=None).dropna()
    flagged = returns[returns.abs() > _CONTINUITY_FLAG_THRESHOLD]
    if not flagged.empty:
        n = len(flagged)
        sample = flagged.index[:5].tolist()
        return [
            f"[MANUAL REVIEW] {n} day(s) with |return| > "
            f"{_CONTINUITY_FLAG_THRESHOLD:.0%}; first dates: {sample}"
        ]
    return []


def _check_vox_reconstitution(df: pd.DataFrame, ticker: str) -> List[str]:
    """
    Check 9: annotate the VOX September 2018 reconstitution.

    The Communication Services sector ETF was fundamentally reconstituted in
    September 2018, shifting from a telecom-heavy index to one dominated by
    FAANG and media companies.  The backtest treats the series as continuous
    per the specification, but the structural break must be documented.
    """
    if ticker != "VOX":
        return []
    if df.index.min() < _VOX_RECONSTITUTION_DATE < df.index.max():
        return [
            "[INFO] VOX reconstituted September 2018: pre-2018 series tracked "
            "a different industry composition (telecom-heavy vs FAANG/media-heavy "
            "post-2018).  Backtest treats the series as continuous per spec §4.3."
        ]
    return []


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def validate_ohlc(df: pd.DataFrame, ticker: str) -> List[str]:
    """
    Run all Section 4.3 data quality checks on a single ticker's OHLCV DataFrame.

    Check 8 (cross-ticker calendar alignment) is handled at the universe level
    by :func:`align_trading_calendar` and is therefore not included here.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame with DatetimeIndex.  Expected columns:
        ``["Open", "High", "Low", "Close", "Volume"]``.
    ticker : str
        Ticker symbol.  Used in issue messages and for the VOX-specific check.

    Returns
    -------
    List[str]
        Issue descriptions.  Empty list → all checks passed.
        Items prefixed ``[MANUAL REVIEW]`` are warnings, not hard failures.
        Items prefixed ``[INFO]`` are informational annotations.
    """
    # Run duplicate-date check first; downstream checks assume a unique index.
    issues: List[str] = []
    issues += _check_duplicate_dates(df)
    issues += _check_no_nans(df)
    issues += _check_ohlc_consistency(df)
    issues += _check_volume(df)
    issues += _check_missing_trading_days(df, ticker)
    issues += _check_adjusted_prices(df, ticker)
    issues += _check_price_continuity(df, ticker)
    issues += _check_vox_reconstitution(df, ticker)
    return issues


def validate_universe(
    data_dict: Dict[str, pd.DataFrame],
) -> Dict[str, List[str]]:
    """
    Run :func:`validate_ohlc` on every ticker in the universe.

    Parameters
    ----------
    data_dict : Dict[str, pd.DataFrame]
        Mapping of ticker → OHLCV DataFrame.

    Returns
    -------
    Dict[str, List[str]]
        Mapping of ticker → list of issue strings.  Empty list → clean.
    """
    results: Dict[str, List[str]] = {}
    for ticker, df in data_dict.items():
        issues = validate_ohlc(df, ticker)
        results[ticker] = issues
        for issue in issues:
            if issue.startswith("[INFO]"):
                logger.info("%s: %s", ticker, issue)
            else:
                # Both [MANUAL REVIEW] and hard failures go to WARNING so they
                # are always visible in the console output.
                logger.warning("%s: %s", ticker, issue)
        if not issues:
            logger.debug("%s: all checks passed.", ticker)

    n_clean = sum(1 for v in results.values() if not v)
    logger.info(
        "Validation complete — %d/%d tickers clean, %d with issues.",
        n_clean, len(results), len(results) - n_clean,
    )
    return results


def align_trading_calendar(
    data_dict: Dict[str, pd.DataFrame],
) -> Dict[str, pd.DataFrame]:
    """
    Align all ticker series to the common NYSE trading calendar (check 8).

    Determines the overlapping date range across all tickers (so that no
    ticker is ever missing data at any rebalancing date), reindexes each
    series to the NYSE session schedule for that range, and forward-fills
    gaps of up to 3 consecutive days.  Gaps larger than 3 trading days
    produce a WARNING — they likely indicate a data problem rather than a
    legitimate market closure and should be investigated manually.

    Parameters
    ----------
    data_dict : Dict[str, pd.DataFrame]
        Mapping of ticker → OHLCV DataFrame.

    Returns
    -------
    Dict[str, pd.DataFrame]
        Mapping of ticker → aligned OHLCV DataFrame.  Every DataFrame in
        the returned dict shares an identical DatetimeIndex.

    Raises
    ------
    ValueError
        If the tickers have no overlapping date range.
    """
    if not data_dict:
        return {}

    # Use the latest start and earliest end so that every ticker has data
    # on every date in the aligned window.  This naturally enforces the UUP
    # inception constraint without special-casing it here.
    common_start = max(df.index.min() for df in data_dict.values())
    common_end = min(df.index.max() for df in data_dict.values())

    if common_start >= common_end:
        raise ValueError(
            f"Tickers share no common date range "
            f"(computed start={common_start.date()}, end={common_end.date()})."
        )

    logger.info(
        "Aligning %d tickers to NYSE calendar: %s → %s.",
        len(data_dict), common_start.date(), common_end.date(),
    )

    # Build the reference index from NYSE sessions.  Converting each session
    # to a naive midnight Timestamp matches the format produced by the downloader
    # (which strips timezone via tz_localize(None) + normalize()).
    cal = _get_nyse()
    sessions = cal.sessions_in_range(common_start, common_end)
    ref_index = pd.DatetimeIndex(
        [pd.Timestamp(s.date()) for s in sessions], name="Date"
    )

    aligned: Dict[str, pd.DataFrame] = {}
    for ticker, df in data_dict.items():
        df_clipped = df.loc[common_start:common_end]
        df_reindexed = df_clipped.reindex(ref_index)

        # Forward-fill up to 3 days to bridge minor gaps (e.g. an ETF whose
        # primary exchange observes a holiday not on the NYSE schedule).
        df_filled = df_reindexed.ffill(limit=3)

        remaining_nan_rows = int(df_filled.isna().any(axis=1).sum())
        if remaining_nan_rows > 0:
            logger.warning(
                "%s: %d row(s) still contain NaN after 3-day forward-fill — "
                "possible data gap exceeding 3 consecutive trading days.",
                ticker, remaining_nan_rows,
            )

        aligned[ticker] = df_filled
        logger.debug("%s: aligned to %d sessions.", ticker, len(df_filled))

    logger.info(
        "Alignment complete — %d tickers × %d sessions.",
        len(aligned), len(ref_index),
    )
    return aligned
