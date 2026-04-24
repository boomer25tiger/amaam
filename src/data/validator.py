"""
Data validation for AMAAM (Section 4.3 of the specification).

Implements the nine data-quality checks for each ticker's OHLCV DataFrame
and the cross-ticker NYSE calendar alignment step. Returns structured issue
lists so callers can distinguish hard failures from soft warnings.
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

# The earliest date the NYSE calendar must cover.  Set to 2003-01-01 so that
# data extended back to 2004 (with EEM inception 2003-04-07) is always within
# the calendar's range.  The default exchange_calendars start is 2006-04-21.
_NYSE_CALENDAR_START: str = "2003-01-01"


def _get_nyse() -> xcals.ExchangeCalendar:
    """Return the cached NYSE (XNYS) exchange calendar, initialising on first call.

    Notes
    -----
    The calendar is requested with an explicit early start date because the
    exchange_calendars default start (2006-04-21) would raise DateOutOfBounds
    when validating proxy-extended series that reach back to 2004.
    """
    global _NYSE_CALENDAR
    if _NYSE_CALENDAR is None:
        # Explicitly request an early start so the calendar covers our full
        # 2004-01-01 backtest history.  The default library start is 2006-04-21,
        # which would raise DateOutOfBounds when validating proxy-extended series.
        _NYSE_CALENDAR = xcals.get_calendar("XNYS", start=_NYSE_CALENDAR_START)
    return _NYSE_CALENDAR


def _nyse_session_dates(start: pd.Timestamp, end: pd.Timestamp) -> Set:
    """Return the set of NYSE trading dates (as ``datetime.date``) in [start, end].

    Notes
    -----
    Dates are returned as ``datetime.date`` objects rather than Timestamps to
    avoid timezone-comparison mismatches: exchange_calendars yields UTC-midnight
    sessions while our DataFrames carry naive timestamps. ``_EXTRA_NYSE_HOLIDAYS``
    removes ad-hoc closures not yet recorded in the exchange_calendars database.
    """
    cal = _get_nyse()
    sessions = cal.sessions_in_range(start, end)
    return {pd.Timestamp(s).date() for s in sessions} - _EXTRA_NYSE_HOLIDAYS


# ---------------------------------------------------------------------------
# Individual check functions — each returns a List[str] of issue descriptions.
# Empty list means the check passed.
# ---------------------------------------------------------------------------

def _check_duplicate_dates(df: pd.DataFrame) -> List[str]:
    """Detect duplicate index entries (spec check 5), which would corrupt any positional slice."""
    dupes = df.index[df.index.duplicated()].tolist()
    if dupes:
        return [f"Duplicate dates ({len(dupes)}): {dupes[:10]}"]
    return []


def _check_no_nans(df: pd.DataFrame) -> List[str]:
    """Verify that no OHLCV column contains NaN values (spec check 2), which would silently corrupt factor calculations."""
    nan_counts = df.isna().sum()
    bad = nan_counts[nan_counts > 0]
    if not bad.empty:
        return [f"NaN values — {bad.to_dict()}"]
    return []


def _check_ohlc_consistency(df: pd.DataFrame) -> List[str]:
    """Confirm that High >= max(Open, Close) and Low <= min(Open, Close) on every row (spec check 3).

    Notes
    -----
    Violations are emitted as ``[MANUAL REVIEW]`` rather than hard failures
    because yfinance ``auto_adjust=True`` applies an exact ratio to Close but
    rounds Open/High/Low separately, occasionally producing penny-level breaches
    (e.g. High < Close by $0.01) that are adjustment artifacts rather than real
    data errors. Large or numerous violations still warrant operator judgment.
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
    """Flag zero or negative volume on trading days (spec check 4).

    Notes
    -----
    Emitted as ``[MANUAL REVIEW]`` rather than a hard failure because
    early-history rows for newly-launched ETFs (e.g. VOX in 2004) can
    legitimately show zero volume. Systematic zero-volume blocks spanning
    hundreds of rows always indicate a data problem and should be investigated.
    """
    bad = df["Volume"] <= 0
    if bad.any():
        n = int(bad.sum())
        sample = df.index[bad][:5].tolist()
        return [
            f"[MANUAL REVIEW] Non-positive volume on {n} row(s); "
            f"first dates: {sample}"
        ]
    return []


def _check_missing_trading_days(df: pd.DataFrame, ticker: str) -> List[str]:
    """Verify that every NYSE session within the DataFrame's date range is present (spec check 1).

    Notes
    -----
    Gaps of 3 days or fewer are soft ``[MANUAL REVIEW]`` warnings rather than
    hard failures because the alignment step will forward-fill them; they
    typically arise from proxy-source indices (e.g. ^BCOM) that skip certain
    NYSE half-days. Gaps larger than 3 days are reported as hard failures.
    """
    if df.empty:
        return ["DataFrame is empty."]

    expected = _nyse_session_dates(df.index.min(), df.index.max())
    actual = {pd.Timestamp(d).date() for d in df.index}
    missing = sorted(expected - actual)

    if missing:
        # Report total count and a sample to keep log messages manageable.
        sample = missing[:10]
        n = len(missing)
        if n <= 3:
            # Small gaps will be resolved by the alignment forward-fill (limit=3).
            # Flag as soft warning rather than a hard failure.
            return [
                f"[MANUAL REVIEW] Missing {n} NYSE trading day(s) "
                f"(≤3; will be forward-filled by alignment); dates: {sample}"
            ]
        return [
            f"Missing {n} NYSE trading day(s); first 10: {sample}"
        ]
    return []


def _check_adjusted_prices(df: pd.DataFrame, ticker: str) -> List[str]:
    """Detect residual split/dividend adjustment artifacts by flagging single-day returns above ±40 % (spec check 6).

    Notes
    -----
    The ±40 % threshold targets unadjusted corporate actions that survived
    ``auto_adjust=True``; it is intentionally higher than the ±25 % continuity
    threshold (check 7) so the two checks cover distinct severity bands.
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
    """Flag single-day Close returns exceeding ±25 % as ``[MANUAL REVIEW]`` items (spec check 7).

    Notes
    -----
    The ±25 % threshold is aggressive for diversified ETFs; legitimate spikes
    exist (e.g. SH during the 2020 COVID crash), so the check never hard-fails —
    it only surfaces occurrences that a human should verify before trusting the data.
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
    """Annotate the September 2018 VOX index reconstitution as an ``[INFO]`` item (spec check 9).

    Notes
    -----
    The reconstitution fundamentally changed VOX from a telecom-heavy index to
    a FAANG/media-heavy one, creating a structural break in the return series.
    The spec treats the series as continuous, but the annotation ensures the
    break is always visible in validation output rather than silently assumed away.
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
    """Run all per-ticker Section 4.3 data-quality checks and return a list of issue strings.

    Notes
    -----
    Check 8 (cross-ticker calendar alignment) is intentionally absent here; it
    is handled at the universe level by :func:`align_trading_calendar`. Issue
    strings prefixed ``[MANUAL REVIEW]`` are soft warnings; ``[INFO]`` items
    are purely informational; all others are hard failures.
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
    """Run :func:`validate_ohlc` on every ticker in the universe and log a summary."""
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
    force_start: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    """Reindex every ticker to the shared NYSE session grid and forward-fill gaps of up to 3 days (spec check 8).

    Notes
    -----
    ``force_start`` exists because late-inception benchmark tickers such as
    IGOV (2009) would otherwise clip the entire universe to their start date.
    Gaps exceeding 3 consecutive trading days are not filled and produce a
    WARNING — they almost always indicate a data problem, not a legitimate
    closure, and must be investigated before running the backtest.
    """
    if not data_dict:
        return {}

    # Determine the common window.  When force_start is given, use it as the
    # alignment start instead of the latest ticker start — this prevents
    # benchmark-only tickers with late inceptions (e.g. IGOV starts 2009)
    # from clipping the model universe back to an unnecessarily late date.
    if force_start is not None:
        common_start = pd.Timestamp(force_start)
        logger.info("align_trading_calendar: force_start=%s overrides auto start.", force_start)
    else:
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
