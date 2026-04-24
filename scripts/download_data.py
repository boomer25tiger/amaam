"""
Data download script for AMAAM.

One-time (and periodic refresh) entry point that downloads daily OHLCV data
for every ticker in the universe, validates all nine Section 4.3 checks, aligns
to the NYSE trading calendar, and writes clean CSVs to data/raw/ and
data/processed/.

Usage (from project root)::

    python scripts/download_data.py                   # first-time download
    python scripts/download_data.py --force           # force re-download
    python scripts/download_data.py --start 2004-01-01 --end 2026-04-10
    python scripts/download_data.py -v                # verbose DEBUG output
    python scripts/download_data.py --no-proxy        # skip proxy construction
"""

import argparse
import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

# Allow running from the project root without installing the package.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from config.default_config import ModelConfig                          # noqa: E402
from config.etf_universe import ALL_TICKERS                            # noqa: E402
from src.data.downloader import (                                      # noqa: E402
    download_historical_data,
    load_raw_data,
    save_raw_data,
)
from src.data.proxy import construct_all_proxies                       # noqa: E402
from src.data.validator import align_trading_calendar, validate_universe  # noqa: E402

# Proxy source tickers — needed for pre-inception series construction but
# must NOT be written to the processed/ directory as model tickers.
_PROXY_SOURCE_TICKERS: list = ["^BCOM", "DX-Y.NYB"]


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def _configure_logging(verbose: bool) -> None:
    """
    Configure console (INFO) and rotating file (DEBUG) logging per Section 11.

    The console handler gives the user a clean progress view; the file handler
    retains full DEBUG detail (factor values, per-row checks) for post-mortem
    analysis without flooding stdout.
    """
    console_level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s [%(levelname)-8s] %(name)s: %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    logging.basicConfig(level=console_level, format=fmt, datefmt=datefmt,
                        stream=sys.stdout)

    log_dir = _PROJECT_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    file_handler = RotatingFileHandler(
        log_dir / "download_data.log",
        maxBytes=10 * 1024 * 1024,   # 10 MB per file
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
    logging.getLogger().addHandler(file_handler)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    cfg = ModelConfig()
    parser = argparse.ArgumentParser(
        description="Download, validate, and save AMAAM market data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--start", default=cfg.backtest_start, metavar="YYYY-MM-DD",
        help="Download start date (inclusive).",
    )
    parser.add_argument(
        "--end", default=cfg.backtest_end, metavar="YYYY-MM-DD",
        help="Download end date (exclusive per yfinance convention).",
    )
    parser.add_argument(
        "--raw-dir", type=Path,
        default=_PROJECT_ROOT / "data" / "raw",
        help="Directory for raw downloaded CSVs.",
    )
    parser.add_argument(
        "--processed-dir", type=Path,
        default=_PROJECT_ROOT / "data" / "processed",
        help="Directory for validated, calendar-aligned CSVs.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-download even if raw CSVs already exist.",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable DEBUG-level console output.",
    )
    parser.add_argument(
        "--no-proxy", action="store_true",
        help=(
            "Skip proxy construction for SH, DBC, and UUP.  "
            "Use this flag for testing or debugging when only raw downloads are needed."
        ),
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> int:  # noqa: C901
    """Execute the full download → proxy → validate → align → save pipeline."""
    args = _parse_args()
    _configure_logging(args.verbose)
    log = logging.getLogger(__name__)

    # Decide which tickers to download: model universe + proxy sources (unless
    # --no-proxy is set, in which case proxy source tickers are not needed).
    download_tickers = list(ALL_TICKERS)
    if not args.no_proxy:
        # Add proxy source tickers that are not already in the model universe.
        for t in _PROXY_SOURCE_TICKERS:
            if t not in download_tickers:
                download_tickers.append(t)

    log.info("=" * 60)
    log.info("AMAAM Data Download")
    log.info("Model tickers   : %d", len(ALL_TICKERS))
    log.info("Proxy sources   : %s", _PROXY_SOURCE_TICKERS if not args.no_proxy else "skipped")
    log.info("Total to fetch  : %d", len(download_tickers))
    log.info("Period          : %s → %s", args.start, args.end)
    log.info("=" * 60)

    # ------------------------------------------------------------------
    # Step 1: Download raw data (skip if cached and --force not set)
    # ------------------------------------------------------------------
    raw_csvs_exist = (
        args.raw_dir.exists()
        and any(args.raw_dir.glob("*.csv"))
    )

    if raw_csvs_exist and not args.force:
        log.info(
            "Raw CSVs already exist in %s — loading from cache. "
            "Pass --force to re-download.",
            args.raw_dir,
        )
        raw_data = load_raw_data(args.raw_dir)
    else:
        raw_data = download_historical_data(
            tickers=download_tickers,
            start_date=args.start,
            end_date=args.end,
        )
        if not raw_data:
            log.error("No data downloaded. Check your network connection and try again.")
            return 1
        save_raw_data(raw_data, args.raw_dir)

    # ------------------------------------------------------------------
    # Step 2 (optional): Build and splice proxy series for SH, DBC, UUP
    # ------------------------------------------------------------------
    if args.no_proxy:
        log.info("--no-proxy flag set — skipping proxy construction.")
        # Exclude proxy source tickers from subsequent processing since they
        # were not downloaded.
        proxied_data = {t: df for t, df in raw_data.items() if t in ALL_TICKERS}
    else:
        log.info(
            "Constructing proxy series for SH, DBC, UUP back to %s...", args.start
        )
        try:
            # construct_all_proxies returns the model universe with SH/DBC/UUP
            # replaced by spliced series; proxy source tickers are stripped out.
            proxied_data = construct_all_proxies(raw_data, start_date=args.start)
        except ValueError as exc:
            log.error("Proxy construction failed: %s", exc)
            return 1

        # Confirm proxy source tickers were removed.
        for src in _PROXY_SOURCE_TICKERS:
            if src in proxied_data:
                proxied_data.pop(src)
                log.debug("Removed proxy source ticker %s from processed set.", src)

    # ------------------------------------------------------------------
    # Step 3: Validate all tickers against Section 4.3 checks
    # ------------------------------------------------------------------
    log.info("Running Section 4.3 validation checks...")
    issues = validate_universe(proxied_data)

    # Distinguish hard failures (unexpected data errors) from soft warnings
    # ([MANUAL REVIEW] items that need a human look but don't block processing)
    # and informational annotations ([INFO] items like the VOX note).
    hard_failures: dict = {}
    soft_warnings: dict = {}
    for ticker, msgs in issues.items():
        hard = [m for m in msgs if not m.startswith("[")]
        soft = [m for m in msgs if m.startswith("[")]
        if hard:
            hard_failures[ticker] = hard
        if soft:
            soft_warnings[ticker] = soft

    if hard_failures:
        log.warning(
            "%d ticker(s) have hard validation failures and will be "
            "EXCLUDED from the processed output:", len(hard_failures),
        )
        for ticker, msgs in hard_failures.items():
            for msg in msgs:
                log.warning("  %-6s %s", ticker + ":", msg)

    if soft_warnings:
        log.warning(
            "%d ticker(s) have soft warnings requiring manual review:",
            len(soft_warnings),
        )
        for ticker, msgs in soft_warnings.items():
            for msg in msgs:
                log.warning("  %-6s %s", ticker + ":", msg)

    # Remove hard-failure tickers before alignment so they don't propagate.
    clean_data = {t: df for t, df in proxied_data.items() if t not in hard_failures}

    if not clean_data:
        log.error("No tickers passed validation. Aborting.")
        return 1

    # ------------------------------------------------------------------
    # Step 4: Align all series to the NYSE trading calendar
    # ------------------------------------------------------------------
    log.info("Aligning to NYSE trading calendar (Section 4.3, check 8)...")
    try:
        # Pass force_start so that late-inception benchmark-only tickers
        # (e.g. IGOV, which starts 2009-01-30) do not restrict the alignment
        # window for the full model universe.  Those tickers will be
        # forward-filled from their actual first date.
        aligned_data = align_trading_calendar(clean_data, force_start=args.start)
    except ValueError as exc:
        log.error("Calendar alignment failed: %s", exc)
        return 1

    # ------------------------------------------------------------------
    # Step 5: Save processed data
    # Proxy source tickers (^BCOM, DX-Y.NYB) were removed in Step 2 and are
    # never written here.  Only model-universe tickers reach this point.
    # ------------------------------------------------------------------
    save_raw_data(aligned_data, args.processed_dir)
    log.info("Processed data written to %s.", args.processed_dir)

    # ------------------------------------------------------------------
    # Step 6: Summary
    # ------------------------------------------------------------------
    any_df = next(iter(aligned_data.values()))
    n_sessions = len(any_df)
    date_min = any_df.index.min().date()
    date_max = any_df.index.max().date()

    log.info("=" * 60)
    log.info("Summary")
    log.info("  Tickers downloaded      : %d / %d", len(raw_data), len(download_tickers))
    log.info("  Tickers in processed/   : %d", len(aligned_data))
    log.info("  Trading sessions        : %d (%s → %s)", n_sessions, date_min, date_max)
    log.info(
        "  Hard failures excluded  : %d  (%s)",
        len(hard_failures), list(hard_failures.keys()) or "none",
    )
    log.info(
        "  Soft warnings           : %d  (see log for details)",
        len(soft_warnings),
    )
    log.info("Done.")
    log.info("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
