"""
Data downloader for AMAAM.

Handles historical data acquisition via yfinance (for backtesting) and the
Schwab API (for live signal generation). Downloads daily OHLC data for all
ETFs in the universe, saves each ticker as a local CSV for reproducibility,
and provides a loader for previously saved files. See Section 9.3 of the
specification.
"""

import logging
import time
from pathlib import Path
from typing import Dict, List

import pandas as pd
import yfinance as yf
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Canonical column order stored in every CSV and expected by all downstream code.
_OHLCV_COLUMNS: List[str] = ["Open", "High", "Low", "Close", "Volume"]


def _normalize_yfinance_df(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Standardize a raw yfinance ``history()`` DataFrame to the canonical format.

    Keeps only OHLCV columns, removes timezone info, and normalises the index
    to midnight timestamps. Stripping the timezone keeps downstream code
    timezone-agnostic and ensures consistent CSV round-trips; we only ever
    care about the date, not the intraday time returned by yfinance.

    Parameters
    ----------
    raw : pd.DataFrame
        Output of ``yf.Ticker.history()``.
    ticker : str
        Ticker symbol used only for warning messages.

    Returns
    -------
    pd.DataFrame
        DataFrame with ``DatetimeIndex`` named ``"Date"`` and columns
        ``["Open", "High", "Low", "Close", "Volume"]``.
    """
    available = [c for c in _OHLCV_COLUMNS if c in raw.columns]
    if len(available) < len(_OHLCV_COLUMNS):
        missing = set(_OHLCV_COLUMNS) - set(available)
        logger.warning("%s: missing columns %s in yfinance output.", ticker, missing)
    df = raw[available].copy()

    # yfinance returns tz-aware index (America/New_York or UTC).
    # Convert to naive midnight timestamps so the index is a plain date series.
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.index = df.index.normalize()
    df.index.name = "Date"
    return df


def download_historical_data(
    tickers: List[str],
    start_date: str,
    end_date: str,
    source: str = "yfinance",
) -> Dict[str, pd.DataFrame]:
    """
    Download daily OHLCV data for a list of tickers.

    Each ticker is downloaded individually (rather than in bulk) to avoid the
    MultiIndex alignment issues in ``yf.download()`` when tickers have
    different inception dates.

    Parameters
    ----------
    tickers : List[str]
        Ticker symbols to download.
    start_date : str
        Inclusive start date, ISO format ``"YYYY-MM-DD"``.
    end_date : str
        Exclusive end date per yfinance convention, ISO format ``"YYYY-MM-DD"``.
    source : str, optional
        Data source. Only ``"yfinance"`` is supported here; use
        :func:`download_schwab_data` for live data. Default ``"yfinance"``.

    Returns
    -------
    Dict[str, pd.DataFrame]
        Mapping of ticker → OHLCV DataFrame. Tickers with no data returned
        are omitted and logged as warnings.

    Raises
    ------
    ValueError
        If ``source`` is not ``"yfinance"``.
    """
    if source != "yfinance":
        raise ValueError(
            f"Source '{source}' is not supported for historical download. "
            "Use source='yfinance' or call download_schwab_data() for live data."
        )

    logger.info(
        "Downloading %d tickers via yfinance (%s → %s).",
        len(tickers), start_date, end_date,
    )

    data: Dict[str, pd.DataFrame] = {}
    for ticker in tqdm(tickers, desc="Downloading", unit="ticker"):
        try:
            raw = yf.Ticker(ticker).history(
                start=start_date,
                end=end_date,
                # Receive split- and dividend-adjusted prices so downstream
                # momentum and return calculations reflect true economic returns
                # without requiring manual adjustment logic.
                auto_adjust=True,
                # Exclude the Dividends and Stock Splits event columns; our
                # factors only consume OHLCV prices, and the extra columns
                # would require explicit filtering in every downstream module.
                actions=False,
            )
            if raw.empty:
                logger.warning("%s: yfinance returned empty DataFrame — skipping.", ticker)
                continue
            data[ticker] = _normalize_yfinance_df(raw, ticker)
            logger.debug("%s: %d rows downloaded.", ticker, len(data[ticker]))
            # Brief pause to avoid overwhelming the yfinance rate limit.
            time.sleep(0.15)
        except Exception as exc:  # noqa: BLE001
            logger.error("%s: download failed — %s", ticker, exc)

    logger.info(
        "Download complete: %d / %d tickers retrieved.", len(data), len(tickers)
    )
    return data


def download_schwab_data(
    tickers: List[str],
    start_date: str,
    end_date: str,
    credentials: Dict,
) -> Dict[str, pd.DataFrame]:
    """
    Download daily OHLCV data via the Schwab API.

    Intended for live signal generation in Phase 9. Returns data in the same
    format as :func:`download_historical_data` so callers are source-agnostic.

    Parameters
    ----------
    tickers : List[str]
        Ticker symbols to download.
    start_date : str
        Start date, ISO format ``"YYYY-MM-DD"``.
    end_date : str
        End date, ISO format ``"YYYY-MM-DD"``.
    credentials : Dict
        Schwab API credentials. Expected keys: ``"api_key"``, ``"app_secret"``,
        ``"callback_url"``. See schwab-py documentation for OAuth setup.

    Returns
    -------
    Dict[str, pd.DataFrame]
        Mapping of ticker → OHLCV DataFrame.

    Raises
    ------
    NotImplementedError
        Implemented in Phase 9 once schwab-py authentication is configured.
    """
    raise NotImplementedError(
        "Schwab API integration is implemented in Phase 9. "
        "Install schwab-py (pip install schwab-py) and configure credentials "
        "in a .env file before calling this function."
    )


def save_raw_data(
    data_dict: Dict[str, pd.DataFrame],
    output_dir: Path,
) -> None:
    """
    Persist each ticker's OHLCV DataFrame to a CSV file.

    Parameters
    ----------
    data_dict : Dict[str, pd.DataFrame]
        Mapping of ticker → OHLCV DataFrame.
    output_dir : Path
        Destination directory. Created if it does not exist.

    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for ticker, df in data_dict.items():
        path = output_dir / f"{ticker}.csv"
        df.to_csv(path, index=True)
        logger.debug("Saved %s → %s (%d rows).", ticker, path.name, len(df))
    logger.info("Saved %d tickers to %s.", len(data_dict), output_dir)


def load_raw_data(data_dir: Path) -> Dict[str, pd.DataFrame]:
    """
    Load previously saved raw OHLCV CSVs from *data_dir*.

    Parameters
    ----------
    data_dir : Path
        Directory containing ``{TICKER}.csv`` files.

    Returns
    -------
    Dict[str, pd.DataFrame]
        Mapping of ticker → OHLCV DataFrame with DatetimeIndex.
    """
    data_dir = Path(data_dir)
    data: Dict[str, pd.DataFrame] = {}
    for csv_path in sorted(data_dir.glob("*.csv")):
        ticker = csv_path.stem
        df = pd.read_csv(csv_path, index_col="Date", parse_dates=True)
        data[ticker] = df
        logger.debug("Loaded %s: %d rows.", ticker, len(df))
    logger.info("Loaded %d tickers from %s.", len(data), data_dir)
    return data
