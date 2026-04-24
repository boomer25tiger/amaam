"""
Proxy series construction for AMAAM pre-inception data extension.

Builds synthetic or rebased OHLCV series for SH, DBC, and UUP so the backtest
can start on 2004-01-01, well before each ETF's inception date.
See AMAAM specification Section 4.2 for the rationale.
"""

import logging
from typing import Dict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Splice dates — fixed historical facts (not tunable parameters)
# ---------------------------------------------------------------------------

# SH first traded on 2006-06-19; all dates before are synthetic -1× SPY.
SH_SPLICE_DATE: str = "2006-06-19"

# DBC first traded on 2006-02-06; all dates before are rebased ^BCOM.
DBC_SPLICE_DATE: str = "2006-02-06"

# UUP first traded on 2007-03-01; all dates before are rebased DX-Y.NYB.
UUP_SPLICE_DATE: str = "2007-03-01"

# Proxy source tickers — only needed for construction, never saved to processed/.
_BCOM_TICKER: str = "^BCOM"
_DXY_TICKER: str = "DX-Y.NYB"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_synthetic_inverse(
    base_df: pd.DataFrame,
    target_first_price: float,
    end_date: str,
) -> pd.DataFrame:
    """
    Construct a synthetic -1× inverse price series from *base_df* to proxy SH before its inception.

    The series is built forward from an arbitrary base of 100, then uniformly rescaled so the
    final synthetic Close equals *target_first_price*, producing a seamless join without
    distorting any returns.

    Notes
    -----
    OHLC inversion swaps the High/Low roles: a base Low becomes the synthetic High and vice versa,
    because inverting the return direction flips the intraday range.  OHLC consistency
    (High >= max(Open, Close)) is enforced after inversion to absorb any rounding violations.
    """
    end_ts = pd.Timestamp(end_date)

    # Keep only the proxy window (strictly before the splice date).
    proxy_base = base_df.loc[base_df.index < end_ts].copy()

    if len(proxy_base) < 2:
        raise ValueError(
            f"build_synthetic_inverse: base_df has fewer than 2 rows before "
            f"{end_date} (got {len(proxy_base)}). Cannot construct synthetic SH."
        )

    n = len(proxy_base)
    close_base = proxy_base["Close"].values
    open_base = proxy_base["Open"].values
    high_base = proxy_base["High"].values
    low_base = proxy_base["Low"].values
    vol_base = proxy_base["Volume"].values

    synth_close = np.empty(n)
    synth_open = np.empty(n)
    synth_high = np.empty(n)
    synth_low = np.empty(n)

    # Day-0 anchor (no prior close in window): ratio = 1.0; rebase absorbs it.
    ratio_close = np.empty(n)
    ratio_open = np.empty(n)
    ratio_high = np.empty(n)
    ratio_low = np.empty(n)

    ratio_close[0] = ratio_open[0] = ratio_high[0] = ratio_low[0] = 1.0

    for t in range(1, n):
        prev_c = close_base[t - 1]
        ratio_close[t] = 2.0 - close_base[t] / prev_c
        ratio_open[t]  = 2.0 - open_base[t]  / prev_c
        # High/Low roles swap under inversion: base Low → synthetic High.
        ratio_high[t]  = 2.0 - low_base[t]   / prev_c
        ratio_low[t]   = 2.0 - high_base[t]  / prev_c

    synth_close[0] = 100.0
    for t in range(1, n):
        synth_close[t] = synth_close[t - 1] * ratio_close[t]

    scale = target_first_price / synth_close[-1]
    synth_close *= scale

    synth_open[0] = synth_high[0] = synth_low[0] = synth_close[0]
    for t in range(1, n):
        prev_sc = synth_close[t - 1]
        synth_open[t] = prev_sc * ratio_open[t]
        synth_high[t] = prev_sc * ratio_high[t]
        synth_low[t] = prev_sc * ratio_low[t]

        # Enforce OHLC consistency (inversion can occasionally flip H/L due to
        # rounding); ensure High >= max(Open, Close) and Low <= min(Open, Close).
        curr_c = synth_close[t]
        hi_candidates = [synth_open[t], curr_c, synth_high[t]]
        lo_candidates = [synth_open[t], curr_c, synth_low[t]]
        synth_high[t] = max(hi_candidates)
        synth_low[t] = min(lo_candidates)

    result = pd.DataFrame(
        {
            "Open": synth_open,
            "High": synth_high,
            "Low": synth_low,
            "Close": synth_close,
            "Volume": vol_base,
        },
        index=proxy_base.index,
    )
    result.index.name = "Date"

    # SPY volume is valid and is used as-is for the synthetic SH proxy.
    # If for any reason the base volume is all zeros, substitute a placeholder
    # so the Volume validation check does not fail.
    if (result["Volume"] <= 0).all():
        logger.warning(
            "build_synthetic_inverse: all base Volume values are zero; "
            "substituting placeholder volume of 1."
        )
        result["Volume"] = 1.0

    return result


def build_rebased_proxy(
    proxy_df: pd.DataFrame,
    target_first_price: float,
    end_date: str,
    fill_volume: float = 0.0,
) -> pd.DataFrame:
    """
    Scale *proxy_df* so its final Close before *end_date* equals *target_first_price*, producing
    a level-matched proxy that preserves all historical returns exactly.

    Notes
    -----
    Index-based proxies (^BCOM, DX-Y.NYB) report zero volume; pass the target ETF's mean volume
    as *fill_volume* so the spliced series passes the non-positive-volume validation check without
    affecting price data.  A single-day forward-fill (limit=1) is applied to cover commodity
    calendar gaps on NYSE trading days.
    """
    if target_first_price <= 0:
        raise ValueError(
            f"build_rebased_proxy: target_first_price must be positive, "
            f"got {target_first_price}."
        )

    end_ts = pd.Timestamp(end_date)
    window = proxy_df.loc[proxy_df.index < end_ts].copy()

    if window.empty:
        raise ValueError(
            f"build_rebased_proxy: proxy_df has no rows before {end_date}."
        )

    # Forward-fill single-day commodity calendar gaps (limit=1 avoids masking real missing data).
    window = window.ffill(limit=1)

    last_proxy_close = window["Close"].iloc[-1]
    if last_proxy_close <= 0:
        raise ValueError(
            f"build_rebased_proxy: last proxy Close before {end_date} is "
            f"{last_proxy_close} — cannot compute scale factor."
        )

    scale = target_first_price / last_proxy_close
    logger.debug(
        "build_rebased_proxy: last_proxy_close=%.4f, target=%.4f, scale=%.6f",
        last_proxy_close, target_first_price, scale,
    )

    result = window.copy()
    for col in ("Open", "High", "Low", "Close"):
        if col in result.columns:
            result[col] = result[col] * scale

    # Index-based proxies (^BCOM, DX-Y.NYB) report zero volume; fill with ETF mean.
    if fill_volume > 0 and "Volume" in result.columns:
        zero_vol_mask = result["Volume"] <= 0
        if zero_vol_mask.any():
            # Cast to int to match the integer dtype used by yfinance Volume columns.
            fill_int = int(round(fill_volume))
            result = result.copy()
            result["Volume"] = result["Volume"].astype(float)
            result.loc[zero_vol_mask, "Volume"] = float(fill_int)
            logger.debug(
                "build_rebased_proxy: filled %d zero-volume rows with %d.",
                int(zero_vol_mask.sum()), fill_int,
            )

    result.index.name = "Date"
    return result


def splice_series(
    proxy_df: pd.DataFrame,
    real_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Concatenate *proxy_df* (earlier dates) with *real_df* (later dates) into a single sorted
    OHLCV series, guarding against date overlap or accidental duplicates.
    """
    if proxy_df.empty:
        logger.warning("splice_series: proxy_df is empty — returning real_df unchanged.")
        return real_df.copy()

    if real_df.empty:
        logger.warning("splice_series: real_df is empty — returning proxy_df unchanged.")
        return proxy_df.copy()

    real_start = real_df.index.min()
    proxy_max = proxy_df.index.max()

    if proxy_max >= real_start:
        raise ValueError(
            f"splice_series: proxy_df has rows on or after real_df's first date "
            f"({real_start.date()}).  Latest proxy date: {proxy_max.date()}.  "
            "Ensure proxy_df ends strictly before the real series begins."
        )

    combined = pd.concat([proxy_df, real_df], axis=0)
    combined = combined.sort_index()
    combined.index.name = "Date"

    # Guard against accidental duplicates introduced by the caller.
    if combined.index.duplicated().any():
        n_dupes = int(combined.index.duplicated().sum())
        logger.warning(
            "splice_series: %d duplicate date(s) detected after concat — "
            "keeping first occurrence per date.",
            n_dupes,
        )
        combined = combined[~combined.index.duplicated(keep="first")]

    return combined


def construct_all_proxies(
    data_dict: Dict[str, pd.DataFrame],
    start_date: str = "2004-01-01",
) -> Dict[str, pd.DataFrame]:
    """
    Orchestrate proxy construction and splicing for all three pre-inception ETFs (SH, DBC, UUP),
    then remove the raw proxy-source tickers from the returned dict so they are not saved downstream.

    Notes
    -----
    The splice boundary is derived from the actual first date in each real series rather than the
    hard-coded inception constants, because yfinance occasionally returns data 1–2 trading days
    after the official inception date.
    """
    result = {k: v.copy() for k, v in data_dict.items()}

    _require_tickers(result, ["SPY", "SH", "DBC", "UUP", _BCOM_TICKER, _DXY_TICKER])

    # ------------------------------------------------------------------
    # 1.  SH  — synthetic -1× SPY
    # ------------------------------------------------------------------
    spy_df = result["SPY"]
    sh_real = result["SH"]

    sh_real_first = sh_real.index.min().strftime("%Y-%m-%d")
    sh_first_close = float(sh_real["Close"].iloc[0])
    sh_splice_ts = sh_real.index.min()

    logger.info(
        "[PROXY] SH: synthetic -1×SPY proxy for dates before %s "
        "(first real SH close = %.4f; expected inception %s).",
        sh_real_first, sh_first_close, SH_SPLICE_DATE,
    )

    sh_proxy = build_synthetic_inverse(
        base_df=spy_df,
        target_first_price=sh_first_close,
        end_date=sh_real_first,
    )

    # Trim proxy to the requested start date.
    sh_proxy = sh_proxy.loc[sh_proxy.index >= pd.Timestamp(start_date)]

    n_proxy_days_sh = len(sh_proxy)
    logger.info("[PROXY] SH: %d proxy days added (from %s).", n_proxy_days_sh, start_date)
    _log_splice_correlation("SH", sh_proxy, sh_real, sh_splice_ts, window=252)

    result["SH"] = splice_series(sh_proxy, sh_real)

    # ------------------------------------------------------------------
    # 2.  DBC  — rebased ^BCOM
    # ------------------------------------------------------------------
    bcom_df = result[_BCOM_TICKER]
    dbc_real = result["DBC"]

    dbc_mean_volume = float(dbc_real["Volume"].mean())
    dbc_real_first = dbc_real.index.min().strftime("%Y-%m-%d")
    dbc_first_close = float(dbc_real["Close"].iloc[0])
    dbc_splice_ts = dbc_real.index.min()

    logger.info(
        "[PROXY] DBC: rebased ^BCOM proxy for dates before %s "
        "(first real DBC close = %.4f; expected inception %s).",
        dbc_real_first, dbc_first_close, DBC_SPLICE_DATE,
    )

    dbc_proxy = build_rebased_proxy(
        proxy_df=bcom_df,
        target_first_price=dbc_first_close,
        end_date=dbc_real_first,
        fill_volume=dbc_mean_volume,
    )

    dbc_proxy = dbc_proxy.loc[dbc_proxy.index >= pd.Timestamp(start_date)]

    n_proxy_days_dbc = len(dbc_proxy)
    logger.info("[PROXY] DBC: %d proxy days added (from %s).", n_proxy_days_dbc, start_date)

    _log_splice_correlation("DBC", dbc_proxy, dbc_real, dbc_splice_ts, window=252)

    result["DBC"] = splice_series(dbc_proxy, dbc_real)

    # ------------------------------------------------------------------
    # 3.  UUP  — rebased DX-Y.NYB
    # ------------------------------------------------------------------
    dxy_df = result[_DXY_TICKER]
    uup_real = result["UUP"]

    uup_mean_volume = float(uup_real["Volume"].mean())
    uup_real_first = uup_real.index.min().strftime("%Y-%m-%d")
    uup_first_close = float(uup_real["Close"].iloc[0])
    uup_splice_ts = uup_real.index.min()

    logger.info(
        "[PROXY] UUP: rebased DX-Y.NYB proxy for dates before %s "
        "(first real UUP close = %.4f; expected inception %s).",
        uup_real_first, uup_first_close, UUP_SPLICE_DATE,
    )

    uup_proxy = build_rebased_proxy(
        proxy_df=dxy_df,
        target_first_price=uup_first_close,
        end_date=uup_real_first,
        fill_volume=uup_mean_volume,
    )

    uup_proxy = uup_proxy.loc[uup_proxy.index >= pd.Timestamp(start_date)]

    n_proxy_days_uup = len(uup_proxy)
    logger.info("[PROXY] UUP: %d proxy days added (from %s).", n_proxy_days_uup, start_date)

    _log_splice_correlation("UUP", uup_proxy, uup_real, uup_splice_ts, window=252)

    result["UUP"] = splice_series(uup_proxy, uup_real)

    # ------------------------------------------------------------------
    # Remove proxy source tickers — they should NOT be saved to processed/.
    # ------------------------------------------------------------------
    result.pop(_BCOM_TICKER, None)
    result.pop(_DXY_TICKER, None)

    logger.info(
        "[PROXY] construct_all_proxies complete. "
        "SH=%d proxy days, DBC=%d proxy days, UUP=%d proxy days.",
        n_proxy_days_sh, n_proxy_days_dbc, n_proxy_days_uup,
    )

    return result


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _require_tickers(data_dict: Dict[str, pd.DataFrame], tickers: list) -> None:
    """
    Raise ``ValueError`` if any required ticker is absent or holds an empty DataFrame in
    *data_dict*, providing a clear diagnostic before proxy construction begins.
    """
    missing = [t for t in tickers if t not in data_dict or data_dict[t].empty]
    if missing:
        raise ValueError(
            f"construct_all_proxies: the following required tickers are missing "
            f"or empty in data_dict: {missing}.  "
            "Ensure ^BCOM and DX-Y.NYB are downloaded alongside the main universe."
        )


def _log_splice_correlation(
    ticker: str,
    proxy_df: pd.DataFrame,
    real_df: pd.DataFrame,
    splice_ts: pd.Timestamp,
    window: int = 252,
) -> None:
    """
    Log the Pearson correlation between the proxy's last *window* daily returns and the real
    series' first *window* daily returns as a sanity check on proxy quality at the splice point.
    """
    try:
        real_ret  = real_df["Close"].head(window).pct_change().dropna()
        proxy_ret = proxy_df["Close"].tail(window).pct_change().dropna()

        if len(real_ret) < 10 or len(proxy_ret) < 10:
            logger.info(
                "[PROXY] %s: insufficient data for overlap correlation (real=%d, proxy=%d days).",
                ticker, len(real_ret), len(proxy_ret),
            )
            return

        corr = float(np.corrcoef(proxy_ret.values[-len(real_ret):],
                                  real_ret.values[:len(proxy_ret)])[0, 1])
        n_overlap = min(len(real_ret), len(proxy_ret))
        logger.info(
            "[PROXY] %s: proxy–real return correlation = %.4f "
            "(over %d days around splice %s).",
            ticker, corr, n_overlap, splice_ts.date(),
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "[PROXY] %s: could not compute splice correlation — %s", ticker, exc
        )
