"""
Static Matplotlib chart library for AMAAM.

Produces all 24 charts defined in Section 7.1 of the specification as PNG files
saved to reports/figures/. Chart style constants (fonts, colors, line widths,
figure dimensions) are defined once at the top of this module and applied
consistently across all functions. One public function per chart. Intended for
the GitHub README and the research summary PDF. See Section 9.18 of the
specification.
"""

import os
from pathlib import Path
from typing import Dict, List, TYPE_CHECKING

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy import stats as sp_stats

from src.backtest.metrics import compute_rolling_metrics

if TYPE_CHECKING:
    from config.default_config import ModelConfig
    from src.backtest.engine import BacktestResult

# ---------------------------------------------------------------------------
# Style constants — applied consistently to all 24 charts (Section 7.1)
# ---------------------------------------------------------------------------

AMAAM_COLOR    = "#2196F3"  # blue
SPY_COLOR      = "#F44336"  # red
SIXTY_40_COLOR = "#4CAF50"  # green
SEVEN_12_COLOR = "#9C27B0"  # purple
POSITIVE_COLOR = "#4CAF50"
NEGATIVE_COLOR = "#F44336"
NEUTRAL_COLOR  = "#9E9E9E"

TITLE_SIZE  = 14
LABEL_SIZE  = 11
TICK_SIZE   = 9
LINE_WIDTH  = 1.8
FIGURE_DPI  = 150
FIGURE_SIZE = (12, 6)

BENCHMARK_COLORS: Dict[str, str] = {
    "AMAAM":   AMAAM_COLOR,
    "SPY B&H": SPY_COLOR,
    "60/40":   SIXTY_40_COLOR,
    "7Twelve": SEVEN_12_COLOR,
}

# IS/OOS split date used by the backtest (Section 5.2).
_IS_OOS_SPLIT = "2018-01-01"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ensure_dir(output_dir: str) -> Path:
    """Create *output_dir* if it does not exist and return it as a Path."""
    p = Path(output_dir)
    os.makedirs(p, exist_ok=True)
    return p


def _pct_fmt(x: float, _pos: int) -> str:
    """Format axis tick as percentage string."""
    return f"{x * 100:.0f}%"


def _annual_returns(monthly: pd.Series) -> pd.Series:
    """Compound monthly returns into calendar-year returns."""
    return monthly.groupby(monthly.index.year).apply(lambda r: (1 + r).prod() - 1)


# ---------------------------------------------------------------------------
# 01 — Equity curves (log scale)
# ---------------------------------------------------------------------------

def plot_equity_curves(
    equity_curves: Dict[str, pd.Series],
    output_dir: str,
    filename: str = "01_equity_curves.png",
) -> Path:
    """Log-scale cumulative equity curves for AMAAM and all benchmarks.

    Parameters
    ----------
    equity_curves : Dict[str, pd.Series]
        Strategy label → equity series (already normalised to start at 1.0).
    output_dir : str
        Directory where the PNG is saved.
    filename : str
        Output file name.

    Returns
    -------
    Path
        Absolute path to the saved PNG.
    """
    out = _ensure_dir(output_dir)
    fig, ax = plt.subplots(figsize=FIGURE_SIZE, dpi=FIGURE_DPI)

    for label, series in equity_curves.items():
        lw = LINE_WIDTH * 1.4 if label == "AMAAM" else LINE_WIDTH
        color = BENCHMARK_COLORS.get(label, NEUTRAL_COLOR)
        ax.semilogy(series.index, series.values, label=label, color=color, linewidth=lw)

    ax.axvline(pd.Timestamp(_IS_OOS_SPLIT), color="gray", linestyle="--",
               linewidth=1.0, label="IS/OOS split")
    ax.set_title("Cumulative Performance (Log Scale)", fontsize=TITLE_SIZE)
    ax.set_ylabel("Portfolio Value (log)", fontsize=LABEL_SIZE)
    ax.tick_params(labelsize=TICK_SIZE)
    ax.legend(fontsize=TICK_SIZE)
    ax.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    path = out / filename
    fig.savefig(path, dpi=FIGURE_DPI)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 02 — Drawdown over time
# ---------------------------------------------------------------------------

def plot_drawdowns(
    equity_curves: Dict[str, pd.Series],
    output_dir: str,
    filename: str = "02_drawdowns.png",
) -> Path:
    """Peak-to-trough drawdown over time for all strategies.

    Parameters
    ----------
    equity_curves : Dict[str, pd.Series]
        Strategy label → equity series (normalised to 1.0).
    output_dir : str
        Directory where the PNG is saved.
    filename : str
        Output file name.

    Returns
    -------
    Path
        Absolute path to the saved PNG.
    """
    out = _ensure_dir(output_dir)
    fig, ax = plt.subplots(figsize=FIGURE_SIZE, dpi=FIGURE_DPI)

    for label, series in equity_curves.items():
        dd = series / series.cummax() - 1
        color = BENCHMARK_COLORS.get(label, NEUTRAL_COLOR)
        lw = LINE_WIDTH * 1.4 if label == "AMAAM" else LINE_WIDTH
        ax.plot(dd.index, dd.values, label=label, color=color, linewidth=lw)
        if label == "AMAAM":
            ax.fill_between(dd.index, dd.values, 0, color=color, alpha=0.15)

    ax.set_title("Drawdown Over Time", fontsize=TITLE_SIZE)
    ax.set_ylabel("Drawdown", fontsize=LABEL_SIZE)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_pct_fmt))
    ax.tick_params(labelsize=TICK_SIZE)
    ax.legend(fontsize=TICK_SIZE)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = out / filename
    fig.savefig(path, dpi=FIGURE_DPI)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 03 — Monthly return heatmap
# ---------------------------------------------------------------------------

def plot_monthly_return_heatmap(
    monthly_returns: pd.Series,
    output_dir: str,
    filename: str = "03_monthly_heatmap.png",
) -> Path:
    """Year x month heatmap of AMAAM monthly returns.

    Parameters
    ----------
    monthly_returns : pd.Series
        AMAAM monthly return series.
    output_dir : str
        Directory where the PNG is saved.
    filename : str
        Output file name.

    Returns
    -------
    Path
        Absolute path to the saved PNG.
    """
    out = _ensure_dir(output_dir)

    pivot = monthly_returns.groupby([monthly_returns.index.year,
                                     monthly_returns.index.month]).mean().unstack()
    pivot.columns = ["Jan","Feb","Mar","Apr","May","Jun",
                     "Jul","Aug","Sep","Oct","Nov","Dec"][:len(pivot.columns)]

    fig, ax = plt.subplots(figsize=(14, max(4, len(pivot) * 0.45)), dpi=FIGURE_DPI)
    vabs = max(abs(pivot.values[~np.isnan(pivot.values)])) if pivot.size else 0.05
    im = ax.imshow(pivot.values, cmap="RdYlGn", vmin=-vabs, vmax=vabs, aspect="auto")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, fontsize=TICK_SIZE)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=TICK_SIZE)

    for r in range(pivot.shape[0]):
        for c in range(pivot.shape[1]):
            val = pivot.iloc[r, c]
            if not np.isnan(val):
                ax.text(c, r, f"{val*100:.1f}%", ha="center", va="center",
                        fontsize=7, color="black")

    plt.colorbar(im, ax=ax, format=mticker.FuncFormatter(_pct_fmt))
    ax.set_title("AMAAM Monthly Returns Heatmap", fontsize=TITLE_SIZE)

    plt.tight_layout()
    path = out / filename
    fig.savefig(path, dpi=FIGURE_DPI)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 04 — Annual returns bar chart
# ---------------------------------------------------------------------------

def plot_annual_returns(
    returns_dict: Dict[str, pd.Series],
    output_dir: str,
    filename: str = "04_annual_returns.png",
) -> Path:
    """Grouped bar chart of calendar-year returns for AMAAM and benchmarks.

    Parameters
    ----------
    returns_dict : Dict[str, pd.Series]
        Strategy label → monthly return series.
    output_dir : str
        Directory where the PNG is saved.
    filename : str
        Output file name.

    Returns
    -------
    Path
        Absolute path to the saved PNG.
    """
    out = _ensure_dir(output_dir)

    annual: Dict[str, pd.Series] = {lbl: _annual_returns(r) for lbl, r in returns_dict.items()}
    all_years = sorted({yr for s in annual.values() for yr in s.index})

    n_strats = len(annual)
    width = 0.8 / max(n_strats, 1)
    offsets = np.linspace(-(n_strats - 1) * width / 2, (n_strats - 1) * width / 2, n_strats)

    fig, ax = plt.subplots(figsize=FIGURE_SIZE, dpi=FIGURE_DPI)

    for (label, series), offset in zip(annual.items(), offsets):
        x = np.array([all_years.index(yr) for yr in series.index])
        color = BENCHMARK_COLORS.get(label, NEUTRAL_COLOR)
        ax.bar(x + offset, series.values, width=width * 0.9, label=label,
               color=color, alpha=0.85)

    ax.set_xticks(range(len(all_years)))
    ax.set_xticklabels(all_years, fontsize=TICK_SIZE, rotation=45)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_pct_fmt))
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Calendar-Year Returns", fontsize=TITLE_SIZE)
    ax.tick_params(labelsize=TICK_SIZE)
    ax.legend(fontsize=TICK_SIZE)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    path = out / filename
    fig.savefig(path, dpi=FIGURE_DPI)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 05 — Rolling 12-month returns
# ---------------------------------------------------------------------------

def plot_rolling_returns(
    returns_dict: Dict[str, pd.Series],
    output_dir: str,
    filename: str = "05_rolling_returns.png",
) -> Path:
    """Rolling 12-month return for all strategies.

    Parameters
    ----------
    returns_dict : Dict[str, pd.Series]
        Strategy label → monthly return series.
    output_dir : str
        Directory where the PNG is saved.
    filename : str
        Output file name.

    Returns
    -------
    Path
        Absolute path to the saved PNG.
    """
    out = _ensure_dir(output_dir)
    fig, ax = plt.subplots(figsize=FIGURE_SIZE, dpi=FIGURE_DPI)

    for label, rets in returns_dict.items():
        rolling = (1 + rets).rolling(12, min_periods=12).apply(np.prod, raw=True) - 1
        color = BENCHMARK_COLORS.get(label, NEUTRAL_COLOR)
        lw = LINE_WIDTH * 1.4 if label == "AMAAM" else LINE_WIDTH
        ax.plot(rolling.index, rolling.values, label=label, color=color, linewidth=lw)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Rolling 12-Month Return", fontsize=TITLE_SIZE)
    ax.set_ylabel("Return", fontsize=LABEL_SIZE)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_pct_fmt))
    ax.tick_params(labelsize=TICK_SIZE)
    ax.legend(fontsize=TICK_SIZE)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = out / filename
    fig.savefig(path, dpi=FIGURE_DPI)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 06 — Rolling Sharpe ratio
# ---------------------------------------------------------------------------

def plot_rolling_sharpe(
    returns_dict: Dict[str, pd.Series],
    output_dir: str,
    filename: str = "06_rolling_sharpe.png",
) -> Path:
    """Rolling 12-month Sharpe ratio over time.

    Parameters
    ----------
    returns_dict : Dict[str, pd.Series]
        Strategy label → monthly return series.
    output_dir : str
        Directory where the PNG is saved.
    filename : str
        Output file name.

    Returns
    -------
    Path
        Absolute path to the saved PNG.
    """
    out = _ensure_dir(output_dir)
    fig, ax = plt.subplots(figsize=FIGURE_SIZE, dpi=FIGURE_DPI)

    for label, rets in returns_dict.items():
        rm = compute_rolling_metrics(rets, window=12)
        color = BENCHMARK_COLORS.get(label, NEUTRAL_COLOR)
        lw = LINE_WIDTH * 1.4 if label == "AMAAM" else LINE_WIDTH
        ax.plot(rm.index, rm["rolling_sharpe"], label=label, color=color, linewidth=lw)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title("Rolling 12-Month Sharpe Ratio", fontsize=TITLE_SIZE)
    ax.set_ylabel("Sharpe Ratio", fontsize=LABEL_SIZE)
    ax.tick_params(labelsize=TICK_SIZE)
    ax.legend(fontsize=TICK_SIZE)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = out / filename
    fig.savefig(path, dpi=FIGURE_DPI)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 07 — Rolling volatility
# ---------------------------------------------------------------------------

def plot_rolling_volatility(
    returns_dict: Dict[str, pd.Series],
    output_dir: str,
    filename: str = "07_rolling_vol.png",
) -> Path:
    """Rolling 12-month annualised volatility.

    Parameters
    ----------
    returns_dict : Dict[str, pd.Series]
        Strategy label → monthly return series.
    output_dir : str
        Directory where the PNG is saved.
    filename : str
        Output file name.

    Returns
    -------
    Path
        Absolute path to the saved PNG.
    """
    out = _ensure_dir(output_dir)
    fig, ax = plt.subplots(figsize=FIGURE_SIZE, dpi=FIGURE_DPI)

    for label, rets in returns_dict.items():
        rm = compute_rolling_metrics(rets, window=12)
        color = BENCHMARK_COLORS.get(label, NEUTRAL_COLOR)
        lw = LINE_WIDTH * 1.4 if label == "AMAAM" else LINE_WIDTH
        ax.plot(rm.index, rm["rolling_vol"], label=label, color=color, linewidth=lw)

    ax.set_title("Rolling 12-Month Annualised Volatility", fontsize=TITLE_SIZE)
    ax.set_ylabel("Volatility", fontsize=LABEL_SIZE)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_pct_fmt))
    ax.tick_params(labelsize=TICK_SIZE)
    ax.legend(fontsize=TICK_SIZE)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = out / filename
    fig.savefig(path, dpi=FIGURE_DPI)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 08 — Rolling drawdown
# ---------------------------------------------------------------------------

def plot_rolling_drawdown(
    equity_curves: Dict[str, pd.Series],
    output_dir: str,
    filename: str = "08_rolling_drawdown.png",
) -> Path:
    """Rolling 12-month worst drawdown (trailing window).

    Parameters
    ----------
    equity_curves : Dict[str, pd.Series]
        Strategy label → equity series (normalised to 1.0).
    output_dir : str
        Directory where the PNG is saved.
    filename : str
        Output file name.

    Returns
    -------
    Path
        Absolute path to the saved PNG.
    """
    out = _ensure_dir(output_dir)
    fig, ax = plt.subplots(figsize=FIGURE_SIZE, dpi=FIGURE_DPI)

    for label, series in equity_curves.items():
        dd = series / series.cummax() - 1
        rolling_worst = dd.rolling(12, min_periods=12).min()
        color = BENCHMARK_COLORS.get(label, NEUTRAL_COLOR)
        lw = LINE_WIDTH * 1.4 if label == "AMAAM" else LINE_WIDTH
        ax.plot(rolling_worst.index, rolling_worst.values,
                label=label, color=color, linewidth=lw)

    ax.set_title("Rolling 12-Month Worst Drawdown", fontsize=TITLE_SIZE)
    ax.set_ylabel("Drawdown", fontsize=LABEL_SIZE)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_pct_fmt))
    ax.tick_params(labelsize=TICK_SIZE)
    ax.legend(fontsize=TICK_SIZE)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = out / filename
    fig.savefig(path, dpi=FIGURE_DPI)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 09 — Return distribution
# ---------------------------------------------------------------------------

def plot_return_distribution(
    returns_dict: Dict[str, pd.Series],
    output_dir: str,
    filename: str = "09_return_distribution.png",
) -> Path:
    """Histogram of AMAAM monthly returns with normal distribution overlay.

    Parameters
    ----------
    returns_dict : Dict[str, pd.Series]
        Strategy label → monthly return series (AMAAM key required).
    output_dir : str
        Directory where the PNG is saved.
    filename : str
        Output file name.

    Returns
    -------
    Path
        Absolute path to the saved PNG.
    """
    out = _ensure_dir(output_dir)
    rets = returns_dict.get("AMAAM", next(iter(returns_dict.values())))

    fig, ax = plt.subplots(figsize=FIGURE_SIZE, dpi=FIGURE_DPI)

    ax.hist(rets.dropna(), bins=40, color=AMAAM_COLOR, alpha=0.6,
            density=True, label="AMAAM monthly returns")

    mu, sigma = rets.mean(), rets.std()
    x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 200)
    ax.plot(x, sp_stats.norm.pdf(x, mu, sigma), color="black",
            linestyle="--", linewidth=LINE_WIDTH, label="Normal fit")

    ax.axvline(0, color=NEGATIVE_COLOR, linewidth=1.0, linestyle=":")
    ax.set_title("AMAAM Monthly Return Distribution", fontsize=TITLE_SIZE)
    ax.set_xlabel("Monthly Return", fontsize=LABEL_SIZE)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_pct_fmt))
    ax.tick_params(labelsize=TICK_SIZE)
    ax.legend(fontsize=TICK_SIZE)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = out / filename
    fig.savefig(path, dpi=FIGURE_DPI)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 10 — Main sleeve allocation (stacked area)
# ---------------------------------------------------------------------------

def plot_main_sleeve_allocation(
    allocations: pd.DataFrame,
    main_tickers: List[str],
    output_dir: str,
    filename: str = "10_main_sleeve.png",
) -> Path:
    """Stacked area chart of main sleeve holdings over time.

    Parameters
    ----------
    allocations : pd.DataFrame
        Weight history (index=signal dates, columns=tickers).
    main_tickers : List[str]
        Main sleeve tickers to include.
    output_dir : str
        Directory where the PNG is saved.
    filename : str
        Output file name.

    Returns
    -------
    Path
        Absolute path to the saved PNG.
    """
    out = _ensure_dir(output_dir)
    cols = [t for t in main_tickers if t in allocations.columns]
    data = allocations[cols].fillna(0) if cols else pd.DataFrame()

    fig, ax = plt.subplots(figsize=FIGURE_SIZE, dpi=FIGURE_DPI)
    if not data.empty:
        ax.stackplot(data.index, *[data[c].values for c in cols], labels=cols, alpha=0.85)
        ax.legend(fontsize=TICK_SIZE, loc="upper left", ncol=3)

    ax.set_title("Main Sleeve Allocation Over Time", fontsize=TITLE_SIZE)
    ax.set_ylabel("Weight", fontsize=LABEL_SIZE)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_pct_fmt))
    ax.tick_params(labelsize=TICK_SIZE)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = out / filename
    fig.savefig(path, dpi=FIGURE_DPI)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 11 — Hedging sleeve allocation (stacked area)
# ---------------------------------------------------------------------------

def plot_hedging_sleeve_allocation(
    allocations: pd.DataFrame,
    hedge_tickers: List[str],
    output_dir: str,
    filename: str = "11_hedging_sleeve.png",
) -> Path:
    """Stacked area chart of hedging sleeve holdings over time.

    Parameters
    ----------
    allocations : pd.DataFrame
        Weight history (index=signal dates, columns=tickers).
    hedge_tickers : List[str]
        Hedging sleeve tickers to include.
    output_dir : str
        Directory where the PNG is saved.
    filename : str
        Output file name.

    Returns
    -------
    Path
        Absolute path to the saved PNG.
    """
    out = _ensure_dir(output_dir)
    cols = [t for t in hedge_tickers if t in allocations.columns]
    data = allocations[cols].fillna(0) if cols else pd.DataFrame()

    fig, ax = plt.subplots(figsize=FIGURE_SIZE, dpi=FIGURE_DPI)
    if not data.empty:
        ax.stackplot(data.index, *[data[c].values for c in cols], labels=cols, alpha=0.85)
        ax.legend(fontsize=TICK_SIZE, loc="upper left")

    ax.set_title("Hedging Sleeve Allocation Over Time", fontsize=TITLE_SIZE)
    ax.set_ylabel("Weight", fontsize=LABEL_SIZE)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_pct_fmt))
    ax.tick_params(labelsize=TICK_SIZE)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = out / filename
    fig.savefig(path, dpi=FIGURE_DPI)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 12 — Total hedging weight over time
# ---------------------------------------------------------------------------

def plot_hedging_weight_over_time(
    allocations: pd.DataFrame,
    hedge_tickers: List[str],
    output_dir: str,
    filename: str = "12_hedging_weight.png",
) -> Path:
    """Line chart of total hedging sleeve allocation over time.

    Parameters
    ----------
    allocations : pd.DataFrame
        Weight history (index=signal dates, columns=tickers).
    hedge_tickers : List[str]
        Hedging sleeve tickers to sum.
    output_dir : str
        Directory where the PNG is saved.
    filename : str
        Output file name.

    Returns
    -------
    Path
        Absolute path to the saved PNG.
    """
    out = _ensure_dir(output_dir)
    cols = [t for t in hedge_tickers if t in allocations.columns]
    hedge_total = allocations[cols].fillna(0).sum(axis=1) if cols else pd.Series(dtype=float)

    fig, ax = plt.subplots(figsize=FIGURE_SIZE, dpi=FIGURE_DPI)
    ax.plot(hedge_total.index, hedge_total.values, color=SPY_COLOR, linewidth=LINE_WIDTH)
    ax.fill_between(hedge_total.index, hedge_total.values, alpha=0.2, color=SPY_COLOR)
    ax.set_title("Total Hedging Sleeve Weight Over Time", fontsize=TITLE_SIZE)
    ax.set_ylabel("Hedging Weight", fontsize=LABEL_SIZE)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_pct_fmt))
    ax.tick_params(labelsize=TICK_SIZE)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = out / filename
    fig.savefig(path, dpi=FIGURE_DPI)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 13 — Turnover bar chart
# ---------------------------------------------------------------------------

def plot_turnover(
    turnover: pd.Series,
    output_dir: str,
    filename: str = "13_turnover.png",
) -> Path:
    """Monthly turnover bar chart with rolling 12-month average overlay.

    Parameters
    ----------
    turnover : pd.Series
        Per-period portfolio turnover values.
    output_dir : str
        Directory where the PNG is saved.
    filename : str
        Output file name.

    Returns
    -------
    Path
        Absolute path to the saved PNG.
    """
    out = _ensure_dir(output_dir)
    fig, ax = plt.subplots(figsize=FIGURE_SIZE, dpi=FIGURE_DPI)

    ax.bar(turnover.index, turnover.values, color=AMAAM_COLOR, alpha=0.5, label="Turnover")
    rolling_avg = turnover.rolling(12, min_periods=1).mean()
    ax.plot(rolling_avg.index, rolling_avg.values, color=NEGATIVE_COLOR,
            linewidth=LINE_WIDTH, label="12-month rolling avg")

    ax.set_title("Portfolio Turnover Over Time", fontsize=TITLE_SIZE)
    ax.set_ylabel("Turnover (sum |Δw|)", fontsize=LABEL_SIZE)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_pct_fmt))
    ax.tick_params(labelsize=TICK_SIZE)
    ax.legend(fontsize=TICK_SIZE)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    path = out / filename
    fig.savefig(path, dpi=FIGURE_DPI)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 14 — TRank factor weights (static bar)
# ---------------------------------------------------------------------------

def plot_factor_weights(
    config: "ModelConfig",
    output_dir: str,
    filename: str = "14_factor_weights.png",
) -> Path:
    """Static bar chart of the configured TRank factor weights.

    Parameters
    ----------
    config : ModelConfig
        Model configuration containing factor weight fields.
    output_dir : str
        Directory where the PNG is saved.
    filename : str
        Output file name.

    Returns
    -------
    Path
        Absolute path to the saved PNG.
    """
    out = _ensure_dir(output_dir)

    # wT is a raw scale factor, not a ranked factor, so it is shown separately.
    labels = ["wM (Momentum)", "wV (Volatility)", "wC (Correlation)", "wT (Trend)"]
    values = [config.weight_momentum, config.weight_volatility,
              config.weight_correlation, config.weight_trend]
    colors = [AMAAM_COLOR, SIXTY_40_COLOR, SEVEN_12_COLOR, SPY_COLOR]

    fig, ax = plt.subplots(figsize=(8, 5), dpi=FIGURE_DPI)
    bars = ax.bar(labels, values, color=colors, alpha=0.85)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.2f}", ha="center", va="bottom", fontsize=LABEL_SIZE)

    ax.set_title("TRank Factor Weights", fontsize=TITLE_SIZE)
    ax.set_ylabel("Weight / Scale", fontsize=LABEL_SIZE)
    ax.tick_params(labelsize=TICK_SIZE)
    fig.text(0.5, 0.01,
             "Fixed weights (no walk-forward optimisation)",
             ha="center", fontsize=TICK_SIZE, color="gray")
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    path = out / filename
    fig.savefig(path, dpi=FIGURE_DPI)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 15 — Return decomposition (main vs hedging sleeve)
# ---------------------------------------------------------------------------

def plot_sleeve_return_decomposition(
    monthly_returns: pd.Series,
    allocations: pd.DataFrame,
    main_tickers: List[str],
    hedge_tickers: List[str],
    output_dir: str,
    filename: str = "15_return_decomp.png",
) -> Path:
    """Approximate return decomposition: main vs hedging sleeve contribution.

    Parameters
    ----------
    monthly_returns : pd.Series
        AMAAM monthly returns.
    allocations : pd.DataFrame
        Weight history (index=signal dates, columns=tickers).
    main_tickers : List[str]
        Main sleeve tickers.
    hedge_tickers : List[str]
        Hedging sleeve tickers.
    output_dir : str
        Directory where the PNG is saved.
    filename : str
        Output file name.

    Returns
    -------
    Path
        Absolute path to the saved PNG.
    """
    out = _ensure_dir(output_dir)
    main_cols  = [t for t in main_tickers  if t in allocations.columns]
    hedge_cols = [t for t in hedge_tickers if t in allocations.columns]

    _empty = pd.Series(dtype=float)
    main_w  = allocations[main_cols].fillna(0).sum(axis=1)  if main_cols  else _empty
    hedge_w = allocations[hedge_cols].fillna(0).sum(axis=1) if hedge_cols else _empty

    fig, ax = plt.subplots(figsize=FIGURE_SIZE, dpi=FIGURE_DPI)
    if not main_w.empty:
        ax.stackplot(main_w.index, main_w.values, hedge_w.values,
                     labels=["Main sleeve weight", "Hedging sleeve weight"],
                     colors=[AMAAM_COLOR, SPY_COLOR], alpha=0.6)

    ax2 = ax.twinx()
    ax2.plot(monthly_returns.index, monthly_returns.values, color="black",
             linewidth=1.0, alpha=0.6, label="Monthly return")
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(_pct_fmt))
    ax2.set_ylabel("Monthly Return", fontsize=LABEL_SIZE)

    ax.set_title("Sleeve Weight Decomposition Over Time", fontsize=TITLE_SIZE)
    ax.set_ylabel("Portfolio Weight", fontsize=LABEL_SIZE)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_pct_fmt))
    ax.tick_params(labelsize=TICK_SIZE)
    ax.legend(loc="upper left", fontsize=TICK_SIZE)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = out / filename
    fig.savefig(path, dpi=FIGURE_DPI)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 16 — Correlation matrix of main sleeve
# ---------------------------------------------------------------------------

def plot_correlation_matrix(
    data_dict: Dict[str, pd.DataFrame],
    main_tickers: List[str],
    output_dir: str,
    filename: str = "16_correlation_matrix.png",
) -> Path:
    """Full-period correlation matrix of main sleeve ETFs.

    Parameters
    ----------
    data_dict : Dict[str, pd.DataFrame]
        Processed OHLCV data keyed by ticker.
    main_tickers : List[str]
        Main sleeve tickers to include.
    output_dir : str
        Directory where the PNG is saved.
    filename : str
        Output file name.

    Returns
    -------
    Path
        Absolute path to the saved PNG.
    """
    out = _ensure_dir(output_dir)
    avail = [t for t in main_tickers if t in data_dict]
    closes = pd.DataFrame({t: data_dict[t]["Close"] for t in avail}).ffill()
    rets = closes.pct_change().dropna()
    corr = rets.corr()

    fig, ax = plt.subplots(figsize=(10, 8), dpi=FIGURE_DPI)
    im = ax.imshow(corr.values, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax)

    ax.set_xticks(range(len(avail)))
    ax.set_xticklabels(avail, rotation=45, ha="right", fontsize=TICK_SIZE)
    ax.set_yticks(range(len(avail)))
    ax.set_yticklabels(avail, fontsize=TICK_SIZE)

    for i in range(len(avail)):
        for j in range(len(avail)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center",
                    fontsize=6, color="black")

    ax.set_title("Main Sleeve ETF Correlation Matrix", fontsize=TITLE_SIZE)

    plt.tight_layout()
    path = out / filename
    fig.savefig(path, dpi=FIGURE_DPI)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 17 — Regime performance (grouped bar)
# ---------------------------------------------------------------------------

def plot_regime_performance(
    regime_df: pd.DataFrame,
    output_dir: str,
    filename: str = "17_regime_performance.png",
) -> Path:
    """Grouped bar chart of Total Return per strategy per regime.

    Parameters
    ----------
    regime_df : pd.DataFrame
        Output of compute_regime_metrics() with MultiIndex (Strategy, Regime).
    output_dir : str
        Directory where the PNG is saved.
    filename : str
        Output file name.

    Returns
    -------
    Path
        Absolute path to the saved PNG.
    """
    out = _ensure_dir(output_dir)
    if regime_df.empty:
        fig, ax = plt.subplots(figsize=FIGURE_SIZE, dpi=FIGURE_DPI)
        ax.set_title("Regime Performance (no data)", fontsize=TITLE_SIZE)
        path = out / filename
        plt.tight_layout()
        fig.savefig(path, dpi=FIGURE_DPI)
        plt.close(fig)
        return path

    pivot = regime_df["Total Return"].unstack(level="Strategy")
    strategies = pivot.columns.tolist()
    regimes = pivot.index.tolist()

    n_strats = len(strategies)
    width = 0.8 / max(n_strats, 1)
    offsets = np.linspace(-(n_strats - 1) * width / 2, (n_strats - 1) * width / 2, n_strats)

    fig, ax = plt.subplots(figsize=(14, 6), dpi=FIGURE_DPI)
    for strat, offset in zip(strategies, offsets):
        vals = pivot[strat].values
        x = np.arange(len(regimes))
        bar_colors = [POSITIVE_COLOR if v >= 0 else NEGATIVE_COLOR for v in vals]
        ax.bar(x + offset, vals, width=width * 0.9, color=bar_colors,
               alpha=0.8, label=strat)

    ax.set_xticks(range(len(regimes)))
    ax.set_xticklabels(regimes, fontsize=TICK_SIZE, rotation=20, ha="right")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_pct_fmt))
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Total Return During Market Stress Regimes", fontsize=TITLE_SIZE)
    ax.tick_params(labelsize=TICK_SIZE)
    ax.legend(fontsize=TICK_SIZE)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    path = out / filename
    fig.savefig(path, dpi=FIGURE_DPI)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 18 — Weight sensitivity heatmap (horizontal bars)
# ---------------------------------------------------------------------------

def plot_weight_sensitivity_heatmap(
    weight_df: pd.DataFrame,
    output_dir: str,
    filename: str = "18_weight_sensitivity.png",
) -> Path:
    """Horizontal bar chart of Sharpe across wM sweep.

    Parameters
    ----------
    weight_df : pd.DataFrame
        Output of run_weight_sensitivity(); index=label, column "Sharpe Ratio".
    output_dir : str
        Directory where the PNG is saved.
    filename : str
        Output file name.

    Returns
    -------
    Path
        Absolute path to the saved PNG.
    """
    out = _ensure_dir(output_dir)
    fig, ax = plt.subplots(figsize=(10, 6), dpi=FIGURE_DPI)

    sharpe = weight_df["Sharpe Ratio"]
    colors = [AMAAM_COLOR if "0.65" in idx else NEUTRAL_COLOR for idx in sharpe.index]
    ax.barh(sharpe.index, sharpe.values, color=colors, alpha=0.85)

    ax.set_title("Sharpe Ratio vs Momentum Weight (wM)", fontsize=TITLE_SIZE)
    ax.set_xlabel("Sharpe Ratio", fontsize=LABEL_SIZE)
    ax.tick_params(labelsize=TICK_SIZE)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    path = out / filename
    fig.savefig(path, dpi=FIGURE_DPI)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 19 — Selection sensitivity
# ---------------------------------------------------------------------------

def plot_selection_sensitivity(
    selection_df: pd.DataFrame,
    output_dir: str,
    filename: str = "19_selection_sensitivity.png",
) -> Path:
    """Grouped bar chart of key metrics across top-N variants.

    Parameters
    ----------
    selection_df : pd.DataFrame
        Output of run_selection_sensitivity(); index=Top N.
    output_dir : str
        Directory where the PNG is saved.
    filename : str
        Output file name.

    Returns
    -------
    Path
        Absolute path to the saved PNG.
    """
    out = _ensure_dir(output_dir)
    metrics_to_show = ["Sharpe Ratio", "Calmar Ratio", "Max Drawdown"]
    avail = [m for m in metrics_to_show if m in selection_df.columns]

    n_metrics = len(avail)
    width = 0.8 / max(n_metrics, 1)
    x = np.arange(len(selection_df))
    offsets = np.linspace(-(n_metrics - 1) * width / 2, (n_metrics - 1) * width / 2, n_metrics)

    fig, ax = plt.subplots(figsize=FIGURE_SIZE, dpi=FIGURE_DPI)
    palette = [AMAAM_COLOR, SIXTY_40_COLOR, NEGATIVE_COLOR]
    for metric, offset, color in zip(avail, offsets, palette):
        ax.bar(x + offset, selection_df[metric].values, width=width * 0.9,
               label=metric, color=color, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([f"Top {n}" for n in selection_df.index], fontsize=TICK_SIZE)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Performance vs Main Sleeve Top-N Selection", fontsize=TITLE_SIZE)
    ax.tick_params(labelsize=TICK_SIZE)
    ax.legend(fontsize=TICK_SIZE)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    path = out / filename
    fig.savefig(path, dpi=FIGURE_DPI)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 20 — Weighting scheme comparison
# ---------------------------------------------------------------------------

def plot_weighting_scheme_comparison(
    results_dict: Dict[str, Dict[str, float]],
    output_dir: str,
    filename: str = "20_weighting_comparison.png",
) -> Path:
    """Side-by-side bar chart comparing equal vs inverse-vol weighting schemes.

    Parameters
    ----------
    results_dict : Dict[str, Dict[str, float]]
        {"Equal Weight": metrics_dict, "Inverse Vol": metrics_dict}
    output_dir : str
        Directory where the PNG is saved.
    filename : str
        Output file name.

    Returns
    -------
    Path
        Absolute path to the saved PNG.
    """
    out = _ensure_dir(output_dir)
    metrics_to_show = ["Sharpe Ratio", "Sortino Ratio", "Calmar Ratio", "Max Drawdown"]
    schemes = list(results_dict.keys())

    n_metrics = len(metrics_to_show)
    width = 0.35
    x = np.arange(n_metrics)

    fig, ax = plt.subplots(figsize=FIGURE_SIZE, dpi=FIGURE_DPI)
    for i, (scheme, color) in enumerate(zip(schemes, [AMAAM_COLOR, SIXTY_40_COLOR])):
        vals = [results_dict[scheme].get(m, float("nan")) for m in metrics_to_show]
        ax.bar(x + (i - 0.5) * width, vals, width=width * 0.9,
               label=scheme, color=color, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics_to_show, fontsize=TICK_SIZE)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Weighting Scheme Comparison", fontsize=TITLE_SIZE)
    ax.tick_params(labelsize=TICK_SIZE)
    ax.legend(fontsize=TICK_SIZE)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    path = out / filename
    fig.savefig(path, dpi=FIGURE_DPI)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 21 — Transaction cost scenario equity curves
# ---------------------------------------------------------------------------

def plot_cost_scenarios_equity(
    equity_curves: Dict[str, pd.Series],
    output_dir: str,
    filename: str = "21_cost_scenarios.png",
) -> Path:
    """Equity curves at 0, 10, and 15 bps transaction cost scenarios.

    Parameters
    ----------
    equity_curves : Dict[str, pd.Series]
        {"0 bps": ..., "10 bps": ..., "15 bps": ...} equity series.
    output_dir : str
        Directory where the PNG is saved.
    filename : str
        Output file name.

    Returns
    -------
    Path
        Absolute path to the saved PNG.
    """
    out = _ensure_dir(output_dir)
    palette = [POSITIVE_COLOR, AMAAM_COLOR, NEGATIVE_COLOR]

    fig, ax = plt.subplots(figsize=FIGURE_SIZE, dpi=FIGURE_DPI)
    for (label, series), color in zip(equity_curves.items(), palette):
        lw = LINE_WIDTH * 1.4 if "10" in label else LINE_WIDTH
        ax.plot(series.index, series.values, label=label, color=color, linewidth=lw)

    ax.set_title("Equity Curves — Transaction Cost Scenarios", fontsize=TITLE_SIZE)
    ax.set_ylabel("Portfolio Value", fontsize=LABEL_SIZE)
    ax.tick_params(labelsize=TICK_SIZE)
    ax.legend(fontsize=TICK_SIZE)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = out / filename
    fig.savefig(path, dpi=FIGURE_DPI)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 22 — Transaction cost metrics table
# ---------------------------------------------------------------------------

def plot_cost_scenarios_table(
    metrics_dict: Dict[str, Dict[str, float]],
    output_dir: str,
    filename: str = "22_cost_table.png",
) -> Path:
    """Matplotlib table of key metrics across cost scenarios.

    Parameters
    ----------
    metrics_dict : Dict[str, Dict[str, float]]
        {"0 bps": metrics, "10 bps": metrics, "15 bps": metrics}
    output_dir : str
        Directory where the PNG is saved.
    filename : str
        Output file name.

    Returns
    -------
    Path
        Absolute path to the saved PNG.
    """
    out = _ensure_dir(output_dir)
    row_keys = [
        "Annualized Return", "Annualized Volatility", "Sharpe Ratio",
        "Sortino Ratio", "Calmar Ratio", "Max Drawdown",
        "Best Year", "Worst Year",
    ]
    scenarios = list(metrics_dict.keys())

    def _fmt(val: float, key: str) -> str:
        if np.isnan(val):
            return "—"
        pct_keys = {"Annualized Return", "Annualized Volatility", "Max Drawdown",
                    "Best Year", "Worst Year"}
        return f"{val*100:.2f}%" if key in pct_keys else f"{val:.3f}"

    cell_data = [[_fmt(metrics_dict[s].get(k, float("nan")), k) for s in scenarios]
                 for k in row_keys]

    fig, ax = plt.subplots(figsize=(10, 5), dpi=FIGURE_DPI)
    ax.axis("off")
    tbl = ax.table(
        cellText=cell_data,
        rowLabels=row_keys,
        colLabels=scenarios,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(TICK_SIZE)
    tbl.scale(1.2, 1.5)
    ax.set_title("Key Metrics — Transaction Cost Scenarios", fontsize=TITLE_SIZE, pad=20)

    plt.tight_layout()
    path = out / filename
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 23 — IS vs OOS equity curves (side by side)
# ---------------------------------------------------------------------------

def plot_is_oos_equity(
    is_result: "BacktestResult",
    oos_result: "BacktestResult",
    output_dir: str,
    filename: str = "23_is_oos_equity.png",
) -> Path:
    """Side-by-side equity curves for IS and OOS periods.

    Each panel's equity curve is independently normalised to 1.0 at the start
    of its period so the scaling is directly comparable.

    Parameters
    ----------
    is_result : BacktestResult
        In-sample backtest result.
    oos_result : BacktestResult
        Out-of-sample backtest result.
    output_dir : str
        Directory where the PNG is saved.
    filename : str
        Output file name.

    Returns
    -------
    Path
        Absolute path to the saved PNG.
    """
    out = _ensure_dir(output_dir)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=FIGURE_DPI)

    for ax, result, label in zip(axes, [is_result, oos_result], ["In-Sample", "Out-of-Sample"]):
        eq = result.equity_curve
        eq_norm = eq / eq.iloc[0] if len(eq) > 0 else eq
        ax.plot(eq_norm.index, eq_norm.values, color=AMAAM_COLOR,
                linewidth=LINE_WIDTH, label="AMAAM")
        ax.set_title(f"{label} Equity Curve", fontsize=TITLE_SIZE)
        ax.set_ylabel("Portfolio Value", fontsize=LABEL_SIZE)
        ax.tick_params(labelsize=TICK_SIZE)
        ax.legend(fontsize=TICK_SIZE)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = out / filename
    fig.savefig(path, dpi=FIGURE_DPI)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 24 — IS vs OOS metrics table
# ---------------------------------------------------------------------------

def plot_is_oos_stats_table(
    is_result: "BacktestResult",
    oos_result: "BacktestResult",
    output_dir: str,
    filename: str = "24_is_oos_stats.png",
) -> Path:
    """Side-by-side metrics table for IS vs OOS periods.

    Parameters
    ----------
    is_result : BacktestResult
        In-sample backtest result.
    oos_result : BacktestResult
        Out-of-sample backtest result.
    output_dir : str
        Directory where the PNG is saved.
    filename : str
        Output file name.

    Returns
    -------
    Path
        Absolute path to the saved PNG.
    """
    out = _ensure_dir(output_dir)
    key_metrics = [
        "Annualized Return", "Annualized Volatility", "Sharpe Ratio",
        "Sortino Ratio", "Calmar Ratio", "Max Drawdown", "Worst Year",
    ]

    def _fmt(val: float, key: str) -> str:
        if np.isnan(val):
            return "—"
        pct_keys = {"Annualized Return", "Annualized Volatility", "Max Drawdown", "Worst Year"}
        return f"{val*100:.2f}%" if key in pct_keys else f"{val:.3f}"

    is_vals  = [_fmt(is_result.metrics.get(k,  float("nan")), k) for k in key_metrics]
    oos_vals = [_fmt(oos_result.metrics.get(k, float("nan")), k) for k in key_metrics]

    fig, ax = plt.subplots(figsize=(10, 5), dpi=FIGURE_DPI)
    ax.axis("off")
    tbl = ax.table(
        cellText=[[is_v, oos_v] for is_v, oos_v in zip(is_vals, oos_vals)],
        rowLabels=key_metrics,
        colLabels=["In-Sample", "Out-of-Sample"],
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(TICK_SIZE)
    tbl.scale(1.2, 1.5)
    ax.set_title("IS vs OOS Performance Metrics", fontsize=TITLE_SIZE, pad=20)

    plt.tight_layout()
    path = out / filename
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    return path
