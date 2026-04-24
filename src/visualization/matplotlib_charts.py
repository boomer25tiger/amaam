"""
Static Matplotlib chart library for AMAAM.

Produces all 24 charts defined in Section 7.1 of the specification as PNG files
saved to reports/figures/. Chart style constants (fonts, colors, line widths,
figure dimensions) are defined once at the top of this module and applied
consistently across all functions. One public function per chart. Intended for
the GitHub README and the research summary PDF. See Section 9.18 of the
specification.
"""

# Common call signature for all chart functions:
#   output_dir (Path)  — directory where the file is saved
#   filename   (str)   — output filename (PNG for this module, HTML for plotly)
#   returns    (Path)  — absolute path to the saved file
# These parameters are not individually documented in each function's docstring.

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


def _style_table(
    tbl,
    header_color: str = "#1565C0",
) -> None:
    """Apply alternating-row colours and a bold header to a Matplotlib table."""
    light_row  = "#EBF5FB"   # very light blue for odd data rows
    alt_row    = "#FFFFFF"   # white for even data rows
    row_lbl_bg = "#D6EAF8"   # slightly stronger blue for row label column
    edge_col   = "#BDBDBD"   # grid-line colour

    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor(edge_col)
        cell.set_linewidth(0.5)
        if row == 0:
            # Column header row
            cell.set_facecolor(header_color)
            cell.get_text().set_color("white")
            cell.get_text().set_fontweight("bold")
        elif col == -1:
            # Row label column
            cell.set_facecolor(row_lbl_bg)
            cell.get_text().set_fontweight("bold")
            cell.get_text().set_color("#1A1A1A")
        else:
            # Alternating data-row backgrounds
            cell.set_facecolor(light_row if row % 2 == 1 else alt_row)
            cell.get_text().set_color("#1A1A1A")


# ---------------------------------------------------------------------------
# 01 — Equity curves (linear scale, base = 100)
# ---------------------------------------------------------------------------

def plot_equity_curves(
    equity_curves: Dict[str, pd.Series],
    output_dir: str,
    filename: str = "01_equity_curves.png",
) -> Path:
    """Linear-scale cumulative equity curves for AMAAM and all benchmarks.

    All series are rescaled so that the portfolio starts at 100.  Each series
    in *equity_curves* is assumed to already be normalised to 1.0 at inception;
    multiplying by 100 gives the conventional "growth of $100" presentation.

    Parameters
    ----------
    equity_curves : Dict[str, pd.Series]
        Strategy label → equity series (normalised to start at 1.0).
    """
    out = _ensure_dir(output_dir)
    fig, ax = plt.subplots(figsize=FIGURE_SIZE, dpi=FIGURE_DPI)

    for label, series in equity_curves.items():
        lw = LINE_WIDTH * 1.6 if label == "AMAAM" else LINE_WIDTH
        color = BENCHMARK_COLORS.get(label, NEUTRAL_COLOR)
        # Rescale: series starts at 1.0, multiply by 100 → "growth of $100"
        ax.plot(series.index, series.values * 100, label=label, color=color, linewidth=lw)

    ax.axvline(pd.Timestamp(_IS_OOS_SPLIT), color="gray", linestyle="--",
               linewidth=1.0, label="IS/OOS split")
    ax.set_title("Cumulative Performance (Growth of $100)", fontsize=TITLE_SIZE)
    ax.set_ylabel("Portfolio Value ($)", fontsize=LABEL_SIZE)
    ax.tick_params(labelsize=TICK_SIZE)
    ax.legend(fontsize=TICK_SIZE)
    ax.grid(True, alpha=0.3)

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
# 14 — Return decomposition (main vs hedging sleeve)
# ---------------------------------------------------------------------------

def plot_sleeve_return_decomposition(
    monthly_returns: pd.Series,
    allocations: pd.DataFrame,
    main_tickers: List[str],
    hedge_tickers: List[str],
    output_dir: str,
    filename: str = "14_return_decomp.png",
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
# 15 — Correlation matrix of main sleeve
# ---------------------------------------------------------------------------

def plot_correlation_matrix(
    data_dict: Dict[str, pd.DataFrame],
    main_tickers: List[str],
    output_dir: str,
    filename: str = "15_correlation_matrix.png",
) -> Path:
    """Full-period correlation matrix of main sleeve ETFs.

    Parameters
    ----------
    data_dict : Dict[str, pd.DataFrame]
        Processed OHLCV data keyed by ticker.
    main_tickers : List[str]
        Main sleeve tickers to include.
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
# 16 — Regime performance (grouped bar)
# ---------------------------------------------------------------------------

def plot_regime_performance(
    regime_df: pd.DataFrame,
    output_dir: str,
    filename: str = "16_regime_performance.png",
) -> Path:
    """Grouped bar chart of Total Return per strategy per regime.

    Parameters
    ----------
    regime_df : pd.DataFrame
        Output of compute_regime_metrics() with MultiIndex (Strategy, Regime).
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

    fig, ax = plt.subplots(figsize=(15, 7), dpi=FIGURE_DPI)

    for strat, offset in zip(strategies, offsets):
        vals = pivot[strat].values
        x = np.arange(len(regimes))
        is_amaam = (strat == "AMAAM")
        # Each strategy keeps its own identity colour across all regimes so
        # viewers can track a single strategy visually.  AMAAM is additionally
        # distinguished by a thick black edge and slightly taller zorder.
        color     = BENCHMARK_COLORS.get(strat, NEUTRAL_COLOR)
        edge_col  = "black" if is_amaam else color
        edge_lw   = 1.5    if is_amaam else 0.4
        alpha     = 0.92   if is_amaam else 0.78
        zorder    = 3      if is_amaam else 2

        bars = ax.bar(
            x + offset, vals, width=width * 0.9,
            color=color, alpha=alpha, label=strat,
            edgecolor=edge_col, linewidth=edge_lw, zorder=zorder,
        )

        # Data labels — shown for all strategies to enable exact comparisons.
        for bar, val in zip(bars, vals):
            if np.isnan(val):
                continue
            sign = 1 if val >= 0 else -1
            ypos = bar.get_height() if val >= 0 else bar.get_y()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                ypos + sign * 0.008,
                f"{val * 100:.0f}%",
                ha="center",
                va="bottom" if val >= 0 else "top",
                fontsize=6.5,
                color="#1A1A1A",
                fontweight="bold" if is_amaam else "normal",
                zorder=zorder + 1,
            )

    ax.set_xticks(range(len(regimes)))
    ax.set_xticklabels(regimes, fontsize=TICK_SIZE, rotation=20, ha="right")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_pct_fmt))
    ax.axhline(0, color="black", linewidth=0.9, zorder=1)
    ax.set_title("Total Return During Market Stress Regimes", fontsize=TITLE_SIZE)
    ax.tick_params(labelsize=TICK_SIZE)
    ax.legend(fontsize=TICK_SIZE, framealpha=0.9, loc="upper right")
    ax.grid(True, axis="y", alpha=0.3, zorder=0)

    plt.tight_layout()
    path = out / filename
    fig.savefig(path, dpi=FIGURE_DPI)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 17 — Weight sensitivity heatmap (horizontal bars)
# ---------------------------------------------------------------------------

def plot_weight_sensitivity_heatmap(
    weight_df: pd.DataFrame,
    output_dir: str,
    filename: str = "17_weight_sensitivity.png",
) -> Path:
    """Horizontal bar chart of Sharpe across wM sweep.

    Parameters
    ----------
    weight_df : pd.DataFrame
        Output of run_weight_sensitivity(); index=label, column "Sharpe Ratio".
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
# 18 — Selection sensitivity
# ---------------------------------------------------------------------------

def plot_selection_sensitivity(
    selection_df: pd.DataFrame,
    output_dir: str,
    filename: str = "18_selection_sensitivity.png",
) -> Path:
    """Grouped bar chart of key metrics across top-N variants.

    Parameters
    ----------
    selection_df : pd.DataFrame
        Output of run_selection_sensitivity(); index=Top N.
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
        bars = ax.bar(x + offset, selection_df[metric].values, width=width * 0.9,
                      label=metric, color=color, alpha=0.85)
        # Data label: percentages for drawdown, two-decimal ratio for others.
        is_pct = "Drawdown" in metric
        for bar in bars:
            val = bar.get_height()
            if np.isnan(val):
                continue
            label_str = f"{val * 100:.1f}%" if is_pct else f"{val:.2f}"
            sign = 1 if val >= 0 else -1
            ypos = val + sign * (0.008 if is_pct else 0.015)
            ax.text(
                bar.get_x() + bar.get_width() / 2, ypos,
                label_str,
                ha="center", va="bottom" if val >= 0 else "top",
                fontsize=7, color=color, fontweight="bold",
            )

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
# 19 — Weighting scheme comparison
# ---------------------------------------------------------------------------

def plot_weighting_scheme_comparison(
    results_dict: Dict[str, Dict[str, float]],
    output_dir: str,
    filename: str = "19_weighting_comparison.png",
) -> Path:
    """Side-by-side bar chart comparing equal vs inverse-vol weighting schemes.

    Parameters
    ----------
    results_dict : Dict[str, Dict[str, float]]
        {"Equal Weight": metrics_dict, "Inverse Vol": metrics_dict}
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
        bars = ax.bar(x + (i - 0.5) * width, vals, width=width * 0.9,
                      label=scheme, color=color, alpha=0.85)
        # Data label: percentages for drawdown-type metrics, two-decimal for ratios.
        for bar, metric, val in zip(bars, metrics_to_show, vals):
            if np.isnan(val):
                continue
            is_pct = "Drawdown" in metric
            label_str = f"{val * 100:.1f}%" if is_pct else f"{val:.2f}"
            sign = 1 if val >= 0 else -1
            ypos = val + sign * (0.005 if is_pct else 0.02)
            ax.text(
                bar.get_x() + bar.get_width() / 2, ypos,
                label_str,
                ha="center", va="bottom" if val >= 0 else "top",
                fontsize=8, color=color, fontweight="bold",
            )

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
# 20 — Transaction cost scenario equity curves
# ---------------------------------------------------------------------------

def plot_cost_scenarios_equity(
    equity_curves: Dict[str, pd.Series],
    output_dir: str,
    filename: str = "20_cost_scenarios.png",
) -> Path:
    """Equity curves at 0, 5, and 10 bps transaction cost scenarios.

    The 5 bps series (the model's base cost assumption) is drawn with a
    thicker line to distinguish it as the primary scenario.

    Parameters
    ----------
    equity_curves : Dict[str, pd.Series]
        {"0 bps": ..., "5 bps": ..., "10 bps": ...} equity series.
    """
    out = _ensure_dir(output_dir)
    palette = [POSITIVE_COLOR, AMAAM_COLOR, NEGATIVE_COLOR]

    fig, ax = plt.subplots(figsize=FIGURE_SIZE, dpi=FIGURE_DPI)
    for (label, series), color in zip(equity_curves.items(), palette):
        # 5 bps is the model's base case — draw it slightly bolder.
        lw = LINE_WIDTH * 1.4 if "5" in label else LINE_WIDTH
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
# 21 — Transaction cost metrics table
# ---------------------------------------------------------------------------

def plot_cost_scenarios_table(
    metrics_dict: Dict[str, Dict[str, float]],
    output_dir: str,
    filename: str = "21_cost_table.png",
) -> Path:
    """Matplotlib table of key metrics across cost scenarios.

    Parameters
    ----------
    metrics_dict : Dict[str, Dict[str, float]]
        {"0 bps": metrics, "10 bps": metrics, "15 bps": metrics}
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
    tbl.set_fontsize(TICK_SIZE + 1)
    tbl.scale(1.3, 1.8)
    _style_table(tbl)
    ax.set_title("Key Metrics — Transaction Cost Scenarios",
                 fontsize=TITLE_SIZE, pad=20, fontweight="bold")

    plt.tight_layout()
    path = out / filename
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 22 — IS vs OOS equity curves (side by side)
# ---------------------------------------------------------------------------

def plot_is_oos_equity(
    is_result: "BacktestResult",
    oos_result: "BacktestResult",
    output_dir: str,
    filename: str = "22_is_oos_equity.png",
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
# 23 — IS vs OOS metrics table
# ---------------------------------------------------------------------------

def plot_is_oos_stats_table(
    is_result: "BacktestResult",
    oos_result: "BacktestResult",
    output_dir: str,
    filename: str = "23_is_oos_stats.png",
) -> Path:
    """Side-by-side metrics table for IS vs OOS periods.

    Parameters
    ----------
    is_result : BacktestResult
        In-sample backtest result.
    oos_result : BacktestResult
        Out-of-sample backtest result.
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
        colLabels=["In-Sample (2004–2017)", "Out-of-Sample (2018–2026)"],
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(TICK_SIZE + 1)
    tbl.scale(1.3, 1.8)
    _style_table(tbl)
    ax.set_title("IS vs OOS Performance Metrics",
                 fontsize=TITLE_SIZE, pad=20, fontweight="bold")

    plt.tight_layout()
    path = out / filename
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 24 — Risk–Return scatter
# ---------------------------------------------------------------------------

def plot_risk_return_scatter(
    returns_dict: Dict[str, pd.Series],
    output_dir: str,
    filename: str = "24_risk_return_scatter.png",
) -> Path:
    """Annualised return vs annualised volatility scatter for all strategies.

    Iso-Sharpe reference lines at 0.5 and 1.0 (rf = 2 %) frame where each
    strategy sits on the risk-return plane.  The classic institutional
    presentation — a single chart that quantifies the risk/reward trade-off.

    Parameters
    ----------
    returns_dict : Dict[str, pd.Series]
        Strategy label → monthly return series.
    """
    out = _ensure_dir(output_dir)
    PERIODS = 12
    RF = 0.02  # risk-free rate used for iso-Sharpe lines

    fig, ax = plt.subplots(figsize=(9, 7), dpi=FIGURE_DPI)

    points: Dict[str, tuple] = {}
    for label, rets in returns_dict.items():
        r = rets.dropna()
        if len(r) < PERIODS:
            continue
        ann_ret = float((1 + r).prod() ** (PERIODS / len(r)) - 1)
        ann_vol = float(r.std() * np.sqrt(PERIODS))
        points[label] = (ann_vol, ann_ret)

    # Iso-Sharpe reference lines (Sharpe = (ret - rf) / vol)
    max_vol = max((v for v, _ in points.values()), default=0.25)
    vol_range = np.linspace(0.0, max_vol * 1.35, 300)
    for sharpe, ls, lbl in [(0.5, ":", "Sharpe = 0.5"), (1.0, "--", "Sharpe = 1.0")]:
        ax.plot(vol_range, RF + sharpe * vol_range, color="gray",
                linewidth=0.9, linestyle=ls, label=lbl, zorder=1)

    for label, (ann_vol, ann_ret) in points.items():
        color    = BENCHMARK_COLORS.get(label, NEUTRAL_COLOR)
        is_amaam = (label == "AMAAM")
        ax.scatter(
            ann_vol, ann_ret,
            color=color, s=200 if is_amaam else 120,
            edgecolors="black" if is_amaam else color,
            linewidths=1.8 if is_amaam else 0,
            zorder=3 if is_amaam else 2,
        )
        ax.annotate(
            label, (ann_vol, ann_ret),
            textcoords="offset points", xytext=(10, 5),
            fontsize=TICK_SIZE, color=color,
            fontweight="bold" if is_amaam else "normal",
        )

    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_pct_fmt))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_pct_fmt))
    ax.set_xlabel("Annualised Volatility", fontsize=LABEL_SIZE)
    ax.set_ylabel("Annualised Return", fontsize=LABEL_SIZE)
    ax.set_title("Risk–Return Profile", fontsize=TITLE_SIZE)
    ax.legend(fontsize=TICK_SIZE - 1, loc="upper left")
    ax.tick_params(labelsize=TICK_SIZE)
    ax.grid(True, alpha=0.3, zorder=0)
    ax.axhline(0, color="black", linewidth=0.5, zorder=1)

    plt.tight_layout()
    path = out / filename
    fig.savefig(path, dpi=FIGURE_DPI)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 25 — Monthly return scatter vs SPY (beta plot)
# ---------------------------------------------------------------------------

def plot_beta_scatter(
    returns_dict: Dict[str, pd.Series],
    output_dir: str,
    filename: str = "25_beta_scatter.png",
) -> Path:
    """Scatter of AMAAM monthly returns (y) against SPY monthly returns (x).

    The OLS regression line and its annotation (beta, annualised alpha, R2)
    make the near-zero systematic exposure immediately visible.  A flat
    regression line against the SPY scatter cloud is the most persuasive
    single image for demonstrating genuine alpha orthogonal to equity beta.

    Parameters
    ----------
    returns_dict : Dict[str, pd.Series]
        Strategy label → monthly return series. Requires "AMAAM" and "SPY B&H".
    """
    out = _ensure_dir(output_dir)
    fig, ax = plt.subplots(figsize=(9, 7), dpi=FIGURE_DPI)

    amaam_rets = returns_dict.get("AMAAM")
    spy_rets   = returns_dict.get("SPY B&H")

    if amaam_rets is None or spy_rets is None:
        ax.set_title("Beta Scatter (data unavailable)", fontsize=TITLE_SIZE)
        path = out / filename
        plt.tight_layout()
        fig.savefig(path, dpi=FIGURE_DPI)
        plt.close(fig)
        return path

    df = pd.DataFrame({"AMAAM": amaam_rets, "SPY": spy_rets}).dropna()
    slope, intercept, r_value, _pval, _stderr = sp_stats.linregress(
        df["SPY"].values, df["AMAAM"].values
    )
    ann_alpha = float((1.0 + intercept) ** 12 - 1.0)

    ax.scatter(df["SPY"], df["AMAAM"], color=AMAAM_COLOR, alpha=0.45, s=25, zorder=2)
    x_line = np.linspace(df["SPY"].min(), df["SPY"].max(), 300)
    ax.plot(
        x_line, intercept + slope * x_line,
        color="black", linewidth=1.8, zorder=3,
        label=(
            f"β = {slope:.3f}"
            f"     α = {ann_alpha:+.2%}/yr"
            f"     R\u00b2 = {r_value**2:.3f}"
        ),
    )
    ax.axhline(0, color="gray", linewidth=0.6, linestyle=":")
    ax.axvline(0, color="gray", linewidth=0.6, linestyle=":")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_pct_fmt))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_pct_fmt))
    ax.set_xlabel("SPY Monthly Return", fontsize=LABEL_SIZE)
    ax.set_ylabel("AMAAM Monthly Return", fontsize=LABEL_SIZE)
    ax.set_title("AMAAM vs SPY \u2014 Monthly Return Scatter", fontsize=TITLE_SIZE)
    ax.legend(fontsize=TICK_SIZE, loc="upper left")
    ax.tick_params(labelsize=TICK_SIZE)
    ax.grid(True, alpha=0.3, zorder=0)

    plt.tight_layout()
    path = out / filename
    fig.savefig(path, dpi=FIGURE_DPI)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 26 — Rolling 12m / 24m correlation to SPY
# ---------------------------------------------------------------------------

def plot_rolling_spy_correlation(
    returns_dict: Dict[str, pd.Series],
    output_dir: str,
    filename: str = "26_rolling_spy_correlation.png",
) -> Path:
    """Rolling 12- and 24-month Pearson correlation of AMAAM returns to SPY.

    Confirms that the near-zero full-period beta is a persistent property
    rather than a statistical average masking high-correlation sub-periods.
    Reference bands at +/-0.3 mark a "moderate correlation" threshold.

    Parameters
    ----------
    returns_dict : Dict[str, pd.Series]
        Strategy label → monthly return series. Requires "AMAAM" and "SPY B&H".
    """
    out = _ensure_dir(output_dir)
    fig, ax = plt.subplots(figsize=FIGURE_SIZE, dpi=FIGURE_DPI)

    amaam_rets = returns_dict.get("AMAAM")
    spy_rets   = returns_dict.get("SPY B&H")

    if amaam_rets is None or spy_rets is None:
        ax.set_title("Rolling SPY Correlation (data unavailable)", fontsize=TITLE_SIZE)
        path = out / filename
        plt.tight_layout()
        fig.savefig(path, dpi=FIGURE_DPI)
        plt.close(fig)
        return path

    df     = pd.DataFrame({"AMAAM": amaam_rets, "SPY": spy_rets}).dropna()
    roll12 = df["AMAAM"].rolling(12, min_periods=12).corr(df["SPY"])
    roll24 = df["AMAAM"].rolling(24, min_periods=24).corr(df["SPY"])

    ax.fill_between(roll12.index, roll12, 0, alpha=0.12, color=AMAAM_COLOR)
    ax.plot(roll12.index, roll12, color=AMAAM_COLOR, linewidth=LINE_WIDTH * 1.3,
            label="12m rolling correlation")
    ax.plot(roll24.index, roll24, color=SPY_COLOR, linewidth=LINE_WIDTH,
            linestyle="--", label="24m rolling correlation")

    for level in (0.3, -0.3):
        ax.axhline(level, color="gray", linewidth=0.7, linestyle=":", alpha=0.8)
    ax.axhline(0, color="black", linewidth=1.0, linestyle="--")

    ax.set_ylim(-1.05, 1.05)
    ax.set_title("Rolling Correlation \u2014 AMAAM vs SPY", fontsize=TITLE_SIZE)
    ax.set_ylabel("Pearson Correlation", fontsize=LABEL_SIZE)
    ax.tick_params(labelsize=TICK_SIZE)
    ax.legend(fontsize=TICK_SIZE)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = out / filename
    fig.savefig(path, dpi=FIGURE_DPI)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Helper — drawdown event extraction
# ---------------------------------------------------------------------------

def _find_drawdown_periods(equity: pd.Series) -> list:
    """Extract distinct drawdown events from a normalised equity series.

    Each event is a dict with keys: start, trough, end (None if ongoing),
    depth (negative fraction from peak), months_to_trough, and
    months_to_recovery (None if ongoing).
    """
    high     = float(equity.iloc[0])
    high_idx = equity.index[0]
    in_dd    = False
    trough_v = high
    trough_i = high_idx
    events: list = []

    for dt, val in equity.items():
        val = float(val)
        if val >= high:
            if in_dd:
                events.append({
                    "start":  high_idx,
                    "trough": trough_i,
                    "end":    dt,
                    "depth":  trough_v / high - 1.0,
                    "months_to_trough": (
                        (trough_i.year - high_idx.year) * 12
                        + trough_i.month - high_idx.month
                    ),
                    "months_to_recovery": (
                        (dt.year - high_idx.year) * 12
                        + dt.month - high_idx.month
                    ),
                })
                in_dd = False
            high     = val
            high_idx = dt
            trough_v = val
            trough_i = dt
        else:
            in_dd = True
            if val < trough_v:
                trough_v = val
                trough_i = dt

    if in_dd:  # drawdown ongoing at end of series
        events.append({
            "start":  high_idx,
            "trough": trough_i,
            "end":    None,
            "depth":  trough_v / high - 1.0,
            "months_to_trough": (
                (trough_i.year - high_idx.year) * 12
                + trough_i.month - high_idx.month
            ),
            "months_to_recovery": None,
        })
    return events


# ---------------------------------------------------------------------------
# 27 — Drawdown depth and recovery duration
# ---------------------------------------------------------------------------

def plot_drawdown_duration(
    equity_curves: Dict[str, pd.Series],
    output_dir: str,
    filename: str = "27_drawdown_duration.png",
) -> Path:
    """Two-panel drawdown analysis for AMAAM: depth (left) and duration (right).

    Left: horizontal bars showing peak-to-trough loss for every drawdown
    deeper than 1 %, sorted worst-first.
    Right: months-to-trough (blue) stacked with additional months-to-recovery
    (green), so the total bar length equals the full recovery time.
    Ongoing drawdowns are shown in neutral grey and marked with a star.

    Parameters
    ----------
    equity_curves : Dict[str, pd.Series]
        Strategy label → normalised equity series. "AMAAM" key is used.
    """
    out = _ensure_dir(output_dir)
    amaam_eq = equity_curves.get("AMAAM")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=FIGURE_DPI)

    if amaam_eq is None:
        for ax in (ax1, ax2):
            ax.set_title("Drawdown Duration (data unavailable)", fontsize=TITLE_SIZE)
        path = out / filename
        plt.tight_layout()
        fig.savefig(path, dpi=FIGURE_DPI)
        plt.close(fig)
        return path

    eq     = amaam_eq / amaam_eq.iloc[0]
    events = _find_drawdown_periods(eq)
    events = [e for e in events if e["depth"] < -0.01]          # drop trivial < 1 %
    events = sorted(events, key=lambda e: e["depth"])            # worst first (top)

    labels = [
        e["start"].strftime("%b %Y") + (" \u2605" if e["end"] is None else "")
        for e in events
    ]
    depths        = [e["depth"] for e in events]
    m_trough      = [e["months_to_trough"] for e in events]
    m_recovery    = [
        e["months_to_recovery"] if e["months_to_recovery"] is not None
        else e["months_to_trough"]
        for e in events
    ]
    bar_colors = [
        NEUTRAL_COLOR if e["end"] is None else NEGATIVE_COLOR
        for e in events
    ]
    y = np.arange(len(labels))

    # Left panel — depth
    ax1.barh(y, depths, color=bar_colors, alpha=0.85, edgecolor="white", linewidth=0.4)
    for i, dep in enumerate(depths):
        ax1.text(dep - 0.003, i, f"{dep*100:.1f}%",
                 ha="right", va="center", fontsize=7.5,
                 color="white", fontweight="bold")
    ax1.set_yticks(y)
    ax1.set_yticklabels(labels, fontsize=TICK_SIZE - 1)
    ax1.xaxis.set_major_formatter(mticker.FuncFormatter(_pct_fmt))
    ax1.set_title("Drawdown Depth", fontsize=TITLE_SIZE)
    ax1.set_xlabel("Peak-to-Trough Loss", fontsize=LABEL_SIZE)
    ax1.tick_params(labelsize=TICK_SIZE)
    ax1.grid(True, axis="x", alpha=0.3)

    # Right panel — duration
    extra = [max(0, r - t) for r, t in zip(m_recovery, m_trough)]
    ax2.barh(y, m_trough, color=AMAAM_COLOR, alpha=0.75, label="Months to trough")
    ax2.barh(y, extra, left=m_trough, color=POSITIVE_COLOR, alpha=0.75,
             label="Additional months to full recovery")
    for i, (e, tr, rec) in enumerate(zip(events, m_trough, m_recovery)):
        ongoing = e["end"] is None
        txt = f"{tr}m+" if ongoing else f"{rec}m total"
        ax2.text(rec + 0.3, i, txt, ha="left", va="center",
                 fontsize=7.5, color=NEUTRAL_COLOR if ongoing else "#1A1A1A")
    ax2.set_yticks(y)
    ax2.set_yticklabels(labels, fontsize=TICK_SIZE - 1)
    ax2.set_title("Recovery Duration (months)", fontsize=TITLE_SIZE)
    ax2.set_xlabel("Months from Peak", fontsize=LABEL_SIZE)
    ax2.tick_params(labelsize=TICK_SIZE)
    ax2.legend(fontsize=TICK_SIZE - 1, loc="upper right")
    ax2.grid(True, axis="x", alpha=0.3)

    fig.suptitle("AMAAM Drawdown Events  (\u2605 = ongoing)",
                 fontsize=TITLE_SIZE + 1, fontweight="bold")
    plt.tight_layout()
    path = out / filename
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 28 — Win rate / payoff statistics table
# ---------------------------------------------------------------------------

def plot_win_rate_stats(
    returns_dict: Dict[str, pd.Series],
    output_dir: str,
    filename: str = "28_win_rate_table.png",
) -> Path:
    """Styled table of monthly win-rate and payoff statistics for all strategies.

    Rows: Win Rate, Avg Win, Avg Loss, Payoff Ratio, Best Month, Worst Month,
    Max Consecutive Wins, Max Consecutive Losses, Skewness, Excess Kurtosis.
    Surfaces the asymmetric payoff profile that summary risk-adjusted ratios
    do not convey on their own.

    Parameters
    ----------
    returns_dict : Dict[str, pd.Series]
        Strategy label → monthly return series.
    """
    out = _ensure_dir(output_dir)

    def _max_consec(rets: pd.Series, positive: bool) -> int:
        best = cur = 0
        for v in rets:
            if (v > 0) is positive:
                cur  += 1
                best  = max(best, cur)
            else:
                cur = 0
        return best

    def _build_stats(rets: pd.Series) -> dict:
        r      = rets.dropna()
        wins   = r[r > 0]
        losses = r[r <= 0]
        payoff = (
            wins.mean() / abs(losses.mean())
            if len(losses) > 0 and losses.mean() != 0 else float("nan")
        )
        return {
            "Win Rate":           f"{len(wins)/len(r)*100:.1f}%",
            "Avg Win":            f"{wins.mean()*100:+.2f}%",
            "Avg Loss":           f"{losses.mean()*100:+.2f}%",
            "Payoff Ratio":       f"{payoff:.2f}\u00d7",
            "Best Month":         f"{r.max()*100:+.2f}%",
            "Worst Month":        f"{r.min()*100:+.2f}%",
            "Max Consec. Wins":   str(_max_consec(r, True)),
            "Max Consec. Losses": str(_max_consec(r, False)),
            "Skewness":           f"{float(r.skew()):+.3f}",
            "Excess Kurtosis":    f"{float(r.kurt()):+.3f}",
        }

    stat_keys  = [
        "Win Rate", "Avg Win", "Avg Loss", "Payoff Ratio",
        "Best Month", "Worst Month",
        "Max Consec. Wins", "Max Consec. Losses",
        "Skewness", "Excess Kurtosis",
    ]
    strategies = list(returns_dict.keys())
    all_stats  = {lbl: _build_stats(r) for lbl, r in returns_dict.items()}
    cell_data  = [[all_stats[s][k] for s in strategies] for k in stat_keys]

    fig, ax = plt.subplots(figsize=(13, 5), dpi=FIGURE_DPI)
    ax.axis("off")
    tbl = ax.table(
        cellText=cell_data,
        rowLabels=stat_keys,
        colLabels=strategies,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(TICK_SIZE + 1)
    tbl.scale(1.3, 1.7)
    _style_table(tbl)
    ax.set_title("Monthly Win Rate & Payoff Statistics",
                 fontsize=TITLE_SIZE, pad=20, fontweight="bold")

    plt.tight_layout()
    path = out / filename
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 29 — Historical VaR and CVaR comparison
# ---------------------------------------------------------------------------

def plot_var_cvar(
    returns_dict: Dict[str, pd.Series],
    output_dir: str,
    filename: str = "29_var_cvar.png",
) -> Path:
    """Historical VaR and CVaR (Expected Shortfall) at 95% and 99%.

    All values are expressed as positive loss magnitudes so that a shorter
    bar indicates less tail risk.  Strategy-consistent colours and per-bar
    data labels enable direct numerical comparison across scenarios.

    Parameters
    ----------
    returns_dict : Dict[str, pd.Series]
        Strategy label → monthly return series.
    """
    out = _ensure_dir(output_dir)

    def _vc(rets: pd.Series) -> dict:
        r  = rets.dropna().values
        q5 = np.percentile(r, 5)
        q1 = np.percentile(r, 1)
        return {
            "VaR 95%":  float(-q5),
            "CVaR 95%": float(-r[r <= q5].mean()),
            "VaR 99%":  float(-q1),
            "CVaR 99%": float(-r[r <= q1].mean()),
        }

    metric_keys = ["VaR 95%", "CVaR 95%", "VaR 99%", "CVaR 99%"]
    strategies  = list(returns_dict.keys())
    all_vc      = {lbl: _vc(r) for lbl, r in returns_dict.items()}

    n_strats = len(strategies)
    width    = 0.8 / max(n_strats, 1)
    x        = np.arange(len(metric_keys))
    offsets  = np.linspace(-(n_strats - 1) * width / 2,
                            (n_strats - 1) * width / 2, n_strats)

    fig, ax = plt.subplots(figsize=FIGURE_SIZE, dpi=FIGURE_DPI)
    for strat, offset in zip(strategies, offsets):
        color    = BENCHMARK_COLORS.get(strat, NEUTRAL_COLOR)
        is_amaam = (strat == "AMAAM")
        vals     = [all_vc[strat][m] for m in metric_keys]
        bars = ax.bar(
            x + offset, vals, width=width * 0.9,
            label=strat, color=color, alpha=0.85,
            edgecolor="black" if is_amaam else color,
            linewidth=1.3 if is_amaam else 0.3,
            zorder=3 if is_amaam else 2,
        )
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2, val + 0.001,
                f"{val*100:.1f}%",
                ha="center", va="bottom", fontsize=7,
                color=color, fontweight="bold" if is_amaam else "normal",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(metric_keys, fontsize=TICK_SIZE)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_pct_fmt))
    ax.set_title(
        "Historical VaR & CVaR  (monthly, positive = loss magnitude)",
        fontsize=TITLE_SIZE,
    )
    ax.set_ylabel("Loss Magnitude", fontsize=LABEL_SIZE)
    ax.tick_params(labelsize=TICK_SIZE)
    ax.legend(fontsize=TICK_SIZE)
    ax.grid(True, axis="y", alpha=0.3, zorder=0)

    plt.tight_layout()
    path = out / filename
    fig.savefig(path, dpi=FIGURE_DPI)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 30 — Rolling Calmar ratio
# ---------------------------------------------------------------------------

def plot_rolling_calmar(
    returns_dict: Dict[str, pd.Series],
    output_dir: str,
    window: int = 36,
    filename: str = "30_rolling_calmar.png",
) -> Path:
    """Rolling Calmar ratio (annualised return / |max drawdown|) over time.

    A 36-month trailing window is the industry standard — long enough to
    capture a meaningful drawdown, short enough to reflect regime changes.
    A horizontal reference at Calmar = 1.0 marks the conventional threshold
    separating adequate from strong risk-adjusted performance.

    Parameters
    ----------
    returns_dict : Dict[str, pd.Series]
        Strategy label → monthly return series.
    window : int
        Trailing window in months (default 36).
    """
    out = _ensure_dir(output_dir)
    fig, ax = plt.subplots(figsize=FIGURE_SIZE, dpi=FIGURE_DPI)

    for label, rets in returns_dict.items():
        r = rets.dropna()
        if len(r) < window:
            continue
        calmars: list = []
        idx_list: list = []
        for i in range(window, len(r) + 1):
            w       = r.iloc[i - window: i]
            ann_ret = float((1 + w).prod() ** (12.0 / window) - 1)
            eq_w    = (1 + w).cumprod()
            mdd     = float((eq_w / eq_w.cummax() - 1).min())
            calmar  = ann_ret / abs(mdd) if mdd < -1e-6 else float("nan")
            calmars.append(calmar)
            idx_list.append(r.index[i - 1])

        s     = pd.Series(calmars, index=idx_list)
        color = BENCHMARK_COLORS.get(label, NEUTRAL_COLOR)
        lw    = LINE_WIDTH * 1.5 if label == "AMAAM" else LINE_WIDTH
        ax.plot(s.index, s, label=label, color=color, linewidth=lw)

    ax.axhline(0,   color="black", linewidth=0.9, linestyle="--")
    ax.axhline(1.0, color="gray",  linewidth=0.7, linestyle=":",
               label="Calmar = 1.0")
    ax.set_title(f"Rolling {window}-Month Calmar Ratio", fontsize=TITLE_SIZE)
    ax.set_ylabel("Calmar Ratio  (Ann. Return / |Max DD|)", fontsize=LABEL_SIZE)
    ax.tick_params(labelsize=TICK_SIZE)
    ax.legend(fontsize=TICK_SIZE)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = out / filename
    fig.savefig(path, dpi=FIGURE_DPI)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 31 — Return autocorrelation (ACF)
# ---------------------------------------------------------------------------

def plot_return_autocorrelation(
    returns_dict: Dict[str, pd.Series],
    output_dir: str,
    nlags: int = 24,
    filename: str = "31_return_acf.png",
) -> Path:
    """Autocorrelation function (ACF) of monthly returns — AMAAM and SPY.

    Confidence bands at +/-1.96/sqrt(N) (Bartlett white-noise approximation)
    are shaded.  Bars within the band are consistent with serially independent
    returns, supporting the absence of look-ahead bias and ruling out stale-
    price momentum artifacts from the monthly rebalancing cycle.

    Parameters
    ----------
    returns_dict : Dict[str, pd.Series]
        Strategy label → monthly return series.
    nlags : int
        Number of lags to display (default 24 months).
    """
    out  = _ensure_dir(output_dir)
    keys = [k for k in ("AMAAM", "SPY B&H") if k in returns_dict]
    lags = np.arange(1, nlags + 1)

    fig, axes = plt.subplots(
        1, len(keys),
        figsize=(7 * len(keys), 5),
        dpi=FIGURE_DPI,
        sharey=True,
    )
    if len(keys) == 1:
        axes = [axes]

    for ax, label in zip(axes, keys):
        r     = returns_dict[label].dropna()
        n     = len(r)
        conf  = 1.96 / np.sqrt(n)
        color = BENCHMARK_COLORS.get(label, NEUTRAL_COLOR)

        # Compute ACF manually — no statsmodels dependency
        mu  = float(r.mean())
        var = float(r.var())
        acf_vals = np.array([
            float(((r.iloc[:-lag] - mu) * (r.iloc[lag:].values - mu)).mean() / var)
            for lag in lags
        ])

        ax.bar(lags, acf_vals, color=color, alpha=0.72, width=0.65)
        ax.fill_between([0.5, nlags + 0.5], -conf, conf,
                        color="red", alpha=0.08, zorder=0)
        ax.axhline(conf,  color="red", linewidth=1.1, linestyle="--",
                   label=f"95% CI  (\u00b1{conf:.3f})")
        ax.axhline(-conf, color="red", linewidth=1.1, linestyle="--")
        ax.axhline(0,     color="black", linewidth=0.8)

        ax.set_title(f"{label}  \u2014  Return ACF", fontsize=TITLE_SIZE)
        ax.set_xlabel("Lag (months)", fontsize=LABEL_SIZE)
        if ax is axes[0]:
            ax.set_ylabel("Autocorrelation", fontsize=LABEL_SIZE)
        ax.set_xticks(lags[::2])
        ax.set_xlim(0.5, nlags + 0.5)
        ax.tick_params(labelsize=TICK_SIZE)
        ax.legend(fontsize=TICK_SIZE - 1)
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Monthly Return Autocorrelation (ACF)",
                 fontsize=TITLE_SIZE + 1, fontweight="bold")
    plt.tight_layout()
    path = out / filename
    fig.savefig(path, dpi=FIGURE_DPI)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 32 — Walk-forward validation
# ---------------------------------------------------------------------------

def plot_walk_forward(
    fold_df: pd.DataFrame,
    stacked_returns: Dict[str, pd.Series],
    output_dir: str,
    filename: str = "32_walk_forward.png",
) -> Path:
    """Two-panel walk-forward validation chart.

    Left panel: per-fold Sharpe ratio for the canonical config (wM=0.65) and
    the equal-weight baseline, with data labels.  A zero-reference line
    separates positive from negative periods.

    Right panel: stacked OOS equity curves — each fold's test window
    concatenated back-to-back starting at $100, with fold boundaries
    marked as dashed verticals.

    Parameters
    ----------
    fold_df : pd.DataFrame
        One row per fold with columns: fold, test_start, test_end,
        candidate_sr, baseline_sr.
    stacked_returns : Dict[str, pd.Series]
        ``{"AMAAM (canonical)": Series, "Baseline": Series}`` of monthly
        returns covering the concatenated test windows.
    """
    out = _ensure_dir(output_dir)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=FIGURE_DPI)

    # ── Left: per-fold Sharpe bars ───────────────────────────────────────────
    folds    = fold_df["fold"].tolist()
    cand_srs = fold_df["candidate_sr"].tolist()
    base_srs = fold_df["baseline_sr"].tolist()
    x        = np.arange(len(folds))
    bar_w    = 0.35

    bars_cand = ax1.bar(x - bar_w / 2, cand_srs, bar_w,
                        label="AMAAM (canonical)", color=AMAAM_COLOR, alpha=0.88,
                        edgecolor="black", linewidth=0.8)
    bars_base = ax1.bar(x + bar_w / 2, base_srs, bar_w,
                        label="Baseline (equal-weight)", color=NEUTRAL_COLOR, alpha=0.75,
                        edgecolor="gray", linewidth=0.5)

    for bar, val in list(zip(bars_cand, cand_srs)) + list(zip(bars_base, base_srs)):
        if np.isnan(val):
            continue
        sign = 1 if val >= 0 else -1
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            val + sign * 0.04,
            f"{val:.2f}",
            ha="center", va="bottom" if val >= 0 else "top",
            fontsize=7.5, fontweight="bold",
            color=bar.get_facecolor(),
        )

    ax1.axhline(0, color="black", linewidth=0.9, linestyle="--")
    ax1.set_xticks(x)
    ax1.set_xticklabels(
        [f"{r['fold']}\n{r['test_start'][:7]}–{r['test_end'][:7]}"
         for _, r in fold_df.iterrows()],
        fontsize=TICK_SIZE - 1,
    )
    ax1.set_title("Per-Fold OOS Sharpe Ratio", fontsize=TITLE_SIZE)
    ax1.set_ylabel("Sharpe Ratio (rf = 2%)", fontsize=LABEL_SIZE)
    ax1.tick_params(labelsize=TICK_SIZE)
    ax1.legend(fontsize=TICK_SIZE - 1)
    ax1.grid(True, axis="y", alpha=0.3)

    # ── Right: stacked OOS equity curves ─────────────────────────────────────
    palette = {"AMAAM (canonical)": AMAAM_COLOR, "Baseline": NEUTRAL_COLOR}
    lws     = {"AMAAM (canonical)": LINE_WIDTH * 1.5, "Baseline": LINE_WIDTH}

    for label, rets in stacked_returns.items():
        eq = (1 + rets.sort_index()).cumprod() * 100
        ax2.plot(eq.index, eq.values,
                 color=palette.get(label, NEUTRAL_COLOR),
                 linewidth=lws.get(label, LINE_WIDTH),
                 label=label)

    # Fold boundary verticals + fold labels
    y_bot = ax2.get_ylim()[0]
    for i, (_, row) in enumerate(fold_df.iterrows()):
        ax2.axvline(pd.Timestamp(row["test_start"]),
                    color="gray", linewidth=0.7, linestyle=":", alpha=0.6)
        ax2.text(
            pd.Timestamp(row["test_start"]), y_bot,
            f" F{i+1}", fontsize=6.5, color="gray", va="bottom",
        )

    ax2.axhline(100, color="black", linewidth=0.6, linestyle="--", alpha=0.4)
    ax2.set_title("Stacked OOS Equity Curves (Growth of $100)", fontsize=TITLE_SIZE)
    ax2.set_ylabel("Portfolio Value ($)", fontsize=LABEL_SIZE)
    ax2.tick_params(labelsize=TICK_SIZE)
    ax2.legend(fontsize=TICK_SIZE - 1)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(
        "Walk-Forward Validation  \u2014  6 Folds, Expanding Training Window, 2-Year Test Windows",
        fontsize=TITLE_SIZE + 1, fontweight="bold",
    )
    plt.tight_layout()
    path = out / filename
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    return path
    return path
