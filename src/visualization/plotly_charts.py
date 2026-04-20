"""
Interactive Plotly chart library for AMAAM.

Mirrors matplotlib_charts.py but produces interactive HTML files saved to
reports/interactive/. Intended for personal analysis where hovering, zooming,
and toggling series are useful. Chart function signatures match their Matplotlib
counterparts to allow shared calling code in generate_reports.py. See Section
9.19 of the specification.
"""

import os
from pathlib import Path
from typing import Dict, List, TYPE_CHECKING

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats as sp_stats

from src.backtest.metrics import compute_rolling_metrics

if TYPE_CHECKING:
    from config.default_config import ModelConfig
    from src.backtest.engine import BacktestResult

# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------

COLORS: Dict[str, str] = {
    "AMAAM":   "#2196F3",
    "SPY B&H": "#F44336",
    "60/40":   "#4CAF50",
    "7Twelve": "#9C27B0",
}
POSITIVE_COLOR = "#4CAF50"
NEGATIVE_COLOR = "#F44336"
NEUTRAL_COLOR  = "#9E9E9E"
TEMPLATE       = "plotly_white"

# IS/OOS split date from Section 5.2.
_IS_OOS_SPLIT = "2018-01-01"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ensure_dir(output_dir: str) -> Path:
    """Create *output_dir* if it does not exist and return it as a Path."""
    p = Path(output_dir)
    os.makedirs(p, exist_ok=True)
    return p


def _annual_returns(monthly: pd.Series) -> pd.Series:
    """Compound monthly returns into calendar-year returns."""
    return monthly.groupby(monthly.index.year).apply(lambda r: (1 + r).prod() - 1)


def _color_for(label: str) -> str:
    """Return the configured color for *label* or a neutral fallback."""
    return COLORS.get(label, NEUTRAL_COLOR)


# ---------------------------------------------------------------------------
# 01 — Equity curves (log scale)
# ---------------------------------------------------------------------------

def plot_equity_curves(
    equity_curves: Dict[str, pd.Series],
    output_dir: str,
    filename: str = "01_equity_curves.html",
) -> Path:
    """Log-scale cumulative equity curves for AMAAM and all benchmarks.

    Parameters
    ----------
    equity_curves : Dict[str, pd.Series]
        Strategy label → equity series (normalised to 1.0).
    output_dir : str
        Directory where the HTML is saved.
    filename : str
        Output file name.

    Returns
    -------
    Path
        Absolute path to the saved HTML.
    """
    out = _ensure_dir(output_dir)
    fig = go.Figure()
    for label, series in equity_curves.items():
        width = 3 if label == "AMAAM" else 1.5
        fig.add_trace(go.Scatter(
            x=series.index, y=series.values, name=label,
            line=dict(color=_color_for(label), width=width),
        ))
    # add_vline with annotation_text fails on string datetime axes in Plotly;
    # use add_shape + add_annotation separately to avoid the bug.
    fig.add_shape(type="line", xref="x", yref="paper",
                  x0=_IS_OOS_SPLIT, x1=_IS_OOS_SPLIT, y0=0, y1=1,
                  line=dict(color="gray", dash="dash"))
    fig.add_annotation(x=_IS_OOS_SPLIT, yref="paper", y=0.97,
                       text="IS/OOS", showarrow=False, font=dict(color="gray"))
    fig.update_layout(
        template=TEMPLATE, title="Cumulative Performance (Log Scale)",
        xaxis_title="Date", yaxis_title="Portfolio Value",
        yaxis_type="log", hovermode="x unified",
    )
    path = out / filename
    fig.write_html(str(path))
    return path


# ---------------------------------------------------------------------------
# 02 — Drawdown over time
# ---------------------------------------------------------------------------

def plot_drawdowns(
    equity_curves: Dict[str, pd.Series],
    output_dir: str,
    filename: str = "02_drawdowns.html",
) -> Path:
    """Peak-to-trough drawdown over time for all strategies.

    Parameters
    ----------
    equity_curves : Dict[str, pd.Series]
        Strategy label → equity series (normalised to 1.0).
    output_dir : str
        Directory where the HTML is saved.
    filename : str
        Output file name.

    Returns
    -------
    Path
        Absolute path to the saved HTML.
    """
    out = _ensure_dir(output_dir)
    fig = go.Figure()
    for label, series in equity_curves.items():
        dd = series / series.cummax() - 1
        width = 2.5 if label == "AMAAM" else 1.5
        fig.add_trace(go.Scatter(
            x=dd.index, y=dd.values * 100, name=label,
            line=dict(color=_color_for(label), width=width),
            fill="tozeroy" if label == "AMAAM" else None,
            fillcolor="rgba(33,150,243,0.1)" if label == "AMAAM" else None,
        ))
    fig.update_layout(
        template=TEMPLATE, title="Drawdown Over Time",
        xaxis_title="Date", yaxis_title="Drawdown (%)", hovermode="x unified",
    )
    path = out / filename
    fig.write_html(str(path))
    return path


# ---------------------------------------------------------------------------
# 03 — Monthly return heatmap
# ---------------------------------------------------------------------------

def plot_monthly_return_heatmap(
    monthly_returns: pd.Series,
    output_dir: str,
    filename: str = "03_monthly_heatmap.html",
) -> Path:
    """Year x month heatmap of AMAAM monthly returns.

    Parameters
    ----------
    monthly_returns : pd.Series
        AMAAM monthly return series.
    output_dir : str
        Directory where the HTML is saved.
    filename : str
        Output file name.

    Returns
    -------
    Path
        Absolute path to the saved HTML.
    """
    out = _ensure_dir(output_dir)
    pivot = monthly_returns.groupby([monthly_returns.index.year,
                                     monthly_returns.index.month]).mean().unstack()
    month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    months = [month_names[c - 1] for c in pivot.columns]
    years  = [str(y) for y in pivot.index]

    text_vals = [[f"{v*100:.1f}%" if not np.isnan(v) else ""
                  for v in row] for row in pivot.values]
    fig = go.Figure(go.Heatmap(
        z=pivot.values * 100, x=months, y=years,
        colorscale="RdYlGn", zmid=0,
        text=text_vals, texttemplate="%{text}",
        colorbar=dict(title="Return (%)"),
    ))
    fig.update_layout(
        template=TEMPLATE, title="AMAAM Monthly Returns Heatmap",
        xaxis_title="Month", yaxis_title="Year",
    )
    path = out / filename
    fig.write_html(str(path))
    return path


# ---------------------------------------------------------------------------
# 04 — Annual returns grouped bar
# ---------------------------------------------------------------------------

def plot_annual_returns(
    returns_dict: Dict[str, pd.Series],
    output_dir: str,
    filename: str = "04_annual_returns.html",
) -> Path:
    """Grouped bar chart of calendar-year returns for AMAAM and benchmarks.

    Parameters
    ----------
    returns_dict : Dict[str, pd.Series]
        Strategy label → monthly return series.
    output_dir : str
        Directory where the HTML is saved.
    filename : str
        Output file name.

    Returns
    -------
    Path
        Absolute path to the saved HTML.
    """
    out = _ensure_dir(output_dir)
    fig = go.Figure()
    for label, rets in returns_dict.items():
        annual = _annual_returns(rets)
        fig.add_trace(go.Bar(
            x=annual.index.astype(str), y=annual.values * 100,
            name=label, marker_color=_color_for(label),
        ))
    fig.update_layout(
        template=TEMPLATE, title="Calendar-Year Returns",
        xaxis_title="Year", yaxis_title="Return (%)",
        barmode="group", hovermode="x unified",
    )
    path = out / filename
    fig.write_html(str(path))
    return path


# ---------------------------------------------------------------------------
# 05 — Rolling 12-month returns
# ---------------------------------------------------------------------------

def plot_rolling_returns(
    returns_dict: Dict[str, pd.Series],
    output_dir: str,
    filename: str = "05_rolling_returns.html",
) -> Path:
    """Rolling 12-month return for all strategies.

    Parameters
    ----------
    returns_dict : Dict[str, pd.Series]
        Strategy label → monthly return series.
    output_dir : str
        Directory where the HTML is saved.
    filename : str
        Output file name.

    Returns
    -------
    Path
        Absolute path to the saved HTML.
    """
    out = _ensure_dir(output_dir)
    fig = go.Figure()
    for label, rets in returns_dict.items():
        rolling = ((1 + rets).rolling(12, min_periods=12).apply(np.prod, raw=True) - 1) * 100
        fig.add_trace(go.Scatter(
            x=rolling.index, y=rolling.values, name=label,
            line=dict(color=_color_for(label), width=2.5 if label == "AMAAM" else 1.5),
        ))
    fig.update_layout(
        template=TEMPLATE, title="Rolling 12-Month Return",
        xaxis_title="Date", yaxis_title="Return (%)", hovermode="x unified",
    )
    path = out / filename
    fig.write_html(str(path))
    return path


# ---------------------------------------------------------------------------
# 06 — Rolling Sharpe
# ---------------------------------------------------------------------------

def plot_rolling_sharpe(
    returns_dict: Dict[str, pd.Series],
    output_dir: str,
    filename: str = "06_rolling_sharpe.html",
) -> Path:
    """Rolling 12-month Sharpe ratio over time.

    Parameters
    ----------
    returns_dict : Dict[str, pd.Series]
        Strategy label → monthly return series.
    output_dir : str
        Directory where the HTML is saved.
    filename : str
        Output file name.

    Returns
    -------
    Path
        Absolute path to the saved HTML.
    """
    out = _ensure_dir(output_dir)
    fig = go.Figure()
    for label, rets in returns_dict.items():
        rm = compute_rolling_metrics(rets, window=12)
        fig.add_trace(go.Scatter(
            x=rm.index, y=rm["rolling_sharpe"], name=label,
            line=dict(color=_color_for(label), width=2.5 if label == "AMAAM" else 1.5),
        ))
    fig.add_hline(y=0, line=dict(color="black", dash="dash"))
    fig.update_layout(
        template=TEMPLATE, title="Rolling 12-Month Sharpe Ratio",
        xaxis_title="Date", yaxis_title="Sharpe Ratio", hovermode="x unified",
    )
    path = out / filename
    fig.write_html(str(path))
    return path


# ---------------------------------------------------------------------------
# 07 — Rolling volatility
# ---------------------------------------------------------------------------

def plot_rolling_volatility(
    returns_dict: Dict[str, pd.Series],
    output_dir: str,
    filename: str = "07_rolling_vol.html",
) -> Path:
    """Rolling 12-month annualised volatility.

    Parameters
    ----------
    returns_dict : Dict[str, pd.Series]
        Strategy label → monthly return series.
    output_dir : str
        Directory where the HTML is saved.
    filename : str
        Output file name.

    Returns
    -------
    Path
        Absolute path to the saved HTML.
    """
    out = _ensure_dir(output_dir)
    fig = go.Figure()
    for label, rets in returns_dict.items():
        rm = compute_rolling_metrics(rets, window=12)
        fig.add_trace(go.Scatter(
            x=rm.index, y=rm["rolling_vol"] * 100, name=label,
            line=dict(color=_color_for(label), width=2.5 if label == "AMAAM" else 1.5),
        ))
    fig.update_layout(
        template=TEMPLATE, title="Rolling 12-Month Annualised Volatility",
        xaxis_title="Date", yaxis_title="Volatility (%)", hovermode="x unified",
    )
    path = out / filename
    fig.write_html(str(path))
    return path


# ---------------------------------------------------------------------------
# 08 — Rolling drawdown
# ---------------------------------------------------------------------------

def plot_rolling_drawdown(
    equity_curves: Dict[str, pd.Series],
    output_dir: str,
    filename: str = "08_rolling_drawdown.html",
) -> Path:
    """Rolling 12-month worst drawdown (trailing window).

    Parameters
    ----------
    equity_curves : Dict[str, pd.Series]
        Strategy label → equity series (normalised to 1.0).
    output_dir : str
        Directory where the HTML is saved.
    filename : str
        Output file name.

    Returns
    -------
    Path
        Absolute path to the saved HTML.
    """
    out = _ensure_dir(output_dir)
    fig = go.Figure()
    for label, series in equity_curves.items():
        dd = series / series.cummax() - 1
        rolling_worst = dd.rolling(12, min_periods=12).min() * 100
        fig.add_trace(go.Scatter(
            x=rolling_worst.index, y=rolling_worst.values, name=label,
            line=dict(color=_color_for(label), width=2.5 if label == "AMAAM" else 1.5),
        ))
    fig.update_layout(
        template=TEMPLATE, title="Rolling 12-Month Worst Drawdown",
        xaxis_title="Date", yaxis_title="Drawdown (%)", hovermode="x unified",
    )
    path = out / filename
    fig.write_html(str(path))
    return path


# ---------------------------------------------------------------------------
# 09 — Return distribution
# ---------------------------------------------------------------------------

def plot_return_distribution(
    returns_dict: Dict[str, pd.Series],
    output_dir: str,
    filename: str = "09_return_distribution.html",
) -> Path:
    """Histogram of AMAAM monthly returns with normal distribution overlay.

    Parameters
    ----------
    returns_dict : Dict[str, pd.Series]
        Strategy label → monthly return series (AMAAM key required).
    output_dir : str
        Directory where the HTML is saved.
    filename : str
        Output file name.

    Returns
    -------
    Path
        Absolute path to the saved HTML.
    """
    out = _ensure_dir(output_dir)
    rets = returns_dict.get("AMAAM", next(iter(returns_dict.values()))).dropna()
    mu, sigma = rets.mean(), rets.std()
    x_norm = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 200)

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=rets.values * 100, name="Monthly returns",
        marker_color=COLORS["AMAAM"], opacity=0.6,
        histnorm="probability density",
    ))
    fig.add_trace(go.Scatter(
        x=x_norm * 100, y=sp_stats.norm.pdf(x_norm, mu, sigma) / 100,
        name="Normal fit", line=dict(color="black", dash="dash", width=2),
    ))
    fig.add_vline(x=0, line=dict(color=NEGATIVE_COLOR, dash="dot"))
    fig.update_layout(
        template=TEMPLATE, title="AMAAM Monthly Return Distribution",
        xaxis_title="Monthly Return (%)", yaxis_title="Density",
    )
    path = out / filename
    fig.write_html(str(path))
    return path


# ---------------------------------------------------------------------------
# 10 — Main sleeve allocation
# ---------------------------------------------------------------------------

def plot_main_sleeve_allocation(
    allocations: pd.DataFrame,
    main_tickers: List[str],
    output_dir: str,
    filename: str = "10_main_sleeve.html",
) -> Path:
    """Stacked area chart of main sleeve holdings over time.

    Parameters
    ----------
    allocations : pd.DataFrame
        Weight history (index=signal dates, columns=tickers).
    main_tickers : List[str]
        Main sleeve tickers to include.
    output_dir : str
        Directory where the HTML is saved.
    filename : str
        Output file name.

    Returns
    -------
    Path
        Absolute path to the saved HTML.
    """
    out = _ensure_dir(output_dir)
    cols = [t for t in main_tickers if t in allocations.columns]
    data = allocations[cols].fillna(0) if cols else pd.DataFrame()

    fig = go.Figure()
    for col in cols:
        fig.add_trace(go.Scatter(
            x=data.index, y=data[col].values * 100,
            name=col, mode="lines", stackgroup="one",
        ))
    fig.update_layout(
        template=TEMPLATE, title="Main Sleeve Allocation Over Time",
        xaxis_title="Date", yaxis_title="Weight (%)", hovermode="x unified",
    )
    path = out / filename
    fig.write_html(str(path))
    return path


# ---------------------------------------------------------------------------
# 11 — Hedging sleeve allocation
# ---------------------------------------------------------------------------

def plot_hedging_sleeve_allocation(
    allocations: pd.DataFrame,
    hedge_tickers: List[str],
    output_dir: str,
    filename: str = "11_hedging_sleeve.html",
) -> Path:
    """Stacked area chart of hedging sleeve holdings over time.

    Parameters
    ----------
    allocations : pd.DataFrame
        Weight history (index=signal dates, columns=tickers).
    hedge_tickers : List[str]
        Hedging sleeve tickers to include.
    output_dir : str
        Directory where the HTML is saved.
    filename : str
        Output file name.

    Returns
    -------
    Path
        Absolute path to the saved HTML.
    """
    out = _ensure_dir(output_dir)
    cols = [t for t in hedge_tickers if t in allocations.columns]
    data = allocations[cols].fillna(0) if cols else pd.DataFrame()

    fig = go.Figure()
    for col in cols:
        fig.add_trace(go.Scatter(
            x=data.index, y=data[col].values * 100,
            name=col, mode="lines", stackgroup="one",
        ))
    fig.update_layout(
        template=TEMPLATE, title="Hedging Sleeve Allocation Over Time",
        xaxis_title="Date", yaxis_title="Weight (%)", hovermode="x unified",
    )
    path = out / filename
    fig.write_html(str(path))
    return path


# ---------------------------------------------------------------------------
# 12 — Total hedging weight
# ---------------------------------------------------------------------------

def plot_hedging_weight_over_time(
    allocations: pd.DataFrame,
    hedge_tickers: List[str],
    output_dir: str,
    filename: str = "12_hedging_weight.html",
) -> Path:
    """Line chart of total hedging sleeve allocation over time.

    Parameters
    ----------
    allocations : pd.DataFrame
        Weight history (index=signal dates, columns=tickers).
    hedge_tickers : List[str]
        Hedging sleeve tickers to sum.
    output_dir : str
        Directory where the HTML is saved.
    filename : str
        Output file name.

    Returns
    -------
    Path
        Absolute path to the saved HTML.
    """
    out = _ensure_dir(output_dir)
    cols = [t for t in hedge_tickers if t in allocations.columns]
    hedge_total = allocations[cols].fillna(0).sum(axis=1) if cols else pd.Series(dtype=float)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hedge_total.index, y=hedge_total.values * 100,
        name="Hedging Weight", line=dict(color=COLORS["SPY B&H"], width=2),
        fill="tozeroy", fillcolor="rgba(244,67,54,0.1)",
    ))
    fig.update_layout(
        template=TEMPLATE, title="Total Hedging Sleeve Weight Over Time",
        xaxis_title="Date", yaxis_title="Hedging Weight (%)", hovermode="x unified",
    )
    path = out / filename
    fig.write_html(str(path))
    return path


# ---------------------------------------------------------------------------
# 13 — Turnover
# ---------------------------------------------------------------------------

def plot_turnover(
    turnover: pd.Series,
    output_dir: str,
    filename: str = "13_turnover.html",
) -> Path:
    """Monthly turnover bar chart with rolling 12-month average overlay.

    Parameters
    ----------
    turnover : pd.Series
        Per-period portfolio turnover values.
    output_dir : str
        Directory where the HTML is saved.
    filename : str
        Output file name.

    Returns
    -------
    Path
        Absolute path to the saved HTML.
    """
    out = _ensure_dir(output_dir)
    rolling_avg = turnover.rolling(12, min_periods=1).mean()

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=turnover.index, y=turnover.values * 100,
        name="Turnover", marker_color=COLORS["AMAAM"], opacity=0.6,
    ))
    fig.add_trace(go.Scatter(
        x=rolling_avg.index, y=rolling_avg.values * 100,
        name="12M Rolling Avg", line=dict(color=NEGATIVE_COLOR, width=2),
    ))
    fig.update_layout(
        template=TEMPLATE, title="Portfolio Turnover Over Time",
        xaxis_title="Date", yaxis_title="Turnover (%)", hovermode="x unified",
    )
    path = out / filename
    fig.write_html(str(path))
    return path


# ---------------------------------------------------------------------------
# 14 — Factor weights (static bar)
# ---------------------------------------------------------------------------

def plot_factor_weights(
    config: "ModelConfig",
    output_dir: str,
    filename: str = "14_factor_weights.html",
) -> Path:
    """Static bar chart of the configured TRank factor weights.

    Parameters
    ----------
    config : ModelConfig
        Model configuration containing factor weight fields.
    output_dir : str
        Directory where the HTML is saved.
    filename : str
        Output file name.

    Returns
    -------
    Path
        Absolute path to the saved HTML.
    """
    out = _ensure_dir(output_dir)
    labels = ["wM (Momentum)", "wV (Volatility)", "wC (Correlation)", "wT (Trend)"]
    values = [config.weight_momentum, config.weight_volatility,
              config.weight_correlation, config.weight_trend]
    colors = [COLORS["AMAAM"], COLORS["60/40"], COLORS["7Twelve"], COLORS["SPY B&H"]]

    fig = go.Figure(go.Bar(
        x=labels, y=values,
        marker_color=colors,
        text=[f"{v:.2f}" for v in values], textposition="outside",
    ))
    fig.update_layout(
        template=TEMPLATE,
        title="TRank Factor Weights<br><sub>Fixed weights (no walk-forward optimisation)</sub>",
        xaxis_title="Factor", yaxis_title="Weight / Scale",
    )
    path = out / filename
    fig.write_html(str(path))
    return path


# ---------------------------------------------------------------------------
# 15 — Return decomposition
# ---------------------------------------------------------------------------

def plot_sleeve_return_decomposition(
    monthly_returns: pd.Series,
    allocations: pd.DataFrame,
    main_tickers: List[str],
    hedge_tickers: List[str],
    output_dir: str,
    filename: str = "15_return_decomp.html",
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
        Directory where the HTML is saved.
    filename : str
        Output file name.

    Returns
    -------
    Path
        Absolute path to the saved HTML.
    """
    out = _ensure_dir(output_dir)
    main_cols  = [t for t in main_tickers  if t in allocations.columns]
    hedge_cols = [t for t in hedge_tickers if t in allocations.columns]
    _empty = pd.Series(dtype=float)
    main_w  = allocations[main_cols].fillna(0).sum(axis=1)  if main_cols  else _empty
    hedge_w = allocations[hedge_cols].fillna(0).sum(axis=1) if hedge_cols else _empty

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    if not main_w.empty:
        fig.add_trace(go.Scatter(
            x=main_w.index, y=main_w.values * 100,
            name="Main sleeve", mode="lines", stackgroup="one",
            line=dict(color=COLORS["AMAAM"]),
        ), secondary_y=False)
        fig.add_trace(go.Scatter(
            x=hedge_w.index, y=hedge_w.values * 100,
            name="Hedging sleeve", mode="lines", stackgroup="one",
            line=dict(color=COLORS["SPY B&H"]),
        ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=monthly_returns.index, y=monthly_returns.values * 100,
        name="Monthly return", line=dict(color="black", width=1, dash="dot"),
        opacity=0.6,
    ), secondary_y=True)

    fig.update_layout(
        template=TEMPLATE, title="Sleeve Weight Decomposition Over Time",
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="Weight (%)", secondary_y=False)
    fig.update_yaxes(title_text="Monthly Return (%)", secondary_y=True)
    path = out / filename
    fig.write_html(str(path))
    return path


# ---------------------------------------------------------------------------
# 16 — Correlation matrix
# ---------------------------------------------------------------------------

def plot_correlation_matrix(
    data_dict: Dict[str, pd.DataFrame],
    main_tickers: List[str],
    output_dir: str,
    filename: str = "16_correlation_matrix.html",
) -> Path:
    """Full-period correlation matrix of main sleeve ETFs.

    Parameters
    ----------
    data_dict : Dict[str, pd.DataFrame]
        Processed OHLCV data keyed by ticker.
    main_tickers : List[str]
        Main sleeve tickers to include.
    output_dir : str
        Directory where the HTML is saved.
    filename : str
        Output file name.

    Returns
    -------
    Path
        Absolute path to the saved HTML.
    """
    out = _ensure_dir(output_dir)
    avail = [t for t in main_tickers if t in data_dict]
    closes = pd.DataFrame({t: data_dict[t]["Close"] for t in avail}).ffill()
    corr = closes.pct_change().dropna().corr()

    text_vals = [[f"{corr.iloc[i, j]:.2f}" for j in range(len(avail))]
                 for i in range(len(avail))]
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=avail, y=avail,
        colorscale="RdYlGn", zmin=-1, zmax=1,
        text=text_vals, texttemplate="%{text}",
        colorbar=dict(title="Correlation"),
    ))
    fig.update_layout(
        template=TEMPLATE, title="Main Sleeve ETF Correlation Matrix",
    )
    path = out / filename
    fig.write_html(str(path))
    return path


# ---------------------------------------------------------------------------
# 17 — Regime performance
# ---------------------------------------------------------------------------

def plot_regime_performance(
    regime_df: pd.DataFrame,
    output_dir: str,
    filename: str = "17_regime_performance.html",
) -> Path:
    """Grouped bar chart of Total Return per strategy per regime.

    Parameters
    ----------
    regime_df : pd.DataFrame
        Output of compute_regime_metrics() with MultiIndex (Strategy, Regime).
    output_dir : str
        Directory where the HTML is saved.
    filename : str
        Output file name.

    Returns
    -------
    Path
        Absolute path to the saved HTML.
    """
    out = _ensure_dir(output_dir)
    fig = go.Figure()
    if not regime_df.empty and "Total Return" in regime_df.columns:
        pivot = regime_df["Total Return"].unstack(level="Strategy")
        for strat in pivot.columns:
            fig.add_trace(go.Bar(
                x=pivot.index.tolist(),
                y=(pivot[strat] * 100).tolist(),
                name=strat,
                marker_color=_color_for(strat),
            ))
    fig.update_layout(
        template=TEMPLATE, title="Total Return During Market Stress Regimes",
        xaxis_title="Regime", yaxis_title="Total Return (%)",
        barmode="group", hovermode="x unified",
    )
    path = out / filename
    fig.write_html(str(path))
    return path


# ---------------------------------------------------------------------------
# 18 — Weight sensitivity
# ---------------------------------------------------------------------------

def plot_weight_sensitivity_heatmap(
    weight_df: pd.DataFrame,
    output_dir: str,
    filename: str = "18_weight_sensitivity.html",
) -> Path:
    """Horizontal bar chart of Sharpe across wM sweep.

    Parameters
    ----------
    weight_df : pd.DataFrame
        Output of run_weight_sensitivity(); index=label, column "Sharpe Ratio".
    output_dir : str
        Directory where the HTML is saved.
    filename : str
        Output file name.

    Returns
    -------
    Path
        Absolute path to the saved HTML.
    """
    out = _ensure_dir(output_dir)
    sharpe = weight_df["Sharpe Ratio"]
    colors = [COLORS["AMAAM"] if "0.65" in idx else NEUTRAL_COLOR for idx in sharpe.index]

    fig = go.Figure(go.Bar(
        x=sharpe.values, y=sharpe.index.tolist(),
        orientation="h", marker_color=colors,
        text=[f"{v:.3f}" for v in sharpe.values], textposition="outside",
    ))
    fig.update_layout(
        template=TEMPLATE, title="Sharpe Ratio vs Momentum Weight (wM)",
        xaxis_title="Sharpe Ratio", yaxis_title="Configuration",
    )
    path = out / filename
    fig.write_html(str(path))
    return path


# ---------------------------------------------------------------------------
# 19 — Selection sensitivity
# ---------------------------------------------------------------------------

def plot_selection_sensitivity(
    selection_df: pd.DataFrame,
    output_dir: str,
    filename: str = "19_selection_sensitivity.html",
) -> Path:
    """Grouped bar chart of key metrics across top-N variants.

    Parameters
    ----------
    selection_df : pd.DataFrame
        Output of run_selection_sensitivity(); index=Top N.
    output_dir : str
        Directory where the HTML is saved.
    filename : str
        Output file name.

    Returns
    -------
    Path
        Absolute path to the saved HTML.
    """
    out = _ensure_dir(output_dir)
    metrics_to_show = ["Sharpe Ratio", "Calmar Ratio", "Max Drawdown"]
    avail = [m for m in metrics_to_show if m in selection_df.columns]
    palette = [COLORS["AMAAM"], COLORS["60/40"], COLORS["SPY B&H"]]

    fig = go.Figure()
    for metric, color in zip(avail, palette):
        fig.add_trace(go.Bar(
            x=[f"Top {n}" for n in selection_df.index],
            y=selection_df[metric].values,
            name=metric, marker_color=color,
        ))
    fig.update_layout(
        template=TEMPLATE, title="Performance vs Main Sleeve Top-N Selection",
        xaxis_title="Top N", barmode="group", hovermode="x unified",
    )
    path = out / filename
    fig.write_html(str(path))
    return path


# ---------------------------------------------------------------------------
# 20 — Weighting scheme comparison
# ---------------------------------------------------------------------------

def plot_weighting_scheme_comparison(
    results_dict: Dict[str, Dict[str, float]],
    output_dir: str,
    filename: str = "20_weighting_comparison.html",
) -> Path:
    """Side-by-side bar chart comparing equal vs inverse-vol weighting schemes.

    Parameters
    ----------
    results_dict : Dict[str, Dict[str, float]]
        {"Equal Weight": metrics_dict, "Inverse Vol": metrics_dict}
    output_dir : str
        Directory where the HTML is saved.
    filename : str
        Output file name.

    Returns
    -------
    Path
        Absolute path to the saved HTML.
    """
    out = _ensure_dir(output_dir)
    metrics_to_show = ["Sharpe Ratio", "Sortino Ratio", "Calmar Ratio", "Max Drawdown"]
    palette = [COLORS["AMAAM"], COLORS["60/40"]]

    fig = go.Figure()
    for (scheme, mets), color in zip(results_dict.items(), palette):
        vals = [mets.get(m, float("nan")) for m in metrics_to_show]
        fig.add_trace(go.Bar(
            x=metrics_to_show, y=vals,
            name=scheme, marker_color=color,
        ))
    fig.update_layout(
        template=TEMPLATE, title="Weighting Scheme Comparison",
        xaxis_title="Metric", barmode="group", hovermode="x unified",
    )
    path = out / filename
    fig.write_html(str(path))
    return path


# ---------------------------------------------------------------------------
# 21 — Transaction cost equity curves
# ---------------------------------------------------------------------------

def plot_cost_scenarios_equity(
    equity_curves: Dict[str, pd.Series],
    output_dir: str,
    filename: str = "21_cost_scenarios.html",
) -> Path:
    """Equity curves at 0, 10, and 15 bps transaction cost scenarios.

    Parameters
    ----------
    equity_curves : Dict[str, pd.Series]
        {"0 bps": ..., "10 bps": ..., "15 bps": ...} equity series.
    output_dir : str
        Directory where the HTML is saved.
    filename : str
        Output file name.

    Returns
    -------
    Path
        Absolute path to the saved HTML.
    """
    out = _ensure_dir(output_dir)
    palette = ["#4CAF50", COLORS["AMAAM"], COLORS["SPY B&H"]]

    fig = go.Figure()
    for (label, series), color in zip(equity_curves.items(), palette):
        width = 2.5 if "10" in label else 1.5
        fig.add_trace(go.Scatter(
            x=series.index, y=series.values, name=label,
            line=dict(color=color, width=width),
        ))
    fig.update_layout(
        template=TEMPLATE, title="Equity Curves — Transaction Cost Scenarios",
        xaxis_title="Date", yaxis_title="Portfolio Value", hovermode="x unified",
    )
    path = out / filename
    fig.write_html(str(path))
    return path


# ---------------------------------------------------------------------------
# 22 — Transaction cost metrics table
# ---------------------------------------------------------------------------

def plot_cost_scenarios_table(
    metrics_dict: Dict[str, Dict[str, float]],
    output_dir: str,
    filename: str = "22_cost_table.html",
) -> Path:
    """Plotly table of key metrics across cost scenarios.

    Parameters
    ----------
    metrics_dict : Dict[str, Dict[str, float]]
        {"0 bps": metrics, "10 bps": metrics, "15 bps": metrics}
    output_dir : str
        Directory where the HTML is saved.
    filename : str
        Output file name.

    Returns
    -------
    Path
        Absolute path to the saved HTML.
    """
    out = _ensure_dir(output_dir)
    row_keys = [
        "Annualized Return", "Annualized Volatility", "Sharpe Ratio",
        "Sortino Ratio", "Calmar Ratio", "Max Drawdown",
        "Best Year", "Worst Year",
    ]
    scenarios = list(metrics_dict.keys())
    pct_keys = {
        "Annualized Return", "Annualized Volatility",
        "Max Drawdown", "Best Year", "Worst Year",
    }

    def _fmt(val: float, key: str) -> str:
        if np.isnan(val):
            return "—"
        return f"{val*100:.2f}%" if key in pct_keys else f"{val:.3f}"

    header_vals = ["Metric"] + scenarios
    cell_cols: List[List[str]] = [row_keys]
    for s in scenarios:
        cell_cols.append([_fmt(metrics_dict[s].get(k, float("nan")), k) for k in row_keys])

    fig = go.Figure(go.Table(
        header=dict(values=header_vals,
                    fill_color=COLORS["AMAAM"],
                    font=dict(color="white", size=12)),
        cells=dict(values=cell_cols, fill_color="white", align="center"),
    ))
    fig.update_layout(title="Key Metrics — Transaction Cost Scenarios")
    path = out / filename
    fig.write_html(str(path))
    return path


# ---------------------------------------------------------------------------
# 23 — IS vs OOS equity
# ---------------------------------------------------------------------------

def plot_is_oos_equity(
    is_result: "BacktestResult",
    oos_result: "BacktestResult",
    output_dir: str,
    filename: str = "23_is_oos_equity.html",
) -> Path:
    """Side-by-side equity curves for IS and OOS periods.

    Parameters
    ----------
    is_result : BacktestResult
        In-sample backtest result.
    oos_result : BacktestResult
        Out-of-sample backtest result.
    output_dir : str
        Directory where the HTML is saved.
    filename : str
        Output file name.

    Returns
    -------
    Path
        Absolute path to the saved HTML.
    """
    out = _ensure_dir(output_dir)
    fig = make_subplots(rows=1, cols=2, subplot_titles=["In-Sample", "Out-of-Sample"])

    for col, result in enumerate([is_result, oos_result], start=1):
        eq = result.equity_curve
        eq_norm = eq / eq.iloc[0] if len(eq) > 0 else eq
        fig.add_trace(go.Scatter(
            x=eq_norm.index, y=eq_norm.values,
            name="AMAAM", line=dict(color=COLORS["AMAAM"], width=2),
            showlegend=(col == 1),
        ), row=1, col=col)

    fig.update_layout(
        template=TEMPLATE, title="IS vs OOS Equity Curves", hovermode="x unified",
    )
    path = out / filename
    fig.write_html(str(path))
    return path


# ---------------------------------------------------------------------------
# 24 — IS vs OOS stats table
# ---------------------------------------------------------------------------

def plot_is_oos_stats_table(
    is_result: "BacktestResult",
    oos_result: "BacktestResult",
    output_dir: str,
    filename: str = "24_is_oos_stats.html",
) -> Path:
    """Side-by-side metrics table for IS vs OOS periods.

    Parameters
    ----------
    is_result : BacktestResult
        In-sample backtest result.
    oos_result : BacktestResult
        Out-of-sample backtest result.
    output_dir : str
        Directory where the HTML is saved.
    filename : str
        Output file name.

    Returns
    -------
    Path
        Absolute path to the saved HTML.
    """
    out = _ensure_dir(output_dir)
    key_metrics = [
        "Annualized Return", "Annualized Volatility", "Sharpe Ratio",
        "Sortino Ratio", "Calmar Ratio", "Max Drawdown", "Worst Year",
    ]
    pct_keys = {"Annualized Return", "Annualized Volatility", "Max Drawdown", "Worst Year"}

    def _fmt(val: float, key: str) -> str:
        if np.isnan(val):
            return "—"
        return f"{val*100:.2f}%" if key in pct_keys else f"{val:.3f}"

    is_vals  = [_fmt(is_result.metrics.get(k,  float("nan")), k) for k in key_metrics]
    oos_vals = [_fmt(oos_result.metrics.get(k, float("nan")), k) for k in key_metrics]

    fig = go.Figure(go.Table(
        header=dict(values=["Metric", "In-Sample", "Out-of-Sample"],
                    fill_color=COLORS["AMAAM"],
                    font=dict(color="white", size=12)),
        cells=dict(values=[key_metrics, is_vals, oos_vals],
                   fill_color="white", align="center"),
    ))
    fig.update_layout(title="IS vs OOS Performance Metrics")
    path = out / filename
    fig.write_html(str(path))
    return path
