"""TeraLens figures (team: The Click).

Every function returns a ``matplotlib.figure.Figure`` built without pyplot's
global state, so it is safe in Streamlit and in tests.

Colour roles: blue = observed data, orange = linear trend / forecast,
grey dashed = Sen's slope or the long-term-average baseline. Diverging maps
(blue-red for temperature, brown-teal for moisture) have a neutral midpoint
at "normal".
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.figure import Figure

import analysis as an
import forecast as fc

COLOR_DATA = "#2a78d6"
COLOR_FIT = "#eb6834"
COLOR_SEN = "#52514e"
COLOR_GOOD = "#1baf7a"
COLOR_MUTED = "#b5b4ae"
COLOR_GRID = "#e6e5e0"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
DIVERGING = {"T2M": "RdBu_r", "PRECTOTCORR": "BrBG", "RH2M": "BrBG"}


def style(ax) -> None:
    """Recessive axes: no top/right spines, light horizontal grid."""
    ax.set_facecolor("white")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(COLOR_MUTED)
    ax.tick_params(colors=TEXT_2, labelsize=9)
    ax.grid(axis="y", color=COLOR_GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _title(ax, text: str) -> None:
    ax.set_title(text, loc="left", fontsize=11, color=TEXT)


def _legend(ax, **kw) -> None:
    ax.legend(frameon=False, fontsize=8.5, labelcolor=TEXT_2, **kw)


# --------------------------------------------------------------------------- #
# Core views
# --------------------------------------------------------------------------- #


def plot_trend(series: pd.Series, result: an.TrendResult,
               rolling: Optional[int] = None) -> Figure:
    """Time series with OLS line, shaded 95% CI of the line, Sen's slope and
    an optional centred rolling mean (in points)."""
    s = series.dropna()
    x, y = s.index.to_numpy(float), s.to_numpy(float)
    fig = Figure(figsize=(9, 4.2), dpi=110)
    ax = fig.add_subplot(111)
    style(ax)
    dense = len(s) > 60
    ax.plot(x, y, color=COLOR_DATA, linewidth=1.0 if dense else 2,
            alpha=0.6 if (dense and rolling) else 1,
            marker=None if dense else "o", markersize=5, label="Data")
    if rolling and rolling > 1:
        rm = s.rolling(rolling, center=True, min_periods=max(2, rolling // 2)).mean()
        ax.plot(rm.index, rm.to_numpy(), color=TEXT, linewidth=1.8,
                label=f"{rolling}-point moving average")
    xx = np.linspace(x.min(), x.max(), 200)
    fit, lo, hi = an.regression_band(xx, x, y)
    ax.fill_between(xx, lo, hi, color=COLOR_FIT, alpha=0.18, linewidth=0,
                    label="95% CI of trend line")
    ax.plot(xx, fit, color=COLOR_FIT, linewidth=2,
            label=f"Linear trend {result.slope:+.3f} {result.slope_units}")
    ax.plot(xx, result.sens_intercept + result.sens_slope * xx, color=COLOR_SEN,
            linewidth=1.5, linestyle="--",
            label=f"Sen's slope {result.sens_slope:+.3f} {result.slope_units}")
    if result.label.endswith("(anomalies)"):
        ax.axhline(0, color=COLOR_MUTED, linewidth=0.8)
    _title(ax, f"{result.label}: {result.flag} ({an.format_p(result.p_value)})")
    ax.set_ylabel(result.units, color=TEXT_2)
    ax.set_xlabel("Year", color=TEXT_2)
    _legend(ax, loc="upper left")
    fig.tight_layout()
    return fig


def plot_values(series: pd.Series, units: str, label: str, bars: bool) -> Figure:
    """Plain bars (yearly) or line (monthly) of a series."""
    fig = Figure(figsize=(9, 3.8), dpi=110)
    ax = fig.add_subplot(111)
    style(ax)
    s = series.dropna()
    if bars:
        ax.bar(s.index, s.to_numpy(), color=COLOR_DATA, width=0.8,
               edgecolor="white", linewidth=1)
    else:
        ax.plot(s.index, s.to_numpy(), color=COLOR_DATA, linewidth=1.2)
    _title(ax, label)
    ax.set_ylabel(units, color=TEXT_2)
    ax.set_xlabel("Year", color=TEXT_2)
    fig.tight_layout()
    return fig


def plot_seasonal_panels(df: pd.DataFrame, parameter: str) -> Figure:
    """12 small multiples (one per month) of anomalies with each month's trend.

    Shared y-axis so slopes are comparable; '*' p < 0.05, '(*)' p < 0.10.
    """
    clim = an.monthly_climatology(df, parameter)
    fig = Figure(figsize=(11, 7.5), dpi=100)
    axes = fig.subplots(3, 4, sharex=True, sharey=True)
    marks = {"significant": " *", "suggestive": " (*)", "not significant": ""}
    for m, ax in enumerate(axes.flat, start=1):
        style(ax)
        raw = an.month_series(df, parameter, m).dropna()
        r = an.analyze_trend(raw, parameter, an.UNITS[parameter])
        x = raw.index.to_numpy(float)
        hot = r.flag != "not significant"
        ax.axhline(0, color=COLOR_MUTED, linewidth=0.8)
        ax.plot(x, raw.to_numpy(float) - clim[m], color=COLOR_DATA, linewidth=1.3)
        ax.plot(x, r.intercept - clim[m] + r.slope * x,
                color=COLOR_FIT if hot else COLOR_SEN, linewidth=2 if hot else 1.2)
        ax.set_title(f"{an.MONTH_NAMES[m - 1][:3]}  {r.slope:+.3f}/yr{marks[r.flag]}",
                     fontsize=9.5, loc="left", color=TEXT if hot else TEXT_2)
    fig.suptitle(f"{an.LONG_NAMES[parameter]}: each month vs its own 2001–2025 average "
                 f"({an.UNITS[parameter]}); * p < 0.05, (*) p < 0.10",
                 x=0.01, ha="left", fontsize=11, color=TEXT)
    fig.tight_layout()
    return fig


def plot_annual_bars(df: pd.DataFrame, parameter: str) -> Figure:
    """Annual totals (rain) or annual means (temperature, humidity) as bars."""
    s = an.annual_series(df, parameter)
    fig = Figure(figsize=(9, 3.8), dpi=110)
    ax = fig.add_subplot(111)
    style(ax)
    ax.bar(s.index, s.to_numpy(), color=COLOR_DATA, width=0.8,
           edgecolor="white", linewidth=1)
    mean = float(s.mean())
    ax.axhline(mean, color=TEXT_2, linewidth=1, linestyle=":")
    ax.annotate(f"25-yr mean {mean:.1f} {an.ANNUAL_UNITS[parameter]}",
                xy=(s.index.min() - 0.4, mean), xytext=(0, 5),
                textcoords="offset points", ha="left", va="bottom", fontsize=8.5,
                color=TEXT_2, bbox=dict(facecolor="white", edgecolor="none", pad=1))
    if parameter != "PRECTOTCORR":  # means: a zero baseline would flatten the bars
        pad = (s.max() - s.min()) * 0.3
        ax.set_ylim(s.min() - pad, s.max() + pad)
    _title(ax, f"{an.LONG_NAMES[parameter]}, {an.ANNUAL_AGG_LABEL[parameter]} "
               f"({an.ANNUAL_UNITS[parameter]})")
    ax.set_xlabel("Year", color=TEXT_2)
    fig.tight_layout()
    return fig


def plot_compare_years(df: pd.DataFrame, parameter: str,
                       years: Sequence[int]) -> Figure:
    """Seasonal cycle of chosen years against the 2001–2025 range and mean."""
    piv = an._subset(df, parameter).pivot(index="month", columns="year", values="value")
    months = np.arange(1, 13)
    fig = Figure(figsize=(9, 4.2), dpi=110)
    ax = fig.add_subplot(111)
    style(ax)
    ax.fill_between(months, piv.min(axis=1), piv.max(axis=1), color=COLOR_MUTED,
                    alpha=0.25, linewidth=0, label="2001–2025 range")
    ax.plot(months, piv.mean(axis=1), color=TEXT_2, linewidth=1.5, linestyle="--",
            label="2001–2025 mean")
    for color, yr in zip(CATEGORICAL, years):
        if yr in piv.columns:
            ax.plot(months, piv[yr], color=color, linewidth=2, marker="o",
                    markersize=4, label=str(yr))
    ax.set_xticks(months, [m[:3] for m in an.MONTH_NAMES])
    ax.set_ylabel(an.UNITS[parameter], color=TEXT_2)
    _title(ax, f"{an.LONG_NAMES[parameter]}: chosen years vs the 25-year record")
    _legend(ax)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Forecast views
# --------------------------------------------------------------------------- #


def plot_forecast(series: pd.Series, proj: fc.Projection,
                  threshold: Optional[float] = None) -> Figure:
    """History, fitted trend, projection with average-line and single-year bands,
    and the long-term-average baseline."""
    s = series.dropna()
    x, y = s.index.to_numpy(float), s.to_numpy(float)
    fig = Figure(figsize=(9.5, 4.4), dpi=110)
    ax = fig.add_subplot(111)
    style(ax)
    ax.plot(x, y, color=COLOR_DATA, linewidth=2, marker="o", markersize=4,
            label="Observed")
    xh = np.linspace(x.min(), x.max(), 50)
    ax.plot(xh, proj.intercept + proj.slope * xh, color=COLOR_FIT, linewidth=1.5)
    yrs = np.concatenate([[x.max()], proj.years])
    join = proj.intercept + proj.slope * x.max()
    ax.fill_between(proj.years, proj.pi_low, proj.pi_high, color=COLOR_FIT, alpha=0.12,
                    linewidth=0, label="95% range for a single year")
    ax.fill_between(proj.years, proj.ci_low, proj.ci_high, color=COLOR_FIT, alpha=0.28,
                    linewidth=0, label="95% range for the average line")
    ax.plot(yrs, np.concatenate([[join], proj.mean]), color=COLOR_FIT, linewidth=2.2,
            linestyle=(0, (4, 2)), label="Trend forecast")
    ax.hlines(proj.clim_mean, x.min(), proj.years.max(), color=COLOR_SEN, linewidth=1.2,
              linestyle=":", label=f"Long-term average {proj.clim_mean:.2f}")
    if threshold is not None:
        ax.axhline(threshold, color=TEXT, linewidth=0.8, linestyle="-.",
                   label=f"Threshold {threshold:.2f}")
    ax.axvline(x.max() + 0.5, color=COLOR_MUTED, linewidth=0.8)
    ax.annotate("forecast →", xy=(x.max() + 0.7, 1), xycoords=("data", "axes fraction"),
                va="top", fontsize=8.5, color=TEXT_2)
    _title(ax, f"{proj.label}: projection to {proj.years.max()}")
    ax.set_ylabel(proj.units, color=TEXT_2)
    ax.set_xlabel("Year", color=TEXT_2)
    _legend(ax, loc="upper left", ncols=2)
    fig.tight_layout()
    return fig


def plot_backtest(series: pd.Series, bt: fc.BacktestResult, units: str) -> Figure:
    """The 'time-travel' test: forecasts made from older years vs what happened."""
    s = series.dropna()
    fig = Figure(figsize=(9, 3.8), dpi=110)
    ax = fig.add_subplot(111)
    style(ax)
    train = s[s.index <= bt.train_end]
    ax.plot(train.index, train.to_numpy(), color=COLOR_MUTED, linewidth=1.5,
            marker="o", markersize=3, label=f"Training years (≤ {bt.train_end})")
    ax.fill_between(bt.years, bt.pi_low, bt.pi_high, color=COLOR_FIT, alpha=0.15,
                    linewidth=0, label="Forecast 95% range")
    ax.plot(bt.years, bt.trend_pred, color=COLOR_FIT, linewidth=2,
            label=f"Trend forecast (error {bt.mae_trend:.2f})")
    ax.hlines(bt.clim_pred, bt.years.min(), bt.years.max(), color=COLOR_SEN,
              linestyle=":", linewidth=1.5, label=f"Average forecast (error {bt.mae_clim:.2f})")
    ax.plot(bt.years, bt.actual, color=COLOR_DATA, linewidth=2, marker="o",
            markersize=5, label="What actually happened")
    _title(ax, f"Time-travel test: skill {bt.skill:+.0%} vs the average")
    ax.set_ylabel(units, color=TEXT_2)
    _legend(ax, loc="upper left")
    fig.tight_layout()
    return fig


def plot_monthly_forecast(df: pd.DataFrame, parameter: str,
                          mf: pd.DataFrame, history_years: int = 3) -> Figure:
    """Last few observed years month by month, then the projected months."""
    sub = an._subset(df, parameter)
    last = int(sub["year"].max())
    hist = sub[sub["year"] > last - history_years]
    xh = (hist["year"] + (hist["month"] - 0.5) / 12).to_numpy()
    fig = Figure(figsize=(10, 4), dpi=110)
    ax = fig.add_subplot(111)
    style(ax)
    ax.plot(xh, hist["value"].to_numpy(), color=COLOR_DATA, linewidth=1.8,
            marker="o", markersize=3, label="Observed")
    ax.fill_between(mf["x"], mf["pi_low"], mf["pi_high"], color=COLOR_FIT, alpha=0.15,
                    linewidth=0, label="95% range")
    ax.plot(mf["x"], mf["climatology"], color=COLOR_SEN, linestyle=":", linewidth=1.3,
            label="Normal for that month (2001–2025)")
    ax.plot(mf["x"], mf["mean"], color=COLOR_FIT, linewidth=2, label="Projection")
    ax.axvline(last + 1, color=COLOR_MUTED, linewidth=0.8)
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + (hi - lo) * 0.22)  # headroom for the legend
    _title(ax, f"{an.LONG_NAMES[parameter]}: month-by-month projection")
    ax.set_ylabel(an.UNITS[parameter], color=TEXT_2)
    _legend(ax, loc="upper left", ncols=2)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Shift, extremes, relationships
# --------------------------------------------------------------------------- #


def plot_changepoint(series: pd.Series, cp: an.ChangePoint, units: str,
                     label: str) -> Figure:
    """Series with before/after means of the Pettitt change point."""
    s = series.dropna()
    fig = Figure(figsize=(9, 3.8), dpi=110)
    ax = fig.add_subplot(111)
    style(ax)
    dense = len(s) > 60
    ax.plot(s.index, s.to_numpy(), color=COLOR_DATA, linewidth=1 if dense else 2,
            marker=None if dense else "o", markersize=4, label="Data")
    split = (cp.last_before + cp.first_after) / 2
    line_color = COLOR_FIT if cp.flag != "not significant" else COLOR_SEN
    ax.hlines(cp.mean_before, s.index.min(), split, color=line_color, linewidth=2.5,
              label=f"Average before: {cp.mean_before:.2f}")
    ax.hlines(cp.mean_after, split, s.index.max(), color=line_color, linewidth=2.5,
              linestyle="--", label=f"Average after: {cp.mean_after:.2f}")
    ax.axvline(split, color=TEXT_2, linewidth=0.8, linestyle=":")
    _title(ax, f"{label}: most likely shift point "
               f"({cp.flag}, {an.format_p(cp.p_value)})")
    ax.set_ylabel(units, color=TEXT_2)
    _legend(ax, loc="upper left")
    fig.tight_layout()
    return fig


def plot_anomaly_heatmap(df: pd.DataFrame, parameter: str) -> Figure:
    """Year x month grid coloured by departure from normal ("climate stripes" grid)."""
    mat = an.anomaly_matrix(df, parameter)
    vmax = float(np.nanpercentile(np.abs(mat.to_numpy()), 97)) or 1.0
    fig = Figure(figsize=(9.5, 6.2), dpi=110)
    ax = fig.add_subplot(111)
    im = ax.imshow(mat.to_numpy(), aspect="auto", cmap=DIVERGING[parameter],
                   norm=TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax))
    ax.set_xticks(range(12), [m[:3] for m in an.MONTH_NAMES])
    ax.set_yticks(range(len(mat.index)), mat.index)
    ax.tick_params(colors=TEXT_2, labelsize=8, length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_xticks(np.arange(-0.5, 12), minor=True)
    ax.set_yticks(np.arange(-0.5, len(mat.index)), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", length=0)
    cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cb.set_label(f"Difference from that month's normal ({an.UNITS[parameter]})",
                 color=TEXT_2, fontsize=9)
    cb.ax.tick_params(colors=TEXT_2, labelsize=8)
    cb.outline.set_visible(False)
    _title(ax, f"{an.LONG_NAMES[parameter]}: every month vs normal, 2001–2025")
    fig.tight_layout()
    return fig


def plot_relationship(data: pd.DataFrame, rel: an.Relationship, units_x: str,
                      units_y: str) -> Figure:
    """Scatter of two variables with their fitted line."""
    fig = Figure(figsize=(5.2, 4.2), dpi=110)
    ax = fig.add_subplot(111)
    style(ax)
    ax.grid(axis="x", color=COLOR_GRID, linewidth=0.8)
    ax.scatter(data[rel.x], data[rel.y], s=22, color=COLOR_DATA, alpha=0.7,
               edgecolor="white", linewidth=0.5)
    xx = np.linspace(data[rel.x].min(), data[rel.x].max(), 50)
    ax.plot(xx, rel.intercept + rel.slope * xx,
            color=COLOR_FIT if rel.flag == "significant" else COLOR_SEN, linewidth=2)
    ax.set_xlabel(f"{an.LONG_NAMES[rel.x]} ({units_x})", color=TEXT_2)
    ax.set_ylabel(f"{an.LONG_NAMES[rel.y]} ({units_y})", color=TEXT_2)
    _title(ax, f"r = {rel.r:+.2f} ({an.format_p(rel.p_value)})")
    fig.tight_layout()
    return fig


def plot_correlation_matrix(data: pd.DataFrame) -> Figure:
    """3 x 3 correlation grid with values written in each cell."""
    corr = data.corr()
    names = [an.LONG_NAMES.get(c, c) for c in corr.columns]
    fig = Figure(figsize=(4.8, 4), dpi=110)
    ax = fig.add_subplot(111)
    im = ax.imshow(corr.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(names)), names, rotation=20, ha="right")
    ax.set_yticks(range(len(names)), names)
    ax.tick_params(colors=TEXT_2, labelsize=9, length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    for i in range(len(names)):
        for j in range(len(names)):
            v = corr.iat[i, j]
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=10,
                    color="white" if abs(v) > 0.6 else TEXT)
    cb = fig.colorbar(im, ax=ax, shrink=0.8)
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=TEXT_2, labelsize=8)
    _title(ax, "Correlation (r)")
    fig.tight_layout()
    return fig
