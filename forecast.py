"""TeraLens forecasting (team: The Click).

Honest, simple, explainable projections, built for teaching rather than for
operational use:

* :func:`project_series` extends the fitted straight line into the future with
  a 95% confidence band (where the *average* line may go) and a wider 95%
  prediction interval (where a *single year* may land). It also returns a
  climatology baseline ("just use the long-term average") for comparison.
* :func:`backtest` hides the most recent years, forecasts them from the older
  years, and scores the trend forecast against the climatology baseline
  (mean absolute error, skill score, interval coverage).
* :func:`exceedance_probability` turns a projection into the chance that a
  future year beats a threshold (e.g. the current record).
* :func:`project_monthly` projects the seasonal cycle month by month.

These are statistical extrapolations of a 25-year record, not climate-model
projections; every user-facing text says so.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

import analysis as an

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _clip(values, parameter: str):
    lo, hi = an.PHYSICAL_BOUNDS.get(parameter, (None, None))
    return np.clip(values, lo if lo is not None else -np.inf,
                   hi if hi is not None else np.inf)


def _xy(series: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    s = series.dropna()
    return s.index.to_numpy(float), s.to_numpy(float)


# --------------------------------------------------------------------------- #
# Projection
# --------------------------------------------------------------------------- #


@dataclass
class Projection:
    """A straight-line projection with uncertainty and a climatology baseline."""

    parameter: str
    label: str
    units: str
    level: float
    years: np.ndarray            # future years
    mean: np.ndarray             # projected trend value
    ci_low: np.ndarray           # band for the average line
    ci_high: np.ndarray
    pi_low: np.ndarray           # band for a single year
    pi_high: np.ndarray
    clim_mean: float             # baseline: long-term average
    clim_low: float
    clim_high: float
    n: int
    slope: float
    intercept: float
    resid_sd: float
    x_mean: float
    sxx: float
    y_mean: float
    y_sd: float
    trend: an.TrendResult = field(repr=False)

    def at(self, year: int) -> dict:
        """Projected values for one future year."""
        i = int(np.where(self.years == year)[0][0])
        return {"year": int(year), "mean": float(self.mean[i]),
                "ci_low": float(self.ci_low[i]), "ci_high": float(self.ci_high[i]),
                "pi_low": float(self.pi_low[i]), "pi_high": float(self.pi_high[i])}

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame({
            "year": self.years, "trend forecast": self.mean,
            "average-line low": self.ci_low, "average-line high": self.ci_high,
            "single-year low": self.pi_low, "single-year high": self.pi_high,
            "baseline (long-term average)": self.clim_mean,
        }).round(3)


def project_series(series: pd.Series, parameter: str, units: str, label: str = "",
                   horizon_end: int = 2035, level: float = 0.95) -> Projection:
    """Extend the OLS trend of a yearly series to ``horizon_end``.

    ``series`` must be indexed by integer years (annual or single-month scope).
    Values are clipped to physical limits (no negative rain, humidity <= 100%).
    """
    x, y = _xy(series)
    n = len(y)
    if n < 6:
        raise ValueError("Need at least 6 years to project a trend.")
    last = int(x.max())
    if horizon_end <= last:
        raise ValueError(f"horizon_end must be after the last data year ({last}).")

    trend = an.analyze_trend(series, parameter, units, label)
    lr = stats.linregress(x, y)
    resid = y - (lr.intercept + lr.slope * x)
    s = float(np.sqrt(np.sum(resid ** 2) / (n - 2)))
    xm = float(x.mean())
    sxx = float(np.sum((x - xm) ** 2))
    t = stats.t.ppf(0.5 + level / 2, n - 2)

    years = np.arange(last + 1, horizon_end + 1)
    mean = lr.intercept + lr.slope * years
    lev = (years - xm) ** 2 / sxx
    ci_half = t * s * np.sqrt(1 / n + lev)
    pi_half = t * s * np.sqrt(1 + 1 / n + lev)

    y_mean, y_sd = float(y.mean()), float(y.std(ddof=1))
    clim_half = stats.t.ppf(0.5 + level / 2, n - 1) * y_sd * np.sqrt(1 + 1 / n)

    return Projection(
        parameter=parameter, label=label or trend.label, units=units, level=level,
        years=years, mean=_clip(mean, parameter),
        ci_low=_clip(mean - ci_half, parameter), ci_high=_clip(mean + ci_half, parameter),
        pi_low=_clip(mean - pi_half, parameter), pi_high=_clip(mean + pi_half, parameter),
        clim_mean=y_mean, clim_low=float(_clip(y_mean - clim_half, parameter)),
        clim_high=float(_clip(y_mean + clim_half, parameter)),
        n=n, slope=float(lr.slope), intercept=float(lr.intercept), resid_sd=s,
        x_mean=xm, sxx=sxx, y_mean=y_mean, y_sd=y_sd, trend=trend,
    )


def exceedance_probability(proj: Projection, year: int, threshold: float,
                           baseline: bool = False) -> float:
    """Chance that the value in ``year`` exceeds ``threshold``.

    Uses the Student-t predictive distribution of either the trend model or,
    with ``baseline=True``, the no-trend long-term average model.
    """
    if baseline:
        scale = proj.y_sd * np.sqrt(1 + 1 / proj.n)
        return float(stats.t.sf((threshold - proj.y_mean) / scale, proj.n - 1))
    center = proj.intercept + proj.slope * year
    scale = proj.resid_sd * np.sqrt(1 + 1 / proj.n + (year - proj.x_mean) ** 2 / proj.sxx)
    return float(stats.t.sf((threshold - center) / scale, proj.n - 2))


# --------------------------------------------------------------------------- #
# Backtest
# --------------------------------------------------------------------------- #


@dataclass
class BacktestResult:
    """How a trend forecast made from older years did on newer, hidden years."""

    train_start: int
    train_end: int
    years: np.ndarray
    actual: np.ndarray
    trend_pred: np.ndarray
    clim_pred: float
    pi_low: np.ndarray
    pi_high: np.ndarray
    mae_trend: float
    mae_clim: float
    skill: float          # 1 - MAE_trend / MAE_clim
    coverage: float       # share of actual values inside the prediction interval
    explanation: str = ""

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame({
            "year": self.years, "actual": self.actual,
            "trend forecast": self.trend_pred, "average forecast": self.clim_pred,
            "range low": self.pi_low, "range high": self.pi_high,
        }).round(3)


def backtest(series: pd.Series, parameter: str, train_end: int = 2018,
             level: float = 0.95) -> BacktestResult:
    """Fit on years <= ``train_end``, forecast the rest, score against the average."""
    x, y = _xy(series)
    train = x <= train_end
    test = ~train
    if train.sum() < 8 or test.sum() < 2:
        raise ValueError("Backtest needs >= 8 training years and >= 2 test years.")
    xt, yt = x[train], y[train]
    n = len(yt)
    lr = stats.linregress(xt, yt)
    resid = yt - (lr.intercept + lr.slope * xt)
    s = np.sqrt(np.sum(resid ** 2) / (n - 2))
    xm, sxx = xt.mean(), np.sum((xt - xt.mean()) ** 2)
    t = stats.t.ppf(0.5 + level / 2, n - 2)

    xf, actual = x[test], y[test]
    pred = _clip(lr.intercept + lr.slope * xf, parameter)
    half = t * s * np.sqrt(1 + 1 / n + (xf - xm) ** 2 / sxx)
    lo, hi = _clip(pred - half, parameter), _clip(pred + half, parameter)
    clim = float(yt.mean())
    mae_t = float(np.mean(np.abs(actual - pred)))
    mae_c = float(np.mean(np.abs(actual - clim)))
    skill = 1 - mae_t / mae_c if mae_c > 0 else 0.0
    coverage = float(np.mean((actual >= lo) & (actual <= hi)))

    bt = BacktestResult(train_start=int(xt.min()), train_end=int(train_end),
                        years=xf.astype(int), actual=actual, trend_pred=pred,
                        clim_pred=clim, pi_low=lo, pi_high=hi, mae_trend=mae_t,
                        mae_clim=mae_c, skill=float(skill), coverage=coverage)
    bt.explanation = explain_backtest(bt)
    return bt


def explain_backtest(bt: BacktestResult) -> str:
    if bt.skill > 0.1:
        judge = "so the trend added real forecasting value"
    elif bt.skill >= -0.1:
        judge = "so the trend was no better than simply using the average"
    else:
        judge = "so following the trend made forecasts worse than using the average"
    return (f"Time-travel test: using only {bt.train_start}–{bt.train_end}, we 'forecast' "
            f"{bt.years.min()}–{bt.years.max()} and compared with what happened. Typical "
            f"error was {bt.mae_trend:.2f} for the trend vs {bt.mae_clim:.2f} for the "
            f"average (skill {bt.skill:+.0%}), {judge}. {bt.coverage:.0%} of real values "
            f"fell inside the 95% range (ideally about 95%).")


def forecast_verdict(proj: Projection, bt: Optional[BacktestResult] = None) -> str:
    """Plain-English forecast summary that states its own limits."""
    year = int(proj.years[-1])
    v = proj.at(year)
    u = proj.units
    text = (f"If the {proj.trend.start:.0f}–{proj.trend.end:.0f} straight-line trend "
            f"simply continued, {proj.label.lower()} in {year} would be about "
            f"{v['mean']:.2f} {u}, with a single year likely between {v['pi_low']:.2f} "
            f"and {v['pi_high']:.2f} {u}.")
    if proj.trend.flag_adjusted == "significant" and proj.trend.flag == "significant":
        text += (" The trend is statistically significant, so extending it a few years "
                 "is reasonable, but the further out, the shakier.")
    else:
        text += (f" But the trend is not firmly established, so the long-term average "
                 f"({proj.clim_mean:.2f} {u}, likely range {proj.clim_low:.2f}–"
                 f"{proj.clim_high:.2f}) is an equally defensible forecast.")
    if bt is not None:
        text += " " + bt.explanation
    text += (" This is a statistical extrapolation, not a climate-model projection: it "
             "knows nothing about emissions, El Niño or city growth.")
    return text


# --------------------------------------------------------------------------- #
# Monthly projection
# --------------------------------------------------------------------------- #


def project_monthly(df: pd.DataFrame, parameter: str, horizon_end: int = 2030,
                    level: float = 0.95) -> pd.DataFrame:
    """Project each future month as climatology + anomaly trend.

    The spread uses each calendar month's own residual variability, so the
    monsoon months get wide ranges and dry months narrow ones. For rainfall the
    trend is applied *in proportion* to each month's normal (a wetter climate
    adds more rain to July than to dry January); for temperature and humidity
    it is added equally to every month.
    """
    data = an._subset(an.add_anomalies(df), parameter).dropna(subset=["value"])
    x = (data["year"] + (data["month"] - 0.5) / 12).to_numpy(float)
    a = data["anomaly"].to_numpy(float)
    n = len(a)
    lr = stats.linregress(x, a)
    resid = pd.Series(a - (lr.intercept + lr.slope * x), index=data["month"].to_numpy())
    sd_m = resid.groupby(level=0).std(ddof=1)
    clim = an.monthly_climatology(df, parameter)
    xm, sxx = x.mean(), np.sum((x - x.mean()) ** 2)
    t = stats.t.ppf(0.5 + level / 2, n - 2)

    overall = float(data["value"].mean())
    proportional = parameter == "PRECTOTCORR" and overall > 0
    last = int(data["year"].max())
    rows = []
    for yr in range(last + 1, horizon_end + 1):
        for m in range(1, 13):
            xf = yr + (m - 0.5) / 12
            shift = lr.intercept + lr.slope * xf
            if proportional:
                ratio = max(0.0, (overall + shift) / overall)
                mean = clim[m] * ratio
                half = t * sd_m[m] * ratio * np.sqrt(1 + 1 / n + (xf - xm) ** 2 / sxx)
            else:
                mean = clim[m] + shift
                half = t * sd_m[m] * np.sqrt(1 + 1 / n + (xf - xm) ** 2 / sxx)
            rows.append({"year": yr, "month": m, "x": xf, "mean": mean,
                         "pi_low": mean - half, "pi_high": mean + half,
                         "climatology": clim[m]})
    out = pd.DataFrame(rows)
    for c in ("mean", "pi_low", "pi_high"):
        out[c] = _clip(out[c].to_numpy(), parameter)
    return out
