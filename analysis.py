"""TeraLens analysis core (team: The Click).

UI-free, testable building blocks for the TeraLens climate explorer:

* loader for NASA/POWER monthly point CSVs (header block skipped, -999 -> NaN),
* series builders (annual, one calendar month, all months as anomalies),
* trend tests: OLS (``scipy.stats.linregress``), Mann-Kendall (+ Hamed-Rao
  autocorrelation correction), Sen's slope, effective-sample-size p-value,
* Pettitt change-point test ("steady trend or sudden shift?"),
* extremes, record years, year x month anomaly matrix,
* correlations between variables,
* plain-English verdicts, per-statistic explanations and a Markdown report.

Forecasting lives in ``forecast.py``; figures in ``plots.py``.
Run ``python analysis.py [path/to/csv]`` for a printed summary.
"""

from __future__ import annotations

import calendar
import io
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import IO, Literal, Optional, Sequence, Union

import numpy as np
import pandas as pd
import pymannkendall as mk
from scipy import stats

APP_NAME = "TeraLens"
TEAM_NAME = "The Click"

# --------------------------------------------------------------------------- #
# Constants & plain-language text
# --------------------------------------------------------------------------- #

DEFAULT_CSV = "POWER_Point_Monthly_20010101_20251231_024d86N_067d01E_LST.csv"
MISSING_VALUE = -999
MONTH_COLS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
              "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
MONTH_NAMES = list(calendar.month_name)[1:]
PARAMETERS = ["PRECTOTCORR", "RH2M", "T2M"]

DEFINITIONS: dict[str, str] = {
    "PRECTOTCORR": "Average daily rainfall in mm per day. Higher = wetter month.",
    "RH2M": "Percent of moisture in the air. 100% = saturated.",
    "T2M": "Air temperature in °C at about 2 m above ground.",
}
LONG_NAMES: dict[str, str] = {
    "PRECTOTCORR": "Precipitation",
    "RH2M": "Relative humidity",
    "T2M": "Temperature",
}
UNITS: dict[str, str] = {"PRECTOTCORR": "mm/day", "RH2M": "%", "T2M": "°C"}
# Annual aggregation differs by variable: rain is summed to a yearly total,
# temperature and humidity are averaged.
ANNUAL_UNITS: dict[str, str] = {"PRECTOTCORR": "mm", "RH2M": "%", "T2M": "°C"}
ANNUAL_AGG_LABEL: dict[str, str] = {
    "PRECTOTCORR": "annual total",
    "RH2M": "annual mean",
    "T2M": "annual mean",
}
TREND_WORDS: dict[str, tuple[str, str]] = {
    "T2M": ("warming", "cooling"),
    "PRECTOTCORR": ("wetting", "drying"),
    "RH2M": ("moistening", "drying"),
}
# Values a variable can physically take (used to clip forecasts).
PHYSICAL_BOUNDS: dict[str, tuple[Optional[float], Optional[float]]] = {
    "PRECTOTCORR": (0.0, None),
    "RH2M": (0.0, 100.0),
    "T2M": (None, None),
}

SIGNIFICANT_P = 0.05
SUGGESTIVE_P = 0.10

STAT_GLOSSARY: dict[str, str] = {
    "linear_regression": (
        "Fits the straight line that best follows the points; its slope is the "
        "average change per year."
    ),
    "p_value": (
        "If there were truly no trend, the chance of seeing a slope at least this "
        "steep from random ups and downs alone."
    ),
    "mann_kendall": (
        "Counts how often later values are higher than earlier ones. It ignores "
        "the size of jumps, so one extreme year cannot dominate it."
    ),
    "sens_slope": (
        "The median of the slopes between every pair of years; a 'typical' rate "
        "of change that outliers barely move."
    ),
    "autocorrelation": (
        "When each value resembles the one before it, the data holds less "
        "independent information than its length suggests, so p-values must be "
        "adjusted upward."
    ),
    "anomaly": (
        "How far a month is above or below its own 2001–2025 average, which "
        "removes the normal seasonal cycle."
    ),
    "change_point": (
        "The Pettitt test looks for the single moment where the average level "
        "jumped, to tell a sudden shift apart from a steady trend."
    ),
    "prediction_interval": (
        "The range a single future year is expected to fall in 95% of the time; "
        "wider than the trend line's range because individual years are noisy."
    ),
    "skill_score": (
        "How much better a forecast did than simply guessing the long-term "
        "average: positive is better, zero is no better, negative is worse."
    ),
    "correlation": (
        "A number from -1 to +1 showing how strongly two variables move together; "
        "it does not show that one causes the other."
    ),
}

Source = Union[str, Path, IO[str], IO[bytes]]
Scope = Literal["annual", "month", "monthly"]

# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def _read_text(source: Source) -> str:
    """Return the full text of a path or file-like object (str or bytes)."""
    if isinstance(source, (str, Path)):
        return Path(source).read_text(encoding="utf-8-sig")
    data = source.read()
    return data.decode("utf-8-sig") if isinstance(data, bytes) else data


def load_power_wide(source: Source) -> pd.DataFrame:
    """Parse a NASA/POWER monthly CSV into its wide table (one row per parameter-year).

    The metadata block (``-BEGIN HEADER-`` ... ``-END HEADER-``) is skipped by
    locating the ``PARAMETER,YEAR,...`` row, so files with or without a header
    block both work. Any value <= -999 becomes NaN.
    """
    lines = _read_text(source).splitlines()
    try:
        start = next(i for i, line in enumerate(lines)
                     if line.strip().upper().startswith("PARAMETER,"))
    except StopIteration as exc:
        raise ValueError("Could not find the 'PARAMETER,YEAR,...' column row.") from exc

    df = pd.read_csv(io.StringIO("\n".join(lines[start:])))
    df.columns = [str(c).strip().upper() for c in df.columns]
    required = {"PARAMETER", "YEAR", *MONTH_COLS}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing expected columns: {sorted(missing)}")

    df["PARAMETER"] = df["PARAMETER"].astype(str).str.strip()
    df["YEAR"] = pd.to_numeric(df["YEAR"], errors="raise").astype(int)
    value_cols = [c for c in df.columns if c not in ("PARAMETER", "YEAR")]
    df[value_cols] = df[value_cols].apply(pd.to_numeric, errors="coerce")
    df[value_cols] = df[value_cols].mask(df[value_cols] <= MISSING_VALUE)
    return df


def load_power_csv(source: Source) -> pd.DataFrame:
    """Load a NASA/POWER monthly CSV as a tidy DataFrame.

    Returns columns ``year`` (int), ``month`` (1-12), ``parameter`` (str) and
    ``value`` (float, NaN where the source had -999). The ``ANN`` column is
    dropped; annual values are recomputed by :func:`annual_series` so the
    aggregation rule is explicit.
    """
    wide = load_power_wide(source)
    long = wide.melt(id_vars=["PARAMETER", "YEAR"], value_vars=MONTH_COLS,
                     var_name="month_abbr", value_name="value")
    long["month"] = long["month_abbr"].map({m: i + 1 for i, m in enumerate(MONTH_COLS)})
    long = long.rename(columns={"PARAMETER": "parameter", "YEAR": "year"})
    long = long[["year", "month", "parameter", "value"]]
    return long.sort_values(["parameter", "year", "month"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Series builders
# --------------------------------------------------------------------------- #


def _subset(df: pd.DataFrame, parameter: str) -> pd.DataFrame:
    sub = df[df["parameter"] == parameter]
    if sub.empty:
        raise KeyError(f"Parameter {parameter!r} not found in data.")
    return sub


def monthly_climatology(df: pd.DataFrame, parameter: str) -> pd.Series:
    """Long-term mean for each calendar month (index 1-12)."""
    return _subset(df, parameter).groupby("month")["value"].mean()


def add_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with an ``anomaly`` column: value minus that month's climatology."""
    out = df.copy()
    clim = out.groupby(["parameter", "month"])["value"].transform("mean")
    out["anomaly"] = out["value"] - clim
    return out


def annual_series(df: pd.DataFrame, parameter: str, min_months: int = 12) -> pd.Series:
    """One value per year.

    * PRECTOTCORR: annual **total** in mm = sum(mm/day x days in month).
    * RH2M, T2M: annual **mean** of the monthly means.

    Years with fewer than ``min_months`` valid months are NaN, so a partly
    missing year cannot masquerade as a dry or cool one.
    """
    sub = _subset(df, parameter).copy()
    if parameter == "PRECTOTCORR":
        days = [calendar.monthrange(int(y), int(m))[1]
                for y, m in zip(sub["year"], sub["month"])]
        sub["agg"] = sub["value"] * np.asarray(days)
        values = sub.groupby("year")["agg"].sum(min_count=1)
    else:
        values = sub.groupby("year")["value"].mean()
    counts = sub.groupby("year")["value"].count()
    values[counts < min_months] = np.nan
    values.index = values.index.astype(int)
    return values.rename(f"{parameter} {ANNUAL_AGG_LABEL[parameter]}")


def month_series(df: pd.DataFrame, parameter: str, month: int) -> pd.Series:
    """Values for one calendar month across years (e.g. every July)."""
    if not 1 <= month <= 12:
        raise ValueError("month must be 1-12")
    sub = _subset(df, parameter)
    s = sub[sub["month"] == month].set_index("year")["value"]
    s.index = s.index.astype(int)
    return s.rename(f"{parameter} {MONTH_NAMES[month - 1]}")


def monthly_series(df: pd.DataFrame, parameter: str, anomalies: bool = True) -> pd.Series:
    """Every month as one series indexed by decimal year (mid-month).

    With ``anomalies=True`` (recommended) the seasonal cycle is removed first;
    otherwise the summer/winter swing swamps any long-term trend.
    """
    sub = _subset(add_anomalies(df) if anomalies else df, parameter)
    col = "anomaly" if anomalies else "value"
    x = sub["year"] + (sub["month"] - 0.5) / 12.0
    s = pd.Series(sub[col].to_numpy(), index=x.to_numpy(), name=parameter)
    return s.sort_index()


def build_series(df: pd.DataFrame, parameter: str, scope: Scope,
                 month: Optional[int] = None,
                 anomalies: bool = True) -> tuple[pd.Series, str, str]:
    """Dispatch to the right builder. Returns ``(series, units, label)``."""
    name = LONG_NAMES[parameter]
    if scope == "annual":
        return (annual_series(df, parameter), ANNUAL_UNITS[parameter],
                f"{name}, {ANNUAL_AGG_LABEL[parameter]}")
    if scope == "month":
        if month is None:
            raise ValueError("scope='month' needs a month (1-12)")
        return (month_series(df, parameter, month), UNITS[parameter],
                f"{name}, {MONTH_NAMES[month - 1]} only")
    if scope == "monthly":
        label = f"{name}, all months" + (" (anomalies)" if anomalies else "")
        return monthly_series(df, parameter, anomalies), UNITS[parameter], label
    raise ValueError(f"Unknown scope {scope!r}")


# --------------------------------------------------------------------------- #
# Trend analysis
# --------------------------------------------------------------------------- #


@dataclass
class TrendResult:
    """All trend statistics for one series, plus a plain-English verdict."""

    parameter: str
    label: str
    units: str
    n: int
    start: float
    end: float
    slope: float            # OLS, units per year
    intercept: float
    r: float
    p_value: float
    stderr: float
    ci_low: float           # 95% CI on OLS slope
    ci_high: float
    mk_trend: str           # 'increasing' | 'decreasing' | 'no trend'
    mk_p: float
    mk_tau: float
    mk_hr_p: float          # Hamed-Rao autocorrelation-corrected Mann-Kendall p
    sens_slope: float       # units per year
    sens_intercept: float
    lag1_autocorr: float    # of OLS residuals
    n_eff: float            # effective sample size given lag-1 autocorrelation
    p_adjusted: float       # OLS p using n_eff
    flag: str               # from p_value: 'significant' | 'suggestive' | 'not significant'
    flag_adjusted: str      # from p_adjusted
    verdict: str = ""

    @property
    def r_squared(self) -> float:
        return self.r ** 2

    @property
    def slope_units(self) -> str:
        return slope_unit_label(self.units)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["r_squared"] = self.r_squared
        return d


def slope_unit_label(units: str) -> str:
    """'°C' -> '°C/yr'; 'mm/day' -> 'mm/day per yr'."""
    return f"{units} per yr" if "/" in units else f"{units}/yr"


def classify_p(p: float) -> str:
    """p < 0.05 -> 'significant'; 0.05 <= p < 0.10 -> 'suggestive'; else 'not significant'."""
    if not np.isfinite(p):
        return "not significant"
    if p < SIGNIFICANT_P:
        return "significant"
    if p < SUGGESTIVE_P:
        return "suggestive"
    return "not significant"


def format_p(p: float) -> str:
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def sens_slope(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Sen's (Theil-Sen) slope using the real x spacing; returns (slope, intercept).

    Computed directly rather than via ``pymannkendall.sens_slope`` because that
    assumes evenly spaced points, which breaks once missing values are dropped.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    i, j = np.triu_indices(len(x), k=1)
    dx = x[j] - x[i]
    keep = dx != 0
    slope = float(np.median((y[j] - y[i])[keep] / dx[keep]))
    intercept = float(np.median(y - slope * x))
    return slope, intercept


def lag1_autocorrelation(values: np.ndarray) -> float:
    """Lag-1 autocorrelation; values near 0 mean consecutive points are independent."""
    v = np.asarray(values, float) - np.mean(values)
    denom = np.sum(v ** 2)
    return float(np.sum(v[1:] * v[:-1]) / denom) if denom > 0 else 0.0


def effective_sample_size(n: int, r1: float) -> float:
    """n_eff = n (1 - r1) / (1 + r1) for positive lag-1 autocorrelation (else n).

    Clipped to [3, n] so the adjusted test always has at least 1 degree of freedom.
    """
    if r1 <= 0:
        return float(n)
    return float(np.clip(n * (1 - r1) / (1 + r1), 3, n))


def _adjusted_p(slope: float, stderr: float, n: int, n_eff: float) -> float:
    """Two-sided OLS slope p-value after shrinking the sample to n_eff."""
    if stderr == 0 or not np.isfinite(stderr):
        return 0.0 if slope != 0 else 1.0
    inflated = stderr * np.sqrt((n - 2) / max(n_eff - 2, 1e-9))
    t = slope / inflated
    return float(2 * stats.t.sf(abs(t), df=max(n_eff - 2, 1)))


def analyze_trend(series: pd.Series, parameter: str, units: str,
                  label: str = "") -> TrendResult:
    """Run OLS, Mann-Kendall (+Hamed-Rao) and Sen's slope on a series indexed by year."""
    s = series.dropna()
    if len(s) < 4:
        raise ValueError("Need at least 4 non-missing points for a trend test.")
    x = s.index.to_numpy(dtype=float)
    y = s.to_numpy(dtype=float)
    n = len(y)

    lr = stats.linregress(x, y)
    t_crit = stats.t.ppf(0.975, n - 2)
    ci_half = t_crit * lr.stderr
    mk_res = mk.original_test(y)
    try:
        mk_hr_p = float(mk.hamed_rao_modification_test(y).p)
    except Exception:  # very short or constant series
        mk_hr_p = float(mk_res.p)
    sen_m, sen_b = sens_slope(x, y)
    resid = y - (lr.intercept + lr.slope * x)
    r1 = lag1_autocorrelation(resid)
    n_eff = effective_sample_size(n, r1)
    p_adj = _adjusted_p(float(lr.slope), float(lr.stderr), n, n_eff)

    result = TrendResult(
        parameter=parameter, label=label or parameter, units=units, n=n,
        start=float(x.min()), end=float(x.max()),
        slope=float(lr.slope), intercept=float(lr.intercept), r=float(lr.rvalue),
        p_value=float(lr.pvalue), stderr=float(lr.stderr),
        ci_low=float(lr.slope - ci_half), ci_high=float(lr.slope + ci_half),
        mk_trend=str(mk_res.trend), mk_p=float(mk_res.p), mk_tau=float(mk_res.Tau),
        mk_hr_p=mk_hr_p, sens_slope=sen_m, sens_intercept=sen_b,
        lag1_autocorr=r1, n_eff=n_eff, p_adjusted=p_adj,
        flag=classify_p(float(lr.pvalue)), flag_adjusted=classify_p(p_adj),
    )
    result.verdict = make_verdict(result)
    return result


def make_verdict(r: TrendResult) -> str:
    """Plain-English conclusion that never overclaims."""
    up, down = TREND_WORDS.get(r.parameter, ("increasing", "decreasing"))
    word = up if r.slope > 0 else down
    su = r.slope_units
    rate = f"{r.slope:+.3f} {su}"
    per_decade = f"{r.slope * 10:+.2f} {r.units} per decade"

    if r.flag == "significant":
        text = (f"Significant {word} trend of {rate} ({format_p(r.p_value)}), "
                f"roughly {per_decade}.")
    elif r.flag == "suggestive":
        text = (f"Suggestive but not conclusive {word} trend of {rate} "
                f"({format_p(r.p_value)}). The evidence is weak; treat it as a hint "
                f"worth checking with more data, not a finding.")
    else:
        text = (f"No statistically significant trend ({format_p(r.p_value)}). "
                f"This does not mean nothing changed: the record may be too short "
                f"or too noisy to tell a real change of this size ({rate}) apart "
                f"from random variation.")

    mk_flag = classify_p(r.mk_p)
    if (mk_flag == "significant") != (r.flag == "significant"):
        text += (f" Caution: the Mann–Kendall test disagrees ({format_p(r.mk_p)}), "
                 f"so the result depends on a few unusual values.")
    else:
        text += f" The Mann–Kendall test agrees ({format_p(r.mk_p)})."
    if (r.sens_slope > 0) != (r.slope > 0) and r.slope != 0:
        text += " Sen's slope points the other way, another sign the trend is fragile."
    if r.lag1_autocorr > 0.1 and r.n_eff < r.n - 0.5:
        text += (f" Neighbouring points are correlated (lag-1 r = {r.lag1_autocorr:.2f}), "
                 f"so the {r.n} points behave like about {r.n_eff:.0f} independent ones; "
                 f"after that correction {format_p(r.p_adjusted)} ({r.flag_adjusted}).")
    return text


def explain_stats(r: TrendResult) -> pd.DataFrame:
    """Stats table: every number paired with a one-sentence translation."""
    su = r.slope_units
    rows = [
        ("Points used (n)", f"{r.n}",
         f"The test used {r.n} values from {r.start:.0f} to {r.end:.0f}."),
        ("Linear slope", f"{r.slope:+.4f} {su}",
         f"On average the value changed by {r.slope:+.4f} {r.units} each year."),
        ("95% CI of slope", f"[{r.ci_low:+.4f}, {r.ci_high:+.4f}]",
         "The true rate is plausibly anywhere in this range"
         + ("; it includes zero, so 'no change' is not ruled out."
            if r.ci_low <= 0 <= r.ci_high else "; it excludes zero.")),
        ("p-value (linear)", f"{r.p_value:.4f}",
         f"If there were no real trend, a slope this steep would appear by chance "
         f"about {r.p_value * 100:.1f}% of the time."),
        ("R²", f"{r.r_squared:.3f}",
         f"The straight line explains {r.r_squared * 100:.0f}% of the ups and downs; "
         f"the rest is variability."),
        ("Mann–Kendall", f"{r.mk_trend}, {format_p(r.mk_p)}, τ = {r.mk_tau:+.2f}",
         "A rank-based check on direction that one extreme year cannot dominate; "
         "τ runs from -1 (always falling) to +1 (always rising)."),
        ("Sen's slope", f"{r.sens_slope:+.4f} {su}",
         "The typical rate of change, robust to outliers; compare it with the "
         "linear slope."),
        ("Residual lag-1 autocorrelation", f"{r.lag1_autocorr:+.2f}",
         "Near 0 means each point is independent of the previous one, which the "
         "basic p-values assume."),
        ("Effective sample size", f"{r.n_eff:.0f} of {r.n}",
         "How many truly independent points the data is worth once "
         "autocorrelation is taken into account."),
        ("p-value, autocorrelation-corrected", f"{r.p_adjusted:.4f} (MK Hamed–Rao "
         f"{r.mk_hr_p:.4f})",
         "The same tests redone honestly for correlated data; trust these over the "
         "uncorrected ones when they differ."),
        ("Verdict", f"{r.flag} (corrected: {r.flag_adjusted})",
         "p < 0.05 is flagged 'significant', 0.05–0.10 'suggestive', otherwise "
         "'not significant'."),
    ]
    return pd.DataFrame(rows, columns=["Statistic", "Value", "In plain English"])


def regression_band(x: np.ndarray, x_obs: np.ndarray, y_obs: np.ndarray,
                    level: float = 0.95) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """OLS fit line and confidence band for the mean line, evaluated at ``x``."""
    x_obs = np.asarray(x_obs, float)
    y_obs = np.asarray(y_obs, float)
    n = len(x_obs)
    lr = stats.linregress(x_obs, y_obs)
    fit = lr.intercept + lr.slope * np.asarray(x, float)
    resid = y_obs - (lr.intercept + lr.slope * x_obs)
    s = np.sqrt(np.sum(resid ** 2) / (n - 2))
    sxx = np.sum((x_obs - x_obs.mean()) ** 2)
    se = s * np.sqrt(1.0 / n + (np.asarray(x, float) - x_obs.mean()) ** 2 / sxx)
    half = stats.t.ppf(0.5 + level / 2, n - 2) * se
    return fit, fit - half, fit + half


def seasonal_trends(df: pd.DataFrame, parameter: str) -> pd.DataFrame:
    """Trend for each calendar month separately: which months drive the change?"""
    rows = []
    for m in range(1, 13):
        r = analyze_trend(month_series(df, parameter, m), parameter, UNITS[parameter])
        rows.append({"month": MONTH_NAMES[m - 1][:3], "slope": r.slope,
                     "p_value": r.p_value, "mk_p": r.mk_p,
                     "sens_slope": r.sens_slope, "flag": r.flag})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Change point (Pettitt test)
# --------------------------------------------------------------------------- #


@dataclass
class ChangePoint:
    """Result of the Pettitt test for a single shift in the average level."""

    last_before: float      # last x value of the 'before' segment
    first_after: float      # first x value of the 'after' segment
    statistic: float        # K = max |U_t|
    p_value: float
    mean_before: float
    mean_after: float
    flag: str
    verdict: str = ""

    @property
    def shift(self) -> float:
        return self.mean_after - self.mean_before


def pettitt_test(series: pd.Series, units: str = "") -> ChangePoint:
    """Non-parametric Pettitt (1979) test for one abrupt change in the mean.

    U_t = sum_{i<=t} sum_{j>t} sign(x_j - x_i); K = max|U_t|;
    p ~= 2 exp(-6 K^2 / (n^3 + n^2)), capped at 1.
    """
    s = series.dropna()
    if len(s) < 6:
        raise ValueError("Need at least 6 points for a change-point test.")
    x = s.index.to_numpy(float)
    y = s.to_numpy(float)
    n = len(y)
    sign = np.sign(y[None, :] - y[:, None])          # sign(y_j - y_i)
    # U_t for t = 0..n-2: rows i<=t, cols j>t
    cum = np.cumsum(sign, axis=0)                    # sum over i<=t
    u = np.array([cum[t, t + 1:].sum() for t in range(n - 1)])
    t_star = int(np.argmax(np.abs(u)))
    k = float(np.abs(u[t_star]))
    p = float(min(1.0, 2 * np.exp(-6 * k ** 2 / (n ** 3 + n ** 2))))
    before, after = y[: t_star + 1], y[t_star + 1:]
    cp = ChangePoint(last_before=float(x[t_star]), first_after=float(x[t_star + 1]),
                     statistic=k, p_value=p, mean_before=float(before.mean()),
                     mean_after=float(after.mean()), flag=classify_p(p))
    when = f"{cp.first_after:.0f}" if float(cp.first_after).is_integer() \
        else f"{cp.first_after:.1f}"
    if cp.flag == "significant":
        cp.verdict = (f"Likely a sudden shift around {when} ({format_p(p)}): the average "
                      f"moved from {cp.mean_before:.2f} to {cp.mean_after:.2f} {units}. "
                      f"A 'trend' in this series may really be a step change, so a "
                      f"straight line can misdescribe it.")
    elif cp.flag == "suggestive":
        cp.verdict = (f"Weak hint of a shift around {when} ({format_p(p)}); not strong "
                      f"enough to claim a step change.")
    else:
        cp.verdict = (f"No clear sudden shift ({format_p(p)}). Any change looks gradual "
                      f"or is too small to detect in this record.")
    return cp


# --------------------------------------------------------------------------- #
# Extremes & records
# --------------------------------------------------------------------------- #


def anomaly_matrix(df: pd.DataFrame, parameter: str) -> pd.DataFrame:
    """Year x month table of anomalies (rows = years, columns = 1..12)."""
    sub = _subset(add_anomalies(df), parameter)
    return sub.pivot(index="year", columns="month", values="anomaly")


def extremes_table(df: pd.DataFrame, parameter: str, n: int = 5,
                   which: Literal["high", "low"] = "high") -> pd.DataFrame:
    """The n most extreme months relative to their own month's normal.

    Ranked by z-score (anomaly / that month's standard deviation), so a hot
    January counts as much as a hot June.
    """
    sub = _subset(add_anomalies(df), parameter).copy()
    sd = sub.groupby("month")["value"].transform("std")
    sub["z"] = sub["anomaly"] / sd.replace(0, np.nan)
    sub = sub.dropna(subset=["z"])
    sub = sub.nlargest(n, "z") if which == "high" else sub.nsmallest(n, "z")
    out = sub.assign(month=sub["month"].map(lambda m: MONTH_NAMES[m - 1]))
    return out[["year", "month", "value", "anomaly", "z"]].round(2).reset_index(drop=True)


def record_years(series: pd.Series, which: Literal["high", "low"] = "high") -> list[int]:
    """Years that set a new running record (after the first year)."""
    s = series.dropna()
    best = s.cummax() if which == "high" else s.cummin()
    is_record = (s == best) & (best != best.shift(1))
    return [int(y) for y in s.index[is_record][1:]]


# --------------------------------------------------------------------------- #
# Relationships between variables
# --------------------------------------------------------------------------- #


def wide_values(df: pd.DataFrame, scope: Scope = "monthly",
                month: Optional[int] = None) -> pd.DataFrame:
    """One column per parameter, aligned in time, for the chosen scope.

    'monthly' uses anomalies so the shared seasonal cycle does not create a
    fake relationship (rain and humidity both peak in summer).
    """
    cols = {}
    for p in sorted(df["parameter"].unique()):
        s, _, _ = build_series(df, p, scope, month, anomalies=True)
        cols[p] = s
    return pd.DataFrame(cols).dropna()


@dataclass
class Relationship:
    x: str
    y: str
    n: int
    r: float
    p_value: float
    slope: float
    intercept: float
    flag: str
    verdict: str = ""


def relationship(data: pd.DataFrame, x: str, y: str) -> Relationship:
    """Pearson correlation and fitted line between two columns of ``wide_values``."""
    d = data[[x, y]].dropna()
    lr = stats.linregress(d[x], d[y])
    rel = Relationship(x=x, y=y, n=len(d), r=float(lr.rvalue), p_value=float(lr.pvalue),
                       slope=float(lr.slope), intercept=float(lr.intercept),
                       flag=classify_p(float(lr.pvalue)))
    strength = ("strong" if abs(rel.r) >= 0.6 else "moderate" if abs(rel.r) >= 0.3
                else "weak")
    direction = "together" if rel.r > 0 else "in opposite directions"
    if rel.flag == "significant":
        rel.verdict = (f"{LONG_NAMES[x]} and {LONG_NAMES[y]} have a {strength} tendency "
                       f"to move {direction} (r = {rel.r:+.2f}, {format_p(rel.p_value)}). "
                       f"This is an association, not proof that one causes the other.")
    else:
        rel.verdict = (f"No reliable link between {LONG_NAMES[x]} and {LONG_NAMES[y]} "
                       f"here (r = {rel.r:+.2f}, {format_p(rel.p_value)}).")
    return rel


# --------------------------------------------------------------------------- #
# Reporting & demo mode
# --------------------------------------------------------------------------- #


def report_markdown(result: TrendResult, extra_sections: Optional[dict[str, str]] = None
                    ) -> str:
    """A downloadable Markdown summary of an investigation."""
    lines = [f"# {APP_NAME} investigation report", "",
             f"*Team {TEAM_NAME} · NASA POWER (MERRA-2), 24.86°N 67.01°E*", "",
             f"## {result.label}", "", f"**Verdict:** {result.verdict}", "",
             "| Statistic | Value | In plain English |", "|---|---|---|"]
    for _, row in explain_stats(result).iterrows():
        lines.append(f"| {row['Statistic']} | {row['Value']} | {row['In plain English']} |")
    for title, body in (extra_sections or {}).items():
        lines += ["", f"## {title}", "", body]
    lines += ["", "## Caveats", "",
              "25-year record; single grid point; straight-line assumption; MERRA-2 "
              "reanalysis (model data), not station observations. 'No significant "
              "trend' does not mean 'no change'."]
    return "\n".join(lines)


def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Annual and monthly-anomaly trends for every parameter plus July rain."""
    cases: list[tuple[str, Scope, Optional[int]]] = [
        (p, "annual", None) for p in PARAMETERS
    ] + [(p, "monthly", None) for p in PARAMETERS] + [("PRECTOTCORR", "month", 7)]
    rows = []
    for p, scope, month in cases:
        s, units, label = build_series(df, p, scope, month)
        r = analyze_trend(s, p, units, label)
        rows.append({"series": label, "n": r.n,
                     "slope (per yr)": f"{r.slope:+.4f} {units}",
                     "95% CI": f"[{r.ci_low:+.4f}, {r.ci_high:+.4f}]",
                     "p (OLS)": round(r.p_value, 4), "p (MK)": round(r.mk_p, 4),
                     "p (corrected)": round(r.p_adjusted, 4),
                     "Sen": round(r.sens_slope, 4), "flag": r.flag,
                     "flag (corrected)": r.flag_adjusted})
    return pd.DataFrame(rows)


def main(argv: Sequence[str]) -> None:
    path = Path(argv[1]) if len(argv) > 1 else Path(__file__).with_name(DEFAULT_CSV)
    df = load_power_csv(path)
    print(f"{APP_NAME} by {TEAM_NAME}\nLoaded {len(df)} rows: "
          f"{df['year'].min()}–{df['year'].max()}, {df['parameter'].nunique()} "
          f"parameters, {int(df['value'].isna().sum())} missing values\n")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(summary_table(df).to_string(index=False))
    for p, scope, month in [("T2M", "annual", None), ("PRECTOTCORR", "month", 7)]:
        s, units, label = build_series(df, p, scope, month)
        print(f"\n{label}:\n  {analyze_trend(s, p, units, label).verdict}")
    rain = annual_series(df, "PRECTOTCORR")
    print(f"\nChange point, annual rainfall:\n  {pettitt_test(rain, 'mm').verdict}")
    try:
        import forecast as fc
        t = annual_series(df, "T2M")
        proj = fc.project_series(t, "T2M", "°C", "Temperature, annual mean", 2035)
        bt = fc.backtest(t, "T2M", train_end=2018)
        print(f"\nForecast, annual temperature:\n  {fc.forecast_verdict(proj, bt)}")
    except ImportError:
        pass


if __name__ == "__main__":
    main(sys.argv)
