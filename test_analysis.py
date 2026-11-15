"""Unit tests for analysis.py. Run with:  pytest -q"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import analysis as an

HEADER = """-BEGIN HEADER-
NASA/POWER Source Native Resolution Monthly and Annual
Location: Latitude  24.86   Longitude 67.01
The value for missing source data ...: -999
-END HEADER-
"""
COLS = "PARAMETER,YEAR,JAN,FEB,MAR,APR,MAY,JUN,JUL,AUG,SEP,OCT,NOV,DEC,ANN\n"


def make_csv(years=range(2001, 2026), params=("PRECTOTCORR", "RH2M", "T2M"),
             missing: tuple[str, int, int] | None = None, header: bool = True) -> str:
    """Synthetic POWER-style CSV. ``missing=(param, year, month)`` writes -999 there."""
    lines = []
    for p in params:
        for y in years:
            vals = [float(m) for m in range(1, 13)]
            if missing and missing[0] == p and missing[1] == y:
                vals[missing[2] - 1] = -999
            lines.append(",".join([p, str(y), *map(str, vals), "6.5"]))
    return (HEADER if header else "") + COLS + "\n".join(lines) + "\n"


# ----------------------------- loader --------------------------------------


def test_header_is_skipped_and_columns_are_tidy():
    df = an.load_power_csv(io.StringIO(make_csv()))
    assert list(df.columns) == ["year", "month", "parameter", "value"]
    assert set(df["parameter"]) == {"PRECTOTCORR", "RH2M", "T2M"}
    assert df["month"].between(1, 12).all()
    assert df["year"].dtype.kind == "i"


def test_works_without_header_block_and_from_bytes():
    df = an.load_power_csv(io.BytesIO(make_csv(header=False).encode()))
    assert len(df) == 25 * 3 * 12


def test_row_count_is_25_years_by_3_params():
    wide = an.load_power_wide(io.StringIO(make_csv()))
    assert len(wide) == 25 * 3
    df = an.load_power_csv(io.StringIO(make_csv()))
    assert len(df) == 25 * 3 * 12
    assert df.groupby("parameter")["year"].nunique().eq(25).all()


def test_minus_999_becomes_nan():
    df = an.load_power_csv(io.StringIO(make_csv(missing=("T2M", 2010, 7))))
    cell = df.query("parameter == 'T2M' and year == 2010 and month == 7")["value"]
    assert cell.isna().all()
    assert df["value"].isna().sum() == 1
    assert (df["value"] <= -999).sum() == 0


def test_missing_month_blanks_that_year_in_annual_series():
    df = an.load_power_csv(io.StringIO(make_csv(missing=("T2M", 2010, 7))))
    s = an.annual_series(df, "T2M")
    assert np.isnan(s[2010])
    assert s.drop(2010).notna().all()


def test_no_column_row_raises():
    with pytest.raises(ValueError):
        an.load_power_csv(io.StringIO("just,some\n1,2\n"))


def test_annual_precip_is_a_total_in_mm():
    df = an.load_power_csv(io.StringIO(make_csv(years=[2001, 2004])))
    s = an.annual_series(df, "PRECTOTCORR")
    expected_2001 = sum(m * d for m, d in zip(range(1, 13),
                        [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]))
    assert s[2001] == pytest.approx(expected_2001)
    assert s[2004] == pytest.approx(expected_2001 + 2)  # leap year: Feb has 29 days


REAL = Path(__file__).with_name(an.DEFAULT_CSV)


@pytest.mark.skipif(not REAL.exists(), reason="real NASA/POWER CSV not present")
def test_real_file_shape():
    df = an.load_power_csv(REAL)
    assert len(df) == 25 * 3 * 12
    assert (df["year"].min(), df["year"].max()) == (2001, 2025)


# ----------------------------- trends --------------------------------------


@pytest.fixture
def known_slope_series() -> pd.Series:
    rng = np.random.default_rng(42)
    years = np.arange(2001, 2026)
    return pd.Series(20 + 0.05 * (years - 2001) + rng.normal(0, 0.05, len(years)),
                     index=years)


def test_recovers_known_slope(known_slope_series):
    r = an.analyze_trend(known_slope_series, "T2M", "°C")
    assert r.slope == pytest.approx(0.05, abs=0.005)
    assert r.sens_slope == pytest.approx(0.05, abs=0.005)
    assert r.ci_low < 0.05 < r.ci_high
    assert r.flag == "significant"
    assert r.mk_trend == "increasing"
    assert r.verdict.startswith("Significant warming trend of +0.05")


def test_pure_noise_is_not_significant():
    rng = np.random.default_rng(0)
    s = pd.Series(rng.normal(0, 1, 25), index=np.arange(2001, 2026))
    r = an.analyze_trend(s, "PRECTOTCORR", "mm/day")
    assert r.flag == "not significant"
    assert "does not mean nothing changed" in r.verdict


def test_negative_slope_wording():
    years = np.arange(2001, 2026)
    s = pd.Series(10 - 0.2 * (years - 2001), index=years, dtype=float)
    s.iloc[3] += 0.1  # avoid a perfect fit
    r = an.analyze_trend(s, "PRECTOTCORR", "mm/day")
    assert r.slope == pytest.approx(-0.2, abs=0.01)
    assert "drying" in r.verdict


def test_sens_slope_handles_gaps_and_outliers():
    x = np.array([2001, 2002, 2004, 2005, 2008, 2010], float)
    y = 3 * (x - 2001)
    y[2] += 100  # one wild outlier
    slope, _ = an.sens_slope(x, y)
    assert slope == pytest.approx(3.0)


def test_nan_values_are_dropped(known_slope_series):
    s = known_slope_series.copy()
    s.iloc[[2, 10]] = np.nan
    r = an.analyze_trend(s, "T2M", "°C")
    assert r.n == 23
    assert r.slope == pytest.approx(0.05, abs=0.006)


@pytest.mark.parametrize("p,flag", [(0.001, "significant"), (0.0499, "significant"),
                                    (0.05, "suggestive"), (0.0999, "suggestive"),
                                    (0.10, "not significant"), (0.8, "not significant")])
def test_classify_p_thresholds(p, flag):
    assert an.classify_p(p) == flag


def test_monthly_anomalies_remove_seasonal_cycle():
    df = an.load_power_csv(io.StringIO(make_csv()))
    s = an.monthly_series(df, "T2M", anomalies=True)
    assert len(s) == 300
    assert np.allclose(s.to_numpy(), 0)  # synthetic data is a pure seasonal cycle


def test_regression_band_brackets_fit(known_slope_series):
    x = known_slope_series.index.to_numpy(float)
    fit, lo, hi = an.regression_band(x, x, known_slope_series.to_numpy())
    assert np.all(lo < fit) and np.all(fit < hi)


def test_explain_stats_has_plain_english_for_every_row(known_slope_series):
    table = an.explain_stats(an.analyze_trend(known_slope_series, "T2M", "°C"))
    assert table["In plain English"].str.len().gt(20).all()


# ----------------------- autocorrelation, shifts, extremes -----------------


def test_effective_sample_size():
    assert an.effective_sample_size(30, -0.2) == 30
    assert an.effective_sample_size(30, 0.5) == pytest.approx(10)
    assert an.effective_sample_size(10, 0.99) == 3  # clipped


def test_autocorrelated_series_gets_larger_p():
    rng = np.random.default_rng(1)
    e = np.zeros(120)
    for i in range(1, 120):
        e[i] = 0.8 * e[i - 1] + rng.normal()
    s = pd.Series(0.01 * np.arange(120) + e, index=2000 + np.arange(120) / 12)
    r = an.analyze_trend(s, "T2M", "°C")
    assert r.lag1_autocorr > 0.5
    assert r.n_eff < r.n / 2
    assert r.p_adjusted > r.p_value
    assert "independent ones" in r.verdict


def test_pettitt_finds_a_step():
    rng = np.random.default_rng(3)
    years = np.arange(2001, 2026)
    y = np.where(years < 2015, 10.0, 15.0) + rng.normal(0, 0.5, len(years))
    cp = an.pettitt_test(pd.Series(y, index=years), "mm")
    assert cp.first_after == 2015
    assert cp.flag == "significant"
    assert cp.shift == pytest.approx(5, abs=0.6)


def test_pettitt_quiet_on_noise():
    rng = np.random.default_rng(4)
    cp = an.pettitt_test(pd.Series(rng.normal(0, 1, 25), index=np.arange(2001, 2026)))
    assert cp.flag == "not significant"


def test_extremes_and_records():
    df = an.load_power_csv(io.StringIO(make_csv()))
    df.loc[(df.parameter == "T2M") & (df.year == 2020) & (df.month == 6), "value"] += 5
    top = an.extremes_table(df, "T2M", 3, "high")
    assert len(top) == 3
    assert (top.iloc[0]["year"], top.iloc[0]["month"]) == (2020, "June")
    assert top["z"].is_monotonic_decreasing
    s = pd.Series([1, 3, 2, 4, 4, 5], index=range(2001, 2007))
    assert an.record_years(s, "high") == [2002, 2004, 2006]


def test_relationship_detects_link():
    rng = np.random.default_rng(5)
    a = rng.normal(size=200)
    data = pd.DataFrame({"PRECTOTCORR": a, "RH2M": 2 * a + rng.normal(0, 0.5, 200)})
    rel = an.relationship(data, "PRECTOTCORR", "RH2M")
    assert rel.r > 0.9 and rel.flag == "significant"
    assert "not proof" in rel.verdict


def test_report_markdown(known_slope_series):
    md = an.report_markdown(an.analyze_trend(known_slope_series, "T2M", "°C"),
                            {"Forecast": "hello"})
    assert an.APP_NAME in md and an.TEAM_NAME in md
    assert "## Forecast" in md and "| Linear slope |" in md
