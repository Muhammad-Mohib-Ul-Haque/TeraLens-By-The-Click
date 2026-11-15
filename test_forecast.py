"""Unit tests for forecast.py. Run with:  pytest -q"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest

import analysis as an
import forecast as fc
from test_analysis import make_csv

YEARS = np.arange(2001, 2026)


@pytest.fixture
def linear() -> pd.Series:
    rng = np.random.default_rng(7)
    return pd.Series(20 + 0.1 * (YEARS - 2001) + rng.normal(0, 0.1, len(YEARS)),
                     index=YEARS)


def test_projection_extends_the_line(linear):
    proj = fc.project_series(linear, "T2M", "°C", "", 2035)
    assert proj.years[0] == 2026 and proj.years[-1] == 2035
    assert proj.at(2035)["mean"] == pytest.approx(20 + 0.1 * 34, abs=0.15)
    assert np.all(proj.pi_low < proj.ci_low) and np.all(proj.pi_high > proj.ci_high)
    widths = proj.pi_high - proj.pi_low
    assert np.all(np.diff(widths) > 0)  # uncertainty grows with distance


def test_projection_respects_physical_bounds():
    s = pd.Series(np.linspace(5, 0.2, 25), index=YEARS)
    s.iloc[::3] += 0.3
    proj = fc.project_series(s, "PRECTOTCORR", "mm/day", "", 2045)
    assert proj.mean.min() >= 0 and proj.pi_low.min() >= 0


def test_horizon_must_be_in_future(linear):
    with pytest.raises(ValueError):
        fc.project_series(linear, "T2M", "°C", "", 2025)


def test_exceedance_probability(linear):
    proj = fc.project_series(linear, "T2M", "°C", "", 2030)
    centre = proj.intercept + proj.slope * 2030
    assert fc.exceedance_probability(proj, 2030, centre) == pytest.approx(0.5, abs=1e-6)
    assert fc.exceedance_probability(proj, 2030, centre + 5) < 0.001
    # with a real upward trend, the no-trend baseline underestimates the chance
    assert (fc.exceedance_probability(proj, 2030, linear.max())
            > fc.exceedance_probability(proj, 2030, linear.max(), baseline=True))


def test_backtest_rewards_real_trend(linear):
    bt = fc.backtest(linear, "T2M", train_end=2018)
    assert list(bt.years) == list(range(2019, 2026))
    assert bt.skill > 0.5
    assert bt.coverage >= 0.8
    assert "trend added real forecasting value" in bt.explanation


def test_backtest_no_skill_on_noise():
    rng = np.random.default_rng(8)
    s = pd.Series(rng.normal(0, 1, 25), index=YEARS)
    assert fc.backtest(s, "T2M", 2018).skill < 0.2


def test_backtest_needs_enough_years(linear):
    with pytest.raises(ValueError):
        fc.backtest(linear, "T2M", train_end=2005)


def test_project_monthly_shape_and_bounds():
    df = an.load_power_csv(io.StringIO(make_csv()))
    mf = fc.project_monthly(df, "PRECTOTCORR", 2028)
    assert len(mf) == 3 * 12
    assert (mf["pi_low"] >= 0).all()
    assert mf.groupby("year").size().eq(12).all()


def test_forecast_verdict_states_limits(linear):
    proj = fc.project_series(linear, "T2M", "°C", "Temperature", 2035)
    text = fc.forecast_verdict(proj, fc.backtest(linear, "T2M", 2018))
    assert "not a climate-model projection" in text
    assert "Time-travel test" in text
