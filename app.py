"""TeraLens: a guided climate-data detective tool (team: The Click).

Run:  streamlit run app.py
Statistics live in analysis.py and forecast.py, figures in plots.py; this file
is only the question flow and layout.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.figure import Figure

import analysis as an
import forecast as fc
import plots as pl

st.set_page_config(page_title=f"{an.APP_NAME} · {an.TEAM_NAME}", page_icon="🔭",
                   layout="wide")

VARIABLE_CHOICES = {
    "Precipitation": ["PRECTOTCORR"],
    "Humidity": ["RH2M"],
    "Temperature": ["T2M"],
    "Compare all three": ["PRECTOTCORR", "RH2M", "T2M"],
}
TIMEFRAMES = ["Annual", "Monthly", "Specific month"]
QUESTIONS = [
    "Is there a trend?",
    "What might happen next?",
    "Was there a sudden shift?",
    "Find the extremes",
    "How are the variables related?",
    "Show me the data",
    "Compare years",
]
SCOPE = {"Annual": "annual", "Monthly": "monthly", "Specific month": "month"}
FLAG_BOX = {"significant": st.success, "suggestive": st.warning,
            "not significant": st.info}

# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #


@st.cache_data(show_spinner=False)
def load_from_bytes(raw: bytes) -> pd.DataFrame:
    return an.load_power_csv(io.BytesIO(raw))


def get_data() -> Optional[pd.DataFrame]:
    st.sidebar.header("Data")
    upload = st.sidebar.file_uploader("Use a different NASA/POWER monthly CSV",
                                      type="csv")
    default = Path(__file__).with_name(an.DEFAULT_CSV)
    if upload is not None:
        raw, name = upload.getvalue(), upload.name
    elif default.exists():
        raw, name = default.read_bytes(), default.name
    else:
        st.warning(f"Put `{an.DEFAULT_CSV}` next to app.py or upload a CSV in the sidebar.")
        return None
    try:
        df = load_from_bytes(raw)
    except ValueError as exc:
        st.error(f"Could not read {name}: {exc}")
        return None
    st.sidebar.caption(f"{name}: {df['year'].min()}–{df['year'].max()}, "
                       f"{int(df['value'].isna().sum())} missing values")
    return df


# --------------------------------------------------------------------------- #
# Demos (they pre-fill the questions, then the normal flow runs)
# --------------------------------------------------------------------------- #

DEMOS = {
    "A": {"title": "A · Is Karachi getting hotter?",
          "state": {"q1": "Temperature", "q2": "Annual", "q3": "Is there a trend?"}},
    "B": {"title": "B · Is the monsoon getting wetter?",
          "state": {"q1": "Precipitation", "q2": "Specific month", "month": "July",
                    "q3": "Is there a trend?"}},
    "C": {"title": "C · How warm could 2035 be?",
          "state": {"q1": "Temperature", "q2": "Annual", "q3": "What might happen next?",
                    "horizon": 2035, "train_end": 2018}},
    "D": {"title": "D · Trend or sudden shift?",
          "state": {"q1": "Precipitation", "q2": "Annual",
                    "q3": "Was there a sudden shift?"}},
}


def start_demo(key: str) -> None:
    for k, v in DEMOS[key]["state"].items():
        st.session_state[k] = v
    st.session_state["demo"] = key


def end_demo() -> None:
    st.session_state["demo"] = None


def demo_narrative(df: pd.DataFrame, key: str) -> None:
    """Walkthrough text built from live results, so it never contradicts the data."""
    if key == "A":
        s, u, lab = an.build_series(df, "T2M", "annual")
        r = an.analyze_trend(s, "T2M", u, lab)
        sm, um, labm = an.build_series(df, "T2M", "monthly", anomalies=True)
        rm = an.analyze_trend(sm, "T2M", um, labm)
        st.markdown(f"""
**The question.** Is the air near Karachi getting warmer?

**What we did.** Averaged the 12 monthly temperatures into one value per year
(2001–2025, 25 points) and fitted a straight line.

**What we found.** The line slopes **{r.slope:+.3f} °C/yr** (about {r.slope*10:+.2f} °C
per decade), but **{an.format_p(r.p_value)}**, so the flag is **{r.flag}**. The 95%
range for the slope, [{r.ci_low:+.3f}, {r.ci_high:+.3f}],
{"includes" if r.ci_low <= 0 else "excludes"} zero.

**Detective's lesson.** A warming-looking line is not enough. Switch *Which timeframe?*
to **Monthly**: the same rate ({rm.slope:+.3f} °C/yr) gives {an.format_p(rm.p_value)}
with {rm.n} points. But neighbouring months are linked, so those points are worth only
about **{rm.n_eff:.0f}** independent ones; after correcting, {an.format_p(rm.p_adjusted)}
(**{rm.flag_adjusted}**). Bigger data is not automatically better evidence.
""")
    elif key == "B":
        s, u, lab = an.build_series(df, "PRECTOTCORR", "month", 7)
        r = an.analyze_trend(s, "PRECTOTCORR", u, lab)
        st.markdown(f"""
**The question.** Is July, the core of the monsoon, getting wetter?

**What we did.** Took only the 25 Julys and tested for a trend.

**What we found.** Slope **{r.slope:+.3f} mm/day per year**, **{an.format_p(r.p_value)}**,
flag **{r.flag}**. July {int(s.idxmax())} averaged {s.max():.1f} mm/day while
several Julys were below 0.2 mm/day.

**Detective's lesson: noise vs signal.** When single years jump this much, a trend has
to be large before a test can separate it from luck. Sen's slope ({r.sens_slope:+.3f})
is much smaller than the linear slope, a sign that a few extreme Julys pull the line up.
""")
    elif key == "C":
        s, u, lab = an.build_series(df, "T2M", "annual")
        proj = fc.project_series(s, "T2M", u, lab, 2035)
        bt = fc.backtest(s, "T2M", 2018)
        v = proj.at(2035)
        p_rec = fc.exceedance_probability(proj, 2035, float(s.max()))
        st.markdown(f"""
**The question.** If things carry on as they have, how warm might 2035 be?

**What we did.** Extended the 2001–2025 straight line to 2035 and drew two ranges: where
the *average* line could be, and where a *single year* could land.

**What we found.** About **{v['mean']:.2f} °C** in 2035, but a single year could plausibly
be anywhere from {v['pi_low']:.2f} to {v['pi_high']:.2f} °C. Chance that 2035 beats the
{int(s.idxmax())} record ({s.max():.2f} °C): **{p_rec:.0%}** with the trend, versus
{fc.exceedance_probability(proj, 2035, float(s.max()), baseline=True):.0%} with no trend.

**Detective's lesson.** Before trusting a forecast, test it on the past. Trained only on
2001–2018, the trend forecast of 2019–2025 had skill **{bt.skill:+.0%}** compared with
just using the average. Weak trends make weak forecasts, and the range matters more
than the single number.
""")
    else:
        s, u, lab = an.build_series(df, "PRECTOTCORR", "annual")
        r = an.analyze_trend(s, "PRECTOTCORR", u, lab)
        cp = an.pettitt_test(s, u)
        st.markdown(f"""
**The question.** Annual rainfall shows a *significant* upward trend
({r.slope:+.1f} mm/yr, {an.format_p(r.p_value)}). Is it really rising steadily?

**What we did.** Ran the Pettitt test, which hunts for one moment where the average
level jumped.

**What we found.** {cp.verdict}

**Detective's lesson.** A straight line fitted to "low, low, low, HIGH, HIGH" will report
a trend, even if the truth is a step change or a wet spell. Different questions need
different tools. Open *What might happen next?* for rainfall: the time-travel test shows
how poorly a straight line forecasts this kind of series.
""")


# --------------------------------------------------------------------------- #
# Result renderers
# --------------------------------------------------------------------------- #


def show_fig(fig: Figure) -> None:
    st.pyplot(fig, clear_figure=False, width="stretch")


def download_md(text: str, name: str, key: str) -> None:
    st.download_button("📄 Download report (Markdown)", text.encode(), file_name=name,
                       mime="text/markdown", key=key)


def render_trend(df, param, scope, month, anomalies, rolling) -> None:
    s, units, label = an.build_series(df, param, scope, month, anomalies)
    r = an.analyze_trend(s, param, units, label)
    FLAG_BOX[r.flag](r.verdict)
    window = (5 if scope != "monthly" else 13) if rolling else None
    show_fig(pl.plot_trend(s, r, rolling=window))
    st.caption("Blue: the data. Orange line: best straight-line fit; the orange band is "
               "its 95% confidence interval. Grey dashed: Sen's slope, a fit that "
               "outliers barely move."
               + (" Black: moving average, which shows wiggles a straight line hides."
                  if window else ""))
    st.dataframe(an.explain_stats(r), hide_index=True, width="stretch")
    if scope != "month":
        with st.expander("Which months drive the trend? (12-panel view)"):
            show_fig(pl.plot_seasonal_panels(df, param))
            st.caption("Each panel is one calendar month across 2001–2025. "
                       "With 12 tests, about one may look 'significant' by chance alone.")
    download_md(an.report_markdown(r), f"teralens_{param}_{scope}_trend.md",
                f"rep_t_{param}")


def render_forecast(df, param, scope, month, anomalies, horizon, train_end) -> None:
    if scope == "monthly":
        last = int(df["year"].max())
        end = min(horizon, last + 5)
        mf = fc.project_monthly(df, param, end)
        st.info(f"Month-by-month projection to {end} (capped at 5 years ahead: monthly "
                "ranges get wide fast). Each month = its normal value + the long-term "
                "trend; the band uses that month's own variability."
                + (" For rain, the trend is scaled to each month's normal."
                   if param == "PRECTOTCORR" else ""))
        show_fig(pl.plot_monthly_forecast(df, param, mf))
        table = mf.assign(month=mf["month"].map(lambda m: an.MONTH_NAMES[m - 1][:3]))
        table = table[["year", "month", "mean", "pi_low", "pi_high", "climatology"]].round(2)
        st.dataframe(table, hide_index=True, width="stretch", height=240)
        st.download_button("Download projection CSV", table.to_csv(index=False).encode(),
                           file_name=f"teralens_{param}_monthly_projection.csv",
                           key=f"dl_mf_{param}")
        return

    s, units, label = an.build_series(df, param, scope, month, anomalies)
    proj = fc.project_series(s, param, units, label, horizon)
    try:
        bt = fc.backtest(s, param, train_end)
    except ValueError:
        bt = None
    record = float(s.max())
    thr = st.number_input(
        f"Threshold to check ({units})", value=record, format="%.2f",
        key=f"thr_{param}_{scope}_{month}",
        help=f"Defaults to the record: {record:.2f} in {int(s.idxmax())}.")

    st.info(fc.forecast_verdict(proj, bt))
    show_fig(pl.plot_forecast(s, proj, threshold=thr))
    st.caption("Dashed orange: the trend extended. Dark band: where the average line could "
               "be. Light band: where a single year could land (wider, because single "
               "years are noisy). Dotted grey: the 'no trend' forecast, the long-term "
               "average.")

    v = proj.at(horizon)
    p_trend = fc.exceedance_probability(proj, horizon, thr)
    p_base = fc.exceedance_probability(proj, horizon, thr, baseline=True)
    p_any = 1 - np.prod([1 - fc.exceedance_probability(proj, int(y), thr)
                         for y in proj.years])
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"Trend forecast {horizon}", f"{v['mean']:.2f} {units}",
              help="The straight line extended; the single most likely value.")
    c2.metric("Single-year range", f"{v['pi_low']:.2f} – {v['pi_high']:.2f}",
              help=an.STAT_GLOSSARY["prediction_interval"])
    c3.metric(f"Chance {horizon} > threshold", f"{p_trend:.0%}",
              delta=f"{(p_trend - p_base) * 100:+.0f} pts vs no-trend", delta_color="off",
              help=f"With no trend it would be {p_base:.0%}.")
    c4.metric(f"Chance any year to {horizon} > threshold", f"{p_any:.0%}",
              help="Assumes years are independent, which is roughly true here.")

    if bt is not None:
        st.markdown("##### Can we trust it? The time-travel test")
        show_fig(pl.plot_backtest(s, bt, units))
        FLAG_BOX["significant" if bt.skill > 0.1 else "suggestive" if bt.skill >= -0.1
                 else "not significant"](bt.explanation)

    with st.expander("Projection table"):
        table = proj.to_frame()
        st.dataframe(table, hide_index=True, width="stretch")
        st.download_button("Download projection CSV", table.to_csv(index=False).encode(),
                           file_name=f"teralens_{param}_{scope}_projection.csv",
                           key=f"dl_proj_{param}")
    download_md(an.report_markdown(proj.trend, {"Forecast": fc.forecast_verdict(proj, bt)}),
                f"teralens_{param}_{scope}_forecast.md", f"rep_f_{param}")


def render_shift(df, param, scope, month, anomalies) -> None:
    s, units, label = an.build_series(df, param, scope, month, anomalies)
    cp = an.pettitt_test(s, units)
    r = an.analyze_trend(s, param, units, label)
    FLAG_BOX[cp.flag](cp.verdict)
    show_fig(pl.plot_changepoint(s, cp, units, label))
    st.markdown(f"""
| Tool | Asks | Answer here |
|---|---|---|
| Trend test | Is there a steady rise or fall? | {r.flag} ({an.format_p(r.p_value)}) |
| Pettitt shift test | Did the average jump at one moment? | {cp.flag} ({an.format_p(cp.p_value)}) |
""")
    st.caption(an.STAT_GLOSSARY["change_point"] + " If both tests fire, look at the chart: "
               "a staircase and a ramp can produce the same trend p-value.")


def render_extremes(df, param, scope, month, anomalies) -> None:
    show_fig(pl.plot_anomaly_heatmap(df, param))
    st.caption("Each cell is one month. Colour shows how far it was from that month's "
               "2001–2025 normal; white is normal.")
    c1, c2 = st.columns(2)
    c1.markdown("**Most extreme high months**")
    c1.dataframe(an.extremes_table(df, param, 5, "high"), hide_index=True, width="stretch")
    c2.markdown("**Most extreme low months**")
    c2.dataframe(an.extremes_table(df, param, 5, "low"), hide_index=True, width="stretch")
    st.caption("z = how many 'typical swings' (standard deviations) away from normal. "
               "Beyond ±2 happens in roughly 1 month in 20.")
    s, units, label = an.build_series(df, param, scope, month, anomalies)
    if scope != "monthly":
        highs, lows = an.record_years(s, "high"), an.record_years(s, "low")
        st.markdown(f"**Running records ({label}).** New highs: "
                    f"{', '.join(map(str, highs)) or 'none'}. New lows: "
                    f"{', '.join(map(str, lows)) or 'none'}. In a stable climate, "
                    f"new records get rarer over time; a cluster of recent highs is a clue.")


def render_relationships(df, scope, month) -> None:
    data = an.wide_values(df, scope, month)
    units = {p: (an.ANNUAL_UNITS if scope == "annual" else an.UNITS)[p] for p in data}
    basis = {"annual": "yearly values", "month": "values for the chosen month",
             "monthly": "monthly anomalies (seasonal cycle removed)"}[scope]
    st.caption(f"Based on {basis}, n = {len(data)}. "
               + ("Anomalies are used so that the shared summer peak does not create a "
                  "fake link." if scope == "monthly" else "")
               + " " + an.STAT_GLOSSARY["correlation"])
    left, right = st.columns([1, 2])
    with left:
        show_fig(pl.plot_correlation_matrix(data))
    pairs = [("PRECTOTCORR", "RH2M"), ("RH2M", "T2M"), ("PRECTOTCORR", "T2M")]
    with right:
        tabs = st.tabs([f"{an.LONG_NAMES[a]} vs {an.LONG_NAMES[b]}" for a, b in pairs])
        for tab, (a, b) in zip(tabs, pairs):
            with tab:
                rel = an.relationship(data, a, b)
                FLAG_BOX[rel.flag](rel.verdict)
                show_fig(pl.plot_relationship(data, rel, units[a], units[b]))


def render_data(df, param, scope, month, anomalies) -> None:
    s, units, label = an.build_series(df, param, scope, month, anomalies)
    if scope == "annual":
        show_fig(pl.plot_annual_bars(df, param))
        st.caption("Rain is summed into a yearly total (mm); temperature and humidity "
                   "are averaged, because adding up temperatures has no meaning.")
    else:
        show_fig(pl.plot_values(s, units, label, bars=(scope == "month")))
    table = s.rename("value").rename_axis("year").reset_index()
    c1, c2 = st.columns([3, 1])
    c1.dataframe(table, hide_index=True, width="stretch", height=240)
    c2.metric("Mean", f"{s.mean():.2f} {units}")
    c2.metric("Lowest", f"{s.min():.2f}", help=f"in {s.idxmin():.0f}")
    c2.metric("Highest", f"{s.max():.2f}", help=f"in {s.idxmax():.0f}")
    c2.download_button("Download CSV", table.to_csv(index=False).encode(),
                       file_name=f"teralens_{param}_{scope}.csv", key=f"dl_{param}")


def render_compare_years(df, param, years) -> None:
    show_fig(pl.plot_compare_years(df, param, years))
    clim = an.monthly_climatology(df, param)
    rows = {}
    for y in years:
        vals = df.query("parameter == @param and year == @y").set_index("month")["value"]
        rows[str(y)] = (vals - clim).round(2)
    if rows:
        diff = pd.DataFrame(rows)
        diff.index = [m[:3] for m in an.MONTH_NAMES]
        st.caption(f"Difference from the 2001–2025 average for each month "
                   f"({an.UNITS[param]}); positive = above normal.")
        st.dataframe(diff.T, width="stretch")


# --------------------------------------------------------------------------- #
# Tabs
# --------------------------------------------------------------------------- #

DEFAULT_STATE = {"q1": "Temperature", "q2": "Annual", "q3": "Is there a trend?",
                 "month": "July", "anom": True, "years": [2001, 2022, 2025],
                 "demo": None, "horizon": 2035, "train_end": 2018, "rolling": False}

CAVEATS = """
- **Short record (25 years).** Climate trends are usually judged over 30+ years; with
  25 points, modest real trends often fail to reach significance.
- **Single location.** One grid point (about 50 × 60 km) is not "Pakistan" or even
  all of Karachi.
- **Straight-line assumption.** A jump, a cycle or a recent acceleration can be missed or
  misrepresented. The *sudden shift* question checks for one kind of break.
- **Autocorrelation.** Basic tests assume independent points. TeraLens reports corrected
  p-values (effective sample size, Hamed–Rao Mann–Kendall); trust those when they differ.
- **Forecasts are extrapolations.** They extend past behaviour; they are not climate-model
  projections and know nothing about emissions, El Niño or city growth.
- **Model data, not observations.** MERRA-2 is a reanalysis: a weather model constrained
  by observations. Rainfall is the least reliable of the three variables.
- **Many tests, some false alarms.** Trying many variables and months means about 1 in 20
  tests will look "significant" by chance.
- **"No significant trend" ≠ "no change".** It means this data cannot tell a change apart
  from natural variability.
"""


def investigate_tab(df: pd.DataFrame) -> None:
    st.subheader("What the variables mean")
    cols = st.columns(3)
    for col, p in zip(cols, an.PARAMETERS):
        col.info(f"**{an.LONG_NAMES[p]}** (`{p}`, {an.UNITS[p]})\n\n{an.DEFINITIONS[p]}")

    st.subheader("Try a guided demo")
    dcols = st.columns(5)
    for col, key in zip(dcols, DEMOS):
        col.button(DEMOS[key]["title"], on_click=start_demo, args=(key,), width="stretch")
    if st.session_state.get("demo"):
        dcols[4].button("Exit demo", on_click=end_demo, width="stretch")

    st.subheader("Your investigation")
    q1c, q2c, q3c = st.columns(3)
    var = q1c.radio("Q1. What would you like to investigate?", list(VARIABLE_CHOICES),
                    key="q1", on_change=end_demo,
                    help=" | ".join(f"{an.LONG_NAMES[p]}: {an.DEFINITIONS[p]}"
                                    for p in an.PARAMETERS))
    timeframe = q2c.radio("Q2. Which timeframe?", TIMEFRAMES, key="q2", on_change=end_demo,
                          help="Annual: one value per year. Monthly: all 300 months. "
                               "Specific month: e.g. only Julys.")
    question = q3c.radio("Q3. What do you want to know?", QUESTIONS, key="q3",
                         on_change=end_demo)

    month: Optional[int] = None
    anomalies = True
    if timeframe == "Specific month":
        name = q2c.selectbox("Which month?", an.MONTH_NAMES, key="month", on_change=end_demo)
        month = an.MONTH_NAMES.index(name) + 1
    elif timeframe == "Monthly":
        anomalies = q2c.checkbox("Remove the seasonal cycle (use anomalies)", key="anom",
                                 help=an.STAT_GLOSSARY["anomaly"])
        if question in ("Is there a trend?", "Was there a sudden shift?") and not anomalies:
            q2c.caption("⚠️ Without anomalies, the summer/winter swing dominates the test.")

    last_year = int(df["year"].max())
    rolling, horizon, train_end, years = False, 2035, 2018, []
    if question == "Is there a trend?":
        rolling = q3c.checkbox("Show a moving average", key="rolling")
    elif question == "What might happen next?":
        horizon = q3c.slider("Forecast up to", last_year + 1, last_year + 25, key="horizon")
        train_end = q3c.slider("Time-travel test: train on years up to", 2008,
                               last_year - 3, key="train_end",
                               help="The later years are hidden, forecast, then compared "
                                    "with what really happened.")
    elif question == "Compare years":
        years = q3c.multiselect("Which years?", sorted(df["year"].unique().tolist()),
                                max_selections=6, key="years")
        q3c.caption("Year comparison always uses the 12 monthly values.")
    elif question in ("Find the extremes",):
        q3c.caption("The heatmap and extremes always use all months.")

    demo = st.session_state.get("demo")
    if demo:
        with st.container(border=True):
            st.markdown(f"### Demo {DEMOS[demo]['title']}")
            demo_narrative(df, demo)

    params = VARIABLE_CHOICES[var]
    scope = SCOPE[timeframe]
    st.divider()

    if question == "How are the variables related?":
        render_relationships(df, scope, month)
    else:
        if question == "Is there a trend?" and len(params) > 1:
            rows = []
            for p in params:
                s, u, lab = an.build_series(df, p, scope, month, anomalies)
                r = an.analyze_trend(s, p, u, lab)
                rows.append({"Variable": lab, "Slope": f"{r.slope:+.4f} {r.slope_units}",
                             "95% CI": f"[{r.ci_low:+.4f}, {r.ci_high:+.4f}]",
                             "p (linear)": an.format_p(r.p_value),
                             "p (corrected)": an.format_p(r.p_adjusted),
                             "p (Mann–Kendall)": an.format_p(r.mk_p), "Flag": r.flag})
            st.markdown("#### Side-by-side summary")
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
            st.caption("Slopes are in different units, so compare flags and p-values, "
                       "not the size of the numbers.")

        tabs = (st.tabs([an.LONG_NAMES[p] for p in params]) if len(params) > 1
                else [st.container()])
        for tab, p in zip(tabs, params):
            with tab:
                st.markdown(f"#### {an.LONG_NAMES[p]}  \n*{an.DEFINITIONS[p]}*")
                if question == "Is there a trend?":
                    render_trend(df, p, scope, month, anomalies, rolling)
                elif question == "What might happen next?":
                    render_forecast(df, p, scope, month, anomalies, horizon, train_end)
                elif question == "Was there a sudden shift?":
                    render_shift(df, p, scope, month, anomalies)
                elif question == "Find the extremes":
                    render_extremes(df, p, scope, month, anomalies)
                elif question == "Show me the data":
                    render_data(df, p, scope, month, anomalies)
                else:
                    render_compare_years(df, p, years)

    with st.expander("⚠️ Caveats: read before drawing conclusions"):
        st.markdown(CAVEATS)


def build_quiz(df: pd.DataFrame) -> list[dict]:
    """Questions generated from the live data, each with an explanation."""
    rain = an.annual_series(df, "PRECTOTCORR")
    ranked = rain.sort_values(ascending=False)
    options = sorted({int(ranked.index[0]), int(ranked.index[1]), int(rain.idxmin()),
                      int(ranked.index[len(ranked) // 2])})
    temp = an.analyze_trend(an.annual_series(df, "T2M"), "T2M", "°C")
    cp = an.pettitt_test(rain, "mm")
    t_ans = {"significant": "Yes, p < 0.05", "suggestive": "Only suggestive",
             "not significant": "No"}[temp.flag]
    shift_ans = (f"A jump around {cp.first_after:.0f}" if cp.flag == "significant"
                 else "No clear jump")
    return [
        {"q": "Which year had the highest total rainfall?",
         "options": [str(o) for o in options], "answer": str(int(ranked.index[0])),
         "why": f"{int(ranked.index[0])} had {ranked.iloc[0]:.0f} mm, about "
                f"{ranked.iloc[0] / rain.mean():.1f}× the 25-year average."},
        {"q": "Is the annual temperature trend statistically significant?",
         "options": ["Yes, p < 0.05", "Only suggestive", "No"], "answer": t_ans,
         "why": f"Slope {temp.slope:+.3f} °C/yr with {an.format_p(temp.p_value)}."},
        {"q": "Did annual rainfall rise steadily, or jump?",
         "options": [f"A jump around {cp.first_after:.0f}", "A steady rise every year",
                     "No clear jump"], "answer": shift_ans, "why": cp.verdict},
        {"q": "A trend test gives p = 0.30. What can you conclude?",
         "options": ["Nothing is changing",
                     "The data cannot tell a change apart from random variation",
                     "There is a 30% chance the trend is real"],
         "answer": "The data cannot tell a change apart from random variation",
         "why": "A large p-value means 'not proven', not 'proven absent'. It is also not "
                "the probability that the trend is real."},
        {"q": "A forecast has a skill score of −20%. What does that mean?",
         "options": ["It was 20% accurate", "It did worse than just using the average",
                     "It was 20% better than the average"],
         "answer": "It did worse than just using the average",
         "why": an.STAT_GLOSSARY["skill_score"]},
        {"q": "Rain and humidity both peak in summer. Why does TeraLens remove the "
              "seasonal cycle before correlating monthly values?",
         "options": ["To make the numbers smaller",
                     "Because a shared season creates a link even if they are unrelated",
                     "Because NASA requires it"],
         "answer": "Because a shared season creates a link even if they are unrelated",
         "why": "Anything that peaks in summer would 'correlate' with rain. Anomalies "
                "ask the better question: is a wetter-than-normal month also more humid "
                "than normal?"},
    ]


def challenge_tab(df: pd.DataFrame) -> None:
    st.subheader("🧩 Detective challenge")
    st.write("Answer from what you found in *Investigate*. Every question is built from "
             "the data loaded right now.")
    quiz = build_quiz(df)
    score = answered = 0
    for i, item in enumerate(quiz, start=1):
        choice = st.radio(f"**{i}. {item['q']}**", item["options"], index=None,
                          key=f"quiz_{i}")
        if choice is not None:
            answered += 1
            if choice == item["answer"]:
                score += 1
                st.success(f"Correct. {item['why']}")
            else:
                st.error(f"Not quite: the answer is “{item['answer']}”. {item['why']}")
    if answered:
        st.metric("Score", f"{score} / {len(quiz)}", help=f"{answered} answered")
        if answered == len(quiz) and score == len(quiz):
            st.balloons()


def about_tab() -> None:
    st.subheader(f"About {an.APP_NAME}")
    st.markdown(f"""
**{an.APP_NAME}** is an educational climate-data tool built by **Team {an.TEAM_NAME}**.
It lets students and curious readers investigate 25 years of NASA climate data
for Karachi and learn to tell real signals from noise, without writing code.

**Data.** NASA POWER, MERRA-2 reanalysis, monthly values for 24.86°N, 67.01°E,
2001–2025: rainfall (PRECTOTCORR), relative humidity (RH2M), temperature (T2M).

**Methods.**
- Trend: linear regression with 95% CI · Mann–Kendall · Sen's slope
- Autocorrelation correction: effective sample size · Hamed–Rao Mann–Kendall
- Sudden shifts: Pettitt change-point test
- Forecasts: trend extrapolation with prediction intervals, long-term-average baseline,
  time-travel backtest with skill score, threshold-exceedance probabilities
- Extremes: anomaly heatmap, z-scored extreme months, running records
- Relationships: Pearson correlation on anomalies

**Significance labels.** p < 0.05 *significant* · 0.05–0.10 *suggestive* · otherwise
*not significant*.
""")
    st.markdown("**Caveats**")
    st.markdown(CAVEATS)


def main() -> None:
    for k, v in DEFAULT_STATE.items():
        st.session_state.setdefault(k, v)
    st.title(f"🔭 {an.APP_NAME}")
    st.markdown(f"**Be a climate data detective** · Karachi, 2001–2025 · "
                f"by Team *{an.TEAM_NAME}*")

    df = get_data()
    with st.sidebar.expander("Glossary of statistics", expanded=False):
        for k, v in an.STAT_GLOSSARY.items():
            st.markdown(f"**{k.replace('_', ' ').capitalize()}**: {v}")
    st.sidebar.caption(f"{an.APP_NAME} · Team {an.TEAM_NAME} · Data: NASA POWER / MERRA-2")
    if df is None:
        return

    t1, t2, t3 = st.tabs(["🔎 Investigate", "🧩 Detective challenge", "ℹ️ About"])
    with t1:
        investigate_tab(df)
    with t2:
        challenge_tab(df)
    with t3:
        about_tab()


main()
