# 🔭 TeraLens

**Be a climate data detective.** An educational Streamlit app by **Team The Click**.

TeraLens lets high-school and college students, and scientifically curious readers,
explore 25 years of NASA climate data for Karachi (24.86°N, 67.01°E, 2001–2025). It
helps them tell a real trend from noise, spot sudden shifts and extremes, and make honest
forecasts, all without writing code.

## Files

| File | What it is |
|---|---|
| `analysis.py` | Loader, series builders, trend tests (+ autocorrelation correction), Pettitt shift test, extremes, correlations, Markdown report. |
| `forecast.py` | Trend projection with uncertainty, long-term-average baseline, time-travel backtest, threshold probabilities, monthly projection. |
| `plots.py` | All matplotlib figures. |
| `app.py` | Streamlit UI: guided questions, 4 demos, quiz, About page. |
| `test_analysis.py`, `test_forecast.py` | 38 pytest tests. |
| `requirements.txt` | pandas, numpy, scipy, pymannkendall, matplotlib, streamlit, pytest. |
| `POWER_Point_Monthly_..._LST.csv` | The data. Keep it next to `app.py`, or upload another POWER CSV in the sidebar. |

## Run it

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt

pytest -q                       # 38 tests
python analysis.py              # demo mode: summary table, demo verdicts, shift + forecast
streamlit run app.py            # opens TeraLens in your browser
```

## What students can do

**🔎 Investigate: three guided questions**

1. **What would you like to investigate?** Precipitation · Humidity · Temperature · Compare all three
2. **Which timeframe?** Annual · Monthly (seasonal cycle removed) · Specific month (e.g. only Julys)
3. **What do you want to know?**

| Question | What TeraLens does |
|---|---|
| Is there a trend? | Linear fit with 95% band, Mann–Kendall, Sen's slope, autocorrelation-corrected p, plain-English verdict, 12-month panel, optional moving average |
| What might happen next? | Projects the trend (up to 25 years ahead) with two uncertainty bands, compares with "no trend", runs a time-travel backtest, gives the chance of beating a threshold (default: the record) |
| Was there a sudden shift? | Pettitt change-point test, before/after averages, side-by-side with the trend test |
| Find the extremes | Year × month anomaly heatmap, top-5 extreme months (z-scored), running records |
| How are the variables related? | Correlation matrix and scatter plots, on anomalies so the shared seasonal cycle cannot fake a link |
| Show me the data | Bars or lines, table, summary metrics, CSV download |
| Compare years | Chosen years' seasonal cycles against the 25-year range |

**Demos** pre-fill the questions and add a walkthrough written from the live numbers:
A · Is Karachi getting hotter? · B · Is the monsoon getting wetter? · C · How warm could
2035 be? · D · Trend or sudden shift?

**🧩 Detective challenge.** Six quiz questions generated from the loaded data, with instant
feedback and explanations.

**📄 Reports.** Trend and forecast results can be downloaded as a Markdown report.

## The statistics in plain English

- **Linear regression**: the straight line that best follows the points. Its *slope* is the
  average change per year. The *95% CI* is the range the true slope plausibly lies in.
- **p-value**: if there were truly no trend, the chance of a slope this steep from random
  ups and downs. Labels: p < 0.05 *significant* · 0.05–0.10 *suggestive* · otherwise
  *not significant*.
- **R²**: the share of the variation the line explains.
- **Mann–Kendall**: asks only "are later values usually higher?", so one freak year can't
  dominate it.
- **Sen's slope**: the median of all pairwise slopes. It gives a typical rate that outliers
  barely move.
- **Autocorrelation correction**: when each month resembles the one before, 300 months
  carry less information than 300 independent points. TeraLens computes an *effective
  sample size* and corrected p-values (including Hamed–Rao Mann–Kendall).
- **Pettitt test**: finds the single moment where the average level most likely jumped.
  A staircase and a ramp can both produce a "significant trend".
- **Prediction interval**: where a *single future year* may land. It is wider than the
  band for the average line, because single years are noisy.
- **Backtest and skill score**: hide recent years, forecast them from older years, and
  compare with what happened. Skill > 0 means the trend beat "just use the average".
- **Exceedance probability**: the chance a future year beats a threshold, from the
  forecast's spread.
- **Correlation (r)**: −1 to +1, how strongly two variables move together. It does not
  show cause and effect.
- **"No significant trend" ≠ "no change"**: it means this record cannot separate a change
  from natural variability.

## What the data shows

| Series | Slope / yr | p | p corrected | Flag (corrected) |
|---|---|---|---|---|
| Rainfall, annual total | +14.9 mm | 0.013 | 0.022 | significant |
| Humidity, annual mean | +0.062 % | 0.29 | 0.35 | not significant |
| Temperature, annual mean | +0.012 °C | 0.12 | 0.12 | not significant |
| Rainfall, monthly anomalies | +0.040 mm/day | <0.001 | 0.002 | significant |
| Humidity, monthly anomalies | +0.062 % | 0.067 | 0.12 | not significant |
| Temperature, monthly anomalies | +0.012 °C | 0.034 | 0.11 | not significant |
| Rainfall, July only | +0.106 mm/day | 0.15 | 0.16 | not significant |

- **Temperature**: positive but not significant. The monthly test looks significant
  (p = 0.03) until autocorrelation is corrected (p = 0.11).
- **Rainfall**: the "significant trend" is better described as a **step change around
  2019** (Pettitt p = 0.027): the average rose from about 146 mm to 492 mm a year. The
  backtest confirms a straight line forecasts it badly (skill −7%, only 43% of years
  inside the 95% range).
- **Forecast to 2035, temperature**: about 26.4 °C (single-year range 25.8–27.1 °C).
  The chance that 2035 beats the 2024 record is 28% with the trend, versus 6% without it.
  Backtest skill is +12%.
- **Relationships (monthly anomalies)**: rain–humidity r = +0.28, humidity–temperature
  r = +0.23, rain–temperature not significant.

## Caveats (also in the app)

- 25-year record and a single grid point.
- Straight-line assumption.
- Autocorrelation (corrected where possible).
- Forecasts are statistical extrapolations, not climate-model projections.
- MERRA-2 is reanalysis (model data constrained by observations), not station data.
- Many tests will produce some false alarms.

## How TeraLens addresses the challenge

- **Guided flow**: three questions map to variable, series type and analysis, so students
  never have to pick a statistical method themselves. The seven Q3 options cover
  description, inference, forecasting and data exploration.
- **Definitions everywhere**: variable cards, tooltips, a statistics glossary, and a
  plain-English sentence for every number in every stats table.
- **Demos and quiz**: four walkthroughs built from live results, plus a data-driven quiz
  that checks understanding.
- **Honest significance testing**: three trend tests, autocorrelation-corrected p-values,
  a shift test that challenges misleading trends, and verdicts that say clearly when
  something is *not* shown.
- **Forecasts with humility**: two uncertainty bands, a no-trend baseline, a backtest
  that scores the forecast on real past data, and an explicit "not a climate model" note.
- **Engineering**: analysis, forecasting and plotting are separate testable modules with
  type hints and docstrings, covered by 38 tests. No ML dependencies.

---
*TeraLens · Team The Click · Data: NASA POWER / MERRA-2*
