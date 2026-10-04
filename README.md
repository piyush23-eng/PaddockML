# PaddockML

**A Formula 1 performance intelligence workspace for exploring race results, pit-stop execution, predictive modelling and strategy sensitivity.**

PaddockML turns historical Formula 1 data into an interactive Streamlit app designed for data-science and performance-engineering conversations. It combines a season dashboard, race-by-race comparisons, pit-stop analytics, a time-aware machine-learning benchmark and a transparent strategy what-if tool.

> The aim is decision support, not false precision: model quality and data limitations are visible in the app.

## Highlights

- Explore Grand Prix race-result data from **2012 to 2025**.
- Compare driver form race by race and inspect qualifying-position versus finishing-position patterns.
- Analyse pit-stop duration by season and driver, with outlier filtering.
- Benchmark a median baseline, Ridge regression and Random Forest using a final-season holdout rather than a random split.
- Run an interactive starting-grid and pit-stop sensitivity scenario for a selected race and driver.
- View final **2025 driver and constructor championship standings** separately from the Grand Prix-only points analysis.

## Dashboard modules

| Module | What it provides |
| --- | --- |
| Overview | Season KPIs, driver race-result points, constructor distribution and final 2025 standings snapshot. |
| Season Explorer | Cumulative driver form, grid-versus-finish scatter plot and position-gain table. |
| Pit Stops | Driver-level median, fastest and average pit-stop duration, plus an inspectable summary table. |
| ML & Strategy | Model benchmark, held-out prediction diagnostics and a strategy sensitivity scenario. |
| Methodology | Data coverage, assumptions and modelling limitations. |

## Machine learning approach

The performance model estimates a driver's **final classified position** (lower is better). It uses only pre-race information available in the local dataset:

- qualifying / grid position and calendar round;
- circuit and constructor identity;
- lagged five-race driver points and classified-finish form;
- lagged five-race constructor points.

For the current dataset, the models train on seasons through 2024 and hold out **2025** for evaluation. This avoids future-race leakage that a random train/test split would introduce.

| Model | 2025 holdout MAE | RMSE | R² |
| --- | ---: | ---: | ---: |
| Random Forest | 3.28 positions | 4.28 | 0.45 |
| Ridge Regression | 3.41 positions | 4.32 | 0.44 |
| Median baseline | 4.99 positions | 5.78 | -0.01 |

Metrics are reproducible from the bundled data and fixed model seed. They describe this snapshot only; they are not a live race forecast.

## Strategy sensitivity

The strategy page lets you alter a driver's planned grid position and pit-stop execution relative to plan. The main finishing-position estimate comes from the validated pre-race model. Pit-stop sensitivity is based on an **observational** historical association, not a causal effect: safety cars, tyre state, traffic, failures and race strategy are confounders. The app surfaces this caveat deliberately.

## Data coverage and provenance

The app ships with local CSV snapshots, so it runs without live API access.

- Grand Prix races: 2012–2025
- 2025 race data: 24 races, 479 classifications and 821 pit-stop records
- Final 2025 championship snapshots: 21 drivers and 10 constructors
- 2024–2025 updates: [Jolpica F1 API](https://api.jolpi.ca/docs/), the maintained Ergast-compatible data service
- Earlier historical files: included with the source project

Grand Prix race-result points are labelled separately from final championship standings because the historical race-results table does not include sprint-session points.

## Run locally

```bash
git clone https://github.com/piyush23-eng/PaddockML.git
cd PaddockML

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
streamlit run app.py
```

## Refreshing data

To reproduce the 2024–2025 data refresh:

```bash
python scripts/update_season_data.py --seasons 2024 2025
```

The script appends missing race rounds and refreshes final standings from Jolpica. It requests results and pit stops per round to avoid incomplete pagination when a result page splits a race classification.

## Project structure

```text
app.py                         # Streamlit user interface and dashboard views
modeling.py                    # Feature engineering, model evaluation and pit sensitivity
scripts/update_season_data.py  # Reproducible 2024–2025 local data refresh
*_1.csv                        # Local race, result, driver, constructor and pit-stop snapshots
CurrentStanding*.csv           # Final 2025 championship snapshots
```

## Technology

Python · Streamlit · Pandas · Plotly · scikit-learn · NumPy

## Limitations and next steps

- The model does not yet use weather, tyre compounds, stint data, safety-car periods, sector timing or race-control events.
- A production model should use rolling-origin cross-validation, prediction intervals and race-specific feature monitoring.
- Strategy sensitivity is a research aid and should be validated against race-engineering context before being used for decisions.

---

Built as a data-science portfolio project focused on transparent Formula 1 performance analysis.
