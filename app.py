"""Formula 1 Analytics Lab — an interactive Streamlit portfolio project."""

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from modeling import pit_stop_association, train_and_evaluate


BASE_DIR = Path(__file__).resolve().parent
F1_RED = "#e10600"
F1_DARK = "#0e1117"
PLOTLY_LAYOUT = {
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(0,0,0,0)",
    "font": {"color": "#e8eaed", "family": "Inter, Arial, sans-serif"},
    "margin": {"l": 10, "r": 10, "t": 45, "b": 10},
}


@st.cache_data(show_spinner="Loading race data…")
def load_data():
    """Load once per session and keep file paths independent of the launch directory."""
    races = pd.read_csv(BASE_DIR / "races_1.csv")
    results = pd.read_csv(BASE_DIR / "results_1.csv")
    drivers = pd.read_csv(BASE_DIR / "drivers_1.csv")
    constructors = pd.read_csv(BASE_DIR / "constructors_1.csv")
    pits = pd.read_csv(BASE_DIR / "pit_stops_1.csv")
    predictions = pd.read_csv(BASE_DIR / "DriverPrediction.csv")
    current_drivers = pd.read_csv(BASE_DIR / "CurrentStanding.csv")
    current_constructors = pd.read_csv(BASE_DIR / "CurrentStanding2.csv")

    races["date"] = pd.to_datetime(races["date"], dayfirst=True, errors="coerce")
    drivers["driver_name"] = drivers["forename"] + " " + drivers["surname"]
    results["points"] = pd.to_numeric(results["points"], errors="coerce").fillna(0)
    results["positionOrder"] = pd.to_numeric(results["positionOrder"], errors="coerce")
    results["grid"] = pd.to_numeric(results["grid"], errors="coerce")
    pits["seconds"] = pd.to_numeric(pits["milliseconds"], errors="coerce") / 1000

    enriched = (
        results.merge(races[["raceId", "year", "round", "circuitId", "name", "date"]], on="raceId", how="left")
        .merge(drivers[["driverId", "driver_name", "driverRef", "nationality"]], on="driverId", how="left")
        .merge(constructors[["constructorId", "name"]].rename(columns={"name": "constructor"}), on="constructorId", how="left")
    )
    return races, enriched, drivers, constructors, pits, predictions, current_drivers, current_constructors


def inject_styles():
    st.markdown(
        """
        <style>
        .stApp { background: radial-gradient(circle at 80% -20%, #401517 0, #111217 34%, #0a0c10 72%); }
        [data-testid="stSidebar"] { background: #101217; border-right: 1px solid #30343b; }
        .hero { padding: 1.8rem 0 1rem; }
        .eyebrow { color: #ff554d; font-weight: 700; letter-spacing: .12em; font-size: .78rem; }
        .hero h1 { font-size: clamp(2.1rem, 4vw, 3.9rem); margin: .1rem 0; letter-spacing: -.045em; }
        .hero p { color: #b7bdc8; max-width: 720px; font-size: 1.08rem; }
        .card { background: linear-gradient(145deg, #1a1d25, #111319); border: 1px solid #323640; border-radius: 14px; padding: 1.1rem 1.25rem; min-height: 90px; }
        .card .label { color: #aab1bd; font-size: .85rem; text-transform: uppercase; letter-spacing: .07em; }
        .card .value { font-size: clamp(1.35rem, 2.1vw, 2rem); font-weight: 750; margin-top: .18rem; overflow-wrap: anywhere; }
        .card .detail { color: #7fd7a4; font-size: .82rem; }
        [data-testid="stMetric"] { background: #171a21; border: 1px solid #30343b; border-radius: 12px; padding: 12px; }
        h2, h3 { letter-spacing: -.025em; }
        div[data-testid="stDataFrame"] { border: 1px solid #30343b; border-radius: 12px; overflow: hidden; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def chart(fig, height=380):
    fig.update_layout(**PLOTLY_LAYOUT, height=height)
    fig.update_xaxes(gridcolor="#292d36", zerolinecolor="#292d36")
    fig.update_yaxes(gridcolor="#292d36", zerolinecolor="#292d36")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def season_data(results, year):
    return results.loc[results["year"] == year].copy()


def driver_table(season):
    table = (
        season.groupby("driver_name", as_index=False)
        .agg(
            Points=("points", "sum"),
            Wins=("positionOrder", lambda s: (s == 1).sum()),
            Podiums=("positionOrder", lambda s: (s <= 3).sum()),
            Starts=("raceId", "nunique"),
            Avg_finish=("positionOrder", "mean"),
            Avg_grid=("grid", "mean"),
        )
    )
    table["Position gain"] = table["Avg_grid"] - table["Avg_finish"]
    return table.sort_values(["Points", "Wins"], ascending=False).reset_index(drop=True)


def constructor_table(season):
    return (
        season.groupby("constructor", as_index=False)
        .agg(Points=("points", "sum"), Wins=("positionOrder", lambda s: (s == 1).sum()), Starts=("raceId", "nunique"))
        .sort_values(["Points", "Wins"], ascending=False)
        .reset_index(drop=True)
    )


def render_hero(selected_year):
    st.markdown(
        f"""<div class="hero"><div class="eyebrow">DATA-DRIVEN MOTORSPORT</div>
        <h1>Formula 1 Analytics Lab</h1>
        <p>Explore {selected_year} season performance, race-by-race form, pit-stop execution and model-based championship scenarios from the included historical dataset.</p></div>""",
        unsafe_allow_html=True,
    )


def overview(results, current_drivers, current_constructors, year):
    season = season_data(results, year)
    standings = driver_table(season)
    teams = constructor_table(season)
    races = season["raceId"].nunique()
    leader = standings.iloc[0]
    best_team = teams.iloc[0]

    cols = st.columns(4)
    metrics = [
        ("Races analysed", str(races), f"{year} season"),
        ("Race-points leader", leader["driver_name"].split()[-1], f"{leader['driver_name']} · {leader['Points']:.0f} points"),
        ("Most wins", str(int(leader["Wins"])), leader["driver_name"]),
        ("Top constructor", best_team["constructor"], f"{best_team['Points']:.0f} race points"),
    ]
    for col, (label, value, detail) in zip(cols, metrics):
        col.markdown(f'<div class="card"><div class="label">{label}</div><div class="value">{value}</div><div class="detail">{detail}</div></div>', unsafe_allow_html=True)

    left, right = st.columns((1.25, 1))
    with left:
        st.subheader("Driver race-result points")
        fig = px.bar(standings.head(12).sort_values("Points"), x="Points", y="driver_name", orientation="h", color="Points", color_continuous_scale=["#4f0b10", F1_RED], text="Points")
        fig.update_coloraxes(showscale=False)
        fig.update_traces(texttemplate="%{text:.0f}", textposition="outside", cliponaxis=False)
        chart(fig, 430)
    with right:
        st.subheader("Constructor race-result points")
        fig = px.pie(teams, names="constructor", values="Points", hole=.58, color_discrete_sequence=px.colors.sequential.Reds_r)
        fig.update_traces(textinfo="percent", hovertemplate="%{label}<br>%{value:.0f} points<extra></extra>")
        chart(fig, 430)

    st.caption("Race-result points exclude sprint-session scoring. The final 2025 championship snapshot, including sprint points, is available below.")
    with st.expander("Final 2025 championship standings snapshot", expanded=False):
        a, b = st.columns(2)
        a.dataframe(current_drivers, use_container_width=True, hide_index=True)
        b.dataframe(current_constructors, use_container_width=True, hide_index=True)


def season_explorer(results, year):
    season = season_data(results, year)
    standings = driver_table(season)
    st.subheader("Race-by-race form")
    candidates = standings.head(10)["driver_name"].tolist()
    selected = st.multiselect("Compare drivers", candidates, default=candidates[:4], max_selections=8)
    if selected:
        form = season[season["driver_name"].isin(selected)].sort_values(["driver_name", "round"])
        form["Cumulative points"] = form.groupby("driver_name")["points"].cumsum()
        fig = px.line(form, x="round", y="Cumulative points", color="driver_name", markers=True, color_discrete_sequence=px.colors.qualitative.Bold)
        fig.update_layout(legend_title_text="Driver", xaxis_title="Round")
        chart(fig)
    else:
        st.info("Choose at least one driver to compare race-by-race form.")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Starting position vs finish")
        tooltip = {"driver_name": True, "constructor": True, "grid": ":.0f", "positionOrder": ":.0f", "points": ":.0f"}
        fig = px.scatter(season, x="grid", y="positionOrder", color="constructor", hover_name="driver_name", hover_data=tooltip, opacity=.72)
        fig.update_yaxes(autorange="reversed", title="Finish position (lower is better)")
        fig.update_xaxes(title="Grid position")
        chart(fig, 390)
    with c2:
        st.subheader("Overperformers")
        display = standings[["driver_name", "Points", "Wins", "Podiums", "Avg_grid", "Avg_finish", "Position gain"]].copy()
        display = display.sort_values("Position gain", ascending=False).head(12)
        display[["Points", "Avg_grid", "Avg_finish", "Position gain"]] = display[["Points", "Avg_grid", "Avg_finish", "Position gain"]].round(1)
        st.dataframe(display.rename(columns={"driver_name": "Driver"}), use_container_width=True, hide_index=True)


def pit_stop_intelligence(results, pits, drivers, year):
    st.subheader("Pit-stop intelligence")
    st.caption("Pit-stop data is filtered by season and aggregated by driver. Values include total pit-lane duration from the source dataset.")
    races = results.loc[results["year"] == year, ["raceId"]].drop_duplicates()
    pit_data = pits.merge(races, on="raceId", how="inner").merge(drivers[["driverId", "driver_name"]], on="driverId", how="left")
    pit_data = pit_data[pit_data["seconds"].between(10, 90)].copy()
    if pit_data.empty:
        st.warning(f"No usable pit-stop records are available for {year}.")
        return
    summary = pit_data.groupby("driver_name", as_index=False).agg(
        Stops=("stop", "count"), Median_seconds=("seconds", "median"), Fastest_seconds=("seconds", "min"), Average_seconds=("seconds", "mean")
    )
    summary = summary[summary["Stops"] >= 3].sort_values("Median_seconds")
    fastest = pit_data.loc[pit_data["seconds"].idxmin()]
    a, b, c = st.columns(3)
    a.metric("Recorded pit stops", f"{len(pit_data):,}")
    b.metric("Fastest stop", f"{fastest['seconds']:.2f}s")
    c.metric("Fastest driver", fastest["driver_name"])
    fig = px.bar(summary.head(15).sort_values("Median_seconds"), x="Median_seconds", y="driver_name", orientation="h", color="Median_seconds", color_continuous_scale=["#4f0b10", F1_RED], hover_data=["Stops", "Fastest_seconds", "Average_seconds"])
    fig.update_coloraxes(showscale=False)
    fig.update_xaxes(title="Median pit-stop duration (seconds)")
    fig.update_yaxes(title="")
    chart(fig, 460)
    with st.expander("Inspect pit-stop summary"):
        st.dataframe(summary.round(2), use_container_width=True, hide_index=True)


@st.cache_resource(show_spinner="Training and evaluating performance models…")
def get_model_artifacts(results):
    return train_and_evaluate(results)


def ml_strategy_lab(results, pits):
    st.subheader("ML performance model & strategy sensitivity")
    artifacts = get_model_artifacts(results)
    holdout_year = artifacts["holdout_year"]
    st.caption(f"The model predicts final classified position (lower is better), using only information available before a race: qualifying grid, round, driver form, constructor form and circuit. It is validated on the held-out {holdout_year} season.")
    metrics = artifacts["metrics"].copy()
    for column in ["MAE (positions)", "RMSE", "R²"]:
        metrics[column] = metrics[column].round(2)
    st.markdown("#### Time-aware model benchmark")
    st.dataframe(metrics, use_container_width=True, hide_index=True)
    best = metrics.iloc[0]
    a, b, c = st.columns(3)
    a.metric("Selected model", str(best["Model"]))
    b.metric("Holdout MAE", f"{best['MAE (positions)']:.2f} positions")
    c.metric("Holdout R²", f"{best['R²']:.2f}")
    st.caption(f"This {holdout_year} final-season holdout is stronger than a random split because it prevents future races leaking into model training. A negative R² is still valuable evidence: it means the model has not yet beaten a simple holdout baseline reliably.")

    tabs = st.tabs(["Prediction diagnostics", "Strategy what-if", "Research notes"])
    with tabs[0]:
        check = artifacts["test_predictions"].copy()
        check["Predicted finish"] = check["Predicted finish"].clip(1, 25).round(1)
        check["Actual finish"] = check["positionOrder"].round(0)
        selected_drivers = st.multiselect("Drivers to inspect", sorted(check["driver_name"].unique()), default=sorted(check["driver_name"].unique())[:5], key="ml_driver_filter")
        plot_data = check[check["driver_name"].isin(selected_drivers)]
        if not plot_data.empty:
            fig = px.scatter(plot_data, x="Actual finish", y="Predicted finish", color="driver_name", hover_data=["name", "constructor", "grid"], opacity=.75)
            fig.add_shape(type="line", x0=1, y0=1, x1=25, y1=25, line={"color": "#aab1bd", "dash": "dash"})
            fig.update_xaxes(range=[0, 25], dtick=5)
            fig.update_yaxes(range=[25, 0], dtick=5, title="Predicted classified position")
            chart(fig, 450)
        st.caption("Points on the diagonal are exact predictions. Results are exploratory estimates, not a replacement for race engineering judgement.")

    with tabs[1]:
        test = artifacts["test_predictions"].copy()
        race_names = test.sort_values("round")["name"].drop_duplicates().tolist()
        selected_race = st.selectbox("Race scenario", race_names)
        candidates = test.loc[test["name"] == selected_race].sort_values("grid")
        selected_driver = st.selectbox("Driver", candidates["driver_name"].tolist())
        record = candidates.loc[candidates["driver_name"] == selected_driver].iloc[[0]].copy()
        grid_value = st.slider("Planned qualifying / starting position", min_value=1, max_value=20, value=int(record["grid"].iloc[0] if pd.notna(record["grid"].iloc[0]) else 10))
        pit_delta = st.slider("Pit-stop execution versus plan (seconds)", min_value=-3.0, max_value=3.0, value=0.0, step=0.1, help="Negative is a faster stop; positive is a slower stop.")
        record["grid"] = grid_value
        model_finish = float(artifacts["best_model"].predict(record[["grid", "round", "driver_recent_points", "driver_recent_finish", "team_recent_points", "constructor", "circuitId"]])[0])
        pit_effect = pit_stop_association(results, pits)
        adjusted_finish = model_finish + pit_delta * pit_effect["slope"]
        x, y, z = st.columns(3)
        x.metric("Model expected finish", f"P{max(1, min(25, model_finish)):.1f}")
        y.metric("Pit-time sensitivity", f"P{max(1, min(25, adjusted_finish)):.1f}", delta=f"{adjusted_finish - model_finish:+.2f} positions")
        z.metric("Historical pit sample", f"{len(pit_effect['sample']):,} driver-races")
        st.markdown(f"**Decision interpretation:** For this scenario, moving to grid P{grid_value} yields an expected classified position around P{model_finish:.1f}. The pit-time sensitivity applies an observational historical association of **{pit_effect['slope']:.3f} finishing positions per additional pit-stop second**.")
        st.warning("This is a sensitivity analysis, not a causal claim: pit timing, safety cars, tyre choices and traffic are confounded in the historic data. Use it to frame questions, then validate against race-engineering context.")

    with tabs[2]:
        st.markdown("""
        **Why this is credible portfolio work**

        - The target is held out by time: all prior seasons train the models and the latest complete season is the test set.
        - A median baseline and a regularised linear model are compared against a non-linear random forest.
        - Features are lagged at driver and constructor level, so current-race results are not used to predict themselves.
        - Strategy output distinguishes a predictive model from an observational association and surfaces its limitation.

        **Next iteration:** ingest weather, tyre stint, safety-car and sector-time data; use rolling-origin cross-validation; add prediction intervals and calibrate simulations with race-specific tyre-degradation models.
        """)


def methodology(results, pits):
    st.subheader("Project notes")
    st.markdown("""
    This upgraded portfolio version is designed to make the analysis easier to inspect in an interview:

    - All local files are resolved from the app directory, so `streamlit run app.py` works from any folder.
    - Data loading is cached, and the selected season drives every dashboard view.
    - Race position, points, pit-stop timing and projection outputs are kept visibly separate to avoid overstating what the data can support.
    """)
    st.subheader("Data coverage")
    years = results["year"].dropna().astype(int)
    a, b, c = st.columns(3)
    a.metric("Historical seasons", f"{years.min()}–{years.max()}")
    b.metric("Race result records", f"{len(results):,}")
    c.metric("Pit-stop records", f"{len(pits):,}")
    st.caption("Source data was supplied with the original repository. The app does not make external calls at runtime.")


def main():
    st.set_page_config(page_title="F1 Analytics Lab", page_icon="🏎️", layout="wide", initial_sidebar_state="expanded")
    inject_styles()
    races, results, drivers, _, pits, predictions, current_drivers, current_constructors = load_data()
    years = sorted(races["year"].dropna().astype(int).unique(), reverse=True)
    with st.sidebar:
        st.image(str(BASE_DIR / "formula1projectlogo.jfif"), width=100)
        st.title("Analytics Lab")
        page = st.radio("Workspace", ["Overview", "Season Explorer", "Pit Stops", "ML & Strategy", "Methodology"])
        year = st.selectbox("Season", years, index=0)
        st.divider()
        st.caption(f"Built with Streamlit, Pandas and Plotly. Historical dataset through {max(years)}.")
    render_hero(year)
    if page == "Overview":
        overview(results, current_drivers, current_constructors, year)
    elif page == "Season Explorer":
        season_explorer(results, year)
    elif page == "Pit Stops":
        pit_stop_intelligence(results, pits, drivers, year)
    elif page == "ML & Strategy":
        ml_strategy_lab(results, pits)
    else:
        methodology(results, pits)


if __name__ == "__main__":
    main()
