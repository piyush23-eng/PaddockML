"""Reproducible, time-aware modelling helpers for Formula 1 performance analysis.

The features below are available before a race starts. They intentionally exclude
post-race values such as finishing position, race points and observed pit duration.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NUMERIC_FEATURES = ["grid", "round", "driver_recent_points", "driver_recent_finish", "team_recent_points"]
CATEGORICAL_FEATURES = ["constructor", "circuitId"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def _rolling_history(frame: pd.DataFrame, group: str, value: str, window: int = 5) -> pd.Series:
    return frame.groupby(group, group_keys=False)[value].apply(
        lambda values: values.shift().rolling(window=window, min_periods=1).mean()
    )


def build_feature_frame(results: pd.DataFrame) -> pd.DataFrame:
    """Create leakage-safe pre-race features ordered chronologically."""
    columns = [
        "raceId", "year", "round", "date", "name", "circuitId", "driverId", "driver_name",
        "constructorId", "constructor", "grid", "points", "positionOrder",
    ]
    frame = results[columns].dropna(subset=["year", "round", "driverId", "constructorId"]).copy()
    frame = frame.sort_values(["date", "raceId", "driverId"]).reset_index(drop=True)
    frame["grid"] = frame["grid"].replace(0, np.nan)
    frame["driver_recent_points"] = _rolling_history(frame, "driverId", "points")
    frame["driver_recent_finish"] = _rolling_history(frame, "driverId", "positionOrder")

    # A constructor's strength is the mean points scored in its prior five races.
    team_race = (
        frame.groupby(["raceId", "constructorId"], as_index=False)
        .agg(team_points=("points", "sum"), date=("date", "first"))
        .sort_values(["date", "raceId"])
    )
    team_race["team_recent_points"] = _rolling_history(team_race, "constructorId", "team_points")
    frame = frame.merge(team_race[["raceId", "constructorId", "team_recent_points"]], on=["raceId", "constructorId"], how="left")
    return frame


def _preprocessor(scale_numeric: bool) -> ColumnTransformer:
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scale", StandardScaler()))
    return ColumnTransformer(
        [
            ("numeric", Pipeline(numeric_steps), NUMERIC_FEATURES),
            ("categorical", Pipeline([( "imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), CATEGORICAL_FEATURES),
        ]
    )


def train_and_evaluate(results: pd.DataFrame, holdout_year: int | None = None) -> dict:
    """Benchmark models using a final-season holdout, never a random train/test split."""
    frame = build_feature_frame(results).dropna(subset=["positionOrder"]).copy()
    if holdout_year is None:
        holdout_year = int(frame["year"].max())
    train = frame.loc[frame["year"] < holdout_year].copy()
    test = frame.loc[frame["year"] == holdout_year].copy()
    if train.empty or test.empty:
        raise ValueError("The supplied data does not contain both training seasons and the requested holdout season.")

    x_train, y_train = train[FEATURES], train["positionOrder"]
    x_test, y_test = test[FEATURES], test["positionOrder"]
    models = {
        "Median baseline": Pipeline([( "prep", _preprocessor(False)), ("model", DummyRegressor(strategy="median"))]),
        "Ridge regression": Pipeline([( "prep", _preprocessor(True)), ("model", Ridge(alpha=6.0))]),
        "Random forest": Pipeline([( "prep", _preprocessor(False)), ("model", RandomForestRegressor(n_estimators=180, min_samples_leaf=2, max_features=0.75, random_state=42, n_jobs=-1))]),
    }
    rows, fitted = [], {}
    for name, model in models.items():
        model.fit(x_train, y_train)
        predicted = model.predict(x_test)
        rows.append(
            {
                "Model": name,
                "MAE (positions)": mean_absolute_error(y_test, predicted),
                "RMSE": mean_squared_error(y_test, predicted) ** 0.5,
                "R²": r2_score(y_test, predicted),
            }
        )
        fitted[name] = model
    metrics = pd.DataFrame(rows).sort_values("MAE (positions)").reset_index(drop=True)
    best_name = metrics.iloc[0]["Model"]
    display_columns = ["raceId", "year", "round", "name", "driver_name", "constructor", "grid", "positionOrder"]
    test = test[list(dict.fromkeys(display_columns + FEATURES))].copy()
    test["Predicted finish"] = fitted[best_name].predict(test[FEATURES])
    test["Error"] = test["Predicted finish"] - test["positionOrder"]
    return {"metrics": metrics, "test_predictions": test, "best_model": fitted[best_name], "best_name": best_name, "features": frame, "holdout_year": holdout_year}


def pit_stop_association(results: pd.DataFrame, pits: pd.DataFrame) -> dict:
    """Estimate an observational pit-time / final-position relationship for sensitivity analysis."""
    summary = pits.groupby(["raceId", "driverId"], as_index=False).agg(median_pit_seconds=("seconds", "median"), stops=("stop", "count"))
    outcome = results[["raceId", "driverId", "grid", "positionOrder"]].drop_duplicates(["raceId", "driverId"])
    sample = summary.merge(outcome, on=["raceId", "driverId"], how="inner")
    sample = sample[sample["median_pit_seconds"].between(12, 60) & sample["positionOrder"].between(1, 25)].dropna()
    if len(sample) < 20:
        return {"sample": sample, "slope": 0.0, "correlation": np.nan}
    slope = float(np.polyfit(sample["median_pit_seconds"], sample["positionOrder"], 1)[0])
    return {"sample": sample, "slope": slope, "correlation": float(sample["median_pit_seconds"].corr(sample["positionOrder"], method="spearman"))}
