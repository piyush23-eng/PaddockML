"""Refresh local F1 results from the Jolpica (Ergast-compatible) API.

Usage: python scripts/update_season_data.py --seasons 2024 2025

The script appends only missing race rounds and is safe to rerun. It deliberately
keeps the project self-contained: Streamlit reads the CSV snapshots locally and
does not call an API when a recruiter opens the app.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
API_ROOT = "https://api.jolpi.ca/ergast/f1"
HEADERS = {"User-Agent": "F1AnalyticsLab/1.0 (portfolio data refresh)"}


def fetch(path: str, **params) -> dict:
    query = urlencode({key: value for key, value in params.items() if value is not None})
    url = f"{API_ROOT}/{path}.json" + (f"?{query}" if query else "")
    request = Request(url, headers=HEADERS)
    with urlopen(request, timeout=30) as response:  # noqa: S310 - fixed HTTPS endpoint
        payload = json.load(response)
    time.sleep(0.15)  # Respect the public service while keeping a refresh quick.
    return payload["MRData"]


def round_results(season: int, round_number: int) -> list[dict]:
    """Fetch per-round results to avoid a page splitting a race's 20 entries."""
    return fetch(f"{season}/{round_number}/results", limit=100)["RaceTable"]["Races"]


def final_standings(season: int, type_name: str) -> list[dict]:
    data = fetch(f"{season}/{type_name}", limit=100)
    return data["StandingsTable"]["StandingsLists"][0]["{0}".format("DriverStandings" if type_name == "driverstandings" else "ConstructorStandings")]


def as_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def as_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def update(seasons: list[int]):
    races = pd.read_csv(ROOT / "races_1.csv")
    results = pd.read_csv(ROOT / "results_1.csv")
    drivers = pd.read_csv(ROOT / "drivers_1.csv")
    constructors = pd.read_csv(ROOT / "constructors_1.csv")
    pits = pd.read_csv(ROOT / "pit_stops_1.csv")

    race_lookup = {(int(row.year), int(row["round"])): int(row.raceId) for _, row in races.iterrows()}
    next_race_id = int(races["raceId"].max()) + 1
    next_result_id = int(results["resultId"].max()) + 1
    next_driver_id = int(drivers["driverId"].max()) + 1
    next_constructor_id = int(constructors["constructorId"].max()) + 1
    driver_ids = dict(zip(drivers["driverRef"], drivers["driverId"]))
    constructor_ids = dict(zip(constructors["constructorRef"], constructors["constructorId"]))
    result_keys = set(zip(results["raceId"], results["driverId"]))
    pit_keys = set(zip(pits["raceId"], pits["driverId"], pits["stop"]))

    new_races, new_results, new_drivers, new_constructors, new_pits = [], [], [], [], []
    latest_driver_standings = latest_constructor_standings = None
    for season in sorted(seasons):
        rounds_with_results = []
        # Both target seasons have 24 scheduled rounds. Per-round fetching keeps
        # every result intact even when an API page would otherwise split a race.
        for round_number in range(1, 25):
            result_races = round_results(season, round_number)
            if not result_races:
                continue
            race = result_races[0]
            rounds_with_results.append(round_number)
            key = (season, round_number)
            race_id = race_lookup.get(key)
            if race_id is None:
                race_id = next_race_id
                next_race_id += 1
                race_lookup[key] = race_id
                new_races.append({
                    "raceId": race_id, "year": season, "round": round_number,
                    "circuitId": race["Circuit"]["circuitId"], "name": race["raceName"], "date": race["date"],
                })
            for item in race["Results"]:
                driver = item["Driver"]
                constructor = item["Constructor"]
                driver_ref = driver["driverId"]
                constructor_ref = constructor["constructorId"]
                if driver_ref not in driver_ids:
                    driver_ids[driver_ref] = next_driver_id
                    new_drivers.append({
                        "driverId": next_driver_id, "driverRef": driver_ref, "forename": driver["givenName"],
                        "surname": driver["familyName"], "dob": driver.get("dateOfBirth", ""), "nationality": driver.get("nationality", ""),
                    })
                    next_driver_id += 1
                if constructor_ref not in constructor_ids:
                    constructor_ids[constructor_ref] = next_constructor_id
                    new_constructors.append({
                        "constructorId": next_constructor_id, "constructorRef": constructor_ref,
                        "name": constructor["name"], "nationality": constructor.get("nationality", ""),
                    })
                    next_constructor_id += 1
                result_key = (race_id, driver_ids[driver_ref])
                if result_key in result_keys:
                    continue
                fastest = item.get("FastestLap", {})
                new_results.append({
                    "resultId": next_result_id, "raceId": race_id, "driverId": driver_ids[driver_ref],
                    "constructorId": constructor_ids[constructor_ref], "number": as_int(item.get("number")),
                    "grid": as_int(item.get("grid")), "position": item.get("position", "\\N"),
                    "positionText": item.get("positionText", "\\N"), "positionOrder": as_int(item.get("position"), 25),
                    "points": as_float(item.get("points")), "laps": as_int(item.get("laps")),
                    "time": item.get("Time", {}).get("time", item.get("status", "")),
                    "milliseconds": as_int(item.get("Time", {}).get("millis")), "fastestLap": as_int(fastest.get("lap")),
                    "rank": as_int(fastest.get("AverageSpeed", {}).get("rank")),
                    "fastestLapTime": fastest.get("Time", {}).get("time", ""),
                    "fastestLapSpeed": as_float(fastest.get("AverageSpeed", {}).get("speed")),
                    "statusId": 1 if item.get("position") else 0,
                })
                next_result_id += 1
                result_keys.add(result_key)

        # Pit-stop endpoint is only available per round. These records drive the
        # pit-stop dashboard and strategy sensitivity, so refresh them too.
        for round_number in rounds_with_results:
            race_id = race_lookup.get((season, round_number))
            if race_id is None:
                continue
            pit_races = fetch(f"{season}/{round_number}/pitstops", limit=100)["RaceTable"]["Races"]
            for pit_race in pit_races:
                for stop in pit_race.get("PitStops", []):
                    driver_id = driver_ids.get(stop["driverId"])
                    key = (race_id, driver_id, as_int(stop.get("stop")))
                    if driver_id is None or key in pit_keys:
                        continue
                    new_pits.append({
                        "raceId": race_id, "driverId": driver_id, "stop": as_int(stop.get("stop")),
                        "lap": as_int(stop.get("lap")), "milliseconds": int(as_float(stop.get("duration")) * 1000),
                    })
                    pit_keys.add(key)
        latest_driver_standings = final_standings(season, "driverstandings")
        latest_constructor_standings = final_standings(season, "constructorstandings")

    for current, additions, filename in [
        (races, new_races, "races_1.csv"), (results, new_results, "results_1.csv"),
        (drivers, new_drivers, "drivers_1.csv"), (constructors, new_constructors, "constructors_1.csv"),
        (pits, new_pits, "pit_stops_1.csv"),
    ]:
        if additions:
            pd.concat([current, pd.DataFrame(additions)], ignore_index=True).to_csv(ROOT / filename, index=False)

    if latest_driver_standings:
        snapshot = pd.DataFrame([
            {"Position": as_int(row["position"]), "Driver": f"{row['Driver']['givenName']} {row['Driver']['familyName']}", "Points": as_float(row["points"])}
            for row in latest_driver_standings
        ])
        snapshot.to_csv(ROOT / "CurrentStanding.csv", index=False)
    if latest_constructor_standings:
        snapshot = pd.DataFrame([
            {"Position": as_int(row["position"]), "Constructor": row["Constructor"]["name"], "Points": as_float(row["points"])}
            for row in latest_constructor_standings
        ])
        snapshot.to_csv(ROOT / "CurrentStanding2.csv", index=False)

    print(f"Added {len(new_races)} races, {len(new_results)} results, {len(new_pits)} pit stops, {len(new_drivers)} drivers and {len(new_constructors)} constructors.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seasons", nargs="+", type=int, default=[2024, 2025])
    update(parser.parse_args().seasons)
