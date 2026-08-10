#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import ROOT, ensure_dirs, parse_lap_time


def load_races(endpoint: str) -> list[dict]:
    races = []
    for path in sorted((ROOT / "data" / "raw" / "jolpica").glob(f"{endpoint}_*_offset_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        races.extend(payload["MRData"]["RaceTable"].get("Races", []))
    return races


def flatten_qualifying() -> pd.DataFrame:
    rows = []
    for race in load_races("qualifying"):
        for result in race.get("QualifyingResults", []):
            driver = result.get("Driver", {})
            constructor = result.get("Constructor", {})
            rows.append({
                "season": int(race["season"]),
                "round": int(race["round"]),
                "race_name": race.get("raceName"),
                "race_date": race.get("date"),
                "circuit_id": race.get("Circuit", {}).get("circuitId"),
                "driver_id": driver.get("driverId"),
                "driver_code": driver.get("code"),
                "driver_given_name": driver.get("givenName"),
                "driver_family_name": driver.get("familyName"),
                "constructor_id": constructor.get("constructorId"),
                "qualifying_position": pd.to_numeric(result.get("position"), errors="coerce"),
                "q1_raw": result.get("Q1"),
                "q2_raw": result.get("Q2"),
                "q3_raw": result.get("Q3"),
                "q1_sec": parse_lap_time(result.get("Q1")),
                "q2_sec": parse_lap_time(result.get("Q2")),
                "q3_sec": parse_lap_time(result.get("Q3")),
            })
    frame = pd.DataFrame(rows).drop_duplicates(["season", "round", "driver_id"], keep="last")
    return frame.sort_values(["season", "round", "qualifying_position", "driver_id"])


def flatten_results() -> pd.DataFrame:
    rows = []
    for race in load_races("results"):
        for result in race.get("Results", []):
            driver = result.get("Driver", {})
            constructor = result.get("Constructor", {})
            rows.append({
                "season": int(race["season"]),
                "round": int(race["round"]),
                "race_name": race.get("raceName"),
                "race_date": race.get("date"),
                "driver_id": driver.get("driverId"),
                "driver_code": driver.get("code"),
                "constructor_id": constructor.get("constructorId"),
                "finish_position": pd.to_numeric(result.get("position"), errors="coerce"),
                "position_text": result.get("positionText"),
                "race_points": pd.to_numeric(result.get("points"), errors="coerce"),
                "grid_position": pd.to_numeric(result.get("grid"), errors="coerce"),
                "laps": pd.to_numeric(result.get("laps"), errors="coerce"),
                "status": result.get("status"),
            })
    frame = pd.DataFrame(rows).drop_duplicates(["season", "round", "driver_id"], keep="last")
    frame["top_ten"] = (frame["finish_position"] <= 10).astype(float)
    non_starts = {"Did not start", "Did not qualify", "Did not prequalify", "Withdrawn"}
    frame["counted_start"] = (~frame["status"].isin(non_starts)).astype(int)
    return frame.sort_values(["season", "round", "finish_position", "driver_id"])


def main() -> None:
    ensure_dirs()
    qualifying = flatten_qualifying()
    results = flatten_results()
    qualifying.to_csv(ROOT / "data" / "interim" / "qualifying.csv", index=False)
    results.to_csv(ROOT / "data" / "interim" / "results.csv", index=False)

    prior = (
        results.groupby(["season", "driver_id"], as_index=False)
        .agg(
            season_points=("race_points", "sum"),
            season_result_records=("round", "nunique"),
            season_starts=("counted_start", "sum"),
        )
    )
    prior["season_points_per_start"] = np.where(
        prior["season_starts"] > 0,
        prior["season_points"] / prior["season_starts"],
        np.nan,
    )
    prior.to_csv(ROOT / "data" / "interim" / "driver_season_history.csv", index=False)
    log = {
        "qualifying_rows": int(len(qualifying)),
        "qualifying_races_2010_2019": int(
            qualifying.loc[qualifying["season"].between(2010, 2019), ["season", "round"]].drop_duplicates().shape[0]
        ),
        "result_rows": int(len(results)),
        "duplicate_qualifying_keys": int(qualifying.duplicated(["season", "round", "driver_id"]).sum()),
        "duplicate_result_keys": int(results.duplicated(["season", "round", "driver_id"]).sum()),
    }
    (ROOT / "output" / "logs" / "01_clean.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(json.dumps(log, indent=2))


if __name__ == "__main__":
    main()
