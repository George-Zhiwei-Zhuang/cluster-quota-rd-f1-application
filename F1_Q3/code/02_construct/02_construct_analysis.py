#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import ROOT, ensure_dirs, load_config


def rank_q2(group: pd.DataFrame) -> pd.DataFrame:
    group = group.copy()
    eligible = group["q2_sec"].notna()
    ordered = group.loc[eligible].sort_values(["q2_sec", "qualifying_position", "driver_id"])
    group["q2_rank"] = np.nan
    group.loc[ordered.index, "q2_rank"] = np.arange(1, len(ordered) + 1)
    return group


def audit_reason(row: pd.Series) -> str:
    reasons = []
    if not (12 <= row["q2_count"] <= 17):
        reasons.append(f"q2_count={int(row['q2_count'])}")
    if row["focal_rank_count"] != 2:
        reasons.append(f"focal_rank_count={int(row['focal_rank_count'])}")
    if pd.notna(row["focal_gap_sec"]) and row["focal_gap_sec"] <= 0:
        reasons.append("nonpositive_focal_gap")
    if row["q2_result_merge_missing"] > 0:
        reasons.append(f"missing_q2_race_results={int(row['q2_result_merge_missing'])}")
    if pd.notna(row.get("manual_reason_code")):
        reasons.append(str(row["manual_reason_code"]))
    return "; ".join(reasons) if reasons else "included"


def make_focal_pairs(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    variables = ["race_points", "top_ten", "grid_position", "q1_sec"]
    for (season, round_number), group in panel.groupby(["season", "round"], sort=True):
        by_rank = group.set_index("q2_rank")
        if 10 not in by_rank.index or 11 not in by_rank.index:
            continue
        r10, r11 = by_rank.loc[10], by_rank.loc[11]
        row = {
            "race_id": r10["race_id"],
            "season": int(season),
            "round": int(round_number),
            "race_name": r10["race_name"],
            "rank10_driver_id": r10["driver_id"],
            "rank11_driver_id": r11["driver_id"],
            "rank10_driver": r10["driver_name"],
            "rank11_driver": r11["driver_name"],
            "score_gap_sec": float(r11["q2_sec"] - r10["q2_sec"]),
            "rank10_has_prior_history": int(r10["has_prior_history"]),
            "rank11_has_prior_history": int(r11["has_prior_history"]),
            "complete_prior_pair": int(r10["has_prior_history"] and r11["has_prior_history"]),
            "no_history_diff": float(r10["no_prior_history"] - r11["no_prior_history"]),
            "prior_history_diff": float(r10["has_prior_history"] - r11["has_prior_history"]),
            "prior_points_zero_diff": float(r10["prior_points_zero"] - r11["prior_points_zero"]),
        }
        for variable in variables:
            row[f"{variable}_diff"] = float(r10[variable] - r11[variable]) if pd.notna(r10[variable]) and pd.notna(r11[variable]) else np.nan
        row["q1_advantage_sec"] = -row["q1_sec_diff"] if pd.notna(row["q1_sec_diff"]) else np.nan
        if row["complete_prior_pair"]:
            row["prior_points_complete_diff"] = float(r10["prior_season_points"] - r11["prior_season_points"])
            row["prior_pps_complete_diff"] = float(r10["prior_points_per_start"] - r11["prior_points_per_start"])
        else:
            row["prior_points_complete_diff"] = np.nan
            row["prior_pps_complete_diff"] = np.nan
        rows.append(row)
    pairs = pd.DataFrame(rows)
    pairs["gap_quartile"] = pd.qcut(
        pairs["score_gap_sec"].rank(method="first"), 4, labels=[1, 2, 3, 4]
    ).astype(int)
    return pairs


def main() -> None:
    ensure_dirs()
    config = load_config()
    qualifying = pd.read_csv(ROOT / "data" / "interim" / "qualifying.csv")
    results = pd.read_csv(ROOT / "data" / "interim" / "results.csv")
    history = pd.read_csv(ROOT / "data" / "interim" / "driver_season_history.csv")
    qualifying = qualifying[qualifying["season"].isin(config["qualifying_seasons"])].copy()
    qualifying["race_id"] = qualifying["season"].astype(str) + "_" + qualifying["round"].astype(str).str.zfill(2)
    qualifying["driver_name"] = qualifying["driver_given_name"].fillna("") + " " + qualifying["driver_family_name"].fillna("")
    qualifying["_source_row"] = np.arange(len(qualifying))
    ranked = (
        qualifying.loc[qualifying["q2_sec"].notna()]
        .sort_values(["season", "round", "q2_sec", "qualifying_position", "driver_id"])
        .copy()
    )
    ranked["q2_rank"] = ranked.groupby(["season", "round"]).cumcount() + 1
    qualifying = qualifying.merge(
        ranked[["_source_row", "q2_rank"]], on="_source_row", how="left", validate="one_to_one"
    ).drop(columns="_source_row")

    result_cols = ["season", "round", "driver_id", "finish_position", "race_points", "grid_position", "top_ten", "status"]
    merged = qualifying.merge(results[result_cols], on=["season", "round", "driver_id"], how="left", validate="one_to_one")
    merged["is_focal_rank"] = merged["q2_rank"].isin([10, 11])
    merged["q2_result_missing"] = merged["q2_sec"].notna() & merged["race_points"].isna()
    audit = (
        merged.groupby(["season", "round", "race_id", "race_name"], as_index=False)
        .agg(
            qualifying_records=("driver_id", "size"),
            q2_count=("q2_sec", lambda x: int(x.notna().sum())),
            q3_count=("q3_sec", lambda x: int(x.notna().sum())),
            focal_rank_count=("is_focal_rank", lambda x: int(x.sum())),
            q2_result_merge_missing=("q2_result_missing", lambda x: int(x.sum())),
        )
    )
    focal_wide = (
        merged.loc[merged["q2_rank"].isin([10, 11])]
        .pivot_table(index=["season", "round"], columns="q2_rank", values="q2_sec", aggfunc="first")
        .reset_index()
    )
    if 10 in focal_wide.columns and 11 in focal_wide.columns:
        focal_wide["focal_gap_sec"] = focal_wide[11] - focal_wide[10]
    else:
        focal_wide["focal_gap_sec"] = np.nan
    audit = audit.merge(
        focal_wide[["season", "round", "focal_gap_sec"]], on=["season", "round"], how="left"
    )
    manual = pd.read_csv(ROOT / "config" / "manual_exclusions.csv")
    audit = audit.merge(
        manual.rename(columns={"reason_code": "manual_reason_code", "reason": "manual_reason"}),
        on=["season", "round"], how="left"
    )
    audit["reason"] = audit.apply(audit_reason, axis=1)
    audit["included"] = audit["reason"].eq("included")
    audit.to_csv(ROOT / "data" / "processed" / "race_audit.csv", index=False)

    included_ids = set(audit.loc[audit["included"], "race_id"])
    panel = merged[merged["race_id"].isin(included_ids) & merged["q2_sec"].notna()].copy()
    history = history.rename(columns={
        "season": "prior_season",
        "season_points": "prior_season_points",
        "season_starts": "prior_season_starts",
        "season_result_records": "prior_season_result_records",
        "season_points_per_start": "prior_points_per_start",
    })
    panel["prior_season"] = panel["season"] - 1
    panel = panel.merge(
        history[["prior_season", "driver_id", "prior_season_points", "prior_season_starts", "prior_season_result_records", "prior_points_per_start"]],
        on=["prior_season", "driver_id"], how="left", validate="many_to_one"
    )
    panel["has_prior_history"] = (panel["prior_season_starts"].fillna(0) > 0).astype(int)
    panel["no_prior_history"] = 1 - panel["has_prior_history"]
    panel["prior_points_zero"] = panel["prior_season_points"].fillna(0.0)
    panel.loc[panel["has_prior_history"].eq(0), "prior_points_per_start"] = np.nan
    panel["treated_q3"] = (panel["q2_rank"] <= 10).astype(int)
    panel.to_csv(ROOT / "data" / "processed" / "q2_driver_race_panel.csv", index=False)

    neighbor_panel = panel[panel["q2_rank"].isin(config["neighbor_ranks"])].copy()
    neighbor_panel.to_csv(ROOT / "data" / "processed" / "rank_8_13_panel.csv", index=False)
    pairs = make_focal_pairs(panel)
    pairs.to_csv(ROOT / "data" / "processed" / "focal_pairs.csv", index=False)

    log = {
        "audited_races": int(len(audit)),
        "included_races": int(audit["included"].sum()),
        "excluded_races": int((~audit["included"]).sum()),
        "q2_driver_race_rows": int(len(panel)),
        "q2_count_min": int(audit.loc[audit["included"], "q2_count"].min()),
        "q2_count_max": int(audit.loc[audit["included"], "q2_count"].max()),
        "focal_pairs": int(len(pairs)),
        "complete_prior_pairs": int(pairs["complete_prior_pair"].sum()),
        "pairs_with_any_no_prior_history": int((pairs["complete_prior_pair"] == 0).sum()),
        "rank10_no_history": int((pairs["rank10_has_prior_history"] == 0).sum()),
        "rank11_no_history": int((pairs["rank11_has_prior_history"] == 0).sum()),
        "unmatched_result_rows_in_included_races": int(panel["race_points"].isna().sum()),
    }
    (ROOT / "output" / "logs" / "02_construct.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(json.dumps(log, indent=2))


if __name__ == "__main__":
    main()
