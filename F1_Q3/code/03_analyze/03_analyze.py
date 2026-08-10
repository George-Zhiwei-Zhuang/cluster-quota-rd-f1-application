#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import (
    ROOT,
    ensure_dirs,
    load_config,
    mean_cluster,
    mean_iid,
    season_block_bootstrap,
    sign_enumeration_pvalue,
    slope_cluster,
    two_way_cluster_ols,
)


VARIABLES = {
    "race_points": ("Race points", "race_points_diff", "race_points", None),
    "top_ten": ("Top-ten finish", "top_ten_diff", "top_ten", None),
    "grid_position": ("Starting-grid position", "grid_position_diff", "grid_position", None),
    "prior_points_complete": (
        "Prior-season points (complete pair)",
        "prior_points_complete_diff",
        "prior_season_points",
        "complete_prior_pair",
    ),
    "prior_pps_complete": (
        "Prior-season points per start (complete pair)",
        "prior_pps_complete_diff",
        "prior_points_per_start",
        "complete_prior_pair",
    ),
    "no_prior_history": (
        "No-prior-season-history indicator",
        "no_history_diff",
        "no_prior_history",
        None,
    ),
    "q1_advantage": ("Q1 lap-time advantage, seconds", "q1_advantage_sec", "q1_score", None),
}


def stacked_two_way(panel: pd.DataFrame, value_column: str, complete_pair: bool) -> dict:
    data = panel[panel["q2_rank"].isin([10, 11])].copy()
    if value_column == "q1_score":
        data["analysis_value"] = -data["q1_sec"]
    else:
        data["analysis_value"] = data[value_column]
    if complete_pair:
        good = (
            data.groupby("race_id")["has_prior_history"].agg(lambda x: len(x) == 2 and bool(x.eq(1).all()))
        )
        data = data[data["race_id"].isin(good[good].index)]
    data = data.dropna(subset=["analysis_value"])
    valid_races = data.groupby("race_id").size()
    data = data[data["race_id"].isin(valid_races[valid_races.eq(2)].index)].copy()
    data["treated"] = data["q2_rank"].eq(10).astype(float)
    # Absorb race fixed effects before forming the two-way race-by-driver
    # sandwich. This preserves the paired-difference estimand and prevents
    # cross-track lap-time levels from entering the driver-cluster score.
    data["analysis_value_within"] = data["analysis_value"] - data.groupby("race_id")["analysis_value"].transform("mean")
    data["treated_within"] = data["treated"] - data.groupby("race_id")["treated"].transform("mean")
    X = data[["treated_within"]].to_numpy(float)
    return two_way_cluster_ols(
        data["analysis_value_within"], X, data["race_id"], data["driver_id"], coefficient=0
    )


def primary_contrasts(pairs: pd.DataFrame, panel: pd.DataFrame, config: dict) -> pd.DataFrame:
    rows = []
    for offset, (key, (label, pair_col, driver_col, complete_flag)) in enumerate(VARIABLES.items()):
        values = pairs[pair_col]
        iid = mean_iid(values)
        cluster = mean_cluster(values, pairs["season"])
        bootstrap = season_block_bootstrap(
            values,
            pairs["season"],
            int(config["bootstrap_replications"]),
            int(config["seed"]) + offset,
        )
        two_way = stacked_two_way(panel, driver_col, complete_flag is not None)
        for method, result in (
            ("race_iid", iid),
            ("season_cluster", cluster),
            ("season_block_bootstrap", bootstrap),
            ("race_driver_two_way", two_way),
        ):
            rows.append({
                "variable": key,
                "label": label,
                "method": method,
                "estimate": result.get("estimate"),
                "se": result.get("se"),
                "ci_low": result.get("ci_low"),
                "ci_high": result.get("ci_high"),
                "p_value": result.get("p_value"),
                "n": result.get("n"),
                "clusters": result.get("clusters", result.get("race_clusters")),
                "driver_clusters": result.get("driver_clusters"),
            })
    return pd.DataFrame(rows)


def quartile_diagnostics(pairs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    variables = {
        "race_points": "race_points_diff",
        "prior_points_complete": "prior_points_complete_diff",
        "prior_pps_complete": "prior_pps_complete_diff",
        "no_prior_history": "no_history_diff",
    }
    for quartile in range(1, 5):
        subset = pairs[pairs["gap_quartile"].eq(quartile)]
        for key, column in variables.items():
            result = mean_iid(subset[column])
            rows.append({"gap_quartile": quartile, "variable": key, **result})
        rows.append({
            "gap_quartile": quartile,
            "variable": "any_no_prior_history_share",
            **mean_iid(1 - subset["complete_prior_pair"]),
        })
    return pd.DataFrame(rows)


def continuous_gap(pairs: pd.DataFrame) -> pd.DataFrame:
    log_gap = np.log2(pairs["score_gap_sec"])
    rows = []
    for key, (_, column, _, _) in VARIABLES.items():
        result = slope_cluster(pairs[column], log_gap, pairs["season"])
        rows.append({"variable": key, **result})
    return pd.DataFrame(rows)


def adjacent_pair_data(panel: pd.DataFrame, variable: str, upper: int, lower: int) -> pd.DataFrame:
    subset = panel[panel["q2_rank"].isin([upper, lower])].copy()
    if variable == "q1_advantage":
        value = "q1_sec"
        multiplier = -1.0
    else:
        value = variable
        multiplier = 1.0
    wide = subset.pivot_table(
        index=["race_id", "season"], columns="q2_rank", values=value, aggfunc="first"
    ).reset_index()
    if upper not in wide.columns or lower not in wide.columns:
        wide["difference"] = np.nan
    else:
        wide["difference"] = multiplier * (wide[upper] - wide[lower])
    # The prior-season variables remain missing for drivers without prior-season
    # starts, so the pivot-table dropna logic itself enforces complete pairs.
    return wide[["race_id", "season", "difference"]]


def adjacent_results(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    pairs = [(8, 9), (9, 10), (10, 11), (11, 12), (12, 13)]
    variables = {
        "race_points": "race_points",
        "top_ten": "top_ten",
        "grid_position": "grid_position",
        "prior_points_complete": "prior_season_points",
        "no_prior_history": "no_prior_history",
    }
    for key, column in variables.items():
        for upper, lower in pairs:
            data = adjacent_pair_data(panel, column, upper, lower)
            result = mean_cluster(data["difference"], data["season"])
            rows.append({
                "variable": key,
                "upper_rank": upper,
                "lower_rank": lower,
                "pair": f"{upper}-{lower}",
                **result,
            })
    return pd.DataFrame(rows)


def boundary_excess(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    variables = {
        "race_points": "race_points",
        "top_ten": "top_ten",
        "grid_position": "grid_position",
        "prior_points_complete": "prior_season_points",
        "no_prior_history": "no_prior_history",
    }
    nearest_rows = []
    all_rows = []
    nearest_race_points = None
    for key, value in variables.items():
        sub = panel[panel["q2_rank"].isin([9, 10, 11, 12])]
        wide = sub.pivot_table(index=["race_id", "season"], columns="q2_rank", values=value, aggfunc="first").reset_index()
        required = [9, 10, 11, 12]
        if all(rank in wide.columns for rank in required):
            wide = wide.dropna(subset=required)
            if key == "prior_points_complete":
                hist = sub.pivot_table(index=["race_id", "season"], columns="q2_rank", values="has_prior_history", aggfunc="first").reset_index()
                hist = hist.dropna(subset=required)
                complete_ids = hist.loc[hist[required].eq(1).all(axis=1), "race_id"]
                wide = wide[wide["race_id"].isin(complete_ids)]
            wide["focal"] = wide[10] - wide[11]
            wide["neighbor_mean"] = 0.5 * ((wide[9] - wide[10]) + (wide[11] - wide[12]))
            wide["excess"] = wide["focal"] - wide["neighbor_mean"]
            result = mean_cluster(wide["excess"], wide["season"])
            nearest_rows.append({
                "variable": key,
                "focal_contrast": wide["focal"].mean(),
                "neighbor_mean": wide["neighbor_mean"].mean(),
                **result,
            })
            if key == "race_points":
                nearest_race_points = wide[["season", "excess"]].copy()

        sub6 = panel[panel["q2_rank"].isin([8, 9, 10, 11, 12, 13])]
        wide6 = sub6.pivot_table(index=["race_id", "season"], columns="q2_rank", values=value, aggfunc="first").reset_index()
        required6 = [8, 9, 10, 11, 12, 13]
        if all(rank in wide6.columns for rank in required6):
            wide6 = wide6.dropna(subset=required6)
            if key == "prior_points_complete":
                hist6 = sub6.pivot_table(index=["race_id", "season"], columns="q2_rank", values="has_prior_history", aggfunc="first").reset_index().dropna(subset=required6)
                complete_ids = hist6.loc[hist6[required6].eq(1).all(axis=1), "race_id"]
                wide6 = wide6[wide6["race_id"].isin(complete_ids)]
            wide6["focal"] = wide6[10] - wide6[11]
            wide6["placebo_mean"] = 0.25 * (
                (wide6[8] - wide6[9]) + (wide6[9] - wide6[10])
                + (wide6[11] - wide6[12]) + (wide6[12] - wide6[13])
            )
            wide6["excess"] = wide6["focal"] - wide6["placebo_mean"]
            result = mean_cluster(wide6["excess"], wide6["season"])
            all_rows.append({
                "variable": key,
                "focal_contrast": wide6["focal"].mean(),
                "all_placebo_mean": wide6["placebo_mean"].mean(),
                **result,
            })
    nearest = pd.DataFrame(nearest_rows)
    if nearest_race_points is not None:
        sign = sign_enumeration_pvalue(nearest_race_points["excess"], nearest_race_points["season"])
        nearest.loc[nearest["variable"].eq("race_points"), "sign_enumeration_p"] = sign["p_value"]
        nearest.loc[nearest["variable"].eq("race_points"), "sign_enumerations"] = sign["enumerations"]
    return nearest, pd.DataFrame(all_rows)


def extreme_gap(pairs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    ordered = pairs.sort_values("score_gap_sec", ascending=False)
    for removed in [0, 1, 5, 10, 20]:
        subset = ordered.iloc[removed:]
        for variable in ["race_points_diff", "top_ten_diff"]:
            result = mean_iid(subset[variable])
            rows.append({"removed_largest_gaps": removed, "variable": variable, **result})
    return pd.DataFrame(rows)


def leave_one_season(pairs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for season in sorted(pairs["season"].unique()):
        subset = pairs[pairs["season"].ne(season)]
        rows.append({
            "omitted_season": season,
            "race_points_estimate": subset["race_points_diff"].mean(),
            "top_ten_estimate": subset["top_ten_diff"].mean(),
            "n": len(subset),
        })
    return pd.DataFrame(rows)


def rookie_audit(pairs: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {"metric": "focal_pairs", "value": len(pairs)},
        {"metric": "complete_prior_pairs", "value": int(pairs["complete_prior_pair"].sum())},
        {"metric": "pairs_with_any_no_prior_history", "value": int((1 - pairs["complete_prior_pair"]).sum())},
        {"metric": "rank10_no_prior_history", "value": int((1 - pairs["rank10_has_prior_history"]).sum())},
        {"metric": "rank11_no_prior_history", "value": int((1 - pairs["rank11_has_prior_history"]).sum())},
    ]
    for name, column in (
        ("prior_points_zero_inclusive_difference", "prior_points_zero_diff"),
        ("prior_points_complete_pair_difference", "prior_points_complete_diff"),
        ("prior_points_per_start_complete_pair_difference", "prior_pps_complete_diff"),
        ("no_prior_history_indicator_difference", "no_history_diff"),
    ):
        result = mean_cluster(pairs[column], pairs["season"])
        rows.append({"metric": name, "value": result["estimate"], "se": result["se"],
                     "ci_low": result["ci_low"], "ci_high": result["ci_high"],
                     "p_value": result["p_value"], "n": result["n"], "clusters": result["clusters"]})
    return pd.DataFrame(rows)


def main() -> None:
    ensure_dirs()
    config = load_config()
    pairs = pd.read_csv(ROOT / "data" / "processed" / "focal_pairs.csv")
    panel = pd.read_csv(ROOT / "data" / "processed" / "q2_driver_race_panel.csv")
    primary = primary_contrasts(pairs, panel, config)
    quartiles = quartile_diagnostics(pairs)
    continuous = continuous_gap(pairs)
    adjacent = adjacent_results(panel)
    nearest, all_placebos = boundary_excess(panel)
    extreme = extreme_gap(pairs)
    loo = leave_one_season(pairs)
    rookie = rookie_audit(pairs)

    quantile_levels = [0, .05, .25, .50, .75, .95, 1]
    gap_stats = pd.DataFrame({
        "quantile": quantile_levels,
        "score_gap_sec": pairs["score_gap_sec"].quantile(quantile_levels).to_numpy(),
    })
    outputs = {
        "primary_contrasts_long.csv": primary,
        "gap_quartile_diagnostics.csv": quartiles,
        "continuous_gap_diagnostics.csv": continuous,
        "adjacent_rank_contrasts.csv": adjacent,
        "boundary_excess_nearest.csv": nearest,
        "boundary_excess_all_placebos.csv": all_placebos,
        "extreme_gap_sensitivity.csv": extreme,
        "leave_one_season_out.csv": loo,
        "rookie_history_audit.csv": rookie,
        "score_gap_quantiles.csv": gap_stats,
    }
    for filename, frame in outputs.items():
        frame.to_csv(ROOT / "data" / "processed" / filename, index=False)

    main_est = primary[(primary["method"] == "season_cluster")].set_index("variable")
    log = {
        "races": int(len(pairs)),
        "race_points_estimate": float(main_est.loc["race_points", "estimate"]),
        "top_ten_estimate": float(main_est.loc["top_ten", "estimate"]),
        "grid_position_estimate": float(main_est.loc["grid_position", "estimate"]),
        "prior_points_complete_estimate": float(main_est.loc["prior_points_complete", "estimate"]),
        "prior_pps_complete_estimate": float(main_est.loc["prior_pps_complete", "estimate"]),
        "q1_advantage_estimate": float(main_est.loc["q1_advantage", "estimate"]),
        "minimum_gap": float(pairs["score_gap_sec"].min()),
        "maximum_gap": float(pairs["score_gap_sec"].max()),
    }
    (ROOT / "output" / "logs" / "03_analyze.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(json.dumps(log, indent=2))


if __name__ == "__main__":
    main()
