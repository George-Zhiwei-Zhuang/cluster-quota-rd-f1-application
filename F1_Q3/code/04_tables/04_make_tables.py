#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import ROOT, ensure_dirs, fmt_number


METHOD_LABELS = {
    "race_iid": "Race-i.i.d. 95% CI",
    "season_cluster": "Season-cluster 95% CI",
    "season_block_bootstrap": "Season-bootstrap 95% CI",
    "race_driver_two_way": "Race x driver 95% CI",
}


def interval(row) -> str:
    return f"[{fmt_number(row.ci_low)}, {fmt_number(row.ci_high)}]"


def write_markdown(frame: pd.DataFrame, path: Path, title: str, note: str = "") -> None:
    def cell(value) -> str:
        if pd.isna(value):
            return ""
        return str(value).replace("|", "\\|").replace("\n", " ")

    header = "| " + " | ".join(cell(column) for column in frame.columns) + " |"
    rule = "| " + " | ".join("---" for _ in frame.columns) + " |"
    body = ["| " + " | ".join(cell(value) for value in row) + " |" for row in frame.itertuples(index=False, name=None)]
    text = f"# {title}\n\n" + "\n".join([header, rule, *body]) + "\n"
    if note:
        text += f"\nNotes: {note}\n"
    path.write_text(text, encoding="utf-8")


def table2(primary: pd.DataFrame) -> pd.DataFrame:
    order = [
        "race_points", "top_ten", "grid_position", "prior_points_complete",
        "prior_pps_complete", "no_prior_history", "q1_advantage",
    ]
    labels = primary.drop_duplicates("variable").set_index("variable")["label"]
    rows = []
    for variable in order:
        subset = primary[primary["variable"].eq(variable)].set_index("method")
        base = subset.loc["season_cluster"]
        row = {"Variable": labels[variable], "Estimate": fmt_number(base["estimate"]), "N": int(base["n"])}
        for method, label in METHOD_LABELS.items():
            row[label] = interval(subset.loc[method])
        rows.append(row)
    return pd.DataFrame(rows)


def table3(continuous: pd.DataFrame) -> pd.DataFrame:
    labels = {
        "race_points": "Race points",
        "top_ten": "Top-ten finish",
        "prior_points_complete": "Prior-season points (complete pair)",
        "prior_pps_complete": "Prior-season points per start (complete pair)",
        "no_prior_history": "No-prior-history indicator",
        "q1_advantage": "Q1 lap-time advantage",
    }
    rows = []
    for key, label in labels.items():
        row = continuous.loc[continuous["variable"].eq(key)].iloc[0]
        rows.append({
            "Rank-10 minus rank-11 variable": label,
            "Change per doubling of gap": fmt_number(row["estimate"]),
            "Season-cluster 95% CI": interval(row),
            "p-value": fmt_number(row["p_value"]),
            "N": int(row["n"]),
        })
    return pd.DataFrame(rows)


def table4(nearest: pd.DataFrame) -> pd.DataFrame:
    labels = {
        "race_points": "Race points",
        "top_ten": "Top-ten finish",
        "grid_position": "Starting-grid position",
        "prior_points_complete": "Prior-season points (common complete-history sample)",
        "no_prior_history": "No-prior-history indicator",
    }
    rows = []
    for key, label in labels.items():
        row = nearest.loc[nearest["variable"].eq(key)].iloc[0]
        rows.append({
            "Variable": label,
            "Q3 boundary contrast": fmt_number(row["focal_contrast"]),
            "Nearest-neighbor mean": fmt_number(row["neighbor_mean"]),
            "Boundary excess": fmt_number(row["estimate"]),
            "Season-cluster 95% CI": interval(row),
            "p-value": fmt_number(row["p_value"]),
            "N": int(row["n"]),
        })
    return pd.DataFrame(rows)


def sample_audit(audit: pd.DataFrame) -> pd.DataFrame:
    columns = ["season", "round", "race_name", "q2_count", "q3_count", "focal_gap_sec", "reason"]
    result = audit.loc[~audit["included"], columns].copy()
    result.columns = ["Season", "Round", "Race", "Q2 recorded times", "Q3 recorded times", "Rank 10-11 gap (sec)", "Exclusion reason"]
    return result


def manuscript_comparison(primary: pd.DataFrame, pairs: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    main = primary[primary["method"].eq("season_cluster")].set_index("variable")
    reference = {
        "Included races": 192,
        "Q2 driver-race observations": 2994,
        "Race-points contrast": 0.818,
        "Top-ten contrast": 0.083,
        "Starting-grid contrast": -2.469,
        "Prior-season-points contrast (zero-coded draft)": 5.568,
        "Prior-season points per start (zero-coded draft)": 0.632,
        "Q1 lap-time advantage": 0.047,
    }
    rebuilt = {
        "Included races": len(pairs),
        "Q2 driver-race observations": len(panel),
        "Race-points contrast": main.loc["race_points", "estimate"],
        "Top-ten contrast": main.loc["top_ten", "estimate"],
        "Starting-grid contrast": main.loc["grid_position", "estimate"],
        "Prior-season-points contrast (zero-coded draft)": pairs["prior_points_zero_diff"].mean(),
        "Prior-season points per start (zero-coded draft)": np.nan,
        "Q1 lap-time advantage": main.loc["q1_advantage", "estimate"],
    }
    rows = []
    for metric in reference:
        rows.append({"Metric": metric, "Current draft": reference[metric], "Rule-based rebuild": rebuilt[metric]})
    return pd.DataFrame(rows)


def main() -> None:
    ensure_dirs()
    processed = ROOT / "data" / "processed"
    out = ROOT / "output" / "tables"
    primary = pd.read_csv(processed / "primary_contrasts_long.csv")
    continuous = pd.read_csv(processed / "continuous_gap_diagnostics.csv")
    nearest = pd.read_csv(processed / "boundary_excess_nearest.csv")
    audit = pd.read_csv(processed / "race_audit.csv")
    rookie = pd.read_csv(processed / "rookie_history_audit.csv")
    pairs = pd.read_csv(processed / "focal_pairs.csv")
    panel = pd.read_csv(processed / "q2_driver_race_panel.csv")
    all_placebos = pd.read_csv(processed / "boundary_excess_all_placebos.csv")
    loo = pd.read_csv(processed / "leave_one_season_out.csv")
    adjacent = pd.read_csv(processed / "adjacent_rank_contrasts.csv")
    quartiles = pd.read_csv(processed / "gap_quartile_diagnostics.csv")
    gap_quantiles = pd.read_csv(processed / "score_gap_quantiles.csv")

    tables = {
        "table_2_primary_contrasts": table2(primary),
        "table_3_continuous_gap_diagnostics": table3(continuous),
        "table_4_boundary_vs_neighboring": table4(nearest),
        "table_a1_excluded_races": sample_audit(audit),
        "table_a2_rookie_history_audit": rookie,
        "table_a3_all_placebo_boundary_excess": all_placebos,
        "table_a4_leave_one_season_out": loo,
        "table_a5_adjacent_rank_contrasts": adjacent,
        "table_a6_gap_quartile_diagnostics": quartiles,
        "table_a7_score_gap_quantiles": gap_quantiles,
        "table_a8_draft_vs_rebuild": manuscript_comparison(primary, pairs, panel),
    }
    notes = {
        "table_2_primary_contrasts": "Prior-season performance variables use only races in which both focal drivers had at least one start in the preceding season. The history indicator is reported separately. Intervals are sampling-based sensitivities, not design-based randomization intervals.",
        "table_4_boundary_vs_neighboring": "Predetermined prior-season points use a common complete-history sample across ranks 9-12.",
        "table_a1_excluded_races": "Exclusions are generated by explicit institutional and data-integrity rules in config/manual_exclusions.csv and code/02_construct/02_construct_analysis.py.",
        "table_a8_draft_vs_rebuild": "The draft values are transcribed from Cluster_Based_RDD_Rewritten(20260808-123526).docx. The rebuilt column uses the current Jolpica snapshot and explicit rule-based exclusions.",
    }
    for stem, frame in tables.items():
        frame.to_csv(out / f"{stem}.csv", index=False)
        write_markdown(frame, out / f"{stem}.md", stem.replace("_", " ").title(), notes.get(stem, ""))
    log = {"tables": sorted([f"{stem}.csv" for stem in tables])}
    (ROOT / "output" / "logs" / "04_tables.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(json.dumps(log, indent=2))


if __name__ == "__main__":
    main()
