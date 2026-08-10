#!/usr/bin/env python3
from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import scipy

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import ROOT, ensure_dirs, sha256_file


def require(condition: bool, message: str, checks: list[dict]) -> None:
    checks.append({"check": message, "passed": bool(condition)})
    if not condition:
        raise AssertionError(message)


def main() -> None:
    ensure_dirs()
    checks = []
    manifest_path = ROOT / "data" / "raw" / "jolpica" / "manifest.json"
    require(manifest_path.exists(), "raw manifest exists", checks)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(len(manifest) == 100, "raw manifest contains 100 paginated responses", checks)
    for row in manifest:
        path = ROOT / row["file"]
        require(path.exists(), f"raw file exists: {row['file']}", checks)
        require(sha256_file(path) == row["sha256"], f"raw checksum matches: {row['file']}", checks)

    audit = pd.read_csv(ROOT / "data" / "processed" / "race_audit.csv")
    pairs = pd.read_csv(ROOT / "data" / "processed" / "focal_pairs.csv")
    panel = pd.read_csv(ROOT / "data" / "processed" / "q2_driver_race_panel.csv")
    require(len(audit) == 198, "198 race weekends audited", checks)
    require(int(audit["included"].sum()) == 193, "193 rule-based race weekends included", checks)
    require(len(pairs) == 193, "one focal pair per included weekend", checks)
    require(len(panel) == 3010, "3010 classified Q2 driver-race observations", checks)
    require(not pairs.duplicated("race_id").any(), "no duplicate focal race keys", checks)
    require(pairs["score_gap_sec"].gt(0).all(), "all included focal score gaps are positive", checks)
    require(np.isclose(pairs["score_gap_sec"].min(), .002, atol=1e-9), "minimum score gap is 0.002 seconds", checks)
    require(np.isclose(pairs["score_gap_sec"].max(), 1.564, atol=1e-9), "maximum score gap is 1.564 seconds", checks)
    require(int(pairs["complete_prior_pair"].sum()) == 130, "130 complete prior-history focal pairs", checks)

    expected_tables = [
        "table_2_primary_contrasts.csv", "table_3_continuous_gap_diagnostics.csv",
        "table_4_boundary_vs_neighboring.csv", "table_a1_excluded_races.csv",
        "table_a2_rookie_history_audit.csv", "table_a8_draft_vs_rebuild.csv",
    ]
    expected_figures = [
        "figure_5_score_gap_distribution.png", "figure_6_gap_quartile_diagnostics.png",
        "figure_7_quota_vs_neighboring_gradients.png",
        "figure_8_dependence_and_extreme_gap_sensitivity.png",
        "figure_a1_rookie_history_sensitivity.png",
    ]
    for filename in expected_tables:
        require((ROOT / "output" / "tables" / filename).exists(), f"table exists: {filename}", checks)
    for filename in expected_figures:
        require((ROOT / "output" / "figures" / filename).exists(), f"figure exists: {filename}", checks)

    checksum_rows = []
    for directory in [ROOT / "data" / "processed", ROOT / "output" / "tables", ROOT / "output" / "figures"]:
        for path in sorted(directory.glob("*")):
            if path.is_file():
                checksum_rows.append({"file": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    pd.DataFrame(checksum_rows).to_csv(ROOT / "output" / "logs" / "output_checksums.csv", index=False)
    session = {
        "python": sys.version,
        "platform": platform.platform(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__,
        "checks_passed": int(sum(item["passed"] for item in checks)),
        "checks_total": int(len(checks)),
    }
    (ROOT / "output" / "logs" / "session_info.json").write_text(json.dumps(session, indent=2), encoding="utf-8")
    (ROOT / "output" / "logs" / "06_validation.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    print(json.dumps(session, indent=2))


if __name__ == "__main__":
    main()
