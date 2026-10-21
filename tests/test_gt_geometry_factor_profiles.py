import csv
import json
from pathlib import Path

import pytest

from scripts.failure_mining.analyze_gt_geometry_factor_profiles import (
    DIRECTIONS,
    STAGES,
    analyze,
)
from scripts.failure_mining.compare_baselines import sha256


def write_input(root: Path) -> None:
    models = ["m"]
    scenes = ["positive", "none", "tie"]
    rows = []
    stage_values = {
        "positive": (0.0, 0.125, 0.25, 0.75),
        "none": (0.75, 0.5, 0.25, 0.0),
        "tie": (0.0, 0.25, 0.5, 0.625),
    }
    for scene in scenes:
        for metric, direction in DIRECTIONS.items():
            values = stage_values[scene]
            if direction == "lower_is_better":
                values = tuple(-value for value in values)
            gains = [values[index + 1] - values[index] for index in range(3)]
            if direction == "lower_is_better":
                gains = [-value for value in gains]
            rows.append(
                {
                    "model": "m",
                    "scene": scene,
                    "metric": metric,
                    "direction": direction,
                    **{f"{stage}_value": value for stage, value in zip(STAGES, values)},
                    "extrinsics_gain_vs_native": gains[0],
                    "intrinsics_gain_after_extrinsics": gains[1],
                    "depth_gain_after_extrinsics_intrinsics": gains[2],
                    "full_gt_input_gain_vs_native": sum(gains),
                }
            )
    case_path = root / "case_metric_stages.csv"
    with case_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "models": models,
        "scenes": scenes,
        "cases": len(models) * len(scenes),
        "case_metric_rows": len(rows),
        "case_output_sha256": sha256(case_path),
        "stages": list(STAGES),
        "uses_ground_truth_intervention": True,
        "test_time_policy_input": False,
        "success_threshold": None,
    }
    (root / "manifest.json").write_text(json.dumps(manifest) + "\n")


def test_profiles_preserve_positive_none_and_tied_cases(tmp_path: Path) -> None:
    write_input(tmp_path)
    profiles, aggregate, manifest = analyze(tmp_path)
    fscore = {row["scene"]: row for row in profiles if row["metric"] == "fscore"}
    assert fscore["positive"]["best_positive_conditional_factors"] == "depth_after_extrinsics_intrinsics"
    assert fscore["none"]["best_positive_conditional_factors"] == "none"
    assert fscore["tie"]["best_positive_conditional_factors"] == "extrinsics;intrinsics_after_extrinsics"
    summary = next(row for row in aggregate if row["metric"] == "fscore")
    assert summary["tied_best_positive_cases"] == 1
    assert summary["no_positive_factor_cases"] == 1
    assert summary["depth_after_extrinsics_intrinsics_unique_best_positive_cases"] == 1
    assert manifest["success_threshold"] is None


def test_profiles_reject_stored_gain_mismatch(tmp_path: Path) -> None:
    write_input(tmp_path)
    path = tmp_path / "case_metric_stages.csv"
    rows = list(csv.DictReader(path.open()))
    rows[0]["extrinsics_gain_vs_native"] = "123"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    manifest["case_output_sha256"] = sha256(path)
    (tmp_path / "manifest.json").write_text(json.dumps(manifest) + "\n")
    with pytest.raises(ValueError, match="Stored gain mismatch"):
        analyze(tmp_path)
