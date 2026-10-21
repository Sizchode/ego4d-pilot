import csv
import json
from pathlib import Path

import pytest

from scripts.failure_mining.analyze_gt_geometry_factor_profiles import FACTORS
from scripts.failure_mining.compare_baselines import sha256, write_csv
from scripts.failure_mining.prepare_factor_targets import build_targets


def build_profiles(root: Path) -> None:
    scenes = ["a", "b"]
    rows = []
    for scene_index, scene in enumerate(scenes):
        row = {"model": "m", "scene": scene, "metric": "fscore"}
        for factor_index, (factor, _field) in enumerate(FACTORS):
            row[f"{factor}_conditional_gain"] = scene_index + factor_index / 10
        rows.append(row)
    path = root / "case_metric_factor_profiles.csv"
    write_csv(path, rows)
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "models": ["m"],
                "scenes": scenes,
                "profile_output_sha256": sha256(path),
                "uses_ground_truth_intervention": True,
                "test_time_policy_input": False,
                "success_threshold": None,
            }
        )
    )


def test_builds_one_continuous_target_per_scene_and_factor(tmp_path: Path) -> None:
    build_profiles(tmp_path)
    rows, manifest = build_targets(
        tmp_path,
        dataset="d",
        model="m",
        reconstruction_metric="fscore",
    )
    assert len(rows) == 6
    assert [(row["scene"], row["factor"]) for row in rows] == [
        (scene, factor) for scene in ["a", "b"] for factor, _field in FACTORS
    ]
    assert rows[-1]["conditional_gain"] == 1.2
    assert manifest["test_time_policy_input"] is False


def test_rejects_tampered_profile_table(tmp_path: Path) -> None:
    build_profiles(tmp_path)
    path = tmp_path / "case_metric_factor_profiles.csv"
    rows = list(csv.DictReader(path.open()))
    rows[0]["extrinsics_conditional_gain"] = "999"
    write_csv(path, rows)
    with pytest.raises(ValueError, match="hash mismatch"):
        build_targets(
            tmp_path,
            dataset="d",
            model="m",
            reconstruction_metric="fscore",
        )
