import csv
import json
from pathlib import Path

import pytest

from scripts.audit.gt_depth_error_outputs import PREFIXES, audit_gt_depth_error
from scripts.failure_mining.compare_baselines import sha256, write_csv


def build_outputs(root: Path) -> None:
    views = []
    for index, values in enumerate(([0.0, 1.0], [2.0, 3.0])):
        row = {
            "dataset": "scannetpp",
            "scene": "s",
            "view_index": index,
            "pixels": 4,
            "predicted_valid_count": 3,
            "ground_truth_valid_count": 4,
            "joint_valid_count": 2,
        }
        for prefix in PREFIXES:
            row.update(
                {
                    f"{prefix}.count": 2,
                    f"{prefix}.finite_count": 2,
                    f"{prefix}.mean": sum(values) / 2,
                    f"{prefix}.min": min(values),
                    f"{prefix}.max": max(values),
                    f"{prefix}.p50": sum(values) / 2,
                    f"{prefix}.p90": max(values),
                    f"{prefix}.p95": max(values),
                    f"{prefix}.p99": max(values),
                }
            )
        views.append(row)
    scene = {
        "dataset": "scannetpp",
        "scene": "s",
        "views": 2,
        "pixels": 8,
        "predicted_valid_count": 6,
        "ground_truth_valid_count": 8,
        "joint_valid_count": 4,
        "predicted_valid_ratio": 0.75,
        "ground_truth_valid_ratio": 1.0,
        "joint_valid_ratio": 0.5,
        "joint_over_ground_truth_valid_ratio": 0.5,
    }
    for prefix in PREFIXES:
        scene.update(
            {
                f"{prefix}.count": 4,
                f"{prefix}.finite_count": 4,
                f"{prefix}.mean": 1.5,
                f"{prefix}.min": 0.0,
                f"{prefix}.max": 3.0,
                f"{prefix}.p50": 1.5,
                f"{prefix}.p90": 3.0,
                f"{prefix}.p95": 3.0,
                f"{prefix}.p99": 3.0,
            }
        )
    view_path = root / "view_gt_depth_error.csv"
    scene_path = root / "scene_gt_depth_error.csv"
    write_csv(view_path, views)
    write_csv(scene_path, [scene])
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "scenes": ["s"],
                "scene_rows": 1,
                "view_rows": 2,
                "view_output_sha256": sha256(view_path),
                "scene_output_sha256": sha256(scene_path),
                "uses_ground_truth_analysis": True,
                "test_time_policy_input": False,
                "failure_labels": None,
                "quality_thresholds": None,
            }
        )
    )


def test_recomputes_scene_counts_ratios_and_moments(tmp_path: Path) -> None:
    build_outputs(tmp_path)
    report = audit_gt_depth_error(tmp_path)
    assert report["passed"] is True
    assert report["view_rows"] == 2


def test_rejects_tampered_scene_mean(tmp_path: Path) -> None:
    build_outputs(tmp_path)
    path = tmp_path / "scene_gt_depth_error.csv"
    path.write_text(path.read_text().replace("1.5", "9.5", 1))
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    manifest["scene_output_sha256"] = sha256(path)
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="mean differs"):
        audit_gt_depth_error(tmp_path)


def test_accepts_empty_metric_summary_when_view_has_no_joint_pixels(tmp_path: Path) -> None:
    build_outputs(tmp_path)
    view_path = tmp_path / "view_gt_depth_error.csv"
    with view_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows[1]["predicted_valid_count"] = "0"
    rows[1]["joint_valid_count"] = "0"
    for prefix in PREFIXES:
        rows[1][f"{prefix}.count"] = "0"
        rows[1][f"{prefix}.finite_count"] = "0"
        for field in ("mean", "min", "max"):
            rows[1][f"{prefix}.{field}"] = ""
    write_csv(view_path, rows)

    scene_path = tmp_path / "scene_gt_depth_error.csv"
    with scene_path.open(newline="", encoding="utf-8") as handle:
        scene = list(csv.DictReader(handle))[0]
    scene["predicted_valid_count"] = "3"
    scene["joint_valid_count"] = "2"
    scene["predicted_valid_ratio"] = str(3 / 8)
    scene["joint_valid_ratio"] = str(2 / 8)
    scene["joint_over_ground_truth_valid_ratio"] = str(2 / 8)
    for prefix in PREFIXES:
        scene[f"{prefix}.count"] = "2"
        scene[f"{prefix}.finite_count"] = "2"
        scene[f"{prefix}.mean"] = "0.5"
        scene[f"{prefix}.min"] = "0.0"
        scene[f"{prefix}.max"] = "1.0"
    write_csv(scene_path, [scene])

    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["view_output_sha256"] = sha256(view_path)
    manifest["scene_output_sha256"] = sha256(scene_path)
    manifest_path.write_text(json.dumps(manifest))
    assert audit_gt_depth_error(tmp_path)["passed"] is True
