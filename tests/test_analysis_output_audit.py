import csv
import json
from pathlib import Path

import pytest

from scripts.audit.analysis_outputs import audit_analysis_outputs


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_analyses(root: Path) -> tuple[Path, Path]:
    standard, visibility = root / "standard", root / "visibility"
    batch_manifest_hash, batch_results_hash = "a" * 64, "b" * 64
    standard_manifest = {
        "expected_scenes": 2,
        "records": 2,
        "completed_or_reused": 2,
        "execution_failures": 0,
        "scene_rows": 2,
        "pair_rows": 2,
        "surface_rows": 1,
        "batch_manifest_sha256": batch_manifest_hash,
        "batch_results_sha256": batch_results_hash,
    }
    visibility_manifest = {
        "expected_scenes": 2,
        "execution_failures": 0,
        "scene_rows": 2,
        "pair_rows": 2,
        "batch_manifest_sha256": batch_manifest_hash,
        "batch_results_sha256": batch_results_hash,
    }
    write_json(standard / "manifest.json", standard_manifest)
    write_json(visibility / "manifest.json", visibility_manifest)
    write_json(standard / "failed_runs.json", [])
    write_json(visibility / "failed_runs.json", [])
    scenes = [
        {"batch_index": 0, "dataset": "set", "scene": "one", "state": "reused_verified_run"},
        {"batch_index": 1, "dataset": "set", "scene": "two", "state": "completed"},
    ]
    write_csv(standard / "scenes.csv", scenes)
    write_csv(visibility / "scene_visibility.csv", scenes)
    pairs = [
        {"dataset": "set", "scene": "one", "source_index": 0, "target_index": 1},
        {"dataset": "set", "scene": "two", "source_index": 0, "target_index": 1},
    ]
    write_csv(standard / "pair_diagnostics.csv", pairs)
    write_csv(visibility / "pair_visibility.csv", pairs)
    write_csv(standard / "surface_diagnostics.csv", [{"dataset": "set", "scene": "one", "direction": "prediction_to_reference"}])
    write_csv(
        standard / "rankings.csv",
        [
            {"dataset": "set", "scene": "one", "metric": "metric.pose.auc30", "population": 2, "failure_rank": 1, "direction": "lower_is_worse"},
            {"dataset": "set", "scene": "two", "metric": "metric.pose.auc30", "population": 2, "failure_rank": 2, "direction": "lower_is_worse"},
        ],
    )
    return standard, visibility


def test_audit_accepts_matching_complete_analyses(tmp_path: Path) -> None:
    standard, visibility = make_analyses(tmp_path)
    report = audit_analysis_outputs(standard, visibility, expected_scenes=2)
    assert report["passed"] is True
    assert report["expected_scenes"] == 2
    assert report["directed_pairs"] == 2


def test_audit_rejects_visibility_pair_population_change(tmp_path: Path) -> None:
    standard, visibility = make_analyses(tmp_path)
    write_csv(
        visibility / "pair_visibility.csv",
        [
            {"dataset": "set", "scene": "one", "source_index": 0, "target_index": 1},
            {"dataset": "set", "scene": "two", "source_index": 1, "target_index": 0},
        ],
    )
    with pytest.raises(ValueError, match="directed-pair populations differ"):
        audit_analysis_outputs(standard, visibility, expected_scenes=2)


def test_audit_rejects_incomplete_ranking_group(tmp_path: Path) -> None:
    standard, visibility = make_analyses(tmp_path)
    write_csv(
        standard / "rankings.csv",
        [{"dataset": "set", "scene": "one", "metric": "metric.pose.auc30", "population": 2, "failure_rank": 1, "direction": "lower_is_worse"}],
    )
    with pytest.raises(ValueError, match="Incomplete ranks"):
        audit_analysis_outputs(standard, visibility, expected_scenes=2)
