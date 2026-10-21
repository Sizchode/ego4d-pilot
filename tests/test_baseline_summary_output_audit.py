import csv
import json
from pathlib import Path

import pytest

from scripts.audit.baseline_summary_outputs import audit_baseline_summary
from scripts.failure_mining.compare_baselines import sha256, write_csv
from scripts.failure_mining.summarize_baseline_results import (
    RUNTIME_FIELDS,
    summarize,
)
from tests.test_cross_model_output_audit import make_cross_outputs, write_json


def make_fixture(root: Path) -> tuple[Path, Path]:
    cross, _visibility, _batches = make_cross_outputs(root)
    cross_manifest_path = cross / "manifest.json"
    cross_manifest = json.loads(cross_manifest_path.read_text())
    for model_index, model in enumerate(cross_manifest["models"]):
        scenes_path = Path(cross_manifest["inputs"][model]["analysis_dir"]) / "scenes.csv"
        with scenes_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        for scene_index, row in enumerate(rows):
            row["run"] = f"/{model}/{scene_index}"
            row["views"] = str(scene_index + 2)
            for field_index, field in enumerate(RUNTIME_FIELDS):
                row[field] = str(10 + model_index + scene_index + field_index)
            row["metric.pose.auc30"] = str(0.5 + 0.1 * scene_index + 0.01 * model_index)
        write_csv(scenes_path, rows)
        cross_manifest["inputs"][model]["scenes_sha256"] = sha256(scenes_path)
    write_json(cross_manifest_path, cross_manifest)

    scenes, aggregates, provenance = summarize(cross)
    output = root / "summary"
    output.mkdir()
    scene_path = output / "scene_model_results.csv"
    aggregate_path = output / "dataset_model_summary.csv"
    write_csv(scene_path, scenes)
    write_csv(aggregate_path, aggregates)
    manifest = {
        **provenance,
        "scene_rows": len(scenes),
        "summary_rows": len(aggregates),
        "scene_output_sha256": sha256(scene_path),
        "summary_output_sha256": sha256(aggregate_path),
        "failure_labels": None,
        "composite_score": None,
    }
    # The production writer sorts object keys, so input-object key order is not
    # expected to match the explicit model list order.
    (output / "manifest.json").write_text(json.dumps(manifest, sort_keys=True) + "\n")
    return output, cross


def test_audit_accepts_recomputed_baseline_summary(tmp_path: Path) -> None:
    summary, cross = make_fixture(tmp_path)
    report = audit_baseline_summary(summary, cross, expected_scenes=2)
    assert report["passed"] is True
    assert report["scene_rows"] == 4


def test_audit_rejects_changed_aggregate_mean(tmp_path: Path) -> None:
    summary, cross = make_fixture(tmp_path)
    path = summary / "dataset_model_summary.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows[0]["mean"] = "999"
    write_csv(path, rows)
    manifest = json.loads((summary / "manifest.json").read_text())
    manifest["summary_output_sha256"] = sha256(path)
    (summary / "manifest.json").write_text(json.dumps(manifest) + "\n")
    with pytest.raises(ValueError, match="Aggregate mean mismatch"):
        audit_baseline_summary(summary, cross)
