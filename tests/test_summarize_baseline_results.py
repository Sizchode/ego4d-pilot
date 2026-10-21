import csv
import json
from pathlib import Path

from scripts.failure_mining.compare_baselines import sha256
from scripts.failure_mining.summarize_baseline_results import summarize
from tests.test_cross_model_output_audit import make_cross_outputs, write_json


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_summary_preserves_scene_population_and_separates_fields(tmp_path: Path) -> None:
    cross_dir, _visibility, _batches = make_cross_outputs(tmp_path)
    manifest_path = cross_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for model_index, model in enumerate(manifest["models"]):
        scenes_path = Path(manifest["inputs"][model]["analysis_dir"]) / "scenes.csv"
        with scenes_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        for scene_index, row in enumerate(rows):
            row["run"] = f"/{model}/run-{scene_index}"
            row["views"] = str(scene_index + 2)
            for runtime_index, field in enumerate(
                (
                    "elapsed_seconds",
                    "inference_seconds",
                    "evaluation_seconds",
                    "diagnostics_seconds",
                    "peak_cuda_allocated_bytes",
                    "peak_cuda_reserved_bytes",
                )
            ):
                row[field] = str(10 + model_index + scene_index + runtime_index)
            row["metric.pose.auc30"] = str(0.5 + 0.1 * scene_index + 0.01 * model_index)
        write_csv(scenes_path, rows)
        manifest["inputs"][model]["scenes_sha256"] = sha256(scenes_path)
    write_json(manifest_path, manifest)

    scenes, summaries, provenance = summarize(cross_dir)
    assert len(scenes) == 4
    assert provenance["scene_population_per_model"] == 2
    pose = [row for row in summaries if row["field"] == "metric.pose.auc30"]
    runtime = [row for row in summaries if row["field"] == "elapsed_seconds"]
    assert len(pose) == len(runtime) == 2
    assert all(row["count"] == 2 for row in pose + runtime)
    assert all(row["population_stddev"] >= 0 for row in pose + runtime)
    assert all("composite" not in row["field"] for row in summaries)


def test_summary_rejects_partial_metric_population(tmp_path: Path) -> None:
    cross_dir, _visibility, _batches = make_cross_outputs(tmp_path)
    manifest_path = cross_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for model in manifest["models"]:
        scenes_path = Path(manifest["inputs"][model]["analysis_dir"]) / "scenes.csv"
        with scenes_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        for index, row in enumerate(rows):
            row["run"] = f"/{model}/run-{index}"
            row["views"] = "2"
            for field in (
                "elapsed_seconds",
                "inference_seconds",
                "evaluation_seconds",
                "diagnostics_seconds",
                "peak_cuda_allocated_bytes",
                "peak_cuda_reserved_bytes",
            ):
                row[field] = "1"
            row["metric.pose.auc30"] = "0.5" if index == 0 else ""
        write_csv(scenes_path, rows)
        manifest["inputs"][model]["scenes_sha256"] = sha256(scenes_path)
    write_json(manifest_path, manifest)
    import pytest

    with pytest.raises(ValueError, match="Partial field population"):
        summarize(cross_dir)
