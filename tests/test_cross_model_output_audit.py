import csv
import json
from pathlib import Path

import pytest

from scripts.audit.analysis_outputs import sha256
from scripts.audit.cross_model_outputs import audit_cross_model_outputs
from tests.test_analysis_output_audit import make_analyses


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def write_rows(path: Path, count: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["row"])
        writer.writeheader()
        writer.writerows({"row": index} for index in range(count))


def make_cross_outputs(root: Path) -> tuple[Path, Path, Path]:
    models = ["model-a", "model-b"]
    standard_inputs, visibility_inputs = {}, {}
    batch_audits = root / "batch-audits"
    for index, model in enumerate(models):
        standard, visibility = make_analyses(root / model)
        # Multiple model runs may intentionally share one frozen scene manifest.
        # The results hash, together with that manifest hash, identifies a batch.
        manifest_hash = "1" * 64
        results_hash = str(index + 3) * 64
        for directory in (standard, visibility):
            path = directory / "manifest.json"
            data = json.loads(path.read_text())
            data["batch_manifest_sha256"] = manifest_hash
            data["batch_results_sha256"] = results_hash
            write_json(path, data)
        standard_inputs[model] = {
            "analysis_dir": str(standard.resolve()),
            "analysis_manifest_sha256": sha256(standard / "manifest.json"),
            "scenes_sha256": sha256(standard / "scenes.csv"),
            "rankings_sha256": sha256(standard / "rankings.csv"),
        }
        visibility_inputs[model] = {
            "analysis_dir": str(visibility.resolve()),
            "manifest_sha256": sha256(visibility / "manifest.json"),
            "scene_visibility_sha256": sha256(visibility / "scene_visibility.csv"),
        }
        write_json(
            batch_audits / f"{model}.json",
            {
                "passed": True,
                "expected_scenes": 2,
                "batch_file_sha256": {
                    "manifest.json": manifest_hash,
                    "results.jsonl": results_hash,
                },
            },
        )

    standard_cross, visibility_cross = root / "standard-cross", root / "visibility-cross"
    standard_counts = {
        "execution_rows": 4,
        "model_metric_rank_rows": 4,
        "scene_metric_spread_rows": 2,
        "pairwise_rank_agreement_rows": 1,
    }
    standard_files = {
        "execution_rows": "execution_matrix.csv",
        "model_metric_rank_rows": "model_metric_rank_rows.csv",
        "scene_metric_spread_rows": "scene_metric_spread.csv",
        "pairwise_rank_agreement_rows": "pairwise_rank_agreement.csv",
    }
    for key, filename in standard_files.items():
        write_rows(standard_cross / filename, standard_counts[key])
    write_json(
        standard_cross / "manifest.json",
        {
            "models": models,
            "scene_population": 2,
            "population_order": [
                {"dataset": "set", "scene": "one"},
                {"dataset": "set", "scene": "two"},
            ],
            "inputs": standard_inputs,
            "outputs": standard_counts,
        },
    )

    visibility_counts = {
        "model_scene_rows": 4,
        "rank_rows": 4,
        "pairwise_rows": 1,
        "scene_spread_rows": 2,
    }
    visibility_files = {
        "model_scene_rows": "model_scene_visibility.csv",
        "rank_rows": "model_visibility_ranks.csv",
        "pairwise_rows": "pairwise_visibility_rank_agreement.csv",
        "scene_spread_rows": "scene_visibility_rank_spread.csv",
    }
    for key, filename in visibility_files.items():
        write_rows(visibility_cross / filename, visibility_counts[key])
    write_json(
        visibility_cross / "manifest.json",
        {
            "models": models,
            "scene_population": 2,
            "inputs": visibility_inputs,
            "outputs": visibility_counts,
        },
    )
    return standard_cross, visibility_cross, batch_audits


def test_audit_accepts_complete_cross_model_outputs(tmp_path: Path) -> None:
    standard, visibility, batches = make_cross_outputs(tmp_path)
    report = audit_cross_model_outputs(standard, visibility, batches, expected_scenes=2)
    assert report["passed"] is True
    assert report["models"] == ["model-a", "model-b"]
    assert set(report["model_audits"]) == {"model-a", "model-b"}


def test_audit_rejects_cross_manifest_input_hash_change(tmp_path: Path) -> None:
    standard, visibility, batches = make_cross_outputs(tmp_path)
    path = standard / "manifest.json"
    data = json.loads(path.read_text())
    data["inputs"]["model-b"]["scenes_sha256"] = "0" * 64
    write_json(path, data)
    with pytest.raises(ValueError, match="scenes hash mismatch"):
        audit_cross_model_outputs(standard, visibility, batches, expected_scenes=2)


def test_audit_rejects_duplicate_manifest_and_results_provenance(tmp_path: Path) -> None:
    standard, visibility, batches = make_cross_outputs(tmp_path)
    original = json.loads((batches / "model-a.json").read_text())
    write_json(batches / "duplicate.json", original)
    with pytest.raises(ValueError, match="Duplicate batch audit provenance"):
        audit_cross_model_outputs(standard, visibility, batches, expected_scenes=2)
