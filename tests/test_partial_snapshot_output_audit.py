import json
from pathlib import Path

import pytest

from scripts.audit.partial_snapshot_outputs import audit
from scripts.failure_mining.compare_baselines import sha256, write_csv
from scripts.failure_mining.compare_snapshot_metrics import compare


MODELS = ["a", "b"]


def make_snapshot(root: Path, counts: dict[str, int]) -> tuple[Path, Path]:
    snapshot = root / "snapshot"
    comparison = root / "comparison"
    snapshot.mkdir(parents=True)
    sources = []
    metrics = []
    scenes = ("one", "two", "three")
    for model in MODELS:
        records = []
        for index, scene in enumerate(scenes[: counts[model]]):
            record = {
                "dataset": "set",
                "scene": scene,
                "run": f"/{model}/{scene}",
                "state": "completed",
            }
            records.append(record)
            metrics.append(
                {
                    "model": model,
                    "dataset": "set",
                    "scene": scene,
                    "run": record["run"],
                    "state": "completed",
                    "views": 2,
                    "metrics": {"pose": {"auc30": 0.1 + index + (0.01 if model == "b" else 0)}},
                }
            )
        result_path = snapshot / f"{model}.results.jsonl"
        result_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in records))
        result_hash = sha256(result_path)
        sources.append(
            {
                "model": model,
                "batch_dir": f"/{model}",
                "manifest_rows": 3,
                "manifest_sha256": "0" * 64,
                "results_rows": len(records),
                "results_sha256": result_hash,
                "completed_validated_rows": len(records),
                "failure_rows": 0,
                "by_dataset": {"set": len(records)},
                "last_identity": {"dataset": "set", "scene": records[-1]["scene"]},
                "copied_results": result_path.name,
                "copied_results_sha256": result_hash,
            }
        )
    metrics_path = snapshot / "completed_metrics.jsonl"
    metrics_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in metrics))
    snapshot_manifest = {
        "status": "non-final immutable snapshot",
        "models": MODELS,
        "sources": sources,
        "completed_metric_rows": len(metrics),
        "completed_metrics_sha256": sha256(metrics_path),
    }
    (snapshot / "manifest.json").write_text(json.dumps(snapshot_manifest) + "\n")

    outputs = compare(metrics, MODELS)
    comparison.mkdir()
    output_meta = {}
    for name, rows in outputs.items():
        path = comparison / f"{name}.csv"
        write_csv(path, rows)
        output_meta[name] = {"rows": len(rows), "sha256": sha256(path)}
    comparison_manifest = {
        "status": "non-final common-population comparison",
        "snapshot_manifest_sha256": sha256(snapshot / "manifest.json"),
        "snapshot_metrics_sha256": sha256(metrics_path),
        "models": MODELS,
        "outputs": output_meta,
        "failure_labels": None,
        "composite_score": None,
    }
    (comparison / "manifest.json").write_text(json.dumps(comparison_manifest) + "\n")
    return snapshot, comparison


def test_audits_recomputation_and_prefix_stability(tmp_path: Path) -> None:
    old_snapshot, old_comparison = make_snapshot(
        tmp_path / "old", {"a": 2, "b": 2}
    )
    new_snapshot, new_comparison = make_snapshot(
        tmp_path / "new", {"a": 3, "b": 2}
    )
    report = audit(
        new_snapshot,
        new_comparison,
        previous_snapshot_dir=old_snapshot,
        previous_comparison_dir=old_comparison,
    )
    assert report["passed"] is True
    assert report["source_rows"] == {"a": 3, "b": 2}
    assert report["common_populations"] == {"set": 2}
    assert report["stability_against_previous"]["result_prefix_checks"]["a"] == {
        "previous_rows": 2,
        "current_rows": 3,
    }


def test_rejects_semantically_changed_comparison_with_updated_hash(tmp_path: Path) -> None:
    snapshot, comparison = make_snapshot(tmp_path, {"a": 2, "b": 2})
    path = comparison / "model_metric_rank_rows.csv"
    payload = path.read_text().replace("0.1", "0.9", 1)
    path.write_text(payload)
    manifest_path = comparison / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["outputs"]["model_metric_rank_rows"]["sha256"] = sha256(path)
    manifest_path.write_text(json.dumps(manifest) + "\n")
    with pytest.raises(ValueError, match="independent recomputation"):
        audit(snapshot, comparison)
