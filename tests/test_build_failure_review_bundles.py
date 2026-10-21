import csv
import json
import tarfile
from pathlib import Path

from scripts.failure_mining.archive_case_bundles import REQUIRED_FILES
from scripts.failure_mining.build_failure_review_bundles import build_bundles
from scripts.failure_mining.compare_baselines import sha256, write_csv


def make_fixture(root: Path) -> tuple[Path, Path, Path]:
    models = ["a", "b"]
    selection = root / "selection"
    summary = root / "summary"
    selection.mkdir()
    summary.mkdir()
    selected = [
        {"model": "a", "dataset": "set", "scene": "one", "metric": "geometry", "selection_reasons": "consensus_hardest"},
    ]
    selected_path = selection / "selected_review_cases.csv"
    write_csv(selected_path, selected)
    selection_manifest = {
        "models": models,
        "cases_output_sha256": sha256(selected_path),
        "selection_threshold": None,
        "top_k": None,
    }
    (selection / "manifest.json").write_text(json.dumps(selection_manifest) + "\n")

    scene_rows = []
    for model in models:
        run = root / f"run-{model}"
        for relative in REQUIRED_FILES:
            path = run / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"data")
        (run / "status.json").write_text(json.dumps({"state": "completed"}))
        (run / "validation.json").write_text(json.dumps({"passed": True}))
        (run / "run_config.json").write_text(json.dumps({"dataset": "set", "scene": "one"}))
        scene_rows.append({"model": model, "dataset": "set", "scene": "one", "run": str(run), "state": "completed", "views": 2})
    scene_path = summary / "scene_model_results.csv"
    write_csv(scene_path, scene_rows)
    summary_manifest = {"models": models, "scene_output_sha256": sha256(scene_path)}
    summary_manifest_path = summary / "manifest.json"
    summary_manifest_path.write_text(json.dumps(summary_manifest) + "\n")
    audit_path = root / "summary-audit.json"
    audit_path.write_text(json.dumps({"passed": True, "summary_manifest_sha256": sha256(summary_manifest_path)}) + "\n")
    return selection, summary, audit_path


def test_builds_cross_model_bundle_from_selected_scene(tmp_path: Path) -> None:
    selection, summary, audit = make_fixture(tmp_path)
    output = tmp_path / "bundles"
    manifest = build_bundles(selection, summary, audit, output)
    assert manifest["selected_unique_scenes"] == 1
    assert manifest["expanded_cross_model_cases"] == 2
    assert manifest["test_time_policy_input"] is False
    bundle = manifest["bundles"][0]
    assert bundle["cases"] == 2
    assert sha256(output / bundle["tar"]) == bundle["tar_sha256"]
    with tarfile.open(output / bundle["tar"]) as archive:
        archived = json.load(archive.extractfile("archive_manifest.json"))
    assert archived["case_count"] == 2
