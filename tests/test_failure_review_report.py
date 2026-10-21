import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from scripts.audit.analysis_outputs import sha256
from scripts.audit.failure_review_report import audit_report
from scripts.failure_mining.archive_case_bundles import build_archive
from scripts.failure_mining.build_failure_review_report import build_report
from scripts.failure_mining.compare_baselines import write_csv


def make_fixture(root: Path) -> tuple[Path, Path, Path]:
    models = ["a", "b"]
    rows = []
    for index, model in enumerate(models):
        run = root / f"run-{model}"
        (run / "diagnostics").mkdir(parents=True)
        (run / "evaluation_capture").mkdir(parents=True)
        (run / "status.json").write_text(json.dumps({"state": "completed"}))
        (run / "validation.json").write_text(json.dumps({"passed": True}))
        (run / "run_config.json").write_text(json.dumps({"dataset": "set", "scene": "one", "model": model}))
        (run / "input_manifest.json").write_text(json.dumps([{"index": 0}]))
        (run / "diagnostics" / "protocol.json").write_text(json.dumps({"pairs": [[0, 1]]}))
        Image.new("RGB", (32, 24), (index * 100, 40, 80)).save(run / "preview.png")
        Image.new("RGB", (30, 20), (20, index * 100, 60)).save(run / "camera_errors.png")
        np.savez(run / "evaluation_capture" / "camera_pairs.npz", index_a=np.array([0]), index_b=np.array([1]))
        rows.append({"model": model, "dataset": "set", "scene": "one", "run": str(run)})

    bundles = root / "bundles"
    bundles.mkdir()
    selected_runs = bundles / "selected_runs.jsonl"
    selected_runs.write_text("".join(json.dumps(row) + "\n" for row in rows))
    tar_path = bundles / "set-review-cases.tar"
    archive_manifest = build_archive(selected_runs, "set", ["one"], models, tar_path)
    selection = bundles / "selected_review_cases.csv"
    write_csv(selection, [{"model": "a", "dataset": "set", "scene": "one", "metric": "pose", "selection_reasons": "consensus_hardest"}])
    bundle_manifest = {
        "models": models,
        "test_time_policy_input": False,
        "bundles": [{"dataset": "set", "scenes": ["one"], "models": models, "tar": tar_path.name, "tar_sha256": sha256(tar_path)}],
    }
    bundle_manifest_path = bundles / "manifest.json"
    bundle_manifest_path.write_text(json.dumps(bundle_manifest) + "\n")

    views = root / "views"
    (views / "set" / "one").mkdir(parents=True)
    rendered = views / "set" / "one" / "view-000.jpg"
    Image.new("RGB", (28, 22), (100, 120, 140)).save(rendered)
    views_manifest = {
        "models": models,
        "test_time_policy_input": False,
        "scene_outputs": [{
            "dataset": "set",
            "scene": "one",
            "views": [{"index": 0, "output_path": "set/one/view-000.jpg", "output_sha256": sha256(rendered), "selection_reasons": [{"model": "a", "metric": "rotation_degrees", "extremum": "maximum"}]}],
        }],
    }
    views_manifest_path = views / "manifest.json"
    views_manifest_path.write_text(json.dumps(views_manifest) + "\n")
    views_audit = root / "views-audit.json"
    views_audit.write_text(json.dumps({
        "passed": True,
        "bundle_manifest_sha256": sha256(bundle_manifest_path),
        "views_manifest_sha256": sha256(views_manifest_path),
    }) + "\n")
    assert archive_manifest["case_count"] == 2
    return bundles, views, views_audit


def test_builds_and_audits_cross_model_review_report(tmp_path: Path) -> None:
    bundles, views, views_audit = make_fixture(tmp_path)
    output = tmp_path / "report"
    manifest = build_report(bundles, views, views_audit, output)
    assert manifest["selected_scenes"] == 1
    assert manifest["expanded_cross_model_cases"] == 2
    assert manifest["test_time_policy_input"] is False
    assert (output / "index.html").is_file()
    report = audit_report(bundles, views, views_audit, output)
    assert report["passed"] is True


def test_audit_rejects_mutated_report_asset(tmp_path: Path) -> None:
    bundles, views, views_audit = make_fixture(tmp_path)
    output = tmp_path / "report"
    manifest = build_report(bundles, views, views_audit, output)
    asset = next(path for path in manifest["outputs"] if path.endswith("preview.png"))
    (output / asset).write_bytes((output / asset).read_bytes() + b"mutated")
    with pytest.raises(ValueError, match="Report output hash differs"):
        audit_report(bundles, views, views_audit, output)
