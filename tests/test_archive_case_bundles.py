import json
import tarfile
from pathlib import Path

import pytest

from scripts.failure_mining.archive_case_bundles import REQUIRED_FILES, build_archive


def make_snapshot(root: Path) -> Path:
    rows = []
    for model in ("a", "b"):
        run = root / f"run-{model}"
        for relative in REQUIRED_FILES:
            path = run / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"data")
        (run / "status.json").write_text(json.dumps({"state": "completed"}))
        (run / "validation.json").write_text(json.dumps({"passed": True}))
        (run / "run_config.json").write_text(json.dumps({"dataset": "set", "scene": "one"}))
        rows.append({"model": model, "dataset": "set", "scene": "one", "run": str(run)})
    path = root / "completed_metrics.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def test_archive_contains_exact_required_evidence_and_manifest(tmp_path: Path) -> None:
    metrics = make_snapshot(tmp_path)
    output = tmp_path / "cases.tar"
    manifest = build_archive(metrics, "set", ["one"], ["a", "b"], output)
    assert manifest["case_count"] == 2
    assert manifest["file_count"] == 2 * len(REQUIRED_FILES)
    with tarfile.open(output) as archive:
        names = set(archive.getnames())
        assert "archive_manifest.json" in names
        assert all(f"a-one/{relative}" in names for relative in REQUIRED_FILES)
        archived = json.load(archive.extractfile("archive_manifest.json"))
    assert archived["metrics_jsonl_sha256"] == manifest["metrics_jsonl_sha256"]


def test_archive_rejects_missing_model_case(tmp_path: Path) -> None:
    metrics = make_snapshot(tmp_path)
    with pytest.raises(ValueError, match="does not contain selected case"):
        build_archive(metrics, "set", ["one"], ["a", "missing"], tmp_path / "cases.tar")
