import hashlib
import json
from pathlib import Path

import pytest

from scripts.audit.batch_outputs import audit_batch


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def make_batch(root: Path) -> Path:
    batch = root / "batch"
    batch.mkdir()
    config = {"model_revision": "pinned-revision", "process_res": 504}
    tasks = [
        {"dataset": "set-a", "scene": "scene-1"},
        {"dataset": "set-b", "scene": "scene-2"},
    ]
    write_json(batch / "manifest.json", tasks)
    write_json(batch / "dataset_manifests.json", {"set-a": ["scene-1"], "set-b": ["scene-2"]})
    write_json(batch / "config.json", config)
    write_json(batch / "runner.json", {"batch_runner_sha256": "a" * 64})
    results = []
    for index, task in enumerate(tasks):
        run = root / f"run-{index}"
        run.mkdir()
        image = root / f"input-{index}.jpg"
        image.write_bytes(f"image-{index}".encode())
        write_json(run / "run_config.json", {**task, "config": config})
        write_json(run / "status.json", {**task, "state": "completed"})
        write_json(run / "validation.json", {"passed": True, "views": 1})
        write_json(
            run / "input_manifest.json",
            [
                {
                    "index": 0,
                    "path": str(image),
                    "sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
                }
            ],
        )
        results.append(
            {
                **task,
                "run": str(run),
                "state": "reused_verified_run" if index == 0 else "completed",
                **({} if index == 0 else {"exit_code": 0}),
            }
        )
    (batch / "results.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in results),
        encoding="utf-8",
    )
    write_json(batch / "summary.json", {"total": 2, "failed": 0, "finished": True})
    return batch


def test_audit_accepts_complete_exact_batch(tmp_path: Path) -> None:
    report = audit_batch(make_batch(tmp_path))
    assert report["passed"] is True
    assert report["expected_scenes"] == 2
    assert report["result_records"] == 2
    assert report["completed_or_reused"] == 2
    assert report["input_images"] == 2
    assert report["datasets"] == {"set-a": 1, "set-b": 1}
    assert all(item["inputs"][0]["observed_sha256"] for item in report["runs"])


def test_audit_rejects_result_population_reordering(tmp_path: Path) -> None:
    batch = make_batch(tmp_path)
    path = batch / "results.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(reversed(lines)) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="ordered batch manifest"):
        audit_batch(batch)


def test_audit_rejects_changed_input(tmp_path: Path) -> None:
    batch = make_batch(tmp_path)
    (tmp_path / "input-1.jpg").write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        audit_batch(batch)


def test_audit_rejects_failed_validation(tmp_path: Path) -> None:
    batch = make_batch(tmp_path)
    write_json(tmp_path / "run-1" / "validation.json", {"passed": False, "views": 1})
    with pytest.raises(ValueError, match="validation did not pass"):
        audit_batch(batch)
