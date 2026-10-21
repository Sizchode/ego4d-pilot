import json
from pathlib import Path

import pytest

from scripts.failure_mining.snapshot_baseline_batches import BatchInput, snapshot_batch


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def make_batch(root: Path) -> Path:
    batch = root / "batch"
    batch.mkdir()
    tasks = [
        {"dataset": "set", "scene": "one"},
        {"dataset": "set", "scene": "two"},
    ]
    write_json(batch / "manifest.json", tasks)
    records = []
    for index, task in enumerate(tasks):
        run = root / f"run-{index}"
        write_json(run / "run_config.json", task)
        write_json(run / "status.json", {"state": "completed"})
        write_json(run / "validation.json", {"passed": True, "views": index + 2, "metrics": {"pose": {"auc30": 0.9}}})
        records.append({**task, "run": str(run), "state": "completed", "exit_code": 0})
    (batch / "results.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in records), encoding="utf-8"
    )
    return batch


def test_snapshot_captures_exact_completed_prefix(tmp_path: Path) -> None:
    batch = make_batch(tmp_path)
    output = tmp_path / "snapshot"
    output.mkdir()
    metrics, evidence = snapshot_batch(BatchInput("model", batch), output)
    assert len(metrics) == 2
    assert evidence["results_rows"] == 2
    assert evidence["completed_validated_rows"] == 2
    assert evidence["by_dataset"] == {"set": 2}
    assert (output / "model.results.jsonl").read_bytes() == (batch / "results.jsonl").read_bytes()


def test_snapshot_rejects_nonprefix_results(tmp_path: Path) -> None:
    batch = make_batch(tmp_path)
    lines = (batch / "results.jsonl").read_text().splitlines()
    (batch / "results.jsonl").write_text("\n".join(reversed(lines)) + "\n")
    output = tmp_path / "snapshot"
    output.mkdir()
    with pytest.raises(ValueError, match="ordered manifest prefix"):
        snapshot_batch(BatchInput("model", batch), output)


def test_snapshot_rejects_partial_trailing_jsonl_record(tmp_path: Path) -> None:
    batch = make_batch(tmp_path)
    with (batch / "results.jsonl").open("ab") as handle:
        handle.write(b'{"dataset":"set"')
    output = tmp_path / "snapshot"
    output.mkdir()
    with pytest.raises(ValueError, match="partial JSONL record"):
        snapshot_batch(BatchInput("model", batch), output)
