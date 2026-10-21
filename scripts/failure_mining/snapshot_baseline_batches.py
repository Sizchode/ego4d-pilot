"""Create an immutable, prefix-consistent snapshot of active baseline batches."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


COMPLETED_STATES = frozenset({"completed", "reused_verified_run"})


@dataclass(frozen=True)
class BatchInput:
    model: str
    directory: Path


def parse_batch(value: str) -> BatchInput:
    model, separator, directory = value.partition("=")
    if not separator or not model or not directory:
        raise argparse.ArgumentTypeError("batch must be MODEL=BATCH_DIRECTORY")
    return BatchInput(model, Path(directory))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def snapshot_batch(item: BatchInput, output: Path) -> tuple[list[dict], dict]:
    manifest_path = item.directory / "manifest.json"
    results_path = item.directory / "results.jsonl"
    manifest_bytes = manifest_path.read_bytes()
    results_bytes = results_path.read_bytes()
    if results_bytes and not results_bytes.endswith(b"\n"):
        raise ValueError(f"{item.model}: results snapshot ends with a partial JSONL record")
    manifest = json.loads(manifest_bytes)
    records = [json.loads(line) for line in results_bytes.splitlines() if line.strip()]
    expected = [(row["dataset"], row["scene"]) for row in manifest]
    observed = [(row["dataset"], row["scene"]) for row in records]
    if observed != expected[: len(observed)]:
        raise ValueError(f"{item.model}: results are not an ordered manifest prefix")
    if len(expected) != len(set(expected)):
        raise ValueError(f"{item.model}: duplicate identities in batch manifest")

    copied_results = output / f"{item.model}.results.jsonl"
    copied_results.write_bytes(results_bytes)
    metrics = []
    failures = 0
    for record in records:
        state = record.get("state")
        if state not in COMPLETED_STATES:
            failures += 1
            continue
        run = Path(record["run"])
        status = load_json(run / "status.json")
        validation = load_json(run / "validation.json")
        config = load_json(run / "run_config.json")
        identity = (record["dataset"], record["scene"])
        if (config.get("dataset"), config.get("scene")) != identity:
            raise ValueError(f"{item.model}: run identity mismatch for {identity}")
        if status.get("state") != "completed" or validation.get("passed") is not True:
            raise ValueError(f"{item.model}: completed record lacks completed validation for {identity}")
        metrics.append(
            {
                "model": item.model,
                "dataset": identity[0],
                "scene": identity[1],
                "run": str(run),
                "state": state,
                "views": validation.get("views"),
                "metrics": validation.get("metrics", {}),
            }
        )
    by_dataset = dict(sorted(Counter(row["dataset"] for row in metrics).items()))
    evidence = {
        "model": item.model,
        "batch_dir": str(item.directory.resolve()),
        "manifest_rows": len(manifest),
        "manifest_sha256": sha256_bytes(manifest_bytes),
        "results_rows": len(records),
        "results_sha256": sha256_bytes(results_bytes),
        "completed_validated_rows": len(metrics),
        "failure_rows": failures,
        "by_dataset": by_dataset,
        "last_identity": (
            {"dataset": records[-1]["dataset"], "scene": records[-1]["scene"]}
            if records
            else None
        ),
        "copied_results": copied_results.name,
        "copied_results_sha256": sha256_bytes(copied_results.read_bytes()),
    }
    return metrics, evidence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", action="append", type=parse_batch, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    names = [item.model for item in args.batch]
    if len(names) != len(set(names)):
        parser.error("model labels must be unique")
    args.output.mkdir(parents=True, exist_ok=False)
    all_metrics = []
    sources = []
    for item in args.batch:
        metrics, evidence = snapshot_batch(item, args.output)
        all_metrics.extend(metrics)
        sources.append(evidence)
    metrics_path = args.output / "completed_metrics.jsonl"
    metrics_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in all_metrics),
        encoding="utf-8",
    )
    result = {
        "status": "non-final immutable snapshot",
        "models": names,
        "sources": sources,
        "completed_metric_rows": len(all_metrics),
        "completed_metrics_sha256": sha256_bytes(metrics_path.read_bytes()),
        "analysis_script_sha256": sha256_bytes(Path(__file__).read_bytes()),
    }
    (args.output / "manifest.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
