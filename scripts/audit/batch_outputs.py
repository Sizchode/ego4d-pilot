"""Audit a finished multi-baseline benchmark batch from its recorded evidence.

The batch manifest is the source of truth for the population.  This audit does
not infer success from a scheduler exit code: it checks the ordered result
population, every run's identity and validation state, the exact input list,
and (by default) every recorded input SHA-256.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


COMPLETED_STATES = frozenset({"completed", "reused_verified_run"})
RUN_EVIDENCE_FILES = (
    "run_config.json",
    "status.json",
    "validation.json",
    "input_manifest.json",
)


def load_json(path: Path) -> Any:
    if not path.is_file():
        raise ValueError(f"Required file is missing: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in {path}: {error}") from error


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ValueError(f"Required file is missing: {path}")
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSON on {path}:{line_number}: {error}") from error
        if not isinstance(record, dict):
            raise ValueError(f"Expected an object on {path}:{line_number}")
        records.append(record)
    return records


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def identity(record: dict[str, Any], source: str) -> tuple[str, str]:
    dataset = record.get("dataset")
    scene = record.get("scene")
    require(isinstance(dataset, str) and dataset, f"Missing dataset identity in {source}")
    require(isinstance(scene, str) and scene, f"Missing scene identity in {source}")
    return dataset, scene


def audit_input_manifest(
    path: Path,
    expected_views: int,
    verify_input_hashes: bool,
) -> tuple[int, list[dict[str, Any]]]:
    entries = load_json(path)
    require(isinstance(entries, list) and entries, f"Input manifest is empty or invalid: {path}")
    require(len(entries) == expected_views, f"Input count does not match validation views: {path}")
    require(
        [entry.get("index") for entry in entries] == list(range(len(entries))),
        f"Input indices are not the exact ordered range: {path}",
    )
    paths = [entry.get("path") for entry in entries]
    require(all(isinstance(item, str) and item for item in paths), f"Invalid input path: {path}")
    require(len(paths) == len(set(paths)), f"Duplicate input path: {path}")

    evidence = []
    for entry in entries:
        expected_hash = entry.get("sha256")
        require(
            isinstance(expected_hash, str)
            and len(expected_hash) == 64
            and all(char in "0123456789abcdef" for char in expected_hash),
            f"Invalid SHA-256 in {path} at index {entry.get('index')}",
        )
        input_path = Path(entry["path"])
        observed_hash = None
        if verify_input_hashes:
            require(input_path.is_file(), f"Recorded input is missing: {input_path}")
            observed_hash = sha256(input_path)
            require(
                observed_hash == expected_hash,
                f"Recorded input hash mismatch: {input_path}",
            )
        evidence.append(
            {
                "index": entry["index"],
                "path": str(input_path),
                "expected_sha256": expected_hash,
                "observed_sha256": observed_hash,
            }
        )
    return len(entries), evidence


def audit_batch(batch_dir: Path, verify_input_hashes: bool = True) -> dict[str, Any]:
    batch_dir = batch_dir.resolve()
    manifest_path = batch_dir / "manifest.json"
    results_path = batch_dir / "results.jsonl"
    dataset_manifests_path = batch_dir / "dataset_manifests.json"
    config_path = batch_dir / "config.json"
    runner_path = batch_dir / "runner.json"
    summary_path = batch_dir / "summary.json"

    manifest = load_json(manifest_path)
    require(isinstance(manifest, list) and manifest, "Batch manifest must be a non-empty list")
    expected = [identity(item, f"{manifest_path}[{index}]") for index, item in enumerate(manifest)]
    require(len(expected) == len(set(expected)), "Batch manifest contains duplicate dataset/scene identities")

    dataset_manifests = load_json(dataset_manifests_path)
    require(isinstance(dataset_manifests, dict) and dataset_manifests, "Invalid dataset manifests")
    reconstructed = []
    for dataset, scenes in dataset_manifests.items():
        require(isinstance(dataset, str) and dataset, "Invalid dataset name in dataset manifests")
        require(isinstance(scenes, list) and scenes, f"Empty scene manifest for {dataset}")
        require(len(scenes) == len(set(scenes)), f"Duplicate scenes in manifest for {dataset}")
        reconstructed.extend((dataset, scene) for scene in scenes)
    require(reconstructed == expected, "Dataset manifests do not exactly reconstruct the ordered batch manifest")

    config = load_json(config_path)
    require(isinstance(config, dict) and config, "Batch config must be a non-empty object")
    runner = load_json(runner_path)
    require(isinstance(runner, dict) and runner, "Runner evidence must be a non-empty object")

    results = load_jsonl(results_path)
    observed = [identity(item, f"{results_path}[{index}]") for index, item in enumerate(results)]
    require(observed == expected, "Results do not exactly match the ordered batch manifest")

    summary = load_json(summary_path)
    require(summary.get("finished") is True, "Batch summary is not marked finished")
    require(summary.get("total") == len(expected), "Batch summary total does not match the manifest")
    require(summary.get("failed") == 0, "Batch summary records execution failures")

    run_evidence = []
    total_inputs = 0
    for index, record in enumerate(results):
        record_identity = expected[index]
        state = record.get("state")
        require(state in COMPLETED_STATES, f"Run is not complete for {record_identity}: {state}")
        if "exit_code" in record:
            require(record["exit_code"] == 0, f"Nonzero exit code for {record_identity}")
        run_value = record.get("run")
        require(isinstance(run_value, str) and run_value, f"Missing run directory for {record_identity}")
        run = Path(run_value)
        require(run.is_dir(), f"Run directory is missing for {record_identity}: {run}")

        evidence_hashes = {}
        for relative in RUN_EVIDENCE_FILES:
            evidence_path = run / relative
            require(evidence_path.is_file(), f"Missing run evidence for {record_identity}: {evidence_path}")
            evidence_hashes[relative] = sha256(evidence_path)

        run_config = load_json(run / "run_config.json")
        status = load_json(run / "status.json")
        validation = load_json(run / "validation.json")
        require(identity(run_config, str(run / "run_config.json")) == record_identity, f"Run-config identity mismatch for {record_identity}")
        require(run_config.get("config") == config, f"Run config differs from batch config for {record_identity}")
        require(status.get("state") == "completed", f"Run status is not completed for {record_identity}")
        if "dataset" in status or "scene" in status:
            require(identity(status, str(run / "status.json")) == record_identity, f"Status identity mismatch for {record_identity}")
        require(validation.get("passed") is True, f"Run validation did not pass for {record_identity}")
        views = validation.get("views")
        require(isinstance(views, int) and views > 0, f"Invalid validation view count for {record_identity}")

        input_count, inputs = audit_input_manifest(
            run / "input_manifest.json",
            expected_views=views,
            verify_input_hashes=verify_input_hashes,
        )
        total_inputs += input_count
        run_evidence.append(
            {
                "batch_index": index,
                "dataset": record_identity[0],
                "scene": record_identity[1],
                "state": state,
                "run": str(run),
                "views": views,
                "evidence_sha256": evidence_hashes,
                "inputs": inputs,
            }
        )

    return {
        "schema_version": 1,
        "batch_dir": str(batch_dir),
        "passed": True,
        "verify_input_hashes": verify_input_hashes,
        "expected_scenes": len(expected),
        "result_records": len(results),
        "completed_or_reused": len(results),
        "execution_failures": 0,
        "datasets": {dataset: len(scenes) for dataset, scenes in dataset_manifests.items()},
        "input_images": total_inputs,
        "batch_file_sha256": {
            path.name: sha256(path)
            for path in (
                manifest_path,
                results_path,
                dataset_manifests_path,
                config_path,
                runner_path,
                summary_path,
            )
        },
        "runs": run_evidence,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--skip-input-file-hashes",
        action="store_true",
        help="Validate recorded hashes syntactically without rereading the source images.",
    )
    args = parser.parse_args()
    report = audit_batch(args.batch_dir, verify_input_hashes=not args.skip_input_file_hashes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    require(not args.output.exists(), f"Refusing to overwrite existing audit: {args.output}")
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "passed": report["passed"],
                "expected_scenes": report["expected_scenes"],
                "result_records": report["result_records"],
                "input_images": report["input_images"],
                "execution_failures": report["execution_failures"],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
