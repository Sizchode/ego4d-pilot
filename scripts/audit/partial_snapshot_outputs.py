"""Independently audit immutable partial snapshots and their comparisons."""

from __future__ import annotations

import argparse
import csv
import json
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from scripts.failure_mining.compare_baselines import sha256, write_csv
from scripts.failure_mining.compare_snapshot_metrics import compare


COMPLETED_STATES = frozenset({"completed", "reused_verified_run"})
OUTPUT_NAMES = (
    "common_populations",
    "model_metric_rank_rows",
    "pairwise_rank_agreement",
    "scene_metric_spread",
    "oracle_scene_headroom",
    "oracle_summary",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_json(path: Path) -> Any:
    require(path.is_file(), f"Required file is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    require(path.is_file(), f"Required file is missing: {path}")
    payload = path.read_bytes()
    require(not payload or payload.endswith(b"\n"), f"Partial JSONL record: {path}")
    return [json.loads(line) for line in payload.splitlines() if line.strip()]


def read_csv(path: Path) -> list[dict[str, str]]:
    require(path.is_file(), f"Required file is missing: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def verify_snapshot(snapshot_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    manifest_path = snapshot_dir / "manifest.json"
    metrics_path = snapshot_dir / "completed_metrics.jsonl"
    manifest = read_json(manifest_path)
    require(manifest.get("status") == "non-final immutable snapshot", "Unexpected snapshot status")
    models = manifest.get("models")
    require(
        isinstance(models, list) and models and len(models) == len(set(models)),
        "Invalid snapshot model population",
    )
    require(
        manifest.get("completed_metrics_sha256") == sha256(metrics_path),
        "Completed-metrics hash mismatch",
    )
    metrics = read_jsonl(metrics_path)
    require(
        manifest.get("completed_metric_rows") == len(metrics),
        "Completed-metrics row count mismatch",
    )
    sources = manifest.get("sources")
    require(
        isinstance(sources, list) and [row.get("model") for row in sources] == models,
        "Snapshot source/model order differs",
    )

    results_by_model: dict[str, list[dict[str, Any]]] = {}
    expected_metric_identities = []
    source_checks = []
    for source in sources:
        model = source["model"]
        copied_path = snapshot_dir / source["copied_results"]
        observed_hash = sha256(copied_path)
        require(source.get("copied_results_sha256") == observed_hash, f"Copied-result hash mismatch: {model}")
        require(source.get("results_sha256") == observed_hash, f"Source/copied hash mismatch: {model}")
        rows = read_jsonl(copied_path)
        require(source.get("results_rows") == len(rows), f"Result row count mismatch: {model}")
        identities = [(row.get("dataset"), row.get("scene")) for row in rows]
        require(len(identities) == len(set(identities)), f"Duplicate result identity: {model}")
        completed = [row for row in rows if row.get("state") in COMPLETED_STATES]
        failures = [row for row in rows if row.get("state") not in COMPLETED_STATES]
        require(
            source.get("completed_validated_rows") == len(completed),
            f"Completed result count mismatch: {model}",
        )
        require(source.get("failure_rows") == len(failures), f"Failure count mismatch: {model}")
        by_dataset = dict(sorted(Counter(row["dataset"] for row in completed).items()))
        require(source.get("by_dataset") == by_dataset, f"Dataset counts mismatch: {model}")
        expected_last = (
            {"dataset": rows[-1]["dataset"], "scene": rows[-1]["scene"]}
            if rows
            else None
        )
        require(source.get("last_identity") == expected_last, f"Last identity mismatch: {model}")
        expected_metric_identities.extend(
            (model, row["dataset"], row["scene"], row["state"]) for row in completed
        )
        results_by_model[model] = rows
        source_checks.append(
            {
                "model": model,
                "rows": len(rows),
                "completed": len(completed),
                "failures": len(failures),
                "sha256": observed_hash,
            }
        )

    metric_identities = [
        (row.get("model"), row.get("dataset"), row.get("scene"), row.get("state"))
        for row in metrics
    ]
    require(
        metric_identities == expected_metric_identities,
        "Completed-metric population/order differs from copied result streams",
    )
    return manifest, metrics, results_by_model


def verify_comparison(
    comparison_dir: Path,
    snapshot_dir: Path,
    snapshot_manifest: dict[str, Any],
    metrics: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, list[dict[str, str]]]]:
    manifest_path = comparison_dir / "manifest.json"
    manifest = read_json(manifest_path)
    require(
        manifest.get("status") == "non-final common-population comparison",
        "Unexpected comparison status",
    )
    require(
        manifest.get("snapshot_manifest_sha256") == sha256(snapshot_dir / "manifest.json"),
        "Comparison snapshot-manifest hash mismatch",
    )
    require(
        manifest.get("snapshot_metrics_sha256") == snapshot_manifest.get("completed_metrics_sha256"),
        "Comparison snapshot-metrics hash mismatch",
    )
    require(manifest.get("models") == snapshot_manifest.get("models"), "Comparison model order differs")
    require(manifest.get("failure_labels") is None, "Comparison assigns failure labels")
    require(manifest.get("composite_score") is None, "Comparison uses a composite score")
    require(set(manifest.get("outputs", {})) == set(OUTPUT_NAMES), "Unexpected comparison outputs")

    recomputed = compare(metrics, snapshot_manifest["models"])
    rows_by_output: dict[str, list[dict[str, str]]] = {}
    checks = []
    with tempfile.TemporaryDirectory() as directory:
        temporary = Path(directory)
        for name in OUTPUT_NAMES:
            path = comparison_dir / f"{name}.csv"
            rows = read_csv(path)
            meta = manifest["outputs"][name]
            observed_hash = sha256(path)
            require(meta.get("rows") == len(rows), f"Comparison row count mismatch: {name}")
            require(meta.get("sha256") == observed_hash, f"Comparison hash mismatch: {name}")
            expected_path = temporary / f"{name}.csv"
            write_csv(expected_path, recomputed[name])
            require(
                expected_path.read_bytes() == path.read_bytes(),
                f"Comparison bytes differ from independent recomputation: {name}",
            )
            rows_by_output[name] = rows
            checks.append({"name": name, "rows": len(rows), "sha256": observed_hash})
    return manifest, rows_by_output


def audit_stability(
    current_snapshot: dict[str, list[dict[str, Any]]],
    current_outputs: dict[str, list[dict[str, str]]],
    previous_snapshot_dir: Path,
    previous_comparison_dir: Path,
) -> dict[str, Any]:
    previous_manifest, previous_metrics, previous_results = verify_snapshot(previous_snapshot_dir)
    _previous_cross, previous_outputs = verify_comparison(
        previous_comparison_dir,
        previous_snapshot_dir,
        previous_manifest,
        previous_metrics,
    )
    require(set(previous_results) == set(current_snapshot), "Previous/current model populations differ")
    prefix_checks = {}
    for model, old_rows in previous_results.items():
        new_rows = current_snapshot[model]
        require(new_rows[: len(old_rows)] == old_rows, f"Result prefix changed: {model}")
        prefix_checks[model] = {"previous_rows": len(old_rows), "current_rows": len(new_rows)}

    def populations(rows: list[dict[str, str]]) -> dict[str, int]:
        return {row["dataset"]: int(row["common_scenes"]) for row in rows}

    old_population = populations(previous_outputs["common_populations"])
    new_population = populations(current_outputs["common_populations"])
    unchanged = sorted(
        dataset
        for dataset, count in old_population.items()
        if new_population.get(dataset) == count
    )
    stability = {}
    for dataset in unchanged:
        stability[dataset] = {}
        for name in OUTPUT_NAMES:
            if name == "common_populations":
                continue
            old_rows = [row for row in previous_outputs[name] if row.get("dataset") == dataset]
            new_rows = [row for row in current_outputs[name] if row.get("dataset") == dataset]
            require(old_rows == new_rows, f"Unchanged-dataset output changed: {(dataset, name)}")
            stability[dataset][name] = True
    return {
        "previous_snapshot_manifest_sha256": sha256(previous_snapshot_dir / "manifest.json"),
        "result_prefix_checks": prefix_checks,
        "previous_common_populations": old_population,
        "current_common_populations": new_population,
        "unchanged_dataset_outputs": stability,
    }


def audit(
    snapshot_dir: Path,
    comparison_dir: Path,
    *,
    previous_snapshot_dir: Path | None = None,
    previous_comparison_dir: Path | None = None,
) -> dict[str, Any]:
    snapshot_dir = snapshot_dir.resolve()
    comparison_dir = comparison_dir.resolve()
    manifest, metrics, results_by_model = verify_snapshot(snapshot_dir)
    comparison, outputs = verify_comparison(comparison_dir, snapshot_dir, manifest, metrics)
    report: dict[str, Any] = {
        "schema_version": 1,
        "passed": True,
        "snapshot_manifest_sha256": sha256(snapshot_dir / "manifest.json"),
        "snapshot_metrics_sha256": sha256(snapshot_dir / "completed_metrics.jsonl"),
        "snapshot_rows": len(metrics),
        "models": manifest["models"],
        "source_rows": {model: len(rows) for model, rows in results_by_model.items()},
        "comparison_manifest_sha256": sha256(comparison_dir / "manifest.json"),
        "comparison_outputs": {
            name: {"rows": len(rows), "sha256": sha256(comparison_dir / f"{name}.csv")}
            for name, rows in outputs.items()
        },
        "common_populations": {
            row["dataset"]: int(row["common_scenes"])
            for row in outputs["common_populations"]
        },
    }
    require(
        (previous_snapshot_dir is None) == (previous_comparison_dir is None),
        "Previous snapshot and comparison must be provided together",
    )
    if previous_snapshot_dir is not None and previous_comparison_dir is not None:
        report["stability_against_previous"] = audit_stability(
            results_by_model,
            outputs,
            previous_snapshot_dir.resolve(),
            previous_comparison_dir.resolve(),
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--previous-snapshot", type=Path)
    parser.add_argument("--previous-comparison", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(
        args.snapshot,
        args.comparison,
        previous_snapshot_dir=args.previous_snapshot,
        previous_comparison_dir=args.previous_comparison,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    require(not args.output.exists(), f"Refusing to overwrite audit: {args.output}")
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
