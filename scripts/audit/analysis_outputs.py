"""Audit paired standard and visibility failure-mining outputs.

This audit validates populations and provenance.  It does not assign failure
labels, choose thresholds, or combine metrics.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


COMPLETED_STATES = frozenset({"completed", "reused_verified_run"})


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_json(path: Path) -> Any:
    if not path.is_file():
        raise ValueError(f"Required file is missing: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in {path}: {error}") from error


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise ValueError(f"Required file is missing: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scene_keys(rows: list[dict[str, str]], source: str) -> list[tuple[str, str]]:
    keys = []
    for index, row in enumerate(rows):
        dataset, scene = row.get("dataset"), row.get("scene")
        require(dataset is not None and dataset != "", f"Missing dataset in {source} row {index}")
        require(scene is not None and scene != "", f"Missing scene in {source} row {index}")
        keys.append((dataset, scene))
    require(len(keys) == len(set(keys)), f"Duplicate scene identities in {source}")
    return keys


def validate_manifest_population(
    manifest: dict[str, Any], source: str, expected_scenes: int | None
) -> int:
    population = manifest.get("expected_scenes")
    require(isinstance(population, int) and population > 0, f"Invalid expected_scenes in {source}")
    if expected_scenes is not None:
        require(population == expected_scenes, f"Unexpected scene population in {source}")
    require(manifest.get("execution_failures") == 0, f"Execution failures reported by {source}")
    require(manifest.get("scene_rows") == population, f"Scene row count mismatch in {source}")
    return population


def validate_failure_file(path: Path) -> None:
    failures = load_json(path)
    require(isinstance(failures, list), f"Failure file is not a list: {path}")
    require(not failures, f"Failure file is not empty: {path}")


def validate_standard(
    directory: Path, expected_scenes: int | None
) -> tuple[dict[str, Any], list[tuple[str, str]], set[tuple[str, str, int, int]], dict[str, str]]:
    manifest_path = directory / "manifest.json"
    manifest = load_json(manifest_path)
    population = validate_manifest_population(manifest, str(manifest_path), expected_scenes)
    require(manifest.get("records") == population, "Standard analysis record count mismatch")
    require(manifest.get("completed_or_reused") == population, "Standard analysis completion count mismatch")
    validate_failure_file(directory / "failed_runs.json")

    scenes = read_csv(directory / "scenes.csv")
    require(len(scenes) == population, "Standard scenes.csv row count mismatch")
    keys = scene_keys(scenes, "standard scenes.csv")
    require([int(row["batch_index"]) for row in scenes] == list(range(population)), "Standard batch indices are not the exact ordered population")
    require(all(row.get("state") in COMPLETED_STATES for row in scenes), "Standard scenes contain a non-completed state")

    known = set(keys)
    pairs = read_csv(directory / "pair_diagnostics.csv")
    require(len(pairs) == manifest.get("pair_rows"), "Standard pair row count mismatch")
    pair_keys = {
        (row["dataset"], row["scene"], int(row["source_index"]), int(row["target_index"]))
        for row in pairs
    }
    require(len(pair_keys) == len(pairs), "Duplicate standard directed camera pairs")
    require(all(key[:2] in known for key in pair_keys), "Standard pairs reference an unknown scene")

    surfaces = read_csv(directory / "surface_diagnostics.csv")
    require(len(surfaces) == manifest.get("surface_rows"), "Standard surface row count mismatch")
    require(all((row["dataset"], row["scene"]) in known for row in surfaces), "Surface rows reference an unknown scene")

    rankings = read_csv(directory / "rankings.csv")
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    ranking_keys = set()
    for row in rankings:
        key = (row["dataset"], row["scene"], row["metric"])
        require(key not in ranking_keys, f"Duplicate standard ranking row: {key}")
        ranking_keys.add(key)
        require(key[:2] in known, f"Ranking references an unknown scene: {key[:2]}")
        grouped.setdefault((row["dataset"], row["metric"]), []).append(row)
    for group, rows in grouped.items():
        populations = {int(row["population"]) for row in rows}
        directions = {row["direction"] for row in rows}
        require(len(populations) == 1, f"Inconsistent ranking population: {group}")
        require(len(directions) == 1, f"Inconsistent ranking direction: {group}")
        group_population = next(iter(populations))
        ranks = sorted(int(row["failure_rank"]) for row in rows)
        require(ranks == list(range(1, group_population + 1)), f"Incomplete ranks: {group}")

    file_hashes = {
        name: sha256(directory / name)
        for name in (
            "manifest.json",
            "failed_runs.json",
            "scenes.csv",
            "pair_diagnostics.csv",
            "surface_diagnostics.csv",
            "rankings.csv",
        )
    }
    return manifest, keys, pair_keys, file_hashes


def validate_visibility(
    directory: Path,
    expected_scenes: int | None,
    standard_keys: list[tuple[str, str]],
    standard_pair_keys: set[tuple[str, str, int, int]],
) -> tuple[dict[str, Any], dict[str, str]]:
    manifest_path = directory / "manifest.json"
    manifest = load_json(manifest_path)
    population = validate_manifest_population(manifest, str(manifest_path), expected_scenes)
    validate_failure_file(directory / "failed_runs.json")

    scenes = read_csv(directory / "scene_visibility.csv")
    require(len(scenes) == population, "Visibility scene row count mismatch")
    keys = scene_keys(scenes, "visibility scene_visibility.csv")
    require(keys == standard_keys, "Standard and visibility scene population/order differ")
    require([int(row["batch_index"]) for row in scenes] == list(range(population)), "Visibility batch indices are not the exact ordered population")
    require(all(row.get("state") in COMPLETED_STATES for row in scenes), "Visibility scenes contain a non-completed state")

    pairs = read_csv(directory / "pair_visibility.csv")
    require(len(pairs) == manifest.get("pair_rows"), "Visibility pair row count mismatch")
    pair_keys = {
        (row["dataset"], row["scene"], int(row["source_index"]), int(row["target_index"]))
        for row in pairs
    }
    require(len(pair_keys) == len(pairs), "Duplicate visibility directed camera pairs")
    require(pair_keys == standard_pair_keys, "Standard and visibility directed-pair populations differ")

    file_hashes = {
        name: sha256(directory / name)
        for name in ("manifest.json", "failed_runs.json", "scene_visibility.csv", "pair_visibility.csv")
    }
    return manifest, file_hashes


def audit_analysis_outputs(
    standard_dir: Path,
    visibility_dir: Path,
    expected_scenes: int | None = None,
    batch_audit_path: Path | None = None,
) -> dict[str, Any]:
    standard_dir = standard_dir.resolve()
    visibility_dir = visibility_dir.resolve()
    standard, keys, pair_keys, standard_hashes = validate_standard(standard_dir, expected_scenes)
    visibility, visibility_hashes = validate_visibility(
        visibility_dir, expected_scenes, keys, pair_keys
    )
    require(
        standard.get("batch_manifest_sha256") == visibility.get("batch_manifest_sha256"),
        "Standard and visibility analyses use different batch manifests",
    )
    require(
        standard.get("batch_results_sha256") == visibility.get("batch_results_sha256"),
        "Standard and visibility analyses use different batch results",
    )

    batch_audit_sha256 = None
    if batch_audit_path is not None:
        batch_audit = load_json(batch_audit_path)
        require(batch_audit.get("passed") is True, "Referenced batch audit did not pass")
        require(batch_audit.get("expected_scenes") == len(keys), "Batch-audit population mismatch")
        batch_hashes = batch_audit.get("batch_file_sha256", {})
        require(batch_hashes.get("manifest.json") == standard.get("batch_manifest_sha256"), "Analysis manifest hash differs from batch audit")
        require(batch_hashes.get("results.jsonl") == standard.get("batch_results_sha256"), "Analysis results hash differs from batch audit")
        batch_audit_sha256 = sha256(batch_audit_path)

    return {
        "schema_version": 1,
        "passed": True,
        "expected_scenes": len(keys),
        "directed_pairs": len(pair_keys),
        "batch_manifest_sha256": standard["batch_manifest_sha256"],
        "batch_results_sha256": standard["batch_results_sha256"],
        "batch_audit_sha256": batch_audit_sha256,
        "standard_dir": str(standard_dir),
        "visibility_dir": str(visibility_dir),
        "standard_file_sha256": standard_hashes,
        "visibility_file_sha256": visibility_hashes,
        "failure_labels": None,
        "quality_thresholds": None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--standard-dir", type=Path, required=True)
    parser.add_argument("--visibility-dir", type=Path, required=True)
    parser.add_argument("--expected-scenes", type=int)
    parser.add_argument("--batch-audit", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit_analysis_outputs(
        args.standard_dir,
        args.visibility_dir,
        expected_scenes=args.expected_scenes,
        batch_audit_path=args.batch_audit,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    require(not args.output.exists(), f"Refusing to overwrite existing audit: {args.output}")
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("passed", "expected_scenes", "directed_pairs")}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
