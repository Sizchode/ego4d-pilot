"""Audit nested ScanNet++ ground-truth geometry-factor outputs.

The audit verifies population, ordering, per-case artifacts, hashes, metric
schema, and cross-model use of the same image/depth ground truth.  It does not
interpret intervention gains or define a success threshold.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from scripts.audit.analysis_outputs import load_json, require, sha256


METRICS = ("acc", "comp", "overall", "precision", "recall", "fscore")
STAGES = (
    "official_native",
    "gt_extrinsics",
    "gt_extrinsics_intrinsics",
    "gt_extrinsics_intrinsics_depth",
)
METRIC_BLOCKS = {
    "official_native": "official_native_metrics",
    "gt_extrinsics": "gt_extrinsics_metrics",
    "gt_extrinsics_intrinsics": "gt_extrinsics_intrinsics_metrics",
    "gt_extrinsics_intrinsics_depth": "gt_extrinsics_intrinsics_depth_metrics",
}
MESH_OUTPUTS = {
    "gt_extrinsics_intrinsics_mesh_sha256": "gt_extrinsics_intrinsics_mesh.ply",
    "gt_extrinsics_intrinsics_depth_mesh_sha256":
        "gt_extrinsics_intrinsics_depth_mesh.ply",
}
CAPTURE_FILES = {
    "alignment_000.npz",
    "fused_mesh_000.ply",
    "fusion_inputs_000.npz",
    "surface_distance_000.npz",
    "surface_distance_001.npz",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    require(path.is_file(), f"Required file is missing: {path}")
    rows = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSONL at {path}:{line_number}: {error}") from error
        require(isinstance(value, dict), f"JSONL row is not an object: {path}:{line_number}")
        rows.append(value)
    return rows


def is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def require_unique_strings(value: Any, name: str) -> list[str]:
    require(isinstance(value, list) and value, f"{name} must be a non-empty list")
    require(all(isinstance(item, str) and item for item in value), f"Invalid {name}")
    require(len(value) == len(set(value)), f"Duplicate values in {name}")
    return value


def audit_factor_outputs(
    factor_dir: Path,
    *,
    expected_models: list[str] | None = None,
    expected_scenes: list[str] | None = None,
) -> dict[str, Any]:
    factor_dir = factor_dir.resolve()
    manifest_path = factor_dir / "manifest.json"
    results_path = factor_dir / "results.jsonl"
    manifest = load_json(manifest_path)
    rows = read_jsonl(results_path)

    require(manifest.get("dataset") == "scannetpp", "Unexpected factor dataset")
    require(tuple(manifest.get("stages", [])) == STAGES, "Unexpected factor stages")
    require(manifest.get("uses_ground_truth_intervention") is True, "Missing GT provenance")
    require(manifest.get("test_time_policy_input") is False, "GT is marked as policy input")
    require(manifest.get("success_threshold") is None, "Factor output defines a success threshold")
    require(manifest.get("results_sha256") == sha256(results_path), "Factor results hash mismatch")
    models = require_unique_strings(manifest.get("models"), "models")
    scenes = require_unique_strings(manifest.get("scenes"), "scenes")
    if expected_models is not None:
        require(models == expected_models, "Factor model population/order differs from expectation")
    if expected_scenes is not None:
        require(scenes == expected_scenes, "Factor scene population/order differs from expectation")

    expected_identities = [(model, scene) for scene in scenes for model in models]
    require(manifest.get("cases") == len(expected_identities), "Manifest case count is not model × scene")
    require(len(rows) == len(expected_identities), "Factor result row count is incomplete")
    actual_identities = [(row.get("model"), row.get("scene")) for row in rows]
    require(actual_identities == expected_identities, "Factor case population/order differs from manifest")

    population_hashes: dict[str, tuple[str, str]] = {}
    case_artifact_hashes: dict[str, dict[str, str]] = {}
    model_repositories: dict[str, str] = {}
    runs: set[str] = set()
    for row in rows:
        model, scene = row["model"], row["scene"]
        identity = f"{model}:{scene}"
        require(row.get("dataset") == "scannetpp", f"Unexpected dataset: {identity}")
        require(tuple(row.get("stages", [])) == STAGES, f"Stage mismatch: {identity}")
        require(row.get("uses_ground_truth_intervention") is True, f"Missing GT provenance: {identity}")
        require(row.get("test_time_policy_input") is False, f"GT is policy input: {identity}")
        require(isinstance(row.get("views"), int) and row["views"] > 0, f"Invalid view count: {identity}")
        require(row.get("seed") == manifest.get("seed"), f"Seed mismatch: {identity}")
        run = row.get("run")
        require(isinstance(run, str) and run, f"Missing source run: {identity}")
        require(run not in runs, f"Duplicate source run: {run}")
        runs.add(run)
        for stage, block_name in METRIC_BLOCKS.items():
            block = row.get(block_name)
            require(
                isinstance(block, dict) and set(block) == set(METRICS),
                f"Metric schema mismatch for {identity}, {stage}",
            )
            require(
                all(isinstance(value, (int, float)) and math.isfinite(value) for value in block.values()),
                f"Non-finite metric for {identity}, {stage}",
            )

        repository = row.get("model_repository")
        require(
            isinstance(repository, str) and repository,
            f"Missing model repository identity: {identity}",
        )
        if model in model_repositories:
            require(
                model_repositories[model] == repository,
                f"Model repository changes across scenes: {model}",
            )
        else:
            model_repositories[model] = repository
        capture_hashes = row.get("capture_sha256")
        require(
            isinstance(capture_hashes, dict)
            and set(capture_hashes) == CAPTURE_FILES
            and all(is_sha256(value) for value in capture_hashes.values()),
            f"Invalid official-capture provenance: {identity}",
        )
        for count_name in (
            "gt_point_count",
            "gt_extrinsics_intrinsics_point_count",
            "gt_extrinsics_intrinsics_depth_point_count",
        ):
            require(
                isinstance(row.get(count_name), int) and row[count_name] > 0,
                f"Invalid point count {count_name}: {identity}",
            )
        require(is_sha256(row.get("gt_meta_sha256")), f"Invalid GT metadata hash: {identity}")
        hashes = (
            row.get("image_population_sha256"),
            row.get("gt_depth_population_sha256"),
        )
        require(all(is_sha256(value) for value in hashes), f"Invalid population hash: {identity}")
        if scene in population_hashes:
            require(population_hashes[scene] == hashes, f"Cross-model GT population differs: {scene}")
        else:
            population_hashes[scene] = hashes

        case_dir = factor_dir / f"{model}-{scene}"
        result_path = case_dir / "result.json"
        require(load_json(result_path) == row, f"Per-case result differs from JSONL: {identity}")
        artifact_hashes = {"result.json": sha256(result_path)}
        for field, filename in MESH_OUTPUTS.items():
            path = case_dir / filename
            require(path.is_file() and path.stat().st_size > 0, f"Missing mesh: {path}")
            observed = sha256(path)
            require(row.get(field) == observed, f"Mesh hash mismatch: {path}")
            artifact_hashes[filename] = observed
        case_artifact_hashes[identity] = artifact_hashes

    case_directories = sorted(
        path.name for path in factor_dir.iterdir() if path.is_dir()
    )
    expected_directories = sorted(f"{model}-{scene}" for model, scene in expected_identities)
    require(case_directories == expected_directories, "Unexpected or missing factor case directories")
    require(
        len(set(model_repositories.values())) == len(model_repositories),
        "Multiple model IDs map to the same repository identity",
    )
    return {
        "schema_version": 1,
        "passed": True,
        "dataset": "scannetpp",
        "models": models,
        "model_repositories": model_repositories,
        "scenes": scenes,
        "cases": len(rows),
        "stages": list(STAGES),
        "factor_manifest_sha256": sha256(manifest_path),
        "factor_results_sha256": sha256(results_path),
        "case_artifact_sha256": case_artifact_hashes,
        "uses_ground_truth_intervention": True,
        "test_time_policy_input": False,
        "success_threshold": None,
        "quality_interpretation": None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--expected-model", action="append")
    parser.add_argument("--expected-scene", action="append")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit_factor_outputs(
        args.input,
        expected_models=args.expected_model,
        expected_scenes=args.expected_scene,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    require(not args.output.exists(), f"Refusing to overwrite existing audit: {args.output}")
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("passed", "cases", "models", "scenes")}, indent=2))


if __name__ == "__main__":
    main()
