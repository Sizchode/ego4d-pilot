"""Audit final cross-model standard and visibility comparisons."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from scripts.audit.analysis_outputs import (
    audit_analysis_outputs,
    load_json,
    require,
    sha256,
)


STANDARD_OUTPUTS = {
    "execution_rows": "execution_matrix.csv",
    "model_metric_rank_rows": "model_metric_rank_rows.csv",
    "scene_metric_spread_rows": "scene_metric_spread.csv",
    "pairwise_rank_agreement_rows": "pairwise_rank_agreement.csv",
}
VISIBILITY_OUTPUTS = {
    "model_scene_rows": "model_scene_visibility.csv",
    "rank_rows": "model_visibility_ranks.csv",
    "pairwise_rows": "pairwise_visibility_rank_agreement.csv",
    "scene_spread_rows": "scene_visibility_rank_spread.csv",
}


def csv_rows(path: Path) -> int:
    if not path.is_file():
        raise ValueError(f"Required file is missing: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def verify_output_counts(directory: Path, counts: dict[str, Any], mapping: dict[str, str]) -> None:
    require(set(counts) == set(mapping), f"Unexpected output-count keys in {directory}")
    for key, filename in mapping.items():
        require(counts[key] == csv_rows(directory / filename), f"Output row count mismatch: {directory / filename}")


def locate_batch_audits(directory: Path) -> dict[tuple[str, str], Path]:
    reports: dict[tuple[str, str], Path] = {}
    for path in sorted(directory.glob("*.json")):
        report = load_json(path)
        require(report.get("passed") is True, f"Batch audit did not pass: {path}")
        batch_hashes = report.get("batch_file_sha256", {})
        manifest_hash = batch_hashes.get("manifest.json")
        results_hash = batch_hashes.get("results.jsonl")
        require(isinstance(manifest_hash, str), f"Batch audit lacks manifest hash: {path}")
        require(isinstance(results_hash, str), f"Batch audit lacks results hash: {path}")
        provenance_key = (manifest_hash, results_hash)
        require(provenance_key not in reports, f"Duplicate batch audit provenance: {provenance_key}")
        reports[provenance_key] = path
    require(reports, f"No batch-audit reports found in {directory}")
    return reports


def verify_standard_input(model: str, item: dict[str, Any]) -> Path:
    directory = Path(item["analysis_dir"])
    require(sha256(directory / "manifest.json") == item["analysis_manifest_sha256"], f"Standard analysis manifest hash mismatch for {model}")
    require(sha256(directory / "scenes.csv") == item["scenes_sha256"], f"Standard scenes hash mismatch for {model}")
    require(sha256(directory / "rankings.csv") == item["rankings_sha256"], f"Standard rankings hash mismatch for {model}")
    return directory


def verify_visibility_input(model: str, item: dict[str, Any]) -> Path:
    directory = Path(item["analysis_dir"])
    require(sha256(directory / "manifest.json") == item["manifest_sha256"], f"Visibility manifest hash mismatch for {model}")
    require(sha256(directory / "scene_visibility.csv") == item["scene_visibility_sha256"], f"Visibility scenes hash mismatch for {model}")
    return directory


def audit_cross_model_outputs(
    standard_cross_dir: Path,
    visibility_cross_dir: Path,
    batch_audit_dir: Path,
    expected_scenes: int,
) -> dict[str, Any]:
    standard_cross_dir = standard_cross_dir.resolve()
    visibility_cross_dir = visibility_cross_dir.resolve()
    batch_audit_dir = batch_audit_dir.resolve()
    standard_manifest = load_json(standard_cross_dir / "manifest.json")
    visibility_manifest = load_json(visibility_cross_dir / "manifest.json")
    models = standard_manifest.get("models")
    require(isinstance(models, list) and len(models) >= 2 and len(models) == len(set(models)), "Invalid standard cross-model list")
    require(visibility_manifest.get("models") == models, "Standard and visibility model order differ")
    require(standard_manifest.get("scene_population") == expected_scenes, "Unexpected standard cross-model population")
    require(visibility_manifest.get("scene_population") == expected_scenes, "Unexpected visibility cross-model population")
    require(len(standard_manifest.get("population_order", [])) == expected_scenes, "Cross-model population order is incomplete")
    verify_output_counts(standard_cross_dir, standard_manifest.get("outputs", {}), STANDARD_OUTPUTS)
    verify_output_counts(visibility_cross_dir, visibility_manifest.get("outputs", {}), VISIBILITY_OUTPUTS)

    standard_inputs = standard_manifest.get("inputs", {})
    visibility_inputs = visibility_manifest.get("inputs", {})
    require(list(standard_inputs) == models, "Standard input model order differs from model list")
    require(list(visibility_inputs) == models, "Visibility input model order differs from model list")
    batch_audits = locate_batch_audits(batch_audit_dir)
    model_audits = {}
    for model in models:
        standard_dir = verify_standard_input(model, standard_inputs[model])
        visibility_dir = verify_visibility_input(model, visibility_inputs[model])
        standard_analysis_manifest = load_json(standard_dir / "manifest.json")
        batch_manifest_hash = standard_analysis_manifest.get("batch_manifest_sha256")
        batch_results_hash = standard_analysis_manifest.get("batch_results_sha256")
        provenance_key = (batch_manifest_hash, batch_results_hash)
        require(provenance_key in batch_audits, f"No batch audit matches {model}")
        model_audits[model] = audit_analysis_outputs(
            standard_dir,
            visibility_dir,
            expected_scenes=expected_scenes,
            batch_audit_path=batch_audits[provenance_key],
        )

    first_standard_dir = Path(standard_inputs[models[0]]["analysis_dir"])
    with (first_standard_dir / "scenes.csv").open(newline="", encoding="utf-8") as handle:
        first_population = [
            {"dataset": row["dataset"], "scene": row["scene"]}
            for row in csv.DictReader(handle)
        ]
    require(first_population == standard_manifest["population_order"], "Cross-model population order differs from source analyses")

    cross_files = {
        "standard": {
            filename: sha256(standard_cross_dir / filename)
            for filename in ("manifest.json", *STANDARD_OUTPUTS.values())
        },
        "visibility": {
            filename: sha256(visibility_cross_dir / filename)
            for filename in ("manifest.json", *VISIBILITY_OUTPUTS.values())
        },
    }
    return {
        "schema_version": 1,
        "passed": True,
        "models": models,
        "scene_population": expected_scenes,
        "model_audits": model_audits,
        "cross_file_sha256": cross_files,
        "failure_labels": None,
        "quality_thresholds": None,
        "comparison_policy": "within-dataset and within-metric only",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--standard-cross-dir", type=Path, required=True)
    parser.add_argument("--visibility-cross-dir", type=Path, required=True)
    parser.add_argument("--batch-audit-dir", type=Path, required=True)
    parser.add_argument("--expected-scenes", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit_cross_model_outputs(
        args.standard_cross_dir,
        args.visibility_cross_dir,
        args.batch_audit_dir,
        args.expected_scenes,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    require(not args.output.exists(), f"Refusing to overwrite existing audit: {args.output}")
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"passed": True, "models": report["models"], "scene_population": report["scene_population"]}, indent=2))


if __name__ == "__main__":
    main()
