"""Independently reproduce and audit GT geometry-factor profile tables."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

from scripts.failure_mining.analyze_gt_geometry_factor_profiles import analyze
from scripts.failure_mining.compare_baselines import sha256, write_csv


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_json(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"Required file is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"Expected JSON object: {path}")
    return value


def audit_profiles(summary_dir: Path, profile_dir: Path) -> dict[str, Any]:
    summary_manifest_path = summary_dir / "manifest.json"
    case_stage_path = summary_dir / "case_metric_stages.csv"
    profile_manifest_path = profile_dir / "manifest.json"
    case_profile_path = profile_dir / "case_metric_factor_profiles.csv"
    aggregate_path = profile_dir / "model_metric_factor_summary.csv"
    for path in (
        summary_manifest_path,
        case_stage_path,
        profile_manifest_path,
        case_profile_path,
        aggregate_path,
    ):
        require(path.is_file(), f"Required file is missing: {path}")

    stored = load_json(profile_manifest_path)
    profiles, aggregate, reproduced = analyze(summary_dir)
    require(
        stored.get("input_summary_manifest_sha256") == sha256(summary_manifest_path),
        "Profile input-summary-manifest hash differs",
    )
    require(
        stored.get("input_case_table_sha256") == sha256(case_stage_path),
        "Profile input-case-table hash differs",
    )
    for field in (
        "models",
        "scenes",
        "case_metric_profiles",
        "model_metric_rows",
        "factor_order",
        "attribution_scope",
        "uses_ground_truth_intervention",
        "test_time_policy_input",
        "success_threshold",
    ):
        require(stored.get(field) == reproduced.get(field), f"Profile provenance differs: {field}")
    require(stored.get("profile_output_sha256") == sha256(case_profile_path), "Case-profile hash differs")
    require(stored.get("aggregate_output_sha256") == sha256(aggregate_path), "Aggregate-profile hash differs")

    with tempfile.TemporaryDirectory() as directory:
        expected_profiles = Path(directory) / "profiles.csv"
        expected_aggregate = Path(directory) / "aggregate.csv"
        write_csv(expected_profiles, profiles)
        write_csv(expected_aggregate, aggregate)
        require(expected_profiles.read_bytes() == case_profile_path.read_bytes(), "Case profiles do not reproduce")
        require(expected_aggregate.read_bytes() == aggregate_path.read_bytes(), "Aggregate profiles do not reproduce")

    return {
        "schema_version": 1,
        "passed": True,
        "case_metric_profiles": reproduced["case_metric_profiles"],
        "model_metric_rows": reproduced["model_metric_rows"],
        "models": reproduced["models"],
        "scenes": reproduced["scenes"],
        "factor_order": reproduced["factor_order"],
        "attribution_scope": reproduced["attribution_scope"],
        "summary_manifest_sha256": sha256(summary_manifest_path),
        "input_case_table_sha256": sha256(case_stage_path),
        "profile_manifest_sha256": sha256(profile_manifest_path),
        "case_profile_sha256": sha256(case_profile_path),
        "aggregate_profile_sha256": sha256(aggregate_path),
        "uses_ground_truth_intervention": True,
        "test_time_policy_input": False,
        "success_threshold": None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), f"Refusing to overwrite output: {args.output}")
    report = audit_profiles(args.summary, args.profiles)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
