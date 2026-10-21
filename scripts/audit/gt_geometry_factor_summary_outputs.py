"""Independently reproduce and audit GT geometry-factor summary tables."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

from scripts.failure_mining.compare_baselines import sha256, write_csv
from scripts.failure_mining.summarize_gt_geometry_factors import summarize


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"Expected JSON object: {path}")
    return value


def audit_summary(raw_dir: Path, summary_dir: Path, raw_audit_path: Path) -> dict[str, Any]:
    raw_manifest_path = raw_dir / "manifest.json"
    raw_results_path = raw_dir / "results.jsonl"
    summary_manifest_path = summary_dir / "manifest.json"
    case_path = summary_dir / "case_metric_stages.csv"
    aggregate_path = summary_dir / "model_metric_stage_summary.csv"
    for path in (
        raw_manifest_path,
        raw_results_path,
        summary_manifest_path,
        case_path,
        aggregate_path,
        raw_audit_path,
    ):
        require(path.is_file(), f"Required file is missing: {path}")

    raw = load_json(raw_manifest_path)
    raw_audit = load_json(raw_audit_path)
    summary_manifest = load_json(summary_manifest_path)
    require(raw_audit.get("passed") is True, "Raw factor-output audit did not pass")
    require(raw_audit.get("factor_manifest_sha256") == sha256(raw_manifest_path), "Raw manifest hash differs from audit")
    require(raw_audit.get("factor_results_sha256") == sha256(raw_results_path), "Raw results hash differs from audit")
    require(raw_audit.get("cases") == raw.get("cases"), "Raw audit case count differs")
    require(raw_audit.get("models") == raw.get("models"), "Raw audit model population differs")
    require(raw_audit.get("scenes") == raw.get("scenes"), "Raw audit scene population differs")

    case_rows, aggregate_rows, provenance = summarize(raw_dir)
    require(summary_manifest.get("input_manifest_sha256") == sha256(raw_manifest_path), "Summary input-manifest hash differs")
    require(summary_manifest.get("input_results_sha256") == sha256(raw_results_path), "Summary input-results hash differs")
    for field in (
        "cases",
        "case_metric_rows",
        "summary_rows",
        "models",
        "scenes",
        "stages",
        "uses_ground_truth_intervention",
        "test_time_policy_input",
        "success_threshold",
    ):
        require(summary_manifest.get(field) == provenance.get(field), f"Summary provenance differs: {field}")
    require(summary_manifest.get("case_output_sha256") == sha256(case_path), "Case-table hash differs")
    require(summary_manifest.get("summary_output_sha256") == sha256(aggregate_path), "Aggregate-table hash differs")

    with tempfile.TemporaryDirectory() as directory:
        expected_case = Path(directory) / "case.csv"
        expected_aggregate = Path(directory) / "aggregate.csv"
        write_csv(expected_case, case_rows)
        write_csv(expected_aggregate, aggregate_rows)
        require(expected_case.read_bytes() == case_path.read_bytes(), "Case table does not reproduce")
        require(expected_aggregate.read_bytes() == aggregate_path.read_bytes(), "Aggregate table does not reproduce")

    return {
        "schema_version": 1,
        "passed": True,
        "cases": provenance["cases"],
        "case_metric_rows": provenance["case_metric_rows"],
        "summary_rows": provenance["summary_rows"],
        "models": provenance["models"],
        "scenes": provenance["scenes"],
        "raw_audit_sha256": sha256(raw_audit_path),
        "raw_manifest_sha256": sha256(raw_manifest_path),
        "raw_results_sha256": sha256(raw_results_path),
        "summary_manifest_sha256": sha256(summary_manifest_path),
        "case_table_sha256": sha256(case_path),
        "aggregate_table_sha256": sha256(aggregate_path),
        "uses_ground_truth_intervention": True,
        "test_time_policy_input": False,
        "success_threshold": None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--raw-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), f"Refusing to overwrite output: {args.output}")
    report = audit_summary(args.raw, args.summary, args.raw_audit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
