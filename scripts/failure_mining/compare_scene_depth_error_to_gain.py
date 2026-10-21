"""Relate scene-level GT depth error to oracle depth-replacement gain.

This is an offline validation analysis. Ground-truth depth errors and oracle
intervention gains are analysis targets; neither is available to a test-time
policy. The four error fields are declared in advance to avoid selecting a
favorable statistic after seeing the correlations.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

from scipy.stats import spearmanr

from scripts.failure_mining.compare_baselines import sha256, write_csv
from scripts.failure_mining.correlation_fdr import benjamini_hochberg


DEPTH_FACTOR = "depth_after_extrinsics_intrinsics"
EVIDENCE_FIELDS = (
    "depth.absolute_relative.mean",
    "depth.absolute_relative.p95",
    "depth.absolute_log_ratio.mean",
    "depth.gt_uncovered_ratio",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_csv(path: Path) -> list[dict[str, str]]:
    require(path.is_file(), f"Required file is missing: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_manifest(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"Required file is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"Expected JSON object: {path}")
    return value


def merge_scene_targets(
    depth_rows: list[dict[str, str]],
    factor_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    require(depth_rows, "Depth-error table is empty")
    depth_by_key: dict[tuple[str, str], dict[str, str]] = {}
    for row in depth_rows:
        key = (row["dataset"], row["scene"])
        require(key not in depth_by_key, f"Duplicate depth-error scene: {key}")
        depth_by_key[key] = row

    selected = [row for row in factor_rows if row["factor"] == DEPTH_FACTOR]
    require(selected, f"No targets for factor {DEPTH_FACTOR}")
    target_by_key: dict[tuple[str, str], dict[str, str]] = {}
    for row in selected:
        key = (row["dataset"], row["scene"])
        require(key not in target_by_key, f"Duplicate depth-gain target: {key}")
        target_by_key[key] = row
    require(set(depth_by_key) == set(target_by_key), "Depth-error and factor-target populations differ")

    output: list[dict[str, Any]] = []
    for key in sorted(depth_by_key):
        depth = depth_by_key[key]
        target = target_by_key[key]
        coverage = float(depth["joint_over_ground_truth_valid_ratio"])
        require(math.isfinite(coverage) and 0 <= coverage <= 1, f"Invalid GT coverage: {key}")
        row: dict[str, Any] = {
            "dataset": key[0],
            "scene": key[1],
            "factor": DEPTH_FACTOR,
            "conditional_gain": float(target["conditional_gain"]),
            "depth.gt_uncovered_ratio": 1.0 - coverage,
        }
        for field in EVIDENCE_FIELDS[:-1]:
            value = float(depth[field])
            require(math.isfinite(value), f"Non-finite depth error: {(key, field)}")
            row[field] = value
        require(math.isfinite(row["conditional_gain"]), f"Non-finite target: {key}")
        output.append(row)
    return output


def correlations(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    require(len(rows) >= 3, "At least three scenes are required")
    gains = [float(row["conditional_gain"]) for row in rows]
    require(len(set(gains)) >= 2, "Conditional gains are constant")
    output = []
    for field in EVIDENCE_FIELDS:
        values = [float(row[field]) for row in rows]
        require(len(set(values)) >= 2, f"Evidence field is constant: {field}")
        result = spearmanr(values, gains)
        require(
            math.isfinite(result.statistic) and math.isfinite(result.pvalue),
            f"Non-finite correlation: {field}",
        )
        output.append(
            {
                "factor": DEPTH_FACTOR,
                "evidence_field": field,
                "scenes": len(rows),
                "spearman_rho": float(result.statistic),
                "p_value_two_sided": float(result.pvalue),
            }
        )
    q_values = benjamini_hochberg([row["p_value_two_sided"] for row in output])
    for row, q_value in zip(output, q_values, strict=True):
        row["bh_q_value_within_depth_target"] = q_value
        row["tests_in_family"] = len(output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-depth-analysis", type=Path, required=True)
    parser.add_argument("--factor-targets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), f"Refusing to overwrite output: {args.output}")

    depth_manifest_path = args.gt_depth_analysis / "manifest.json"
    depth_table_path = args.gt_depth_analysis / "scene_gt_depth_error.csv"
    target_manifest_path = args.factor_targets / "manifest.json"
    target_table_path = args.factor_targets / "factor_conditional_gain_targets.csv"
    depth_manifest = read_manifest(depth_manifest_path)
    target_manifest = read_manifest(target_manifest_path)
    require(depth_manifest.get("scene_output_sha256") == sha256(depth_table_path), "Depth-error table hash differs")
    require(depth_manifest.get("uses_ground_truth_analysis") is True, "GT depth provenance is missing")
    require(depth_manifest.get("test_time_policy_input") is False, "GT depth is marked as policy input")
    require(target_manifest.get("target_output_sha256") == sha256(target_table_path), "Factor-target table hash differs")
    require(target_manifest.get("uses_ground_truth_intervention") is True, "Oracle intervention provenance is missing")
    require(target_manifest.get("test_time_policy_input") is False, "Oracle gain is marked as policy input")
    require(
        (depth_manifest.get("dataset"), depth_manifest.get("model"))
        == (target_manifest.get("dataset"), target_manifest.get("model")),
        "Dataset/model provenance differs",
    )

    merged = merge_scene_targets(read_csv(depth_table_path), read_csv(target_table_path))
    result = correlations(merged)
    args.output.mkdir(parents=True, exist_ok=False)
    merged_path = args.output / "scene_depth_error_and_gain.csv"
    correlation_path = args.output / "correlations_with_fdr.csv"
    write_csv(merged_path, merged)
    write_csv(correlation_path, result)
    manifest = {
        "dataset": depth_manifest["dataset"],
        "model": depth_manifest["model"],
        "factor": DEPTH_FACTOR,
        "scenes": len(merged),
        "evidence_fields": list(EVIDENCE_FIELDS),
        "depth_manifest_sha256": sha256(depth_manifest_path),
        "depth_table_sha256": sha256(depth_table_path),
        "factor_target_manifest_sha256": sha256(target_manifest_path),
        "factor_target_table_sha256": sha256(target_table_path),
        "merged_output_sha256": sha256(merged_path),
        "correlation_output_sha256": sha256(correlation_path),
        "analysis_script_sha256": sha256(Path(__file__)),
        "multiple_testing": "Benjamini-Hochberg across four predeclared depth-error fields",
        "uses_ground_truth_analysis": True,
        "test_time_policy_input": False,
        "failure_labels": None,
        "quality_thresholds": None,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
