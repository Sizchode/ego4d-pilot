"""Test whether observable per-view evidence localizes GT depth error.

Each scene is the statistical unit.  Spearman correlations are computed within
scene, then tested across scenes; views are never treated as independent scenes.
Ground-truth depth errors are offline targets and are not policy inputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr, wilcoxon

from scripts.failure_mining.compare_baselines import sha256, write_csv
from scripts.failure_mining.correlation_fdr import benjamini_hochberg


IMAGE_FIELDS = (
    "gradient.mean",
    "gradient.p95",
    "luminance.entropy_bits",
    "rgb.any_channel_at_floor_ratio",
    "rgb.any_channel_at_ceiling_ratio",
)
VISIBILITY_FIELDS = (
    "source_depth_not_behind.absolute_relative_z_nearest.p95",
    "source_depth_not_behind_ratio",
)
TARGET_FIELDS = (
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


def load_json(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"Required file is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"Expected JSON object: {path}")
    return value


def finite_float(value: str | None) -> float | None:
    if value in {None, ""}:
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def aggregate_visibility(
    pair_rows: list[dict[str, str]],
    population: set[tuple[str, str, int]],
) -> dict[tuple[str, str, int], dict[str, float | None]]:
    values: dict[tuple[str, str, int], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in pair_rows:
        dataset, scene = row["dataset"], row["scene"]
        source, target = int(row["source_index"]), int(row["target_index"])
        source_key, target_key = (dataset, scene, source), (dataset, scene, target)
        if source_key not in population and target_key not in population:
            continue
        require(source_key in population and target_key in population, f"Visibility pair crosses selected population: {(source_key, target_key)}")
        for field in VISIBILITY_FIELDS:
            value = finite_float(row.get(field))
            if value is None:
                continue
            values[source_key][f"visibility.outgoing.{field}.mean"].append(value)
            values[target_key][f"visibility.incoming.{field}.mean"].append(value)
    output: dict[tuple[str, str, int], dict[str, float | None]] = {}
    expected_fields = {
        f"visibility.{direction}.{field}.mean"
        for direction in ("outgoing", "incoming")
        for field in VISIBILITY_FIELDS
    }
    for key in sorted(population):
        available = values.get(key, {})
        output[key] = {
            field: statistics.fmean(available[field]) if available.get(field) else None
            for field in expected_fields
        }
    return output


def merge_view_evidence(
    target_rows: list[dict[str, str]],
    image_rows: list[dict[str, str]],
    visibility_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    identities = [(row["dataset"], row["scene"], int(row["view_index"])) for row in target_rows]
    require(len(identities) == len(set(identities)) and identities, "Target view population is empty or duplicated")
    population = set(identities)
    image_lookup = {}
    for row in image_rows:
        key = (row["dataset"], row["scene"], int(row["index"]))
        if key not in population:
            continue
        require(key not in image_lookup, f"Duplicate image evidence: {key}")
        image_lookup[key] = row
    require(set(image_lookup) == population, "Image-evidence population differs from GT target views")
    visibility = aggregate_visibility(visibility_rows, population)

    merged = []
    for row, key in zip(target_rows, identities, strict=True):
        image = image_lookup[key]
        output: dict[str, Any] = {
            "dataset": key[0],
            "scene": key[1],
            "view_index": key[2],
        }
        for field in TARGET_FIELDS[:-1]:
            value = finite_float(row.get(field))
            output[field] = value
        coverage = finite_float(row.get("joint_over_ground_truth_valid_ratio"))
        gt_valid_count = int(row["ground_truth_valid_count"])
        joint_valid_count = int(row["joint_valid_count"])
        require(0 <= joint_valid_count <= gt_valid_count, f"Invalid GT/joint counts: {key}")
        if gt_valid_count == 0:
            require(coverage is None, f"Coverage must be undefined without valid GT: {key}")
            output["depth.gt_uncovered_ratio"] = None
        else:
            require(coverage is not None and 0 <= coverage <= 1, f"Invalid GT coverage: {key}")
            require(
                math.isclose(coverage, joint_valid_count / gt_valid_count, rel_tol=1e-12, abs_tol=1e-12),
                f"GT coverage disagrees with counts: {key}",
            )
            output["depth.gt_uncovered_ratio"] = 1.0 - coverage
        for field in IMAGE_FIELDS:
            value = finite_float(image.get(field))
            require(value is not None, f"Missing image evidence: {(key, field)}")
            output[f"image.{field}"] = value
        output.update(visibility[key])
        merged.append(output)
    return merged


def scene_correlations(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["scene"])].append(row)
    evidence_fields = [f"image.{field}" for field in IMAGE_FIELDS] + [
        f"visibility.{direction}.{field}.mean"
        for direction in ("outgoing", "incoming")
        for field in VISIBILITY_FIELDS
    ]
    output = []
    for (dataset, scene), scene_rows in sorted(grouped.items()):
        for target in TARGET_FIELDS:
            for evidence in evidence_fields:
                complete_rows = [
                    row
                    for row in scene_rows
                    if row.get(target) is not None and row.get(evidence) is not None
                ]
                if len(complete_rows) < 3:
                    continue
                target_values = [float(row[target]) for row in complete_rows]
                if len(set(target_values)) < 2:
                    continue
                evidence_values = [float(row[evidence]) for row in complete_rows]
                if len(set(evidence_values)) < 2:
                    continue
                result = spearmanr(evidence_values, target_values)
                if not (math.isfinite(result.statistic) and math.isfinite(result.pvalue)):
                    continue
                output.append(
                    {
                        "dataset": dataset,
                        "scene": scene,
                        "views": len(complete_rows),
                        "target_field": target,
                        "evidence_field": evidence,
                        "spearman_rho": float(result.statistic),
                        "p_value_two_sided_descriptive": float(result.pvalue),
                    }
                )
    return output


def aggregate_correlations(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(row["target_field"], row["evidence_field"])].append(float(row["spearman_rho"]))
    output = []
    by_target: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for (target, evidence), values in sorted(grouped.items()):
        nonzero = np.asarray([value for value in values if value != 0], dtype=np.float64)
        p_value = 1.0 if nonzero.size == 0 else float(wilcoxon(nonzero, alternative="two-sided").pvalue)
        by_target[target].append(
            {
                "target_field": target,
                "evidence_field": evidence,
                "scenes": len(values),
                "mean_scene_spearman_rho": statistics.fmean(values),
                "median_scene_spearman_rho": statistics.median(values),
                "positive_scenes": sum(value > 0 for value in values),
                "zero_scenes": sum(value == 0 for value in values),
                "negative_scenes": sum(value < 0 for value in values),
                "wilcoxon_p_value_two_sided": p_value,
            }
        )
    for target, tests in sorted(by_target.items()):
        q_values = benjamini_hochberg([row["wilcoxon_p_value_two_sided"] for row in tests])
        for row, q_value in zip(tests, q_values, strict=True):
            row["bh_q_value_within_target"] = q_value
            row["tests_in_target_family"] = len(tests)
            output.append(row)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-depth-analysis", type=Path, required=True)
    parser.add_argument("--image-analysis", type=Path, required=True)
    parser.add_argument("--visibility-analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), f"Refusing to overwrite output: {args.output}")
    gt_manifest_path = args.gt_depth_analysis / "manifest.json"
    gt_view_path = args.gt_depth_analysis / "view_gt_depth_error.csv"
    image_manifest_path = args.image_analysis / "manifest.json"
    image_path = args.image_analysis / "image_evidence.csv"
    visibility_manifest_path = args.visibility_analysis / "manifest.json"
    visibility_path = args.visibility_analysis / "pair_visibility.csv"
    gt_manifest = load_json(gt_manifest_path)
    image_manifest = load_json(image_manifest_path)
    visibility_manifest = load_json(visibility_manifest_path)
    require(gt_manifest.get("view_output_sha256") == sha256(gt_view_path), "GT view-table hash differs")
    require(gt_manifest.get("uses_ground_truth_analysis") is True, "GT analysis provenance is missing")
    require(gt_manifest.get("test_time_policy_input") is False, "GT targets are marked as policy input")
    require(gt_manifest.get("quality_thresholds") is None, "GT analysis uses a quality threshold")
    require(image_manifest.get("uses_ground_truth") is False, "Image evidence uses ground truth")
    require(image_manifest.get("execution_failures") == 0, "Image analysis has execution failures")
    require(visibility_manifest.get("visibility_ground_truth") is False, "Visibility evidence is marked as GT")
    require(visibility_manifest.get("execution_failures") == 0, "Visibility analysis has execution failures")

    merged = merge_view_evidence(read_csv(gt_view_path), read_csv(image_path), read_csv(visibility_path))
    per_scene = scene_correlations(merged)
    aggregate = aggregate_correlations(per_scene)
    args.output.mkdir(parents=True, exist_ok=False)
    merged_path = args.output / "merged_view_evidence.csv"
    scene_path = args.output / "scene_view_rank_correlations.csv"
    aggregate_path = args.output / "aggregate_view_rank_correlations.csv"
    write_csv(merged_path, merged)
    write_csv(scene_path, per_scene)
    write_csv(aggregate_path, aggregate)
    manifest = {
        "gt_depth_manifest_sha256": sha256(gt_manifest_path),
        "gt_depth_views_sha256": sha256(gt_view_path),
        "image_manifest_sha256": sha256(image_manifest_path),
        "image_evidence_sha256": sha256(image_path),
        "visibility_manifest_sha256": sha256(visibility_manifest_path),
        "visibility_evidence_sha256": sha256(visibility_path),
        "target_fields": list(TARGET_FIELDS),
        "image_fields": list(IMAGE_FIELDS),
        "visibility_fields": list(VISIBILITY_FIELDS),
        "visibility_aggregation": "separate arithmetic means over outgoing and incoming adjacent directed pairs",
        "missing_value_policy": "within-scene complete cases per target/evidence pair; require at least three views",
        "view_rows": len(merged),
        "scene_correlation_rows": len(per_scene),
        "aggregate_rows": len(aggregate),
        "merged_output_sha256": sha256(merged_path),
        "scene_output_sha256": sha256(scene_path),
        "aggregate_output_sha256": sha256(aggregate_path),
        "analysis_script_sha256": sha256(Path(__file__)),
        "statistical_unit": "scene",
        "multiple_testing": "Benjamini-Hochberg across predeclared evidence fields within each target",
        "uses_ground_truth_targets": True,
        "test_time_policy_input": False,
        "selection_threshold": None,
        "top_k": None,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: manifest[key] for key in ("view_rows", "scene_correlation_rows", "aggregate_rows", "statistical_unit")}, indent=2))


if __name__ == "__main__":
    main()
