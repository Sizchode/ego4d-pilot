"""Audit per-view ScanNet++ GT-depth error outputs without raw GT access."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from scripts.failure_mining.compare_baselines import sha256


PREFIXES = (
    "depth.signed_relative",
    "depth.absolute_relative",
    "depth.signed_log_ratio",
    "depth.absolute_log_ratio",
)
COUNT_FIELDS = (
    "pixels",
    "predicted_valid_count",
    "ground_truth_valid_count",
    "joint_valid_count",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_csv(path: Path) -> list[dict[str, str]]:
    require(path.is_file(), f"Required file is missing: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"Required file is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"Expected JSON object: {path}")
    return value


def close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12)


def audit_gt_depth_error(analysis_dir: Path) -> dict[str, Any]:
    manifest_path = analysis_dir / "manifest.json"
    view_path = analysis_dir / "view_gt_depth_error.csv"
    scene_path = analysis_dir / "scene_gt_depth_error.csv"
    manifest = read_json(manifest_path)
    require(manifest.get("view_output_sha256") == sha256(view_path), "View-table hash differs")
    require(manifest.get("scene_output_sha256") == sha256(scene_path), "Scene-table hash differs")
    require(manifest.get("uses_ground_truth_analysis") is True, "Missing GT-analysis provenance")
    require(manifest.get("test_time_policy_input") is False, "GT is marked as policy input")
    require(manifest.get("failure_labels") is None, "Analysis assigns failure labels")
    require(manifest.get("quality_thresholds") is None, "Analysis uses quality thresholds")
    scenes = manifest.get("scenes")
    require(isinstance(scenes, list) and scenes and len(scenes) == len(set(scenes)), "Invalid scene population")

    view_rows = read_csv(view_path)
    scene_rows = read_csv(scene_path)
    require(len(view_rows) == manifest.get("view_rows"), "View-row count differs")
    require(len(scene_rows) == manifest.get("scene_rows") == len(scenes), "Scene-row count differs")
    scene_lookup = {row["scene"]: row for row in scene_rows}
    require(len(scene_lookup) == len(scene_rows) and set(scene_lookup) == set(scenes), "Scene table population differs")
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    observed_order = []
    for row in view_rows:
        require(row["dataset"] == "scannetpp", "Unexpected view dataset")
        if not observed_order or observed_order[-1] != row["scene"]:
            observed_order.append(row["scene"])
        grouped[row["scene"]].append(row)
    require(observed_order == scenes and set(grouped) == set(scenes), "View scene population/order differs")

    for scene in scenes:
        views = grouped[scene]
        indices = [int(row["view_index"]) for row in views]
        require(indices == list(range(len(views))), f"View indices are not contiguous: {scene}")
        scene_row = scene_lookup[scene]
        require(int(scene_row["views"]) == len(views), f"Scene view count differs: {scene}")
        totals = {field: sum(int(row[field]) for row in views) for field in COUNT_FIELDS}
        for field, total in totals.items():
            require(int(scene_row[field]) == total, f"Scene count differs: {(scene, field)}")
        ratios = {
            "predicted_valid_ratio": totals["predicted_valid_count"] / totals["pixels"],
            "ground_truth_valid_ratio": totals["ground_truth_valid_count"] / totals["pixels"],
            "joint_valid_ratio": totals["joint_valid_count"] / totals["pixels"],
            "joint_over_ground_truth_valid_ratio": totals["joint_valid_count"] / totals["ground_truth_valid_count"],
        }
        for field, value in ratios.items():
            require(close(float(scene_row[field]), value), f"Scene ratio differs: {(scene, field)}")
        for prefix in PREFIXES:
            counts = [int(row[f"{prefix}.finite_count"]) for row in views]
            for row, count in zip(views, counts, strict=True):
                require(
                    int(row[f"{prefix}.count"]) == count == int(row["joint_valid_count"]),
                    f"View metric count differs: {(scene, row['view_index'], prefix)}",
                )
                summary_fields = ("mean", "min", "max")
                if count == 0:
                    require(
                        all(row[f"{prefix}.{field}"] == "" for field in summary_fields),
                        f"Empty view has populated metric summary: {(scene, row['view_index'], prefix)}",
                    )
                else:
                    require(
                        all(math.isfinite(float(row[f"{prefix}.{field}"])) for field in summary_fields),
                        f"Populated view has invalid metric summary: {(scene, row['view_index'], prefix)}",
                    )
            total = sum(counts)
            require(int(scene_row[f"{prefix}.count"]) == total, f"Metric count differs: {(scene, prefix)}")
            require(int(scene_row[f"{prefix}.finite_count"]) == total, f"Metric finite count differs: {(scene, prefix)}")
            require(total > 0, f"Metric has no finite values: {(scene, prefix)}")
            weighted_mean = sum(
                float(row[f"{prefix}.mean"]) * count
                for row, count in zip(views, counts, strict=True)
                if count > 0
            ) / total
            require(close(float(scene_row[f"{prefix}.mean"]), weighted_mean), f"Metric mean differs: {(scene, prefix)}")
            require(
                close(float(scene_row[f"{prefix}.min"]), min(float(row[f"{prefix}.min"]) for row in views if int(row[f"{prefix}.finite_count"]) > 0)),
                f"Metric minimum differs: {(scene, prefix)}",
            )
            require(
                close(float(scene_row[f"{prefix}.max"]), max(float(row[f"{prefix}.max"]) for row in views if int(row[f"{prefix}.finite_count"]) > 0)),
                f"Metric maximum differs: {(scene, prefix)}",
            )

    return {
        "schema_version": 1,
        "passed": True,
        "scenes": scenes,
        "scene_rows": len(scene_rows),
        "view_rows": len(view_rows),
        "manifest_sha256": sha256(manifest_path),
        "view_output_sha256": sha256(view_path),
        "scene_output_sha256": sha256(scene_path),
        "recomputed_fields": [
            *COUNT_FIELDS,
            "validity ratios",
            "per-metric count/finite_count/mean/min/max",
        ],
        "unrecomputed_fields": "pooled quantiles require raw pixels and are covered only by stored file hashes",
        "uses_ground_truth_analysis": True,
        "test_time_policy_input": False,
        "failure_labels": None,
        "quality_thresholds": None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), f"Refusing to overwrite output: {args.output}")
    report = audit_gt_depth_error(args.analysis)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
