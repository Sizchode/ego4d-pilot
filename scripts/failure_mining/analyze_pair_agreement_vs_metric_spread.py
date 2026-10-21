#!/usr/bin/env python3
"""Compare cross-backbone camera-pair agreement with endpoint spread.

The analysis preserves rotation and translation as separate predictors and
pose and reconstruction as separate endpoints. It assigns no failure label.
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

from scripts.failure_mining.compare_baselines import pearson, sha256, write_csv
from scripts.failure_mining.compare_image_evidence import average_ranks


CAMERA_METRICS = ("rotation_degrees", "translation_degrees")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def build_scene_rows(
    pair_rows: list[dict[str, str]],
    rank_rows: list[dict[str, str]],
    dataset: str,
    pose_metric: str,
    geometry_metric: str,
) -> list[dict[str, Any]]:
    require(pair_rows, "Pair-agreement rows are empty")
    require(rank_rows, "Metric-rank rows are empty")

    pair_values: dict[tuple[str, str], list[float]] = defaultdict(list)
    pair_identities: dict[tuple[str, str], set[tuple[str, str]]] = defaultdict(set)
    seen_pairs: set[tuple[str, str, str, str]] = set()
    for row in pair_rows:
        if row.get("dataset") != dataset:
            continue
        metric = row["metric"]
        require(metric in CAMERA_METRICS, f"Unknown camera metric: {metric}")
        left, right = sorted((row["model_left"], row["model_right"]))
        identity = (row["scene"], metric, left, right)
        require(identity not in seen_pairs, f"Duplicate pair-agreement row: {identity}")
        seen_pairs.add(identity)
        value = float(row["spearman_pair_error"])
        require(math.isfinite(value), f"Non-finite pair agreement: {identity}")
        key = (row["scene"], metric)
        pair_values[key].append(value)
        pair_identities[key].add((left, right))

    scenes = sorted({scene for scene, _metric in pair_values})
    require(scenes, f"No pair-agreement rows for dataset {dataset}")
    reference_pairs: set[tuple[str, str]] | None = None
    for scene in scenes:
        for metric in CAMERA_METRICS:
            key = (scene, metric)
            require(key in pair_values, f"Missing camera metric for scene: {key}")
            if reference_pairs is None:
                reference_pairs = pair_identities[key]
            require(
                pair_identities[key] == reference_pairs,
                f"Model-pair population differs for scene/metric: {key}",
            )

    endpoint_values: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    for row in rank_rows:
        if row.get("dataset") != dataset or row.get("metric") not in {
            pose_metric,
            geometry_metric,
        }:
            continue
        key = (row["scene"], row["metric"])
        model = row["model"]
        require(model not in endpoint_values[key], f"Duplicate endpoint row: {key + (model,)}")
        value = float(row["value"])
        require(math.isfinite(value), f"Non-finite endpoint value: {key + (model,)}")
        endpoint_values[key][model] = value

    reference_models: set[str] | None = None
    output: list[dict[str, Any]] = []
    for scene in scenes:
        pose_key = (scene, pose_metric)
        geometry_key = (scene, geometry_metric)
        require(pose_key in endpoint_values, f"Missing pose endpoint for scene: {scene}")
        require(geometry_key in endpoint_values, f"Missing geometry endpoint for scene: {scene}")
        models = set(endpoint_values[pose_key])
        require(models == set(endpoint_values[geometry_key]), f"Endpoint model population differs: {scene}")
        if reference_models is None:
            reference_models = models
        require(models == reference_models, f"Model population differs across scenes: {scene}")
        pose = list(endpoint_values[pose_key].values())
        geometry = list(endpoint_values[geometry_key].values())
        output.append(
            {
                "dataset": dataset,
                "scene": scene,
                "models": len(models),
                "model_pairs": len(reference_pairs or ()),
                "median_rotation_pair_error_agreement": statistics.median(
                    pair_values[(scene, "rotation_degrees")]
                ),
                "median_translation_pair_error_agreement": statistics.median(
                    pair_values[(scene, "translation_degrees")]
                ),
                "pose_metric": pose_metric,
                "pose_model_range": max(pose) - min(pose),
                "geometry_metric": geometry_metric,
                "geometry_model_range": max(geometry) - min(geometry),
            }
        )
    return output


def build_correlations(scene_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    require(len(scene_rows) >= 2, "At least two scenes are required")
    predictors = (
        "median_rotation_pair_error_agreement",
        "median_translation_pair_error_agreement",
    )
    endpoints = ("pose_model_range", "geometry_model_range")
    output: list[dict[str, Any]] = []
    for predictor in predictors:
        predictor_ranks = average_ranks([float(row[predictor]) for row in scene_rows])
        for endpoint in endpoints:
            endpoint_ranks = average_ranks([float(row[endpoint]) for row in scene_rows])
            output.append(
                {
                    "dataset": scene_rows[0]["dataset"],
                    "predictor": predictor,
                    "endpoint": endpoint,
                    "scenes": len(scene_rows),
                    "spearman_predictor_vs_endpoint": pearson(
                        predictor_ranks, endpoint_ranks
                    ),
                    "failure_labels": None,
                    "composite_score": None,
                }
            )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-correlations", action="append", type=Path, required=True)
    parser.add_argument("--metric-rank-rows", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--pose-metric", required=True)
    parser.add_argument("--geometry-metric", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    pair_rows = [row for path in args.pair_correlations for row in read_csv(path)]
    rank_rows = read_csv(args.metric_rank_rows)
    scene_rows = build_scene_rows(
        pair_rows,
        rank_rows,
        args.dataset,
        args.pose_metric,
        args.geometry_metric,
    )
    correlation_rows = build_correlations(scene_rows)

    args.output.mkdir(parents=True, exist_ok=False)
    scene_path = args.output / "scene_pair_agreement_and_metric_spread.csv"
    correlation_path = args.output / "pair_agreement_spread_correlations.csv"
    write_csv(scene_path, scene_rows)
    write_csv(correlation_path, correlation_rows)
    manifest = {
        "dataset": args.dataset,
        "pose_metric": args.pose_metric,
        "geometry_metric": args.geometry_metric,
        "scenes": len(scene_rows),
        "scene_rows": len(scene_rows),
        "correlation_rows": len(correlation_rows),
        "pair_correlation_inputs": [
            {"path": str(path.resolve()), "sha256": sha256(path)}
            for path in args.pair_correlations
        ],
        "metric_rank_rows_sha256": sha256(args.metric_rank_rows),
        "scene_output_sha256": sha256(scene_path),
        "correlation_output_sha256": sha256(correlation_path),
        "analysis_script_sha256": sha256(Path(__file__)),
        "failure_labels": None,
        "composite_score": None,
        "policy": "scene-level median pair agreement; camera and endpoint metrics remain separate",
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
