#!/usr/bin/env python3
"""Relate native metrics to GT-extrinsics geometry intervention gains.

This analysis is diagnostic: both the native pose metric and intervention use
ground truth. It does not define a deployable policy or a success threshold.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from scripts.failure_mining.compare_baselines import pearson, sha256, write_csv
from scripts.failure_mining.compare_image_evidence import average_ranks


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def build_joined_rows(
    rank_rows: list[dict[str, str]],
    intervention_rows: list[dict[str, str]],
    model_map: dict[str, str],
    dataset: str,
    pose_metric: str,
) -> list[dict[str, Any]]:
    require(rank_rows, "Native rank table is empty")
    require(intervention_rows, "Intervention table is empty")
    require(model_map, "Model map is empty")
    require(
        len(set(model_map.values())) == len(model_map),
        "Native model aliases in model map must be unique",
    )

    native_index: dict[tuple[str, str], float] = {}
    for row in rank_rows:
        if row.get("dataset") != dataset or row.get("metric") != pose_metric:
            continue
        key = (row["model"], row["scene"])
        require(key not in native_index, f"Duplicate native pose row: {key}")
        value = float(row["value"])
        require(math.isfinite(value), f"Non-finite native pose value: {key}")
        native_index[key] = value

    case_index: set[tuple[str, str]] = set()
    joined: list[dict[str, Any]] = []
    for row in intervention_rows:
        if row.get("metric") != "fscore":
            continue
        intervention_model = row["model"]
        require(
            intervention_model in model_map,
            f"Intervention model is absent from model map: {intervention_model}",
        )
        native_model = model_map[intervention_model]
        scene = row["scene"]
        case_key = (intervention_model, scene)
        require(case_key not in case_index, f"Duplicate intervention F-score row: {case_key}")
        case_index.add(case_key)
        native_key = (native_model, scene)
        require(native_key in native_index, f"Missing native pose row: {native_key}")

        numeric = {
            "native_pose_value": native_index[native_key],
            "official_native_fscore": float(row["official_native"]),
            "native_resample_control_fscore": float(row["native_resample_control"]),
            "gt_extrinsics_fscore": float(row["gt_extrinsics_intervention"]),
            "fscore_improvement_vs_control": float(row["improvement_vs_control"]),
            "control_delta_vs_official": float(row["control_delta_vs_official"]),
        }
        require(
            all(math.isfinite(value) for value in numeric.values()),
            f"Non-finite intervention value: {case_key}",
        )
        joined.append(
            {
                "dataset": dataset,
                "scene": scene,
                "intervention_model": intervention_model,
                "native_model": native_model,
                "pose_metric": pose_metric,
                **numeric,
            }
        )

    require(joined, "No intervention F-score rows were joined")
    return sorted(joined, key=lambda row: (row["intervention_model"], row["scene"]))


def build_correlations(joined_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    require(joined_rows, "Joined intervention table is empty")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in joined_rows:
        grouped[str(row["intervention_model"])].append(row)

    predictors = (
        "native_pose_value",
        "official_native_fscore",
    )
    output: list[dict[str, Any]] = []
    for model, rows in sorted(grouped.items()):
        require(len(rows) >= 2, f"At least two scenes are required for {model}")
        target = [float(row["fscore_improvement_vs_control"]) for row in rows]
        target_ranks = average_ranks(target)
        for predictor in predictors:
            values = [float(row[predictor]) for row in rows]
            output.append(
                {
                    "model": model,
                    "dataset": rows[0]["dataset"],
                    "pose_metric": rows[0]["pose_metric"],
                    "predictor": predictor,
                    "scenes": len(rows),
                    "spearman_predictor_vs_fscore_improvement": pearson(
                        average_ranks(values), target_ranks
                    ),
                    "uses_ground_truth_native_pose_metric": predictor == "native_pose_value",
                    "uses_ground_truth_intervention": True,
                    "test_time_policy_input": False,
                }
            )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native-rank-rows", type=Path, required=True)
    parser.add_argument("--intervention-case-rows", type=Path, required=True)
    parser.add_argument("--model-map", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--pose-metric", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    model_map = json.loads(args.model_map.read_text(encoding="utf-8"))
    require(isinstance(model_map, dict), "Model map must be a JSON object")
    joined = build_joined_rows(
        read_csv(args.native_rank_rows),
        read_csv(args.intervention_case_rows),
        {str(key): str(value) for key, value in model_map.items()},
        args.dataset,
        args.pose_metric,
    )
    correlations = build_correlations(joined)

    args.output.mkdir(parents=True, exist_ok=False)
    joined_path = args.output / "scene_intervention_predictors.csv"
    correlation_path = args.output / "predictor_gain_correlations.csv"
    write_csv(joined_path, joined)
    write_csv(correlation_path, correlations)
    manifest = {
        "dataset": args.dataset,
        "pose_metric": args.pose_metric,
        "models": sorted({row["intervention_model"] for row in joined}),
        "scenes": sorted({row["scene"] for row in joined}),
        "joined_rows": len(joined),
        "correlation_rows": len(correlations),
        "native_rank_rows_sha256": sha256(args.native_rank_rows),
        "intervention_case_rows_sha256": sha256(args.intervention_case_rows),
        "model_map_sha256": sha256(args.model_map),
        "joined_output_sha256": sha256(joined_path),
        "correlation_output_sha256": sha256(correlation_path),
        "analysis_script_sha256": sha256(Path(__file__)),
        "uses_ground_truth_native_pose_metric": True,
        "uses_ground_truth_intervention": True,
        "test_time_policy_input": False,
        "success_threshold": None,
        "policy": "per-backbone Spearman correlations; all predictors remain separate",
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
