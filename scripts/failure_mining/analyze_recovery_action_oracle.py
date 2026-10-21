#!/usr/bin/env python3
"""Compare privileged recovery-action headroom on a shared scene population.

The oracle uses official metrics and a ground-truth-extrinsics intervention.
It is an offline action-space diagnostic, not a deployable policy.
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

from scripts.failure_mining.compare_baselines import sha256, write_csv


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def improvement(native: float, candidate: float, direction: str) -> float:
    if direction == "higher_is_better":
        return candidate - native
    if direction == "lower_is_better":
        return native - candidate
    raise ValueError(f"Unknown metric direction: {direction}")


def best_value(values: dict[str, float], direction: str) -> float:
    if direction == "higher_is_better":
        return max(values.values())
    if direction == "lower_is_better":
        return min(values.values())
    raise ValueError(f"Unknown metric direction: {direction}")


def build_action_rows(
    rank_rows: list[dict[str, str]],
    intervention_rows: list[dict[str, str]],
    model_map: dict[str, str],
    dataset: str,
    metric: str,
    intervention_metric: str,
    direction: str,
) -> list[dict[str, Any]]:
    require(rank_rows, "Baseline metric table is empty")
    require(intervention_rows, "Intervention table is empty")
    require(model_map, "Model map is empty")
    require(
        len(model_map) == len(set(model_map.values())),
        "Native model aliases in model map must be unique",
    )

    baseline: dict[tuple[str, str], float] = {}
    models_by_scene: dict[str, set[str]] = defaultdict(set)
    for row in rank_rows:
        if row.get("dataset") != dataset or row.get("metric") != metric:
            continue
        key = (row["model"], row["scene"])
        require(key not in baseline, f"Duplicate baseline row: {key}")
        value = float(row["value"])
        require(math.isfinite(value), f"Non-finite baseline value: {key}")
        baseline[key] = value
        models_by_scene[row["scene"]].add(row["model"])

    expected_models = set(model_map.values())
    intervention_cases: set[tuple[str, str]] = set()
    output: list[dict[str, Any]] = []
    for row in intervention_rows:
        if row.get("metric") != intervention_metric:
            continue
        full_model = row["model"]
        require(full_model in model_map, f"Intervention model is absent from model map: {full_model}")
        source_model = model_map[full_model]
        scene = row["scene"]
        case = (source_model, scene)
        require(case not in intervention_cases, f"Duplicate intervention case: {case}")
        intervention_cases.add(case)
        require(models_by_scene.get(scene) == expected_models, f"Baseline model population differs: {scene}")
        require(case in baseline, f"Missing source baseline row: {case}")

        native = baseline[case]
        official_native = float(row["official_native"])
        require(
            math.isclose(native, official_native, rel_tol=0.0, abs_tol=1e-12),
            f"Native metric disagrees with intervention source: {case}",
        )
        gt_extrinsics = float(row["gt_extrinsics_intervention"])
        require(math.isfinite(gt_extrinsics), f"Non-finite intervention value: {case}")

        other_values = {
            model: baseline[(model, scene)]
            for model in sorted(expected_models - {source_model})
        }
        require(other_values, f"No alternative backbone for: {case}")
        best_other = best_value(other_values, direction)
        best_other_models = sorted(
            model for model, value in other_values.items() if value == best_other
        )
        candidates = {
            "no_action": native,
            "gt_extrinsics": gt_extrinsics,
            "switch_backbone": best_other,
        }
        oracle = best_value(candidates, direction)
        oracle_actions = sorted(
            action for action, value in candidates.items() if value == oracle
        )
        output.append(
            {
                "dataset": dataset,
                "scene": scene,
                "source_model": source_model,
                "metric": metric,
                "direction": direction,
                "native_value": native,
                "gt_extrinsics_value": gt_extrinsics,
                "gt_extrinsics_gain": improvement(native, gt_extrinsics, direction),
                "best_other_backbone_value": best_other,
                "best_other_backbone_models": "|".join(best_other_models),
                "switch_backbone_gain": improvement(native, best_other, direction),
                "oracle_value": oracle,
                "oracle_gain": improvement(native, oracle, direction),
                "oracle_actions": "|".join(oracle_actions),
                "uses_ground_truth_intervention": True,
                "uses_ground_truth_metric_oracle": True,
                "test_time_policy_input": False,
            }
        )

    require(output, "No intervention cases matched the requested metric")
    scenes = {row["scene"] for row in output}
    for scene in scenes:
        require(
            {row["source_model"] for row in output if row["scene"] == scene}
            == expected_models,
            f"Intervention model population differs: {scene}",
        )
    return sorted(output, key=lambda row: (row["source_model"], row["scene"]))


def aggregate_action_rows(action_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    require(action_rows, "Action table is empty")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in action_rows:
        grouped[str(row["source_model"])].append(row)

    output: list[dict[str, Any]] = []
    for model, rows in sorted(grouped.items()):
        gt_gains = [float(row["gt_extrinsics_gain"]) for row in rows]
        switch_gains = [float(row["switch_backbone_gain"]) for row in rows]
        oracle_gains = [float(row["oracle_gain"]) for row in rows]
        output.append(
            {
                "source_model": model,
                "dataset": rows[0]["dataset"],
                "metric": rows[0]["metric"],
                "direction": rows[0]["direction"],
                "scenes": len(rows),
                "native_mean": statistics.fmean(float(row["native_value"]) for row in rows),
                "gt_extrinsics_mean": statistics.fmean(float(row["gt_extrinsics_value"]) for row in rows),
                "gt_extrinsics_mean_gain": statistics.fmean(gt_gains),
                "gt_extrinsics_positive_scenes": sum(value > 0 for value in gt_gains),
                "switch_backbone_mean": statistics.fmean(float(row["best_other_backbone_value"]) for row in rows),
                "switch_backbone_mean_gain": statistics.fmean(switch_gains),
                "switch_backbone_positive_scenes": sum(value > 0 for value in switch_gains),
                "combined_oracle_mean": statistics.fmean(float(row["oracle_value"]) for row in rows),
                "combined_oracle_mean_gain": statistics.fmean(oracle_gains),
                "combined_oracle_positive_scenes": sum(value > 0 for value in oracle_gains),
                "oracle_no_action_memberships": sum("no_action" in row["oracle_actions"].split("|") for row in rows),
                "oracle_gt_extrinsics_memberships": sum("gt_extrinsics" in row["oracle_actions"].split("|") for row in rows),
                "oracle_switch_backbone_memberships": sum("switch_backbone" in row["oracle_actions"].split("|") for row in rows),
                "uses_ground_truth_intervention": True,
                "uses_ground_truth_metric_oracle": True,
                "test_time_policy_input": False,
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric-rank-rows", type=Path, required=True)
    parser.add_argument("--intervention-case-rows", type=Path, required=True)
    parser.add_argument("--model-map", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--metric", required=True)
    parser.add_argument("--intervention-metric", required=True)
    parser.add_argument(
        "--direction",
        choices=("higher_is_better", "lower_is_better"),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    raw_model_map = json.loads(args.model_map.read_text(encoding="utf-8"))
    require(isinstance(raw_model_map, dict), "Model map must be a JSON object")
    action_rows = build_action_rows(
        read_csv(args.metric_rank_rows),
        read_csv(args.intervention_case_rows),
        {str(key): str(value) for key, value in raw_model_map.items()},
        args.dataset,
        args.metric,
        args.intervention_metric,
        args.direction,
    )
    summary_rows = aggregate_action_rows(action_rows)

    args.output.mkdir(parents=True, exist_ok=False)
    action_path = args.output / "scene_action_oracle.csv"
    summary_path = args.output / "model_action_oracle_summary.csv"
    write_csv(action_path, action_rows)
    write_csv(summary_path, summary_rows)
    manifest = {
        "dataset": args.dataset,
        "metric": args.metric,
        "intervention_metric": args.intervention_metric,
        "direction": args.direction,
        "models": sorted({row["source_model"] for row in action_rows}),
        "scenes": sorted({row["scene"] for row in action_rows}),
        "action_rows": len(action_rows),
        "summary_rows": len(summary_rows),
        "metric_rank_rows_sha256": sha256(args.metric_rank_rows),
        "intervention_case_rows_sha256": sha256(args.intervention_case_rows),
        "model_map_sha256": sha256(args.model_map),
        "action_output_sha256": sha256(action_path),
        "summary_output_sha256": sha256(summary_path),
        "analysis_script_sha256": sha256(Path(__file__)),
        "actions": ["no_action", "gt_extrinsics", "switch_backbone"],
        "uses_ground_truth_intervention": True,
        "uses_ground_truth_metric_oracle": True,
        "test_time_policy_input": False,
        "success_threshold": None,
        "policy": "per-scene metric oracle; candidates and endpoints remain separate",
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
