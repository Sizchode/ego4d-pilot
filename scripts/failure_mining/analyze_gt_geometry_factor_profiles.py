"""Profile ordered conditional gains from nested GT geometry interventions.

The factors are evaluated in the fixed order extrinsics -> intrinsics -> depth.
Reported gains are conditional on all preceding interventions.  They are not
independent effects or causal attributions, and no success threshold is used.
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


FACTORS = (
    ("extrinsics", "extrinsics_gain_vs_native"),
    ("intrinsics_after_extrinsics", "intrinsics_gain_after_extrinsics"),
    ("depth_after_extrinsics_intrinsics", "depth_gain_after_extrinsics_intrinsics"),
)
STAGES = (
    "official_native",
    "gt_extrinsics",
    "gt_extrinsics_intrinsics",
    "gt_extrinsics_intrinsics_depth",
)
DIRECTIONS = {
    "acc": "lower_is_better",
    "comp": "lower_is_better",
    "overall": "lower_is_better",
    "precision": "higher_is_better",
    "recall": "higher_is_better",
    "fscore": "higher_is_better",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_json(path: Path) -> Any:
    require(path.is_file(), f"Required file is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    require(path.is_file(), f"Required file is missing: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def improvement(before: float, after: float, direction: str) -> float:
    if direction == "higher_is_better":
        return after - before
    if direction == "lower_is_better":
        return before - after
    raise ValueError(f"Unknown direction: {direction}")


def analyze(summary_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    manifest_path = summary_dir / "manifest.json"
    cases_path = summary_dir / "case_metric_stages.csv"
    summary_manifest = read_json(manifest_path)
    require(summary_manifest.get("uses_ground_truth_intervention") is True, "Missing GT provenance")
    require(summary_manifest.get("test_time_policy_input") is False, "GT is marked as policy input")
    require(summary_manifest.get("success_threshold") is None, "Input uses a success threshold")
    require(tuple(summary_manifest.get("stages", [])) == STAGES, "Unexpected stage order")
    require(summary_manifest.get("case_output_sha256") == sha256(cases_path), "Case table hash mismatch")
    models = summary_manifest.get("models")
    scenes = summary_manifest.get("scenes")
    require(isinstance(models, list) and models and len(models) == len(set(models)), "Invalid models")
    require(isinstance(scenes, list) and scenes and len(scenes) == len(set(scenes)), "Invalid scenes")

    source_rows = read_csv(cases_path)
    expected = [
        (model, scene, metric)
        for scene in scenes
        for model in models
        for metric in DIRECTIONS
    ]
    identities = [(row.get("model"), row.get("scene"), row.get("metric")) for row in source_rows]
    require(len(source_rows) == summary_manifest.get("case_metric_rows"), "Case row count mismatch")
    require(set(identities) == set(expected) and len(identities) == len(set(identities)), "Case population is not model × scene × metric")

    profiles = []
    for row in source_rows:
        model, scene, metric = row["model"], row["scene"], row["metric"]
        direction = DIRECTIONS[metric]
        require(row["direction"] == direction, f"Direction mismatch: {(model, scene, metric)}")
        stage_values = {stage: float(row[f"{stage}_value"]) for stage in STAGES}
        require(all(math.isfinite(value) for value in stage_values.values()), f"Non-finite stage value: {(model, scene, metric)}")
        gains = {
            "extrinsics": improvement(stage_values["official_native"], stage_values["gt_extrinsics"], direction),
            "intrinsics_after_extrinsics": improvement(stage_values["gt_extrinsics"], stage_values["gt_extrinsics_intrinsics"], direction),
            "depth_after_extrinsics_intrinsics": improvement(stage_values["gt_extrinsics_intrinsics"], stage_values["gt_extrinsics_intrinsics_depth"], direction),
        }
        for factor, field in FACTORS:
            require(
                math.isclose(float(row[field]), gains[factor], rel_tol=0.0, abs_tol=1e-15),
                f"Stored gain mismatch: {(model, scene, metric, factor)}",
            )
        positive = {factor: value for factor, value in gains.items() if value > 0}
        if positive:
            maximum = max(positive.values())
            best = [factor for factor, value in positive.items() if value == maximum]
            best_label = ";".join(best)
            largest_positive = maximum
        else:
            best = []
            best_label = "none"
            largest_positive = 0.0
        profiles.append(
            {
                "model": model,
                "scene": scene,
                "metric": metric,
                "direction": direction,
                **{f"{stage}_value": stage_values[stage] for stage in STAGES},
                **{f"{factor}_conditional_gain": gains[factor] for factor, _ in FACTORS},
                "best_positive_conditional_factors": best_label,
                "best_positive_factor_count": len(best),
                "largest_positive_conditional_gain": largest_positive,
                "full_gt_input_gain_vs_native": improvement(
                    stage_values["official_native"],
                    stage_values["gt_extrinsics_intrinsics_depth"],
                    direction,
                ),
            }
        )

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in profiles:
        grouped[(row["model"], row["metric"])].append(row)
    aggregate = []
    for (model, metric), rows in sorted(grouped.items()):
        require(len(rows) == len(scenes), f"Incomplete scene population: {(model, metric)}")
        item: dict[str, Any] = {
            "model": model,
            "metric": metric,
            "direction": DIRECTIONS[metric],
            "cases": len(rows),
        }
        for factor, _field in FACTORS:
            values = [float(row[f"{factor}_conditional_gain"]) for row in rows]
            item[f"{factor}_mean_conditional_gain"] = statistics.fmean(values)
            item[f"{factor}_median_conditional_gain"] = statistics.median(values)
            item[f"{factor}_positive_cases"] = sum(value > 0 for value in values)
            item[f"{factor}_zero_cases"] = sum(value == 0 for value in values)
            item[f"{factor}_negative_cases"] = sum(value < 0 for value in values)
            item[f"{factor}_unique_best_positive_cases"] = sum(
                row["best_positive_conditional_factors"] == factor for row in rows
            )
        item["tied_best_positive_cases"] = sum(row["best_positive_factor_count"] > 1 for row in rows)
        item["no_positive_factor_cases"] = sum(row["best_positive_factor_count"] == 0 for row in rows)
        full = [float(row["full_gt_input_gain_vs_native"]) for row in rows]
        item["full_gt_input_mean_gain_vs_native"] = statistics.fmean(full)
        item["full_gt_input_median_gain_vs_native"] = statistics.median(full)
        item["full_gt_input_positive_cases"] = sum(value > 0 for value in full)
        item["full_gt_input_negative_cases"] = sum(value < 0 for value in full)
        aggregate.append(item)

    provenance = {
        "input_summary_manifest_sha256": sha256(manifest_path),
        "input_case_table_sha256": sha256(cases_path),
        "models": models,
        "scenes": scenes,
        "case_metric_profiles": len(profiles),
        "model_metric_rows": len(aggregate),
        "factor_order": [factor for factor, _ in FACTORS],
        "attribution_scope": "ordered nested intervention conditional gains; not independent or causal factor effects",
        "uses_ground_truth_intervention": True,
        "test_time_policy_input": False,
        "success_threshold": None,
    }
    return profiles, aggregate, provenance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), f"Refusing to overwrite output: {args.output}")
    profiles, aggregate, manifest = analyze(args.input)
    args.output.mkdir(parents=True, exist_ok=False)
    profile_path = args.output / "case_metric_factor_profiles.csv"
    aggregate_path = args.output / "model_metric_factor_summary.csv"
    write_csv(profile_path, profiles)
    write_csv(aggregate_path, aggregate)
    manifest.update(
        {
            "profile_output_sha256": sha256(profile_path),
            "aggregate_output_sha256": sha256(aggregate_path),
            "analysis_script_sha256": sha256(Path(__file__)),
        }
    )
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
