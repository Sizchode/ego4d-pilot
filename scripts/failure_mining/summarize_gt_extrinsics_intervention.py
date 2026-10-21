"""Audit and summarize the ScanNet++ GT-extrinsics intervention.

The intervention itself uses ground truth and is diagnostic oracle evidence.
This module adds no success threshold. It validates the immutable result stream,
reports native-resampling error, and computes paired intervention improvements
relative to that resampling control for every official geometry metric.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from scripts.failure_mining.compare_baselines import sha256, write_csv


METRIC_DIRECTIONS = {
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
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def improvement(control: float, intervention: float, direction: str) -> float:
    if direction == "lower_is_better":
        return control - intervention
    if direction == "higher_is_better":
        return intervention - control
    raise ValueError(f"Unknown metric direction: {direction}")


def aggregate_case_rows(case_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in case_rows:
        grouped[(str(row["model"]), str(row["metric"]))].append(row)
    summary_rows: list[dict[str, Any]] = []
    for (model, metric), items in sorted(grouped.items()):
        require(metric in METRIC_DIRECTIONS, f"Unknown metric in case rows: {metric}")
        improvements = [float(item["improvement_vs_control"]) for item in items]
        control_errors = [float(item["abs_control_delta_vs_official"]) for item in items]
        official_values = [float(item["official_native"]) for item in items]
        control_values = [float(item["native_resample_control"]) for item in items]
        intervention_values = [float(item["gt_extrinsics_intervention"]) for item in items]
        summary_rows.append(
            {
                "model": model,
                "metric": metric,
                "direction": METRIC_DIRECTIONS[metric],
                "cases": len(items),
                "official_native_mean": statistics.fmean(official_values),
                "native_resample_control_mean": statistics.fmean(control_values),
                "gt_extrinsics_intervention_mean": statistics.fmean(intervention_values),
                "mean_improvement_vs_control": statistics.fmean(improvements),
                "median_improvement_vs_control": statistics.median(improvements),
                "min_improvement_vs_control": min(improvements),
                "max_improvement_vs_control": max(improvements),
                "positive_improvement_cases": sum(value > 0 for value in improvements),
                "zero_improvement_cases": sum(value == 0 for value in improvements),
                "negative_improvement_cases": sum(value < 0 for value in improvements),
                "mean_abs_control_delta_vs_official": statistics.fmean(control_errors),
                "max_abs_control_delta_vs_official": max(control_errors),
            }
        )
    return summary_rows


def summarize(
    intervention_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    intervention_dir = intervention_dir.resolve()
    manifest_path = intervention_dir / "manifest.json"
    results_path = intervention_dir / "results.jsonl"
    require(manifest_path.is_file(), f"Missing intervention manifest: {manifest_path}")
    require(results_path.is_file(), f"Missing intervention results: {results_path}")

    manifest = read_json(manifest_path)
    require(
        manifest.get("uses_ground_truth_intervention") is True,
        "Manifest does not identify a ground-truth intervention",
    )
    require(
        manifest.get("test_time_policy_input") is False,
        "Ground-truth intervention must not be marked as a test-time policy input",
    )
    require(
        manifest.get("results_sha256") == sha256(results_path),
        "Intervention result hash does not match manifest",
    )

    rows = read_jsonl(results_path)
    expected_cases = manifest.get("cases")
    require(
        isinstance(expected_cases, int) and expected_cases > 0,
        "Manifest has an invalid case count",
    )
    require(len(rows) == expected_cases, "Intervention case count does not match manifest")

    case_rows: list[dict[str, Any]] = []
    identities: set[tuple[str, str]] = set()
    scene_population: set[str] = set()
    model_population: set[str] = set()

    for row in rows:
        model = row.get("model")
        scene = row.get("scene")
        require(isinstance(model, str) and model, "Case has no model identity")
        require(isinstance(scene, str) and scene, "Case has no scene identity")
        identity = (model, scene)
        require(identity not in identities, f"Duplicate intervention case: {identity}")
        identities.add(identity)
        scene_population.add(scene)
        model_population.add(model)
        require(row.get("dataset") == "scannetpp", f"Unexpected dataset for {identity}")
        require(
            row.get("uses_ground_truth_intervention") is True
            and row.get("test_time_policy_input") is False,
            f"Ground-truth provenance flags are invalid for {identity}",
        )

        official = row.get("official_native_metrics")
        control = row.get("native_resample_metrics")
        intervention = row.get("gt_extrinsics_metrics")
        require(
            all(isinstance(item, dict) for item in (official, control, intervention)),
            f"Metric blocks are missing for {identity}",
        )
        require(
            set(official) == set(control) == set(intervention) == set(METRIC_DIRECTIONS),
            f"Metric population differs for {identity}",
        )
        stored_control_delta = row.get("native_resample_delta_vs_official")
        stored_intervention_delta = row.get("gt_extrinsics_delta_vs_native_resample")
        require(
            isinstance(stored_control_delta, dict)
            and isinstance(stored_intervention_delta, dict),
            f"Stored paired deltas are missing for {identity}",
        )

        for metric, direction in METRIC_DIRECTIONS.items():
            values = [float(official[metric]), float(control[metric]), float(intervention[metric])]
            require(all(math.isfinite(value) for value in values), f"Non-finite metric for {identity}")
            official_value, control_value, intervention_value = values
            control_delta = control_value - official_value
            intervention_delta = intervention_value - control_value
            require(
                math.isclose(
                    float(stored_control_delta[metric]),
                    control_delta,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ),
                f"Control delta mismatch for {identity}, {metric}",
            )
            require(
                math.isclose(
                    float(stored_intervention_delta[metric]),
                    intervention_delta,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ),
                f"Intervention delta mismatch for {identity}, {metric}",
            )
            output = {
                "model": model,
                "scene": scene,
                "metric": metric,
                "direction": direction,
                "official_native": official_value,
                "native_resample_control": control_value,
                "gt_extrinsics_intervention": intervention_value,
                "control_delta_vs_official": control_delta,
                "abs_control_delta_vs_official": abs(control_delta),
                "intervention_delta_vs_control": intervention_delta,
                "improvement_vs_control": improvement(
                    control_value, intervention_value, direction
                ),
                "intervention_delta_vs_official": intervention_value - official_value,
            }
            case_rows.append(output)

    require(
        scene_population == set(manifest.get("scenes", [])),
        "Result scene population differs from manifest",
    )

    summary_rows = aggregate_case_rows(case_rows)

    provenance = {
        "intervention_dir": str(intervention_dir),
        "intervention_manifest_sha256": sha256(manifest_path),
        "intervention_results_sha256": sha256(results_path),
        "cases": len(rows),
        "scenes": sorted(scene_population),
        "models": sorted(model_population),
        "case_metric_rows": len(case_rows),
        "summary_rows": len(summary_rows),
        "uses_ground_truth_intervention": True,
        "test_time_policy_input": False,
        "success_threshold": None,
        "primary_pairing": "GT-extrinsics intervention versus same-seed native mesh resampling control",
    }
    return case_rows, summary_rows, provenance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--intervention-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    case_rows, summary_rows, provenance = summarize(args.intervention_dir)
    case_path = args.output / "case_metric_deltas.csv"
    summary_path = args.output / "model_metric_summary.csv"
    write_csv(case_path, case_rows)
    write_csv(summary_path, summary_rows)
    output_manifest = {
        **provenance,
        "case_metric_output_sha256": sha256(case_path),
        "summary_output_sha256": sha256(summary_path),
        "analysis_script_sha256": sha256(Path(__file__)),
    }
    (args.output / "manifest.json").write_text(
        json.dumps(output_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output_manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
