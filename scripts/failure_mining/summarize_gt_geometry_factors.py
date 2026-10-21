"""Audit and summarize nested ScanNet++ ground-truth geometry interventions."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from scripts.failure_mining.compare_baselines import sha256, write_csv
from scripts.failure_mining.summarize_gt_extrinsics_intervention import (
    METRIC_DIRECTIONS,
    improvement,
    read_json,
    read_jsonl,
    require,
)


STAGES = (
    "official_native",
    "gt_extrinsics",
    "gt_extrinsics_intrinsics",
    "gt_extrinsics_intrinsics_depth",
)


def summarize(input_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    manifest_path = input_dir / "manifest.json"
    results_path = input_dir / "results.jsonl"
    require(manifest_path.is_file() and results_path.is_file(), "Factor intervention is incomplete")
    manifest = read_json(manifest_path)
    require(manifest.get("uses_ground_truth_intervention") is True, "Missing GT provenance")
    require(manifest.get("test_time_policy_input") is False, "GT intervention is policy input")
    require(tuple(manifest.get("stages", [])) == STAGES, "Unexpected intervention stages")
    require(manifest.get("results_sha256") == sha256(results_path), "Result hash mismatch")
    rows = read_jsonl(results_path)
    models = manifest.get("models")
    scenes = manifest.get("scenes")
    require(
        isinstance(models, list)
        and models
        and all(isinstance(model, str) and model for model in models)
        and len(models) == len(set(models)),
        "Invalid model population",
    )
    require(
        isinstance(scenes, list)
        and scenes
        and all(isinstance(scene, str) and scene for scene in scenes)
        and len(scenes) == len(set(scenes)),
        "Invalid scene population",
    )
    expected_identities = [(model, scene) for scene in scenes for model in models]
    require(manifest.get("cases") == len(expected_identities), "Manifest cases are not model × scene")
    require(len(rows) == len(expected_identities), "Case count mismatch")
    require(
        [(row.get("model"), row.get("scene")) for row in rows] == expected_identities,
        "Result population/order differs from manifest",
    )

    case_rows: list[dict[str, Any]] = []
    identities: set[tuple[str, str]] = set()
    for row in rows:
        identity = (row["model"], row["scene"])
        require(identity not in identities, f"Duplicate case: {identity}")
        identities.add(identity)
        require(tuple(row.get("stages", [])) == STAGES, f"Stage mismatch: {identity}")
        blocks = {
            "official_native": row["official_native_metrics"],
            "gt_extrinsics": row["gt_extrinsics_metrics"],
            "gt_extrinsics_intrinsics": row["gt_extrinsics_intrinsics_metrics"],
            "gt_extrinsics_intrinsics_depth": row[
                "gt_extrinsics_intrinsics_depth_metrics"
            ],
        }
        require(all(set(block) == set(METRIC_DIRECTIONS) for block in blocks.values()), f"Metric population differs: {identity}")
        for metric, direction in METRIC_DIRECTIONS.items():
            values = {stage: float(blocks[stage][metric]) for stage in STAGES}
            require(all(math.isfinite(value) for value in values.values()), f"Non-finite metric: {identity}, {metric}")
            case_rows.append(
                {
                    "model": row["model"],
                    "scene": row["scene"],
                    "metric": metric,
                    "direction": direction,
                    **{f"{stage}_value": values[stage] for stage in STAGES},
                    "extrinsics_gain_vs_native": improvement(
                        values["official_native"], values["gt_extrinsics"], direction
                    ),
                    "intrinsics_gain_after_extrinsics": improvement(
                        values["gt_extrinsics"],
                        values["gt_extrinsics_intrinsics"],
                        direction,
                    ),
                    "depth_gain_after_extrinsics_intrinsics": improvement(
                        values["gt_extrinsics_intrinsics"],
                        values["gt_extrinsics_intrinsics_depth"],
                        direction,
                    ),
                    "full_gt_input_gain_vs_native": improvement(
                        values["official_native"],
                        values["gt_extrinsics_intrinsics_depth"],
                        direction,
                    ),
                }
            )

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in case_rows:
        grouped[(row["model"], row["metric"])].append(row)
    summary_rows = []
    for (model, metric), items in sorted(grouped.items()):
        summary: dict[str, Any] = {
            "model": model,
            "metric": metric,
            "direction": items[0]["direction"],
            "cases": len(items),
        }
        for stage in STAGES:
            summary[f"{stage}_mean"] = statistics.fmean(
                float(item[f"{stage}_value"]) for item in items
            )
        for field in (
            "extrinsics_gain_vs_native",
            "intrinsics_gain_after_extrinsics",
            "depth_gain_after_extrinsics_intrinsics",
            "full_gt_input_gain_vs_native",
        ):
            values = [float(item[field]) for item in items]
            summary[f"{field}_mean"] = statistics.fmean(values)
            summary[f"{field}_median"] = statistics.median(values)
            summary[f"{field}_positive_cases"] = sum(value > 0 for value in values)
            summary[f"{field}_negative_cases"] = sum(value < 0 for value in values)
        summary_rows.append(summary)

    provenance = {
        "input_manifest_sha256": sha256(manifest_path),
        "input_results_sha256": sha256(results_path),
        "cases": len(rows),
        "case_metric_rows": len(case_rows),
        "summary_rows": len(summary_rows),
        "models": models,
        "scenes": scenes,
        "stages": list(STAGES),
        "uses_ground_truth_intervention": True,
        "test_time_policy_input": False,
        "success_threshold": None,
    }
    return case_rows, summary_rows, provenance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), f"Refusing to overwrite output: {args.output}")
    case_rows, summary_rows, provenance = summarize(args.input)
    args.output.mkdir(parents=True, exist_ok=False)
    case_path = args.output / "case_metric_stages.csv"
    summary_path = args.output / "model_metric_stage_summary.csv"
    write_csv(case_path, case_rows)
    write_csv(summary_path, summary_rows)
    manifest = {
        **provenance,
        "case_output_sha256": sha256(case_path),
        "summary_output_sha256": sha256(summary_path),
        "analysis_script_sha256": sha256(Path(__file__)),
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
