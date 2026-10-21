"""Select exact extrema where bidirectional residual misranks GT depth error."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import rankdata

from scripts.failure_mining.compare_baselines import sha256, write_csv


TARGET = "depth.absolute_relative.mean"
RESIDUAL_FIELDS = (
    "visibility.outgoing.source_depth_not_behind.absolute_relative_z_nearest.p95.mean",
    "visibility.incoming.source_depth_not_behind.absolute_relative_z_nearest.p95.mean",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_csv(path: Path) -> list[dict[str, str]]:
    require(path.is_file(), f"Required file is missing: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def finite(value: str | None) -> float | None:
    if value in {None, ""}:
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) else None


def build_profiles(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["scene"])].append(row)
    require(grouped, "Merged view table is empty")
    output = []
    for (dataset, scene), scene_rows in sorted(grouped.items()):
        provisional = []
        for row in scene_rows:
            target = finite(row.get(TARGET))
            residual_values = [value for field in RESIDUAL_FIELDS if (value := finite(row.get(field))) is not None]
            if target is None or not residual_values:
                continue
            provisional.append(
                {
                    "dataset": dataset,
                    "scene": scene,
                    "view_index": int(row["view_index"]),
                    TARGET: target,
                    "bidirectional_residual_mean": float(np.mean(residual_values)),
                    "residual_directions_available": len(residual_values),
                }
            )
        require(len(provisional) >= 3, f"Too few complete views: {scene}")
        population = len(provisional)
        error_ranks = rankdata([row[TARGET] for row in provisional], method="average")
        residual_ranks = rankdata([row["bidirectional_residual_mean"] for row in provisional], method="average")
        for row, error_rank, residual_rank in zip(provisional, error_ranks, residual_ranks, strict=True):
            error_worstness = float((error_rank - 1) / (population - 1))
            residual_worstness = float((residual_rank - 1) / (population - 1))
            output.append(
                {
                    **row,
                    "complete_view_population": population,
                    "gt_error_worstness_fraction": error_worstness,
                    "residual_worstness_fraction": residual_worstness,
                    "gt_error_minus_residual_worstness": error_worstness - residual_worstness,
                }
            )
    return output


def select_extremes(profiles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in profiles:
        grouped[(row["dataset"], row["scene"])].append(row)
    require(grouped, "Profile table is empty")
    output = []
    for key, rows in sorted(grouped.items()):
        underestimated = max(
            rows,
            key=lambda row: (row["gt_error_minus_residual_worstness"], -row["view_index"]),
        )
        overestimated = min(
            rows,
            key=lambda row: (row["gt_error_minus_residual_worstness"], row["view_index"]),
        )
        output.extend(
            [
                {**underestimated, "role": "residual_underestimates_gt_error"},
                {**overestimated, "role": "residual_overestimates_gt_error"},
            ]
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--view-evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), f"Refusing to overwrite output: {args.output}")
    source_manifest_path = args.view_evidence / "manifest.json"
    merged_path = args.view_evidence / "merged_view_evidence.csv"
    manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    require(manifest.get("merged_output_sha256") == sha256(merged_path), "Merged view table hash differs")
    require(manifest.get("uses_ground_truth_targets") is True, "Missing GT-target provenance")
    require(manifest.get("test_time_policy_input") is False, "GT target is marked as policy input")
    profiles = build_profiles(read_csv(merged_path))
    selections = select_extremes(profiles)
    args.output.mkdir(parents=True, exist_ok=False)
    profile_path = args.output / "view_residual_rank_profiles.csv"
    selection_path = args.output / "residual_rank_disagreement_extremes.csv"
    write_csv(profile_path, profiles)
    write_csv(selection_path, selections)
    output_manifest = {
        "target": TARGET,
        "residual_fields": list(RESIDUAL_FIELDS),
        "residual_aggregation": "arithmetic mean over available incoming/outgoing fields",
        "rank_policy": "within-scene average ranks normalized to [0,1], larger is worse",
        "selection_rule": "exact extrema of GT-error worstness minus residual worstness; ties use lowest view index",
        "scenes": len({(row["dataset"], row["scene"]) for row in profiles}),
        "profile_rows": len(profiles),
        "selected_rows": len(selections),
        "source_manifest_sha256": sha256(source_manifest_path),
        "source_merged_sha256": sha256(merged_path),
        "profile_output_sha256": sha256(profile_path),
        "selection_output_sha256": sha256(selection_path),
        "analysis_script_sha256": sha256(Path(__file__)),
        "uses_ground_truth_analysis": True,
        "test_time_policy_input": False,
        "failure_labels": None,
        "quality_thresholds": None,
        "top_k": None,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(output_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(output_manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
