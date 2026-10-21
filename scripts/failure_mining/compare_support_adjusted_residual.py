"""Evaluate a support-adjusted residual score against offline GT depth error.

Within each scene, the residual and support values are converted to average
ranks.  Ordinary least squares removes the rank-linear component of residual
that is explained by support.  Fitting this calibration uses no GT target; GT
depth error is used only for offline evaluation of the resulting score.
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
from scipy.stats import rankdata, spearmanr, wilcoxon

from scripts.failure_mining.compare_baselines import sha256, write_csv


TARGET = "depth.absolute_relative.mean"
RESIDUAL_FIELDS = (
    "visibility.outgoing.source_depth_not_behind.absolute_relative_z_nearest.p95.mean",
    "visibility.incoming.source_depth_not_behind.absolute_relative_z_nearest.p95.mean",
)
SUPPORT_FIELDS = (
    "visibility.outgoing.source_depth_not_behind_ratio.mean",
    "visibility.incoming.source_depth_not_behind_ratio.mean",
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


def scene_scores(rows: list[dict[str, str]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["scene"])].append(row)
    require(grouped, "Merged view table is empty")

    view_output: list[dict[str, Any]] = []
    scene_output: list[dict[str, Any]] = []
    for (dataset, scene), scene_rows in sorted(grouped.items()):
        complete = []
        for row in scene_rows:
            target = finite(row.get(TARGET))
            residuals = [value for field in RESIDUAL_FIELDS if (value := finite(row.get(field))) is not None]
            supports = [value for field in SUPPORT_FIELDS if (value := finite(row.get(field))) is not None]
            if target is None or not residuals or not supports:
                continue
            complete.append(
                {
                    "dataset": dataset,
                    "scene": scene,
                    "view_index": int(row["view_index"]),
                    "target": target,
                    "residual": statistics.fmean(residuals),
                    "support": statistics.fmean(supports),
                    "residual_directions": len(residuals),
                    "support_directions": len(supports),
                }
            )
        require(len(complete) >= 3, f"Too few complete views: {(dataset, scene)}")
        target_rank = rankdata([row["target"] for row in complete], method="average")
        residual_rank = rankdata([row["residual"] for row in complete], method="average")
        support_rank = rankdata([row["support"] for row in complete], method="average")
        design = np.column_stack((np.ones(len(complete)), support_rank))
        coefficients = np.linalg.lstsq(design, residual_rank, rcond=None)[0]
        adjusted_score = residual_rank - design @ coefficients
        raw = spearmanr(residual_rank, target_rank)
        adjusted = spearmanr(adjusted_score, target_rank)
        require(
            math.isfinite(raw.statistic) and math.isfinite(adjusted.statistic),
            f"Undefined scene correlation: {(dataset, scene)}",
        )
        for row, target_r, residual_r, support_r, score in zip(
            complete, target_rank, residual_rank, support_rank, adjusted_score, strict=True
        ):
            view_output.append(
                {
                    **row,
                    "target_average_rank": float(target_r),
                    "residual_average_rank": float(residual_r),
                    "support_average_rank": float(support_r),
                    "support_adjusted_residual_score": float(score),
                }
            )
        scene_output.append(
            {
                "dataset": dataset,
                "scene": scene,
                "views": len(complete),
                "support_intercept": float(coefficients[0]),
                "support_rank_coefficient": float(coefficients[1]),
                "raw_residual_spearman_rho": float(raw.statistic),
                "support_adjusted_residual_spearman_rho": float(adjusted.statistic),
                "adjusted_minus_raw_rho": float(adjusted.statistic - raw.statistic),
            }
        )
    return view_output, scene_output


def aggregate(scene_rows: list[dict[str, Any]]) -> dict[str, Any]:
    require(scene_rows, "Scene score table is empty")
    raw = [float(row["raw_residual_spearman_rho"]) for row in scene_rows]
    adjusted = [float(row["support_adjusted_residual_spearman_rho"]) for row in scene_rows]
    differences = np.asarray([new - old for old, new in zip(raw, adjusted, strict=True)])
    nonzero = differences[differences != 0]
    p_value = 1.0 if nonzero.size == 0 else float(wilcoxon(nonzero, alternative="two-sided").pvalue)
    return {
        "scenes": len(scene_rows),
        "raw_mean_scene_spearman_rho": statistics.fmean(raw),
        "raw_median_scene_spearman_rho": statistics.median(raw),
        "raw_positive_scenes": sum(value > 0 for value in raw),
        "adjusted_mean_scene_spearman_rho": statistics.fmean(adjusted),
        "adjusted_median_scene_spearman_rho": statistics.median(adjusted),
        "adjusted_positive_scenes": sum(value > 0 for value in adjusted),
        "adjusted_better_scenes": int(np.sum(differences > 0)),
        "adjusted_equal_scenes": int(np.sum(differences == 0)),
        "adjusted_worse_scenes": int(np.sum(differences < 0)),
        "mean_adjusted_minus_raw_rho": float(np.mean(differences)),
        "median_adjusted_minus_raw_rho": float(np.median(differences)),
        "paired_wilcoxon_p_value_two_sided": p_value,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--view-evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), f"Refusing to overwrite output: {args.output}")
    manifest_path = args.view_evidence / "manifest.json"
    evidence_path = args.view_evidence / "merged_view_evidence.csv"
    source_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(source_manifest.get("merged_output_sha256") == sha256(evidence_path), "Evidence hash differs")
    require(source_manifest.get("uses_ground_truth_targets") is True, "Missing offline-GT provenance")
    require(source_manifest.get("test_time_policy_input") is False, "GT is marked as a policy input")

    views, scenes = scene_scores(read_csv(evidence_path))
    result = aggregate(scenes)
    args.output.mkdir(parents=True, exist_ok=False)
    view_path = args.output / "support_adjusted_view_scores.csv"
    scene_path = args.output / "support_adjusted_scene_correlations.csv"
    aggregate_path = args.output / "aggregate.json"
    write_csv(view_path, views)
    write_csv(scene_path, scenes)
    aggregate_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {
        "target": TARGET,
        "residual_fields": list(RESIDUAL_FIELDS),
        "support_fields": list(SUPPORT_FIELDS),
        "score_definition": "within-scene residual average-rank minus its OLS projection on support average-rank and an intercept",
        "score_fit_uses_ground_truth": False,
        "ground_truth_use": "offline evaluation only",
        "statistical_unit": "scene",
        "scenes": len(scenes),
        "view_rows": len(views),
        "source_manifest_sha256": sha256(manifest_path),
        "source_evidence_sha256": sha256(evidence_path),
        "view_output_sha256": sha256(view_path),
        "scene_output_sha256": sha256(scene_path),
        "aggregate_output_sha256": sha256(aggregate_path),
        "analysis_script_sha256": sha256(Path(__file__)),
        "test_time_policy_input": False,
        "quality_thresholds": None,
        "top_k": None,
        "development_status": "exploratory; freeze on an independent development split before policy evaluation",
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
