"""Compare which backbone wins two official endpoints on each scene.

The input is the long-form model/scene/metric table produced by the baseline
comparison.  Comparisons remain within one dataset and scene; values from
different benchmark protocols are never aggregated.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr

from scripts.failure_mining.compare_baselines import sha256, write_csv


DIRECTIONS = {"lower_is_worse", "higher_is_worse"}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def utility(value: float, direction: str) -> float:
    require(direction in DIRECTIONS, f"Unknown direction: {direction}")
    return value if direction == "lower_is_worse" else -value


def compare(
    rows: list[dict[str, str]], metric_left: str, metric_right: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    require(rows, "Metric table must be non-empty")
    require(metric_left != metric_right, "Metrics must differ")
    required = {"dataset", "scene", "model", "metric", "value", "direction"}
    missing = required - set(rows[0])
    require(not missing, f"Metric table is missing columns: {sorted(missing)}")

    grouped: dict[tuple[str, str], dict[str, dict[str, dict[str, str]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for row in rows:
        if row["metric"] not in {metric_left, metric_right}:
            continue
        key = (row["dataset"], row["scene"])
        metric_rows = grouped[key][row["metric"]]
        require(row["model"] not in metric_rows, f"Duplicate metric row: {(*key, row['metric'], row['model'])}")
        metric_rows[row["model"]] = row
    require(grouped, "No rows matched the requested metrics")

    scene_rows = []
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for (dataset, scene), metrics in sorted(grouped.items()):
        require(set(metrics) == {metric_left, metric_right}, f"Scene is missing an endpoint: {(dataset, scene)}")
        left, right = metrics[metric_left], metrics[metric_right]
        require(set(left) == set(right), f"Model populations differ: {(dataset, scene)}")
        models = sorted(left)
        require(len(models) >= 3, f"Need at least three models: {(dataset, scene)}")
        left_directions = {row["direction"] for row in left.values()}
        right_directions = {row["direction"] for row in right.values()}
        require(len(left_directions) == 1 and len(right_directions) == 1, f"Metric direction differs: {(dataset, scene)}")
        left_direction = next(iter(left_directions))
        right_direction = next(iter(right_directions))
        left_utilities = [utility(float(left[model]["value"]), left_direction) for model in models]
        right_utilities = [utility(float(right[model]["value"]), right_direction) for model in models]
        require(all(math.isfinite(value) for value in [*left_utilities, *right_utilities]), f"Non-finite metric: {(dataset, scene)}")
        status = "ok"
        rho: float | None = None
        p_value: float | None = None
        if len(set(left_utilities)) < 2 or len(set(right_utilities)) < 2:
            status = "constant_input"
        else:
            result = spearmanr(left_utilities, right_utilities)
            if math.isfinite(result.statistic) and math.isfinite(result.pvalue):
                rho, p_value = float(result.statistic), float(result.pvalue)
            else:
                status = "non_finite_result"
        left_best, right_best = max(left_utilities), max(right_utilities)
        left_winners = [model for model, value in zip(models, left_utilities) if value == left_best]
        right_winners = [model for model, value in zip(models, right_utilities) if value == right_best]
        row = {
            "dataset": dataset,
            "scene": scene,
            "metric_left": metric_left,
            "metric_right": metric_right,
            "models": len(models),
            "direction_left": left_direction,
            "direction_right": right_direction,
            "rank_status": status,
            "spearman_rho": rho,
            "p_value_two_sided": p_value,
            "left_best_models": ";".join(left_winners),
            "right_best_models": ";".join(right_winners),
            "winner_overlap": bool(set(left_winners) & set(right_winners)),
        }
        scene_rows.append(row)
        by_dataset[dataset].append(row)

    summary_rows = []
    for dataset, selected in sorted(by_dataset.items()):
        finite = [float(row["spearman_rho"]) for row in selected if row["rank_status"] == "ok"]
        overlap = sum(bool(row["winner_overlap"]) for row in selected)
        summary_rows.append(
            {
                "dataset": dataset,
                "metric_left": metric_left,
                "metric_right": metric_right,
                "scenes": len(selected),
                "winner_overlap_scenes": overlap,
                "winner_overlap_fraction": overlap / len(selected),
                "finite_rank_scenes": len(finite),
                "spearman_rho_min": min(finite) if finite else None,
                "spearman_rho_median": float(np.median(finite)) if finite else None,
                "spearman_rho_max": max(finite) if finite else None,
            }
        )
    return scene_rows, summary_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--metric-left", required=True)
    parser.add_argument("--metric-right", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), f"Refusing to overwrite output: {args.output}")
    scene_rows, summary_rows = compare(read_csv(args.input), args.metric_left, args.metric_right)
    args.output.mkdir(parents=True)
    scene_path = args.output / "scene_metric_winner_agreement.csv"
    summary_path = args.output / "metric_winner_agreement_summary.csv"
    write_csv(scene_path, scene_rows)
    write_csv(summary_path, summary_rows)
    manifest = {
        "input": str(args.input.resolve()),
        "input_sha256": sha256(args.input),
        "metric_left": args.metric_left,
        "metric_right": args.metric_right,
        "scene_rows": len(scene_rows),
        "summary_rows": len(summary_rows),
        "scene_output_sha256": sha256(scene_path),
        "summary_output_sha256": sha256(summary_path),
        "analysis_script_sha256": sha256(Path(__file__)),
        "failure_labels": None,
        "quality_thresholds": None,
        "composite_score": None,
        "policy": "within-dataset, within-scene endpoint comparison only",
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
