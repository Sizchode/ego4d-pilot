#!/usr/bin/env python3
"""Measure per-scene backbone-selection headroom for each official metric.

This is an offline upper bound: the oracle uses the official evaluator to pick
the best backbone independently for every scene and metric.  It does not define
a deployable selector and never combines different metrics.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


MAXIMIZE_DIRECTIONS = {"higher_is_better", "lower_is_worse"}
MINIMIZE_DIRECTIONS = {"lower_is_better", "higher_is_worse"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_rankings(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"model", "dataset", "scene", "metric", "direction", "value"}
    if not rows:
        raise ValueError(f"No ranking rows in {path}")
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"Ranking table is missing columns: {sorted(missing)}")
    return rows


def _maximize(direction: str) -> bool:
    if direction in MAXIMIZE_DIRECTIONS:
        return True
    if direction in MINIMIZE_DIRECTIONS:
        return False
    raise ValueError(f"Unknown metric direction: {direction}")


def analyze(rows: list[dict[str, str]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    identities: set[tuple[str, str, str, str]] = set()
    for row in rows:
        identity = (row["model"], row["dataset"], row["scene"], row["metric"])
        if identity in identities:
            raise ValueError(f"Duplicate model/scene/metric row: {identity}")
        identities.add(identity)
        value = float(row["value"])
        if not math.isfinite(value):
            raise ValueError(f"Non-finite metric value: {identity}")
        groups[(row["dataset"], row["metric"])].append(row)

    scene_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for (dataset, metric), group in sorted(groups.items()):
        directions = {row["direction"] for row in group}
        if len(directions) != 1:
            raise ValueError(f"Metric direction differs within {(dataset, metric)}")
        direction = next(iter(directions))
        maximize = _maximize(direction)

        by_scene: dict[str, dict[str, float]] = defaultdict(dict)
        for row in group:
            by_scene[row["scene"]][row["model"]] = float(row["value"])
        model_sets = {tuple(sorted(values)) for values in by_scene.values()}
        if len(model_sets) != 1:
            raise ValueError(f"Model population differs across scenes for {(dataset, metric)}")
        models = list(next(iter(model_sets)))
        if len(models) < 2:
            raise ValueError(f"Backbone oracle requires at least two models for {(dataset, metric)}")

        means = {
            model: statistics.fmean(by_scene[scene][model] for scene in by_scene)
            for model in models
        }
        fixed_value = (max if maximize else min)(means.values())
        fixed_models = sorted(model for model, value in means.items() if value == fixed_value)
        if len(fixed_models) != 1:
            raise ValueError(
                f"Best fixed model is tied for {(dataset, metric)}: {fixed_models}"
            )
        fixed_model = fixed_models[0]

        gains: list[float] = []
        oracle_values: list[float] = []
        fixed_wins = 0
        for scene, values in sorted(by_scene.items()):
            oracle_value = (max if maximize else min)(values.values())
            oracle_models = sorted(
                model for model, value in values.items() if value == oracle_value
            )
            fixed_scene_value = values[fixed_model]
            gain = (
                oracle_value - fixed_scene_value
                if maximize
                else fixed_scene_value - oracle_value
            )
            if gain < -1e-12:
                raise AssertionError(f"Negative oracle gain for {(dataset, scene, metric)}")
            gain = max(gain, 0.0)
            gains.append(gain)
            oracle_values.append(oracle_value)
            if fixed_model in oracle_models:
                fixed_wins += 1
            scene_rows.append(
                {
                    "dataset": dataset,
                    "scene": scene,
                    "metric": metric,
                    "direction": direction,
                    "models": len(models),
                    "best_fixed_model": fixed_model,
                    "best_fixed_value": fixed_scene_value,
                    "oracle_models": "|".join(oracle_models),
                    "oracle_value": oracle_value,
                    "oracle_gain": gain,
                }
            )

        summary_rows.append(
            {
                "dataset": dataset,
                "metric": metric,
                "direction": direction,
                "scenes": len(by_scene),
                "models": len(models),
                "best_fixed_model": fixed_model,
                "best_fixed_mean": means[fixed_model],
                "oracle_mean": statistics.fmean(oracle_values),
                "mean_oracle_gain": statistics.fmean(gains),
                "median_oracle_gain": statistics.median(gains),
                "positive_gain_scenes": sum(gain > 1e-12 for gain in gains),
                "fixed_model_winner_scenes": fixed_wins,
                "max_oracle_gain": max(gains),
            }
        )
    return scene_rows, summary_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rankings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    scene_rows, summary_rows = analyze(read_rankings(args.rankings))
    scene_path = args.output / "oracle_scene_headroom.csv"
    summary_path = args.output / "oracle_summary.csv"
    write_csv(scene_path, scene_rows)
    write_csv(summary_path, summary_rows)
    manifest = {
        "rankings": str(args.rankings.resolve()),
        "rankings_sha256": sha256(args.rankings),
        "analysis_script_sha256": sha256(Path(__file__)),
        "scene_rows": len(scene_rows),
        "summary_rows": len(summary_rows),
        "scene_output_sha256": sha256(scene_path),
        "summary_output_sha256": sha256(summary_path),
        "failure_labels": None,
        "composite_score": None,
        "policy": "per-dataset, per-metric GT oracle; no cross-metric aggregation",
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
