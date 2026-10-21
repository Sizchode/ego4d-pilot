"""Compare threshold-free visibility diagnostics across baseline models.

The output is descriptive.  Metrics are ranked only within one dataset and one
visibility quantity; no direction, failure threshold, or composite is assigned.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from scripts.failure_mining.compare_baselines import (
    AnalysisInput,
    parse_analysis,
    pearson,
    read_csv,
    sha256,
    write_csv,
)


def require_files(item: AnalysisInput) -> None:
    required = ("manifest.json", "scene_visibility.csv", "failed_runs.json")
    missing = [name for name in required if not (item.directory / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{item.model} visibility analysis is missing: {', '.join(missing)}")
    failures = json.loads((item.directory / "failed_runs.json").read_text())
    if failures:
        raise ValueError(f"{item.model} visibility analysis contains execution failures")


def scene_key(row: dict[str, str]) -> tuple[str, str]:
    return row["dataset"], row["scene"]


def validate_population(
    analyses: list[AnalysisInput], rows_by_model: dict[str, list[dict[str, str]]]
) -> list[tuple[str, str]]:
    expected = None
    for item in analyses:
        keys = [scene_key(row) for row in rows_by_model[item.model]]
        if len(keys) != len(set(keys)):
            raise ValueError(f"Duplicate scene identity in {item.model}")
        if expected is None:
            expected = keys
        elif keys != expected:
            raise ValueError(
                f"Scene population or order differs between {analyses[0].model} and {item.model}"
            )
    return expected or []


def visibility_metrics(rows_by_model: dict[str, list[dict[str, str]]]) -> list[str]:
    column_sets = [
        {key for row in rows for key in row if key.startswith("visibility.")}
        for rows in rows_by_model.values()
    ]
    common = set.intersection(*column_sets)
    metrics = []
    for key in sorted(common):
        values = [row[key] for rows in rows_by_model.values() for row in rows]
        if values and all(value != "" and math.isfinite(float(value)) for value in values):
            metrics.append(key)
    return metrics


def average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        average = (start + 1 + end) / 2
        for index in order[start:end]:
            ranks[index] = average
        start = end
    return ranks


def ranked_rows(
    analyses: list[AnalysisInput], rows_by_model: dict[str, list[dict[str, str]]], metrics: list[str]
) -> list[dict]:
    output = []
    for item in analyses:
        datasets = sorted({row["dataset"] for row in rows_by_model[item.model]})
        for dataset in datasets:
            scenes = [row for row in rows_by_model[item.model] if row["dataset"] == dataset]
            for metric in metrics:
                values = [float(row[metric]) for row in scenes]
                ranks = average_ranks(values)
                population = len(values)
                for row, value, rank in zip(scenes, values, ranks, strict=True):
                    output.append(
                        {
                            "model": item.model,
                            "dataset": dataset,
                            "scene": row["scene"],
                            "metric": metric,
                            "value": value,
                            "ascending_rank": rank,
                            "ascending_rank_fraction": (
                                (rank - 1) / (population - 1) if population > 1 else None
                            ),
                            "population": population,
                        }
                    )
    return output


def pairwise_agreement(analyses: list[AnalysisInput], ranks: list[dict]) -> list[dict]:
    lookup = {
        (row["model"], row["dataset"], row["metric"], row["scene"]): row
        for row in ranks
    }
    groups = sorted({(row["dataset"], row["metric"]) for row in ranks})
    output = []
    for dataset, metric in groups:
        scenes = sorted({row["scene"] for row in ranks if row["dataset"] == dataset and row["metric"] == metric})
        for left_index, left in enumerate(analyses):
            for right in analyses[left_index + 1 :]:
                left_ranks = [lookup[(left.model, dataset, metric, scene)]["ascending_rank"] for scene in scenes]
                right_ranks = [lookup[(right.model, dataset, metric, scene)]["ascending_rank"] for scene in scenes]
                output.append(
                    {
                        "dataset": dataset,
                        "metric": metric,
                        "model_left": left.model,
                        "model_right": right.model,
                        "scenes": len(scenes),
                        "spearman_rank": pearson(left_ranks, right_ranks),
                    }
                )
    return output


def scene_rank_spread(ranks: list[dict]) -> list[dict]:
    grouped = {}
    for row in ranks:
        grouped.setdefault((row["dataset"], row["scene"], row["metric"]), []).append(row)
    output = []
    for (dataset, scene, metric), rows in sorted(grouped.items()):
        fractions = [row["ascending_rank_fraction"] for row in rows]
        finite = [value for value in fractions if value is not None]
        output.append(
            {
                "dataset": dataset,
                "scene": scene,
                "metric": metric,
                "models": len(rows),
                "rank_fraction_min": min(finite) if finite else None,
                "rank_fraction_max": max(finite) if finite else None,
                "rank_fraction_range": max(finite) - min(finite) if finite else None,
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", action="append", type=parse_analysis, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    analyses = args.analysis
    if len(analyses) < 2:
        parser.error("at least two --analysis inputs are required")
    names = [item.model for item in analyses]
    if len(names) != len(set(names)):
        parser.error("model names must be unique")
    for item in analyses:
        require_files(item)
    rows_by_model = {
        item.model: read_csv(item.directory / "scene_visibility.csv") for item in analyses
    }
    population = validate_population(analyses, rows_by_model)
    metrics = visibility_metrics(rows_by_model)
    if not metrics:
        raise ValueError("No common finite visibility metrics")

    args.output.mkdir(parents=True, exist_ok=False)
    model_rows = [
        {"model": item.model, **row}
        for item in analyses
        for row in rows_by_model[item.model]
    ]
    ranks = ranked_rows(analyses, rows_by_model, metrics)
    counts = {
        "model_scene_rows": write_csv(args.output / "model_scene_visibility.csv", model_rows),
        "rank_rows": write_csv(args.output / "model_visibility_ranks.csv", ranks),
        "pairwise_rows": write_csv(
            args.output / "pairwise_visibility_rank_agreement.csv",
            pairwise_agreement(analyses, ranks),
        ),
        "scene_spread_rows": write_csv(
            args.output / "scene_visibility_rank_spread.csv", scene_rank_spread(ranks)
        ),
    }
    manifest = {
        "models": names,
        "scene_population": len(population),
        "metrics": metrics,
        "inputs": {
            item.model: {
                "analysis_dir": str(item.directory.resolve()),
                "manifest_sha256": sha256(item.directory / "manifest.json"),
                "scene_visibility_sha256": sha256(item.directory / "scene_visibility.csv"),
            }
            for item in analyses
        },
        "outputs": counts,
        "metric_direction": None,
        "failure_threshold": None,
        "failure_labels": None,
        "comparison_policy": "within-dataset ranks for each visibility quantity; no cross-metric aggregation",
        "analysis_script_sha256": sha256(Path(__file__)),
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
