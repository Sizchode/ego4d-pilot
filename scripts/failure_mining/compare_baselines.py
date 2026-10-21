"""Compare failure rankings across reconstruction baselines.

This module consumes the evidence tables emitted by ``analyze_da3_batch.py``.
It never assigns failure labels and never combines unrelated metrics.  Every
comparison is made within one benchmark dataset and one metric, where the
official metric direction has already been recorded by the per-model analysis.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


SceneKey = tuple[str, str]
MetricKey = tuple[str, str, str]


@dataclass(frozen=True)
class AnalysisInput:
    model: str
    directory: Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict]) -> int:
    materialized = list(rows)
    columns = sorted({key for row in materialized for key in row})
    with path.open("w", newline="") as handle:
        if columns:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(materialized)
    return len(materialized)


def parse_analysis(value: str) -> AnalysisInput:
    model, separator, directory = value.partition("=")
    if not separator or not model or not directory:
        raise argparse.ArgumentTypeError("analysis must be MODEL=ANALYSIS_DIRECTORY")
    return AnalysisInput(model=model, directory=Path(directory))


def require_files(item: AnalysisInput) -> None:
    required = ("manifest.json", "scenes.csv", "rankings.csv")
    missing = [name for name in required if not (item.directory / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{item.model} analysis is missing: {', '.join(missing)}")


def scene_key(row: dict[str, str]) -> SceneKey:
    return row["dataset"], row["scene"]


def validate_scene_population(
    analyses: list[AnalysisInput], scenes_by_model: dict[str, list[dict[str, str]]]
) -> list[SceneKey]:
    expected: list[SceneKey] | None = None
    for item in analyses:
        keys = [scene_key(row) for row in scenes_by_model[item.model]]
        if len(keys) != len(set(keys)):
            raise ValueError(f"Duplicate scene identity in {item.model}/scenes.csv")
        if expected is None:
            expected = keys
        elif keys != expected:
            raise ValueError(
                f"Scene population or order differs between {analyses[0].model} and {item.model}"
            )
    return expected or []


def parse_rankings(
    item: AnalysisInput, rows: list[dict[str, str]], scene_population: set[SceneKey]
) -> dict[MetricKey, dict]:
    parsed: dict[MetricKey, dict] = {}
    directions: dict[tuple[str, str], str] = {}
    grouped_ranks: dict[tuple[str, str], list[int]] = {}
    grouped_populations: dict[tuple[str, str], set[int]] = {}
    for row in rows:
        key = (row["dataset"], row["scene"], row["metric"])
        if key in parsed:
            raise ValueError(f"Duplicate ranking row for {item.model}: {key}")
        if (row["dataset"], row["scene"]) not in scene_population:
            raise ValueError(f"Ranking references unknown scene for {item.model}: {key[:2]}")
        rank = int(row["failure_rank"])
        population = int(row["population"])
        if rank < 1 or rank > population:
            raise ValueError(f"Invalid failure rank for {item.model}: {key}")
        group = (row["dataset"], row["metric"])
        direction = row["direction"]
        previous_direction = directions.setdefault(group, direction)
        if previous_direction != direction:
            raise ValueError(f"Inconsistent metric direction for {item.model}: {group}")
        grouped_ranks.setdefault(group, []).append(rank)
        grouped_populations.setdefault(group, set()).add(population)
        parsed[key] = {
            "model": item.model,
            "dataset": row["dataset"],
            "scene": row["scene"],
            "metric": row["metric"],
            "direction": direction,
            "value": float(row["value"]),
            "failure_rank": rank,
            "population": population,
            "worstness_fraction": (
                (population - rank) / (population - 1) if population > 1 else None
            ),
            "run": row["run"],
        }
    for group, ranks in grouped_ranks.items():
        populations = grouped_populations[group]
        if len(populations) != 1:
            raise ValueError(f"Inconsistent population for {item.model}: {group}")
        population = next(iter(populations))
        if len(ranks) != population or sorted(ranks) != list(range(1, population + 1)):
            raise ValueError(f"Rankings are incomplete for {item.model}: {group}")
    return parsed


def pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    centered_left = [value - left_mean for value in left]
    centered_right = [value - right_mean for value in right]
    numerator = sum(a * b for a, b in zip(centered_left, centered_right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in centered_left))
    right_norm = math.sqrt(sum(value * value for value in centered_right))
    denominator = left_norm * right_norm
    return numerator / denominator if denominator else None


def build_execution_rows(
    analyses: list[AnalysisInput], scenes_by_model: dict[str, list[dict[str, str]]]
) -> list[dict]:
    rows = []
    for item in analyses:
        for scene in scenes_by_model[item.model]:
            rows.append(
                {
                    "model": item.model,
                    "dataset": scene["dataset"],
                    "scene": scene["scene"],
                    "state": scene["state"],
                    "exit_code": scene.get("exit_code", ""),
                    "run": scene["run"],
                }
            )
    return rows


def build_pairwise_agreement(
    analyses: list[AnalysisInput], rankings_by_model: dict[str, dict[MetricKey, dict]]
) -> list[dict]:
    groups = sorted(
        {
            (dataset, metric)
            for rankings in rankings_by_model.values()
            for dataset, _scene, metric in rankings
        }
    )
    rows = []
    for dataset, metric in groups:
        for left_index, left in enumerate(analyses):
            for right in analyses[left_index + 1 :]:
                left_rows = {
                    scene: row
                    for (row_dataset, scene, row_metric), row in rankings_by_model[left.model].items()
                    if (row_dataset, row_metric) == (dataset, metric)
                }
                right_rows = {
                    scene: row
                    for (row_dataset, scene, row_metric), row in rankings_by_model[right.model].items()
                    if (row_dataset, row_metric) == (dataset, metric)
                }
                common = sorted(left_rows.keys() & right_rows.keys())
                left_ranks = [float(left_rows[scene]["failure_rank"]) for scene in common]
                right_ranks = [float(right_rows[scene]["failure_rank"]) for scene in common]
                directions = {
                    row["direction"] for row in [*left_rows.values(), *right_rows.values()]
                }
                if len(directions) > 1:
                    raise ValueError(
                        f"Metric direction differs across models: {(dataset, metric)}"
                    )
                rows.append(
                    {
                        "dataset": dataset,
                        "metric": metric,
                        "direction": next(iter(directions)) if directions else "",
                        "model_left": left.model,
                        "model_right": right.model,
                        "common_scenes": len(common),
                        "spearman_failure_rank": pearson(left_ranks, right_ranks),
                    }
                )
    return rows


def build_scene_metric_spread(rank_rows: list[dict]) -> list[dict]:
    grouped: dict[MetricKey, list[dict]] = {}
    for row in rank_rows:
        grouped.setdefault((row["dataset"], row["scene"], row["metric"]), []).append(row)
    rows = []
    for (dataset, scene, metric), values in sorted(grouped.items()):
        fractions = [row["worstness_fraction"] for row in values]
        finite = [value for value in fractions if value is not None and math.isfinite(value)]
        directions = {row["direction"] for row in values}
        if len(directions) != 1:
            raise ValueError(f"Metric direction differs across models: {(dataset, metric)}")
        rows.append(
            {
                "dataset": dataset,
                "scene": scene,
                "metric": metric,
                "direction": next(iter(directions)),
                "models_with_metric": len(values),
                "worstness_min": min(finite) if finite else None,
                "worstness_max": max(finite) if finite else None,
                "worstness_mean": sum(finite) / len(finite) if finite else None,
                "worstness_range": max(finite) - min(finite) if finite else None,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--analysis",
        action="append",
        type=parse_analysis,
        required=True,
        help="Repeat as MODEL=ANALYSIS_DIRECTORY",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    analyses: list[AnalysisInput] = args.analysis
    names = [item.model for item in analyses]
    if len(analyses) < 2:
        parser.error("at least two --analysis inputs are required")
    if len(names) != len(set(names)):
        parser.error("model names must be unique")
    for item in analyses:
        require_files(item)
    args.output.mkdir(parents=True, exist_ok=False)

    scenes_by_model = {
        item.model: read_csv(item.directory / "scenes.csv") for item in analyses
    }
    population = validate_scene_population(analyses, scenes_by_model)
    population_set = set(population)
    rankings_by_model = {
        item.model: parse_rankings(
            item, read_csv(item.directory / "rankings.csv"), population_set
        )
        for item in analyses
    }
    rank_rows = [
        row
        for item in analyses
        for row in rankings_by_model[item.model].values()
    ]

    output_counts = {
        "execution_rows": write_csv(
            args.output / "execution_matrix.csv",
            build_execution_rows(analyses, scenes_by_model),
        ),
        "model_metric_rank_rows": write_csv(
            args.output / "model_metric_rank_rows.csv", rank_rows
        ),
        "scene_metric_spread_rows": write_csv(
            args.output / "scene_metric_spread.csv", build_scene_metric_spread(rank_rows)
        ),
        "pairwise_rank_agreement_rows": write_csv(
            args.output / "pairwise_rank_agreement.csv",
            build_pairwise_agreement(analyses, rankings_by_model),
        ),
    }
    manifest = {
        "models": names,
        "scene_population": len(population),
        "population_order": [
            {"dataset": dataset, "scene": scene} for dataset, scene in population
        ],
        "inputs": {
            item.model: {
                "analysis_dir": str(item.directory.resolve()),
                "analysis_manifest_sha256": sha256(item.directory / "manifest.json"),
                "scenes_sha256": sha256(item.directory / "scenes.csv"),
                "rankings_sha256": sha256(item.directory / "rankings.csv"),
            }
            for item in analyses
        },
        "outputs": output_counts,
        "weighted_composite_failure_score": None,
        "failure_labels": None,
        "comparison_policy": (
            "within-dataset, within-metric ranks only; no cross-metric aggregation"
        ),
        "analysis_script_sha256": sha256(Path(__file__)),
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
