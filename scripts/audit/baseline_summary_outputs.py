"""Audit final per-scene and per-dataset baseline summary tables."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from scripts.audit.analysis_outputs import load_json, require, sha256
from scripts.failure_mining.summarize_baseline_results import (
    COMPLETED_STATES,
    RUNTIME_FIELDS,
)


def read_csv(path: Path) -> list[dict[str, str]]:
    require(path.is_file(), f"Required file is missing: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=0.0, abs_tol=1e-12)


def audit_baseline_summary(
    summary_dir: Path, cross_dir: Path, expected_scenes: int | None = None
) -> dict[str, Any]:
    summary_dir = summary_dir.resolve()
    cross_dir = cross_dir.resolve()
    manifest_path = summary_dir / "manifest.json"
    scene_path = summary_dir / "scene_model_results.csv"
    aggregate_path = summary_dir / "dataset_model_summary.csv"
    cross_manifest_path = cross_dir / "manifest.json"
    manifest = load_json(manifest_path)
    cross = load_json(cross_manifest_path)

    require(manifest.get("cross_manifest_sha256") == sha256(cross_manifest_path), "Cross manifest hash mismatch")
    require(manifest.get("models") == cross.get("models"), "Model population/order differs")
    population = cross.get("scene_population")
    require(manifest.get("scene_population_per_model") == population, "Scene population differs")
    if expected_scenes is not None:
        require(population == expected_scenes, "Unexpected scene population")
    require(manifest.get("failure_labels") is None, "Summary assigns failure labels")
    require(manifest.get("composite_score") is None, "Summary uses a composite score")
    require(manifest.get("scene_output_sha256") == sha256(scene_path), "Scene output hash mismatch")
    require(manifest.get("summary_output_sha256") == sha256(aggregate_path), "Summary output hash mismatch")

    models = manifest["models"]
    population_order = [
        (item["dataset"], item["scene"]) for item in cross.get("population_order", [])
    ]
    require(len(population_order) == population, "Cross population order is incomplete")
    dataset_population = Counter(dataset for dataset, _scene in population_order)
    require(manifest.get("dataset_scene_population") == dict(sorted(dataset_population.items())), "Dataset population differs")
    summary_inputs = manifest.get("inputs", {})
    require(
        isinstance(summary_inputs, dict) and set(summary_inputs) == set(models),
        "Summary input model population differs",
    )
    for model in models:
        source = cross["inputs"][model]
        item = summary_inputs[model]
        require(item["analysis_dir"] == source["analysis_dir"], f"Analysis directory differs: {model}")
        require(item["analysis_manifest_sha256"] == source["analysis_manifest_sha256"], f"Analysis manifest hash differs: {model}")
        require(item["scenes_sha256"] == source["scenes_sha256"], f"Scene source hash differs: {model}")

    scenes = read_csv(scene_path)
    require(len(scenes) == len(models) * population, "Scene output row count mismatch")
    expected_identities = [
        (model, dataset, scene)
        for model in models
        for dataset, scene in population_order
    ]
    identities = [(row["model"], row["dataset"], row["scene"]) for row in scenes]
    require(identities == expected_identities, "Scene output population/order differs")
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in scenes:
        identity = (row["model"], row["dataset"], row["scene"])
        require(row["state"] in COMPLETED_STATES, f"Incomplete state: {identity}")
        require(int(row["views"]) > 0, f"Invalid views: {identity}")
        require(row["run"], f"Missing run: {identity}")
        for field in RUNTIME_FIELDS:
            raw = row.get(field, "")
            require(raw not in {"", None}, f"Missing runtime {field}: {identity}")
        for field, raw in row.items():
            if not (field.startswith("metric.") or field in RUNTIME_FIELDS):
                continue
            if raw in {"", None}:
                continue
            value = float(raw)
            require(math.isfinite(value), f"Non-finite value {field}: {identity}")
            if field in RUNTIME_FIELDS:
                require(value >= 0, f"Negative runtime {field}: {identity}")
            grouped[(row["model"], row["dataset"], field)].append(value)

    aggregates = read_csv(aggregate_path)
    observed = {
        (row["model"], row["dataset"], row["field"]): row for row in aggregates
    }
    require(len(observed) == len(aggregates), "Duplicate aggregate row")
    require(set(observed) == set(grouped), "Aggregate field population differs from scene table")
    for key, values in grouped.items():
        model, dataset, field = key
        require(len(values) == dataset_population[dataset], f"Partial scene field population: {key}")
        row = observed[key]
        expected = {
            "mean": statistics.fmean(values),
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
            "population_stddev": statistics.pstdev(values),
        }
        require(int(row["count"]) == len(values), f"Aggregate count mismatch: {key}")
        require(int(row["dataset_scenes"]) == dataset_population[dataset], f"Dataset count mismatch: {key}")
        for name, value in expected.items():
            require(close(float(row[name]), value), f"Aggregate {name} mismatch: {key}")

    return {
        "schema_version": 1,
        "passed": True,
        "models": models,
        "scene_population_per_model": population,
        "scene_rows": len(scenes),
        "summary_rows": len(aggregates),
        "dataset_scene_population": dict(sorted(dataset_population.items())),
        "summary_manifest_sha256": sha256(manifest_path),
        "scene_output_sha256": sha256(scene_path),
        "aggregate_output_sha256": sha256(aggregate_path),
        "cross_manifest_sha256": sha256(cross_manifest_path),
        "failure_labels": None,
        "composite_score": None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--cross-dir", type=Path, required=True)
    parser.add_argument("--expected-scenes", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit_baseline_summary(args.summary, args.cross_dir, args.expected_scenes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    require(not args.output.exists(), f"Refusing to overwrite existing audit: {args.output}")
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("passed", "models", "scene_rows", "summary_rows")}, indent=2))


if __name__ == "__main__":
    main()
