"""Create final per-scene and per-dataset tables from audited baseline analyses."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from scripts.failure_mining.compare_baselines import sha256, write_csv


RUNTIME_FIELDS = (
    "elapsed_seconds",
    "inference_seconds",
    "evaluation_seconds",
    "diagnostics_seconds",
    "peak_cuda_allocated_bytes",
    "peak_cuda_reserved_bytes",
)
IDENTITY_FIELDS = ("model", "dataset", "scene", "state", "run", "views")
COMPLETED_STATES = {"completed", "reused_verified_run"}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def summarize(cross_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    cross_dir = cross_dir.resolve()
    manifest_path = cross_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    models = manifest.get("models")
    require(isinstance(models, list) and len(models) >= 2, "Cross manifest has fewer than two models")
    require(len(models) == len(set(models)), "Cross manifest has duplicate models")
    inputs = manifest.get("inputs", {})
    require(list(inputs) == models, "Cross input order differs from model order")
    expected_population = manifest.get("scene_population")
    require(isinstance(expected_population, int) and expected_population > 0, "Invalid cross scene population")
    manifest_order = [
        (item["dataset"], item["scene"])
        for item in manifest.get("population_order", [])
    ]
    require(len(manifest_order) == expected_population, "Cross population order is incomplete")

    scene_rows = []
    input_hashes = {}
    expected_order = None
    for model in models:
        item = inputs[model]
        analysis_dir = Path(item["analysis_dir"])
        scenes_path = analysis_dir / "scenes.csv"
        require(sha256(scenes_path) == item["scenes_sha256"], f"Scene-table hash mismatch for {model}")
        rows = read_csv(scenes_path)
        require(len(rows) == expected_population, f"Scene population mismatch for {model}")
        required_columns = {"dataset", "scene", "state", "run"}
        require(
            bool(rows) and required_columns <= set(rows[0]),
            f"Scene table is missing provenance columns for {model}: "
            f"{sorted(required_columns - set(rows[0]) if rows else required_columns)}",
        )
        order = [(row["dataset"], row["scene"]) for row in rows]
        require(len(order) == len(set(order)), f"Duplicate scene identity for {model}")
        if expected_order is None:
            expected_order = order
        else:
            require(order == expected_order, f"Scene order differs for {model}")
        require(order == manifest_order, f"Scene order differs from cross manifest for {model}")
        metric_fields = sorted({key for row in rows for key in row if key.startswith("metric.")})
        selected_fields = [*RUNTIME_FIELDS, *metric_fields]
        for row in rows:
            identity = (model, row["dataset"], row["scene"])
            require(row["state"] in COMPLETED_STATES, f"Incomplete baseline state: {identity}")
            require(int(row.get("views", "")) > 0, f"Invalid view count: {identity}")
            for field in RUNTIME_FIELDS:
                raw = row.get(field, "")
                require(raw not in {"", None}, f"Missing runtime field {field}: {identity}")
                value = float(raw)
                require(math.isfinite(value) and value >= 0, f"Invalid runtime field {field}: {identity}")
            for field in metric_fields:
                raw = row.get(field, "")
                if raw not in {"", None}:
                    require(math.isfinite(float(raw)), f"Non-finite metric {field}: {identity}")
            output = {
                "model": model,
                "dataset": row["dataset"],
                "scene": row["scene"],
                "state": row["state"],
                "run": row["run"],
                "views": row.get("views", ""),
            }
            output.update({field: row.get(field, "") for field in selected_fields})
            scene_rows.append(output)
        input_hashes[model] = {
            "analysis_dir": str(analysis_dir.resolve()),
            "analysis_manifest_sha256": item["analysis_manifest_sha256"],
            "scenes_sha256": item["scenes_sha256"],
            "metric_fields": metric_fields,
        }

    dataset_population = Counter(dataset for dataset, _scene in (expected_order or []))
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in scene_rows:
        for field, raw in row.items():
            if not (field.startswith("metric.") or field in RUNTIME_FIELDS):
                continue
            if raw in {"", None}:
                continue
            value = float(raw)
            if math.isfinite(value):
                grouped[(row["model"], row["dataset"], field)].append(value)
    summary_rows = []
    for (model, dataset, field), values in sorted(grouped.items()):
        require(
            len(values) == dataset_population[dataset],
            f"Partial field population for {(model, dataset, field)}",
        )
        summary_rows.append(
            {
                "model": model,
                "dataset": dataset,
                "field": field,
                "count": len(values),
                "dataset_scenes": dataset_population[dataset],
                "mean": statistics.fmean(values),
                "median": statistics.median(values),
                "min": min(values),
                "max": max(values),
                "population_stddev": statistics.pstdev(values),
            }
        )
    metric_sets: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in summary_rows:
        if row["field"].startswith("metric."):
            metric_sets[(row["model"], row["dataset"])].add(row["field"])
    for dataset in dataset_population:
        sets = {tuple(sorted(metric_sets[(model, dataset)])) for model in models}
        require(len(sets) == 1, f"Official metric fields differ across models for {dataset}")
    provenance = {
        "models": models,
        "scene_population_per_model": expected_population,
        "dataset_scene_population": dict(sorted(dataset_population.items())),
        "cross_manifest_sha256": sha256(manifest_path),
        "inputs": input_hashes,
    }
    return scene_rows, summary_rows, provenance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cross-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    scenes, summaries, provenance = summarize(args.cross_dir)
    scene_path = args.output / "scene_model_results.csv"
    summary_path = args.output / "dataset_model_summary.csv"
    write_csv(scene_path, scenes)
    write_csv(summary_path, summaries)
    manifest = {
        **provenance,
        "scene_rows": len(scenes),
        "summary_rows": len(summaries),
        "scene_output_sha256": sha256(scene_path),
        "summary_output_sha256": sha256(summary_path),
        "analysis_script_sha256": sha256(Path(__file__)),
        "failure_labels": None,
        "composite_score": None,
        "policy": "per-model, per-dataset, per-field descriptive statistics only",
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
