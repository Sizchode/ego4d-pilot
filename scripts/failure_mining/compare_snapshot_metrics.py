"""Compare official metrics on the common population of an immutable snapshot."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from scripts.failure_mining.analyze_backbone_oracle import analyze as analyze_oracle
from scripts.failure_mining.compare_baselines import pearson, sha256, write_csv


DIRECTIONS = {
    "pose.auc03": "lower_is_worse",
    "pose.auc05": "lower_is_worse",
    "pose.auc15": "lower_is_worse",
    "pose.auc30": "lower_is_worse",
    "recon_unposed.acc": "higher_is_worse",
    "recon_unposed.comp": "higher_is_worse",
    "recon_unposed.overall": "higher_is_worse",
    "recon_unposed.precision": "lower_is_worse",
    "recon_unposed.recall": "lower_is_worse",
    "recon_unposed.fscore": "lower_is_worse",
}


def flatten_metrics(metrics: dict[str, dict[str, Any]]) -> dict[str, float]:
    return {
        f"{mode}.{name}": float(value)
        for mode, values in metrics.items()
        for name, value in values.items()
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def average_ranks(values: list[float], reverse: bool) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index], reverse=reverse)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and values[order[stop]] == values[order[start]]:
            stop += 1
        rank = ((start + 1) + stop) / 2
        for position in order[start:stop]:
            ranks[position] = rank
        start = stop
    return ranks


def compare(rows: list[dict[str, Any]], models: list[str]) -> dict[str, list[dict[str, Any]]]:
    if len(models) < 2 or len(models) != len(set(models)):
        raise ValueError("At least two unique models are required")
    lookup = {}
    order_by_model_dataset: dict[tuple[str, str], list[str]] = defaultdict(list)
    for row in rows:
        key = (row["model"], row["dataset"], row["scene"])
        if key in lookup:
            raise ValueError(f"Duplicate snapshot metric row: {key}")
        if row["model"] not in models:
            raise ValueError(f"Unknown model in snapshot: {row['model']}")
        lookup[key] = flatten_metrics(row["metrics"])
        order_by_model_dataset[(row["model"], row["dataset"])].append(row["scene"])

    datasets = sorted({dataset for _model, dataset in order_by_model_dataset})
    common_populations = []
    rank_rows = []
    for dataset in datasets:
        scene_sets = [set(order_by_model_dataset.get((model, dataset), [])) for model in models]
        common = set.intersection(*scene_sets) if scene_sets else set()
        if not common:
            continue
        ordered_common = [scene for scene in order_by_model_dataset[(models[0], dataset)] if scene in common]
        for model in models[1:]:
            model_order = [scene for scene in order_by_model_dataset[(model, dataset)] if scene in common]
            if model_order != ordered_common:
                raise ValueError(f"Common scene order differs for {(model, dataset)}")
        common_populations.append({"dataset": dataset, "common_scenes": len(ordered_common)})
        metric_sets = []
        for model in models:
            for scene in ordered_common:
                metric_sets.append(set(lookup[(model, dataset, scene)]))
        common_metrics = sorted(set.intersection(*metric_sets))
        unknown = set(common_metrics) - set(DIRECTIONS)
        if unknown:
            raise ValueError(f"Unknown official metric directions: {sorted(unknown)}")
        for model in models:
            for metric in common_metrics:
                values = [lookup[(model, dataset, scene)][metric] for scene in ordered_common]
                if not all(math.isfinite(value) for value in values):
                    raise ValueError(f"Non-finite metric values for {(model, dataset, metric)}")
                direction = DIRECTIONS[metric]
                ranks = average_ranks(values, reverse=direction == "higher_is_worse")
                population = len(ordered_common)
                for scene, value, rank in zip(ordered_common, values, ranks, strict=True):
                    rank_rows.append(
                        {
                            "model": model,
                            "dataset": dataset,
                            "scene": scene,
                            "metric": metric,
                            "direction": direction,
                            "value": value,
                            "failure_rank": rank,
                            "population": population,
                            "worstness_fraction": (
                                (population - rank) / (population - 1)
                                if population > 1
                                else None
                            ),
                        }
                    )

    pairwise = []
    groups = sorted({(row["dataset"], row["metric"]) for row in rank_rows})
    for dataset, metric in groups:
        for left_index, left in enumerate(models):
            for right in models[left_index + 1 :]:
                left_rows = {row["scene"]: row for row in rank_rows if row["model"] == left and row["dataset"] == dataset and row["metric"] == metric}
                right_rows = {row["scene"]: row for row in rank_rows if row["model"] == right and row["dataset"] == dataset and row["metric"] == metric}
                scenes = sorted(left_rows)
                if scenes != sorted(right_rows):
                    raise ValueError(f"Pairwise scene population differs for {(dataset, metric, left, right)}")
                pairwise.append(
                    {
                        "dataset": dataset,
                        "metric": metric,
                        "model_left": left,
                        "model_right": right,
                        "common_scenes": len(scenes),
                        "spearman_failure_rank": pearson(
                            [left_rows[scene]["failure_rank"] for scene in scenes],
                            [right_rows[scene]["failure_rank"] for scene in scenes],
                        ),
                    }
                )

    spread = []
    grouped_rows: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rank_rows:
        grouped_rows[(row["dataset"], row["scene"], row["metric"])].append(row)
    for (dataset, scene, metric), values in sorted(grouped_rows.items()):
        fractions = [row["worstness_fraction"] for row in values if row["worstness_fraction"] is not None]
        spread.append(
            {
                "dataset": dataset,
                "scene": scene,
                "metric": metric,
                "models": len(values),
                "worstness_min": min(fractions) if fractions else None,
                "worstness_max": max(fractions) if fractions else None,
                "worstness_mean": sum(fractions) / len(fractions) if fractions else None,
                "worstness_range": max(fractions) - min(fractions) if fractions else None,
            }
        )
    oracle_scene, oracle_summary = analyze_oracle(rank_rows)
    return {
        "common_populations": common_populations,
        "model_metric_rank_rows": rank_rows,
        "pairwise_rank_agreement": pairwise,
        "scene_metric_spread": spread,
        "oracle_scene_headroom": oracle_scene,
        "oracle_summary": oracle_summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest_path = args.snapshot / "manifest.json"
    metrics_path = args.snapshot / "completed_metrics.jsonl"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "non-final immutable snapshot":
        raise ValueError("Input is not an immutable non-final snapshot")
    if sha256(metrics_path) != manifest.get("completed_metrics_sha256"):
        raise ValueError("Snapshot metrics hash mismatch")
    outputs = compare(read_jsonl(metrics_path), manifest["models"])
    args.output.mkdir(parents=True, exist_ok=False)
    filenames = {name: f"{name}.csv" for name in outputs}
    output_meta = {}
    for name, rows in outputs.items():
        path = args.output / filenames[name]
        write_csv(path, rows)
        output_meta[name] = {"rows": len(rows), "sha256": sha256(path)}
    result = {
        "status": "non-final common-population comparison",
        "snapshot": str(args.snapshot.resolve()),
        "snapshot_manifest_sha256": sha256(manifest_path),
        "snapshot_metrics_sha256": sha256(metrics_path),
        "models": manifest["models"],
        "outputs": output_meta,
        "analysis_script_sha256": sha256(Path(__file__)),
        "failure_labels": None,
        "composite_score": None,
        "policy": "common scene population per dataset; official metrics remain separate",
    }
    (args.output / "manifest.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
