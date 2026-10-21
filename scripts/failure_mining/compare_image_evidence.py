#!/usr/bin/env python3
"""Compare threshold-free image statistics with official failure rankings."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

from scripts.failure_mining.compare_baselines import pearson


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def average_ranks(values: list[float]) -> list[float]:
    """Return one-based average ranks in ascending value order."""
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and values[order[stop]] == values[order[start]]:
            stop += 1
        rank = ((start + 1) + stop) / 2.0
        for position in range(start, stop):
            ranks[order[position]] = rank
        start = stop
    return ranks


def scene_key(row: dict[str, str]) -> tuple[str, str]:
    return row["dataset"], row["scene"]


def validate_population(
    scene_rows: list[dict[str, str]], image_rows: list[dict[str, str]]
) -> list[tuple[str, str]]:
    expected = [scene_key(row) for row in scene_rows]
    observed = [scene_key(row) for row in image_rows]
    if len(expected) != len(set(expected)):
        raise ValueError("Baseline scenes.csv has duplicate scene identities")
    if len(observed) != len(set(observed)):
        raise ValueError("Image analysis has duplicate scene identities")
    if observed != expected:
        raise ValueError("Image-analysis population or order differs from baseline scenes.csv")
    return expected


def parse_failure_rankings(
    rows: list[dict[str, str]], population: set[tuple[str, str]]
) -> dict[tuple[str, str, str], dict]:
    parsed = {}
    for row in rows:
        key = (row["dataset"], row["scene"], row["metric"])
        if key in parsed:
            raise ValueError(f"Duplicate failure ranking: {key}")
        if key[:2] not in population:
            raise ValueError(f"Ranking references an unknown scene: {key}")
        parsed[key] = {
            "failure_rank": float(row["failure_rank"]),
            "population": int(row["population"]),
            "direction": row["direction"],
        }
    return parsed


def build_correlations(
    image_rows: list[dict[str, str]], rankings: dict[tuple[str, str, str], dict]
) -> list[dict]:
    image_metrics = sorted(
        {
            key
            for row in image_rows
            for key, value in row.items()
            if key.startswith("image.") and value not in {"", None}
        }
    )
    grouped_images = {scene_key(row): row for row in image_rows}
    ranking_groups: dict[tuple[str, str], dict[str, dict]] = defaultdict(dict)
    for (dataset, scene, metric), value in rankings.items():
        ranking_groups[(dataset, metric)][scene] = value

    output = []
    for (dataset, reconstruction_metric), scene_rankings in sorted(ranking_groups.items()):
        for image_metric in image_metrics:
            triples = []
            for scene, ranking in sorted(scene_rankings.items()):
                raw = grouped_images[(dataset, scene)].get(image_metric, "")
                if raw in {"", None}:
                    continue
                value = float(raw)
                if math.isfinite(value):
                    triples.append((scene, value, ranking["failure_rank"]))
            if len(triples) < 2:
                continue
            image_ranks = average_ranks([item[1] for item in triples])
            failure_ranks = [item[2] for item in triples]
            directions = {scene_rankings[item[0]]["direction"] for item in triples}
            if len(directions) != 1:
                raise ValueError(
                    f"Inconsistent failure direction for {(dataset, reconstruction_metric)}"
                )
            output.append(
                {
                    "dataset": dataset,
                    "reconstruction_metric": reconstruction_metric,
                    "reconstruction_direction": next(iter(directions)),
                    "image_metric": image_metric,
                    "scenes": len(triples),
                    "spearman_image_value_vs_failure_rank": pearson(
                        image_ranks, failure_ranks
                    ),
                }
            )
    return output


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-analysis", type=Path, required=True)
    parser.add_argument("--image-analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    scenes_path = args.baseline_analysis / "scenes.csv"
    rankings_path = args.baseline_analysis / "rankings.csv"
    image_path = args.image_analysis / "scene_image_evidence.csv"
    scene_rows = read_csv(scenes_path)
    image_rows = read_csv(image_path)
    population = validate_population(scene_rows, image_rows)
    rankings = parse_failure_rankings(read_csv(rankings_path), set(population))
    correlations = build_correlations(image_rows, rankings)
    write_csv(args.output / "image_failure_rank_correlations.csv", correlations)

    manifest = {
        "baseline_analysis": str(args.baseline_analysis.resolve()),
        "image_analysis": str(args.image_analysis.resolve()),
        "scene_population": len(population),
        "correlation_rows": len(correlations),
        "baseline_scenes_sha256": sha256(scenes_path),
        "baseline_rankings_sha256": sha256(rankings_path),
        "image_scenes_sha256": sha256(image_path),
        "analysis_script_sha256": sha256(Path(__file__)),
        "failure_labels": None,
        "composite_score": None,
        "policy": "within-dataset Spearman correlations; image metrics remain separate",
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
