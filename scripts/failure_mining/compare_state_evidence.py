#!/usr/bin/env python3
"""Compare explicit reconstruction-state fields with official failure ranks."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

from scripts.failure_mining.compare_baselines import pearson, sha256, write_csv
from scripts.failure_mining.compare_image_evidence import (
    average_ranks,
    parse_failure_rankings,
    read_csv,
    scene_key,
)


def build_correlations(
    scene_rows: list[dict[str, str]],
    rankings: dict[tuple[str, str, str], dict],
    evidence_fields: list[str],
) -> list[dict]:
    if not scene_rows:
        raise ValueError("No scene rows")
    available_columns = set(scene_rows[0])
    missing = [field for field in evidence_fields if field not in available_columns]
    if missing:
        raise ValueError(f"Evidence fields are absent from scenes.csv: {missing}")
    if len(evidence_fields) != len(set(evidence_fields)):
        raise ValueError("Evidence fields must be unique")

    scenes = {scene_key(row): row for row in scene_rows}
    if len(scenes) != len(scene_rows):
        raise ValueError("Duplicate scene identities in scenes.csv")
    ranking_groups: dict[tuple[str, str], dict[str, dict]] = defaultdict(dict)
    for (dataset, scene, metric), ranking in rankings.items():
        ranking_groups[(dataset, metric)][scene] = ranking

    output = []
    for (dataset, metric), scene_rankings in sorted(ranking_groups.items()):
        for field in evidence_fields:
            triples = []
            for scene, ranking in sorted(scene_rankings.items()):
                raw = scenes[(dataset, scene)].get(field, "")
                if raw in {"", None}:
                    continue
                value = float(raw)
                if math.isfinite(value):
                    triples.append((value, ranking["failure_rank"]))
            if len(triples) < 2:
                continue
            evidence_ranks = average_ranks([value for value, _rank in triples])
            failure_ranks = [rank for _value, rank in triples]
            directions = {
                scene_rankings[scene]["direction"]
                for scene in scene_rankings
                if scenes[(dataset, scene)].get(field, "") not in {"", None}
                and math.isfinite(float(scenes[(dataset, scene)][field]))
            }
            if len(directions) != 1:
                raise ValueError(f"Inconsistent failure direction for {(dataset, metric)}")
            output.append(
                {
                    "dataset": dataset,
                    "reconstruction_metric": metric,
                    "reconstruction_direction": next(iter(directions)),
                    "evidence_field": field,
                    "scenes": len(triples),
                    "spearman_evidence_value_vs_failure_rank": pearson(
                        evidence_ranks, failure_ranks
                    ),
                }
            )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-analysis", type=Path, required=True)
    parser.add_argument("--evidence-field", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    scenes_path = args.baseline_analysis / "scenes.csv"
    rankings_path = args.baseline_analysis / "rankings.csv"
    scene_rows = read_csv(scenes_path)
    population = {scene_key(row) for row in scene_rows}
    rankings = parse_failure_rankings(read_csv(rankings_path), population)
    correlations = build_correlations(scene_rows, rankings, args.evidence_field)
    output_path = args.output / "state_failure_rank_correlations.csv"
    write_csv(output_path, correlations)

    manifest = {
        "baseline_analysis": str(args.baseline_analysis.resolve()),
        "scene_population": len(population),
        "evidence_fields": args.evidence_field,
        "correlation_rows": len(correlations),
        "baseline_scenes_sha256": sha256(scenes_path),
        "baseline_rankings_sha256": sha256(rankings_path),
        "output_sha256": sha256(output_path),
        "analysis_script_sha256": sha256(Path(__file__)),
        "failure_labels": None,
        "composite_score": None,
        "policy": "within-dataset Spearman; explicitly requested state fields remain separate",
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
