"""Run manifest-driven, threshold-free analyses on the final four-model results."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from scripts.failure_mining.analyze_backbone_oracle import analyze as analyze_oracle
from scripts.failure_mining.analyze_backbone_oracle import read_rankings, sha256, write_csv
from scripts.failure_mining.compare_image_evidence import parse_failure_rankings, read_csv, scene_key
from scripts.failure_mining.compare_image_evidence import average_ranks
from scripts.failure_mining.compare_metric_rankings import compare as compare_metrics
from scripts.failure_mining.compare_state_evidence import build_correlations


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def parse_worstness(row: dict[str, str]) -> float:
    value = float(row["worstness_fraction"])
    require(math.isfinite(value) and 0.0 <= value <= 1.0, "Invalid worstness fraction")
    return value


def build_cross_model_consensus(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, list[dict[str, str]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        grouped[(row["dataset"], row["metric"])][row["scene"]].append(row)

    output = []
    for (dataset, metric), by_scene in sorted(grouped.items()):
        model_sets = {tuple(sorted(row["model"] for row in scene_rows)) for scene_rows in by_scene.values()}
        require(len(model_sets) == 1, f"Model population differs across scenes: {(dataset, metric)}")
        scene_values = []
        directions = set()
        for scene, scene_rows in sorted(by_scene.items()):
            values = [parse_worstness(row) for row in scene_rows]
            directions.update(row["direction"] for row in scene_rows)
            scene_values.append(
                {
                    "dataset": dataset,
                    "scene": scene,
                    "metric": metric,
                    "models": len(scene_rows),
                    "worstness_mean": statistics.fmean(values),
                    "worstness_median": statistics.median(values),
                    "worstness_min": min(values),
                    "worstness_max": max(values),
                    "worstness_range": max(values) - min(values),
                }
            )
        require(len(directions) == 1, f"Metric direction differs: {(dataset, metric)}")
        consensus_ranks = average_ranks([-row["worstness_mean"] for row in scene_values])
        disagreement_ranks = average_ranks([-row["worstness_range"] for row in scene_values])
        population = len(scene_values)
        for row, consensus_rank, disagreement_rank in zip(
            scene_values, consensus_ranks, disagreement_ranks, strict=True
        ):
            output.append(
                {
                    **row,
                    "direction": next(iter(directions)),
                    "population": population,
                    "consensus_failure_rank": consensus_rank,
                    "cross_model_disagreement_rank": disagreement_rank,
                }
            )
    return output


def build_pose_geometry_case_profiles(
    rows: list[dict[str, str]], pose_metric: str, geometry_metric: str
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, dict[str, str]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for row in rows:
        if row["metric"] not in {pose_metric, geometry_metric}:
            continue
        key = (row["model"], row["dataset"])
        require(
            row["metric"] not in grouped[key][row["scene"]],
            f"Duplicate case metric: {(key, row['scene'], row['metric'])}",
        )
        grouped[key][row["scene"]][row["metric"]] = row

    output = []
    for (model, dataset), by_scene in sorted(grouped.items()):
        complete = {
            scene: metrics
            for scene, metrics in by_scene.items()
            if set(metrics) == {pose_metric, geometry_metric}
        }
        if not complete:
            continue
        require(len(complete) == len(by_scene), f"Partial pose/geometry population: {(model, dataset)}")
        provisional = []
        for scene, metrics in sorted(complete.items()):
            pose = metrics[pose_metric]
            geometry = metrics[geometry_metric]
            pose_worstness = parse_worstness(pose)
            geometry_worstness = parse_worstness(geometry)
            gap = geometry_worstness - pose_worstness
            profile = (
                "geometry_ranked_worse_than_pose"
                if gap > 0
                else "pose_ranked_worse_than_geometry"
                if gap < 0
                else "equal_failure_rank"
            )
            provisional.append(
                {
                    "model": model,
                    "dataset": dataset,
                    "scene": scene,
                    "pose_metric": pose_metric,
                    "pose_value": float(pose["value"]),
                    "pose_direction": pose["direction"],
                    "pose_failure_rank": float(pose["failure_rank"]),
                    "pose_worstness_fraction": pose_worstness,
                    "geometry_metric": geometry_metric,
                    "geometry_value": float(geometry["value"]),
                    "geometry_direction": geometry["direction"],
                    "geometry_failure_rank": float(geometry["failure_rank"]),
                    "geometry_worstness_fraction": geometry_worstness,
                    "geometry_minus_pose_worstness": gap,
                    "absolute_pose_geometry_rank_gap": abs(gap),
                    "rank_profile": profile,
                }
            )
        gap_ranks = average_ranks(
            [-row["absolute_pose_geometry_rank_gap"] for row in provisional]
        )
        population = len(provisional)
        for row, gap_rank in zip(provisional, gap_ranks, strict=True):
            output.append(
                {
                    **row,
                    "population": population,
                    "pose_geometry_disagreement_rank": gap_rank,
                }
            )
    return output


def analyze_final(
    cross_dir: Path,
    metric_left: str,
    metric_right: str,
    evidence_fields: list[str],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    cross_dir = cross_dir.resolve()
    manifest_path = cross_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    models = manifest.get("models")
    require(isinstance(models, list) and len(models) >= 2, "Cross manifest has fewer than two models")
    require(len(models) == len(set(models)), "Cross manifest has duplicate models")
    inputs = manifest.get("inputs", {})
    require(list(inputs) == models, "Cross-manifest input order differs from model order")

    rankings_path = cross_dir / "model_metric_rank_rows.csv"
    ranking_rows = read_rankings(rankings_path)
    oracle_scene, oracle_summary = analyze_oracle(ranking_rows)
    metric_agreement = compare_metrics(ranking_rows, metric_left, metric_right)
    consensus = build_cross_model_consensus(ranking_rows)
    case_profiles = build_pose_geometry_case_profiles(
        ranking_rows, metric_left, metric_right
    )

    state_correlations = []
    input_hashes = {}
    for model in models:
        item = inputs[model]
        analysis_dir = Path(item["analysis_dir"])
        scenes_path = analysis_dir / "scenes.csv"
        model_rankings_path = analysis_dir / "rankings.csv"
        require(sha256(scenes_path) == item["scenes_sha256"], f"Scene table hash mismatch for {model}")
        require(sha256(model_rankings_path) == item["rankings_sha256"], f"Ranking table hash mismatch for {model}")
        scenes = read_csv(scenes_path)
        population = {scene_key(row) for row in scenes}
        rankings = parse_failure_rankings(read_csv(model_rankings_path), population)
        correlations = build_correlations(scenes, rankings, evidence_fields)
        state_correlations.extend({"model": model, **row} for row in correlations)
        input_hashes[model] = {
            "analysis_dir": str(analysis_dir.resolve()),
            "scenes_sha256": item["scenes_sha256"],
            "rankings_sha256": item["rankings_sha256"],
        }

    outputs = {
        "oracle_scene_headroom": oracle_scene,
        "oracle_summary": oracle_summary,
        "metric_rank_agreement": metric_agreement,
        "cross_model_consensus_failure_ranks": consensus,
        "pose_geometry_case_profiles": case_profiles,
        "state_failure_rank_correlations": state_correlations,
    }
    provenance = {
        "cross_manifest_sha256": sha256(manifest_path),
        "cross_rankings_sha256": sha256(rankings_path),
        "models": models,
        "scene_population": manifest.get("scene_population"),
        "metric_left": metric_left,
        "metric_right": metric_right,
        "evidence_fields": evidence_fields,
        "inputs": input_hashes,
    }
    return outputs, provenance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cross-dir", type=Path, required=True)
    parser.add_argument("--metric-left", required=True)
    parser.add_argument("--metric-right", required=True)
    parser.add_argument("--evidence-field", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(args.metric_left != args.metric_right, "Metrics must differ")
    require(len(args.evidence_field) == len(set(args.evidence_field)), "Evidence fields must be unique")
    args.output.mkdir(parents=True, exist_ok=False)

    outputs, provenance = analyze_final(
        args.cross_dir,
        args.metric_left,
        args.metric_right,
        args.evidence_field,
    )
    filenames = {
        "oracle_scene_headroom": "oracle_scene_headroom.csv",
        "oracle_summary": "oracle_summary.csv",
        "metric_rank_agreement": "metric_rank_agreement.csv",
        "cross_model_consensus_failure_ranks": "cross_model_consensus_failure_ranks.csv",
        "pose_geometry_case_profiles": "pose_geometry_case_profiles.csv",
        "state_failure_rank_correlations": "state_failure_rank_correlations.csv",
    }
    output_meta = {}
    for name, filename in filenames.items():
        path = args.output / filename
        write_csv(path, outputs[name])
        output_meta[name] = {
            "path": filename,
            "rows": len(outputs[name]),
            "sha256": sha256(path),
        }
    result_manifest = {
        **provenance,
        "outputs": output_meta,
        "analysis_script_sha256": sha256(Path(__file__)),
        "failure_labels": None,
        "quality_thresholds": None,
        "composite_score": None,
        "policy": "per-dataset and per-metric analyses only; no cross-metric aggregation",
    }
    (args.output / "manifest.json").write_text(
        json.dumps(result_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result_manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
