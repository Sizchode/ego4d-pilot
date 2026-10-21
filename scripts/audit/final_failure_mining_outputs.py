"""Audit threshold-free final cross-model failure-mining outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from scripts.audit.analysis_outputs import load_json, require, sha256
from scripts.failure_mining.compare_image_evidence import average_ranks


OUTPUTS = {
    "oracle_scene_headroom": "oracle_scene_headroom.csv",
    "oracle_summary": "oracle_summary.csv",
    "metric_rank_agreement": "metric_rank_agreement.csv",
    "cross_model_consensus_failure_ranks": "cross_model_consensus_failure_ranks.csv",
    "pose_geometry_case_profiles": "pose_geometry_case_profiles.csv",
    "state_failure_rank_correlations": "state_failure_rank_correlations.csv",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    require(path.is_file(), f"Required file is missing: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=0.0, abs_tol=1e-12)


def audit_consensus(source: list[dict[str, str]], output: list[dict[str, str]]) -> None:
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in source:
        grouped[(row["dataset"], row["scene"], row["metric"])].append(row)
    observed = {(row["dataset"], row["scene"], row["metric"]): row for row in output}
    require(len(observed) == len(output), "Duplicate consensus case")
    require(set(observed) == set(grouped), "Consensus population differs from source rankings")

    by_group: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for key, source_rows in grouped.items():
        dataset, scene, metric = key
        row = observed[key]
        values = [float(item["worstness_fraction"]) for item in source_rows]
        require(all(math.isfinite(value) and 0 <= value <= 1 for value in values), f"Invalid source worstness: {key}")
        expected = {
            "worstness_mean": statistics.fmean(values),
            "worstness_median": statistics.median(values),
            "worstness_min": min(values),
            "worstness_max": max(values),
            "worstness_range": max(values) - min(values),
        }
        require(int(row["models"]) == len(source_rows), f"Consensus model count mismatch: {key}")
        for field, value in expected.items():
            require(close(float(row[field]), value), f"Consensus {field} mismatch: {key}")
        by_group[(dataset, metric)].append(row)

    for group, rows in by_group.items():
        rows.sort(key=lambda row: row["scene"])
        population = len(rows)
        require(all(int(row["population"]) == population for row in rows), f"Consensus population mismatch: {group}")
        expected_consensus = average_ranks([-float(row["worstness_mean"]) for row in rows])
        expected_disagreement = average_ranks([-float(row["worstness_range"]) for row in rows])
        for row, consensus, disagreement in zip(rows, expected_consensus, expected_disagreement, strict=True):
            require(close(float(row["consensus_failure_rank"]), consensus), f"Consensus rank mismatch: {(group, row['scene'])}")
            require(close(float(row["cross_model_disagreement_rank"]), disagreement), f"Disagreement rank mismatch: {(group, row['scene'])}")


def audit_case_profiles(
    source: list[dict[str, str]], output: list[dict[str, str]], pose_metric: str, geometry_metric: str
) -> None:
    metrics: dict[tuple[str, str, str], dict[str, dict[str, str]]] = defaultdict(dict)
    for row in source:
        if row["metric"] in {pose_metric, geometry_metric}:
            metrics[(row["model"], row["dataset"], row["scene"])][row["metric"]] = row
    expected = {key: value for key, value in metrics.items() if set(value) == {pose_metric, geometry_metric}}
    observed = {(row["model"], row["dataset"], row["scene"]): row for row in output}
    require(len(observed) == len(output), "Duplicate pose/geometry case profile")
    require(set(observed) == set(expected), "Pose/geometry case population differs from source rankings")

    by_group: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for key, pair in expected.items():
        row = observed[key]
        pose = float(pair[pose_metric]["worstness_fraction"])
        geometry = float(pair[geometry_metric]["worstness_fraction"])
        gap = geometry - pose
        profile = (
            "geometry_ranked_worse_than_pose"
            if gap > 0
            else "pose_ranked_worse_than_geometry"
            if gap < 0
            else "equal_failure_rank"
        )
        require(close(float(row["pose_worstness_fraction"]), pose), f"Pose worstness mismatch: {key}")
        require(close(float(row["geometry_worstness_fraction"]), geometry), f"Geometry worstness mismatch: {key}")
        require(close(float(row["geometry_minus_pose_worstness"]), gap), f"Pose/geometry gap mismatch: {key}")
        require(close(float(row["absolute_pose_geometry_rank_gap"]), abs(gap)), f"Absolute gap mismatch: {key}")
        require(row["rank_profile"] == profile, f"Rank profile mismatch: {key}")
        by_group[key[:2]].append(row)

    for group, rows in by_group.items():
        rows.sort(key=lambda row: row["scene"])
        population = len(rows)
        expected_ranks = average_ranks([-float(row["absolute_pose_geometry_rank_gap"]) for row in rows])
        for row, rank in zip(rows, expected_ranks, strict=True):
            require(int(row["population"]) == population, f"Profile population mismatch: {group}")
            require(close(float(row["pose_geometry_disagreement_rank"]), rank), f"Profile rank mismatch: {(group, row['scene'])}")


def audit_final_failure_mining(
    analysis_dir: Path, cross_dir: Path, expected_scene_population: int | None = None
) -> dict[str, Any]:
    analysis_dir = analysis_dir.resolve()
    cross_dir = cross_dir.resolve()
    manifest_path = analysis_dir / "manifest.json"
    cross_manifest_path = cross_dir / "manifest.json"
    rankings_path = cross_dir / "model_metric_rank_rows.csv"
    manifest = load_json(manifest_path)
    cross_manifest = load_json(cross_manifest_path)
    require(manifest.get("cross_manifest_sha256") == sha256(cross_manifest_path), "Cross manifest hash mismatch")
    require(manifest.get("cross_rankings_sha256") == sha256(rankings_path), "Cross ranking hash mismatch")
    require(manifest.get("models") == cross_manifest.get("models"), "Model population/order differs")
    require(manifest.get("scene_population") == cross_manifest.get("scene_population"), "Scene population differs")
    if expected_scene_population is not None:
        require(manifest.get("scene_population") == expected_scene_population, "Unexpected scene population")
    require(manifest.get("failure_labels") is None, "Final analysis assigns failure labels")
    require(manifest.get("quality_thresholds") is None, "Final analysis uses quality thresholds")
    require(manifest.get("composite_score") is None, "Final analysis uses a composite score")
    require(set(manifest.get("outputs", {})) == set(OUTPUTS), "Unexpected final output set")

    output_hashes = {}
    for name, filename in OUTPUTS.items():
        meta = manifest["outputs"][name]
        require(meta.get("path") == filename, f"Output path mismatch: {name}")
        path = analysis_dir / filename
        rows = read_csv(path)
        require(meta.get("rows") == len(rows), f"Output row count mismatch: {name}")
        require(meta.get("sha256") == sha256(path), f"Output hash mismatch: {name}")
        output_hashes[filename] = sha256(path)

    source = read_csv(rankings_path)
    audit_consensus(source, read_csv(analysis_dir / OUTPUTS["cross_model_consensus_failure_ranks"]))
    audit_case_profiles(
        source,
        read_csv(analysis_dir / OUTPUTS["pose_geometry_case_profiles"]),
        manifest["metric_left"],
        manifest["metric_right"],
    )
    return {
        "schema_version": 1,
        "passed": True,
        "models": manifest["models"],
        "scene_population": manifest["scene_population"],
        "analysis_manifest_sha256": sha256(manifest_path),
        "cross_manifest_sha256": sha256(cross_manifest_path),
        "cross_rankings_sha256": sha256(rankings_path),
        "output_sha256": output_hashes,
        "failure_labels": None,
        "quality_thresholds": None,
        "composite_score": None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--cross-dir", type=Path, required=True)
    parser.add_argument("--expected-scenes", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit_final_failure_mining(args.analysis, args.cross_dir, args.expected_scenes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    require(not args.output.exists(), f"Refusing to overwrite existing audit: {args.output}")
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("passed", "models", "scene_population")}, indent=2))


if __name__ == "__main__":
    main()
