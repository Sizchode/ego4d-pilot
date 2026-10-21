"""Select deterministic extrema for visual failure review.

Selection uses only within-dataset/metric ranks from audited final failure
mining.  It has no score threshold or top-k parameter, and preserves all exact
ties at the selected extrema.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from scripts.failure_mining.compare_baselines import sha256, write_csv


REASON_ORDER = (
    "consensus_hardest",
    "consensus_easiest_control",
    "cross_model_most_disputed",
    "largest_gap:geometry_ranked_worse_than_pose",
    "largest_gap:pose_ranked_worse_than_geometry",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_json(path: Path) -> Any:
    require(path.is_file(), f"Required file is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    require(path.is_file(), f"Required file is missing: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def select(
    analysis_dir: Path, audit_path: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest_path = analysis_dir / "manifest.json"
    consensus_path = analysis_dir / "cross_model_consensus_failure_ranks.csv"
    profiles_path = analysis_dir / "pose_geometry_case_profiles.csv"
    manifest = read_json(manifest_path)
    audit = read_json(audit_path)
    require(audit.get("passed") is True, "Final failure-mining audit did not pass")
    require(
        audit.get("analysis_manifest_sha256") == sha256(manifest_path),
        "Audit does not cover this final analysis manifest",
    )
    outputs = manifest.get("outputs", {})
    require(
        outputs.get("cross_model_consensus_failure_ranks", {}).get("sha256")
        == sha256(consensus_path),
        "Consensus table hash mismatch",
    )
    require(
        outputs.get("pose_geometry_case_profiles", {}).get("sha256")
        == sha256(profiles_path),
        "Pose/geometry profile hash mismatch",
    )
    models = manifest.get("models")
    require(isinstance(models, list) and models and len(models) == len(set(models)), "Invalid model population")

    selected: dict[tuple[str, str, str, str], dict[str, Any]] = {}

    def add(model: str, dataset: str, scene: str, metric: str, reason: str, source: dict[str, str]) -> None:
        key = (model, dataset, scene, metric)
        if key not in selected:
            selected[key] = {
                "model": model,
                "dataset": dataset,
                "scene": scene,
                "metric": metric,
                "selection_reasons": [],
                "consensus_failure_rank": source.get("consensus_failure_rank", ""),
                "cross_model_disagreement_rank": source.get("cross_model_disagreement_rank", ""),
                "pose_geometry_disagreement_rank": source.get("pose_geometry_disagreement_rank", ""),
                "rank_profile": source.get("rank_profile", ""),
            }
        if reason not in selected[key]["selection_reasons"]:
            selected[key]["selection_reasons"].append(reason)

    consensus_groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in read_csv(consensus_path):
        consensus_groups[(row["dataset"], row["metric"])].append(row)
    require(consensus_groups, "No consensus groups")
    for (dataset, metric), rows in sorted(consensus_groups.items()):
        consensus_ranks = [float(row["consensus_failure_rank"]) for row in rows]
        disagreement_ranks = [float(row["cross_model_disagreement_rank"]) for row in rows]
        require(all(math.isfinite(value) for value in [*consensus_ranks, *disagreement_ranks]), f"Non-finite selection rank: {(dataset, metric)}")
        extrema = {
            "consensus_hardest": min(consensus_ranks),
            "consensus_easiest_control": max(consensus_ranks),
            "cross_model_most_disputed": min(disagreement_ranks),
        }
        for reason, extreme in extrema.items():
            field = (
                "cross_model_disagreement_rank"
                if reason == "cross_model_most_disputed"
                else "consensus_failure_rank"
            )
            for row in rows:
                if float(row[field]) == extreme:
                    for model in models:
                        add(model, dataset, row["scene"], metric, reason, row)

    profile_groups: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in read_csv(profiles_path):
        profile = row["rank_profile"]
        if profile == "equal_failure_rank":
            continue
        require(
            profile in {
                "geometry_ranked_worse_than_pose",
                "pose_ranked_worse_than_geometry",
            },
            f"Unknown rank profile: {profile}",
        )
        profile_groups[(row["model"], row["dataset"], profile)].append(row)
    for (model, dataset, profile), rows in sorted(profile_groups.items()):
        maximum = max(float(row["absolute_pose_geometry_rank_gap"]) for row in rows)
        reason = f"largest_gap:{profile}"
        for row in rows:
            if float(row["absolute_pose_geometry_rank_gap"]) == maximum:
                add(model, dataset, row["scene"], "pose_vs_geometry", reason, row)

    reason_index = {reason: index for index, reason in enumerate(REASON_ORDER)}
    output = []
    for key in sorted(selected):
        row = selected[key]
        reasons = sorted(row.pop("selection_reasons"), key=reason_index.__getitem__)
        output.append({**row, "selection_reasons": ";".join(reasons)})
    provenance = {
        "input_analysis_manifest_sha256": sha256(manifest_path),
        "input_audit_sha256": sha256(audit_path),
        "input_consensus_sha256": sha256(consensus_path),
        "input_pose_geometry_profiles_sha256": sha256(profiles_path),
        "models": models,
        "selected_model_cases": len(output),
        "selected_unique_scenes": len({(row["dataset"], row["scene"]) for row in output}),
        "selection_rules": list(REASON_ORDER),
        "ties": "all exact ties retained",
        "selection_threshold": None,
        "top_k": None,
        "uses_official_evaluator_ranks": True,
        "intended_use": "offline visual failure review; not deployable policy input",
    }
    return output, provenance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), f"Refusing to overwrite output: {args.output}")
    rows, manifest = select(args.analysis, args.audit)
    args.output.mkdir(parents=True, exist_ok=False)
    cases_path = args.output / "selected_review_cases.csv"
    write_csv(cases_path, rows)
    manifest.update(
        {
            "cases_output_sha256": sha256(cases_path),
            "analysis_script_sha256": sha256(Path(__file__)),
        }
    )
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
