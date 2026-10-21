"""Convert audited factor profiles into continuous intervention-gain targets."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

from scripts.failure_mining.analyze_gt_geometry_factor_profiles import FACTORS
from scripts.failure_mining.compare_baselines import sha256, write_csv


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_csv(path: Path) -> list[dict[str, str]]:
    require(path.is_file(), f"Required file is missing: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def build_targets(
    profile_dir: Path,
    *,
    dataset: str,
    model: str,
    reconstruction_metric: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest_path = profile_dir / "manifest.json"
    profiles_path = profile_dir / "case_metric_factor_profiles.csv"
    require(manifest_path.is_file(), f"Required file is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(manifest.get("profile_output_sha256") == sha256(profiles_path), "Profile table hash mismatch")
    require(manifest.get("uses_ground_truth_intervention") is True, "Missing GT provenance")
    require(manifest.get("test_time_policy_input") is False, "GT is marked as policy input")
    require(manifest.get("success_threshold") is None, "Profiles use a success threshold")
    require(model in manifest.get("models", []), f"Unknown model: {model}")

    source = [
        row
        for row in read_csv(profiles_path)
        if row["model"] == model and row["metric"] == reconstruction_metric
    ]
    scenes = manifest.get("scenes")
    require(isinstance(scenes, list) and scenes, "Invalid scene population")
    require(len(source) == len(scenes), "Incomplete model/metric population")
    by_scene = {row["scene"]: row for row in source}
    require(len(by_scene) == len(source) and set(by_scene) == set(scenes), "Scene population differs from manifest")

    output = []
    for scene in scenes:
        row = by_scene[scene]
        for factor, _stored_gain_field in FACTORS:
            value = float(row[f"{factor}_conditional_gain"])
            require(math.isfinite(value), f"Non-finite gain: {(scene, factor)}")
            output.append(
                {
                    "dataset": dataset,
                    "scene": scene,
                    "factor": factor,
                    "conditional_gain": value,
                    "model": model,
                    "reconstruction_metric": reconstruction_metric,
                }
            )
    provenance = {
        "dataset": dataset,
        "model": model,
        "reconstruction_metric": reconstruction_metric,
        "factors": [factor for factor, _field in FACTORS],
        "scenes": scenes,
        "rows": len(output),
        "profile_manifest_sha256": sha256(manifest_path),
        "profile_table_sha256": sha256(profiles_path),
        "uses_ground_truth_intervention": True,
        "test_time_policy_input": False,
        "success_threshold": None,
    }
    return output, provenance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--metric", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), f"Refusing to overwrite output: {args.output}")
    rows, manifest = build_targets(
        args.profiles,
        dataset=args.dataset,
        model=args.model,
        reconstruction_metric=args.metric,
    )
    args.output.mkdir(parents=True, exist_ok=False)
    target_path = args.output / "factor_conditional_gain_targets.csv"
    write_csv(target_path, rows)
    manifest.update(
        {
            "target_output_sha256": sha256(target_path),
            "analysis_script_sha256": sha256(Path(__file__)),
        }
    )
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
