#!/usr/bin/env python3
"""Compare which camera pairs are difficult across archived baseline cases.

The comparison uses only the ground-truth camera errors already captured by the
official evaluator.  It assigns no failure label and does not aggregate rotation
and translation into a composite score.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from scripts.failure_mining.compare_baselines import pearson
from scripts.failure_mining.compare_image_evidence import average_ranks


CAMERA_METRICS = ("rotation_degrees", "translation_degrees")


@dataclass(frozen=True)
class Case:
    directory: Path
    model: str
    dataset: str
    scene: str
    pairs: dict[tuple[int, int], dict[str, float]]
    camera_pairs_sha256: str


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_case(directory: Path) -> Case:
    run_config_path = directory / "run_config.json"
    status_path = directory / "status.json"
    pairs_path = directory / "evaluation_capture" / "camera_pairs.npz"
    missing = [str(path) for path in (run_config_path, status_path, pairs_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete case bundle {directory}: {missing}")

    run_config = _load_json(run_config_path)
    status = _load_json(status_path)
    config = run_config.get("config", {})
    model = run_config.get("model") or config.get("model_repository")
    dataset = run_config.get("dataset", status.get("dataset"))
    scene = run_config.get("scene", status.get("scene"))
    if not all(isinstance(value, str) and value for value in (model, dataset, scene)):
        raise ValueError(f"Missing model, dataset, or scene identity in {directory}")

    with np.load(pairs_path, allow_pickle=False) as archive:
        required = ("index_a", "index_b", *CAMERA_METRICS)
        absent = [name for name in required if name not in archive]
        if absent:
            raise ValueError(f"Missing camera-pair arrays in {directory}: {absent}")
        arrays = {name: np.asarray(archive[name]).reshape(-1) for name in required}
    lengths = {name: int(value.size) for name, value in arrays.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Camera-pair arrays have unequal lengths in {directory}: {lengths}")

    pairs: dict[tuple[int, int], dict[str, float]] = {}
    for row_index, (raw_a, raw_b) in enumerate(
        zip(arrays["index_a"], arrays["index_b"], strict=True)
    ):
        index_a, index_b = int(raw_a), int(raw_b)
        if index_a == index_b:
            raise ValueError(f"Self-pair {(index_a, index_b)} in {directory}")
        key = tuple(sorted((index_a, index_b)))
        if key in pairs:
            raise ValueError(f"Duplicate camera pair {key} in {directory}")
        pairs[key] = {
            metric: float(arrays[metric][row_index]) for metric in CAMERA_METRICS
        }

    return Case(
        directory=directory.resolve(),
        model=model,
        dataset=dataset,
        scene=scene,
        pairs=pairs,
        camera_pairs_sha256=sha256(pairs_path),
    )


def compare_cases(cases: list[Case]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Case]] = {}
    for case in cases:
        grouped.setdefault((case.dataset, case.scene), []).append(case)

    rows: list[dict[str, Any]] = []
    for (dataset, scene), scene_cases in sorted(grouped.items()):
        models = [case.model for case in scene_cases]
        if len(models) != len(set(models)):
            raise ValueError(f"Duplicate model for {(dataset, scene)}: {models}")
        for left, right in itertools.combinations(sorted(scene_cases, key=lambda case: case.model), 2):
            left_keys = set(left.pairs)
            right_keys = set(right.pairs)
            if left_keys != right_keys:
                raise ValueError(
                    f"Camera-pair population differs for {(dataset, scene)} between "
                    f"{left.model} and {right.model}"
                )
            pair_keys = sorted(left_keys)
            for metric in CAMERA_METRICS:
                paired_values = [
                    (left.pairs[key][metric], right.pairs[key][metric]) for key in pair_keys
                ]
                finite = [
                    (left_value, right_value)
                    for left_value, right_value in paired_values
                    if math.isfinite(left_value) and math.isfinite(right_value)
                ]
                left_ranks = average_ranks([value[0] for value in finite])
                right_ranks = average_ranks([value[1] for value in finite])
                rows.append(
                    {
                        "dataset": dataset,
                        "scene": scene,
                        "model_left": left.model,
                        "model_right": right.model,
                        "metric": metric,
                        "camera_pairs": len(pair_keys),
                        "common_finite_pairs": len(finite),
                        "spearman_pair_error": pearson(left_ranks, right_ranks),
                        "left_case_dir": str(left.directory),
                        "right_case_dir": str(right.directory),
                        "left_camera_pairs_sha256": left.camera_pairs_sha256,
                        "right_camera_pairs_sha256": right.camera_pairs_sha256,
                    }
                )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("case_root", type=Path)
    parser.add_argument("output_csv", type=Path)
    args = parser.parse_args()

    case_dirs = sorted(
        path
        for path in args.case_root.iterdir()
        if path.is_dir() and (path / "run_config.json").is_file()
    )
    cases = [load_case(path) for path in case_dirs]
    if not cases:
        raise ValueError(f"No case directories found under {args.case_root}")
    rows = compare_cases(cases)
    write_csv(args.output_csv, rows)

    manifest = {
        "case_root": str(args.case_root.resolve()),
        "cases": len(cases),
        "scenes": len({(case.dataset, case.scene) for case in cases}),
        "comparison_rows": len(rows),
        "output_csv": str(args.output_csv.resolve()),
        "output_csv_sha256": sha256(args.output_csv),
        "analysis_script_sha256": sha256(Path(__file__)),
        "failure_labels": None,
        "composite_score": None,
        "policy": "within-scene pair-error Spearman; rotation and translation remain separate",
    }
    manifest_path = args.output_csv.with_suffix(args.output_csv.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
