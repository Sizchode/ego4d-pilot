"""Measure DA3 per-view depth error against ScanNet++ GT for offline mining.

Ground truth is used only to construct analysis targets.  The output is never a
test-time policy input, and no quality threshold or failure label is assigned.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from scripts.failure_mining.analyze_da3_batch import finite_summary, sha256, write_csv
from scripts.failure_mining.analyze_gt_geometry_factors import (
    load_scene_gt_inputs,
    read_json,
    read_jsonl,
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def depth_error_summary(
    predicted: np.ndarray,
    ground_truth: np.ndarray,
    *,
    max_depth: float,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    predicted = np.asarray(predicted, dtype=np.float64)
    ground_truth = np.asarray(ground_truth, dtype=np.float64)
    require(predicted.shape == ground_truth.shape and predicted.ndim == 2, "Depth maps must be same-shape 2D arrays")
    require(math.isfinite(max_depth) and max_depth > 0, "max_depth must be finite and positive")
    pred_valid = np.isfinite(predicted) & (predicted > 0) & (predicted <= max_depth)
    gt_valid = np.isfinite(ground_truth) & (ground_truth > 0) & (ground_truth <= max_depth)
    joint = pred_valid & gt_valid
    pred = predicted[joint]
    gt = ground_truth[joint]
    signed_relative = (pred - gt) / gt
    absolute_relative = np.abs(signed_relative)
    signed_log_ratio = np.log(pred / gt)
    result: dict[str, Any] = {
        "pixels": int(predicted.size),
        "predicted_valid_count": int(pred_valid.sum()),
        "ground_truth_valid_count": int(gt_valid.sum()),
        "joint_valid_count": int(joint.sum()),
        "predicted_valid_ratio": float(pred_valid.mean()),
        "ground_truth_valid_ratio": float(gt_valid.mean()),
        "joint_valid_ratio": float(joint.mean()),
        "joint_over_ground_truth_valid_ratio": (
            float(joint.sum() / gt_valid.sum()) if gt_valid.any() else None
        ),
    }
    result.update(finite_summary(signed_relative, "depth.signed_relative"))
    result.update(finite_summary(absolute_relative, "depth.absolute_relative"))
    result.update(finite_summary(signed_log_ratio, "depth.signed_log_ratio"))
    result.update(finite_summary(np.abs(signed_log_ratio), "depth.absolute_log_ratio"))
    return result, {
        "signed_relative": signed_relative,
        "absolute_relative": absolute_relative,
        "signed_log_ratio": signed_log_ratio,
        "absolute_log_ratio": np.abs(signed_log_ratio),
    }


def find_runs(metrics_jsonl: Path, scenes: list[str], model: str) -> dict[str, Path]:
    requested = set(scenes)
    require(len(requested) == len(scenes), "Scenes must be unique")
    output: dict[str, Path] = {}
    for row in read_jsonl(metrics_jsonl):
        if row.get("model") != model or row.get("dataset") != "scannetpp" or row.get("scene") not in requested:
            continue
        scene = row["scene"]
        require(scene not in output, f"Duplicate metrics row: {scene}")
        require(row.get("state") == "completed", f"Selected run is not completed: {scene}")
        output[scene] = Path(row["run"])
    require(set(output) == requested, f"Missing selected scenes: {sorted(requested - set(output))}")
    return output


def analyze_scene(
    run: Path,
    dataset_object: Any,
    scene: str,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, str]]:
    config_path = run / "run_config.json"
    status_path = run / "status.json"
    validation_path = run / "validation.json"
    input_manifest_path = run / "input_manifest.json"
    fusion_path = run / "evaluation_capture/fusion_inputs_000.npz"
    for path in (config_path, status_path, validation_path, input_manifest_path, fusion_path):
        require(path.is_file(), f"Required file is missing: {path}")
    config = read_json(config_path)
    status = read_json(status_path)
    validation = read_json(validation_path)
    require((config.get("dataset"), config.get("scene")) == ("scannetpp", scene), f"Run identity differs: {run}")
    require(status.get("state") == "completed" and validation.get("passed") is True, f"Run is not validated: {run}")
    params = config["dataset_parameters"]
    max_depth = float(params["max_depth"])
    with np.load(fusion_path, allow_pickle=False) as fusion:
        predicted = np.asarray(fusion["depth"], dtype=np.float32)
    ground_truth = load_scene_gt_inputs(
        run,
        dataset_object,
        scene,
        input_h=int(params["input_h"]),
        input_w=int(params["input_w"]),
    )
    gt_depth = ground_truth["depth"]
    require(predicted.shape == gt_depth.shape, f"Predicted/GT population differs: {scene}")

    view_rows = []
    pooled: dict[str, list[np.ndarray]] = {
        "signed_relative": [],
        "absolute_relative": [],
        "signed_log_ratio": [],
        "absolute_log_ratio": [],
    }
    total_counts = {name: 0 for name in ("pixels", "predicted_valid_count", "ground_truth_valid_count", "joint_valid_count")}
    for view_index, (predicted_view, gt_view) in enumerate(zip(predicted, gt_depth, strict=True)):
        summary, arrays = depth_error_summary(predicted_view, gt_view, max_depth=max_depth)
        view_rows.append({"dataset": "scannetpp", "scene": scene, "view_index": view_index, **summary})
        for name in total_counts:
            total_counts[name] += int(summary[name])
        for name, values in arrays.items():
            pooled[name].append(values)

    scene_row: dict[str, Any] = {
        "dataset": "scannetpp",
        "scene": scene,
        "views": len(view_rows),
        "max_depth": max_depth,
        **total_counts,
        "predicted_valid_ratio": total_counts["predicted_valid_count"] / total_counts["pixels"],
        "ground_truth_valid_ratio": total_counts["ground_truth_valid_count"] / total_counts["pixels"],
        "joint_valid_ratio": total_counts["joint_valid_count"] / total_counts["pixels"],
        "joint_over_ground_truth_valid_ratio": total_counts["joint_valid_count"] / total_counts["ground_truth_valid_count"],
    }
    for name, arrays in pooled.items():
        scene_row.update(finite_summary(np.concatenate(arrays), f"depth.{name}"))
    hashes = {
        "run_config.json": sha256(config_path),
        "status.json": sha256(status_path),
        "validation.json": sha256(validation_path),
        "input_manifest.json": sha256(input_manifest_path),
        "fusion_inputs_000.npz": sha256(fusion_path),
        "gt_meta.npz": ground_truth["gt_meta_sha256"],
        "gt_depth_population": ground_truth["gt_depth_population_sha256"],
        "image_population": ground_truth["image_population_sha256"],
    }
    return view_rows, scene_row, hashes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-jsonl", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--scene", action="append", required=True)
    parser.add_argument("--model", default="da3")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(args.metrics_jsonl.is_file(), f"Missing metrics table: {args.metrics_jsonl}")
    require(args.dataset_root.is_dir(), f"Missing dataset root: {args.dataset_root}")
    require(not args.output.exists(), f"Refusing to overwrite output: {args.output}")
    runs = find_runs(args.metrics_jsonl, args.scene, args.model)

    from depth_anything_3.bench.datasets.scannetpp import ScanNetPP

    dataset_object = ScanNetPP()
    dataset_object.data_root = str(args.dataset_root.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{args.output.name}-", dir=args.output.parent))
    try:
        views: list[dict[str, Any]] = []
        scenes: list[dict[str, Any]] = []
        inputs = {}
        for scene in args.scene:
            view_rows, scene_row, hashes = analyze_scene(runs[scene], dataset_object, scene)
            views.extend(view_rows)
            scenes.append(scene_row)
            inputs[scene] = {"run": str(runs[scene].resolve()), "sha256": hashes}
        view_path = temporary / "view_gt_depth_error.csv"
        scene_path = temporary / "scene_gt_depth_error.csv"
        write_csv(view_path, views)
        write_csv(scene_path, scenes)
        manifest = {
            "dataset": "scannetpp",
            "model": args.model,
            "scenes": args.scene,
            "scene_rows": len(scenes),
            "view_rows": len(views),
            "metrics_jsonl": str(args.metrics_jsonl.resolve()),
            "metrics_jsonl_sha256": sha256(args.metrics_jsonl),
            "dataset_root": str(args.dataset_root.resolve()),
            "inputs": inputs,
            "view_output_sha256": sha256(view_path),
            "scene_output_sha256": sha256(scene_path),
            "analysis_script_sha256": sha256(Path(__file__)),
            "mask_policy": "finite positive depth at or below the benchmark max_depth for both prediction and GT",
            "uses_ground_truth_analysis": True,
            "test_time_policy_input": False,
            "failure_labels": None,
            "quality_thresholds": None,
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, args.output)
        print(json.dumps({key: manifest[key] for key in ("model", "scene_rows", "view_rows", "view_output_sha256", "scene_output_sha256")}, indent=2))
    except BaseException:
        import shutil

        shutil.rmtree(temporary, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()
