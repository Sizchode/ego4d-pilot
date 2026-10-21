"""Measure geometry headroom from replacing only predicted extrinsics with GT.

This offline ScanNet++ intervention consumes captures produced by the official
DA3-BENCH evaluator. It keeps each model's scaled predicted depth and predicted
intrinsics fixed, replaces the aligned predicted extrinsics with the evaluator's
sampled reference extrinsics, and repeats the same TSDF geometry protocol.
Ground truth is used explicitly; these results are diagnostic oracle evidence
and cannot be exposed to a test-time recovery policy.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import KDTree


REQUIRED_CAPTURE_FILES = (
    "fusion_inputs_000.npz",
    "alignment_000.npz",
    "surface_distance_000.npz",
    "surface_distance_001.npz",
    "fused_mesh_000.ply",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def metrics_from_distances(
    prediction_to_gt: np.ndarray,
    gt_to_prediction: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    prediction_to_gt = np.asarray(prediction_to_gt, dtype=np.float64).reshape(-1)
    gt_to_prediction = np.asarray(gt_to_prediction, dtype=np.float64).reshape(-1)
    if prediction_to_gt.size == 0 or gt_to_prediction.size == 0:
        raise ValueError("Both point populations must be non-empty")
    if not (
        np.isfinite(prediction_to_gt).all()
        and np.isfinite(gt_to_prediction).all()
        and math.isfinite(threshold)
        and threshold > 0
    ):
        raise ValueError("Distances and threshold must be finite, with positive threshold")
    accuracy = float(np.mean(prediction_to_gt))
    completeness = float(np.mean(gt_to_prediction))
    precision = float(np.mean(prediction_to_gt < threshold))
    recall = float(np.mean(gt_to_prediction < threshold))
    fscore = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "acc": accuracy,
        "comp": completeness,
        "overall": (accuracy + completeness) / 2,
        "precision": precision,
        "recall": recall,
        "fscore": fscore,
    }


def metrics_from_points(
    gt_points: np.ndarray,
    predicted_points: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    gt_points = np.asarray(gt_points, dtype=np.float64)
    predicted_points = np.asarray(predicted_points, dtype=np.float64)
    if gt_points.ndim != 2 or predicted_points.ndim != 2:
        raise ValueError("Point populations must be rank-two arrays")
    if gt_points.shape[1:] != (3,) or predicted_points.shape[1:] != (3,):
        raise ValueError("Point populations must have shape [N, 3]")
    if not np.isfinite(gt_points).all() or not np.isfinite(predicted_points).all():
        raise ValueError("Point populations must be finite")
    prediction_to_gt = KDTree(gt_points).query(predicted_points)[0].reshape(-1)
    gt_to_prediction = KDTree(predicted_points).query(gt_points)[0].reshape(-1)
    return metrics_from_distances(prediction_to_gt, gt_to_prediction, threshold)


def verify_official_capture(
    capture_dir: Path,
    official_metrics: dict[str, float],
    eval_threshold: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, str]]:
    missing = [name for name in REQUIRED_CAPTURE_FILES if not (capture_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Capture is incomplete at {capture_dir}: {missing}")
    with np.load(capture_dir / "surface_distance_000.npz", allow_pickle=False) as first:
        gt_points = np.asarray(first["reference"], dtype=np.float64)
        native_points = np.asarray(first["query"], dtype=np.float64)
        prediction_to_gt = np.asarray(first["distance"], dtype=np.float64)
    with np.load(capture_dir / "surface_distance_001.npz", allow_pickle=False) as second:
        np.testing.assert_array_equal(second["reference"], native_points)
        np.testing.assert_array_equal(second["query"], gt_points)
        gt_to_prediction = np.asarray(second["distance"], dtype=np.float64)
    reproduced = metrics_from_distances(prediction_to_gt, gt_to_prediction, eval_threshold)
    for key, expected in official_metrics.items():
        if key in reproduced and not math.isclose(
            reproduced[key], float(expected), rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(
                f"Captured official metric mismatch at {capture_dir}: "
                f"{key}={reproduced[key]} versus {expected}"
            )
    hashes = {name: sha256(capture_dir / name) for name in REQUIRED_CAPTURE_FILES}
    return gt_points, native_points, hashes


def sample_mesh_for_scannetpp(
    mesh,
    gt_points: np.ndarray,
    *,
    sampling_number: int,
    down_sample: float,
    seed: int,
) -> np.ndarray:
    import open3d as o3d

    if mesh.is_empty():
        raise ValueError("Cannot evaluate an empty mesh")
    o3d.utility.random.seed(seed)
    point_cloud = mesh.sample_points_uniformly(number_of_points=sampling_number)
    points = np.asarray(point_cloud.points)
    minimum = gt_points.min(axis=0) - 0.1
    maximum = gt_points.max(axis=0) + 0.1
    inside = np.all((points >= minimum) & (points <= maximum), axis=1)
    point_cloud = point_cloud.select_by_index(np.flatnonzero(inside))
    if down_sample > 0:
        point_cloud = point_cloud.voxel_down_sample(down_sample)
    output = np.asarray(point_cloud.points, dtype=np.float64)
    if output.size == 0:
        raise ValueError("Mesh has no points inside the ScanNet++ evaluation bounds")
    return output


def fuse_with_extrinsics(
    depth: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    *,
    voxel_length: float,
    sdf_trunc: float,
    max_depth: float,
):
    from depth_anything_3.bench.utils import create_tsdf_volume, fuse_depth_to_tsdf

    count, height, width = depth.shape
    images = np.broadcast_to(
        np.zeros((1, height, width, 3), dtype=np.uint8),
        (count, height, width, 3),
    )
    volume = create_tsdf_volume(voxel_length=voxel_length, sdf_trunc=sdf_trunc)
    return fuse_depth_to_tsdf(
        volume,
        depth,
        images,
        intrinsics,
        extrinsics,
        max_depth=max_depth,
    )


def analyze_case(run: Path, output: Path, *, seed: int) -> dict[str, Any]:
    import open3d as o3d

    config = read_json(run / "run_config.json")
    status = read_json(run / "status.json")
    validation = read_json(run / "validation.json")
    dataset, scene = config.get("dataset"), config.get("scene")
    model = config.get("model") or config.get("config", {}).get("model_repository")
    if dataset != "scannetpp":
        raise ValueError(f"This intervention is defined only for ScanNet++: {run}")
    if status.get("state") != "completed" or validation.get("passed") is not True:
        raise ValueError(f"Run is not completed and validated: {run}")
    if not all(isinstance(value, str) and value for value in (scene, model)):
        raise ValueError(f"Run identity is incomplete: {run}")

    params = config["dataset_parameters"]
    eval_threshold = float(params["eval_threshold"])
    capture = run / "evaluation_capture"
    official_metrics = validation["metrics"]["recon_unposed"]
    gt_points, captured_native_points, capture_hashes = verify_official_capture(
        capture, official_metrics, eval_threshold
    )

    with np.load(capture / "fusion_inputs_000.npz", allow_pickle=False) as fusion:
        depth = np.asarray(fusion["depth"], dtype=np.float32)
        intrinsics = np.asarray(fusion["intrinsics"], dtype=np.float64)
        native_extrinsics = np.asarray(fusion["extrinsics"], dtype=np.float64)
    with np.load(capture / "alignment_000.npz", allow_pickle=False) as alignment:
        aligned_extrinsics = np.asarray(alignment["aligned_world_to_camera"], dtype=np.float64)
        gt_extrinsics = np.asarray(alignment["reference_world_to_camera"], dtype=np.float64)
        scale = float(alignment["scale"])
    np.testing.assert_allclose(native_extrinsics, aligned_extrinsics, rtol=1e-6, atol=1e-6)
    if not (
        depth.ndim == 3
        and intrinsics.shape == (depth.shape[0], 3, 3)
        and gt_extrinsics.shape == (depth.shape[0], 4, 4)
        and np.isfinite(depth).all()
        and np.isfinite(intrinsics).all()
        and np.isfinite(gt_extrinsics).all()
        and math.isfinite(scale)
        and scale > 0
    ):
        raise ValueError(f"Invalid captured fusion/alignment arrays: {run}")

    sampling_number = int(params["sampling_number"])
    down_sample = float(params["down_sample"])
    native_mesh = o3d.io.read_triangle_mesh(str(capture / "fused_mesh_000.ply"))
    native_resample_points = sample_mesh_for_scannetpp(
        native_mesh,
        gt_points,
        sampling_number=sampling_number,
        down_sample=down_sample,
        seed=seed,
    )
    native_resample_metrics = metrics_from_points(
        gt_points, native_resample_points, eval_threshold
    )

    gt_mesh = fuse_with_extrinsics(
        depth,
        intrinsics,
        gt_extrinsics,
        voxel_length=float(params["voxel_length"]),
        sdf_trunc=float(params["sdf_trunc"]),
        max_depth=float(params["max_depth"]),
    )
    output.mkdir(parents=True, exist_ok=False)
    gt_mesh_path = output / "gt_extrinsics_mesh.ply"
    if not o3d.io.write_triangle_mesh(str(gt_mesh_path), gt_mesh):
        raise IOError(f"Failed to save intervention mesh: {gt_mesh_path}")
    gt_points_predicted = sample_mesh_for_scannetpp(
        gt_mesh,
        gt_points,
        sampling_number=sampling_number,
        down_sample=down_sample,
        seed=seed,
    )
    gt_extrinsics_metrics = metrics_from_points(
        gt_points, gt_points_predicted, eval_threshold
    )

    row = {
        "model": model,
        "dataset": dataset,
        "scene": scene,
        "run": str(run.resolve()),
        "views": int(depth.shape[0]),
        "pose_alignment_scale": scale,
        "seed": seed,
        "dataset_parameters": {
            key: params[key]
            for key in (
                "sampling_number",
                "down_sample",
                "eval_threshold",
                "voxel_length",
                "sdf_trunc",
                "max_depth",
            )
        },
        "official_native_metrics": official_metrics,
        "captured_native_point_count": int(captured_native_points.shape[0]),
        "gt_point_count": int(gt_points.shape[0]),
        "native_resample_point_count": int(native_resample_points.shape[0]),
        "native_resample_metrics": native_resample_metrics,
        "gt_extrinsics_point_count": int(gt_points_predicted.shape[0]),
        "gt_extrinsics_metrics": gt_extrinsics_metrics,
        "gt_extrinsics_delta_vs_native_resample": {
            key: gt_extrinsics_metrics[key] - native_resample_metrics[key]
            for key in gt_extrinsics_metrics
        },
        "gt_extrinsics_delta_vs_official_native": {
            key: gt_extrinsics_metrics[key] - float(official_metrics[key])
            for key in gt_extrinsics_metrics
        },
        "native_resample_delta_vs_official": {
            key: native_resample_metrics[key] - float(official_metrics[key])
            for key in native_resample_metrics
        },
        "capture_sha256": capture_hashes,
        "gt_extrinsics_mesh": gt_mesh_path.name,
        "gt_extrinsics_mesh_sha256": sha256(gt_mesh_path),
        "uses_ground_truth_intervention": True,
        "test_time_policy_input": False,
        "fixed_state": ["scaled predicted depth", "predicted intrinsics", "input population"],
        "changed_state": ["aligned predicted extrinsics -> sampled GT reference extrinsics"],
    }
    (output / "result.json").write_text(
        json.dumps(row, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    del native_mesh, gt_mesh
    gc.collect()
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-jsonl", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--scene", action="append", required=True)
    parser.add_argument("--model", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.dataset != "scannetpp":
        parser.error("the intervention is defined only for scannetpp")
    if len(args.scene) != len(set(args.scene)) or len(args.model) != len(set(args.model)):
        parser.error("scene and model lists must be unique")
    if args.output.exists():
        raise ValueError(f"Refusing to overwrite output: {args.output}")

    lookup = {}
    for row in read_jsonl(args.metrics_jsonl):
        key = (row["model"], row["dataset"], row["scene"])
        if key in lookup:
            raise ValueError(f"Duplicate snapshot metric row: {key}")
        lookup[key] = row
    selected = []
    for scene in args.scene:
        for model in args.model:
            key = (model, args.dataset, scene)
            if key not in lookup:
                raise ValueError(f"Snapshot does not contain selected case: {key}")
            selected.append((model, scene, Path(lookup[key]["run"])))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{args.output.name}-", dir=args.output.parent))
    rows = []
    try:
        for model, scene, run in selected:
            case_output = temporary / f"{model}-{scene}"
            row = analyze_case(run, case_output, seed=args.seed)
            if row["scene"] != scene:
                raise ValueError(f"Scene identity mismatch for {run}")
            rows.append(row)
        results_path = temporary / "results.jsonl"
        results_path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
        manifest = {
            "schema_version": 1,
            "metrics_jsonl": str(args.metrics_jsonl.resolve()),
            "metrics_jsonl_sha256": sha256(args.metrics_jsonl),
            "dataset": args.dataset,
            "scenes": args.scene,
            "models": args.model,
            "cases": len(rows),
            "seed": args.seed,
            "results_sha256": sha256(results_path),
            "analysis_script_sha256": sha256(Path(__file__)),
            "uses_ground_truth_intervention": True,
            "test_time_policy_input": False,
            "intervention": (
                "Keep evaluator-scaled predicted depth and predicted intrinsics; replace "
                "aligned predicted extrinsics with sampled GT reference extrinsics; repeat "
                "the ScanNet++ TSDF and point-distance protocol."
            ),
            "control": (
                "Resample the captured native fused mesh with the same fixed Open3D seed "
                "to quantify point-sampling variance against official native metrics."
            ),
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, args.output)
    except BaseException:
        # Preserve a failed temporary directory for post-mortem evidence.
        (temporary / "FAILED").write_text("intervention did not complete\n", encoding="utf-8")
        raise
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
