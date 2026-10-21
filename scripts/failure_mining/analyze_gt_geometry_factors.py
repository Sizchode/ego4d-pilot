"""Factor ScanNet++ geometry error with nested ground-truth interventions.

Starting from an immutable GT-extrinsics intervention, this offline diagnostic
adds ground-truth intrinsics and then ground-truth depth. The ordered image
population and official TSDF/evaluation protocol stay fixed. Ground truth is
never exposed to a deployable policy.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np

from scripts.failure_mining.analyze_gt_extrinsics_intervention import (
    fuse_with_extrinsics,
    metrics_from_points,
    sample_mesh_for_scannetpp,
    sha256,
    verify_official_capture,
)


METRICS = ("acc", "comp", "overall", "precision", "recall", "fscore")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def image_key(path: str | os.PathLike[str], scene: str) -> str:
    normalized = str(path).replace("\\", "/")
    marker = f"/{scene}/merge_dslr_iphone/"
    with_leading = f"/{normalized.lstrip('/')}"
    require(marker in with_leading, f"Image path does not identify scene {scene}: {path}")
    return with_leading.split(marker, 1)[1]


def prepare_gt_intrinsics(
    intrinsics: np.ndarray,
    camera_hw: list[tuple[int, int]],
    *,
    input_h: int,
    input_w: int,
) -> np.ndarray:
    intrinsics = np.asarray(intrinsics, dtype=np.float64)
    require(intrinsics.ndim == 3 and intrinsics.shape[1:] == (3, 3), "Invalid GT intrinsics")
    require(len(camera_hw) == intrinsics.shape[0], "GT intrinsics/camera-size population differs")
    output = []
    for matrix, (camera_h, camera_w) in zip(intrinsics, camera_hw, strict=True):
        require(camera_h > 0 and camera_w > 0, "Camera dimensions must be positive")
        adjusted = matrix.copy()
        adjusted[:2, 2] += 0.5
        adjusted[0, :] *= input_w / camera_w
        adjusted[1, :] *= input_h / camera_h
        output.append(adjusted)
    stacked = np.stack(output)
    require(np.isfinite(stacked).all(), "Prepared GT intrinsics are non-finite")
    return stacked


def load_gt_depth(path: Path, *, input_h: int, input_w: int) -> np.ndarray:
    require(path.is_file(), f"Missing GT depth: {path}")
    depth = np.asarray(imageio.imread(path), dtype=np.float64) / 1000.0
    require(depth.ndim == 2, f"GT depth must be one-channel: {path}")
    if depth.shape != (input_h, input_w):
        import cv2

        depth = cv2.resize(
            depth, (input_w, input_h), interpolation=cv2.INTER_LINEAR
        )
    depth = depth.astype(np.float32)
    valid = np.isfinite(depth) & (depth > 0)
    depth = np.where(valid, depth, 0.0).astype(np.float32)
    return depth


def load_extrinsics_reference(reference_dir: Path) -> tuple[dict[tuple[str, str], dict[str, float]], dict[str, Any]]:
    manifest_path = reference_dir / "manifest.json"
    cases_path = reference_dir / "case_metric_deltas.csv"
    require(manifest_path.is_file(), f"Missing reference manifest: {manifest_path}")
    require(cases_path.is_file(), f"Missing reference case table: {cases_path}")
    manifest = read_json(manifest_path)
    require(manifest.get("uses_ground_truth_intervention") is True, "Reference is not a GT intervention")
    require(manifest.get("test_time_policy_input") is False, "Reference is marked as policy input")
    require(
        manifest.get("case_metric_output_sha256") == sha256(cases_path),
        "Reference case-table hash does not match manifest",
    )
    grouped: dict[tuple[str, str], dict[str, float]] = {}
    with cases_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (row["model"], row["scene"])
            entry = grouped.setdefault(key, {})
            metric = row["metric"]
            require(metric in METRICS, f"Unknown reference metric: {metric}")
            require(f"official_{metric}" not in entry, f"Duplicate reference metric: {key}, {metric}")
            entry[f"official_{metric}"] = float(row["official_native"])
            entry[f"extrinsics_{metric}"] = float(row["gt_extrinsics_intervention"])
    for key, entry in grouped.items():
        require(
            set(entry)
            == {f"official_{metric}" for metric in METRICS}
            | {f"extrinsics_{metric}" for metric in METRICS},
            f"Incomplete reference metrics: {key}",
        )
    return grouped, {
        "reference_manifest_sha256": sha256(manifest_path),
        "reference_case_metrics_sha256": sha256(cases_path),
    }


def gt_meta_path(run: Path, dataset: str, scene: str) -> Path:
    return (
        run
        / "evaluation"
        / "model_results"
        / dataset
        / scene
        / "unposed"
        / "exports"
        / "gt_meta.npz"
    )


def load_scene_gt_inputs(
    run: Path,
    dataset_object: Any,
    scene: str,
    *,
    input_h: int,
    input_w: int,
) -> dict[str, Any]:
    meta_path = gt_meta_path(run, "scannetpp", scene)
    require(meta_path.is_file(), f"Missing sampled GT metadata: {meta_path}")
    with np.load(meta_path, allow_pickle=True) as meta:
        require(set(meta.files) == {"extrinsics", "intrinsics", "image_files"}, "Unexpected GT meta schema")
        meta_extrinsics = np.asarray(meta["extrinsics"], dtype=np.float64)
        meta_intrinsics = np.asarray(meta["intrinsics"], dtype=np.float64)
        meta_images = [str(value) for value in meta["image_files"].tolist()]

    full = dataset_object.get_data(scene)
    full_keys = [image_key(path, scene) for path in full.image_files]
    require(len(full_keys) == len(set(full_keys)), f"Duplicate GT image key: {scene}")
    index_by_key = {key: index for index, key in enumerate(full_keys)}
    sampled_keys = [image_key(path, scene) for path in meta_images]
    require(len(sampled_keys) == len(set(sampled_keys)), f"Duplicate sampled image key: {scene}")
    missing = [key for key in sampled_keys if key not in index_by_key]
    require(not missing, f"Sampled images are absent from GT dataset for {scene}: {missing[:3]}")
    indices = [index_by_key[key] for key in sampled_keys]

    sampled_extrinsics = np.asarray(full.extrinsics, dtype=np.float64)[indices]
    sampled_intrinsics = np.asarray(full.intrinsics, dtype=np.float64)[indices]
    np.testing.assert_allclose(meta_extrinsics, sampled_extrinsics, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(meta_intrinsics, sampled_intrinsics, rtol=1e-6, atol=1e-6)

    camera_hw = [tuple(full.aux.cam_hw_list[index]) for index in indices]
    prepared_intrinsics = prepare_gt_intrinsics(
        sampled_intrinsics, camera_hw, input_h=input_h, input_w=input_w
    )
    depth_paths = [Path(full.aux.gt_depth_files[index]) for index in indices]
    depths = np.stack(
        [load_gt_depth(path, input_h=input_h, input_w=input_w) for path in depth_paths]
    )
    ordered_depth_hash = hashlib.sha256()
    for key, path in zip(sampled_keys, depth_paths, strict=True):
        ordered_depth_hash.update(key.encode("utf-8"))
        ordered_depth_hash.update(b"\0")
        ordered_depth_hash.update(sha256(path).encode("ascii"))
        ordered_depth_hash.update(b"\n")
    image_population_sha256 = hashlib.sha256(
        "\n".join(sampled_keys).encode("utf-8")
    ).hexdigest()
    return {
        "image_keys": sampled_keys,
        "image_population_sha256": image_population_sha256,
        "gt_meta_sha256": sha256(meta_path),
        "gt_depth_population_sha256": ordered_depth_hash.hexdigest(),
        "extrinsics": meta_extrinsics,
        "intrinsics": prepared_intrinsics,
        "depth": depths,
    }


def validate_case_gt_meta(run: Path, scene_gt: dict[str, Any], scene: str) -> None:
    meta_path = gt_meta_path(run, "scannetpp", scene)
    require(meta_path.is_file(), f"Missing sampled GT metadata: {meta_path}")
    with np.load(meta_path, allow_pickle=True) as meta:
        keys = [image_key(path, scene) for path in meta["image_files"].tolist()]
        extrinsics = np.asarray(meta["extrinsics"], dtype=np.float64)
    require(keys == scene_gt["image_keys"], f"Sampled image population differs: {run}")
    np.testing.assert_allclose(extrinsics, scene_gt["extrinsics"], rtol=1e-6, atol=1e-6)


def intervention_metrics(
    depth: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    gt_points: np.ndarray,
    params: dict[str, Any],
    output_path: Path,
    *,
    seed: int,
) -> tuple[dict[str, float], int, str]:
    import open3d as o3d

    mesh = fuse_with_extrinsics(
        depth,
        intrinsics,
        extrinsics,
        voxel_length=float(params["voxel_length"]),
        sdf_trunc=float(params["sdf_trunc"]),
        max_depth=float(params["max_depth"]),
    )
    if not o3d.io.write_triangle_mesh(str(output_path), mesh):
        raise IOError(f"Failed to save intervention mesh: {output_path}")
    points = sample_mesh_for_scannetpp(
        mesh,
        gt_points,
        sampling_number=int(params["sampling_number"]),
        down_sample=float(params["down_sample"]),
        seed=seed,
    )
    metrics = metrics_from_points(gt_points, points, float(params["eval_threshold"]))
    mesh_hash = sha256(output_path)
    point_count = int(points.shape[0])
    del mesh, points
    gc.collect()
    return metrics, point_count, mesh_hash


def analyze_case(
    run: Path,
    reference: dict[str, float],
    scene_gt: dict[str, Any],
    output: Path,
    *,
    model_id: str,
    seed: int,
) -> dict[str, Any]:
    config = read_json(run / "run_config.json")
    status = read_json(run / "status.json")
    validation = read_json(run / "validation.json")
    dataset, scene = config.get("dataset"), config.get("scene")
    model_repository = config.get("model") or config.get("config", {}).get("model_repository")
    require(dataset == "scannetpp", f"Unexpected dataset: {run}")
    require(status.get("state") == "completed" and validation.get("passed") is True, f"Invalid run: {run}")
    require(
        isinstance(scene, str)
        and isinstance(model_id, str)
        and model_id
        and isinstance(model_repository, str),
        f"Incomplete run identity: {run}",
    )
    validate_case_gt_meta(run, scene_gt, scene)

    params = config["dataset_parameters"]
    official = {key: float(value) for key, value in validation["metrics"]["recon_unposed"].items()}
    require(set(official) == set(METRICS), f"Unexpected official metrics: {run}")
    reference_official = {metric: reference[f"official_{metric}"] for metric in METRICS}
    for metric in METRICS:
        require(
            math.isclose(official[metric], reference_official[metric], rel_tol=0.0, abs_tol=1e-12),
            f"Official/reference mismatch for {model_id}, {scene}, {metric}",
        )

    capture = run / "evaluation_capture"
    gt_points, _native_points, capture_hashes = verify_official_capture(
        capture, official, float(params["eval_threshold"])
    )
    with np.load(capture / "fusion_inputs_000.npz", allow_pickle=False) as fusion:
        predicted_depth = np.asarray(fusion["depth"], dtype=np.float32)
    with np.load(capture / "alignment_000.npz", allow_pickle=False) as alignment:
        captured_gt_extrinsics = np.asarray(
            alignment["reference_world_to_camera"], dtype=np.float64
        )
    require(predicted_depth.shape == scene_gt["depth"].shape, f"Depth population differs: {run}")
    np.testing.assert_allclose(
        captured_gt_extrinsics, scene_gt["extrinsics"], rtol=1e-6, atol=1e-6
    )

    output.mkdir(parents=True, exist_ok=False)
    ek_metrics, ek_points, ek_hash = intervention_metrics(
        predicted_depth,
        scene_gt["intrinsics"],
        scene_gt["extrinsics"],
        gt_points,
        params,
        output / "gt_extrinsics_intrinsics_mesh.ply",
        seed=seed,
    )
    ekd_metrics, ekd_points, ekd_hash = intervention_metrics(
        scene_gt["depth"],
        scene_gt["intrinsics"],
        scene_gt["extrinsics"],
        gt_points,
        params,
        output / "gt_extrinsics_intrinsics_depth_mesh.ply",
        seed=seed,
    )
    result = {
        "model": model_id,
        "model_repository": model_repository,
        "dataset": dataset,
        "scene": scene,
        "run": str(run.resolve()),
        "views": int(predicted_depth.shape[0]),
        "seed": seed,
        "official_native_metrics": official,
        "gt_extrinsics_metrics": {
            metric: reference[f"extrinsics_{metric}"] for metric in METRICS
        },
        "gt_extrinsics_intrinsics_metrics": ek_metrics,
        "gt_extrinsics_intrinsics_depth_metrics": ekd_metrics,
        "gt_point_count": int(gt_points.shape[0]),
        "gt_extrinsics_intrinsics_point_count": ek_points,
        "gt_extrinsics_intrinsics_depth_point_count": ekd_points,
        "gt_extrinsics_intrinsics_mesh_sha256": ek_hash,
        "gt_extrinsics_intrinsics_depth_mesh_sha256": ekd_hash,
        "capture_sha256": capture_hashes,
        "gt_meta_sha256": scene_gt["gt_meta_sha256"],
        "image_population_sha256": scene_gt["image_population_sha256"],
        "gt_depth_population_sha256": scene_gt["gt_depth_population_sha256"],
        "stages": [
            "official_native",
            "gt_extrinsics",
            "gt_extrinsics_intrinsics",
            "gt_extrinsics_intrinsics_depth",
        ],
        "uses_ground_truth_intervention": True,
        "test_time_policy_input": False,
    }
    (output / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-jsonl", type=Path, required=True)
    parser.add_argument("--extrinsics-reference", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--scene", action="append", required=True)
    parser.add_argument("--model", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    require(args.dataset_root.is_dir(), f"Dataset root does not exist: {args.dataset_root}")
    require(len(args.scene) == len(set(args.scene)), "Scenes must be unique")
    require(len(args.model) == len(set(args.model)), "Models must be unique")
    require(not args.output.exists(), f"Refusing to overwrite output: {args.output}")

    reference, reference_provenance = load_extrinsics_reference(args.extrinsics_reference)
    lookup: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in read_jsonl(args.metrics_jsonl):
        key = (row["model"], row["dataset"], row["scene"])
        require(key not in lookup, f"Duplicate snapshot row: {key}")
        lookup[key] = row

    selected: list[tuple[str, str, Path, str]] = []
    for scene in args.scene:
        for model in args.model:
            key = (model, "scannetpp", scene)
            require(key in lookup, f"Snapshot does not contain selected case: {key}")
            run = Path(lookup[key]["run"])
            config = read_json(run / "run_config.json")
            full_model = config.get("model") or config.get("config", {}).get("model_repository")
            require((full_model, scene) in reference, f"Reference lacks case: {(full_model, scene)}")
            selected.append((model, scene, run, full_model))

    from depth_anything_3.bench.datasets.scannetpp import ScanNetPP

    dataset_object = ScanNetPP()
    dataset_object.data_root = str(args.dataset_root.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{args.output.name}-", dir=args.output.parent))
    scene_cache: dict[str, dict[str, Any]] = {}
    rows = []
    try:
        for _model, scene, run, full_model in selected:
            config = read_json(run / "run_config.json")
            params = config["dataset_parameters"]
            if scene not in scene_cache:
                scene_cache[scene] = load_scene_gt_inputs(
                    run,
                    dataset_object,
                    scene,
                    input_h=int(params["input_h"]),
                    input_w=int(params["input_w"]),
                )
            row = analyze_case(
                run,
                reference[(full_model, scene)],
                scene_cache[scene],
                temporary / f"{_model}-{scene}",
                model_id=_model,
                seed=args.seed,
            )
            rows.append(row)
        results_path = temporary / "results.jsonl"
        results_path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
        manifest = {
            "dataset": "scannetpp",
            "scenes": args.scene,
            "models": args.model,
            "cases": len(rows),
            "seed": args.seed,
            "metrics_jsonl": str(args.metrics_jsonl.resolve()),
            "metrics_jsonl_sha256": sha256(args.metrics_jsonl),
            "extrinsics_reference": str(args.extrinsics_reference.resolve()),
            **reference_provenance,
            "dataset_root": str(args.dataset_root.resolve()),
            "results_sha256": sha256(results_path),
            "analysis_script_sha256": sha256(Path(__file__)),
            "stages": [
                "official_native",
                "gt_extrinsics",
                "gt_extrinsics_intrinsics",
                "gt_extrinsics_intrinsics_depth",
            ],
            "uses_ground_truth_intervention": True,
            "test_time_policy_input": False,
            "success_threshold": None,
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, args.output)
        print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    except BaseException:
        import shutil

        shutil.rmtree(temporary, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()
