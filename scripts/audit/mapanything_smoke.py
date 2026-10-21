"""Reproduce the numerical and provenance checks for one MapAnything smoke run.

This script does not approve a run.  It writes the machine-checkable evidence
that accompanies a separate visual review of the saved preview images.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


NETWORK_OUTPUTS = {
    "camera_poses.npy",
    "cam_quats.npy",
    "cam_trans.npy",
    "conf.npy",
    "depth_along_ray.npy",
    "depth_z.npy",
    "img_no_norm.npy",
    "intrinsics.npy",
    "mask.npy",
    "metric_scaling_factor.npy",
    "non_ambiguous_mask.npy",
    "non_ambiguous_mask_logits.npy",
    "pts3d.npy",
    "pts3d_cam.npy",
    "ray_directions.npy",
}


def load_json(path: Path):
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def auc(rotation: np.ndarray, translation: np.ndarray, threshold: int) -> float:
    maximum = np.maximum(rotation, translation)
    histogram, _ = np.histogram(maximum, bins=np.arange(threshold + 1))
    return float(np.mean(np.cumsum(histogram.astype(float) / len(maximum))))


def reconstruction_metrics(accuracy: np.ndarray, completeness: np.ndarray) -> dict:
    threshold = 0.05
    acc = float(np.mean(accuracy))
    comp = float(np.mean(completeness))
    precision = float(np.mean(accuracy < threshold))
    recall = float(np.mean(completeness < threshold))
    fscore = 2 * precision * recall / (precision + recall)
    return {
        "acc": acc,
        "comp": comp,
        "overall": (acc + comp) / 2,
        "precision": precision,
        "recall": recall,
        "fscore": fscore,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--runner-root", type=Path, required=True)
    parser.add_argument("--slurm-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    run = args.run.resolve()
    status = load_json(run / "status.json")
    validation = load_json(run / "validation.json")
    run_config = load_json(run / "run_config.json")
    assets = load_json(run / "mapanything_assets.json")
    instrumentation = load_json(run / "instrumentation.json")
    require(status["state"] == "completed", "Smoke status is not completed")
    require(validation["passed"], "Smoke validation did not pass")
    require(validation.get("mapanything_raw_outputs_present"), "Raw-output validation is absent")
    require(not run_config["has_agent"], "Baseline smoke unexpectedly contains an agent")
    require(instrumentation["model_changes"] is None, "Baseline reports model changes")

    pinned_config = load_json(args.runner_root / "configs/mapanything_pinned.json")
    require(run_config["config"] == pinned_config, "Run configuration differs from pinned config")
    expected_revisions = {
        "mapanything": pinned_config["code_revision"],
        "dinov2": pinned_config["dinov2_revision"],
        "benchmark": pinned_config["benchmark_code_revision"],
    }
    require(assets["code_revisions"] == expected_revisions, "Code revision provenance differs")
    model_record = assets["model_files"][pinned_config["model_filename"]]
    require(model_record["sha256"] == pinned_config["model_sha256"], "Weight digest differs")

    manifest = load_json(run / "input_manifest.json")
    require(len(manifest) == validation["views"], "Input and prediction counts differ")
    require(
        all(sha256(Path(item["path"])) == item["sha256"] for item in manifest),
        "At least one input image hash differs",
    )

    network_dir = run / "raw/network"
    network_outputs = {path.name for path in network_dir.glob("*.npy")}
    require(network_outputs == NETWORK_OUTPUTS, "Network output archive is incomplete or unexpected")
    depth = np.load(run / "raw/depth.npy", mmap_mode="r")
    confidence = np.load(run / "raw/conf.npy", mmap_mode="r")
    extrinsics = np.load(run / "raw/extrinsics.npy")
    intrinsics = np.load(run / "raw/intrinsics.npy")
    require(depth.shape == tuple(validation["shape"]), "Depth shape differs from validation")
    require(np.isfinite(extrinsics).all() and np.isfinite(intrinsics).all(), "Non-finite camera")
    determinant_error = float(np.max(np.abs(np.linalg.det(extrinsics[:, :3, :3]) - 1)))
    require(determinant_error <= 1e-3, "Camera rotation determinant check failed")

    camera_to_world = np.load(network_dir / "camera_poses.npy").astype(np.float64)
    inverted = np.linalg.inv(camera_to_world)[:, :3, :]
    np.testing.assert_allclose(inverted, extrinsics, rtol=0, atol=1e-6)
    require(len(list((run / "raw/world_points").glob("*.npz"))) == len(depth), "Point archive count differs")
    protocol = load_json(run / "diagnostics/protocol.json")
    diagnostic_files = list((run / "diagnostics").glob("*.npz"))
    require(len(diagnostic_files) == len(protocol["pairs"]), "Diagnostic pair archive differs")

    pairs = np.load(run / "evaluation_capture/camera_pairs.npz")
    expected_pairs = len(depth) * (len(depth) - 1) // 2
    require(len(pairs["rotation_degrees"]) == expected_pairs, "Camera pair count differs")
    require(
        np.isfinite(pairs["rotation_degrees"]).all()
        and np.isfinite(pairs["translation_degrees"]).all(),
        "Camera error contains non-finite values",
    )
    pose = {
        "auc30": auc(pairs["rotation_degrees"], pairs["translation_degrees"], 30),
        "auc15": auc(pairs["rotation_degrees"], pairs["translation_degrees"], 15),
        "auc05": auc(pairs["rotation_degrees"], pairs["translation_degrees"], 5),
        "auc03": auc(pairs["rotation_degrees"], pairs["translation_degrees"], 3),
    }
    captured = sorted((run / "evaluation_capture").glob("surface_distance_*.npz"))
    require(len(captured) == 2, "Expected two directed surface-distance captures")
    with np.load(captured[0]) as values:
        accuracy = values["distance"]
    with np.load(captured[1]) as values:
        completeness = values["distance"]
    reconstruction = reconstruction_metrics(accuracy, completeness)
    official = validation["metrics"]
    for key, value in pose.items():
        np.testing.assert_allclose(value, official["pose"][key], rtol=0, atol=1e-12)
    for key, value in reconstruction.items():
        np.testing.assert_allclose(value, official["recon_unposed"][key], rtol=0, atol=1e-12)

    exported_path = (
        run / "evaluation/model_results" / run_config["dataset"] / run_config["scene"]
        / "unposed/exports/mini_npz/results.npz"
    )
    with np.load(exported_path) as exported:
        np.testing.assert_array_equal(exported["depth"], np.round(depth, 8))
        np.testing.assert_array_equal(exported["conf"], np.round(confidence, 2))
        np.testing.assert_array_equal(exported["extrinsics"], extrinsics)
        np.testing.assert_array_equal(exported["intrinsics"], intrinsics)

    saved_hashes = load_json(run / "runner_sources.json")
    current_hashes = {key: sha256(args.runner_root / key) for key in saved_hashes}
    require(current_hashes == saved_hashes, "Per-scene runner source changed after smoke")

    capture_counts = load_json(run / "evaluation_capture/capture_counts.json")
    require(capture_counts.get("fusion_inputs") == 1, "Fusion input capture count differs")
    require(capture_counts.get("alignment") == 1, "Alignment capture count differs")
    require(capture_counts.get("fused_mesh") == 1, "Fused mesh capture count differs")
    require(capture_counts.get("surface_distance") == 2, "Surface-distance capture count differs")

    log = args.slurm_log.read_text(errors="replace")
    require("Using cache found in" in log, "Pinned Torch Hub cache use is absent from log")
    require("Downloading:" not in log, "Smoke attempted a network download")
    preview = Image.open(run / "preview.png")
    camera_errors = Image.open(run / "camera_errors.png")

    result = {
        "automated_checks_passed": True,
        "run": str(run),
        "job_id": run_config["slurm_job_id"],
        "scene": run_config["scene"],
        "checks": {
            "status_completed": True,
            "validation_passed": True,
            "offline_torch_hub_cache_verified": True,
            "input_manifest_hashes_match": True,
            "views": len(depth),
            "network_outputs": sorted(network_outputs),
            "world_point_archives": len(depth),
            "directed_adjacent_diagnostics": len(protocol["pairs"]),
            "camera_pairs": expected_pairs,
            "max_rotation_determinant_error": determinant_error,
            "camera_inverse_equals_evaluator_extrinsics": True,
            "surface_distance_counts": [len(accuracy), len(completeness)],
            "saved_errors_reproduce_all_official_metrics_at_atol_1e-12": True,
            "adapter_export_equals_archived_primary_values": True,
            "runner_hashes_match_current_sources": True,
            "code_revisions": expected_revisions,
            "model_sha256": model_record["sha256"],
            "preview": {"size": list(preview.size), "mode": preview.mode},
            "camera_errors": {"size": list(camera_errors.size), "mode": camera_errors.mode},
        },
        "official_metrics": official,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
