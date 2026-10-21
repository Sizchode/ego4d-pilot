"""Run one frozen MapAnything scene through the audited DA3 benchmark protocol.

No repair, model selection, geometric input, ground-truth conditioning, or
adaptive frame reduction is performed.  The official image-only inference path
and its standard ambiguity/edge mask define the primary reconstruction.
"""
import argparse
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import time
import traceback
from types import SimpleNamespace

import numpy as np
import torch

from scripts.da3.run_scene import (
    EvaluationCapture,
    dump,
    npz,
    save_diagnostics,
    save_pose_errors,
    save_previews,
    sha256,
    validate,
)
from scripts.da3.geometry import unproject
from scripts.common.evaluation_canvas import normalize_to_evaluation_canvas
from scripts.mapanything.preprocess import load_images_with_metadata


DATASETS = ("hiroom", "eth3d", "scannetpp", "7scenes", "dtu", "dtu64")
PROTOCOL_MODES = {"dtu": ["recon_unposed"], "dtu64": ["pose"]}


def _numpy(tensor):
    value = tensor.detach().cpu()
    if value.dtype == torch.bfloat16:
        value = value.float()
    return value.numpy()


class MapAnythingAdapter:
    """Evaluator-compatible MapAnything API with lossless raw-output archiving."""

    def __init__(self, model, run_root, cfg):
        self.model = model
        self.run_root = run_root
        self.cfg = cfg
        self.prediction = None

    def inference(self, image_files, *, export_dir, export_format, ref_view_strategy):
        if export_format != "mini_npz":
            raise ValueError(f"Only the audited mini_npz export is supported, received {export_format}")
        raw = self.run_root / "raw"
        network = raw / "network"
        raw.mkdir()
        network.mkdir()

        views, transforms, preprocessing, original_wh, target_wh = load_images_with_metadata(
            image_files, resolution_set=self.cfg["resolution_set"]
        )
        manifest = [
            {
                "index": index,
                "path": str(Path(filename).resolve()),
                "sha256": sha256(filename),
                "original_hw": [original_wh[index][1], original_wh[index][0]],
            }
            for index, filename in enumerate(image_files)
        ]
        dump(self.run_root / "input_manifest.json", manifest)
        np.save(raw / "original_to_processed_pixel.npy", transforms)
        dump(raw / "preprocessing.json", {
            "upstream_function": "mapanything.utils.image.load_images",
            "resize_mode": "fixed_mapping",
            "resolution_set": self.cfg["resolution_set"],
            "sequence_target_wh": list(target_wh),
            "operations": preprocessing,
        })

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        started = time.monotonic()
        start_event.record()
        outputs = self.model.infer(
            views,
            memory_efficient_inference=self.cfg["memory_efficient_inference"],
            minibatch_size=self.cfg["minibatch_size"],
            use_amp=self.cfg["use_amp"],
            amp_dtype=self.cfg["amp_dtype"],
            apply_mask=self.cfg["apply_mask"],
            mask_edges=self.cfg["mask_edges"],
            edge_normal_threshold=self.cfg["edge_normal_threshold"],
            edge_depth_threshold=self.cfg["edge_depth_threshold"],
            apply_confidence_mask=self.cfg["apply_confidence_mask"],
            confidence_percentile=self.cfg["confidence_percentile"],
            use_multiview_confidence=self.cfg["use_multiview_confidence"],
        )
        end_event.record()
        torch.cuda.synchronize()
        dump(self.run_root / "inference_timing.json", {
            "seconds_including_device_transfer": time.monotonic() - started,
            "network_cuda_seconds": start_event.elapsed_time(end_event) / 1000,
        })

        required_keys = {
            "pts3d", "pts3d_cam", "depth_z", "depth_along_ray", "ray_directions",
            "intrinsics", "camera_poses", "cam_trans", "cam_quats", "conf", "mask",
            "non_ambiguous_mask", "non_ambiguous_mask_logits", "metric_scaling_factor",
            "img_no_norm",
        }
        for index, output in enumerate(outputs):
            missing = required_keys - set(output)
            if missing:
                raise RuntimeError(f"MapAnything view {index} is missing official outputs: {sorted(missing)}")
        network_outputs = {
            key: np.stack([_numpy(output[key][0]) for output in outputs])
            for key in sorted(required_keys)
        }
        for key, value in network_outputs.items():
            np.save(network / f"{key}.npy", value)

        network_depth = network_outputs["depth_z"][..., 0]
        network_confidence = network_outputs["conf"]
        network_intrinsics = network_outputs["intrinsics"]
        camera_to_world = network_outputs["camera_poses"].astype(np.float64)
        extrinsics = np.linalg.inv(camera_to_world)[:, :3, :].astype(np.float32)
        official_processed_images = network_outputs["img_no_norm"]

        normalized = normalize_to_evaluation_canvas(
            network_depth,
            network_confidence,
            official_processed_images,
            network_intrinsics,
            original_wh,
            transforms,
            tuple(self.cfg["evaluation_canvas_hw"]),
        )
        depth = normalized["depth"]
        confidence = normalized["confidence"]
        intrinsics = normalized["intrinsics"]
        processed_images = normalized["images"]
        np.save(raw / "processed_to_evaluation_pixel.npy", normalized["processed_to_evaluation"])
        np.save(raw / "original_to_evaluation_pixel.npy", normalized["original_to_evaluation"])
        np.save(raw / "official_processed_images.npy", official_processed_images)

        for name, value in {
            "depth": depth,
            "conf": confidence,
            "extrinsics": extrinsics,
            "intrinsics": intrinsics,
            "processed_images": processed_images,
        }.items():
            np.save(raw / f"{name}.npy", value)
        valid_depth = np.isfinite(depth) & (depth > 0)
        np.save(raw / "valid_depth.npy", valid_depth)

        prediction = SimpleNamespace(
            depth=depth,
            conf=confidence,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            processed_images=processed_images,
            sky=None,
        )
        self.prediction = prediction

        npz(Path(export_dir) / "exports/mini_npz/results.npz",
            depth=np.round(depth, 8), conf=np.round(confidence, 2),
            extrinsics=extrinsics, intrinsics=intrinsics)

        point_dir = raw / "world_points"
        point_dir.mkdir()
        for index, view_depth in enumerate(depth):
            npz(point_dir / f"{index:06d}.npz",
                xyz=unproject(view_depth, intrinsics[index], extrinsics[index]).astype(np.float32),
                rgb=processed_images[index], valid=valid_depth[index])

        dump(raw / "conventions.json", {
            "extrinsics": "OpenCV world-to-camera; X right, Y down, Z forward",
            "depth": "camera Z in the model-predicted metric scale; no GT scale is used",
            "primary_points": "official masked depth_z unprojected with predicted intrinsics and cameras",
            "point_branch": "official pts3d retained at raw/network/pts3d.npy; observational only",
            "camera_pose_source": "official OpenCV camera-to-world output, inverted to benchmark world-to-camera",
            "official_mask": "non-ambiguous mask combined with upstream normal/depth edge mask",
            "confidence_mask": "disabled, matching the official image-only example",
            "evaluation_raster": "official prediction deterministically warped to a full-image square canvas",
            "invalid_crop_regions": "zero depth; never filled or hallucinated",
            "image_hw": list(depth.shape[-2:]),
            "preprocessing_tensor_shape": list(views[0]["img"].shape),
            "confidence": "raw learning-based confidence; not calibrated across models",
            "matching_or_tracks": "track head not queried by this baseline",
            "reference_view_argument": ref_view_strategy,
            "reference_view_effect": "none; MapAnything jointly predicts the ordered image set",
            "gt_conditioning": False,
        })
        del outputs, views
        torch.cuda.empty_cache()
        return prediction


def _source_hashes():
    files = (
        [Path(__file__).parents[1] / "__init__.py"]
        + list(Path(__file__).parent.glob("*.py"))
        + list((Path(__file__).parents[1] / "common").glob("*.py"))
        + list((Path(__file__).parents[1] / "da3").glob("*.py"))
    )
    files = [path for path in files if not path.name.startswith("._")]
    return {str(path.relative_to(Path(__file__).parents[2])): sha256(path) for path in files}


def _git_revision(path):
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def _validate_dinov2_dependency(root, cfg):
    checkout = (root / "code/DINOv2").resolve()
    cache_entry = root / "cache/torch/hub" / cfg["dinov2_torch_hub_cache_name"]
    if not cache_entry.is_symlink() or cache_entry.resolve() != checkout:
        raise ValueError(
            f"Torch Hub DINOv2 cache must resolve to the pinned checkout {checkout}: {cache_entry}"
        )
    revision = _git_revision(checkout)
    if revision != cfg["dinov2_revision"]:
        raise ValueError(f"DINOv2 revision differs from pinned configuration: {revision}")
    subprocess.run(["git", "-C", str(checkout), "diff", "--exit-code", "HEAD"], check=True)
    return revision


def _validate_mapanything_artifacts(run_root, dataset, scene, modes):
    metrics = validate(run_root, dataset, scene, modes)
    expected = [
        run_root / "raw/network/depth_z.npy",
        run_root / "raw/network/pts3d.npy",
        run_root / "raw/network/pts3d_cam.npy",
        run_root / "raw/network/camera_poses.npy",
        run_root / "raw/network/intrinsics.npy",
        run_root / "raw/network/mask.npy",
        run_root / "raw/network/non_ambiguous_mask_logits.npy",
        run_root / "raw/preprocessing.json",
    ]
    for path in expected:
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Missing MapAnything raw artifact: {path}")
    world_points = np.load(run_root / "raw/network/pts3d.npy", mmap_mode="r")
    network_depth = np.load(run_root / "raw/network/depth_z.npy", mmap_mode="r")
    if world_points.shape[:3] != network_depth.shape[:3]:
        raise ValueError(
            f"Point-branch shape {world_points.shape} does not match network depth {network_depth.shape}"
        )
    validation_path = run_root / "validation.json"
    record = json.loads(validation_path.read_text())
    record["mapanything_raw_outputs_present"] = True
    record["primary_geometry_branch"] = "official_masked_depth_z_plus_predicted_camera"
    dump(validation_path, record)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--scene", required=True, help="Exact scene ID, or @first")
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    run_root = args.root / "runs" / args.run_id
    run_root.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    stage = "initialization"
    try:
        cfg = json.loads(args.config.read_text())
        mapanything_code = args.root / "code/MapAnything"
        benchmark_code = args.root / "code/Depth-Anything-3"
        actual_revisions = {
            "mapanything": _git_revision(mapanything_code),
            "dinov2": _validate_dinov2_dependency(args.root, cfg),
            "benchmark": _git_revision(benchmark_code),
        }
        expected_revisions = {
            "mapanything": cfg["code_revision"],
            "dinov2": cfg["dinov2_revision"],
            "benchmark": cfg["benchmark_code_revision"],
        }
        if actual_revisions != expected_revisions:
            raise ValueError(f"Code revisions differ from pinned configuration: {actual_revisions}")
        subprocess.run(["git", "-C", str(mapanything_code), "diff", "--exit-code", "HEAD"], check=True)
        subprocess.run(["git", "-C", str(benchmark_code), "diff", "--exit-code", "HEAD"], check=True)
        os.chdir(benchmark_code)

        from depth_anything_3.bench.evaluator import Evaluator
        from depth_anything_3.bench.registries import MV_REGISTRY
        from mapanything.models import MapAnything
        import open3d as o3d

        random.seed(cfg["sampling_seed"])
        np.random.seed(cfg["sampling_seed"])
        torch.manual_seed(cfg["sampling_seed"])
        o3d.utility.random.seed(cfg["sampling_seed"])
        torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))

        dataset = MV_REGISTRY.get(args.dataset)()
        scene = dataset.SCENES[0] if args.scene == "@first" else args.scene
        if scene not in dataset.SCENES:
            raise ValueError(f"Scene not in official manifest: {scene}")
        modes = PROTOCOL_MODES.get(args.dataset, ["pose", "recon_unposed"])
        evaluator = Evaluator(
            work_dir=str(run_root / "evaluation"), datas=[args.dataset], modes=modes,
            scenes=[scene], max_frames=cfg["max_frames"],
            ref_view_strategy=cfg["ref_view_strategy"], num_fusion_workers=1,
        )
        dump(run_root / "run_config.json", {
            "model": cfg["model_repository"],
            "dataset": args.dataset,
            "scene": scene,
            "modes": modes,
            "config": cfg,
            "gpu": torch.cuda.get_device_name(),
            "dataset_parameters": {
                key: value for key, value in vars(type(dataset)).items()
                if not key.startswith("_") and isinstance(value, (str, int, float, bool))
            },
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "scratch_only": True,
            "has_agent": False,
        })
        for source, destination in [
            (args.root / "provenance/mapanything_assets.json", run_root / "mapanything_assets.json"),
            (args.root / "provenance/mapanything_environment.txt", run_root / "mapanything_environment.txt"),
            (args.root / "provenance/assets.json", run_root / "benchmark_assets.json"),
        ]:
            shutil.copy2(source, destination)
        dump(run_root / "runner_sources.json", _source_hashes())
        shutil.copytree(Path(__file__).parent, run_root / "runner_source/mapanything", ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copytree(Path(__file__).parents[1] / "common", run_root / "runner_source/common", ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copytree(Path(__file__).parents[1] / "da3", run_root / "runner_source/da3", ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copy2(Path(__file__).parents[1] / "__init__.py", run_root / "runner_source/__init__.py")
        dump(run_root / "instrumentation.json", {
            "model_changes": None,
            "model_inputs": "images only; no intrinsics, depth, pose, scale, or GT metadata",
            "primary_output": "official masked depth_z plus predicted camera unprojection",
            "secondary_output": "official pts3d/pts3d_cam branches archived but not evaluated or selected",
            "official_inference_options": {
                key: cfg[key]
                for key in [
                    "memory_efficient_inference", "minibatch_size", "use_amp", "amp_dtype",
                    "apply_mask", "mask_edges", "edge_normal_threshold", "edge_depth_threshold",
                    "apply_confidence_mask", "confidence_percentile", "use_multiview_confidence",
                ]
            },
            "synchronous_export": "mini_npz written before evaluator proceeds",
            "failure_change": "invalid mesh raises; no replacement random points",
            "evaluation_random_seed": cfg["sampling_seed"],
            "gt_metadata": "evaluator-owned files under evaluation; never passed to model",
            "additional_diagnostics": "observational only; not used to alter predictions",
        })

        stage = "model_load"
        torch.cuda.reset_peak_memory_stats()
        model = MapAnything.from_pretrained(str(args.root / "models/MapAnything")).to("cuda").eval()
        adapter = MapAnythingAdapter(model, run_root, cfg)
        stage = "inference"
        evaluator.infer(adapter)
        torch.cuda.synchronize()
        inference_end = time.monotonic()

        stage = "evaluation"
        capture = EvaluationCapture(run_root / "evaluation_capture", benchmark_code)
        with capture.active(dataset):
            evaluator.eval()
        gt = evaluator._sample_frames(dataset.get_data(scene), scene)
        save_pose_errors(adapter.prediction, gt, run_root / "evaluation_capture")
        evaluation_end = time.monotonic()

        stage = "diagnostics"
        save_diagnostics(adapter.prediction, run_root / "diagnostics")
        save_previews(adapter.prediction, run_root)
        dump(run_root / "runtime.json", {
            "seconds_until_inference_complete": inference_end - started,
            "evaluation_seconds": evaluation_end - inference_end,
            "diagnostics_seconds": time.monotonic() - evaluation_end,
            "total_seconds": time.monotonic() - started,
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
            "gpu": torch.cuda.get_device_name(),
        })
        stage = "artifact_validation"
        metrics = _validate_mapanything_artifacts(run_root, args.dataset, scene, modes)
        dump(run_root / "status.json", {
            "state": "completed", "dataset": args.dataset, "scene": scene,
            "elapsed_seconds": time.monotonic() - started,
        })
        print(json.dumps({"run": str(run_root), "validated": True, "metrics": metrics}), flush=True)
    except Exception as exc:
        dump(run_root / "status.json", {
            "state": "failed", "stage": stage, "type": type(exc).__name__,
            "message": str(exc), "elapsed_seconds": time.monotonic() - started,
        })
        (run_root / "traceback.txt").write_text(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
