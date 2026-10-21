"""Run one frozen Pi3X scene through the audited DA3 benchmark protocol.

No repair, model selection, multimodal condition, ground-truth conditioning,
or adaptive frame reduction is performed. The official image-only example
defines intrinsic recovery and confidence/depth-edge masking.
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
from scripts.pi3x.preprocess import stage_and_load_images


DATASETS = ("hiroom", "eth3d", "scannetpp", "7scenes", "dtu", "dtu64")
PROTOCOL_MODES = {"dtu": ["recon_unposed"], "dtu64": ["pose"]}


def _numpy(tensor):
    value = tensor.detach().cpu()
    if value.dtype == torch.bfloat16:
        value = value.float()
    return value.numpy()


class Pi3XAdapter:
    """Evaluator-compatible Pi3X API with complete raw-output archiving."""

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

        images_cpu, transforms, preprocessing, original_wh, target_wh = stage_and_load_images(
            image_files,
            self.run_root / "input_stage",
            pixel_limit=self.cfg["pixel_limit"],
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
            "upstream_function": "pi3.utils.basic.load_images_as_tensor",
            "pixel_limit": self.cfg["pixel_limit"],
            "patch_size": 14,
            "sequence_target_wh": list(target_wh),
            "operations": preprocessing,
        })

        images = images_cpu[None].to("cuda", non_blocking=False)
        dtype = torch.bfloat16 if self.cfg["inference_dtype"] == "bfloat16" else torch.float16
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        started = time.monotonic()
        start_event.record()
        with torch.inference_mode(), torch.amp.autocast("cuda", dtype=dtype):
            outputs = self.model(
                imgs=images,
                depths=None,
                intrinsics=None,
                rays=None,
                poses=None,
            )
        end_event.record()
        torch.cuda.synchronize()
        dump(self.run_root / "inference_timing.json", {
            "seconds_including_device_transfer": time.monotonic() - started,
            "network_cuda_seconds": start_event.elapsed_time(end_event) / 1000,
        })

        required_keys = {"points", "local_points", "rays", "conf", "camera_poses", "metric"}
        missing = required_keys - set(outputs)
        if missing:
            raise RuntimeError(f"Pi3X is missing official outputs: {sorted(missing)}")
        for key in sorted(required_keys):
            np.save(network / f"{key}.npy", _numpy(outputs[key]))

        from pi3.utils.geometry import depth_normal_edge, recover_intrinsic_from_rays_d

        ray_directions = torch.nn.functional.normalize(outputs["local_points"], dim=-1)
        recovered_intrinsics = recover_intrinsic_from_rays_d(
            ray_directions, force_center_principal_point=True
        )
        confidence_probability = torch.sigmoid(outputs["conf"][..., 0])
        confidence_mask = confidence_probability > self.cfg["confidence_probability_threshold"]
        non_edge_mask = ~depth_normal_edge(
            outputs["local_points"],
            rtol=self.cfg["edge_depth_rtol"],
            mask=confidence_mask,
        )
        official_mask = confidence_mask & non_edge_mask
        np.save(network / "recovered_intrinsics.npy", _numpy(recovered_intrinsics))
        np.save(network / "confidence_probability.npy", _numpy(confidence_probability))
        np.save(network / "official_mask.npy", _numpy(official_mask))

        network_depth = _numpy(outputs["local_points"][0, ..., 2])
        network_depth[~_numpy(official_mask[0])] = 0.0
        network_confidence = _numpy(confidence_probability[0])
        network_intrinsics = _numpy(recovered_intrinsics[0])
        camera_to_world = _numpy(outputs["camera_poses"][0]).astype(np.float64)
        extrinsics = np.linalg.inv(camera_to_world)[:, :3, :].astype(np.float32)
        official_processed_images = images_cpu.permute(0, 2, 3, 1).numpy()

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
            "primary_points": "official-example masked local_points Z unprojected with recovered intrinsics and predicted cameras",
            "point_branch": "official points retained at raw/network/points.npy; observational only",
            "camera_pose_source": "official OpenCV camera-to-world output, inverted to benchmark world-to-camera",
            "intrinsics_source": "recover_intrinsic_from_rays_d(normalize(local_points), force_center_principal_point=True)",
            "official_mask": "sigmoid(conf) threshold combined with official depth_normal_edge exclusion",
            "evaluation_raster": "official prediction deterministically warped to a full-image square canvas",
            "original_image_coverage": "full source image; official loader performs direct resize without crop",
            "image_hw": list(depth.shape[-2:]),
            "preprocessing_tensor_shape": list(images_cpu.shape),
            "confidence": "sigmoid of raw confidence logits, matching the official example",
            "matching_or_tracks": "track head not queried by this baseline",
            "reference_view_argument": ref_view_strategy,
            "reference_view_effect": "none; Pi3X jointly predicts the ordered image set",
            "gt_conditioning": False,
        })
        del outputs, images
        torch.cuda.empty_cache()
        return prediction


def _source_hashes():
    files = (
        list(Path(__file__).parent.glob("*.py"))
        + list((Path(__file__).parents[1] / "common").glob("*.py"))
        + list((Path(__file__).parents[1] / "da3").glob("*.py"))
    )
    files = [path for path in files if not path.name.startswith("._")]
    return {str(path.relative_to(Path(__file__).parents[2])): sha256(path) for path in files}


def _validate_pi3x_artifacts(run_root, dataset, scene, modes):
    metrics = validate(run_root, dataset, scene, modes)
    expected = [
        run_root / "raw/network/local_points.npy",
        run_root / "raw/network/points.npy",
        run_root / "raw/network/camera_poses.npy",
        run_root / "raw/network/recovered_intrinsics.npy",
        run_root / "raw/network/confidence_probability.npy",
        run_root / "raw/network/official_mask.npy",
        run_root / "raw/preprocessing.json",
    ]
    for path in expected:
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Missing Pi3X raw artifact: {path}")
    world_points = np.load(run_root / "raw/network/points.npy", mmap_mode="r")
    local_points = np.load(run_root / "raw/network/local_points.npy", mmap_mode="r")
    if world_points.shape != local_points.shape:
        raise ValueError(
            f"World point shape {world_points.shape} does not match local point shape {local_points.shape}"
        )
    recovered_intrinsics = np.load(run_root / "raw/network/recovered_intrinsics.npy")
    official_mask = np.load(run_root / "raw/network/official_mask.npy", mmap_mode="r")
    if not np.isfinite(recovered_intrinsics).all():
        raise ValueError("Pi3X recovered non-finite intrinsics")
    if np.any(recovered_intrinsics[..., 0, 0] <= 0) or np.any(recovered_intrinsics[..., 1, 1] <= 0):
        raise ValueError("Pi3X recovered non-positive focal length")
    if official_mask.shape != local_points.shape[:-1] or not np.any(official_mask):
        raise ValueError("Pi3X official-example mask is invalid")
    validation_path = run_root / "validation.json"
    record = json.loads(validation_path.read_text())
    record["pi3x_raw_outputs_present"] = True
    record["primary_geometry_branch"] = "official_example_masked_local_depth_plus_predicted_camera"
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
        pi3x_code = args.root / "code/Pi3"
        benchmark_code = args.root / "code/Depth-Anything-3"
        actual_revisions = {
            "pi3x": subprocess.check_output(["git", "-C", str(pi3x_code), "rev-parse", "HEAD"], text=True).strip(),
            "benchmark": subprocess.check_output(["git", "-C", str(benchmark_code), "rev-parse", "HEAD"], text=True).strip(),
        }
        expected_revisions = {"pi3x": cfg["code_revision"], "benchmark": cfg["benchmark_code_revision"]}
        if actual_revisions != expected_revisions:
            raise ValueError(f"Code revisions differ from pinned configuration: {actual_revisions}")
        subprocess.run(["git", "-C", str(pi3x_code), "diff", "--exit-code", "HEAD"], check=True)
        subprocess.run(["git", "-C", str(benchmark_code), "diff", "--exit-code", "HEAD"], check=True)
        os.chdir(benchmark_code)

        from depth_anything_3.bench.evaluator import Evaluator
        from depth_anything_3.bench.registries import MV_REGISTRY
        from pi3.models.pi3x import Pi3X
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
            (args.root / "provenance/pi3x_assets.json", run_root / "pi3x_assets.json"),
            (args.root / "provenance/pi3x_environment.txt", run_root / "pi3x_environment.txt"),
            (args.root / "provenance/assets.json", run_root / "benchmark_assets.json"),
        ]:
            shutil.copy2(source, destination)
        dump(run_root / "runner_sources.json", _source_hashes())
        shutil.copytree(Path(__file__).parent, run_root / "runner_source/pi3x", ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copytree(Path(__file__).parents[1] / "common", run_root / "runner_source/common", ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copytree(Path(__file__).parents[1] / "da3", run_root / "runner_source/da3", ignore=shutil.ignore_patterns("__pycache__"))
        dump(run_root / "instrumentation.json", {
            "model_changes": None,
            "model_inputs": "images only; multimodal branch disabled; no intrinsics, rays, depth, pose, scale, or GT metadata",
            "primary_output": "official-example masked local-point Z plus recovered intrinsics and predicted cameras",
            "secondary_output": "official global/local point branches archived but not selected using benchmark scores",
            "official_inference_options": {
                "inference_dtype": cfg["inference_dtype"],
                "pixel_limit": cfg["pixel_limit"],
                "force_center_principal_point": True,
                "confidence_probability_threshold": cfg["confidence_probability_threshold"],
                "edge_depth_rtol": cfg["edge_depth_rtol"],
                "multimodal_branch_disabled": True,
            },
            "synchronous_export": "mini_npz written before evaluator proceeds",
            "failure_change": "invalid mesh raises; no replacement random points",
            "evaluation_random_seed": cfg["sampling_seed"],
            "gt_metadata": "evaluator-owned files under evaluation; never passed to model",
            "additional_diagnostics": "observational only; not used to alter predictions",
        })

        stage = "model_load"
        torch.cuda.reset_peak_memory_stats()
        model = Pi3X.from_pretrained(str(args.root / "models/Pi3X")).eval()
        model.disable_multimodal()
        model = model.to("cuda")
        adapter = Pi3XAdapter(model, run_root, cfg)
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
        metrics = _validate_pi3x_artifacts(run_root, args.dataset, scene, modes)
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
