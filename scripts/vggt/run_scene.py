"""Run one frozen VGGT scene through the audited DA3 benchmark protocol.

No repair, model selection, ground-truth conditioning, or adaptive frame
reduction is performed.  VGGT's camera and depth heads define the primary
reconstruction, following the upstream recommendation.  Its point-map branch is
retained as observational raw output and is never selected using benchmark
scores.
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
from scripts.vggt.preprocess import load_and_preprocess_with_metadata, normalize_to_evaluation_canvas


DATASETS = ("hiroom", "eth3d", "scannetpp", "7scenes", "dtu", "dtu64")
PROTOCOL_MODES = {"dtu": ["recon_unposed"], "dtu64": ["pose"]}


def _numpy(tensor):
    return tensor.detach().float().cpu().numpy()


class VGGTAdapter:
    """Evaluator-compatible VGGT API with lossless raw-output archiving."""

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

        manifest = []
        for index, filename in enumerate(image_files):
            from PIL import Image
            with Image.open(filename) as image:
                width, height = image.size
            manifest.append({
                "index": index,
                "path": str(Path(filename).resolve()),
                "sha256": sha256(filename),
                "original_hw": [height, width],
            })
        dump(self.run_root / "input_manifest.json", manifest)

        images_cpu, transforms, preprocessing = load_and_preprocess_with_metadata(
            image_files, self.cfg["preprocess_mode"]
        )
        np.save(raw / "original_to_processed_pixel.npy", transforms)
        dump(raw / "preprocessing.json", {
            "upstream_function": "vggt.utils.load_fn.load_and_preprocess_images",
            "mode": self.cfg["preprocess_mode"],
            "operations": preprocessing,
        })
        images = images_cpu.to("cuda", non_blocking=False)

        dtype = torch.bfloat16 if self.cfg["inference_dtype"] == "bfloat16" else torch.float16
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        started = time.monotonic()
        start_event.record()
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=dtype):
            outputs = self.model(images)
        end_event.record()
        torch.cuda.synchronize()
        dump(self.run_root / "inference_timing.json", {
            "seconds_including_device_transfer": time.monotonic() - started,
            "network_cuda_seconds": start_event.elapsed_time(end_event) / 1000,
        })

        from vggt.utils.pose_enc import pose_encoding_to_extri_intri
        extrinsics_t, intrinsics_t = pose_encoding_to_extri_intri(
            outputs["pose_enc"], images.shape[-2:]
        )
        network_depth = _numpy(outputs["depth"][0, ..., 0])
        network_confidence = _numpy(outputs["depth_conf"][0])
        extrinsics = _numpy(extrinsics_t[0])
        network_intrinsics = _numpy(intrinsics_t[0])
        official_processed_images = images_cpu.permute(0, 2, 3, 1).numpy()

        normalized = normalize_to_evaluation_canvas(
            network_depth,
            network_confidence,
            official_processed_images,
            network_intrinsics,
            [(entry["original_hw"][1], entry["original_hw"][0]) for entry in manifest],
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

        np.save(network / "pose_enc.npy", _numpy(outputs["pose_enc"]))
        np.save(network / "pose_enc_iterations.npy", np.stack([_numpy(value) for value in outputs["pose_enc_list"]]))
        np.save(network / "depth.npy", _numpy(outputs["depth"]))
        np.save(network / "depth_conf.npy", _numpy(outputs["depth_conf"]))
        np.save(network / "world_points.npy", _numpy(outputs["world_points"]))
        np.save(network / "world_points_conf.npy", _numpy(outputs["world_points_conf"]))

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
            "depth": "camera Z; model-native scale, not assumed metric",
            "primary_points": "depth unprojected with predicted cameras, per official VGGT recommendation",
            "point_branch": "retained at raw/network/world_points.npy; observational only",
            "evaluation_raster": "official prediction deterministically warped to a full-image square canvas",
            "invalid_crop_regions": "zero depth; never filled or hallucinated",
            "image_hw": list(depth.shape[-2:]),
            "preprocessing_tensor_shape": list(images_cpu.shape),
            "confidence": "raw depth-head score; not a calibrated probability and not comparable across models",
            "matching_or_tracks": "track head not queried by this baseline",
            "reference_view_argument": ref_view_strategy,
            "reference_view_effect": "none; VGGT jointly predicts the ordered image set",
            "gt_conditioning": False,
        })
        del outputs, images
        torch.cuda.empty_cache()
        return prediction


def _source_hashes():
    files = list(Path(__file__).parent.glob("*.py")) + list((Path(__file__).parents[1] / "da3").glob("*.py"))
    return {str(path.relative_to(Path(__file__).parents[2])): sha256(path) for path in files}


def _validate_vggt_artifacts(run_root, dataset, scene, modes):
    metrics = validate(run_root, dataset, scene, modes)
    expected = [
        run_root / "raw/network/pose_enc.npy",
        run_root / "raw/network/pose_enc_iterations.npy",
        run_root / "raw/network/world_points.npy",
        run_root / "raw/network/world_points_conf.npy",
        run_root / "raw/preprocessing.json",
    ]
    for path in expected:
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Missing VGGT raw artifact: {path}")
    world_points = np.load(run_root / "raw/network/world_points.npy", mmap_mode="r")
    network_depth = np.load(run_root / "raw/network/depth.npy", mmap_mode="r")
    if world_points.shape[:4] != network_depth.shape[:4]:
        raise ValueError(
            f"Point-branch shape {world_points.shape} does not match network depth {network_depth.shape}"
        )
    validation_path = run_root / "validation.json"
    record = json.loads(validation_path.read_text())
    record["vggt_raw_branches_present"] = True
    record["primary_geometry_branch"] = "depth_plus_predicted_camera"
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
        vggt_code = args.root / "code/VGGT"
        benchmark_code = args.root / "code/Depth-Anything-3"
        actual_revisions = {
            "vggt": subprocess.check_output(["git", "-C", str(vggt_code), "rev-parse", "HEAD"], text=True).strip(),
            "benchmark": subprocess.check_output(["git", "-C", str(benchmark_code), "rev-parse", "HEAD"], text=True).strip(),
        }
        expected_revisions = {"vggt": cfg["code_revision"], "benchmark": cfg["benchmark_code_revision"]}
        if actual_revisions != expected_revisions:
            raise ValueError(f"Code revisions differ from pinned configuration: {actual_revisions}")
        subprocess.run(["git", "-C", str(vggt_code), "diff", "--exit-code", "HEAD"], check=True)
        subprocess.run(["git", "-C", str(benchmark_code), "diff", "--exit-code", "HEAD"], check=True)
        os.chdir(benchmark_code)

        from depth_anything_3.bench.evaluator import Evaluator
        from depth_anything_3.bench.registries import MV_REGISTRY
        from vggt.models.vggt import VGGT
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
            "model": "VGGT-1B",
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
            (args.root / "provenance/vggt_assets.json", run_root / "vggt_assets.json"),
            (args.root / "provenance/vggt_environment.txt", run_root / "vggt_environment.txt"),
            (args.root / "provenance/assets.json", run_root / "benchmark_assets.json"),
        ]:
            shutil.copy2(source, destination)
        dump(run_root / "runner_sources.json", _source_hashes())
        shutil.copytree(Path(__file__).parent, run_root / "runner_source/vggt", ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copytree(Path(__file__).parents[1] / "da3", run_root / "runner_source/da3", ignore=shutil.ignore_patterns("__pycache__"))
        dump(run_root / "instrumentation.json", {
            "model_changes": None,
            "primary_output": "official depth branch plus predicted camera unprojection",
            "secondary_output": "official point-map branch archived but not evaluated or selected",
            "synchronous_export": "mini_npz written before evaluator proceeds",
            "failure_change": "invalid mesh raises; no replacement random points",
            "evaluation_random_seed": cfg["sampling_seed"],
            "gt_metadata": "evaluator-owned files under evaluation; never passed to model",
            "additional_diagnostics": "observational only; not used to alter predictions",
        })

        stage = "model_load"
        torch.cuda.reset_peak_memory_stats()
        model = VGGT.from_pretrained(str(args.root / "models/VGGT-1B")).to("cuda").eval()
        adapter = VGGTAdapter(model, run_root, cfg)
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
        metrics = _validate_vggt_artifacts(run_root, args.dataset, scene, modes)
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
