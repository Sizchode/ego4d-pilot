"""Summarize one baseline batch without assigning heuristic failure labels.

The batch manifest defines the population. Every scene receives either a fully
validated evidence row or an explicit execution-failure row. Rankings are kept
separate by interpretable metric; no weighted aggregate score is constructed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np


QUANTILES = (0.5, 0.9, 0.95, 0.99)


def load_json(path: Path):
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite_summary(values: np.ndarray, prefix: str) -> dict[str, float | int | None]:
    array = np.asarray(values).reshape(-1)
    finite = array[np.isfinite(array)]
    result: dict[str, float | int | None] = {
        f"{prefix}.count": int(array.size),
        f"{prefix}.finite_count": int(finite.size),
    }
    if finite.size == 0:
        for name in ("mean", "min", "max", "p50", "p90", "p95", "p99"):
            result[f"{prefix}.{name}"] = None
        return result
    result.update(
        {
            f"{prefix}.mean": float(finite.mean()),
            f"{prefix}.min": float(finite.min()),
            f"{prefix}.max": float(finite.max()),
        }
    )
    for quantile in QUANTILES:
        result[f"{prefix}.p{int(quantile * 100):02d}"] = float(np.quantile(finite, quantile))
    return result


def flatten_metrics(metrics: dict) -> dict[str, float]:
    flat = {}
    for mode, values in metrics.items():
        for name, value in values.items():
            flat[f"metric.{mode}.{name}"] = float(value)
    return flat


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def load_results(batch_dir: Path, manifest: list[dict]) -> list[dict]:
    result_path = batch_dir / "results.jsonl"
    records = (
        [json.loads(line) for line in result_path.read_text().splitlines() if line.strip()]
        if result_path.is_file()
        else []
    )
    expected = [(item["dataset"], item["scene"]) for item in manifest]
    observed = [(item["dataset"], item["scene"]) for item in records]
    if observed != expected[: len(observed)]:
        raise ValueError("results.jsonl is not an ordered prefix of the batch manifest")
    for item in manifest[len(records) :]:
        records.append(
            {
                "dataset": item["dataset"],
                "scene": item["scene"],
                "run": "",
                "state": "missing_after_batch_termination",
                "exit_code": None,
            }
        )
    return records


def summarize_camera(run: Path) -> dict:
    path = run / "evaluation_capture/camera_pairs.npz"
    with np.load(path) as archive:
        a, b = archive["index_a"], archive["index_b"]
        if len(a) != len(b) or np.any(a >= b):
            raise ValueError(f"Invalid camera-pair indices: {path}")
        result = {"camera.pair_count": int(len(a))}
        result.update(finite_summary(archive["rotation_degrees"], "camera.rotation_degrees"))
        result.update(finite_summary(archive["translation_degrees"], "camera.translation_degrees"))
        return result


def summarize_pairs(run: Path, identity: dict) -> tuple[list[dict], dict]:
    protocol = load_json(run / "diagnostics/protocol.json")
    pair_rows = []
    relative_chunks = []
    roundtrip_chunks = []
    signed_sum = 0.0
    signed_count = 0
    projectable = 0
    pixels = 0
    for source, target in protocol["pairs"]:
        path = run / "diagnostics" / f"{source:06d}_{target:06d}.npz"
        with np.load(path) as archive:
            mask = archive["projectable"].astype(bool)
            pair_pixels = int(mask.size)
            pair_projectable = int(mask.sum())
            signed = archive["signed_z_residual"][mask]
            relative = np.abs(archive["relative_z_residual"][mask])
            roundtrip = archive["roundtrip_pixels"][mask]
        if not (np.isfinite(signed).all() and np.isfinite(relative).all() and np.isfinite(roundtrip).all()):
            raise ValueError(f"Non-finite values inside projectable mask: {path}")
        row = {
            **identity,
            "source_index": source,
            "target_index": target,
            "pixels": pair_pixels,
            "projectable_count": pair_projectable,
            "projectable_ratio": pair_projectable / pair_pixels,
        }
        row.update(finite_summary(signed, "signed_z"))
        row.update(finite_summary(relative, "absolute_relative_z"))
        row.update(finite_summary(roundtrip, "roundtrip_pixels"))
        pair_rows.append(row)
        pixels += pair_pixels
        projectable += pair_projectable
        if signed.size:
            signed_sum += float(signed.sum(dtype=np.float64))
            signed_count += int(signed.size)
            relative_chunks.append(relative)
            roundtrip_chunks.append(roundtrip)
    scene = {
        "diagnostic.directed_pair_count": len(pair_rows),
        "diagnostic.pixel_count": pixels,
        "diagnostic.projectable_count": projectable,
        "diagnostic.projectable_ratio": projectable / pixels,
        "diagnostic.signed_z.mean": signed_sum / signed_count if signed_count else None,
    }
    relative_all = np.concatenate(relative_chunks) if relative_chunks else np.empty(0)
    roundtrip_all = np.concatenate(roundtrip_chunks) if roundtrip_chunks else np.empty(0)
    scene.update(finite_summary(relative_all, "diagnostic.absolute_relative_z"))
    scene.update(finite_summary(roundtrip_all, "diagnostic.roundtrip_pixels"))
    return pair_rows, scene


def summarize_fusion(run: Path) -> dict:
    paths = sorted((run / "evaluation_capture").glob("fusion_inputs_*.npz"))
    if len(paths) != 1:
        raise ValueError(f"Expected exactly one fusion-input capture in {run}, found {len(paths)}")
    with np.load(paths[0]) as archive:
        retained = archive["retained_mask"].astype(bool)
        depth = archive["depth"]
        result = {
            "fusion.pixel_count": int(retained.size),
            "fusion.retained_count": int(retained.sum()),
            "fusion.retained_ratio": float(retained.mean()),
        }
        result.update(finite_summary(depth[retained], "fusion.retained_depth"))
        return result


def surface_row(identity: dict, direction: str, values: np.ndarray) -> dict:
    row = {**identity, "direction": direction}
    row.update(finite_summary(values, "distance"))
    return row


def summarize_surfaces(run: Path, identity: dict) -> tuple[list[dict], dict]:
    directory = run / "evaluation_capture"
    generic = sorted(directory.glob("surface_distance_*.npz"))
    dtu = sorted(directory.glob("dtu_surface_distance_*.npz"))
    rows = []
    if generic:
        if len(generic) != 2:
            raise ValueError(f"Expected two generic surface-distance captures in {run}")
        directions = ("prediction_to_reference", "reference_to_prediction")
        for direction, path in zip(directions, generic, strict=True):
            with np.load(path) as archive:
                rows.append(surface_row(identity, direction, archive["distance"]))
    if dtu:
        if len(dtu) != 1:
            raise ValueError(f"Expected one DTU surface-distance capture in {run}")
        with np.load(dtu[0]) as archive:
            rows.append(surface_row(identity, "prediction_to_reference_included", archive["accuracy_distance"][archive["accuracy_included"]]))
            rows.append(surface_row(identity, "reference_to_prediction_included", archive["completeness_distance"][archive["completeness_included"]]))
    if not rows:
        raise ValueError(f"No surface-distance evidence in {run}")
    scene = {}
    for row in rows:
        direction = row["direction"]
        for key, value in row.items():
            if key.startswith("distance."):
                scene[f"surface.{direction}.{key.removeprefix('distance.')}"] = value
    return rows, scene


def require_completed_run(record: dict) -> tuple[Path, dict, dict, dict]:
    run = Path(record["run"])
    status = load_json(run / "status.json")
    validation = load_json(run / "validation.json")
    config = load_json(run / "run_config.json")
    if status["state"] != "completed" or not validation["passed"]:
        raise ValueError(f"Record says completed but validation failed: {run}")
    if (config["dataset"], config["scene"]) != (record["dataset"], record["scene"]):
        raise ValueError(f"Run identity mismatch: {run}")
    return run, status, validation, config


def build_rankings(scene_rows: list[dict]) -> list[dict]:
    # Direction follows metric definitions, not a learned or hand-weighted score.
    directions = {
        "metric.pose.auc30": "lower_is_worse",
        "metric.pose.auc15": "lower_is_worse",
        "metric.pose.auc05": "lower_is_worse",
        "metric.pose.auc03": "lower_is_worse",
        "metric.recon_unposed.acc": "higher_is_worse",
        "metric.recon_unposed.comp": "higher_is_worse",
        "metric.recon_unposed.overall": "higher_is_worse",
        "metric.recon_unposed.precision": "lower_is_worse",
        "metric.recon_unposed.recall": "lower_is_worse",
        "metric.recon_unposed.fscore": "lower_is_worse",
        "camera.rotation_degrees.p95": "higher_is_worse",
        "camera.translation_degrees.p95": "higher_is_worse",
        "diagnostic.absolute_relative_z.p95": "higher_is_worse",
        "diagnostic.roundtrip_pixels.p95": "higher_is_worse",
    }
    rankings = []
    datasets = sorted({row["dataset"] for row in scene_rows})
    for dataset in datasets:
        population = [row for row in scene_rows if row["dataset"] == dataset and row["state"] in {"completed", "reused_verified_run"}]
        for metric, direction in directions.items():
            available = [row for row in population if row.get(metric) is not None]
            sign = -1 if direction == "higher_is_worse" else 1
            available.sort(key=lambda row: (sign * row[metric], row["scene"]))
            for rank, row in enumerate(available, start=1):
                rankings.append(
                    {
                        "dataset": dataset,
                        "metric": metric,
                        "direction": direction,
                        "failure_rank": rank,
                        "population": len(available),
                        "scene": row["scene"],
                        "value": row[metric],
                        "run": row["run"],
                    }
                )
    return rankings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    manifest = load_json(args.batch_dir / "manifest.json")
    records = load_results(args.batch_dir, manifest)
    scene_rows = []
    pair_rows = []
    surface_rows = []
    failures = []

    for index, record in enumerate(records):
        identity = {
            "batch_index": index,
            "dataset": record["dataset"],
            "scene": record["scene"],
            "run": record["run"],
        }
        row = {**identity, "state": record["state"], "exit_code": record.get("exit_code")}
        if record["state"] not in {"completed", "reused_verified_run"}:
            run = Path(record["run"]) if record["run"] else None
            status_path = run / "status.json" if run is not None else None
            if status_path is not None and status_path.is_file():
                status = load_json(status_path)
                row.update(
                    {
                        "failure_stage": status.get("stage"),
                        "failure_type": status.get("type"),
                        "failure_message": status.get("message"),
                        "elapsed_seconds": status.get("elapsed_seconds"),
                    }
                )
            failures.append(row)
            scene_rows.append(row)
            continue
        run, status, validation, config = require_completed_run(record)
        runtime = load_json(run / "runtime.json")
        row.update(
            {
                "views": validation["views"],
                "image_height": validation["shape"][1],
                "image_width": validation["shape"][2],
                "elapsed_seconds": status["elapsed_seconds"],
                "inference_seconds": runtime["seconds_until_inference_complete"],
                "evaluation_seconds": runtime["evaluation_seconds"],
                "diagnostics_seconds": runtime["diagnostics_seconds"],
                "peak_cuda_allocated_bytes": runtime["peak_cuda_allocated_bytes"],
                "peak_cuda_reserved_bytes": runtime["peak_cuda_reserved_bytes"],
                "gpu": runtime["gpu"],
                "model": config.get("model", "DA3-GIANT"),
                "code_revision": config["config"]["code_revision"],
                "model_revision": config["config"]["model_revision"],
                "dataset_revision": config["config"]["dataset_revision"],
            }
        )
        row.update(flatten_metrics(validation["metrics"]))
        row.update(summarize_camera(run))
        scene_pairs, pair_scene = summarize_pairs(run, identity)
        pair_rows.extend(scene_pairs)
        row.update(pair_scene)
        if "recon_unposed" in validation["metrics"]:
            row.update(summarize_fusion(run))
            scene_surfaces, surface_scene = summarize_surfaces(run, identity)
            surface_rows.extend(scene_surfaces)
            row.update(surface_scene)
        scene_rows.append(row)

    write_csv(args.output / "scenes.csv", scene_rows)
    write_csv(args.output / "pair_diagnostics.csv", pair_rows)
    write_csv(args.output / "surface_diagnostics.csv", surface_rows)
    write_csv(args.output / "rankings.csv", build_rankings(scene_rows))
    (args.output / "failed_runs.json").write_text(json.dumps(failures, indent=2) + "\n")
    completed = sum(row["state"] in {"completed", "reused_verified_run"} for row in scene_rows)
    analysis_manifest = {
        "batch_dir": str(args.batch_dir.resolve()),
        "batch_manifest_sha256": sha256(args.batch_dir / "manifest.json"),
        "batch_results_sha256": (
            sha256(args.batch_dir / "results.jsonl")
            if (args.batch_dir / "results.jsonl").is_file()
            else None
        ),
        "analysis_script_sha256": sha256(Path(__file__)),
        "expected_scenes": len(manifest),
        "records": len(records),
        "completed_or_reused": completed,
        "execution_failures": len(failures),
        "scene_rows": len(scene_rows),
        "pair_rows": len(pair_rows),
        "surface_rows": len(surface_rows),
        "weighted_composite_failure_score": None,
        "failure_labels": None,
        "ranking_policy": "complete per-dataset rankings for each interpretable metric",
    }
    (args.output / "manifest.json").write_text(json.dumps(analysis_manifest, indent=2) + "\n")
    print(json.dumps(analysis_manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
