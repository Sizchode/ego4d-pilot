"""Summarize threshold-free z-buffer layers for one completed baseline batch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.failure_mining.analyze_da3_batch import (
    finite_summary,
    load_json,
    load_results,
    sha256,
    write_csv,
)
from scripts.failure_mining.visibility import pair_visibility_layers


LAYERS = (
    "target_depth_available",
    "target_depth_closer",
    "source_depth_not_behind",
)


def summarize_values(values: np.ndarray, prefix: str) -> dict:
    return finite_summary(np.abs(values), prefix)


def analyze_run(run: Path, identity: dict) -> tuple[list[dict], dict]:
    depth = np.load(run / "raw/depth.npy", mmap_mode="r")
    intrinsics = np.load(run / "raw/intrinsics.npy")
    extrinsics = np.load(run / "raw/extrinsics.npy")
    protocol = load_json(run / "diagnostics/protocol.json")
    if len(depth) != len(intrinsics) or len(depth) != len(extrinsics):
        raise ValueError(f"Prediction array lengths differ: {run}")

    pair_rows = []
    scene_values = {layer: [] for layer in LAYERS}
    scene_counts = {
        "pixels": 0,
        "rasterizable": 0,
        "source_zbuffer_winner": 0,
        **{layer: 0 for layer in LAYERS},
    }
    for source, target in protocol["pairs"]:
        layers = pair_visibility_layers(
            depth[source], depth[target], intrinsics[source], intrinsics[target],
            extrinsics[source], extrinsics[target],
        )
        row = {
            **identity,
            "source_index": source,
            "target_index": target,
            "pixels": int(depth[source].size),
            "rasterizable_count": int(layers["rasterizable"].sum()),
            "source_zbuffer_winner_count": int(layers["source_zbuffer_winner"].sum()),
        }
        row["rasterizable_ratio"] = row["rasterizable_count"] / row["pixels"]
        row["source_zbuffer_winner_ratio"] = (
            row["source_zbuffer_winner_count"] / row["pixels"]
        )
        scene_counts["pixels"] += row["pixels"]
        scene_counts["rasterizable"] += row["rasterizable_count"]
        scene_counts["source_zbuffer_winner"] += row["source_zbuffer_winner_count"]
        for layer in LAYERS:
            mask = layers[layer]
            count = int(mask.sum())
            row[f"{layer}_count"] = count
            row[f"{layer}_ratio"] = count / row["pixels"]
            row.update(
                summarize_values(
                    layers["relative_z_nearest"][mask],
                    f"{layer}.absolute_relative_z_nearest",
                )
            )
            scene_counts[layer] += count
            scene_values[layer].append(layers["relative_z_nearest"][mask])
        pair_rows.append(row)

    scene = {
        "visibility.directed_pair_count": len(pair_rows),
        "visibility.pixel_count": scene_counts["pixels"],
        "visibility.rasterizable_count": scene_counts["rasterizable"],
        "visibility.rasterizable_ratio": scene_counts["rasterizable"] / scene_counts["pixels"],
        "visibility.source_zbuffer_winner_count": scene_counts["source_zbuffer_winner"],
        "visibility.source_zbuffer_winner_ratio": (
            scene_counts["source_zbuffer_winner"] / scene_counts["pixels"]
        ),
    }
    for layer in LAYERS:
        scene[f"visibility.{layer}_count"] = scene_counts[layer]
        scene[f"visibility.{layer}_ratio"] = scene_counts[layer] / scene_counts["pixels"]
        values = np.concatenate(scene_values[layer]) if scene_values[layer] else np.empty(0)
        scene.update(
            summarize_values(values, f"visibility.{layer}.absolute_relative_z_nearest")
        )
    return pair_rows, scene


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    manifest = load_json(args.batch_dir / "manifest.json")
    records = load_results(args.batch_dir, manifest)
    pair_rows = []
    scene_rows = []
    failures = []
    for index, record in enumerate(records):
        identity = {
            "batch_index": index,
            "dataset": record["dataset"],
            "scene": record["scene"],
            "run": record["run"],
        }
        row = {**identity, "state": record["state"]}
        if record["state"] not in {"completed", "reused_verified_run"}:
            failures.append(row)
            scene_rows.append(row)
            continue
        run = Path(record["run"])
        status = load_json(run / "status.json")
        validation = load_json(run / "validation.json")
        if status["state"] != "completed" or not validation["passed"]:
            raise ValueError(f"Completed record has invalid run evidence: {run}")
        run_pairs, run_scene = analyze_run(run, identity)
        pair_rows.extend(run_pairs)
        row.update(run_scene)
        scene_rows.append(row)

    write_csv(args.output / "pair_visibility.csv", pair_rows)
    write_csv(args.output / "scene_visibility.csv", scene_rows)
    (args.output / "failed_runs.json").write_text(json.dumps(failures, indent=2) + "\n")
    analysis_manifest = {
        "batch_dir": str(args.batch_dir.resolve()),
        "batch_manifest_sha256": sha256(args.batch_dir / "manifest.json"),
        "batch_results_sha256": sha256(args.batch_dir / "results.jsonl"),
        "analysis_script_sha256": sha256(Path(__file__)),
        "visibility_module_sha256": sha256(Path(__file__).with_name("visibility.py")),
        "expected_scenes": len(manifest),
        "scene_rows": len(scene_rows),
        "pair_rows": len(pair_rows),
        "execution_failures": len(failures),
        "visibility_ground_truth": False,
        "numeric_tolerance": None,
        "failure_labels": None,
        "policy": (
            "nearest-pixel source z-buffer and strict target/source depth ordering; "
            "descriptive layers only"
        ),
    }
    (args.output / "manifest.json").write_text(json.dumps(analysis_manifest, indent=2) + "\n")
    print(json.dumps(analysis_manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
