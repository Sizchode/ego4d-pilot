"""Render exact per-scene min/max DA3 GT-depth-error views for review."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from scripts.failure_mining.analyze_da3_gt_depth_error import depth_error_summary
from scripts.failure_mining.analyze_gt_geometry_factors import load_scene_gt_inputs, read_json
from scripts.failure_mining.compare_baselines import sha256, write_csv


METRIC = "depth.absolute_relative.mean"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_csv(path: Path) -> list[dict[str, str]]:
    require(path.is_file(), f"Required file is missing: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def select_extremes(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["scene"])].append(row)
    require(grouped, "View table is empty")
    output = []
    for (dataset, scene), scene_rows in sorted(grouped.items()):
        finite = [row for row in scene_rows if row.get(METRIC) not in {None, ""} and math.isfinite(float(row[METRIC]))]
        require(finite, f"Scene has no finite {METRIC}: {scene}")
        minimum = min(finite, key=lambda row: (float(row[METRIC]), int(row["view_index"])))
        maximum = max(finite, key=lambda row: (float(row[METRIC]), -int(row["view_index"])))
        for role, row in (("minimum", minimum), ("maximum", maximum)):
            output.append(
                {
                    "dataset": dataset,
                    "scene": scene,
                    "role": role,
                    "view_index": int(row["view_index"]),
                    METRIC: float(row[METRIC]),
                    "depth.absolute_relative.p95": float(row["depth.absolute_relative.p95"]),
                    "joint_over_ground_truth_valid_ratio": float(row["joint_over_ground_truth_valid_ratio"]),
                }
            )
    return output


def prepare_rgb(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    require(image.ndim == 3 and image.shape[-1] == 3, f"Unexpected RGB shape: {image.shape}")
    output = image.astype(np.float32)
    if output.max() > 1:
        output /= 255.0
    return np.clip(output, 0, 1)


def render_scene(
    output_path: Path,
    scene: str,
    selections: list[dict[str, Any]],
    rgb: np.ndarray,
    predicted: np.ndarray,
    ground_truth: np.ndarray,
    *,
    max_depth: float,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    require(len(selections) == 2, f"Expected two selections: {scene}")
    require(len({row["role"] for row in selections}) == 2, f"Selection roles are not unique: {scene}")
    error_maps = []
    for row in selections:
        index = int(row["view_index"])
        pred, gt = predicted[index], ground_truth[index]
        valid = np.isfinite(pred) & (pred > 0) & (pred <= max_depth) & np.isfinite(gt) & (gt > 0) & (gt <= max_depth)
        error = np.full(gt.shape, np.nan, dtype=np.float64)
        error[valid] = np.log1p(np.abs(pred[valid] - gt[valid]) / gt[valid])
        error_maps.append(error)
    finite_errors = np.concatenate([values[np.isfinite(values)] for values in error_maps])
    require(finite_errors.size > 0, f"No finite selected error pixels: {scene}")
    error_max = float(finite_errors.max())

    fig, axes = plt.subplots(2, 4, figsize=(14, 7), squeeze=False)
    depth_image = error_image = None
    for row_index, (row, error) in enumerate(zip(selections, error_maps, strict=True)):
        index = int(row["view_index"])
        axes[row_index, 0].imshow(prepare_rgb(rgb[index]))
        axes[row_index, 0].set_title(f"{row['role']} view {index}\nmean AbsRel={row[METRIC]:.4f}")
        depth_image = axes[row_index, 1].imshow(predicted[index], cmap="turbo", vmin=0, vmax=max_depth)
        axes[row_index, 1].set_title("DA3 aligned depth")
        axes[row_index, 2].imshow(ground_truth[index], cmap="turbo", vmin=0, vmax=max_depth)
        axes[row_index, 2].set_title("GT depth")
        error_image = axes[row_index, 3].imshow(error, cmap="magma", vmin=0, vmax=error_max)
        axes[row_index, 3].set_title("log(1 + absolute-relative error)")
        for axis in axes[row_index]:
            axis.axis("off")
    require(depth_image is not None and error_image is not None, f"Rendering failed: {scene}")
    fig.colorbar(depth_image, ax=axes[:, 1:3], fraction=0.025, pad=0.02, label="depth")
    fig.colorbar(error_image, ax=axes[:, 3], fraction=0.05, pad=0.02, label="log1p AbsRel")
    fig.suptitle(f"ScanNet++ {scene}: fixed-rule GT-depth review views")
    fig.subplots_adjust(left=0.02, right=0.94, bottom=0.02, top=0.90, wspace=0.05, hspace=0.12)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-depth-analysis", type=Path, required=True)
    parser.add_argument("--selection-csv", type=Path)
    parser.add_argument("--selection-manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(
        (args.selection_csv is None) == (args.selection_manifest is None),
        "Selection CSV and manifest must be provided together",
    )
    require(not args.output.exists(), f"Refusing to overwrite output: {args.output}")
    manifest_path = args.gt_depth_analysis / "manifest.json"
    view_path = args.gt_depth_analysis / "view_gt_depth_error.csv"
    manifest = read_json(manifest_path)
    require(manifest.get("view_output_sha256") == sha256(view_path), "GT-depth view table hash differs")
    require(manifest.get("uses_ground_truth_analysis") is True, "Missing GT provenance")
    selection_provenance: dict[str, Any]
    if args.selection_csv is None:
        selections = select_extremes(read_csv(view_path))
        selection_provenance = {
            "selection_rule": "exact finite within-scene minimum and maximum; ties use lowest view index",
            "selection_source_sha256": sha256(view_path),
        }
    else:
        selection_manifest = read_json(args.selection_manifest)
        require(
            selection_manifest.get("selection_output_sha256") == sha256(args.selection_csv),
            "External selection table hash differs",
        )
        require(selection_manifest.get("uses_ground_truth_analysis") is True, "Missing selection GT provenance")
        require(selection_manifest.get("test_time_policy_input") is False, "Selection marks GT as policy input")
        selections = read_csv(args.selection_csv)
        require(selections, "External selection table is empty")
        for row in selections:
            require(row["dataset"] == "scannetpp", f"Unexpected selection dataset: {row}")
            row["view_index"] = int(row["view_index"])
            row[METRIC] = float(row[METRIC])
        selection_provenance = {
            "selection_rule": selection_manifest["selection_rule"],
            "selection_manifest_sha256": sha256(args.selection_manifest),
            "selection_source_sha256": sha256(args.selection_csv),
        }
    by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in selections:
        by_scene[row["scene"]].append(row)
    require(list(by_scene) == sorted(manifest["scenes"]), "Selection scene population/order differs")
    require(all(len(rows) == 2 for rows in by_scene.values()), "Each scene must have two selected views")

    from depth_anything_3.bench.datasets.scannetpp import ScanNetPP

    dataset = ScanNetPP()
    dataset.data_root = manifest["dataset_root"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{args.output.name}-", dir=args.output.parent))
    try:
        panels = temporary / "panels"
        panels.mkdir()
        output_rows = []
        panel_hashes = {}
        for scene in sorted(by_scene):
            run = Path(manifest["inputs"][scene]["run"])
            config = read_json(run / "run_config.json")
            params = config["dataset_parameters"]
            with np.load(run / "evaluation_capture/fusion_inputs_000.npz", allow_pickle=False) as fusion:
                predicted = np.asarray(fusion["depth"], dtype=np.float32)
            rgb = np.load(run / "raw/processed_images.npy", mmap_mode="r")
            gt = load_scene_gt_inputs(
                run,
                dataset,
                scene,
                input_h=int(params["input_h"]),
                input_w=int(params["input_w"]),
            )
            ground_truth = gt["depth"]
            require(predicted.shape == ground_truth.shape and len(rgb) == len(predicted), f"Population differs: {scene}")
            for row in by_scene[scene]:
                index = int(row["view_index"])
                recomputed, _arrays = depth_error_summary(predicted[index], ground_truth[index], max_depth=float(params["max_depth"]))
                require(
                    math.isclose(float(row[METRIC]), float(recomputed[METRIC]), rel_tol=1e-12, abs_tol=1e-12),
                    f"Selected metric differs: {(scene, index)}",
                )
                row["image_key"] = gt["image_keys"][index]
            panel_path = panels / f"{scene}.png"
            render_scene(
                panel_path,
                scene,
                by_scene[scene],
                rgb,
                predicted,
                ground_truth,
                max_depth=float(params["max_depth"]),
            )
            panel_hashes[scene] = sha256(panel_path)
            output_rows.extend(by_scene[scene])

        selection_path = temporary / "selections.csv"
        write_csv(selection_path, output_rows)
        output_manifest = {
            "dataset": "scannetpp",
            "model": manifest["model"],
            "scenes": sorted(by_scene),
            "selection_metric": METRIC,
            **selection_provenance,
            "selected_views": len(output_rows),
            "gt_depth_manifest_sha256": sha256(manifest_path),
            "gt_depth_view_table_sha256": sha256(view_path),
            "selection_output_sha256": sha256(selection_path),
            "panel_sha256": panel_hashes,
            "analysis_script_sha256": sha256(Path(__file__)),
            "uses_ground_truth_analysis": True,
            "test_time_policy_input": False,
            "failure_labels": None,
            "quality_thresholds": None,
            "top_k": None,
        }
        (temporary / "manifest.json").write_text(
            json.dumps(output_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, args.output)
        print(json.dumps({key: output_manifest[key] for key in ("scenes", "selected_views", "selection_output_sha256")}, indent=2))
    except BaseException:
        import shutil

        shutil.rmtree(temporary, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()
