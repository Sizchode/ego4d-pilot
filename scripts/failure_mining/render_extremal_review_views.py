"""Render threshold-free extremal input views for audited failure-review cases."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps, __version__ as pillow_version

from scripts.failure_mining.render_selected_input_views import (
    CAMERA_METRICS,
    _input_records,
    _load_pair_arrays,
    _manifest_fingerprint,
    read_json,
    read_jsonl,
    sha256,
    slug,
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def per_view_maximum(
    indices: set[int], index_a: np.ndarray, index_b: np.ndarray, values: np.ndarray
) -> dict[int, float]:
    output = {}
    for index in sorted(indices):
        incident = values[(index_a == index) | (index_b == index)]
        finite = incident[np.isfinite(incident)]
        require(finite.size > 0, f"No finite incident pair error for view {index}")
        output[index] = float(np.max(finite))
    return output


def render_extrema(
    bundles_dir: Path,
    output_dir: Path,
    *,
    max_side: int,
    jpeg_quality: int,
) -> dict[str, Any]:
    require(not output_dir.exists(), f"Refusing to overwrite output: {output_dir}")
    require(max_side > 0, "max_side must be positive")
    require(1 <= jpeg_quality <= 95, "jpeg_quality must be in [1, 95]")
    bundle_manifest_path = bundles_dir / "manifest.json"
    selected_runs_path = bundles_dir / "selected_runs.jsonl"
    bundle_manifest = read_json(bundle_manifest_path)
    require(bundle_manifest.get("test_time_policy_input") is False, "Review bundle is policy input")
    require(
        bundle_manifest.get("selected_runs_sha256") == sha256(selected_runs_path),
        "Selected-run hash mismatch",
    )
    models = bundle_manifest.get("models")
    require(isinstance(models, list) and models and len(models) == len(set(models)), "Invalid model population")

    rows = read_jsonl(selected_runs_path)
    grouped: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        key = (row["dataset"], row["scene"])
        require(row["model"] not in grouped[key], f"Duplicate selected run: {(row['model'], *key)}")
        grouped[key][row["model"]] = row
    require(grouped, "No selected scenes")

    plans = []
    for (dataset, scene), by_model in sorted(grouped.items()):
        require(list(by_model) == models, f"Cross-model order/population differs: {(dataset, scene)}")
        reference_inputs = None
        reference_fingerprint = None
        reasons_by_index: dict[int, list[dict[str, Any]]] = defaultdict(list)
        cases = []
        for model in models:
            run = Path(by_model[model]["run"])
            status = read_json(run / "status.json")
            validation = read_json(run / "validation.json")
            config = read_json(run / "run_config.json")
            require(status.get("state") == "completed" and validation.get("passed") is True, f"Invalid selected run: {(model, dataset, scene)}")
            require((config.get("dataset"), config.get("scene")) == (dataset, scene), f"Run identity mismatch: {(model, dataset, scene)}")
            inputs, input_hash = _input_records(run / "input_manifest.json")
            fingerprint = _manifest_fingerprint(inputs)
            if reference_inputs is None:
                reference_inputs = inputs
                reference_fingerprint = fingerprint
            else:
                require(fingerprint == reference_fingerprint, f"Input population differs across models: {(dataset, scene)}")
            pairs_path = run / "evaluation_capture" / "camera_pairs.npz"
            index_a, index_b, pair_metrics = _load_pair_arrays(pairs_path)
            indices = set(inputs)
            require(set(index_a.tolist()) | set(index_b.tolist()) == indices, f"Pair/input population differs: {(model, dataset, scene)}")
            for metric, values in pair_metrics.items():
                per_view = per_view_maximum(indices, index_a, index_b, values)
                maximum = max(per_view.values())
                minimum = min(per_view.values())
                for index, value in per_view.items():
                    for extremum, target in (("maximum", maximum), ("minimum_control", minimum)):
                        if value == target:
                            reasons_by_index[index].append(
                                {
                                    "model": model,
                                    "metric": metric,
                                    "per_view_statistic": "maximum_incident_pair_error",
                                    "extremum": extremum,
                                    "value": value,
                                }
                            )
            cases.append(
                {
                    "model": model,
                    "run": str(run),
                    "input_manifest_sha256": input_hash,
                    "camera_pairs_sha256": sha256(pairs_path),
                }
            )
        assert reference_inputs is not None
        plans.append(
            {
                "dataset": dataset,
                "scene": scene,
                "inputs": reference_inputs,
                "reasons": reasons_by_index,
                "cases": cases,
            }
        )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent))
    try:
        scene_outputs = []
        for plan in plans:
            dataset, scene = plan["dataset"], plan["scene"]
            relative_root = Path(slug(dataset)) / slug(scene)
            (temporary / relative_root).mkdir(parents=True)
            views = []
            for index, reasons in sorted(plan["reasons"].items()):
                source_record = plan["inputs"][index]
                source = Path(source_record["path"])
                require(source.is_file(), f"Selected input is missing: {source}")
                source_hash = sha256(source)
                require(source_hash == source_record["sha256"], f"Selected input hash mismatch: {source}")
                with Image.open(source) as opened:
                    image = ImageOps.exif_transpose(opened).convert("RGB")
                    decoded_wh = list(image.size)
                    image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
                    rendered_wh = list(image.size)
                    relative = relative_root / f"view-{index:03d}.jpg"
                    image.save(temporary / relative, "JPEG", quality=jpeg_quality, optimize=True)
                views.append(
                    {
                        "index": index,
                        "source_path": str(source),
                        "source_sha256": source_hash,
                        "source_original_hw": source_record.get("original_hw"),
                        "decoded_wh": decoded_wh,
                        "rendered_wh": rendered_wh,
                        "output_path": str(relative),
                        "output_sha256": sha256(temporary / relative),
                        "selection_reasons": reasons,
                    }
                )
            scene_outputs.append(
                {
                    "dataset": dataset,
                    "scene": scene,
                    "input_count": len(plan["inputs"]),
                    "selected_view_count": len(views),
                    "cases": plan["cases"],
                    "views": views,
                }
            )
        manifest = {
            "schema_version": 1,
            "bundle_manifest_sha256": sha256(bundle_manifest_path),
            "selected_runs_sha256": sha256(selected_runs_path),
            "models": models,
            "camera_metrics": list(CAMERA_METRICS),
            "scene_outputs": scene_outputs,
            "selected_scenes": len(scene_outputs),
            "selected_views": sum(row["selected_view_count"] for row in scene_outputs),
            "per_view_statistic": "maximum finite incident official-evaluator pair error",
            "extrema": ["maximum", "minimum_control"],
            "selection_threshold": None,
            "top_k": None,
            "ties": "all exact ties retained",
            "uses_ground_truth_selection": True,
            "test_time_policy_input": False,
            "render": {
                "pillow_version": pillow_version,
                "max_side": max_side,
                "jpeg_quality": jpeg_quality,
                "exif_transpose": True,
                "resampling": "LANCZOS",
            },
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, output_dir)
        return manifest
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundles", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-side", type=int, default=1024)
    parser.add_argument("--jpeg-quality", type=int, default=90)
    args = parser.parse_args()
    manifest = render_extrema(
        args.bundles,
        args.output,
        max_side=args.max_side,
        jpeg_quality=args.jpeg_quality,
    )
    print(json.dumps({key: manifest[key] for key in ("selected_scenes", "selected_views", "selection_threshold", "top_k")}, indent=2))


if __name__ == "__main__":
    main()
