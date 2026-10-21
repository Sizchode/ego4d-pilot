"""Render input-view evidence selected by camera-error ranks for offline mining.

The selection uses ground-truth camera errors captured by the official evaluator.
It is therefore an analysis artifact only and must never be exposed to a
test-time recovery policy.  The manifest records this restriction explicitly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps, __version__ as pillow_version


CAMERA_METRICS = ("rotation_degrees", "translation_degrees")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def slug(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")
    if not result:
        raise ValueError(f"Identity cannot be represented safely: {value!r}")
    return result


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _input_records(path: Path) -> tuple[dict[int, dict[str, Any]], str]:
    value = read_json(path)
    if not isinstance(value, list) or not value:
        raise ValueError(f"Expected a non-empty input manifest list: {path}")
    records: dict[int, dict[str, Any]] = {}
    for raw in value:
        if not isinstance(raw, dict):
            raise ValueError(f"Input manifest row is not an object: {path}")
        index = int(raw["index"])
        if index in records:
            raise ValueError(f"Duplicate input index {index}: {path}")
        if not isinstance(raw.get("path"), str) or not isinstance(raw.get("sha256"), str):
            raise ValueError(f"Input manifest row lacks path or SHA-256: {path}")
        records[index] = raw
    return records, sha256(path)


def _manifest_fingerprint(records: dict[int, dict[str, Any]]) -> tuple[tuple[Any, ...], ...]:
    return tuple(
        (
            index,
            row["sha256"],
            tuple(row.get("original_hw", ())),
        )
        for index, row in sorted(records.items())
    )


def _load_pair_arrays(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=False) as archive:
        required = ("index_a", "index_b", *CAMERA_METRICS)
        missing = [name for name in required if name not in archive]
        if missing:
            raise ValueError(f"Missing camera-pair arrays in {path}: {missing}")
        index_a = np.asarray(archive["index_a"], dtype=np.int64).reshape(-1)
        index_b = np.asarray(archive["index_b"], dtype=np.int64).reshape(-1)
        metrics = {
            name: np.asarray(archive[name], dtype=np.float64).reshape(-1)
            for name in CAMERA_METRICS
        }
    sizes = {index_a.size, index_b.size, *(values.size for values in metrics.values())}
    if len(sizes) != 1:
        raise ValueError(f"Camera-pair arrays have unequal lengths in {path}")
    if np.any(index_a == index_b):
        raise ValueError(f"Self-pairs are not valid camera-error evidence: {path}")
    return index_a, index_b, metrics


def _per_view_quantile(
    indices: set[int],
    index_a: np.ndarray,
    index_b: np.ndarray,
    values: np.ndarray,
    quantile: float,
) -> dict[int, float]:
    result: dict[int, float] = {}
    for index in sorted(indices):
        incident = values[(index_a == index) | (index_b == index)]
        finite = incident[np.isfinite(incident)]
        if finite.size == 0:
            raise ValueError(f"No finite incident camera errors for input index {index}")
        result[index] = float(np.quantile(finite, quantile))
    return result


def render_selected_views(
    metrics_jsonl: Path,
    dataset: str,
    scenes: list[str],
    models: list[str],
    output_dir: Path,
    *,
    top_k: int,
    quantile: float,
    max_side: int,
    jpeg_quality: int,
) -> dict[str, Any]:
    if not scenes or len(scenes) != len(set(scenes)):
        raise ValueError("Scenes must be a non-empty unique list")
    if not models or len(models) != len(set(models)):
        raise ValueError("Models must be a non-empty unique list")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be in [0, 1]")
    if max_side <= 0:
        raise ValueError("max_side must be positive")
    if not 1 <= jpeg_quality <= 95:
        raise ValueError("jpeg_quality must be in [1, 95]")
    if output_dir.exists():
        raise ValueError(f"Refusing to overwrite output directory: {output_dir}")

    rows = read_jsonl(metrics_jsonl)
    lookup: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        key = (row["model"], row["dataset"], row["scene"])
        if key in lookup:
            raise ValueError(f"Duplicate metric row: {key}")
        lookup[key] = row

    scene_plans: list[dict[str, Any]] = []
    for scene in scenes:
        reference_inputs: dict[int, dict[str, Any]] | None = None
        reference_fingerprint: tuple[tuple[Any, ...], ...] | None = None
        reasons_by_index: dict[int, list[dict[str, Any]]] = {}
        cases: list[dict[str, Any]] = []
        for model in models:
            key = (model, dataset, scene)
            if key not in lookup:
                raise ValueError(f"Snapshot does not contain selected case: {key}")
            run = Path(lookup[key]["run"])
            status = read_json(run / "status.json")
            validation = read_json(run / "validation.json")
            config = read_json(run / "run_config.json")
            if status.get("state") != "completed" or validation.get("passed") is not True:
                raise ValueError(f"Selected case is not completed and validated: {key}")
            if (config.get("dataset"), config.get("scene")) != (dataset, scene):
                raise ValueError(f"Run identity mismatch: {key}")

            input_manifest_path = run / "input_manifest.json"
            inputs, input_manifest_sha = _input_records(input_manifest_path)
            fingerprint = _manifest_fingerprint(inputs)
            if reference_inputs is None:
                reference_inputs = inputs
                reference_fingerprint = fingerprint
            elif fingerprint != reference_fingerprint:
                raise ValueError(f"Input population differs across models for {(dataset, scene)}")

            pairs_path = run / "evaluation_capture" / "camera_pairs.npz"
            index_a, index_b, pair_metrics = _load_pair_arrays(pairs_path)
            input_indices = set(inputs)
            pair_indices = set(index_a.tolist()) | set(index_b.tolist())
            if pair_indices != input_indices:
                raise ValueError(
                    f"Camera-pair and input populations differ for {key}: "
                    f"pairs={sorted(pair_indices)}, inputs={sorted(input_indices)}"
                )
            for metric, values in pair_metrics.items():
                per_view = _per_view_quantile(
                    input_indices, index_a, index_b, values, quantile
                )
                ranked = sorted(per_view.items(), key=lambda item: (-item[1], item[0]))
                for rank, (index, value) in enumerate(ranked[:top_k], start=1):
                    reasons_by_index.setdefault(index, []).append(
                        {
                            "model": model,
                            "metric": metric,
                            "incident_pair_quantile": quantile,
                            "value": value,
                            "rank": rank,
                        }
                    )
            cases.append(
                {
                    "model": model,
                    "run": str(run),
                    "input_manifest_sha256": input_manifest_sha,
                    "camera_pairs_sha256": sha256(pairs_path),
                }
            )
        assert reference_inputs is not None
        scene_plans.append(
            {
                "scene": scene,
                "inputs": reference_inputs,
                "reasons_by_index": reasons_by_index,
                "cases": cases,
            }
        )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent))
    try:
        scene_outputs: list[dict[str, Any]] = []
        for plan in scene_plans:
            scene = plan["scene"]
            scene_dir = temporary / slug(scene)
            scene_dir.mkdir(parents=True)
            view_outputs: list[dict[str, Any]] = []
            for index, reasons in sorted(plan["reasons_by_index"].items()):
                row = plan["inputs"][index]
                source = Path(row["path"])
                if not source.is_file():
                    raise FileNotFoundError(f"Selected source image is missing: {source}")
                source_sha = sha256(source)
                if source_sha != row["sha256"]:
                    raise ValueError(f"Selected source image SHA-256 mismatch: {source}")
                with Image.open(source) as opened:
                    image = ImageOps.exif_transpose(opened).convert("RGB")
                    original_size = list(image.size)
                    image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
                    rendered_size = list(image.size)
                    relative = Path(slug(scene)) / f"view-{index:03d}.jpg"
                    image.save(
                        temporary / relative,
                        format="JPEG",
                        quality=jpeg_quality,
                        optimize=True,
                    )
                output_path = temporary / relative
                view_outputs.append(
                    {
                        "index": index,
                        "source_path": str(source),
                        "source_sha256": source_sha,
                        "source_original_hw": row.get("original_hw"),
                        "decoded_wh": original_size,
                        "rendered_wh": rendered_size,
                        "output_path": str(relative),
                        "output_sha256": sha256(output_path),
                        "selection_reasons": reasons,
                    }
                )
            scene_outputs.append(
                {
                    "scene": scene,
                    "input_count": len(plan["inputs"]),
                    "selected_view_count": len(view_outputs),
                    "cases": plan["cases"],
                    "views": view_outputs,
                }
            )

        manifest = {
            "schema_version": 1,
            "metrics_jsonl": str(metrics_jsonl.resolve()),
            "metrics_jsonl_sha256": sha256(metrics_jsonl),
            "dataset": dataset,
            "models": models,
            "scenes": scenes,
            "top_k_per_model_metric": top_k,
            "incident_pair_quantile": quantile,
            "camera_metrics": list(CAMERA_METRICS),
            "uses_ground_truth_selection": True,
            "test_time_policy_input": False,
            "selection_policy": (
                "For each model and camera-error metric, rank input views by the requested "
                "quantile of their incident official-evaluator pair errors; retain the union "
                "of the top-k views for offline failure analysis only."
            ),
            "render": {
                "pillow_version": pillow_version,
                "max_side": max_side,
                "jpeg_quality": jpeg_quality,
                "exif_transpose": True,
                "resampling": "LANCZOS",
            },
            "scene_outputs": scene_outputs,
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
    parser.add_argument("--metrics-jsonl", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--scene", action="append", required=True)
    parser.add_argument("--model", action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--quantile", type=float, default=0.95)
    parser.add_argument("--max-side", type=int, default=1024)
    parser.add_argument("--jpeg-quality", type=int, default=90)
    args = parser.parse_args()
    manifest = render_selected_views(
        args.metrics_jsonl,
        args.dataset,
        args.scene,
        args.model,
        args.output_dir,
        top_k=args.top_k,
        quantile=args.quantile,
        max_side=args.max_side,
        jpeg_quality=args.jpeg_quality,
    )
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir.resolve()),
                "scenes": len(manifest["scene_outputs"]),
                "selected_views": sum(
                    scene["selected_view_count"] for scene in manifest["scene_outputs"]
                ),
                "uses_ground_truth_selection": manifest["uses_ground_truth_selection"],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
