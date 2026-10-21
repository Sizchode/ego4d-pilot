"""Independently audit rendered extremal views and their selection evidence."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from scripts.audit.analysis_outputs import load_json, require, sha256
from scripts.failure_mining.render_selected_input_views import (
    CAMERA_METRICS,
    _input_records,
    _load_pair_arrays,
    _manifest_fingerprint,
)


def _per_view_maximum(
    indices: set[int], index_a: np.ndarray, index_b: np.ndarray, values: np.ndarray
) -> dict[int, float]:
    result: dict[int, float] = {}
    for index in sorted(indices):
        incident = values[(index_a == index) | (index_b == index)]
        finite = incident[np.isfinite(incident)]
        require(finite.size > 0, f"No finite incident pair error for view {index}")
        result[index] = float(np.max(finite))
    return result


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def audit_views(bundles_dir: Path, views_dir: Path) -> dict[str, Any]:
    bundles_dir = bundles_dir.resolve()
    views_dir = views_dir.resolve()
    bundle_manifest_path = bundles_dir / "manifest.json"
    selected_runs_path = bundles_dir / "selected_runs.jsonl"
    views_manifest_path = views_dir / "manifest.json"
    bundle_manifest = load_json(bundle_manifest_path)
    manifest = load_json(views_manifest_path)

    require(manifest.get("bundle_manifest_sha256") == sha256(bundle_manifest_path), "Bundle manifest hash mismatch")
    require(manifest.get("selected_runs_sha256") == sha256(selected_runs_path), "Selected-run hash mismatch")
    require(manifest.get("models") == bundle_manifest.get("models"), "Model population/order differs")
    require(manifest.get("camera_metrics") == list(CAMERA_METRICS), "Camera metric population/order differs")
    require(manifest.get("selection_threshold") is None, "Rendered selection uses a threshold")
    require(manifest.get("top_k") is None, "Rendered selection uses top-k")
    require(manifest.get("ties") == "all exact ties retained", "Tie policy differs")
    require(manifest.get("uses_ground_truth_selection") is True, "Ground-truth selection flag is missing")
    require(manifest.get("test_time_policy_input") is False, "Rendered evidence is marked as policy input")

    rows = _read_jsonl(selected_runs_path)
    grouped: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        key = (row["dataset"], row["scene"])
        require(row["model"] not in grouped[key], f"Duplicate selected run: {(row['model'], *key)}")
        grouped[key][row["model"]] = row
    observed_scenes = {
        (row["dataset"], row["scene"]): row for row in manifest.get("scene_outputs", [])
    }
    require(len(observed_scenes) == len(manifest.get("scene_outputs", [])), "Duplicate rendered scene")
    require(set(observed_scenes) == set(grouped), "Rendered scene population differs")

    output_hashes: dict[str, str] = {}
    total_views = 0
    models = manifest["models"]
    max_side = int(manifest["render"]["max_side"])
    require(max_side > 0, "Invalid rendered max side")
    for key, by_model in sorted(grouped.items()):
        dataset, scene = key
        require(list(by_model) == models, f"Cross-model order/population differs: {key}")
        expected_reasons: dict[int, list[dict[str, Any]]] = defaultdict(list)
        expected_cases = []
        reference_fingerprint = None
        reference_inputs = None
        for model in models:
            run = Path(by_model[model]["run"])
            status = load_json(run / "status.json")
            validation = load_json(run / "validation.json")
            config = load_json(run / "run_config.json")
            require(status.get("state") == "completed" and validation.get("passed") is True, f"Invalid selected run: {(model, *key)}")
            require((config.get("dataset"), config.get("scene")) == key, f"Run identity mismatch: {(model, *key)}")
            inputs, input_hash = _input_records(run / "input_manifest.json")
            fingerprint = _manifest_fingerprint(inputs)
            if reference_fingerprint is None:
                reference_fingerprint = fingerprint
                reference_inputs = inputs
            else:
                require(fingerprint == reference_fingerprint, f"Input population differs: {key}")
            pairs_path = run / "evaluation_capture" / "camera_pairs.npz"
            index_a, index_b, pair_metrics = _load_pair_arrays(pairs_path)
            indices = set(inputs)
            require(set(index_a.tolist()) | set(index_b.tolist()) == indices, f"Pair/input population differs: {(model, *key)}")
            for metric in CAMERA_METRICS:
                per_view = _per_view_maximum(indices, index_a, index_b, pair_metrics[metric])
                maximum = max(per_view.values())
                minimum = min(per_view.values())
                for index, value in per_view.items():
                    for extremum, target in (("maximum", maximum), ("minimum_control", minimum)):
                        if value == target:
                            expected_reasons[index].append(
                                {
                                    "model": model,
                                    "metric": metric,
                                    "per_view_statistic": "maximum_incident_pair_error",
                                    "extremum": extremum,
                                    "value": value,
                                }
                            )
            expected_cases.append(
                {
                    "model": model,
                    "run": str(run),
                    "input_manifest_sha256": input_hash,
                    "camera_pairs_sha256": sha256(pairs_path),
                }
            )
        require(reference_inputs is not None, f"No inputs: {key}")
        scene_row = observed_scenes[key]
        require(scene_row.get("cases") == expected_cases, f"Rendered case provenance differs: {key}")
        require(scene_row.get("input_count") == len(reference_inputs), f"Input count differs: {key}")
        views = scene_row.get("views")
        require(isinstance(views, list), f"Invalid rendered views: {key}")
        observed_views = {int(row["index"]): row for row in views}
        require(len(observed_views) == len(views), f"Duplicate rendered view: {key}")
        require(set(observed_views) == set(expected_reasons), f"Extremal view population differs: {key}")
        require(scene_row.get("selected_view_count") == len(views), f"Rendered view count differs: {key}")
        for index, reasons in expected_reasons.items():
            row = observed_views[index]
            source = Path(reference_inputs[index]["path"])
            require(row.get("source_path") == str(source), f"Source path differs: {(*key, index)}")
            require(row.get("source_sha256") == sha256(source), f"Source hash differs: {(*key, index)}")
            require(row.get("selection_reasons") == reasons, f"Selection reasons differ: {(*key, index)}")
            relative = Path(row["output_path"])
            require(not relative.is_absolute() and ".." not in relative.parts, f"Unsafe output path: {relative}")
            rendered = views_dir / relative
            require(rendered.is_file(), f"Rendered view is missing: {rendered}")
            rendered_hash = sha256(rendered)
            require(row.get("output_sha256") == rendered_hash, f"Rendered view hash differs: {rendered}")
            with Image.open(rendered) as image:
                require(image.format == "JPEG", f"Rendered view is not JPEG: {rendered}")
                require(list(image.size) == row.get("rendered_wh"), f"Rendered dimensions differ: {rendered}")
                require(max(image.size) <= max_side, f"Rendered view exceeds max side: {rendered}")
            output_hashes[str(relative)] = rendered_hash
        total_views += len(views)

    require(manifest.get("selected_scenes") == len(grouped), "Selected scene count differs")
    require(manifest.get("selected_views") == total_views, "Selected view count differs")
    expected_files = {"manifest.json", *output_hashes}
    actual_files = {
        str(path.relative_to(views_dir)) for path in views_dir.rglob("*") if path.is_file()
    }
    require(actual_files == expected_files, "Rendered output file population differs")
    return {
        "schema_version": 1,
        "passed": True,
        "selected_scenes": len(grouped),
        "selected_views": total_views,
        "bundle_manifest_sha256": sha256(bundle_manifest_path),
        "selected_runs_sha256": sha256(selected_runs_path),
        "views_manifest_sha256": sha256(views_manifest_path),
        "output_sha256": output_hashes,
        "selection_threshold": None,
        "top_k": None,
        "uses_ground_truth_selection": True,
        "test_time_policy_input": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundles", type=Path, required=True)
    parser.add_argument("--views", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit_views(args.bundles, args.views)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    require(not args.output.exists(), f"Refusing to overwrite existing audit: {args.output}")
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("passed", "selected_scenes", "selected_views")}, indent=2))


if __name__ == "__main__":
    main()
