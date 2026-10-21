"""Relate image-only view statistics to official camera-pair error evidence.

This is an offline failure-mining analysis. Camera errors are derived from the
official evaluator and are never available to a test-time policy.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from scipy.stats import spearmanr

from scripts.failure_mining.compare_baselines import sha256, write_csv
from scripts.failure_mining.correlation_fdr import benjamini_hochberg


IMAGE_PREFIXES = ("gradient.", "luminance.", "rgb.")
EXCLUDED_SUFFIXES = (".count", ".finite_count")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _image_fields(rows: list[dict[str, str]]) -> list[str]:
    fields = sorted(
        field
        for field in rows[0]
        if field.startswith(IMAGE_PREFIXES)
        and not any(field.endswith(suffix) for suffix in EXCLUDED_SUFFIXES)
    )
    if not fields:
        raise ValueError("No image-evidence fields matched the declared policy")
    return fields


def analyze(
    image_rows: list[dict[str, str]],
    view_rows: list[dict[str, str]],
    *,
    datasets: set[str] | None,
    scenes: set[str] | None,
    camera_fields: tuple[str, ...],
) -> list[dict[str, Any]]:
    if not image_rows or not view_rows:
        raise ValueError("Image and camera-view tables must be non-empty")
    if not camera_fields or len(camera_fields) != len(set(camera_fields)):
        raise ValueError("Camera fields must be a non-empty unique tuple")

    images: dict[tuple[str, str, int], dict[str, str]] = {}
    for row in image_rows:
        key = (row["dataset"], row["scene"], int(row["index"]))
        if key in images:
            raise ValueError(f"Duplicate image-evidence view: {key}")
        images[key] = row
    fields = _image_fields(image_rows)

    groups: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    seen = set()
    for row in view_rows:
        dataset, scene = row["dataset"], row["scene"]
        if datasets is not None and dataset not in datasets:
            continue
        if scenes is not None and scene not in scenes:
            continue
        key = (dataset, scene, row["model"], int(row["view_index"]))
        if key in seen:
            raise ValueError(f"Duplicate camera-evidence view: {key}")
        seen.add(key)
        if key[:2] + (key[3],) not in images:
            raise ValueError(f"Camera view has no image evidence: {key}")
        groups[key[:3]].append(row)
    if not groups:
        raise ValueError("No camera-evidence groups matched the requested filters")

    for (dataset, scene, model), rows in groups.items():
        observed = {int(row["view_index"]) for row in rows}
        expected = {index for ds, sc, index in images if (ds, sc) == (dataset, scene)}
        if observed != expected:
            raise ValueError(
                f"Image and camera populations differ for {(dataset, scene, model)}: "
                f"camera={sorted(observed)}, image={sorted(expected)}"
            )

    output: list[dict[str, Any]] = []
    for (dataset, scene, model), rows in sorted(groups.items()):
        ordered = sorted(rows, key=lambda row: int(row["view_index"]))
        for camera_field in camera_fields:
            if camera_field not in ordered[0]:
                raise ValueError(f"Camera field is absent: {camera_field}")
            tests = []
            for image_field in fields:
                pairs = []
                for row in ordered:
                    index = int(row["view_index"])
                    raw_image = images[(dataset, scene, index)].get(image_field, "")
                    raw_camera = row.get(camera_field, "")
                    if raw_image in {"", None} or raw_camera in {"", None}:
                        continue
                    image_value, camera_value = float(raw_image), float(raw_camera)
                    if math.isfinite(image_value) and math.isfinite(camera_value):
                        pairs.append((image_value, camera_value))
                if len(pairs) < 3:
                    continue
                image_values = [value[0] for value in pairs]
                camera_values = [value[1] for value in pairs]
                if len(set(image_values)) < 2 or len(set(camera_values)) < 2:
                    continue
                result = spearmanr(image_values, camera_values)
                if not (math.isfinite(result.statistic) and math.isfinite(result.pvalue)):
                    continue
                tests.append(
                    {
                        "dataset": dataset,
                        "scene": scene,
                        "model": model,
                        "camera_field": camera_field,
                        "image_field": image_field,
                        "views": len(pairs),
                        "spearman_rho": float(result.statistic),
                        "p_value_two_sided": float(result.pvalue),
                    }
                )
            q_values = benjamini_hochberg([row["p_value_two_sided"] for row in tests])
            for row, q_value in zip(tests, q_values, strict=True):
                row["bh_q_value_within_endpoint"] = q_value
                row["tests_in_endpoint_family"] = len(tests)
                output.append(row)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-evidence", type=Path, required=True)
    parser.add_argument("--camera-view-evidence", type=Path, required=True)
    parser.add_argument("--dataset", action="append")
    parser.add_argument("--scene", action="append")
    parser.add_argument(
        "--camera-field",
        action="append",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    camera_fields = tuple(
        args.camera_field
        or (
            "camera.rotation_degrees.p95",
            "camera.translation_degrees.p95",
        )
    )
    rows = analyze(
        read_csv(args.image_evidence),
        read_csv(args.camera_view_evidence),
        datasets=set(args.dataset) if args.dataset else None,
        scenes=set(args.scene) if args.scene else None,
        camera_fields=camera_fields,
    )
    output_path = args.output / "view_image_camera_correlations.csv"
    write_csv(output_path, rows)
    manifest = {
        "image_evidence": str(args.image_evidence.resolve()),
        "camera_view_evidence": str(args.camera_view_evidence.resolve()),
        "image_evidence_sha256": sha256(args.image_evidence),
        "camera_view_evidence_sha256": sha256(args.camera_view_evidence),
        "datasets": args.dataset,
        "scenes": args.scene,
        "camera_fields": list(camera_fields),
        "image_prefixes": list(IMAGE_PREFIXES),
        "excluded_suffixes": list(EXCLUDED_SUFFIXES),
        "rows": len(rows),
        "output_sha256": sha256(output_path),
        "analysis_script_sha256": sha256(Path(__file__)),
        "multiple_testing": "Benjamini-Hochberg within each scene/model/camera-field endpoint",
        "uses_ground_truth_target": True,
        "test_time_policy_input": False,
        "failure_labels": None,
        "quality_thresholds": None,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
