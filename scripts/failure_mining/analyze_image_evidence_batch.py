#!/usr/bin/env python3
"""Extract threshold-free image evidence for every run in a frozen batch."""

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
from scripts.failure_mining.summarize_image_evidence import summarize


SCENE_METRICS = (
    "luminance.entropy_bits",
    "luminance.std",
    "luminance.p01",
    "luminance.p99",
    "gradient.mean",
    "gradient.p90",
    "gradient.p95",
    "rgb.any_channel_at_floor_ratio",
    "rgb.any_channel_at_ceiling_ratio",
    "rgb.all_channels_at_floor_ratio",
    "rgb.all_channels_at_ceiling_ratio",
)


def analyze_run(run: Path, identity: dict) -> tuple[list[dict], dict]:
    entries = load_json(run / "input_manifest.json")
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"Input manifest must be a non-empty list: {run}")
    image_rows = [{**identity, **summarize(entry, None)} for entry in entries]
    indices = [row["index"] for row in image_rows]
    if len(indices) != len(set(indices)):
        raise ValueError(f"Duplicate input indices: {run}")

    scene = {"image.view_count": len(image_rows)}
    for metric in SCENE_METRICS:
        values = np.asarray([row[metric] for row in image_rows], dtype=np.float64)
        scene.update(finite_summary(values, f"image.{metric}"))
    return image_rows, scene


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    manifest = load_json(args.batch_dir / "manifest.json")
    records = load_results(args.batch_dir, manifest)
    image_rows = []
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
        run_images, run_scene = analyze_run(run, identity)
        image_rows.extend(run_images)
        row.update(run_scene)
        scene_rows.append(row)

    write_csv(args.output / "image_evidence.csv", image_rows)
    write_csv(args.output / "scene_image_evidence.csv", scene_rows)
    (args.output / "failed_runs.json").write_text(json.dumps(failures, indent=2) + "\n")
    analysis_manifest = {
        "batch_dir": str(args.batch_dir.resolve()),
        "batch_manifest_sha256": sha256(args.batch_dir / "manifest.json"),
        "batch_results_sha256": sha256(args.batch_dir / "results.jsonl"),
        "analysis_script_sha256": sha256(Path(__file__)),
        "image_module_sha256": sha256(Path(__file__).with_name("summarize_image_evidence.py")),
        "expected_scenes": len(manifest),
        "scene_rows": len(scene_rows),
        "image_rows": len(image_rows),
        "execution_failures": len(failures),
        "uses_ground_truth": False,
        "numeric_tolerance": None,
        "failure_labels": None,
        "policy": "descriptive RGB luminance, histogram entropy, and gradient distributions only",
    }
    (args.output / "manifest.json").write_text(
        json.dumps(analysis_manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(analysis_manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
