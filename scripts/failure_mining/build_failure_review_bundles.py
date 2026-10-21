"""Build audited cross-model evidence bundles for selected review scenes."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

from scripts.failure_mining.archive_case_bundles import REQUIRED_FILES, build_archive
from scripts.failure_mining.compare_baselines import sha256


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_json(path: Path) -> Any:
    require(path.is_file(), f"Required file is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    require(path.is_file(), f"Required file is missing: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def build_bundles(
    selection_dir: Path,
    summary_dir: Path,
    summary_audit_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    require(not output_dir.exists(), f"Refusing to overwrite output: {output_dir}")
    selection_manifest_path = selection_dir / "manifest.json"
    selection_cases_path = selection_dir / "selected_review_cases.csv"
    summary_manifest_path = summary_dir / "manifest.json"
    summary_cases_path = summary_dir / "scene_model_results.csv"
    selection_manifest = read_json(selection_manifest_path)
    summary_manifest = read_json(summary_manifest_path)
    summary_audit = read_json(summary_audit_path)
    require(
        selection_manifest.get("cases_output_sha256") == sha256(selection_cases_path),
        "Selection table hash mismatch",
    )
    require(selection_manifest.get("selection_threshold") is None, "Selection uses a threshold")
    require(selection_manifest.get("top_k") is None, "Selection uses top-k")
    require(summary_audit.get("passed") is True, "Baseline summary audit did not pass")
    require(
        summary_audit.get("summary_manifest_sha256") == sha256(summary_manifest_path),
        "Summary audit covers a different manifest",
    )
    require(
        summary_manifest.get("scene_output_sha256") == sha256(summary_cases_path),
        "Summary scene table hash mismatch",
    )
    models = selection_manifest.get("models")
    require(models == summary_manifest.get("models"), "Selection and summary model order differ")

    selected_rows = read_csv(selection_cases_path)
    require(selected_rows, "No selected review cases")
    scenes_by_dataset: dict[str, set[str]] = defaultdict(set)
    for row in selected_rows:
        require(row["model"] in models, f"Unknown selected model: {row['model']}")
        scenes_by_dataset[row["dataset"]].add(row["scene"])

    summary_rows = read_csv(summary_cases_path)
    lookup: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in summary_rows:
        key = (row["model"], row["dataset"], row["scene"])
        require(key not in lookup, f"Duplicate summary case: {key}")
        lookup[key] = row
    expanded = []
    for dataset in sorted(scenes_by_dataset):
        for scene in sorted(scenes_by_dataset[dataset]):
            for model in models:
                key = (model, dataset, scene)
                require(key in lookup, f"Summary does not contain selected cross-model case: {key}")
                source = lookup[key]
                expanded.append(
                    {
                        "model": model,
                        "dataset": dataset,
                        "scene": scene,
                        "run": source["run"],
                        "state": source["state"],
                        "views": int(source["views"]),
                    }
                )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent))
    try:
        selected_runs = temporary / "selected_runs.jsonl"
        selected_runs.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in expanded),
            encoding="utf-8",
        )
        shutil.copy2(selection_cases_path, temporary / "selected_review_cases.csv")
        shutil.copy2(selection_manifest_path, temporary / "selection_manifest.json")
        shutil.copy2(summary_manifest_path, temporary / "baseline_summary_manifest.json")
        shutil.copy2(summary_audit_path, temporary / "baseline_summary_audit.json")

        bundles = []
        for dataset in sorted(scenes_by_dataset):
            scenes = sorted(scenes_by_dataset[dataset])
            tar_path = temporary / f"{dataset}-review-cases.tar"
            archive_manifest = build_archive(
                selected_runs, dataset, scenes, models, tar_path
            )
            bundles.append(
                {
                    "dataset": dataset,
                    "scenes": scenes,
                    "models": models,
                    "cases": archive_manifest["case_count"],
                    "files": archive_manifest["file_count"],
                    "tar": tar_path.name,
                    "tar_bytes": tar_path.stat().st_size,
                    "tar_sha256": sha256(tar_path),
                }
            )
        manifest = {
            "schema_version": 1,
            "selection_manifest_sha256": sha256(selection_manifest_path),
            "selection_cases_sha256": sha256(selection_cases_path),
            "baseline_summary_manifest_sha256": sha256(summary_manifest_path),
            "baseline_summary_audit_sha256": sha256(summary_audit_path),
            "selected_runs_sha256": sha256(selected_runs),
            "models": models,
            "selected_review_rows": len(selected_rows),
            "selected_unique_scenes": sum(len(scenes) for scenes in scenes_by_dataset.values()),
            "expanded_cross_model_cases": len(expanded),
            "required_files_per_case": list(REQUIRED_FILES),
            "expansion_policy": "every selected dataset/scene is archived for every baseline model",
            "bundles": bundles,
            "uses_official_evaluator_selection": True,
            "test_time_policy_input": False,
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
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--summary-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = build_bundles(
        args.selection, args.summary, args.summary_audit, args.output
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
