"""Audit a materialized offline failure-review report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image

from scripts.audit.analysis_outputs import load_json, require, sha256


def audit_report(
    bundles_dir: Path,
    views_dir: Path,
    views_audit_path: Path,
    report_dir: Path,
) -> dict[str, Any]:
    bundle_manifest_path = bundles_dir / "manifest.json"
    selection_path = bundles_dir / "selected_review_cases.csv"
    views_manifest_path = views_dir / "manifest.json"
    report_manifest_path = report_dir / "manifest.json"
    bundle_manifest = load_json(bundle_manifest_path)
    views_audit = load_json(views_audit_path)
    manifest = load_json(report_manifest_path)
    require(views_audit.get("passed") is True, "Extremal-view audit did not pass")
    require(manifest.get("bundle_manifest_sha256") == sha256(bundle_manifest_path), "Bundle manifest hash mismatch")
    require(manifest.get("selection_cases_sha256") == sha256(selection_path), "Selection table hash mismatch")
    require(manifest.get("views_manifest_sha256") == sha256(views_manifest_path), "View manifest hash mismatch")
    require(manifest.get("views_audit_sha256") == sha256(views_audit_path), "View audit hash mismatch")
    require(manifest.get("failure_labels") is None, "Report assigns failure labels")
    require(manifest.get("selection_threshold") is None, "Report uses a threshold")
    require(manifest.get("top_k") is None, "Report uses top-k")
    require(manifest.get("uses_ground_truth_selection") is True, "Ground-truth selection flag is missing")
    require(manifest.get("test_time_policy_input") is False, "Report is marked as policy input")
    require(manifest.get("models") == bundle_manifest.get("models"), "Report model population/order differs")
    expected_tars = [
        {"path": row["tar"], "sha256": row["tar_sha256"]}
        for row in bundle_manifest.get("bundles", [])
    ]
    require(manifest.get("source_tars") == expected_tars, "Report tar provenance differs")
    for row in expected_tars:
        require(sha256(bundles_dir / row["path"]) == row["sha256"], f"Source tar hash differs: {row['path']}")

    outputs = manifest.get("outputs")
    require(isinstance(outputs, dict) and "index.html" in outputs, "Invalid report output manifest")
    actual_files = {str(path.relative_to(report_dir)) for path in report_dir.rglob("*") if path.is_file()}
    require(actual_files == {"manifest.json", *outputs}, "Report file population differs")
    for relative, expected_hash in outputs.items():
        relative_path = Path(relative)
        require(not relative_path.is_absolute() and ".." not in relative_path.parts, f"Unsafe report output path: {relative}")
        path = report_dir / relative
        require(sha256(path) == expected_hash, f"Report output hash differs: {relative}")
        if path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            with Image.open(path) as image:
                image.verify()
    scene_records = manifest.get("scene_records")
    require(isinstance(scene_records, list), "Invalid scene records")
    require(manifest.get("selected_scenes") == len(scene_records), "Selected scene count differs")
    models = manifest["models"]
    referenced_outputs = {"index.html"}
    scene_keys = set()
    for scene in scene_records:
        key = (scene["dataset"], scene["scene"])
        require(key not in scene_keys, f"Duplicate report scene: {key}")
        scene_keys.add(key)
        require([row["model"] for row in scene.get("models", [])] == models, f"Scene model population/order differs: {key}")
        for view in scene.get("views", []):
            referenced_outputs.add(view["path"])
            require(outputs.get(view["path"]) == view["sha256"], f"View asset provenance differs: {(key, view.get('index'))}")
        for model in scene.get("models", []):
            for asset in model.get("assets", {}).values():
                referenced_outputs.add(asset["path"])
                require(outputs.get(asset["path"]) == asset["sha256"], f"Model asset provenance differs: {(key, model['model'], asset['path'])}")
    require(referenced_outputs == set(outputs), "Manifest contains unreferenced report outputs")
    expected_cases = sum(len(row.get("models", [])) for row in scene_records)
    require(manifest.get("expanded_cross_model_cases") == expected_cases, "Expanded case count differs")
    return {
        "schema_version": 1,
        "passed": True,
        "selected_scenes": len(scene_records),
        "expanded_cross_model_cases": expected_cases,
        "report_manifest_sha256": sha256(report_manifest_path),
        "output_sha256": outputs,
        "failure_labels": None,
        "selection_threshold": None,
        "top_k": None,
        "uses_ground_truth_selection": True,
        "test_time_policy_input": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundles", type=Path, required=True)
    parser.add_argument("--views", type=Path, required=True)
    parser.add_argument("--views-audit", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit_report(args.bundles, args.views, args.views_audit, args.report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    require(not args.output.exists(), f"Refusing to overwrite existing audit: {args.output}")
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("passed", "selected_scenes", "expanded_cross_model_cases")}, indent=2))


if __name__ == "__main__":
    main()
