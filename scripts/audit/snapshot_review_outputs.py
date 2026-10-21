"""Audit archived-case review tables and rendered evidence.

This audit connects a selected evidence archive to an immutable snapshot.  It
verifies that descriptive case metrics reproduce the snapshot's official
metrics exactly, that per-view tables cover every selected view exactly once,
and that rendered/derived outputs match their manifests.  Ground-truth camera
errors remain explicitly offline failure-analysis evidence.
"""

from __future__ import annotations

import argparse
import csv
import json
import tempfile
from pathlib import Path
from typing import Any

from PIL import Image

from scripts.failure_mining.compare_baselines import sha256, write_csv
from scripts.failure_mining.compare_view_error_profiles import compare


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"Expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    values = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            value = json.loads(line)
            require(isinstance(value, dict), f"Expected JSON object at {path}:{line_number}")
            values.append(value)
    return values


def nested_metric(metrics: dict[str, Any], dotted: str) -> float:
    value: Any = metrics
    for part in dotted.split("."):
        require(isinstance(value, dict) and part in value, f"Missing metric {dotted}")
        value = value[part]
    require(isinstance(value, (int, float)), f"Metric is not numeric: {dotted}")
    return float(value)


def audit(
    *,
    archive_dir: Path,
    case_metrics_path: Path,
    view_metrics_path: Path,
    snapshot_metrics_path: Path,
    contact_sheets_dir: Path,
    view_profiles_dir: Path,
    dataset: str,
) -> dict[str, Any]:
    archive_manifest_path = archive_dir / "archive_manifest.json"
    archive = read_json(archive_manifest_path)
    require(archive.get("dataset") == dataset, "Archive dataset differs")
    require(archive.get("failure_labels") is None, "Archive contains failure labels")
    cases = archive.get("cases")
    require(isinstance(cases, list) and cases, "Archive contains no cases")
    require(archive.get("case_count") == len(cases), "Archive case count differs")

    by_prefix: dict[str, dict[str, Any]] = {}
    selected_keys: set[tuple[str, str, str]] = set()
    models: list[str] = []
    scenes: list[str] = []
    for case in cases:
        prefix = case["archive_prefix"]
        require(prefix not in by_prefix, f"Duplicate archive prefix: {prefix}")
        require(case["dataset"] == dataset, f"Case dataset differs: {prefix}")
        require((archive_dir / prefix).is_dir(), f"Missing archived case: {prefix}")
        by_prefix[prefix] = case
        selected_keys.add((dataset, case["scene"], case["model"]))
        if case["model"] not in models:
            models.append(case["model"])
        if case["scene"] not in scenes:
            scenes.append(case["scene"])
    require(len(selected_keys) == len(cases), "Archive case identities are not unique")
    require(archive.get("models") == models, "Archive model order differs from cases")
    require(archive.get("scenes") == scenes, "Archive scene order differs from cases")

    snapshot_rows = read_jsonl(snapshot_metrics_path)
    selected_snapshot: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in snapshot_rows:
        key = (row.get("dataset"), row.get("scene"), row.get("model"))
        if key in selected_keys:
            require(key not in selected_snapshot, f"Duplicate snapshot metric row: {key}")
            selected_snapshot[key] = row
    require(set(selected_snapshot) == selected_keys, "Snapshot population differs from archive")

    case_rows = read_csv(case_metrics_path)
    require(len(case_rows) == len(cases), "Case-metric row count differs")
    case_by_prefix: dict[str, dict[str, str]] = {}
    metric_columns = sorted(
        column[len("metric.") :]
        for column in case_rows[0]
        if column.startswith("metric.")
    )
    require(metric_columns, "Case table contains no official metrics")
    official_metric_comparisons = 0
    for row in case_rows:
        prefix = Path(row["case_dir"]).name
        require(prefix in by_prefix, f"Unknown case directory: {prefix}")
        require(prefix not in case_by_prefix, f"Duplicate case row: {prefix}")
        case = by_prefix[prefix]
        expected_dir = (archive_dir / prefix).resolve()
        require(Path(row["case_dir"]).resolve() == expected_dir, f"Case path differs: {prefix}")
        require(row["dataset"] == dataset and row["scene"] == case["scene"], f"Case identity differs: {prefix}")
        require(row["state"] in {"completed", "reused_verified_run"}, f"Incomplete case: {prefix}")
        require(row["validation_passed"].lower() == "true", f"Unvalidated case: {prefix}")
        snapshot = selected_snapshot[(dataset, case["scene"], case["model"])]
        require(int(row["views"]) == int(snapshot["views"]), f"View count differs: {prefix}")
        for metric in metric_columns:
            observed = float(row[f"metric.{metric}"])
            expected = nested_metric(snapshot["metrics"], metric)
            require(observed == expected, f"Official metric differs: {prefix} {metric}")
            official_metric_comparisons += 1
        case_by_prefix[prefix] = row
    require(set(case_by_prefix) == set(by_prefix), "Case-table population differs")

    view_rows = read_csv(view_metrics_path)
    expected_view_rows = sum(int(row["views"]) for row in case_rows)
    require(len(view_rows) == expected_view_rows, "Per-view row count differs")
    view_populations: dict[str, set[int]] = {prefix: set() for prefix in by_prefix}
    for row in view_rows:
        prefix = Path(row["case_dir"]).name
        require(prefix in case_by_prefix, f"Unknown per-view case: {prefix}")
        case_row = case_by_prefix[prefix]
        require(row["dataset"] == dataset and row["scene"] == case_row["scene"], f"Per-view identity differs: {prefix}")
        require(row["model"] == case_row["model"], f"Per-view model differs: {prefix}")
        view_index = int(row["view_index"])
        require(view_index not in view_populations[prefix], f"Duplicate per-view row: {(prefix, view_index)}")
        view_populations[prefix].add(view_index)
        views = int(case_row["views"])
        require(int(row["incident_pair_count"]) == views - 1, f"Incident-pair count differs: {prefix}")
    for prefix, population in view_populations.items():
        require(population == set(range(int(case_by_prefix[prefix]["views"]))), f"View population differs: {prefix}")

    sheets_manifest_path = contact_sheets_dir / "manifest.json"
    sheets = read_json(sheets_manifest_path)
    require(sheets.get("dataset") == dataset, "Contact-sheet dataset differs")
    require(sheets.get("models") == models, "Contact-sheet model order differs")
    require(sheets.get("scenes") == sorted(scenes), "Contact-sheet scene population differs")
    require(sheets.get("test_time_policy_input") is False, "Contact sheets marked as test-time input")
    require(sheets.get("failure_labels") is None, "Contact sheets contain failure labels")
    archive_refs = sheets.get("archives")
    require(isinstance(archive_refs, list) and len(archive_refs) == 1, "Contact-sheet archive reference differs")
    require(archive_refs[0]["manifest_sha256"] == sha256(archive_manifest_path), "Contact-sheet archive hash differs")
    for output in sheets["outputs"]:
        path = contact_sheets_dir / output["output"]
        require(path.is_file() and sha256(path) == output["sha256"], f"Contact-sheet hash differs: {path}")
        with Image.open(path) as image:
            require(list(image.size) == [output["width"], output["height"]], f"Contact-sheet dimensions differ: {path}")

    profiles_manifest_path = view_profiles_dir / "manifest.json"
    profiles = read_json(profiles_manifest_path)
    require(profiles.get("input_sha256") == sha256(view_metrics_path), "View-profile input hash differs")
    require(profiles.get("uses_ground_truth_target") is True, "View profiles do not declare GT use")
    require(profiles.get("test_time_policy_input") is False, "View profiles marked as test-time input")
    pair_path = view_profiles_dir / "pairwise_view_error_rank_agreement.csv"
    summary_path = view_profiles_dir / "view_error_rank_agreement_summary.csv"
    require(profiles.get("pairwise_sha256") == sha256(pair_path), "Pairwise profile hash differs")
    require(profiles.get("summary_sha256") == sha256(summary_path), "Profile summary hash differs")
    pair_rows, summary_rows = compare(
        view_rows,
        fields=tuple(profiles["fields"]),
        datasets={dataset},
    )
    with tempfile.TemporaryDirectory() as directory:
        expected_pair = Path(directory) / "pair.csv"
        expected_summary = Path(directory) / "summary.csv"
        write_csv(expected_pair, pair_rows)
        write_csv(expected_summary, summary_rows)
        require(expected_pair.read_bytes() == pair_path.read_bytes(), "Pairwise profiles do not reproduce")
        require(expected_summary.read_bytes() == summary_path.read_bytes(), "Profile summaries do not reproduce")

    return {
        "schema_version": 1,
        "passed": True,
        "dataset": dataset,
        "models": models,
        "scenes": scenes,
        "case_count": len(cases),
        "case_metric_rows": len(case_rows),
        "per_view_rows": len(view_rows),
        "official_metric_columns": metric_columns,
        "official_metric_comparisons": official_metric_comparisons,
        "contact_sheet_count": len(sheets["outputs"]),
        "profile_pairwise_rows": len(pair_rows),
        "profile_summary_rows": len(summary_rows),
        "archive_manifest_sha256": sha256(archive_manifest_path),
        "snapshot_metrics_sha256": sha256(snapshot_metrics_path),
        "case_metrics_sha256": sha256(case_metrics_path),
        "view_metrics_sha256": sha256(view_metrics_path),
        "contact_sheets_manifest_sha256": sha256(sheets_manifest_path),
        "view_profiles_manifest_sha256": sha256(profiles_manifest_path),
        "uses_ground_truth_camera_errors": True,
        "test_time_policy_input": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-dir", type=Path, required=True)
    parser.add_argument("--case-metrics", type=Path, required=True)
    parser.add_argument("--view-metrics", type=Path, required=True)
    parser.add_argument("--snapshot-metrics", type=Path, required=True)
    parser.add_argument("--contact-sheets-dir", type=Path, required=True)
    parser.add_argument("--view-profiles-dir", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), f"Refusing to overwrite output: {args.output}")
    result = audit(
        archive_dir=args.archive_dir,
        case_metrics_path=args.case_metrics,
        view_metrics_path=args.view_metrics,
        snapshot_metrics_path=args.snapshot_metrics,
        contact_sheets_dir=args.contact_sheets_dir,
        view_profiles_dir=args.view_profiles_dir,
        dataset=args.dataset,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
