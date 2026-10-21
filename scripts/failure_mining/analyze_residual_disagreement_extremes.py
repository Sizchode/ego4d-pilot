"""Compare test-time observables at exact residual-misranking extrema.

The residual-underestimate and residual-overestimate view in each scene form a
paired observation.  Ground-truth depth error is used only to select these
offline review cases; it is never included among the compared observables.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any

from scipy.stats import wilcoxon

from scripts.failure_mining.compare_baselines import sha256, write_csv
from scripts.failure_mining.correlation_fdr import benjamini_hochberg


UNDER = "residual_underestimates_gt_error"
OVER = "residual_overestimates_gt_error"
OBSERVABLE_FIELDS = (
    "image.gradient.mean",
    "image.gradient.p95",
    "image.luminance.entropy_bits",
    "image.rgb.any_channel_at_floor_ratio",
    "image.rgb.any_channel_at_ceiling_ratio",
    "visibility.incoming.source_depth_not_behind_ratio.mean",
    "visibility.outgoing.source_depth_not_behind_ratio.mean",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_csv(path: Path) -> list[dict[str, str]]:
    require(path.is_file(), f"Required file is missing: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def finite(value: str | None) -> float | None:
    if value in {None, ""}:
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) else None


def pair_observables(
    selection_rows: list[dict[str, str]],
    evidence_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    evidence: dict[tuple[str, str, int], dict[str, str]] = {}
    for row in evidence_rows:
        key = (row["dataset"], row["scene"], int(row["view_index"]))
        require(key not in evidence, f"Duplicate evidence row: {key}")
        evidence[key] = row
    require(evidence, "Evidence table is empty")

    selections: dict[tuple[str, str], dict[str, dict[str, str]]] = {}
    for row in selection_rows:
        scene_key = (row["dataset"], row["scene"])
        role = row["role"]
        require(role in {UNDER, OVER}, f"Unexpected selection role: {role}")
        role_rows = selections.setdefault(scene_key, {})
        require(role not in role_rows, f"Duplicate role for scene: {(scene_key, role)}")
        role_rows[role] = row
    require(selections, "Selection table is empty")

    paired = []
    for (dataset, scene), role_rows in sorted(selections.items()):
        require(set(role_rows) == {UNDER, OVER}, f"Scene lacks a complete pair: {(dataset, scene)}")
        under_selection, over_selection = role_rows[UNDER], role_rows[OVER]
        under_key = (dataset, scene, int(under_selection["view_index"]))
        over_key = (dataset, scene, int(over_selection["view_index"]))
        require(under_key in evidence and over_key in evidence, f"Selected view lacks evidence: {(under_key, over_key)}")
        under_evidence, over_evidence = evidence[under_key], evidence[over_key]
        for field in OBSERVABLE_FIELDS:
            under_value, over_value = finite(under_evidence.get(field)), finite(over_evidence.get(field))
            require(under_value is not None and over_value is not None, f"Missing paired observable: {(dataset, scene, field)}")
            paired.append(
                {
                    "dataset": dataset,
                    "scene": scene,
                    "evidence_field": field,
                    "underestimate_view_index": under_key[2],
                    "overestimate_view_index": over_key[2],
                    "underestimate_value": under_value,
                    "overestimate_value": over_value,
                    "underestimate_minus_overestimate": under_value - over_value,
                }
            )
    return paired


def summarize_pairs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    require(rows, "Paired observable table is empty")
    tests = []
    for field in OBSERVABLE_FIELDS:
        field_rows = [row for row in rows if row["evidence_field"] == field]
        require(field_rows, f"No paired rows for observable: {field}")
        differences = [float(row["underestimate_minus_overestimate"]) for row in field_rows]
        nonzero = [value for value in differences if value != 0]
        p_value = 1.0 if not nonzero else float(wilcoxon(nonzero, alternative="two-sided").pvalue)
        tests.append(
            {
                "evidence_field": field,
                "scene_pairs": len(field_rows),
                "mean_underestimate_minus_overestimate": statistics.fmean(differences),
                "median_underestimate_minus_overestimate": statistics.median(differences),
                "underestimate_greater_scenes": sum(value > 0 for value in differences),
                "equal_scenes": sum(value == 0 for value in differences),
                "underestimate_lower_scenes": sum(value < 0 for value in differences),
                "wilcoxon_p_value_two_sided": p_value,
            }
        )
    q_values = benjamini_hochberg([row["wilcoxon_p_value_two_sided"] for row in tests])
    for row, q_value in zip(tests, q_values, strict=True):
        row["bh_q_value"] = q_value
        row["tests_in_family"] = len(tests)
    return tests


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-analysis", type=Path, required=True)
    parser.add_argument("--view-evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), f"Refusing to overwrite output: {args.output}")

    selection_manifest_path = args.selection_analysis / "manifest.json"
    selection_path = args.selection_analysis / "residual_rank_disagreement_extremes.csv"
    evidence_manifest_path = args.view_evidence / "manifest.json"
    evidence_path = args.view_evidence / "merged_view_evidence.csv"
    selection_manifest = json.loads(selection_manifest_path.read_text(encoding="utf-8"))
    evidence_manifest = json.loads(evidence_manifest_path.read_text(encoding="utf-8"))
    require(selection_manifest.get("selection_output_sha256") == sha256(selection_path), "Selection hash differs")
    require(selection_manifest.get("uses_ground_truth_analysis") is True, "Missing offline-GT provenance")
    require(selection_manifest.get("test_time_policy_input") is False, "Selection GT is marked as a policy input")
    require(selection_manifest.get("quality_thresholds") is None, "Selection used a quality threshold")
    require(selection_manifest.get("top_k") is None, "Selection used top-k tuning")
    require(evidence_manifest.get("merged_output_sha256") == sha256(evidence_path), "Evidence hash differs")

    paired = pair_observables(read_csv(selection_path), read_csv(evidence_path))
    summary = summarize_pairs(paired)
    args.output.mkdir(parents=True, exist_ok=False)
    paired_path = args.output / "paired_extreme_observables.csv"
    summary_path = args.output / "paired_observable_summary.csv"
    write_csv(paired_path, paired)
    write_csv(summary_path, summary)
    manifest = {
        "scenes": len({(row["dataset"], row["scene"]) for row in paired}),
        "paired_rows": len(paired),
        "observable_fields": list(OBSERVABLE_FIELDS),
        "pairing": "within scene: exact residual-underestimate extreme minus exact residual-overestimate extreme",
        "statistical_unit": "scene pair",
        "multiple_testing": "Benjamini-Hochberg over all predeclared test-time observables",
        "selection_manifest_sha256": sha256(selection_manifest_path),
        "selection_sha256": sha256(selection_path),
        "evidence_manifest_sha256": sha256(evidence_manifest_path),
        "evidence_sha256": sha256(evidence_path),
        "paired_output_sha256": sha256(paired_path),
        "summary_output_sha256": sha256(summary_path),
        "analysis_script_sha256": sha256(Path(__file__)),
        "uses_ground_truth_for_offline_case_selection": True,
        "test_time_policy_input": False,
        "failure_labels": None,
        "quality_thresholds": None,
        "top_k": None,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
