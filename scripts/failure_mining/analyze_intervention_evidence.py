#!/usr/bin/env python3
"""Relate test-time reconstruction evidence to offline intervention gains."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

from scipy.stats import spearmanr

from scripts.failure_mining.compare_baselines import sha256, write_csv
from scripts.failure_mining.correlation_fdr import benjamini_hochberg


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def merge_evidence(
    sources: list[tuple[str, list[dict[str, str]]]],
    dataset: str,
    fields: list[str],
) -> dict[str, dict[str, float]]:
    if not sources:
        raise ValueError("At least one evidence source is required")
    if not fields or len(fields) != len(set(fields)):
        raise ValueError("Evidence fields must be a non-empty unique list")

    owners: dict[str, str] = {}
    merged: dict[str, dict[str, float]] = {}
    for source_name, rows in sources:
        if not rows:
            raise ValueError(f"Evidence source is empty: {source_name}")
        columns = set(rows[0])
        for field in fields:
            if field in columns:
                if field in owners:
                    raise ValueError(
                        f"Evidence field appears in multiple sources: {field} "
                        f"({owners[field]}, {source_name})"
                    )
                owners[field] = source_name

        seen: set[str] = set()
        source_fields = [field for field in fields if owners.get(field) == source_name]
        if not source_fields:
            continue
        for row in rows:
            if row.get("dataset") != dataset:
                continue
            scene = row["scene"]
            if scene in seen:
                raise ValueError(f"Duplicate evidence scene in {source_name}: {scene}")
            seen.add(scene)
            values = merged.setdefault(scene, {})
            for field in source_fields:
                raw = row.get(field, "")
                if raw in {"", None}:
                    continue
                value = float(raw)
                if math.isfinite(value):
                    values[field] = value

    missing = [field for field in fields if field not in owners]
    if missing:
        raise ValueError(f"Evidence fields are absent from all sources: {missing}")
    return merged


def join_intervention(
    intervention_rows: list[dict[str, str]],
    evidence: dict[str, dict[str, float]],
    dataset: str,
    model: str,
    metric: str,
    fields: list[str],
) -> list[dict[str, Any]]:
    joined: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in intervention_rows:
        if row.get("model") != model or row.get("metric") != metric:
            continue
        scene = row["scene"]
        if scene in seen:
            raise ValueError(f"Duplicate intervention scene: {scene}")
        seen.add(scene)
        if scene not in evidence:
            raise ValueError(f"Intervention scene has no evidence: {scene}")
        gain = float(row["improvement_vs_control"])
        if not math.isfinite(gain):
            raise ValueError(f"Non-finite intervention gain: {scene}")
        joined.append(
            {
                "dataset": dataset,
                "scene": scene,
                "model": model,
                "metric": metric,
                "improvement_vs_control": gain,
                **{field: evidence[scene].get(field, "") for field in fields},
            }
        )
    if not joined:
        raise ValueError(f"No intervention rows matched model={model!r}, metric={metric!r}")
    return sorted(joined, key=lambda row: row["scene"])


def correlate(joined: list[dict[str, Any]], fields: list[str]) -> list[dict[str, Any]]:
    tests: list[dict[str, Any]] = []
    for field in fields:
        pairs = [
            (float(row[field]), float(row["improvement_vs_control"]))
            for row in joined
            if row[field] not in {"", None}
        ]
        if len(pairs) < 3:
            continue
        evidence_values = [pair[0] for pair in pairs]
        gains = [pair[1] for pair in pairs]
        if len(set(evidence_values)) < 2 or len(set(gains)) < 2:
            continue
        result = spearmanr(evidence_values, gains)
        if not (math.isfinite(result.statistic) and math.isfinite(result.pvalue)):
            continue
        tests.append(
            {
                "evidence_field": field,
                "scenes": len(pairs),
                "spearman_evidence_vs_intervention_gain": float(result.statistic),
                "p_value_two_sided": float(result.pvalue),
            }
        )
    q_values = benjamini_hochberg([row["p_value_two_sided"] for row in tests])
    for row, q_value in zip(tests, q_values, strict=True):
        row["bh_q_value_within_analysis"] = q_value
        row["tests_in_family"] = len(tests)
    return tests


def parse_source(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not name or not path:
        raise argparse.ArgumentTypeError("Evidence source must have NAME=PATH form")
    return name, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", action="append", type=parse_source, required=True)
    parser.add_argument("--evidence-field", action="append", required=True)
    parser.add_argument("--intervention-case-rows", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--metric", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    evidence_sources = [(name, read_csv(path)) for name, path in args.evidence]
    evidence = merge_evidence(evidence_sources, args.dataset, args.evidence_field)
    joined = join_intervention(
        read_csv(args.intervention_case_rows),
        evidence,
        args.dataset,
        args.model,
        args.metric,
        args.evidence_field,
    )
    correlations = correlate(joined, args.evidence_field)

    args.output.mkdir(parents=True, exist_ok=False)
    joined_path = args.output / "scene_intervention_evidence.csv"
    correlation_path = args.output / "evidence_gain_correlations.csv"
    write_csv(joined_path, joined)
    write_csv(correlation_path, correlations)
    manifest = {
        "dataset": args.dataset,
        "model": args.model,
        "metric": args.metric,
        "scenes": len(joined),
        "evidence_fields": args.evidence_field,
        "evidence_sources": {
            name: {"path": str(path.resolve()), "sha256": sha256(path)}
            for name, path in args.evidence
        },
        "intervention_case_rows_sha256": sha256(args.intervention_case_rows),
        "joined_output_sha256": sha256(joined_path),
        "correlation_output_sha256": sha256(correlation_path),
        "analysis_script_sha256": sha256(Path(__file__)),
        "correlation_rows": len(correlations),
        "multiple_testing": "Benjamini-Hochberg across requested evidence fields",
        "evidence_uses_ground_truth": False,
        "target_uses_ground_truth_intervention": True,
        "test_time_policy": None,
        "selection_threshold": None,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
