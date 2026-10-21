"""Apply within-endpoint FDR correction to exhaustive evidence correlations."""

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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    if any(not math.isfinite(value) or value < 0 or value > 1 for value in p_values):
        raise ValueError("p-values must be finite and in [0, 1]")
    order = sorted(range(len(p_values)), key=p_values.__getitem__)
    adjusted = [0.0] * len(p_values)
    running = 1.0
    total = len(p_values)
    for reverse_rank, index in enumerate(reversed(order), 1):
        rank = total - reverse_rank + 1
        running = min(running, p_values[index] * total / rank)
        adjusted[index] = min(running, 1.0)
    return adjusted


def analyze(
    evidence_rows: list[dict[str, str]],
    target_rows: list[dict[str, str]],
    evidence_prefix: str,
    excluded_suffixes: tuple[str, ...],
    target_value_field: str,
    target_metric_field: str,
    datasets: set[str] | None,
    metrics: set[str] | None,
    evidence_fields: list[str] | None = None,
) -> list[dict[str, Any]]:
    if not evidence_rows or not target_rows:
        raise ValueError("Evidence and target tables must be non-empty")
    evidence = {}
    for row in evidence_rows:
        key = (row["dataset"], row["scene"])
        if key in evidence:
            raise ValueError(f"Duplicate evidence scene: {key}")
        evidence[key] = row
    if evidence_fields is None:
        fields = sorted(
            field
            for field in evidence_rows[0]
            if field.startswith(evidence_prefix)
            and not any(field.endswith(suffix) for suffix in excluded_suffixes)
        )
    else:
        if not evidence_fields or len(evidence_fields) != len(set(evidence_fields)):
            raise ValueError("Explicit evidence fields must be non-empty and unique")
        missing = set(evidence_fields) - set(evidence_rows[0])
        if missing:
            raise ValueError(f"Explicit evidence fields are absent: {sorted(missing)}")
        fields = list(evidence_fields)
    if not fields:
        raise ValueError("No evidence fields matched the requested prefix/suffix policy")

    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    seen = set()
    for row in target_rows:
        dataset, scene, metric = row["dataset"], row["scene"], row[target_metric_field]
        if datasets is not None and dataset not in datasets:
            continue
        if metrics is not None and metric not in metrics:
            continue
        key = (dataset, scene, metric)
        if key in seen:
            raise ValueError(f"Duplicate target scene/metric: {key}")
        seen.add(key)
        if (dataset, scene) not in evidence:
            raise ValueError(f"Target scene has no evidence row: {(dataset, scene)}")
        groups[(dataset, metric)].append(row)
    if not groups:
        raise ValueError("No target groups matched the requested filters")

    output = []
    for (dataset, metric), rows in sorted(groups.items()):
        tests = []
        for field in fields:
            pairs = []
            for row in rows:
                raw_evidence = evidence[(dataset, row["scene"])].get(field, "")
                raw_target = row.get(target_value_field, "")
                if raw_evidence in {"", None} or raw_target in {"", None}:
                    continue
                evidence_value, target_value = float(raw_evidence), float(raw_target)
                if math.isfinite(evidence_value) and math.isfinite(target_value):
                    pairs.append((evidence_value, target_value))
            if len(pairs) < 3:
                continue
            evidence_values = [pair[0] for pair in pairs]
            target_values = [pair[1] for pair in pairs]
            if len(set(evidence_values)) < 2 or len(set(target_values)) < 2:
                continue
            result = spearmanr(evidence_values, target_values)
            if not (math.isfinite(result.statistic) and math.isfinite(result.pvalue)):
                continue
            tests.append(
                {
                    "dataset": dataset,
                    "target_metric": metric,
                    "target_value_field": target_value_field,
                    "evidence_field": field,
                    "scenes": len(pairs),
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
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--evidence-prefix")
    parser.add_argument("--evidence-field", action="append")
    parser.add_argument("--exclude-suffix", action="append", default=[])
    parser.add_argument("--target-value-field", required=True)
    parser.add_argument("--target-metric-field", required=True)
    parser.add_argument("--dataset", action="append")
    parser.add_argument("--metric", action="append")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (args.evidence_prefix is None) == (args.evidence_field is None):
        raise ValueError("Specify exactly one of --evidence-prefix or --evidence-field")
    args.output.mkdir(parents=True, exist_ok=False)
    rows = analyze(
        read_csv(args.evidence),
        read_csv(args.targets),
        args.evidence_prefix or "",
        tuple(args.exclude_suffix),
        args.target_value_field,
        args.target_metric_field,
        set(args.dataset) if args.dataset else None,
        set(args.metric) if args.metric else None,
        args.evidence_field,
    )
    output_path = args.output / "correlations_with_fdr.csv"
    write_csv(output_path, rows)
    manifest = {
        "evidence": str(args.evidence.resolve()),
        "targets": str(args.targets.resolve()),
        "evidence_sha256": sha256(args.evidence),
        "targets_sha256": sha256(args.targets),
        "evidence_prefix": args.evidence_prefix,
        "evidence_fields": args.evidence_field,
        "excluded_suffixes": args.exclude_suffix,
        "target_value_field": args.target_value_field,
        "target_metric_field": args.target_metric_field,
        "datasets": args.dataset,
        "metrics": args.metric,
        "rows": len(rows),
        "output_sha256": sha256(output_path),
        "analysis_script_sha256": sha256(Path(__file__)),
        "multiple_testing": "Benjamini-Hochberg within each dataset/target-metric family",
        "failure_labels": None,
        "quality_thresholds": None,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
