#!/usr/bin/env python3
"""Compare observable scene evidence with per-metric backbone-oracle gain."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

from scripts.failure_mining.compare_baselines import pearson, sha256, write_csv
from scripts.failure_mining.compare_image_evidence import average_ranks


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def build_correlations(
    evidence_rows: list[dict[str, str]],
    oracle_rows: list[dict[str, str]],
    evidence_fields: list[str],
) -> list[dict]:
    if not evidence_rows or not oracle_rows:
        raise ValueError("Evidence and oracle tables must both be non-empty")
    missing = [field for field in evidence_fields if field not in evidence_rows[0]]
    if missing:
        raise ValueError(f"Evidence fields are absent: {missing}")
    if len(evidence_fields) != len(set(evidence_fields)):
        raise ValueError("Evidence fields must be unique")

    evidence = {}
    for row in evidence_rows:
        key = (row["dataset"], row["scene"])
        if key in evidence:
            raise ValueError(f"Duplicate evidence scene: {key}")
        evidence[key] = row

    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    oracle_keys = set()
    for row in oracle_rows:
        key = (row["dataset"], row["scene"], row["metric"])
        if key in oracle_keys:
            raise ValueError(f"Duplicate oracle scene/metric: {key}")
        oracle_keys.add(key)
        if key[:2] not in evidence:
            raise ValueError(f"Oracle scene has no evidence row: {key[:2]}")
        gain = float(row["oracle_gain"])
        if not math.isfinite(gain) or gain < 0:
            raise ValueError(f"Invalid oracle gain for {key}: {gain}")
        grouped[(row["dataset"], row["metric"])].append(row)

    output = []
    for (dataset, metric), rows in sorted(grouped.items()):
        directions = {row["direction"] for row in rows}
        if len(directions) != 1:
            raise ValueError(f"Oracle direction differs for {(dataset, metric)}")
        for field in evidence_fields:
            pairs = []
            for row in sorted(rows, key=lambda item: item["scene"]):
                raw = evidence[(dataset, row["scene"])].get(field, "")
                if raw in {"", None}:
                    continue
                value = float(raw)
                if math.isfinite(value):
                    pairs.append((value, float(row["oracle_gain"])))
            if len(pairs) < 2:
                continue
            output.append(
                {
                    "dataset": dataset,
                    "metric": metric,
                    "direction": next(iter(directions)),
                    "evidence_field": field,
                    "scenes": len(pairs),
                    "spearman_evidence_value_vs_oracle_gain": pearson(
                        average_ranks([value for value, _gain in pairs]),
                        average_ranks([gain for _value, gain in pairs]),
                    ),
                }
            )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-scenes", type=Path, required=True)
    parser.add_argument("--oracle-scenes", type=Path, required=True)
    parser.add_argument("--evidence-field", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    correlations = build_correlations(
        read_csv(args.evidence_scenes),
        read_csv(args.oracle_scenes),
        args.evidence_field,
    )
    output_path = args.output / "evidence_oracle_gain_correlations.csv"
    write_csv(output_path, correlations)
    manifest = {
        "evidence_scenes": str(args.evidence_scenes.resolve()),
        "oracle_scenes": str(args.oracle_scenes.resolve()),
        "evidence_fields": args.evidence_field,
        "correlation_rows": len(correlations),
        "evidence_scenes_sha256": sha256(args.evidence_scenes),
        "oracle_scenes_sha256": sha256(args.oracle_scenes),
        "output_sha256": sha256(output_path),
        "analysis_script_sha256": sha256(Path(__file__)),
        "failure_labels": None,
        "composite_score": None,
        "policy": "within-dataset, per-metric Spearman against GT oracle gain",
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
