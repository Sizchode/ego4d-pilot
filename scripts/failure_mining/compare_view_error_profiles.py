"""Compare GT camera-error profiles across reconstruction backbones.

The input contains one row per model and input view. Camera errors come from
the official evaluator, so this module is strictly for offline failure mining;
its outputs must never be used as test-time policy inputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr

from scripts.failure_mining.compare_baselines import sha256, write_csv


DEFAULT_FIELDS = (
    "camera.rotation_degrees.p95",
    "camera.translation_degrees.p95",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def compare(
    rows: list[dict[str, str]],
    *,
    fields: tuple[str, ...],
    datasets: set[str] | None = None,
    scenes: set[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not rows:
        raise ValueError("View-error table must be non-empty")
    if not fields or len(fields) != len(set(fields)):
        raise ValueError("Fields must be a non-empty unique tuple")
    required = {"dataset", "scene", "model", "view_index", *fields}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"View-error table is missing columns: {sorted(missing)}")

    grouped: dict[tuple[str, str], dict[str, dict[int, dict[str, str]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for row in rows:
        dataset, scene = row["dataset"], row["scene"]
        if datasets is not None and dataset not in datasets:
            continue
        if scenes is not None and scene not in scenes:
            continue
        model, index = row["model"], int(row["view_index"])
        if index in grouped[(dataset, scene)][model]:
            raise ValueError(f"Duplicate model/view row: {(dataset, scene, model, index)}")
        grouped[(dataset, scene)][model][index] = row
    if not grouped:
        raise ValueError("No view-error rows matched the requested filters")

    pair_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for (dataset, scene), by_model in sorted(grouped.items()):
        models = sorted(by_model)
        if len(models) < 2:
            raise ValueError(f"Need at least two models for {(dataset, scene)}")
        populations = {model: set(by_model[model]) for model in models}
        expected = populations[models[0]]
        for model in models[1:]:
            if populations[model] != expected:
                raise ValueError(
                    f"View populations differ for {(dataset, scene)}: "
                    f"{models[0]}={sorted(expected)}, {model}={sorted(populations[model])}"
                )

        for field in fields:
            field_rows = []
            for model_a, model_b in combinations(models, 2):
                pairs = []
                for index in sorted(expected):
                    raw_a = by_model[model_a][index][field]
                    raw_b = by_model[model_b][index][field]
                    if raw_a in {"", None} or raw_b in {"", None}:
                        continue
                    value_a, value_b = float(raw_a), float(raw_b)
                    if math.isfinite(value_a) and math.isfinite(value_b):
                        pairs.append((value_a, value_b))
                record: dict[str, Any] = {
                    "dataset": dataset,
                    "scene": scene,
                    "field": field,
                    "model_a": model_a,
                    "model_b": model_b,
                    "shared_views": len(expected),
                    "finite_pairs": len(pairs),
                    "spearman_rho": None,
                    "p_value_two_sided": None,
                }
                if len(pairs) < 3:
                    record["status"] = "insufficient_finite_pairs"
                elif len({pair[0] for pair in pairs}) < 2 or len({pair[1] for pair in pairs}) < 2:
                    record["status"] = "constant_input"
                else:
                    result = spearmanr(
                        [pair[0] for pair in pairs], [pair[1] for pair in pairs]
                    )
                    if not (
                        math.isfinite(result.statistic) and math.isfinite(result.pvalue)
                    ):
                        record["status"] = "non_finite_result"
                    else:
                        record["status"] = "ok"
                        record["spearman_rho"] = float(result.statistic)
                        record["p_value_two_sided"] = float(result.pvalue)
                pair_rows.append(record)
                field_rows.append(record)

            finite = [
                float(row["spearman_rho"])
                for row in field_rows
                if row["status"] == "ok"
            ]
            summary_rows.append(
                {
                    "dataset": dataset,
                    "scene": scene,
                    "field": field,
                    "models": len(models),
                    "views": len(expected),
                    "possible_model_pairs": len(field_rows),
                    "finite_model_pairs": len(finite),
                    "spearman_rho_min": min(finite) if finite else None,
                    "spearman_rho_median": float(np.median(finite)) if finite else None,
                    "spearman_rho_max": max(finite) if finite else None,
                }
            )
    return pair_rows, summary_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--field", action="append")
    parser.add_argument("--dataset", action="append")
    parser.add_argument("--scene", action="append")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    fields = tuple(args.field or DEFAULT_FIELDS)
    pair_rows, summary_rows = compare(
        read_csv(args.input),
        fields=fields,
        datasets=set(args.dataset) if args.dataset else None,
        scenes=set(args.scene) if args.scene else None,
    )
    args.output.mkdir(parents=True, exist_ok=False)
    pair_path = args.output / "pairwise_view_error_rank_agreement.csv"
    summary_path = args.output / "view_error_rank_agreement_summary.csv"
    write_csv(pair_path, pair_rows)
    write_csv(summary_path, summary_rows)
    manifest = {
        "input": str(args.input.resolve()),
        "input_sha256": sha256(args.input),
        "fields": list(fields),
        "datasets": args.dataset,
        "scenes": args.scene,
        "pairwise_rows": len(pair_rows),
        "summary_rows": len(summary_rows),
        "pairwise_sha256": sha256(pair_path),
        "summary_sha256": sha256(summary_path),
        "analysis_script_sha256": sha256(Path(__file__)),
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
