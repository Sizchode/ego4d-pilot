"""Compare model rankings from a diagnostic proxy and an official endpoint."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr

from scripts.failure_mining.compare_baselines import sha256, write_csv


DIRECTIONS = {"lower_is_better", "higher_is_better"}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def utility(value: float, direction: str) -> float:
    require(direction in DIRECTIONS, f"Unknown direction: {direction}")
    return value if direction == "higher_is_better" else -value


def compare(
    rows: list[dict[str, str]],
    *,
    proxy_field: str,
    target_field: str,
    proxy_direction: str,
    target_direction: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    require(rows, "Case table must be non-empty")
    require(proxy_field != target_field, "Proxy and target fields must differ")
    require(proxy_direction in DIRECTIONS and target_direction in DIRECTIONS, "Invalid direction")
    required = {"dataset", "scene", "model", proxy_field, target_field}
    missing = required - set(rows[0])
    require(not missing, f"Case table is missing columns: {sorted(missing)}")

    grouped: dict[tuple[str, str], dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        key = (row["dataset"], row["scene"])
        model = row["model"]
        require(model not in grouped[key], f"Duplicate case/model row: {(*key, model)}")
        grouped[key][model] = row

    scene_rows = []
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for (dataset, scene), by_model in sorted(grouped.items()):
        require(len(by_model) >= 3, f"Need at least three models: {(dataset, scene)}")
        models = sorted(by_model)
        proxy_values = [float(by_model[model][proxy_field]) for model in models]
        target_values = [float(by_model[model][target_field]) for model in models]
        require(all(math.isfinite(value) for value in [*proxy_values, *target_values]), f"Non-finite metric: {(dataset, scene)}")
        proxy_utilities = [utility(value, proxy_direction) for value in proxy_values]
        target_utilities = [utility(value, target_direction) for value in target_values]
        result = spearmanr(proxy_utilities, target_utilities)
        require(math.isfinite(result.statistic) and math.isfinite(result.pvalue), f"Undefined rank agreement: {(dataset, scene)}")
        proxy_best = max(proxy_utilities)
        target_best = max(target_utilities)
        proxy_winners = [model for model, value in zip(models, proxy_utilities) if value == proxy_best]
        target_winners = [model for model, value in zip(models, target_utilities) if value == target_best]
        row = {
            "dataset": dataset,
            "scene": scene,
            "proxy_field": proxy_field,
            "target_field": target_field,
            "models": len(models),
            "spearman_rho": float(result.statistic),
            "p_value_two_sided": float(result.pvalue),
            "proxy_best_models": ";".join(proxy_winners),
            "target_best_models": ";".join(target_winners),
            "winner_overlap": bool(set(proxy_winners) & set(target_winners)),
        }
        scene_rows.append(row)
        by_dataset[dataset].append(row)

    summary_rows = []
    for dataset, selected in sorted(by_dataset.items()):
        coefficients = [float(row["spearman_rho"]) for row in selected]
        summary_rows.append(
            {
                "dataset": dataset,
                "proxy_field": proxy_field,
                "target_field": target_field,
                "scenes": len(selected),
                "winner_overlap_scenes": sum(bool(row["winner_overlap"]) for row in selected),
                "spearman_rho_min": min(coefficients),
                "spearman_rho_median": float(np.median(coefficients)),
                "spearman_rho_max": max(coefficients),
            }
        )
    return scene_rows, summary_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--proxy-field", required=True)
    parser.add_argument("--target-field", required=True)
    parser.add_argument("--proxy-direction", choices=sorted(DIRECTIONS), required=True)
    parser.add_argument("--target-direction", choices=sorted(DIRECTIONS), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    scene_rows, summary_rows = compare(
        read_csv(args.input),
        proxy_field=args.proxy_field,
        target_field=args.target_field,
        proxy_direction=args.proxy_direction,
        target_direction=args.target_direction,
    )
    require(not args.output.exists(), f"Refusing to overwrite output: {args.output}")
    args.output.mkdir(parents=True)
    scene_path = args.output / "scene_rank_agreement.csv"
    summary_path = args.output / "rank_agreement_summary.csv"
    write_csv(scene_path, scene_rows)
    write_csv(summary_path, summary_rows)
    manifest = {
        "input": str(args.input.resolve()),
        "input_sha256": sha256(args.input),
        "proxy_field": args.proxy_field,
        "target_field": args.target_field,
        "proxy_direction": args.proxy_direction,
        "target_direction": args.target_direction,
        "scene_rows": len(scene_rows),
        "summary_rows": len(summary_rows),
        "scene_output_sha256": sha256(scene_path),
        "summary_output_sha256": sha256(summary_path),
        "analysis_script_sha256": sha256(Path(__file__)),
        "uses_ground_truth_proxy": True,
        "test_time_policy_input": False,
        "failure_labels": None,
        "quality_thresholds": None,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
