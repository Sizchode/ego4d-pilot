"""Join endpoint-winner disagreements with per-scene oracle headroom."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from scripts.failure_mining.compare_baselines import sha256, write_csv


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def summarize(
    winner_rows: list[dict[str, str]],
    oracle_rows: list[dict[str, str]],
    target_metric: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    require(winner_rows and oracle_rows, "Inputs must be non-empty")
    winners: dict[tuple[str, str], dict[str, str]] = {}
    for row in winner_rows:
        key = (row["dataset"], row["scene"])
        require(key not in winners, f"Duplicate winner row: {key}")
        require(row["metric_right"] == target_metric, f"Winner target metric differs: {key}")
        winners[key] = row

    oracle: dict[tuple[str, str], dict[str, str]] = {}
    for row in oracle_rows:
        if row["metric"] != target_metric:
            continue
        key = (row["dataset"], row["scene"])
        require(key not in oracle, f"Duplicate oracle row: {key}")
        oracle[key] = row
    require(set(winners) <= set(oracle), "Oracle table is missing winner-comparison scenes")

    disagreements = []
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for key, winner in sorted(winners.items()):
        if winner["winner_overlap"].lower() == "true":
            continue
        item = oracle[key]
        gain = float(item["oracle_gain"])
        require(math.isfinite(gain) and gain >= 0, f"Invalid oracle gain: {key}")
        row = {
            "dataset": key[0],
            "scene": key[1],
            "pose_metric": winner["metric_left"],
            "geometry_metric": target_metric,
            "pose_best_models": winner["left_best_models"],
            "geometry_best_models": winner["right_best_models"],
            "model_rank_spearman_rho": winner["spearman_rho"],
            "best_fixed_geometry_model": item["best_fixed_model"],
            "best_fixed_geometry_value": float(item["best_fixed_value"]),
            "oracle_geometry_models": item["oracle_models"],
            "oracle_geometry_value": float(item["oracle_value"]),
            "oracle_gain": gain,
            "positive_oracle_headroom": gain > 0,
        }
        disagreements.append(row)
        by_dataset[key[0]].append(row)

    disagreements.sort(key=lambda row: (row["dataset"], -float(row["oracle_gain"]), row["scene"]))
    summary_rows = []
    for dataset, rows in sorted(by_dataset.items()):
        gains = [float(row["oracle_gain"]) for row in rows]
        summary_rows.append(
            {
                "dataset": dataset,
                "pose_metric": rows[0]["pose_metric"],
                "geometry_metric": target_metric,
                "winner_disagreement_scenes": len(rows),
                "positive_oracle_headroom_scenes": sum(value > 0 for value in gains),
                "zero_oracle_headroom_scenes": sum(value == 0 for value in gains),
                "oracle_gain_mean": statistics.fmean(gains),
                "oracle_gain_median": statistics.median(gains),
                "oracle_gain_max": max(gains),
            }
        )
    return disagreements, summary_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--winner-rows", type=Path, required=True)
    parser.add_argument("--oracle-rows", type=Path, required=True)
    parser.add_argument("--target-metric", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), f"Refusing to overwrite output: {args.output}")
    disagreement_rows, summary_rows = summarize(
        read_csv(args.winner_rows), read_csv(args.oracle_rows), args.target_metric
    )
    args.output.mkdir(parents=True)
    disagreements_path = args.output / "endpoint_winner_disagreements.csv"
    summary_path = args.output / "endpoint_disagreement_summary.csv"
    write_csv(disagreements_path, disagreement_rows)
    write_csv(summary_path, summary_rows)
    manifest = {
        "winner_rows": str(args.winner_rows.resolve()),
        "winner_rows_sha256": sha256(args.winner_rows),
        "oracle_rows": str(args.oracle_rows.resolve()),
        "oracle_rows_sha256": sha256(args.oracle_rows),
        "target_metric": args.target_metric,
        "disagreement_rows": len(disagreement_rows),
        "summary_rows": len(summary_rows),
        "disagreements_output_sha256": sha256(disagreements_path),
        "summary_output_sha256": sha256(summary_path),
        "analysis_script_sha256": sha256(Path(__file__)),
        "failure_labels": None,
        "selection_threshold": None,
        "top_k": None,
        "uses_ground_truth_metric_oracle": True,
        "test_time_policy_input": False,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
