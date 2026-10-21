#!/usr/bin/env python3
"""Compare two official failure rankings within each model and dataset."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from scripts.failure_mining.compare_baselines import pearson, sha256, write_csv


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def compare(
    rows: list[dict[str, str]], metric_left: str, metric_right: str
) -> list[dict]:
    selected = [row for row in rows if row["metric"] in {metric_left, metric_right}]
    groups: dict[tuple[str, str], dict[str, dict[str, dict[str, str]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for row in selected:
        key = (row["model"], row["dataset"])
        metric_rows = groups[key][row["metric"]]
        if row["scene"] in metric_rows:
            raise ValueError(
                f"Duplicate scene ranking for {(row['model'], row['dataset'], row['metric'], row['scene'])}"
            )
        metric_rows[row["scene"]] = row

    output = []
    for (model, dataset), metrics in sorted(groups.items()):
        if metric_left not in metrics or metric_right not in metrics:
            continue
        left = metrics[metric_left]
        right = metrics[metric_right]
        if set(left) != set(right):
            raise ValueError(
                f"Scene population differs between {metric_left} and {metric_right} "
                f"for {(model, dataset)}"
            )
        scenes = sorted(left)
        left_directions = {row["direction"] for row in left.values()}
        right_directions = {row["direction"] for row in right.values()}
        if len(left_directions) != 1 or len(right_directions) != 1:
            raise ValueError(f"Metric direction is inconsistent for {(model, dataset)}")
        output.append(
            {
                "model": model,
                "dataset": dataset,
                "metric_left": metric_left,
                "metric_right": metric_right,
                "direction_left": next(iter(left_directions)),
                "direction_right": next(iter(right_directions)),
                "scenes": len(scenes),
                "spearman_failure_rank": pearson(
                    [float(left[scene]["failure_rank"]) for scene in scenes],
                    [float(right[scene]["failure_rank"]) for scene in scenes],
                ),
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rankings", type=Path, required=True)
    parser.add_argument("--metric-left", required=True)
    parser.add_argument("--metric-right", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.metric_left == args.metric_right:
        raise ValueError("Metrics must differ")
    args.output.mkdir(parents=True, exist_ok=False)

    rows = compare(read_csv(args.rankings), args.metric_left, args.metric_right)
    output_path = args.output / "metric_rank_agreement.csv"
    write_csv(output_path, rows)
    manifest = {
        "rankings": str(args.rankings.resolve()),
        "rankings_sha256": sha256(args.rankings),
        "metric_left": args.metric_left,
        "metric_right": args.metric_right,
        "comparison_rows": len(rows),
        "output_sha256": sha256(output_path),
        "analysis_script_sha256": sha256(Path(__file__)),
        "failure_labels": None,
        "composite_score": None,
        "policy": "within-model, within-dataset Spearman of separate official failure ranks",
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
