#!/usr/bin/env python3
"""Select exact per-model intervention-gain extrema for evidence review."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from scripts.failure_mining.compare_baselines import sha256, write_csv


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def select_extrema(
    rows: list[dict[str, str]], metric: str
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    identities: set[tuple[str, str, str]] = set()
    for row in rows:
        if row.get("metric") != metric:
            continue
        identity = (row["model"], row["scene"], row["metric"])
        if identity in identities:
            raise ValueError(f"Duplicate intervention row: {identity}")
        identities.add(identity)
        gain = float(row["improvement_vs_control"])
        if not math.isfinite(gain):
            raise ValueError(f"Non-finite intervention gain: {identity}")
        grouped[row["model"]].append({**row, "improvement_vs_control": gain})

    if not grouped:
        raise ValueError(f"No rows found for metric: {metric}")

    selected: list[dict[str, Any]] = []
    for model, model_rows in sorted(grouped.items()):
        gains = [float(row["improvement_vs_control"]) for row in model_rows]
        extrema = (("minimum_gain", min(gains)), ("maximum_gain", max(gains)))
        for extremum, value in extrema:
            for row in model_rows:
                if float(row["improvement_vs_control"]) == value:
                    selected.append(
                        {
                            "model": model,
                            "metric": metric,
                            "extremum": extremum,
                            "scene": row["scene"],
                            "improvement_vs_control": value,
                            "native_resample_control": float(
                                row["native_resample_control"]
                            ),
                            "gt_extrinsics_intervention": float(
                                row["gt_extrinsics_intervention"]
                            ),
                        }
                    )
    return sorted(
        selected,
        key=lambda row: (row["model"], row["extremum"], row["scene"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-rows", type=Path, required=True)
    parser.add_argument("--metric", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    selected = select_extrema(read_csv(args.case_rows), args.metric)
    args.output.mkdir(parents=True, exist_ok=False)
    selection_path = args.output / "selected_intervention_contrasts.csv"
    write_csv(selection_path, selected)
    manifest = {
        "metric": args.metric,
        "models": sorted({row["model"] for row in selected}),
        "scenes": sorted({row["scene"] for row in selected}),
        "selected_rows": len(selected),
        "case_rows_sha256": sha256(args.case_rows),
        "selection_sha256": sha256(selection_path),
        "selector_sha256": sha256(Path(__file__)),
        "selection_policy": (
            "all exact per-model minima and maxima of improvement_vs_control; "
            "ties retained; no threshold or top-k"
        ),
        "failure_labels": None,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
