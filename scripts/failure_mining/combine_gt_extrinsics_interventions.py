"""Combine disjoint, independently audited GT-extrinsics interventions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.failure_mining.compare_baselines import sha256, write_csv
from scripts.failure_mining.summarize_gt_extrinsics_intervention import (
    aggregate_case_rows,
    summarize,
)


def combine(
    intervention_dirs: list[Path],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if len(intervention_dirs) < 2:
        raise ValueError("At least two intervention directories are required")
    resolved = [path.resolve() for path in intervention_dirs]
    if len(resolved) != len(set(resolved)):
        raise ValueError("Intervention directories must be unique")

    rows: list[dict[str, Any]] = []
    sources = []
    identities: set[tuple[str, str, str]] = set()
    for path in resolved:
        case_rows, _summaries, provenance = summarize(path)
        for row in case_rows:
            identity = (str(row["model"]), str(row["scene"]), str(row["metric"]))
            if identity in identities:
                raise ValueError(f"Duplicate combined intervention row: {identity}")
            identities.add(identity)
            rows.append(row)
        sources.append(provenance)

    rows.sort(key=lambda row: (str(row["scene"]), str(row["model"]), str(row["metric"])))
    summaries = aggregate_case_rows(rows)
    provenance = {
        "sources": sources,
        "source_count": len(sources),
        "cases": len({(row["model"], row["scene"]) for row in rows}),
        "case_metric_rows": len(rows),
        "summary_rows": len(summaries),
        "scenes": sorted({str(row["scene"]) for row in rows}),
        "models": sorted({str(row["model"]) for row in rows}),
        "uses_ground_truth_intervention": True,
        "test_time_policy_input": False,
        "success_threshold": None,
        "combination_policy": "union of disjoint, independently hash-verified case rows",
    }
    return rows, summaries, provenance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--intervention-dir", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    case_rows, summary_rows, provenance = combine(args.intervention_dir)
    case_path = args.output / "case_metric_deltas.csv"
    summary_path = args.output / "model_metric_summary.csv"
    write_csv(case_path, case_rows)
    write_csv(summary_path, summary_rows)
    manifest = {
        **provenance,
        "case_metric_output_sha256": sha256(case_path),
        "summary_output_sha256": sha256(summary_path),
        "analysis_script_sha256": sha256(Path(__file__)),
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
