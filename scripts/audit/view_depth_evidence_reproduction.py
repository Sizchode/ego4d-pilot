"""Audit an independent reproduction of view-level depth evidence analysis."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

from scripts.failure_mining.compare_baselines import sha256


IDENTITY_FIELDS = ("dataset", "scene", "target_field", "evidence_field")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_csv(path: Path) -> list[dict[str, str]]:
    require(path.is_file(), f"Required file is missing: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def audit_reproduction(stored: Path, reproduced: Path) -> dict[str, Any]:
    filenames = {
        "merged": "merged_view_evidence.csv",
        "scene": "scene_view_rank_correlations.csv",
        "aggregate": "aggregate_view_rank_correlations.csv",
    }
    paths = {
        label: (stored / filename, reproduced / filename)
        for label, filename in filenames.items()
    }
    for pair in paths.values():
        require(all(path.is_file() for path in pair), f"Required output is missing: {pair}")

    require(paths["merged"][0].read_bytes() == paths["merged"][1].read_bytes(), "Merged view table is not byte-identical")
    require(paths["aggregate"][0].read_bytes() == paths["aggregate"][1].read_bytes(), "Aggregate table is not byte-identical")
    stored_rows, reproduced_rows = map(read_csv, paths["scene"])
    require(len(stored_rows) == len(reproduced_rows), "Scene-correlation row count differs")
    maximum_p_difference = 0.0
    for stored_row, reproduced_row in zip(stored_rows, reproduced_rows, strict=True):
        identity = tuple(stored_row[field] for field in IDENTITY_FIELDS)
        require(
            identity == tuple(reproduced_row[field] for field in IDENTITY_FIELDS),
            f"Scene-correlation identity/order differs: {identity}",
        )
        require(stored_row["views"] == reproduced_row["views"], f"View count differs: {identity}")
        require(
            float(stored_row["spearman_rho"]) == float(reproduced_row["spearman_rho"]),
            f"Spearman rho differs: {identity}",
        )
        stored_p = float(stored_row["p_value_two_sided_descriptive"])
        reproduced_p = float(reproduced_row["p_value_two_sided_descriptive"])
        require(math.isfinite(stored_p) and math.isfinite(reproduced_p), f"Non-finite descriptive p-value: {identity}")
        difference = abs(stored_p - reproduced_p)
        maximum_p_difference = max(maximum_p_difference, difference)
        require(
            math.isclose(stored_p, reproduced_p, rel_tol=1e-12, abs_tol=1e-15),
            f"Descriptive p-value exceeds floating-point tolerance: {identity}",
        )

    return {
        "schema_version": 1,
        "passed": True,
        "merged_view_table_byte_identical": True,
        "aggregate_table_byte_identical": True,
        "scene_rows": len(stored_rows),
        "scene_identity_views_and_rho_exact": True,
        "scene_descriptive_p_value_tolerance": {"relative": 1e-12, "absolute": 1e-15},
        "scene_descriptive_p_value_maximum_absolute_difference": maximum_p_difference,
        "stored": {label: sha256(pair[0]) for label, pair in paths.items()},
        "reproduced": {label: sha256(pair[1]) for label, pair in paths.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stored", type=Path, required=True)
    parser.add_argument("--reproduced", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), f"Refusing to overwrite output: {args.output}")
    report = audit_reproduction(args.stored, args.reproduced)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
