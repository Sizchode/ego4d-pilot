#!/usr/bin/env python3
"""Summarize archived reconstruction case bundles without assigning labels.

Each case directory must contain the files archived by the baseline runners:
``run_config.json``, ``status.json``, ``validation.json``, and
``evaluation_capture/camera_pairs.npz``.  The output preserves the official
evaluation metrics and adds only descriptive camera-error quantiles.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


CAMERA_FIELDS = ("rotation_degrees", "translation_degrees")
QUANTILES = (0.0, 0.5, 0.9, 0.95, 1.0)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return value


def _flatten(prefix: str, value: Any, output: dict[str, Any]) -> None:
    if isinstance(value, dict):
        for key in sorted(value):
            child = f"{prefix}.{key}" if prefix else str(key)
            _flatten(child, value[key], output)
        return
    output[prefix] = value


def _camera_quantiles(values: np.ndarray) -> dict[str, float | int]:
    finite = values[np.isfinite(values)]
    output: dict[str, float | int] = {"finite_count": int(finite.size)}
    if finite.size:
        quantiles = np.quantile(finite, QUANTILES)
        for quantile, value in zip(QUANTILES, quantiles, strict=True):
            output[f"p{int(quantile * 100):02d}"] = float(value)
    return output


def summarize_case(case_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    required = {
        "run_config": case_dir / "run_config.json",
        "status": case_dir / "status.json",
        "validation": case_dir / "validation.json",
        "camera_pairs": case_dir / "evaluation_capture" / "camera_pairs.npz",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete case bundle {case_dir}: {missing}")

    run_config = _load_json(required["run_config"])
    status = _load_json(required["status"])
    validation = _load_json(required["validation"])

    config = run_config.get("config", {})
    row: dict[str, Any] = {
        "case_dir": str(case_dir.resolve()),
        "model": run_config.get("model") or config.get("model_repository"),
        "dataset": run_config.get("dataset", status.get("dataset")),
        "scene": run_config.get("scene", status.get("scene")),
        "state": status.get("state"),
        "views": validation.get("views"),
        "validation_passed": validation.get("passed"),
        "code_revision": config.get("code_revision"),
        "model_revision": config.get("model_revision"),
    }
    _flatten("metric", validation.get("metrics", {}), row)

    view_rows: list[dict[str, Any]] = []
    with np.load(required["camera_pairs"], allow_pickle=False) as pairs:
        lengths = {field: int(np.asarray(pairs[field]).size) for field in CAMERA_FIELDS}
        if len(set(lengths.values())) != 1:
            raise ValueError(f"Camera fields have unequal lengths in {case_dir}: {lengths}")
        index_a = np.asarray(pairs["index_a"], dtype=np.int64).reshape(-1)
        index_b = np.asarray(pairs["index_b"], dtype=np.int64).reshape(-1)
        if index_a.size != next(iter(lengths.values())) or index_b.size != index_a.size:
            raise ValueError(f"Camera indices have unequal lengths in {case_dir}")
        row["camera.pair_count"] = next(iter(lengths.values()))
        for field in CAMERA_FIELDS:
            values = np.asarray(pairs[field], dtype=np.float64).reshape(-1)
            for name, value in _camera_quantiles(values).items():
                row[f"camera.{field}.{name}"] = value

        view_indices = sorted(set(index_a.tolist()) | set(index_b.tolist()))
        for view_index in view_indices:
            incident = (index_a == view_index) | (index_b == view_index)
            view_row: dict[str, Any] = {
                "case_dir": str(case_dir.resolve()),
                "model": row["model"],
                "dataset": row["dataset"],
                "scene": row["scene"],
                "view_index": view_index,
                "incident_pair_count": int(np.count_nonzero(incident)),
            }
            for field in CAMERA_FIELDS:
                values = np.asarray(pairs[field], dtype=np.float64).reshape(-1)[incident]
                for name, value in _camera_quantiles(values).items():
                    view_row[f"camera.{field}.{name}"] = value
            view_rows.append(view_row)
    return row, view_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("case_root", type=Path)
    parser.add_argument("output_csv", type=Path)
    parser.add_argument(
        "--view-output",
        type=Path,
        help="Optional CSV containing descriptive camera-error statistics per input view",
    )
    args = parser.parse_args()

    case_dirs = sorted(
        path
        for path in args.case_root.iterdir()
        if path.is_dir() and (path / "run_config.json").is_file()
    )
    summaries = [summarize_case(path) for path in case_dirs]
    rows = [summary for summary, _view_rows in summaries]
    if not rows:
        raise ValueError(f"No case directories found under {args.case_root}")

    fields = sorted({key for row in rows for key in row})
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    view_row_count = 0
    if args.view_output:
        view_rows = [row for _summary, case_rows in summaries for row in case_rows]
        view_fields = sorted({key for row in view_rows for key in row})
        args.view_output.parent.mkdir(parents=True, exist_ok=True)
        with args.view_output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=view_fields)
            writer.writeheader()
            writer.writerows(view_rows)
        view_row_count = len(view_rows)

    print(
        json.dumps(
            {
                "cases": len(rows),
                "output": str(args.output_csv),
                "view_output": str(args.view_output) if args.view_output else None,
                "view_rows": view_row_count,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
