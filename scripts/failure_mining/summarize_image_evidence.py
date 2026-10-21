#!/usr/bin/env python3
"""Extract threshold-free image statistics for reconstruction failure analysis.

The script consumes a runner ``input_manifest.json`` and writes one row per
input image.  It records descriptive intensity and gradient distributions only;
it does not assign quality labels or choose reconstruction actions.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


QUANTILES = (0.0, 0.01, 0.5, 0.9, 0.95, 0.99, 1.0)
LUMA_WEIGHTS = np.asarray((0.2126, 0.7152, 0.0722), dtype=np.float64)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def distribution(prefix: str, values: np.ndarray) -> dict[str, float | int]:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = flat[np.isfinite(flat)]
    output: dict[str, float | int] = {
        f"{prefix}.count": int(flat.size),
        f"{prefix}.finite_count": int(finite.size),
    }
    if finite.size:
        output[f"{prefix}.mean"] = float(np.mean(finite))
        output[f"{prefix}.std"] = float(np.std(finite))
        for quantile, value in zip(QUANTILES, np.quantile(finite, QUANTILES), strict=True):
            output[f"{prefix}.p{int(quantile * 100):02d}"] = float(value)
    return output


def entropy_bits(luminance: np.ndarray) -> float:
    bins = np.clip(np.rint(luminance * 255.0), 0, 255).astype(np.uint8)
    counts = np.bincount(bins.reshape(-1), minlength=256).astype(np.float64)
    probabilities = counts[counts > 0] / counts.sum()
    return float(-np.sum(probabilities * np.log2(probabilities)))


def resolve_image(entry: dict[str, Any], image_root: Path | None) -> Path:
    source = Path(entry["path"])
    return image_root / source.name if image_root else source


def summarize(entry: dict[str, Any], image_root: Path | None) -> dict[str, Any]:
    path = resolve_image(entry, image_root)
    if not path.is_file():
        raise FileNotFoundError(path)
    observed_hash = sha256(path)
    expected_hash = entry.get("sha256")
    if expected_hash and observed_hash != expected_hash:
        raise ValueError(f"Image hash mismatch for {path}")

    with Image.open(path) as image:
        rgb_u8 = np.asarray(image.convert("RGB"), dtype=np.uint8)
    height, width = rgb_u8.shape[:2]
    expected_hw = entry.get("original_hw")
    if expected_hw and [height, width] != list(expected_hw):
        raise ValueError(f"Image shape mismatch for {path}: {(height, width)} != {expected_hw}")

    rgb = rgb_u8.astype(np.float64) / 255.0
    luminance = rgb @ LUMA_WEIGHTS
    gradient_y, gradient_x = np.gradient(luminance)
    gradient = np.hypot(gradient_x, gradient_y)

    row: dict[str, Any] = {
        "index": int(entry["index"]),
        "source_path": entry["path"],
        "resolved_path": str(path.resolve()),
        "sha256": observed_hash,
        "height": height,
        "width": width,
        "luminance.entropy_bits": entropy_bits(luminance),
        "rgb.any_channel_at_floor_ratio": float(np.mean(np.any(rgb_u8 == 0, axis=-1))),
        "rgb.any_channel_at_ceiling_ratio": float(np.mean(np.any(rgb_u8 == 255, axis=-1))),
        "rgb.all_channels_at_floor_ratio": float(np.mean(np.all(rgb_u8 == 0, axis=-1))),
        "rgb.all_channels_at_ceiling_ratio": float(np.mean(np.all(rgb_u8 == 255, axis=-1))),
    }
    row.update(distribution("luminance", luminance))
    row.update(distribution("gradient", gradient))
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_manifest", type=Path)
    parser.add_argument("output_csv", type=Path)
    parser.add_argument(
        "--image-root",
        type=Path,
        help="Resolve each manifest entry by basename under this directory",
    )
    args = parser.parse_args()

    entries = json.loads(args.input_manifest.read_text(encoding="utf-8"))
    if not isinstance(entries, list) or not entries:
        raise ValueError("Input manifest must be a non-empty JSON list")
    rows = [summarize(entry, args.image_root) for entry in entries]
    indices = [row["index"] for row in rows]
    if len(indices) != len(set(indices)):
        raise ValueError("Input manifest contains duplicate indices")

    fields = sorted({key for row in rows for key in row})
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"images": len(rows), "output": str(args.output_csv)}, sort_keys=True))


if __name__ == "__main__":
    main()
