#!/usr/bin/env python3
"""Validate one frozen-backbone feature tensor before deleting its source video."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("feature", type=Path)
    parser.add_argument("--dimension", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.feature.is_file():
        raise FileNotFoundError(args.feature)
    feature = torch.load(args.feature, map_location="cpu")
    if not isinstance(feature, torch.Tensor) or feature.ndim != 2:
        raise TypeError(f"Expected a matrix tensor in {args.feature}, received {type(feature)}")
    if feature.shape[0] < 1 or feature.shape[1] != args.dimension:
        raise ValueError(f"Unexpected feature shape {tuple(feature.shape)} in {args.feature}")
    if not torch.isfinite(feature).all():
        raise ValueError(f"Non-finite values in {args.feature}")
    print(f"validated {args.feature}: shape={tuple(feature.shape)} dtype={feature.dtype}")


if __name__ == "__main__":
    main()
