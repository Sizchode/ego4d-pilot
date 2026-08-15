#!/usr/bin/env python3
"""Verify the released EgoPack inputs required by Ego4D OSCC/PNR."""

import argparse
import csv
import json
from pathlib import Path

import torch


EXPECTED_SPLIT_SIZES = {"train": 41_085, "val": 28_348}
EXPECTED_FEATURE_SIZE = 1_536


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation-dir", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with (args.feature_dir / "manifest.csv").open(newline="") as handle:
        manifest_uids = {
            row["video_uid"] for row in csv.DictReader(handle) if row["type"] == "file"
        }

    released_files = {path.stem for path in args.feature_dir.glob("*.pt")}
    missing_release = sorted(manifest_uids - released_files)
    unexpected_release = sorted(released_files - manifest_uids)
    if missing_release or unexpected_release:
        raise RuntimeError(
            f"Feature release mismatch: missing={len(missing_release)}, "
            f"unexpected={len(unexpected_release)}"
        )

    task_uids: set[str] = set()
    split_summary = {}
    for split, expected_size in EXPECTED_SPLIT_SIZES.items():
        annotation_path = args.annotation_dir / f"fho_oscc-pnr_{split}.json"
        with annotation_path.open() as handle:
            clips = json.load(handle)["clips"]
        if len(clips) != expected_size:
            raise RuntimeError(
                f"Unexpected {split} split size: {len(clips)} != {expected_size}"
            )
        split_uids = {clip["video_uid"] for clip in clips}
        task_uids.update(split_uids)
        split_summary[split] = {"clips": len(clips), "video_uids": len(split_uids)}

    missing_task_features = sorted(task_uids - released_files)
    if missing_task_features:
        raise RuntimeError(
            f"Missing features for {len(missing_task_features)} OSCC/PNR videos"
        )

    for index, video_uid in enumerate(sorted(task_uids), start=1):
        feature = torch.load(
            args.feature_dir / f"{video_uid}.pt", map_location="cpu", weights_only=True
        )
        if feature.ndim != 2 or feature.shape[1] != EXPECTED_FEATURE_SIZE:
            raise RuntimeError(
                f"Unexpected feature shape for {video_uid}: {tuple(feature.shape)}"
            )
        if index % 100 == 0:
            print(f"Validated {index}/{len(task_uids)} task feature tensors", flush=True)

    print(
        json.dumps(
            {
                "release_feature_files": len(released_files),
                "task_feature_files_validated": len(task_uids),
                "feature_size": EXPECTED_FEATURE_SIZE,
                "splits": split_summary,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
