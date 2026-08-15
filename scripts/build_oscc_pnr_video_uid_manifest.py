#!/usr/bin/env python3
"""Build an Ego4D CLI UID manifest from the official OSCC/PNR splits.

This script does not create a dataset split or convert annotations for a new
loader. It validates the released split files and extracts their `video_uid`
fields in the whitespace-delimited format accepted by the official Ego4D CLI.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


SPLIT_FILES = {
    "train": "fho_oscc-pnr_train.json",
    "val": "fho_oscc-pnr_val.json",
    "test": "fho_oscc-pnr_test_unannotated.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--manifest-splits",
        nargs="+",
        choices=tuple(SPLIT_FILES),
        default=list(SPLIT_FILES),
        help="Released splits whose video UIDs are written to --output.",
    )
    return parser.parse_args()


def load_released_clips(path: Path, expected_split: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        released = json.load(handle)

    if not isinstance(released, dict) or not isinstance(released.get("clips"), list):
        raise ValueError(f"Unexpected official annotation structure in {path}")

    declared_split = released.get("split")
    valid_declared_splits = {expected_split}
    if expected_split == "test":
        valid_declared_splits.add("test_unannotated")
    if declared_split not in valid_declared_splits:
        raise ValueError(
            f"{path} declares split {declared_split!r}; expected one of "
            f"{sorted(valid_declared_splits)}"
        )

    clips = released["clips"]
    required_fields = {
        "unique_id",
        "video_uid",
        "parent_start_sec",
        "parent_end_sec",
        "parent_start_frame",
        "parent_end_frame",
    }
    for index, clip in enumerate(clips):
        missing = required_fields.difference(clip)
        if missing:
            raise ValueError(f"{path}: clip {index} is missing {sorted(missing)}")
        if expected_split in {"train", "val"} and "state_change" not in clip:
            raise ValueError(f"{path}: labeled clip {index} has no state_change field")

    metadata = {
        "source_file": str(path),
        "released_version": released.get("version"),
        "released_date": released.get("date"),
        "declared_split": declared_split,
    }
    return clips, metadata


def main() -> None:
    args = parse_args()
    all_video_uids: set[str] = set()
    manifest_video_uids: set[str] = set()
    all_unique_ids: set[str] = set()
    report: dict[str, Any] = {"splits": {}}

    for split, filename in SPLIT_FILES.items():
        path = args.annotation_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"Required official split is missing: {path}")

        clips, metadata = load_released_clips(path, split)
        split_unique_ids = [str(clip["unique_id"]) for clip in clips]
        if len(split_unique_ids) != len(set(split_unique_ids)):
            raise ValueError(f"Duplicate unique_id values inside {path}")
        overlap = all_unique_ids.intersection(split_unique_ids)
        if overlap:
            raise ValueError(f"unique_id leakage across released splits: {sorted(overlap)[:5]}")
        all_unique_ids.update(split_unique_ids)

        video_uids = {str(clip["video_uid"]) for clip in clips}
        all_video_uids.update(video_uids)
        if split in args.manifest_splits:
            manifest_video_uids.update(video_uids)
        state_counts = Counter(
            str(clip["state_change"]) for clip in clips if "state_change" in clip
        )
        report["splits"][split] = {
            **metadata,
            "clip_count": len(clips),
            "video_uid_count": len(video_uids),
            "state_change_counts": dict(sorted(state_counts.items())),
        }

    report["total_clip_count"] = len(all_unique_ids)
    report["total_video_uid_count"] = len(all_video_uids)
    report["manifest_splits"] = args.manifest_splits
    report["manifest_video_uid_count"] = len(manifest_video_uids)
    report["video_uid_manifest"] = str(args.output)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(sorted(manifest_video_uids)) + "\n", encoding="utf-8")
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
