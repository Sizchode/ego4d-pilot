#!/usr/bin/env python3
"""Build an Ego4D exported-clip manifest from the released OSCC/PNR splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SPLIT_FILES = {
    "train": "fho_oscc-pnr_train.json",
    "val": "fho_oscc-pnr_val.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def load_released_clips(path: Path, expected_split: str) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        released = json.load(handle)

    if released.get("split") != expected_split or not isinstance(released.get("clips"), list):
        raise ValueError(f"Unexpected official split structure in {path}")

    required_fields = {"unique_id", "video_uid", "clip_uid"}
    for index, clip in enumerate(released["clips"]):
        missing = required_fields.difference(clip)
        if missing:
            raise ValueError(f"{path}: clip {index} is missing {sorted(missing)}")
    return released["clips"]


def main() -> None:
    args = parse_args()
    clip_uids: set[str] = set()
    sample_uids: set[str] = set()
    report: dict[str, Any] = {"splits": {}}

    for split, filename in SPLIT_FILES.items():
        path = args.annotation_dir / filename
        clips = load_released_clips(path, split)
        split_sample_uids = {str(clip["unique_id"]) for clip in clips}
        overlap = sample_uids.intersection(split_sample_uids)
        if overlap:
            raise ValueError(f"unique_id leakage across released splits: {sorted(overlap)[:5]}")
        sample_uids.update(split_sample_uids)

        split_clip_uids = {
            str(clip["clip_uid"])
            for clip in clips
            if isinstance(clip.get("clip_uid"), str) and clip["clip_uid"]
        }
        samples_without_clip_uid = sum(
            not isinstance(clip.get("clip_uid"), str) or not clip["clip_uid"]
            for clip in clips
        )
        clip_uids.update(split_clip_uids)
        report["splits"][split] = {
            "source_file": str(path),
            "sample_count": len(clips),
            "clip_uid_count": len(split_clip_uids),
            "samples_without_clip_uid": samples_without_clip_uid,
        }

    report["total_sample_count"] = len(sample_uids)
    report["total_clip_uid_count"] = len(clip_uids)
    report["clip_uid_manifest"] = str(args.output)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(sorted(clip_uids)) + "\n", encoding="utf-8")
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
