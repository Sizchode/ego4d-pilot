"""Render C0--C3 contact sheets for the mandatory visual sampling audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw

from src.temporal_resampling.manifest import read_jsonl
from src.temporal_resampling.protocol import load_canonical_schedules
from src.temporal_resampling.sampling import build_schedule
from src.temporal_resampling.video import sample_video, video_duration


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--canonical-schedules", required=True)
    parser.add_argument("--offset-seconds", type=float)
    parser.add_argument("--conditions", nargs="+", choices=("C0", "C1", "C2", "C3"), default=("C0", "C2"))
    parser.add_argument("--lower-rate-ratio", type=float)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-samples", type=int, default=20)
    args = parser.parse_args()
    if "C2" in args.conditions and args.offset_seconds is None:
        raise ValueError("--offset-seconds is required when rendering C2")
    records = read_jsonl(args.manifest)[: args.max_samples]
    provenance, canonical = load_canonical_schedules(args.canonical_schedules)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for record in records:
        if provenance["dataset"] != record.dataset:
            raise ValueError(f"canonical schedule dataset does not match manifest: {record.sample_id}")
        duration = video_duration(record.video_path)
        for condition in args.conditions:
            schedule = build_schedule(condition, canonical[record.sample_id], duration, offset_seconds=args.offset_seconds, lower_rate_ratio=args.lower_rate_ratio)
            frames, actual = sample_video(record.video_path, schedule)
            tiles = []
            for index, frame in enumerate(frames):
                image = Image.fromarray(frame).convert("RGB")
                image.thumbnail((256, 256))
                tile = Image.new("RGB", (256, 288), "white")
                tile.paste(image, ((256 - image.width) // 2, 0))
                ImageDraw.Draw(tile).text((4, 264), f"{index}: {actual.actual_timestamps[index]:.3f}s", fill="black")
                tiles.append(tile)
            sheet = Image.new("RGB", (256 * len(tiles), 288), "white")
            for index, tile in enumerate(tiles):
                sheet.paste(tile, (256 * index, 0))
            sheet.save(output_dir / f"{record.sample_id}_{condition}.jpg", quality=95)


if __name__ == "__main__":
    main()
