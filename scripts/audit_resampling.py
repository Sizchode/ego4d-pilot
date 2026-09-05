"""Write C0--C3 real-time schedules for a manifest sample without model inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.temporal_resampling.manifest import read_jsonl
from src.temporal_resampling.sampling import build_schedule, map_to_actual_frames
from src.temporal_resampling.protocol import load_canonical_schedules
from src.temporal_resampling.video import decode_video, video_duration


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--canonical-schedules", required=True, help="JSON object sample_id -> official canonical timestamps")
    parser.add_argument("--offset-seconds", type=float)
    parser.add_argument("--conditions", nargs="+", choices=("C0", "C1", "C2", "C3"), default=("C0", "C2"))
    parser.add_argument("--lower-rate-ratio", type=float)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if "C2" in args.conditions and args.offset_seconds is None:
        raise ValueError("--offset-seconds is required when auditing C2")
    records = read_jsonl(args.manifest)
    provenance, canonical_schedules = load_canonical_schedules(args.canonical_schedules)
    audit = []
    for record in records:
        if provenance["dataset"] != record.dataset:
            raise ValueError(f"canonical schedule dataset does not match manifest: {record.sample_id}")
        _, frame_timestamps = decode_video(record.video_path)
        duration = video_duration(record.video_path)
        canonical = canonical_schedules.get(record.sample_id)
        if not isinstance(canonical, list):
            raise ValueError(f"missing official canonical schedule: {record.sample_id}")
        for condition in args.conditions:
            schedule = build_schedule(condition, canonical, duration, offset_seconds=args.offset_seconds, lower_rate_ratio=args.lower_rate_ratio)
            schedule = map_to_actual_frames(schedule, frame_timestamps)
            audit.append({"sample_id": record.sample_id, "condition": condition, "canonical_provenance": provenance, "requested_timestamps": list(schedule.timestamps), "actual_frame_timestamps": list(schedule.actual_timestamps or ()), "frame_indices": list(schedule.frame_indices or ())})
    destination = Path(args.output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
