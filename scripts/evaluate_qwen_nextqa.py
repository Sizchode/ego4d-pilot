"""Evaluate Qwen3-VL NExT-QA under C0--C3 with one fixed prompt per sample."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.models.factory import build_loader
from src.temporal_resampling.manifest import read_jsonl
from src.temporal_resampling.protocol import load_canonical_schedules, load_config, nextqa_metrics
from src.temporal_resampling.sampling import build_schedule
from src.temporal_resampling.video import video_duration


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/temporal_resampling.yaml")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--canonical-schedules", required=True)
    parser.add_argument("--prompts-file", required=True)
    parser.add_argument("--offset-seconds", type=float)
    parser.add_argument("--lower-rate-ratio", type=float)
    parser.add_argument("--output", required=True)
    parser.add_argument("--conditions", nargs="+", choices=("C0", "C1", "C2", "C3"), default=("C0", "C2"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if "C2" in args.conditions and args.offset_seconds is None:
        raise ValueError("--offset-seconds is required when evaluating C2")
    _, config = load_config(args.config)
    records = read_jsonl(args.manifest)
    canonical_provenance, canonical = load_canonical_schedules(args.canonical_schedules)
    if canonical_provenance["dataset"] != "nextqa" or canonical_provenance["model"] != "qwen3_vl":
        raise ValueError("NExT-QA canonical schedule provenance must identify nextqa and qwen3_vl")
    with Path(args.prompts_file).expanduser().resolve().open("r", encoding="utf-8") as handle:
        prompts = json.load(handle)
    if not isinstance(prompts, dict) or any(not isinstance(value, str) or not value.strip() for value in prompts.values()):
        raise ValueError("prompts file must map sample IDs to non-empty fixed prompts")
    loader = build_loader(config["models"]["qwen3_vl"], args.device)
    predictions = {}
    targets = [int(record.answer_index) for record in records]
    categories = [str(record.category) for record in records]
    for condition in args.conditions:
        values = []
        for record in records:
            if record.sample_id not in canonical or record.sample_id not in prompts:
                raise ValueError(f"missing canonical schedule or official prompt: {record.sample_id}")
            duration = video_duration(record.video_path)
            schedule = build_schedule(condition, canonical[record.sample_id], duration, offset_seconds=args.offset_seconds, lower_rate_ratio=args.lower_rate_ratio)
            values.append(loader.answer_multiple_choice(record.video_path, list(schedule.timestamps), prompts[record.sample_id]))
        predictions[condition] = values
    result = {"canonical_provenance": canonical_provenance, "metrics": {condition: nextqa_metrics(targets, values, categories) for condition, values in predictions.items()}}
    destination = Path(args.output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
