"""Extract one timestamp-aware feature shard for each requested condition."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.models.factory import build_loader
from src.temporal_resampling.features import save_shard
from src.temporal_resampling.manifest import read_jsonl
from src.temporal_resampling.protocol import load_canonical_schedules, load_config
from src.temporal_resampling.sampling import build_schedule
from src.temporal_resampling.video import video_duration


def _representation_layer(name: str) -> int | str | None:
    if name.startswith("decoder_layer_") or name.startswith("vision_layer_"):
        return int(name.split("_")[2])
    if name.endswith("_final_tokens") or name.endswith("_final"):
        return "final"
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/temporal_resampling.yaml")
    parser.add_argument("--model", choices=("dinov3", "vjepa2", "qwen3_vl"), required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--canonical-schedules", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prompts-file", help="Exact official prompt per sample_id; required for Qwen3-VL")
    parser.add_argument("--offset-seconds", type=float)
    parser.add_argument("--lower-rate-ratio", type=float)
    parser.add_argument("--conditions", nargs="+", choices=("C0", "C1", "C2", "C3"), default=("C0", "C2"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if "C2" in args.conditions and args.offset_seconds is None:
        raise ValueError("--offset-seconds is required when extracting C2 features")
    _, config = load_config(args.config)
    records = read_jsonl(args.manifest)
    canonical_provenance, canonical = load_canonical_schedules(args.canonical_schedules)
    if canonical_provenance["dataset"] != records[0].dataset or canonical_provenance["model"] != args.model:
        raise ValueError("canonical schedule provenance must match the manifest dataset and requested model")
    prompts = {}
    if args.prompts_file:
        with Path(args.prompts_file).expanduser().resolve().open("r", encoding="utf-8") as handle:
            prompts = json.load(handle)
        if not isinstance(prompts, dict) or any(not isinstance(value, str) or not value.strip() for value in prompts.values()):
            raise ValueError("prompts file must map sample IDs to non-empty fixed prompts")
    if args.model == "qwen3_vl" and any(record.sample_id not in prompts for record in records):
        raise ValueError("Qwen3-VL requires an exact prompt for every sample")
    loader = build_loader(config["models"][args.model], args.device)
    output_dir = Path(args.output_dir).expanduser().resolve()
    for condition in args.conditions:
        outputs = []
        for record in records:
            official = canonical.get(record.sample_id)
            if not isinstance(official, list):
                raise ValueError(f"missing canonical schedule: {record.sample_id}")
            duration = video_duration(record.video_path)
            schedule = build_schedule(condition, official, duration, offset_seconds=args.offset_seconds, lower_rate_ratio=args.lower_rate_ratio)
            encoded = loader.encode_video(record.video_path, list(schedule.timestamps), config["models"][args.model]["family"], prompt=prompts.get(record.sample_id), return_layers=args.model != "dinov3")
            outputs.append((record, schedule, encoded))
        representation_names = set(outputs[0][2].representations)
        if any(set(encoded.representations) != representation_names for _, _, encoded in outputs):
            raise ValueError("model representation keys changed within a sampling-condition shard")
        representations = {name: [encoded.representations[name] for _, _, encoded in outputs] for name in representation_names}
        metadata = []
        for record, schedule, encoded in outputs:
            representation_metadata = {
                name: {
                    "representation_point": name,
                    "layer": _representation_layer(name),
                    "num_visual_tokens": int(value.shape[0]),
                }
                for name, value in encoded.representations.items()
            }
            metadata.append({
                "model": encoded.model,
                "family": encoded.family,
                "video_id": record.sample_id,
                "sampling_condition": condition,
                "sample_id": record.sample_id,
                "requested_timestamps": list(schedule.timestamps),
                "sampled_timestamps": list(encoded.metadata.get("actual_frame_timestamps", ())),
                "actual_frame_timestamps": list(schedule.actual_timestamps or ()),
                "sampled_frame_indices": list(encoded.metadata.get("sampled_frame_indices", ())),
                "num_visual_tokens": {name: values["num_visual_tokens"] for name, values in representation_metadata.items()},
                "representations": representation_metadata,
                **encoded.metadata,
            })
        labels = [record.label_id for record, _, _ in outputs]
        if any(label is None for label in labels):
            raise ValueError("feature extraction requires SSv2 label IDs")
        save_shard(output_dir / f"{args.model}_{condition}.pt", dataset=records[0].dataset, model=args.model, split=records[0].split, condition=condition, canonical_provenance=canonical_provenance, sample_ids=[record.sample_id for record in records], labels=[int(label) for label in labels], representations=representations, metadata=metadata)


if __name__ == "__main__":
    main()
