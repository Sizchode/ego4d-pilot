"""Dataset conversion, published frame sampling, decoding, and loading for Experiment 1."""

from __future__ import annotations

import ast
import csv
import json
import random
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


class ManifestError(ValueError):
    """Raised when released data cannot be represented without ambiguity."""


@dataclass(frozen=True)
class VideoExample:
    example_id: str
    source_id: str
    video_path: str
    label: int
    split: str
    motion_categories: tuple[str, ...] = ()

    def validate(self) -> None:
        if self.label not in (0, 1):
            raise ManifestError(f"label must be 0 (forward) or 1 (backward): {self.example_id}")
        if not Path(self.video_path).is_absolute():
            raise ManifestError(f"video_path must be absolute: {self.example_id}")


def assign_balanced_directions(example_ids: Sequence[str], seed: int) -> dict[str, int]:
    """Apply TAT ``aot.py:assign_balanced_aot`` to sorted video names."""

    if len(set(example_ids)) != len(example_ids):
        raise ManifestError("source example IDs must be unique")
    ordered_ids = sorted(example_ids)
    # Exact TAT implementation: floor(n/2) forward labels, remaining backward,
    # then Python random.shuffle after random.seed(seed).
    ordered_labels = [0] * (len(ordered_ids) // 2) + [1] * (len(ordered_ids) - len(ordered_ids) // 2)
    random.Random(seed).shuffle(ordered_labels)
    label_by_id = dict(zip(ordered_ids, ordered_labels))
    labels = {example_id: label_by_id[example_id] for example_id in example_ids}
    if abs(sum(labels.values()) - (len(labels) - sum(labels.values()))) > 1:
        raise AssertionError("balanced assignment invariant failed")
    return labels


def _resolve_video(video_root: Path, source_id: str) -> Path:
    candidates = [video_root / f"{source_id}{suffix}" for suffix in (".webm", ".mp4", ".avi", ".mkv")]
    existing = [candidate.resolve() for candidate in candidates if candidate.is_file()]
    if len(existing) != 1:
        raise ManifestError(f"expected exactly one video for {source_id}; found {existing}")
    return existing[0]


def build_ssv2_examples(
    annotations: str | Path,
    video_root: str | Path,
    seed: int,
    expected_examples: int | None = None,
) -> list[VideoExample]:
    """Convert the official SSv2 train annotations without changing the split."""

    annotation_path = Path(annotations).expanduser().resolve()
    root = Path(video_root).expanduser().resolve()
    with annotation_path.open("r", encoding="utf-8") as handle:
        rows = json.load(handle)
    if not isinstance(rows, list):
        raise ManifestError("SSv2 annotation file must contain a JSON list")
    source_ids = [str(row["id"]) for row in rows]
    if expected_examples is not None and len(source_ids) != expected_examples:
        raise ManifestError(f"SSv2 train contains {len(source_ids)} rows; expected {expected_examples}")
    labels = assign_balanced_directions(source_ids, seed)
    examples = [
        VideoExample(
            example_id=source_id,
            source_id=source_id,
            video_path=str(_resolve_video(root, source_id)),
            label=labels[source_id],
            split="train",
        )
        for source_id in source_ids
    ]
    return examples


def _parse_categories(value: str) -> tuple[str, ...]:
    parsed = ast.literal_eval(value)
    if not isinstance(parsed, list) or not all(isinstance(item, str) for item in parsed):
        raise ManifestError(f"invalid AoT motion_category value: {value}")
    return tuple(parsed)


def build_aot_examples(
    metadata: str | Path,
    video_root: str | Path,
    expected_source_clips: int | None = None,
) -> list[VideoExample]:
    """Pair every official AoT source clip forward/backward without a new split."""

    metadata_path = Path(metadata).expanduser().resolve()
    root = Path(video_root).expanduser().resolve()
    with metadata_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if expected_source_clips is not None and len(rows) != expected_source_clips:
        raise ManifestError(f"AoT metadata contains {len(rows)} rows; expected {expected_source_clips}")

    source_ids = [Path(row["id"]).stem for row in rows]
    if len(source_ids) != len(set(source_ids)):
        raise ManifestError("AoT source IDs must be unique")

    examples: list[VideoExample] = []
    for row, source_id in zip(rows, source_ids):
        path = _resolve_video(root, source_id)
        categories = _parse_categories(row["motion_category"])
        for label, suffix in ((0, "forward"), (1, "backward")):
            examples.append(
                VideoExample(
                    example_id=f"{source_id}:{suffix}",
                    source_id=source_id,
                    video_path=str(path),
                    label=label,
                    split="test",
                    motion_categories=categories,
                )
            )
    return examples


def write_jsonl(examples: Iterable[VideoExample], output: str | Path) -> None:
    output_path = Path(output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    materialized = list(examples)
    seen: set[str] = set()
    with output_path.open("w", encoding="utf-8") as handle:
        for example in materialized:
            example.validate()
            if example.example_id in seen:
                raise ManifestError(f"duplicate example_id: {example.example_id}")
            seen.add(example.example_id)
            record = asdict(example)
            record["motion_categories"] = list(example.motion_categories)
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def read_jsonl(path: str | Path) -> list[VideoExample]:
    manifest_path = Path(path).expanduser().resolve()
    examples: list[VideoExample] = []
    seen: set[str] = set()
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            record["motion_categories"] = tuple(record.get("motion_categories", ()))
            example = VideoExample(**record)
            example.validate()
            if example.example_id in seen:
                raise ManifestError(f"duplicate example_id at line {line_number}: {example.example_id}")
            if not Path(example.video_path).is_file():
                raise ManifestError(f"missing video at line {line_number}: {example.video_path}")
            seen.add(example.example_id)
            examples.append(example)
    if not examples:
        raise ManifestError(f"manifest is empty: {manifest_path}")
    return examples


def get_frame_indices(num_frames: int, video_length: int, sample: str) -> list[int]:
    """TAT-VQA's released uniform-bin sampler (commit 7f6275b)."""

    if sample not in {"rand", "middle"}:
        raise ValueError("sample must be 'rand' or 'middle'")
    accepted = min(num_frames, video_length)
    intervals = np.linspace(0, video_length, accepted + 1).astype(int)
    ranges = [(start, intervals[index + 1] - 1) for index, start in enumerate(intervals[:-1])]
    if sample == "rand":
        try:
            indices = [random.choice(range(start, end)) for start, end in ranges]
        except IndexError:
            indices = sorted(np.random.permutation(video_length)[:accepted].tolist())
    else:
        indices = [(start + end) // 2 for start, end in ranges]
    if len(indices) < num_frames:
        indices.extend([indices[-1]] * (num_frames - len(indices)))
    return indices


def virtual_reversed_indices(indices: Sequence[int], video_length: int, backward: bool) -> list[int]:
    """Decode the same frames from the virtual time-reversed video."""

    if video_length <= 0 or any(index < 0 or index >= video_length for index in indices):
        raise ValueError("invalid video length or sampled index")
    return [video_length - 1 - index for index in indices] if backward else list(indices)


class ManifestVideoDataset:
    """Decode only the frames required by the published AoT experiment."""

    def __init__(
        self,
        manifest: str | Path,
        frames: int,
        sampling: str,
        apply_tat_spatial_transform: bool = True,
    ):
        self.examples = read_jsonl(manifest)
        self.frames = frames
        self.sampling = sampling
        self.apply_tat_spatial_transform = apply_tat_spatial_transform

    def _spatial_transform(self, video: Any) -> Any:
        import torch
        from torchvision.transforms import InterpolationMode, RandomHorizontalFlip, RandomResizedCrop
        from torchvision.transforms.functional import resize

        frames = torch.from_numpy(video).permute(0, 3, 1, 2)
        if self.sampling == "rand":
            frames = RandomResizedCrop(
                224,
                scale=(0.5, 1.0),
                ratio=(0.75, 4 / 3),
                interpolation=InterpolationMode.BICUBIC,
            )(frames)
            frames = RandomHorizontalFlip(p=0.5)(frames)
        else:
            frames = resize(frames, [224, 224], interpolation=InterpolationMode.BICUBIC, antialias=True)
        return frames.permute(0, 2, 3, 1).numpy()

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        import decord

        example = self.examples[index]
        reader = decord.VideoReader(example.video_path, num_threads=1)
        video_length = len(reader)
        sampled = get_frame_indices(self.frames, video_length, self.sampling)
        decode_indices = virtual_reversed_indices(sampled, video_length, backward=bool(example.label))
        video = reader.get_batch(decode_indices).asnumpy()
        if self.apply_tat_spatial_transform:
            video = self._spatial_transform(video)
        return {
            "example_id": example.example_id,
            "video": video,
            "label": example.label,
            "motion_categories": example.motion_categories,
        }


def video_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    import torch

    return {
        "example_ids": [item["example_id"] for item in batch],
        "videos": [item["video"] for item in batch],
        "labels": torch.tensor([item["label"] for item in batch], dtype=torch.long),
        "motion_categories": [item["motion_categories"] for item in batch],
    }
