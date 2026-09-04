"""Strict manifest builders for the four datasets in the final plan."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .protocol import NEXTQA_CATEGORIES, ProtocolError


VIDEO_SUFFIXES = (".mp4", ".webm", ".avi", ".mkv", ".mov")
DATASET_SPLITS = {"ssv2": {"train", "validation"}, "nextqa": {"train", "val", "test"}}


@dataclass(frozen=True)
class ManifestRecord:
    example_id: str
    dataset: str
    split: str
    video_path: str
    label: str | None = None
    label_id: int | None = None
    question: str | None = None
    choices: tuple[str, ...] = ()
    answer_index: int | None = None
    category: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self, require_video: bool = True) -> None:
        if not self.example_id or not self.dataset or not self.split:
            raise ProtocolError("manifest records require example_id, dataset, and split")
        if self.dataset in DATASET_SPLITS and self.split not in DATASET_SPLITS[self.dataset]:
            raise ProtocolError(f"{self.dataset} split is not official: {self.split}")
        path = Path(self.video_path)
        if not path.is_absolute():
            raise ProtocolError(f"video_path must be absolute: {self.example_id}")
        if require_video and not path.is_file():
            raise ProtocolError(f"video file does not exist: {self.video_path}")
        if self.label_id is not None and self.label_id < 0:
            raise ProtocolError(f"label_id must be non-negative: {self.example_id}")
        if self.dataset == "ssv2":
            if self.label is None:
                raise ProtocolError(f"SSv2 record is missing the official action label: {self.example_id}")
            if self.answer_index is not None or self.category is not None:
                raise ProtocolError(f"SSv2 record contains NExT-QA fields: {self.example_id}")
        if self.dataset == "nextqa":
            if len(self.choices) != 5 or self.answer_index not in range(5):
                raise ProtocolError(f"NExT-QA requires exactly five choices and an answer index: {self.example_id}")
            if self.category not in NEXTQA_CATEGORIES:
                raise ProtocolError(f"invalid NExT-QA category: {self.category}")
            if not self.question:
                raise ProtocolError(f"NExT-QA question is empty: {self.example_id}")

    def to_json(self) -> dict[str, Any]:
        record = asdict(self)
        record["choices"] = list(self.choices)
        return record


def _resolve_video(video_root: Path, source_id: str) -> Path:
    direct = video_root / source_id
    candidates = [direct] if direct.suffix else [video_root / f"{source_id}{suffix}" for suffix in VIDEO_SUFFIXES]
    found = [candidate.resolve() for candidate in candidates if candidate.is_file()]
    if len(found) != 1:
        raise ProtocolError(f"expected exactly one video for {source_id}; found {found}")
    return found[0]


def _read_label_map(path: str | Path | None) -> dict[str, int] | None:
    if path is None:
        return None
    with Path(path).expanduser().resolve().open("r", encoding="utf-8") as handle:
        mapping = json.load(handle)
    if not isinstance(mapping, dict) or not mapping or any(not isinstance(value, int) for value in mapping.values()):
        raise ProtocolError("label map must be a non-empty object mapping official labels to integer IDs")
    return {str(key): value for key, value in mapping.items()}


def build_ssv2_manifest(
    annotations: str | Path,
    video_root: str | Path,
    split: str,
    label_map: str | Path | None = None,
) -> list[ManifestRecord]:
    """Preserve official SSv2 action strings; optional IDs come from an official map."""

    with Path(annotations).expanduser().resolve().open("r", encoding="utf-8") as handle:
        rows = json.load(handle)
    if not isinstance(rows, list):
        raise ProtocolError("SSv2 annotations must be a JSON list")
    mapping = _read_label_map(label_map)
    if mapping is None:
        raise ProtocolError("SSv2 manifest construction requires the official label map")
    root = Path(video_root).expanduser().resolve()
    records: list[ManifestRecord] = []
    for row in rows:
        if not isinstance(row, dict) or "id" not in row:
            raise ProtocolError("each SSv2 annotation must contain id")
        label = row.get("label", row.get("template"))
        if not isinstance(label, (str, int)):
            raise ProtocolError(f"SSv2 annotation has no official label: {row.get('id')}")
        label_text = str(label)
        label_id = mapping.get(label_text) if mapping is not None else (label if isinstance(label, int) else None)
        records.append(
            ManifestRecord(
                example_id=str(row["id"]),
                dataset="ssv2",
                split=split,
                video_path=str(_resolve_video(root, str(row["id"]))),
                label=label_text,
                label_id=label_id,
                metadata={"official_annotation": row},
            )
        )
    _validate_unique(records)
    return records


def _nextqa_category(value: str) -> str:
    categories = {"T": "temporal", "C": "causal", "D": "descriptive"}
    if value not in categories:
        raise ProtocolError(f"NExT-QA type must be one of {sorted(categories)}; got {value!r}")
    return categories[value]


def build_nextqa_manifest(annotations: str | Path, video_root: str | Path, split: str) -> list[ManifestRecord]:
    """Read the official NExT-QA multi-choice CSV without changing its labels or split."""

    with Path(annotations).expanduser().resolve().open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"video", "question", "answer", "type", "qid", *(f"a{index}" for index in range(5))}
    if not rows or not required.issubset(rows[0]):
        raise ProtocolError(f"NExT-QA annotation is missing columns: {sorted(required - set(rows[0]) if rows else required)}")
    root = Path(video_root).expanduser().resolve()
    records: list[ManifestRecord] = []
    for row in rows:
        try:
            answer_index = int(row["answer"])
        except (KeyError, ValueError) as exc:
            raise ProtocolError(f"invalid NExT-QA answer for {row.get('qid')}") from exc
        records.append(
            ManifestRecord(
                example_id=str(row["qid"]),
                dataset="nextqa",
                split=split,
                video_path=str(_resolve_video(root, str(row["video"]))),
                question=row["question"],
                choices=tuple(row[f"a{index}"] for index in range(5)),
                answer_index=answer_index,
                category=_nextqa_category(row["type"]),
                metadata={"official_annotation": row},
            )
        )
    _validate_unique(records)
    return records


def _validate_unique(records: list[ManifestRecord]) -> None:
    ids = [record.example_id for record in records]
    if len(ids) != len(set(ids)):
        raise ProtocolError("manifest example IDs must be unique")
    for record in records:
        record.validate()


def write_jsonl(records: list[ManifestRecord], output: str | Path) -> None:
    _validate_unique(records)
    destination = Path(output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_json(), sort_keys=True) + "\n")


def read_jsonl(path: str | Path, require_video: bool = True) -> list[ManifestRecord]:
    source = Path(path).expanduser().resolve()
    records: list[ManifestRecord] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            value["choices"] = tuple(value.get("choices", ()))
            record = ManifestRecord(**value)
            record.validate(require_video=require_video)
            records.append(record)
    _validate_unique(records)
    if not records:
        raise ProtocolError(f"manifest is empty: {source}")
    dataset_split = {(record.dataset, record.split) for record in records}
    if len(dataset_split) != 1:
        raise ProtocolError(f"manifest must contain one dataset and split: {source}")
    return records
