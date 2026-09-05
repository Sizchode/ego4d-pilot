"""Strict SSv2/NExT-QA manifest records; labels and splits remain official."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .protocol import ProtocolError


@dataclass(frozen=True)
class SampleRecord:
    sample_id: str
    dataset: str
    split: str
    video_path: str
    label_id: int | None = None
    label_text: str | None = None
    question: str | None = None
    choices: tuple[str, ...] = ()
    answer_index: int | None = None
    category: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self, require_video: bool = True) -> None:
        if not self.sample_id or self.dataset not in {"ssv2", "nextqa"}:
            raise ProtocolError("sample_id and supported dataset are required")
        if not Path(self.video_path).is_absolute():
            raise ProtocolError(f"video_path must be absolute: {self.sample_id}")
        if require_video and not Path(self.video_path).is_file():
            raise ProtocolError(f"video does not exist: {self.video_path}")
        if self.dataset == "ssv2":
            if self.split not in {"train", "validation"} or self.label_id is None or not self.label_text:
                raise ProtocolError(f"SSv2 requires official train/validation label fields: {self.sample_id}")
        else:
            if self.split not in {"train", "val"} or len(self.choices) != 5 or self.answer_index not in range(5):
                raise ProtocolError(f"NExT-QA requires official train/val five-choice fields: {self.sample_id}")
            if self.category not in {"temporal", "causal", "descriptive"} or not self.question:
                raise ProtocolError(f"invalid NExT-QA category/question: {self.sample_id}")


def _video(root: Path, source_id: str) -> Path:
    direct = root / source_id
    candidates = [direct] if direct.suffix else [root / f"{source_id}{suffix}" for suffix in (".webm", ".mp4", ".avi", ".mkv", ".mov")]
    found = [candidate.resolve() for candidate in candidates if candidate.is_file()]
    if len(found) != 1:
        raise ProtocolError(f"expected one video for {source_id}; found {found}")
    return found[0]


def _label_map(path: str | Path) -> dict[str, int]:
    with Path(path).expanduser().resolve().open("r", encoding="utf-8") as handle:
        mapping = json.load(handle)
    if not isinstance(mapping, dict) or len(mapping) != 174 or any(not isinstance(value, int) or isinstance(value, bool) for value in mapping.values()) or set(mapping.values()) != set(range(174)):
        raise ProtocolError("official SSv2 label map must contain the contiguous official 174-class ordering")
    return {str(key): value for key, value in mapping.items()}


def build_ssv2(annotations: str | Path, video_root: str | Path, split: str, label_map: str | Path, annotation_revision: str) -> list[SampleRecord]:
    if split not in {"train", "validation"}:
        raise ProtocolError("SSv2 split must be train or validation")
    if not isinstance(annotation_revision, str) or not annotation_revision:
        raise ProtocolError("SSv2 annotation_revision must identify the official source revision")
    with Path(annotations).expanduser().resolve().open("r", encoding="utf-8") as handle:
        rows = json.load(handle)
    if not isinstance(rows, list):
        raise ProtocolError("SSv2 annotations must be a JSON list")
    mapping = _label_map(label_map)
    records = []
    for row in rows:
        if not isinstance(row, dict) or "id" not in row:
            raise ProtocolError("SSv2 row must contain id")
        label = row.get("template", row.get("label"))
        if not isinstance(label, str) or label not in mapping:
            raise ProtocolError(f"SSv2 label is absent from the official label map: {row.get('id')}")
        records.append(SampleRecord(str(row["id"]), "ssv2", split, str(_video(Path(video_root).resolve(), str(row["id"]))), mapping[label], label, metadata={"official": row, "annotation_revision": annotation_revision}))
    validate_records(records)
    return records


def _nextqa_category(value: str) -> str:
    mapping = {"T": "temporal", "C": "causal", "D": "descriptive"}
    if value not in mapping:
        raise ProtocolError(f"unknown official NExT-QA type: {value}")
    return mapping[value]


def build_nextqa(annotations: str | Path, video_root: str | Path, split: str, annotation_revision: str) -> list[SampleRecord]:
    if split not in {"train", "val"}:
        raise ProtocolError("NExT-QA split must be train or val")
    if not isinstance(annotation_revision, str) or not annotation_revision:
        raise ProtocolError("NExT-QA annotation_revision must identify the official source revision")
    with Path(annotations).expanduser().resolve().open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"video", "qid", "question", "answer", "type", *(f"a{index}" for index in range(5))}
    if not rows:
        raise ProtocolError("NExT-QA annotation is empty")
    for row_number, row in enumerate(rows, start=2):
        missing = sorted(required - set(row))
        if missing:
            raise ProtocolError(f"NExT-QA annotation row {row_number} is missing columns: {missing}")
    records = []
    for row in rows:
        try:
            answer_index = int(row["answer"])
        except (TypeError, ValueError) as exc:
            raise ProtocolError(f"invalid NExT-QA answer: {row.get('qid')}") from exc
        if any(not isinstance(row[f"a{index}"], str) or not row[f"a{index}"].strip() for index in range(5)):
            raise ProtocolError(f"NExT-QA choices must be non-empty strings: {row.get('qid')}")
        records.append(SampleRecord(str(row["qid"]), "nextqa", split, str(_video(Path(video_root).resolve(), row["video"])), question=row["question"], choices=tuple(row[f"a{index}"] for index in range(5)), answer_index=answer_index, category=_nextqa_category(row["type"]), metadata={"official": row, "annotation_revision": annotation_revision}))
    validate_records(records)
    return records


def validate_records(records: list[SampleRecord], require_video: bool = True) -> None:
    if not records or len({record.sample_id for record in records}) != len(records):
        raise ProtocolError("manifest must contain non-empty unique sample IDs")
    dataset_splits = {(record.dataset, record.split) for record in records}
    if len(dataset_splits) != 1:
        raise ProtocolError("one manifest must contain exactly one dataset/split")
    for record in records:
        record.validate(require_video=require_video)


def write_jsonl(records: list[SampleRecord], output: str | Path) -> None:
    validate_records(records)
    destination = Path(output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for record in records:
            value = asdict(record)
            value["choices"] = list(record.choices)
            handle.write(json.dumps(value, sort_keys=True) + "\n")


def read_jsonl(path: str | Path, require_video: bool = True) -> list[SampleRecord]:
    records = []
    with Path(path).expanduser().resolve().open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                value["choices"] = tuple(value.get("choices", ()))
                records.append(SampleRecord(**value))
    validate_records(records, require_video=require_video)
    return records
