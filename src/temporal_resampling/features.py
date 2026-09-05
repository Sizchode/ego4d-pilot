"""Versioned representation shard format with per-sample sampling metadata."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .protocol import ProtocolError


FEATURE_SCHEMA_VERSION = 1


def save_shard(
    output: str | Path,
    *,
    dataset: str,
    model: str,
    split: str,
    condition: str,
    canonical_provenance: dict[str, str],
    sample_ids: list[str],
    labels: list[int],
    representations: dict[str, Any],
    metadata: list[dict[str, Any]],
) -> None:
    import torch

    if not sample_ids or len(sample_ids) != len(labels) or len(sample_ids) != len(metadata):
        raise ProtocolError("feature shard sample IDs, labels, and metadata must align")
    if not representations or any(not isinstance(name, str) or not name for name in representations):
        raise ProtocolError("feature shard must contain named representations")
    if any(not hasattr(value, "__len__") or len(value) != len(sample_ids) for value in representations.values()):
        raise ProtocolError("feature shard representations must align with sample IDs")
    if not isinstance(canonical_provenance, dict) or any(
        not isinstance(canonical_provenance.get(key), str) or not canonical_provenance[key]
        for key in ("source", "revision", "dataset", "model")
    ):
        raise ProtocolError("feature shard canonical provenance is incomplete")
    destination = Path(output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": FEATURE_SCHEMA_VERSION,
            "dataset": dataset,
            "model": model,
            "split": split,
            "condition": condition,
            "canonical_provenance": canonical_provenance,
            "sample_ids": sample_ids,
            "labels": torch.tensor(labels, dtype=torch.long),
            "representations": representations,
            "metadata": metadata,
        },
        destination,
    )


def load_shard(path: str | Path) -> dict[str, Any]:
    import torch

    source = Path(path).expanduser().resolve()
    payload = torch.load(source, map_location="cpu", weights_only=False)
    required = {"schema_version", "dataset", "model", "split", "condition", "canonical_provenance", "sample_ids", "labels", "representations", "metadata"}
    if not isinstance(payload, dict) or not required.issubset(payload):
        raise ProtocolError(f"invalid feature shard: {source}")
    if payload["schema_version"] != FEATURE_SCHEMA_VERSION:
        raise ProtocolError(f"unsupported feature shard version: {payload['schema_version']}")
    provenance = payload["canonical_provenance"]
    if not isinstance(provenance, dict) or any(
        not isinstance(provenance.get(key), str) or not provenance[key]
        for key in ("source", "revision", "dataset", "model")
    ):
        raise ProtocolError(f"feature shard is missing canonical schedule provenance: {source}")
    if len(payload["sample_ids"]) != len(payload["labels"]) or len(payload["sample_ids"]) != len(payload["metadata"]):
        raise ProtocolError(f"feature shard fields are misaligned: {source}")
    if not isinstance(payload["representations"], dict) or not payload["representations"] or any(
        not hasattr(value, "__len__") or len(value) != len(payload["sample_ids"])
        for value in payload["representations"].values()
    ):
        raise ProtocolError(f"feature shard representations are misaligned: {source}")
    return payload
