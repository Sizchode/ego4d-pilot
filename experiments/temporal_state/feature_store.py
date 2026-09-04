"""Versioned feature-file contract between HPC extraction and probe jobs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .protocol import ProtocolError


FEATURE_SCHEMA_VERSION = 1


def save_feature_file(
    output: str | Path,
    *,
    dataset: str,
    model: str,
    split: str,
    total_layers: int,
    example_ids: list[str],
    labels: list[int],
    categories: list[str] | None,
    compact: dict[str, Any],
    distributed: dict[str, list[Any]],
) -> None:
    import torch

    if not example_ids or len(example_ids) != len(labels):
        raise ProtocolError("feature file requires non-empty example IDs and equally sized labels")
    if categories is not None and len(categories) != len(example_ids):
        raise ProtocolError("feature categories must match example IDs")
    destination = Path(output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": FEATURE_SCHEMA_VERSION,
            "dataset": dataset,
            "model": model,
            "split": split,
            "total_layers": total_layers,
            "example_ids": example_ids,
            "labels": torch.tensor(labels, dtype=torch.long),
            "categories": categories,
            "compact": compact,
            "distributed": distributed,
        },
        destination,
    )


def load_feature_file(path: str | Path) -> dict[str, Any]:
    import torch

    source = Path(path).expanduser().resolve()
    payload = torch.load(source, map_location="cpu", weights_only=False)
    required = {"schema_version", "dataset", "model", "split", "total_layers", "example_ids", "labels", "categories", "compact", "distributed"}
    if not isinstance(payload, dict) or not required.issubset(payload):
        raise ProtocolError(f"invalid feature file schema: {source}")
    if payload["schema_version"] != FEATURE_SCHEMA_VERSION:
        raise ProtocolError(f"unsupported feature schema: {payload['schema_version']}")
    if not isinstance(payload["total_layers"], int) or payload["total_layers"] <= 0:
        raise ProtocolError(f"feature file has invalid total_layers: {source}")
    if len(payload["example_ids"]) != len(payload["labels"]):
        raise ProtocolError(f"feature labels do not align with example IDs: {source}")
    return payload
