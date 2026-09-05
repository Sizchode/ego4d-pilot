"""Simple representation shards with per-sample sampling metadata."""

from __future__ import annotations

from pathlib import Path
from typing import Any


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

    destination = Path(output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
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
    return torch.load(source, map_location="cpu", weights_only=False)
