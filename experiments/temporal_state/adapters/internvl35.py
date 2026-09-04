"""Explicit boundary for InternVL3.5's official representation integration."""

from __future__ import annotations

from typing import Any

from .base import AdapterError, RepresentationAdapter


class InternVL35Adapter(RepresentationAdapter):
    def __init__(self, config: dict[str, Any], device: Any):
        raise AdapterError(
            "InternVL3.5 adapter is intentionally blocked until the official HF video processor, chat template, and decoder hook paths are pinned in OPEN_QUESTIONS.md"
        )

    @property
    def representation_dimensions(self) -> dict[str, int]:
        raise AdapterError("InternVL3.5 adapter is not configured")

    @property
    def total_layers(self) -> int:
        raise AdapterError("InternVL3.5 adapter is not configured")

    def extract_compact_state(self, video_paths: list[str], prompts: list[str]) -> dict[str, Any]:
        raise AdapterError("InternVL3.5 adapter is not configured")

    def extract_distributed_state(self, video_paths: list[str], prompts: list[str]) -> dict[str, list[Any]]:
        raise AdapterError("InternVL3.5 adapter is not configured")
