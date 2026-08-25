"""Frozen video representation provider interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class RepresentationProvider(ABC):
    """A frozen backbone returning token sequences for each named layer."""

    representation_dimensions: dict[str, int]

    @abstractmethod
    def extract(self, videos: list[Any]) -> dict[str, list[Any]]:
        """Return representation name -> one [tokens, channels] tensor per video."""


def build_provider(
    name: str, model_path: str | Path, model_config: dict[str, Any], device: Any
) -> RepresentationProvider:
    if name == "qwen3":
        from .qwen3 import Qwen3VLProvider

        return Qwen3VLProvider(model_path, model_config, device)
    if name == "vjepa2":
        from .vjepa2 import VJepa2Provider

        return VJepa2Provider(model_path, model_config, device)
    if name == "videomaev2":
        from .videomaev2 import VideoMAEv2Provider

        return VideoMAEv2Provider(model_path, model_config, device)
    raise ValueError(f"unknown provider: {name}")
