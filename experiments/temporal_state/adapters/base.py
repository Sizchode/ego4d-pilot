"""Public seam implemented by every model-specific representation adapter."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class AdapterError(RuntimeError):
    """Raised when an official model API or representation contract is unavailable."""


class RepresentationAdapter(ABC):
    """No adapter may pool, concatenate layers, or synthesize V-JEPA compact state."""

    @property
    @abstractmethod
    def representation_dimensions(self) -> dict[str, int]:
        raise NotImplementedError

    @property
    @abstractmethod
    def total_layers(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def extract_compact_state(self, video_paths: list[str], prompts: list[str]) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def extract_distributed_state(self, video_paths: list[str], prompts: list[str]) -> dict[str, list[Any]]:
        raise NotImplementedError
