"""Common loader seam; model-specific preprocessing remains inside each loader."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


class LoaderError(RuntimeError):
    """Raised when a model cannot be loaded or its official API is ambiguous."""


@dataclass(frozen=True)
class EncodedVideo:
    model: str
    family: str
    representations: dict[str, Any]
    metadata: dict[str, Any]


class VideoLoader(ABC):
    @abstractmethod
    def encode_video(
        self,
        video_path: str,
        timestamps: list[float],
        model_family: str,
        *,
        prompt: str | None = None,
        return_layers: bool = False,
    ) -> EncodedVideo:
        raise NotImplementedError
