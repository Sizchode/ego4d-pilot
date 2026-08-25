"""Published TAT frame sampling with exact virtual time reversal."""

from __future__ import annotations

import importlib.util
from collections.abc import Callable, Sequence
from pathlib import Path
from types import ModuleType


class VideoProtocolError(RuntimeError):
    """Raised when the pinned published sampler cannot be used exactly."""


def virtual_reversed_indices(indices: Sequence[int], video_length: int, backward: bool) -> list[int]:
    """Map indices in a virtual reversed video back to the original file.

    If ``R[j] = V[length - 1 - j]``, decoding ``length - 1 - j`` from the
    original file is exactly equivalent to decoding index ``j`` from a
    physically reversed file. The output order is intentionally not sorted.
    """

    if video_length <= 0:
        raise ValueError("video_length must be positive")
    if any(index < 0 or index >= video_length for index in indices):
        raise ValueError("sample index outside video")
    if not backward:
        return list(indices)
    return [video_length - 1 - index for index in indices]


def load_tat_video_utils(tat_root: str | Path) -> ModuleType:
    """Load the published sampler from the pinned TAT-VQA checkout."""

    source = Path(tat_root).expanduser().resolve() / "dataset" / "video_utils.py"
    if not source.is_file():
        raise VideoProtocolError(f"pinned TAT sampler not found: {source}")
    spec = importlib.util.spec_from_file_location("tat_vqa_video_utils", source)
    if spec is None or spec.loader is None:
        raise VideoProtocolError(f"cannot load TAT sampler: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def get_published_sampler(tat_root: str | Path) -> Callable[..., list[int]]:
    module = load_tat_video_utils(tat_root)
    sampler = getattr(module, "get_frame_indices", None)
    if not callable(sampler):
        raise VideoProtocolError("TAT checkout does not expose get_frame_indices")
    return sampler
