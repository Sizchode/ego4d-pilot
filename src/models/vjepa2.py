"""V-JEPA 2 loader boundary for official frozen-probe preprocessing."""

from __future__ import annotations

from typing import Any

from .base import EncodedVideo, LoaderError, VideoLoader


class VJEPA2Loader(VideoLoader):
    def __init__(self, config: dict[str, Any], device: Any):
        repo_id = config.get("repo_id")
        revision = config.get("revision")
        if not isinstance(repo_id, str) or not repo_id or not isinstance(revision, str) or not revision:
            raise LoaderError("V-JEPA 2 repo_id and immutable revision must be recorded before execution")
        raise LoaderError(
            "V-JEPA 2 requires Meta's official frozen-probe video transform and layer-output path before model loading"
        )

    def encode_video(self, video_path: str, timestamps: list[float], model_family: str, *, prompt: str | None = None, return_layers: bool = False) -> EncodedVideo:
        raise LoaderError(
            "V-JEPA 2 requires Meta's official frozen-probe video transform and layer-output path to be pinned before extraction"
        )
