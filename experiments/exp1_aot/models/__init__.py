"""Frozen backbones compared in Experiment 1."""

from pathlib import Path
from typing import Any


def build_model(name: str, model_root: str | Path, config: dict[str, Any], device: Any) -> Any:
    model_path = Path(model_root).expanduser().resolve() / config["local_directory"]
    if name == "qwen3":
        from .qwen3 import Qwen3VLProvider

        return Qwen3VLProvider(model_path, config, device)
    if name == "vjepa2_1":
        from .vjepa2 import VJepa2Provider

        return VJepa2Provider(model_path, config, device)
    if name == "videomaev2":
        from .videomaev2 import VideoMAEv2Provider

        return VideoMAEv2Provider(model_path, config, device)
    raise ValueError(f"unknown model: {name}")
