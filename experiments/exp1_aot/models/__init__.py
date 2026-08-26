"""Frozen backbones compared in Experiment 1."""

from typing import Any


def build_model(name: str, config: dict[str, Any], device: Any) -> Any:
    if name == "qwen3":
        from .qwen3 import Qwen3VLProvider

        return Qwen3VLProvider(config, device)
    if name == "vjepa2_1":
        from .vjepa2 import VJepa2Provider

        return VJepa2Provider(config, device)
    if name == "videomaev2":
        from .videomaev2 import VideoMAEv2Provider

        return VideoMAEv2Provider(config, device)
    raise ValueError(f"unknown model: {name}")
