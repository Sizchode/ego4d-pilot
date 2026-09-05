"""Config-driven model loader construction."""

from __future__ import annotations

import importlib
from typing import Any

from .base import VideoLoader


def build_loader(model_config: dict[str, Any], device: Any) -> VideoLoader:
    module_name, class_name = str(model_config["loader"]).split(":", 1)
    loader_type = getattr(importlib.import_module(module_name), class_name)
    loader = loader_type(model_config, device)
    if not isinstance(loader, VideoLoader):
        raise TypeError(f"configured loader does not implement VideoLoader: {model_config['loader']}")
    return loader
