"""Model-specific official preprocessing and representation adapters."""

from __future__ import annotations

import importlib
from typing import Any

from .base import AdapterError, RepresentationAdapter


def build_adapter(model_config: dict[str, Any], device: Any) -> RepresentationAdapter:
    module_name, class_name = str(model_config["adapter"]).split(":", 1)
    module = importlib.import_module(module_name)
    adapter_type = getattr(module, class_name)
    return adapter_type(model_config, device)


__all__ = ["AdapterError", "RepresentationAdapter", "build_adapter"]
