"""Official Transformers V-JEPA 2 frozen encoder adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import RepresentationProvider


class VJepa2Provider(RepresentationProvider):
    def __init__(self, model_path: str | Path, config: dict[str, Any], device: Any):
        import torch
        from transformers import AutoModel, AutoVideoProcessor

        dtype = getattr(torch, config["dtype"])
        self.device = device
        self.processor = AutoVideoProcessor.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModel.from_pretrained(
            model_path, local_files_only=True, torch_dtype=dtype, attn_implementation=config["attention"]
        ).to(device)
        self.model.eval().requires_grad_(False)
        actual = int(self.model.config.hidden_size)
        expected = int(config["expected_hidden_size"])
        if actual != expected:
            raise RuntimeError(f"V-JEPA 2 hidden size is {actual}; protocol requires {expected}")
        self.representation_dimensions = {"last_hidden_state": actual}

    def extract(self, videos: list[Any]) -> dict[str, list[Any]]:
        import torch

        inputs = self.processor(videos=videos, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            outputs = self.model(**inputs, skip_predictor=True, return_dict=True)
        return {"last_hidden_state": [tokens for tokens in outputs.last_hidden_state]}
