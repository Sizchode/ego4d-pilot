"""Official VideoMAE V2 custom-checkpoint frozen encoder adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import RepresentationProvider


class VideoMAEv2Provider(RepresentationProvider):
    def __init__(self, model_path: str | Path, config: dict[str, Any], device: Any):
        import torch
        from transformers import AutoImageProcessor, AutoModel

        dtype = getattr(torch, config["dtype"])
        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModel.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=bool(config["trust_remote_code"]),
            torch_dtype=dtype,
        ).to(device)
        self.model.eval().requires_grad_(False)
        actual = int(self.model.model.embed_dim)
        expected = int(config["expected_hidden_size"])
        if actual != expected:
            raise RuntimeError(f"VideoMAE V2 hidden size is {actual}; protocol requires {expected}")
        self.representation_dimensions = {"last_encoder_tokens": actual}

    def extract(self, videos: list[Any]) -> dict[str, list[Any]]:
        import torch

        processed = [
            self.processor(images=list(video), return_tensors="pt")["pixel_values"] for video in videos
        ]
        # The released model's Conv3d patch embed requires B,C,T,H,W.
        pixel_values = torch.stack(processed).permute(0, 2, 1, 3, 4).to(self.device, self.model.dtype)
        captured: list[torch.Tensor] = []

        def capture(_module: Any, _inputs: Any, output: torch.Tensor) -> None:
            captured.append(output)

        handle = self.model.model.blocks[-1].register_forward_hook(capture)
        try:
            with torch.inference_mode():
                self.model.extract_features(pixel_values)
        finally:
            handle.remove()
        if len(captured) != 1:
            raise RuntimeError(f"expected one VideoMAE V2 encoder capture; got {len(captured)}")
        # Match the released encoder's final-token convention by applying its final norm.
        tokens = self.model.model.norm(captured[0])
        return {"last_encoder_tokens": [sample for sample in tokens]}
