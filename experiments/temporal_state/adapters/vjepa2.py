"""V-JEPA 2 ViT-L adapter; compact state is deliberately unsupported."""

from __future__ import annotations

from typing import Any

from .base import AdapterError, RepresentationAdapter


class VJEPA2ViTLAdapter(RepresentationAdapter):
    def __init__(self, config: dict[str, Any], device: Any):
        try:
            import torch
            from transformers import AutoModel
        except ImportError as exc:
            raise AdapterError("V-JEPA 2 extraction requires the pinned transformers environment on HPC") from exc
        self.torch = torch
        self.device = device
        self.config = config
        revision = config.get("revision")
        if not isinstance(revision, str) or not revision:
            raise AdapterError("V-JEPA 2 config must pin an immutable Hugging Face revision before HPC execution")
        self.model = AutoModel.from_pretrained(config["repo_id"], revision=revision, trust_remote_code=True).to(device)
        self.model.eval().requires_grad_(False)
        self.layer_count = int(getattr(self.model.config, "num_hidden_layers"))
        hidden_size = int(self.model.config.hidden_size)
        self._dimensions = {f"layer_{layer}": hidden_size for layer in range(self.layer_count)}

    @property
    def representation_dimensions(self) -> dict[str, int]:
        return dict(self._dimensions)

    @property
    def total_layers(self) -> int:
        return self.layer_count

    def extract_compact_state(self, video_paths: list[str], prompts: list[str]) -> dict[str, Any]:
        raise AdapterError("the final protocol does not define a language-style compact state for V-JEPA 2")

    def extract_distributed_state(self, video_paths: list[str], prompts: list[str]) -> dict[str, list[Any]]:
        raise AdapterError(
            "V-JEPA 2 video decoding must use Meta's official frozen-probe transform; integrate that transform before running this adapter"
        )
