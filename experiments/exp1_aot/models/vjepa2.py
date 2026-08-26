"""Meta's official V-JEPA 2.1 ViT-L/384 frozen encoder adapter."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


def _clean_official_backbone_keys(state_dict: dict[str, Any]) -> dict[str, Any]:
    """Apply the exact key conversion in V-JEPA 2.1 ``src/hub/backbones.py``."""

    cleaned = {}
    for key, value in state_dict.items():
        cleaned[key.replace("module.", "").replace("backbone.", "")] = value
    return cleaned


class VJepa2Provider:
    def __init__(self, model_path: str | Path, config: dict[str, Any], device: Any):
        import torch

        self.device = device
        self.dtype = getattr(torch, config["dtype"])
        self.resolution = int(config["resolution"])
        checkpoint = Path(model_path).expanduser().resolve() / config["checkpoint_filename"]
        if not checkpoint.is_file():
            raise RuntimeError(f"official V-JEPA 2.1 checkpoint is missing: {checkpoint}")

        hub_repository = f"facebookresearch/vjepa2:{config['code_revision']}"
        encoder, _predictor = torch.hub.load(
            hub_repository,
            config["constructor"],
            trust_repo=True,
            pretrained=False,
        )
        hub_cache = Path(torch.hub.get_dir()) / (
            "facebookresearch_vjepa2_" + config["code_revision"].replace("/", "_")
        )
        if not hub_cache.is_dir():
            raise RuntimeError(f"PyTorch Hub did not create the expected pinned cache: {hub_cache}")
        if str(hub_cache) not in sys.path:
            sys.path.insert(0, str(hub_cache))
        released = torch.load(checkpoint, map_location="cpu", weights_only=True)
        weights = _clean_official_backbone_keys(released[config["checkpoint_key"]])
        encoder.load_state_dict(weights, strict=True)
        self.model = encoder.to(device=device, dtype=self.dtype).eval().requires_grad_(False)

        actual = int(self.model.embed_dim)
        expected = int(config["expected_hidden_size"])
        if actual != expected:
            raise RuntimeError(f"V-JEPA 2.1 hidden size is {actual}; config requires {expected}")
        self.representation_dimensions = {"last_hidden_state": actual}

        from evals.video_classification_frozen.utils import make_transforms

        self.train_transform = make_transforms(training=True, crop_size=self.resolution)
        self.evaluation_transform = make_transforms(training=False, crop_size=self.resolution)

    def extract(self, videos: list[Any], training: bool = False) -> dict[str, list[Any]]:
        import torch

        transform = self.train_transform if training else self.evaluation_transform
        processed = [transform(video)[0] for video in videos]
        pixel_values = torch.stack(processed).to(device=self.device, dtype=self.dtype)
        with torch.inference_mode():
            tokens = self.model(pixel_values)
        return {"last_hidden_state": [sample for sample in tokens]}
