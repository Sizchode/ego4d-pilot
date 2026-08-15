"""V-JEPA 2 adapter for the official Ego4D feature extraction pipeline."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import torch
from ego4d.features.config import BaseModelConfig, InferenceConfig
from pytorchvideo.transforms import ApplyTransformToKey, ShortSideScale
from torch import nn
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import CenterCropVideo, NormalizeVideo


@dataclass
class ModelConfig(BaseModelConfig):
    repository_path: str
    checkpoint_path: str
    checkpoint_key: str = "target_encoder"
    side_size: int = 292
    crop_size: int = 256
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    use_bfloat16: bool = True


class VJEPA2GlobalEncoder(nn.Module):
    """Pool the frozen encoder's normalized spatiotemporal tokens."""

    def __init__(self, encoder: nn.Module, use_bfloat16: bool):
        super().__init__()
        self.encoder = encoder
        self.use_bfloat16 = use_bfloat16

    def forward(self, sample: dict[str, torch.Tensor]) -> torch.Tensor:
        with torch.cuda.amp.autocast(
            dtype=torch.bfloat16,
            enabled=self.use_bfloat16 and sample["video"].is_cuda,
        ):
            tokens = self.encoder(sample["video"])
        if tokens.ndim != 3:
            raise ValueError(f"Expected B,N,D encoder output, received {tuple(tokens.shape)}")
        return tokens.mean(dim=1).float()


def _clean_encoder_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        key.replace("module.", "").replace("backbone.", ""): value
        for key, value in state_dict.items()
    }


def load_model(
    inference_config: InferenceConfig,
    config: ModelConfig,
    patch_final_layer: bool = True,
) -> nn.Module:
    if not patch_final_layer:
        raise ValueError("V-JEPA 2 extraction requires the head-free encoder output")

    repository_path = str(Path(config.repository_path).resolve())
    if repository_path not in sys.path:
        sys.path.insert(0, repository_path)
    from src.models import vision_transformer

    encoder = vision_transformer.vit_large(
        img_size=config.crop_size,
        num_frames=inference_config.frame_window,
        patch_size=16,
        tubelet_size=2,
        img_temporal_dim_size=None,
        uniform_power=True,
        use_rope=True,
    )
    checkpoint = torch.load(config.checkpoint_path, map_location="cpu")
    state_dict = _clean_encoder_state_dict(checkpoint[config.checkpoint_key])
    load_result = encoder.load_state_dict(state_dict, strict=False)
    if load_result.missing_keys or load_result.unexpected_keys:
        raise RuntimeError(f"Unexpected V-JEPA 2 checkpoint mismatch: {load_result}")
    del checkpoint

    model = VJEPA2GlobalEncoder(encoder, config.use_bfloat16).eval().to(
        inference_config.device
    )
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model


def _normalize_pixels(video: torch.Tensor) -> torch.Tensor:
    return video / 255.0


def get_transform(inference_config: InferenceConfig, config: ModelConfig):
    if inference_config.frame_window != 32:
        raise ValueError("V-JEPA 2 is aligned to Ego4D's released 32-frame video-feature window")
    transform = Compose(
        [
            Lambda(_normalize_pixels),
            NormalizeVideo(config.mean, config.std),
            ShortSideScale(size=config.side_size),
            CenterCropVideo(config.crop_size),
        ]
    )
    return ApplyTransformToKey(key="video", transform=transform)
