"""DINOv2 adapter for the official Ego4D feature extraction pipeline."""

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
    side_size: int = 256
    crop_size: int = 224
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)


class DINOv2FrameEncoder(nn.Module):
    """Encode the center frame of each released Ego4D 32-frame window."""

    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder = encoder

    def forward(self, sample: dict[str, torch.Tensor]) -> torch.Tensor:
        video = sample["video"]
        if video.ndim != 5 or video.shape[2] < 1:
            raise ValueError(f"Expected B,C,T,H,W input with T >= 1, received {tuple(video.shape)}")
        return self.encoder(video[:, :, video.shape[2] // 2])


def load_model(
    inference_config: InferenceConfig,
    config: ModelConfig,
    patch_final_layer: bool = True,
) -> nn.Module:
    if not patch_final_layer:
        raise ValueError("DINOv2 extraction requires the head-free backbone output")

    repository_path = str(Path(config.repository_path).resolve())
    if repository_path not in sys.path:
        sys.path.insert(0, repository_path)
    from dinov2.hub.backbones import dinov2_vitl14

    encoder = dinov2_vitl14(pretrained=False)
    state_dict = torch.load(config.checkpoint_path, map_location="cpu")
    encoder.load_state_dict(state_dict, strict=True)
    model = DINOv2FrameEncoder(encoder).eval().to(inference_config.device)
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model


def _normalize_pixels(video: torch.Tensor) -> torch.Tensor:
    return video / 255.0


def get_transform(inference_config: InferenceConfig, config: ModelConfig):
    if inference_config.frame_window != 32:
        raise ValueError("DINOv2 must use Ego4D's released 32-frame feature window")
    transform = Compose(
        [
            Lambda(_normalize_pixels),
            NormalizeVideo(config.mean, config.std),
            ShortSideScale(size=config.side_size),
            CenterCropVideo(config.crop_size),
        ]
    )
    return ApplyTransformToKey(key="video", transform=transform)
