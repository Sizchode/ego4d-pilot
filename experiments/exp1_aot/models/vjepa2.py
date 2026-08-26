"""Frozen V-JEPA 2.1 encoder loaded from its verified Hugging Face conversion."""

from __future__ import annotations

import math
import random
from typing import Any

import numpy as np


_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _normalize(clip: Any) -> Any:
    import torch

    mean = torch.tensor(_IMAGENET_MEAN, dtype=clip.dtype)[:, None, None, None]
    std = torch.tensor(_IMAGENET_STD, dtype=clip.dtype)[:, None, None, None]
    return (clip - mean) / std


def _spatial_crop_parameters(
    height: int,
    width: int,
    scale: tuple[float, float] = (0.3, 1.0),
    ratio: tuple[float, float] = (3 / 4, 4 / 3),
) -> tuple[int, int, int, int]:
    """Meta V-JEPA's released ``_get_param_spatial_crop`` implementation."""

    for _ in range(10):
        target_area = random.uniform(*scale) * height * width
        aspect_ratio = math.exp(random.uniform(math.log(ratio[0]), math.log(ratio[1])))
        crop_width = int(round(math.sqrt(target_area * aspect_ratio)))
        crop_height = int(round(math.sqrt(target_area / aspect_ratio)))
        if 0 < crop_width <= width and 0 < crop_height <= height:
            top = random.randint(0, height - crop_height)
            left = random.randint(0, width - crop_width)
            return top, left, crop_height, crop_width

    input_ratio = width / height
    if input_ratio < min(ratio):
        crop_width = width
        crop_height = int(round(crop_width / min(ratio)))
    elif input_ratio > max(ratio):
        crop_height = height
        crop_width = int(round(crop_height * max(ratio)))
    else:
        crop_width, crop_height = width, height
    return (height - crop_height) // 2, (width - crop_width) // 2, crop_height, crop_width


class _OfficialFrozenProbeTransform:
    """Meta V-JEPA 2.1 frozen-probe transform for a single spatial view."""

    def __init__(self, resolution: int, training: bool):
        self.resolution = resolution
        self.training = training

    def __call__(self, video: np.ndarray) -> Any:
        import cv2
        import torch
        from torchvision.transforms.functional import pil_to_tensor, to_pil_image

        if video.ndim != 4 or video.shape[-1] != 3:
            raise ValueError(f"expected THWC RGB video; got {video.shape}")

        if self.training:
            # The released transform performs ToPILImage -> ToTensor per frame.
            frames = torch.stack(
                [pil_to_tensor(to_pil_image(frame)).float().div_(255.0) for frame in video]
            )
            clip = _normalize(frames.permute(1, 0, 2, 3))
            _, _, height, width = clip.shape
            top, left, crop_height, crop_width = _spatial_crop_parameters(height, width)
            clip = torch.nn.functional.interpolate(
                clip[:, :, top : top + crop_height, left : left + crop_width],
                size=(self.resolution, self.resolution),
                mode="bilinear",
                align_corners=False,
            )
            if np.random.uniform() < 0.5:
                clip = clip.flip(-1)
            return clip

        short_side = int(self.resolution * 256 / 224)
        height, width = video.shape[1:3]
        if width < height:
            resized_width = short_side
            resized_height = int(short_side * height / width)
        else:
            resized_height = short_side
            resized_width = int(short_side * width / height)
        resized = np.stack(
            [
                cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
                for frame in video
            ]
        )
        left = int(round((resized_width - self.resolution) / 2.0))
        top = int(round((resized_height - self.resolution) / 2.0))
        cropped = resized[
            :, top : top + self.resolution, left : left + self.resolution, :
        ]
        clip = torch.from_numpy(cropped).permute(3, 0, 1, 2).float().div_(255.0)
        return _normalize(clip)


class VJepa2Provider:
    def __init__(self, config: dict[str, Any], device: Any):
        import torch
        from transformers import AutoModel

        self.device = device
        self.dtype = getattr(torch, config["dtype"])
        self.resolution = int(config["resolution"])
        self.model = AutoModel.from_pretrained(
            config["repo_id"],
            revision=config["revision"],
            trust_remote_code=bool(config["trust_remote_code"]),
            dtype=self.dtype,
        ).to(device)
        self.model.eval().requires_grad_(False)

        actual = int(self.model.config.hidden_size)
        expected = int(config["expected_hidden_size"])
        if actual != expected:
            raise RuntimeError(f"V-JEPA 2.1 hidden size is {actual}; config requires {expected}")
        self.representation_dimensions = {"last_hidden_state": actual}
        self.train_transform = _OfficialFrozenProbeTransform(self.resolution, training=True)
        self.evaluation_transform = _OfficialFrozenProbeTransform(self.resolution, training=False)

    def extract(self, videos: list[Any], training: bool = False) -> dict[str, list[Any]]:
        import torch

        transform = self.train_transform if training else self.evaluation_transform
        pixel_values = torch.stack([transform(video) for video in videos]).to(
            device=self.device, dtype=self.dtype
        )
        with torch.inference_mode():
            tokens = self.model(pixel_values, skip_predictor=True).last_hidden_state
        return {"last_hidden_state": [sample for sample in tokens]}
