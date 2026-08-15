#!/usr/bin/env python3
"""Validate a frozen feature backbone checkpoint and its output contract."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from types import SimpleNamespace

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRATCH_ROOT = Path(
    os.environ.get(
        "EGO4D_PILOT_SCRATCH",
        f"/oscar/scratch/{os.environ.get('USER', 'user')}/ego4d-state-change",
    )
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backbone", choices=("dinov2", "vjepa2"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inference_config = SimpleNamespace(
        device="cuda", batch_size=1, num_workers=0, frame_window=1
    )

    if args.backbone == "dinov2":
        from feature_models.dinov2 import DINOv2FrameEncoder, ModelConfig, load_model

        inference_config.frame_window = 32
        config = ModelConfig(
            repository_path=str(PROJECT_ROOT / "third_party" / "DINOv2"),
            checkpoint_path=str(
                SCRATCH_ROOT / "checkpoints" / "dinov2" / "dinov2_vitl14_pretrain.pth"
            ),
        )
        model = load_model(inference_config, config)
        if not isinstance(model, DINOv2FrameEncoder):
            raise TypeError(type(model))
        sample = {"video": torch.zeros(1, 3, 32, 224, 224, device="cuda")}
    else:
        from feature_models.vjepa2 import ModelConfig, VJEPA2GlobalEncoder, load_model

        inference_config.frame_window = 32
        config = ModelConfig(
            repository_path=str(PROJECT_ROOT / "third_party" / "vjepa2"),
            checkpoint_path=str(SCRATCH_ROOT / "checkpoints" / "vjepa2" / "vitl.pt"),
        )
        model = load_model(inference_config, config)
        if not isinstance(model, VJEPA2GlobalEncoder):
            raise TypeError(type(model))
        sample = {"video": torch.zeros(1, 3, 32, 256, 256, device="cuda")}

    with torch.inference_mode():
        output = model(sample)
    expected_shape = (1, 1024)
    if tuple(output.shape) != expected_shape:
        raise ValueError(f"Expected output {expected_shape}, received {tuple(output.shape)}")
    if not torch.isfinite(output).all():
        raise ValueError("Backbone produced non-finite values")
    print(
        f"{args.backbone}: output_shape={tuple(output.shape)} "
        f"dtype={output.dtype} finite={bool(torch.isfinite(output).all())}"
    )


if __name__ == "__main__":
    main()
