#!/usr/bin/env python3
"""Measure V-JEPA 2 inference memory while increasing the clip batch size."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from types import SimpleNamespace

import torch

from feature_models.vjepa2 import ModelConfig, load_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRATCH_ROOT = Path(
    os.environ.get(
        "EGO4D_PILOT_SCRATCH",
        f"/oscar/scratch/{os.environ.get('USER', 'user')}/ego4d-state-change",
    )
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=(1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 64, 80, 96),
    )
    return parser.parse_args()


def gib(num_bytes: int) -> float:
    return num_bytes / 1024**3


def main() -> None:
    args = parse_args()
    inference_config = SimpleNamespace(device="cuda", frame_window=32)
    model_config = ModelConfig(
        repository_path=str(PROJECT_ROOT / "third_party" / "vjepa2"),
        checkpoint_path=str(SCRATCH_ROOT / "checkpoints" / "vjepa2" / "vitl.pt"),
        use_bfloat16=True,
    )
    model = load_model(inference_config, model_config)
    device_properties = torch.cuda.get_device_properties(0)
    print(
        f"device={device_properties.name} total_memory_gib="
        f"{gib(device_properties.total_memory):.2f}",
        flush=True,
    )

    largest_successful_batch = 0
    for batch_size in args.batch_sizes:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            video = torch.zeros(
                batch_size,
                3,
                32,
                256,
                256,
                dtype=torch.float32,
                device="cuda",
            )
            with torch.inference_mode():
                output = model({"video": video})
            torch.cuda.synchronize()
            allocated = torch.cuda.max_memory_allocated()
            reserved = torch.cuda.max_memory_reserved()
            if output.shape != (batch_size, 1024):
                raise ValueError(f"Unexpected output shape: {tuple(output.shape)}")
            if not torch.isfinite(output).all():
                raise ValueError("Non-finite output")
            largest_successful_batch = batch_size
            print(
                f"batch={batch_size} status=PASS "
                f"peak_allocated_gib={gib(allocated):.2f} "
                f"peak_reserved_gib={gib(reserved):.2f}",
                flush=True,
            )
            del video, output
        except torch.cuda.OutOfMemoryError as error:
            print(f"batch={batch_size} status=OOM error={error}", flush=True)
            break

    print(f"largest_successful_batch={largest_successful_batch}", flush=True)


if __name__ == "__main__":
    main()
