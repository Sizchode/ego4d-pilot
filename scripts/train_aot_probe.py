#!/usr/bin/env python3
"""Train and evaluate the published attentive Arrow-of-Time probe online."""

from __future__ import annotations

import argparse

from temporal_transfer.protocol import load_protocol
from temporal_transfer.trainer import run_probe_training


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--provider", required=True, choices=("qwen3", "vjepa2", "videomaev2"))
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--eval-manifest", required=True)
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tat-root", default="third_party/TAT-VQA")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    run_probe_training(
        load_protocol(args.protocol),
        args.provider,
        args.train_manifest,
        args.eval_manifest,
        args.model_root,
        args.output_dir,
        args.tat_root,
        args.workers,
    )


if __name__ == "__main__":
    main()
