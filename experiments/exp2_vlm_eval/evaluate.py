"""Run the official lmms-eval tasks used by Experiment 2."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    with Path(args.config).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    model = config["model"]
    evaluation = config["evaluation"]
    model_path = Path(args.model_root).expanduser().resolve() / model["local_directory"]
    model_args = ",".join(
        [
            f"pretrained={model_path}",
            f"max_num_frames={model['max_num_frames']}",
            f"min_pixels={model['min_pixels']}",
            f"max_pixels={model['max_pixels']}",
            f"attn_implementation={model['attention']}",
        ]
    )
    command = [
        sys.executable,
        "-m",
        "lmms_eval",
        "--model",
        "qwen3_vl",
        "--model_args",
        model_args,
        "--tasks",
        ",".join(evaluation["tasks"]),
        "--batch_size",
        str(evaluation["batch_size"]),
        "--log_samples",
        "--output_path",
        str(Path(args.output_dir).expanduser().resolve()),
    ]
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
