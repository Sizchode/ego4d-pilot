#!/usr/bin/env python3
"""Download the three published model releases used by the experiments."""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(Path(__file__).parents[1] / "experiments" / "exp1_aot" / "config.yaml"),
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--model", choices=("qwen3", "vjepa2_1", "videomaev2"))
    args = parser.parse_args()
    with Path(args.config).open("r", encoding="utf-8") as handle:
        models = yaml.safe_load(handle)["models"]
    root = Path(args.output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    selected = (
        {args.model: models[args.model]}
        if args.model
        else models
    )
    for model in selected.values():
        destination = root / model["local_directory"]
        destination.mkdir(parents=True, exist_ok=True)
        if "checkpoint_url" in model:
            checkpoint = destination / model["checkpoint_filename"]
            if not checkpoint.is_file():
                temporary = checkpoint.with_suffix(checkpoint.suffix + ".partial")
                urllib.request.urlretrieve(model["checkpoint_url"], temporary)
                temporary.replace(checkpoint)
            print(f"downloaded official release {model['checkpoint_url']} to {checkpoint}")
        else:
            from huggingface_hub import snapshot_download

            snapshot_download(
                repo_id=model["repo_id"],
                revision=model["revision"],
                local_dir=destination,
            )
            print(f"downloaded {model['repo_id']} at {model['revision']} to {destination}")


if __name__ == "__main__":
    main()
