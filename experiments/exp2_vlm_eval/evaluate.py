"""Run the official lmms-eval tasks used by Experiment 2."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tasks", nargs="+", help="Official lmms-eval task names to run")
    args = parser.parse_args()

    with Path(args.config).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    model = config["model"]
    evaluation = config["evaluation"]
    tasks = args.tasks or list(evaluation["tasks"])
    unknown_tasks = sorted(set(tasks) - set(evaluation["tasks"]))
    if unknown_tasks:
        raise ValueError(f"tasks are not part of Experiment 2: {unknown_tasks}")
    from huggingface_hub import snapshot_download

    model_path = snapshot_download(repo_id=model["repo_id"], revision=model["revision"])
    model_args = ",".join(
        [
            f"pretrained={model_path}",
            f"max_num_frames={model['max_num_frames']}",
            f"min_pixels={model['min_pixels']}",
            f"max_pixels={model['max_pixels']}",
            f"attn_implementation={model['attention']}",
        ]
    )
    output_dir = Path(args.output_dir).expanduser().resolve()
    previous_results = set(output_dir.rglob("*_results.json")) if output_dir.exists() else set()
    command = [
        sys.executable,
        "-m",
        "lmms_eval",
        "eval",
        "--model",
        "qwen3_vl",
        "--model_args",
        model_args,
        "--tasks",
        ",".join(tasks),
        "--batch_size",
        str(evaluation["batch_size"]),
        "--log_samples",
        "--output_path",
        str(output_dir),
    ]
    new_results: list[Path] = []
    for attempt in range(1, 4):
        subprocess.run(command, check=True)
        new_results = sorted(set(output_dir.rglob("*_results.json")) - previous_results)
        if len(new_results) == 1:
            break
        if attempt < 3:
            print(
                "lmms-eval produced no aggregate result; waiting 360 seconds before retry "
                f"({attempt}/3)",
                flush=True,
            )
            time.sleep(360)
    if len(new_results) != 1:
        raise RuntimeError(
            "lmms-eval did not produce exactly one new aggregate result file after three attempts; "
            f"found {len(new_results)}"
        )
    with new_results[0].open("r", encoding="utf-8") as handle:
        result = json.load(handle)
    if not isinstance(result.get("results"), dict) or not result["results"]:
        raise RuntimeError(f"lmms-eval aggregate result is incomplete: {new_results[0]}")


if __name__ == "__main__":
    main()
