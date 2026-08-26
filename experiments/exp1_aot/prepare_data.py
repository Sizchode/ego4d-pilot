"""Create Experiment 1 manifests from the published dataset releases."""

from __future__ import annotations

import argparse
from pathlib import Path

from dataset import build_aot_examples, build_ssv2_examples, write_jsonl
from utils import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=("ssv2", "aot_psyphybench"))
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--video-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    _, config = load_config(args.config)

    if args.dataset == "ssv2":
        dataset_config = config["datasets"]["ssv2_train"]
        examples = build_ssv2_examples(
            args.annotations,
            args.video_root,
            seed=int(config["seed"]),
            expected_examples=int(dataset_config["expected_examples"]),
        )
    else:
        dataset_config = config["datasets"]["aot_psyphybench"]
        examples = build_aot_examples(
            args.annotations,
            args.video_root,
            expected_source_clips=int(dataset_config["expected_source_clips"]),
        )
        if len(examples) != int(dataset_config["expected_examples"]):
            raise RuntimeError("AoT-PsyPhyBENCH example count does not match the released benchmark")

    write_jsonl(examples, args.output)
    print(f"wrote {len(examples)} examples to {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
