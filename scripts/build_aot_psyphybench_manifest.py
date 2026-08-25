#!/usr/bin/env python3
"""Build the released AoT-PsyPhyBENCH paired evaluation manifest."""

from __future__ import annotations

import argparse

from temporal_transfer.manifests import build_aot_examples, write_jsonl
from temporal_transfer.protocol import load_protocol


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--video-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    protocol = load_protocol(args.protocol)
    dataset = protocol.data["datasets"]["aot_psyphybench"]
    examples = build_aot_examples(
        args.metadata, args.video_root, expected_source_clips=int(dataset["expected_source_clips"])
    )
    if len(examples) != int(dataset["expected_examples"]):
        raise RuntimeError(f"built {len(examples)} examples; protocol requires {dataset['expected_examples']}")
    write_jsonl(examples, args.output)
    print(f"wrote {len(examples)} examples to {args.output}")


if __name__ == "__main__":
    main()
