#!/usr/bin/env python3
"""Build the SSv2-train Arrow-of-Time manifest fixed by the protocol."""

from __future__ import annotations

import argparse

from temporal_transfer.manifests import build_ssv2_examples, write_jsonl
from temporal_transfer.protocol import load_protocol


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--video-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    protocol = load_protocol(args.protocol)
    dataset = protocol.data["datasets"]["ssv2"]
    examples = build_ssv2_examples(
        args.annotations,
        args.video_root,
        seed=int(dataset["direction_assignment"]["seed"]),
        expected_examples=int(dataset["expected_examples"]),
    )
    write_jsonl(examples, args.output)
    print(f"wrote {len(examples)} examples to {args.output}")


if __name__ == "__main__":
    main()
