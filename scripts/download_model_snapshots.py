#!/usr/bin/env python3
"""Download only the three protocol-pinned model snapshots to scratch."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

from temporal_transfer.protocol import load_protocol


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    protocol = load_protocol(args.protocol)
    root = Path(args.output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    for model in protocol.data["models"].values():
        destination = root / model["local_directory"]
        snapshot_download(
            repo_id=model["repo_id"],
            revision=model["revision"],
            local_dir=destination,
        )
        print(f"downloaded {model['repo_id']} at {model['revision']} to {destination}")


if __name__ == "__main__":
    main()
