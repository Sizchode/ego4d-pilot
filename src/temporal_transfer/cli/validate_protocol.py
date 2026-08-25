"""Validate an experiment protocol."""

from __future__ import annotations

import argparse

from temporal_transfer.protocol import load_protocol


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("protocol")
    args = parser.parse_args()
    protocol = load_protocol(args.protocol)
    print(f"valid protocol: {protocol.path}")


if __name__ == "__main__":
    main()
