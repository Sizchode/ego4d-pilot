#!/usr/bin/env python3
"""Run EgoPack's published feature conversion for Ego4D OSCC and PNR."""

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--egopack-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--features",
        choices=("omnivore_video_swinl", "vjepa2_vitl16"),
        default="omnivore_video_swinl",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(args.egopack_root.resolve()))

    from data.ego4d_oscc import Ego4dOSCCDataset, Ego4dPNRDataset

    dataset_types = (
        Ego4dOSCCDataset,
        Ego4dPNRDataset,
    )
    for dataset_type in dataset_types:
        for split in ("train", "val"):
            dataset = dataset_type(
                split=split,
                root=str(args.data_root),
                features=args.features,
                verbose=True,
            )
            print(
                f"{dataset_type.__name__} {split}: "
                f"samples={len(dataset)} videos={len(dataset.video_uids)}",
                flush=True,
            )


if __name__ == "__main__":
    main()
