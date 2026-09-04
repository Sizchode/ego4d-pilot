"""Download released media and build strict manifests on the HPC node.

This command is intentionally not run during local development.  It never
downloads complete Ego4D and it never invents an EgoTempo or V-STaR split.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .manifest import build_nextqa_manifest, build_ssv2_manifest, write_jsonl
from .protocol import ProtocolError


HF_DATASET_REPOS = {
    "ssv2": "morpheushoc/something-something-v2",
    "nextvideo": "rhymes-ai/NeXTVideo",
    "vstar": "V-STaR-Bench/V-STaR",
}


def download_hf(repo_id: str, output_dir: str | Path, revision: str | None) -> Path:
    if not revision:
        raise ProtocolError("an immutable Hugging Face revision is required for every HPC download")
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ProtocolError("huggingface_hub is required for HPC data preparation") from exc
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    kwargs = {"repo_id": repo_id, "repo_type": "dataset", "local_dir": str(destination)}
    if revision is not None:
        kwargs["revision"] = revision
    return Path(snapshot_download(**kwargs))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    download = subparsers.add_parser("download")
    download.add_argument("dataset", choices=tuple(HF_DATASET_REPOS))
    download.add_argument("--output-dir", required=True)
    download.add_argument("--revision", required=True, help="Exact HF commit/tag for a reproducible HPC run")

    manifest = subparsers.add_parser("manifest")
    manifest.add_argument("dataset", choices=("ssv2", "nextqa"))
    manifest.add_argument("--annotations", required=True)
    manifest.add_argument("--video-root", required=True)
    manifest.add_argument("--split", required=True)
    manifest.add_argument("--label-map", required=False, help="Required for SSv2; official label-to-index JSON")
    manifest.add_argument("--output", required=True)

    args = parser.parse_args()
    if args.command == "download":
        path = download_hf(HF_DATASET_REPOS[args.dataset], args.output_dir, args.revision)
        print(path)
        return
    if args.dataset == "ssv2":
        records = build_ssv2_manifest(args.annotations, args.video_root, args.split, args.label_map)
    else:
        records = build_nextqa_manifest(args.annotations, args.video_root, args.split)
    write_jsonl(records, args.output)
    print(f"wrote {len(records)} records to {Path(args.output).expanduser().resolve()}")


if __name__ == "__main__":
    main()
