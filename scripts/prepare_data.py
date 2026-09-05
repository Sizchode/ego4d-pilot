"""Download only configured pilot media and build official-split manifests."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.temporal_resampling.manifest import build_nextqa, build_ssv2, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    download = subparsers.add_parser("download-hf")
    download.add_argument("--repo-id", required=True)
    download.add_argument("--revision", required=True)
    download.add_argument("--output-dir", required=True)
    manifest = subparsers.add_parser("manifest")
    manifest.add_argument("dataset", choices=("ssv2", "nextqa"))
    manifest.add_argument("--annotations", required=True)
    manifest.add_argument("--video-root", required=True)
    manifest.add_argument("--split", required=True)
    manifest.add_argument("--output", required=True)
    manifest.add_argument("--label-map", help="Required for SSv2 official label IDs")
    manifest.add_argument("--annotation-revision", required=True)
    args = parser.parse_args()
    if args.command == "download-hf":
        from huggingface_hub import snapshot_download

        print(snapshot_download(repo_id=args.repo_id, repo_type="dataset", revision=args.revision, local_dir=args.output_dir))
    elif args.dataset == "ssv2":
        if not args.label_map:
            raise ValueError("--label-map is required for SSv2")
        write_jsonl(build_ssv2(args.annotations, args.video_root, args.split, args.label_map, args.annotation_revision), args.output)
    else:
        write_jsonl(build_nextqa(args.annotations, args.video_root, args.split, args.annotation_revision), args.output)


if __name__ == "__main__":
    main()
