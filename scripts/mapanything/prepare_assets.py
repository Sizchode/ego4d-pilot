"""Download and verify the immutable MapAnything baseline inputs."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from huggingface_hub import snapshot_download


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision(path):
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def require_pinned_dinov2_cache(root, cfg):
    checkout = (root / "code/DINOv2").resolve()
    cache_entry = root / "cache/torch/hub" / cfg["dinov2_torch_hub_cache_name"]
    if not cache_entry.is_symlink():
        raise RuntimeError(f"DINOv2 Torch Hub cache entry is not a symlink: {cache_entry}")
    if cache_entry.resolve() != checkout:
        raise RuntimeError(
            f"DINOv2 Torch Hub cache resolves to {cache_entry.resolve()}, expected {checkout}"
        )
    revision = git_revision(checkout)
    if revision != cfg["dinov2_revision"]:
        raise RuntimeError(f"Unexpected DINOv2 revision: {revision}")
    subprocess.run(["git", "-C", str(checkout), "diff", "--exit-code", "HEAD"], check=True)
    return revision


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())

    actual = {
        "mapanything": git_revision(args.root / "code/MapAnything"),
        "dinov2": require_pinned_dinov2_cache(args.root, cfg),
        "benchmark": git_revision(args.root / "code/Depth-Anything-3"),
    }
    expected = {
        "mapanything": cfg["code_revision"],
        "dinov2": cfg["dinov2_revision"],
        "benchmark": cfg["benchmark_code_revision"],
    }
    if actual != expected:
        raise RuntimeError(f"Unexpected code revisions: {actual}")

    benchmark_assets_path = args.root / "provenance/assets.json"
    benchmark_assets = json.loads(benchmark_assets_path.read_text())
    benchmark_cfg = benchmark_assets["config"]
    if benchmark_cfg["dataset_revision"] != cfg["dataset_revision"]:
        raise RuntimeError("Prepared benchmark revision differs from MapAnything protocol")
    if benchmark_cfg["archives"] != cfg["archives"]:
        raise RuntimeError("Prepared benchmark archive list differs from MapAnything protocol")

    model_dir = args.root / "models/MapAnything"
    snapshot_download(
        cfg["model_repository"],
        revision=cfg["model_revision"],
        local_dir=model_dir,
        allow_patterns=["config.json", cfg["model_filename"]],
        max_workers=1,
    )
    model_file = model_dir / cfg["model_filename"]
    digest = sha256(model_file)
    if digest != cfg["model_sha256"]:
        raise RuntimeError(f"MapAnything weight digest mismatch: {digest}")

    record = {
        "config": cfg,
        "code_revisions": actual,
        "model_files": {
            "config.json": {
                "sha256": sha256(model_dir / "config.json"),
                "bytes": (model_dir / "config.json").stat().st_size,
            },
            cfg["model_filename"]: {"sha256": digest, "bytes": model_file.stat().st_size},
        },
        "benchmark_assets_sha256": sha256(benchmark_assets_path),
    }
    output = args.root / "provenance/mapanything_assets.json"
    output.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2), flush=True)


if __name__ == "__main__":
    main()
