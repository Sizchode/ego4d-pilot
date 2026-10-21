"""Download and verify the immutable VGGT baseline inputs."""
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())

    actual = {
        "vggt": git_revision(args.root / "code/VGGT"),
        "benchmark": git_revision(args.root / "code/Depth-Anything-3"),
    }
    expected = {
        "vggt": cfg["code_revision"],
        "benchmark": cfg["benchmark_code_revision"],
    }
    if actual != expected:
        raise RuntimeError(f"Unexpected code revisions: {actual}")

    benchmark_assets_path = args.root / "provenance/assets.json"
    benchmark_assets = json.loads(benchmark_assets_path.read_text())
    benchmark_cfg = benchmark_assets["config"]
    if benchmark_cfg["dataset_revision"] != cfg["dataset_revision"]:
        raise RuntimeError("Prepared benchmark revision differs from VGGT protocol")
    if benchmark_cfg["archives"] != cfg["archives"]:
        raise RuntimeError("Prepared benchmark archive list differs from VGGT protocol")

    model_dir = args.root / "models/VGGT-1B"
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
        raise RuntimeError(f"VGGT weight digest mismatch: {digest}")

    record = {
        "config": cfg,
        "code_revisions": actual,
        "model_files": {
            "config.json": {
                "sha256": sha256(model_dir / "config.json"),
                "bytes": (model_dir / "config.json").stat().st_size,
            },
            cfg["model_filename"]: {
                "sha256": digest,
                "bytes": model_file.stat().st_size,
            },
        },
        "benchmark_assets_sha256": sha256(benchmark_assets_path),
    }
    output = args.root / "provenance/vggt_assets.json"
    output.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2), flush=True)


if __name__ == "__main__":
    main()
