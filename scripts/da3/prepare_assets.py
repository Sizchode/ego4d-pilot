"""Download immutable DA3 inputs and validate every ZIP before extraction."""
import argparse
import hashlib
import json
import subprocess
import time
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    code = args.root / "code/Depth-Anything-3"
    revision = subprocess.check_output(["git", "-C", str(code), "rev-parse", "HEAD"], text=True).strip()
    if revision != cfg["code_revision"]:
        raise RuntimeError(f"Unexpected code revision: {revision}")
    provenance = args.root / "provenance"
    provenance.mkdir(exist_ok=True)
    model = args.root / "models/DA3-GIANT"
    snapshot_download(cfg["model_repository"], revision=cfg["model_revision"],
                      local_dir=model, allow_patterns=["config.json", "model.safetensors"], max_workers=1)
    record = {"config": cfg, "model_files": {}, "archives": {}}
    for name in ["config.json", "model.safetensors"]:
        path = model / name
        record["model_files"][name] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    (provenance / "assets.json").write_text(json.dumps(record, indent=2))
    destination = args.root / "datasets"
    for name in cfg["archives"]:
        started = time.monotonic()
        path = Path(hf_hub_download(cfg["dataset_repository"], name, repo_type="dataset",
                                   revision=cfg["dataset_revision"], local_dir=destination / "archives"))
        digest = sha256(path)
        marker = provenance / (name + ".json")
        if marker.exists() and json.loads(marker.read_text())["sha256"] == digest:
            record["archives"][name] = json.loads(marker.read_text())
            continue
        with zipfile.ZipFile(path) as archive:
            for member in archive.infolist():
                target = (destination / member.filename).resolve()
                if not target.is_relative_to(destination.resolve()):
                    raise ValueError(f"Archive member escapes destination: {member.filename}")
                if (member.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError(f"Unexpected symlink in benchmark archive: {member.filename}")
            failed = archive.testzip()
            if failed:
                raise ValueError(f"ZIP checksum failure: {failed}")
            archive.extractall(destination)
            entry = {"sha256": digest, "bytes": path.stat().st_size,
                     "members": len(archive.infolist()), "seconds": time.monotonic() - started}
        marker.write_text(json.dumps(entry, indent=2))
        record["archives"][name] = entry
        (provenance / "assets.json").write_text(json.dumps(record, indent=2))
        print(f"Prepared {name}: {entry}", flush=True)
    (provenance / "assets.json").write_text(json.dumps(record, indent=2))
    workspace = code / "workspace"
    workspace.mkdir(exist_ok=True)
    link = workspace / "benchmark_dataset"
    if link.is_symlink():
        if link.resolve() != destination.resolve():
            raise ValueError(f"Unexpected benchmark symlink: {link}")
    elif link.exists():
        raise FileExistsError(link)
    else:
        link.symlink_to(destination, target_is_directory=True)
    print("All immutable assets prepared.", flush=True)


if __name__ == "__main__":
    main()
