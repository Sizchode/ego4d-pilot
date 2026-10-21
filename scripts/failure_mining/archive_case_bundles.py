"""Archive exact evidence files for selected scenes from an immutable snapshot."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import tarfile
from pathlib import Path
from typing import Any


REQUIRED_FILES = (
    "run_config.json",
    "status.json",
    "validation.json",
    "input_manifest.json",
    "preview.png",
    "camera_errors.png",
    "diagnostics/protocol.json",
    "evaluation_capture/camera_pairs.npz",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def slug(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")
    if not result:
        raise ValueError(f"Identity cannot be represented safely: {value!r}")
    return result


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_archive(
    metrics_jsonl: Path,
    dataset: str,
    scenes: list[str],
    models: list[str],
    output_tar: Path,
) -> dict[str, Any]:
    if not scenes or len(scenes) != len(set(scenes)):
        raise ValueError("Scenes must be a non-empty unique list")
    if not models or len(models) != len(set(models)):
        raise ValueError("Models must be a non-empty unique list")
    if output_tar.exists():
        raise ValueError(f"Refusing to overwrite archive: {output_tar}")
    rows = read_jsonl(metrics_jsonl)
    lookup = {}
    for row in rows:
        key = (row["model"], row["dataset"], row["scene"])
        if key in lookup:
            raise ValueError(f"Duplicate metric row: {key}")
        lookup[key] = row

    cases = []
    output_tar.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output_tar, "w") as archive:
        for scene in scenes:
            for model in models:
                key = (model, dataset, scene)
                if key not in lookup:
                    raise ValueError(f"Snapshot does not contain selected case: {key}")
                row = lookup[key]
                run = Path(row["run"])
                status = json.loads((run / "status.json").read_text())
                validation = json.loads((run / "validation.json").read_text())
                config = json.loads((run / "run_config.json").read_text())
                if status.get("state") != "completed" or validation.get("passed") is not True:
                    raise ValueError(f"Selected case is not completed and validated: {key}")
                if (config.get("dataset"), config.get("scene")) != (dataset, scene):
                    raise ValueError(f"Run identity mismatch: {key}")
                prefix = f"{slug(model)}-{slug(scene)}"
                files = []
                for relative in REQUIRED_FILES:
                    path = run / relative
                    if not path.is_file():
                        raise ValueError(f"Required evidence is missing: {path}")
                    archive.add(path, arcname=f"{prefix}/{relative}", recursive=False)
                    files.append(
                        {
                            "path": relative,
                            "bytes": path.stat().st_size,
                            "sha256": sha256(path),
                        }
                    )
                cases.append(
                    {
                        "model": model,
                        "dataset": dataset,
                        "scene": scene,
                        "archive_prefix": prefix,
                        "run": str(run),
                        "files": files,
                    }
                )
        manifest = {
            "schema_version": 1,
            "metrics_jsonl": str(metrics_jsonl.resolve()),
            "metrics_jsonl_sha256": sha256(metrics_jsonl),
            "dataset": dataset,
            "scenes": scenes,
            "models": models,
            "case_count": len(cases),
            "file_count": sum(len(case["files"]) for case in cases),
            "required_files": list(REQUIRED_FILES),
            "cases": cases,
            "failure_labels": None,
            "selection_policy": "explicit scenes selected from separate metric-rank evidence",
        }
        payload = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
        info = tarfile.TarInfo("archive_manifest.json")
        info.size = len(payload)
        info.mtime = 0
        archive.addfile(info, io.BytesIO(payload))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-jsonl", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--scene", action="append", required=True)
    parser.add_argument("--model", action="append", required=True)
    parser.add_argument("--output-tar", type=Path, required=True)
    args = parser.parse_args()
    manifest = build_archive(
        args.metrics_jsonl,
        args.dataset,
        args.scene,
        args.model,
        args.output_tar,
    )
    print(
        json.dumps(
            {
                "output_tar": str(args.output_tar.resolve()),
                "case_count": manifest["case_count"],
                "file_count": manifest["file_count"],
                "scenes": manifest["scenes"],
                "models": manifest["models"],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
