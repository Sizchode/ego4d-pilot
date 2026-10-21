"""Serial Pi3X benchmark execution gated by one reviewed scene."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def build_tasks(root, cfg):
    os.chdir(root / "code/Depth-Anything-3")
    from depth_anything_3.bench.registries import MV_REGISTRY

    dataset_names = [Path(archive).stem for archive in cfg["archives"]]
    if len(dataset_names) != len(set(dataset_names)):
        raise RuntimeError("Duplicate benchmark dataset derived from archive list")
    tasks = []
    manifests = {}
    for name in dataset_names:
        scenes = list(MV_REGISTRY.get(name)().SCENES)
        if not scenes or len(scenes) != len(set(scenes)):
            raise RuntimeError(f"Invalid official scene manifest for {name}")
        manifests[name] = scenes
        tasks.extend({"dataset": name, "scene": scene} for scene in scenes)
    return tasks, manifests


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--verified-run", type=Path, required=True)
    parser.add_argument("--batch-id", required=True)
    args = parser.parse_args()
    args.config = args.config.resolve()
    cfg = json.loads(args.config.read_text())
    validation = json.loads((args.verified_run / "validation.json").read_text())
    smoke = json.loads((args.verified_run / "run_config.json").read_text())
    review = json.loads((args.verified_run / "review.json").read_text())
    if not validation["passed"] or not validation.get("pi3x_raw_outputs_present"):
        raise RuntimeError("Smoke run did not pass Pi3X artifact validation")
    if not review["artifact_review_passed"] or smoke["config"] != cfg:
        raise RuntimeError("Reviewed smoke configuration does not match batch configuration")

    runner_root = Path(__file__).parents[2]
    smoke_hashes = json.loads((args.verified_run / "runner_sources.json").read_text())
    batch_key = str(Path(__file__).relative_to(runner_root))
    expected = {key: value for key, value in smoke_hashes.items() if key != batch_key}
    actual = {key: sha256(runner_root / key) for key in expected}
    if actual != expected:
        raise RuntimeError("Per-scene runner changed after reviewed smoke run")

    tasks, manifests = build_tasks(args.root, cfg)
    batch_dir = args.root / "runs" / args.batch_id
    batch_dir.mkdir(exist_ok=False)
    (batch_dir / "manifest.json").write_text(json.dumps(tasks, indent=2) + "\n")
    (batch_dir / "dataset_manifests.json").write_text(json.dumps(manifests, indent=2) + "\n")
    (batch_dir / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    (batch_dir / "runner.json").write_text(json.dumps({
        "batch_runner_sha256": sha256(Path(__file__)),
        "per_scene_sources_verified_against": str(args.verified_run),
        "per_scene_source_hashes": actual,
    }, indent=2) + "\n")

    failures = 0
    for index, task in enumerate(tasks):
        if (task["dataset"], task["scene"]) == (smoke["dataset"], smoke["scene"]):
            record = {**task, "run": str(args.verified_run), "state": "reused_verified_run"}
        else:
            run_id = f'{args.batch_id}-{index:03d}-{task["dataset"]}'
            command = [
                sys.executable, "-m", "scripts.pi3x.run_scene",
                "--root", str(args.root), "--config", str(args.config),
                "--dataset", task["dataset"], "--scene", task["scene"], "--run-id", run_id,
            ]
            print(f"Start {index + 1}/{len(tasks)}: {task}", flush=True)
            with (batch_dir / f"{index:03d}.log").open("w") as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
            record = {
                **task,
                "run": str(args.root / "runs" / run_id),
                "exit_code": result.returncode,
                "state": "completed" if result.returncode == 0 else "failed",
            }
            failures += result.returncode != 0
        with (batch_dir / "results.jsonl").open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
    (batch_dir / "summary.json").write_text(json.dumps({
        "total": len(tasks), "failed": failures, "finished": True,
    }, indent=2) + "\n")
    return int(failures > 0)


if __name__ == "__main__":
    raise SystemExit(main())
