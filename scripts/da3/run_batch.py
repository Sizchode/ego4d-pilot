"""Serial DA3 benchmark execution, gated by a verified scene. No job arrays."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def build_tasks(root, cfg):
    os.chdir(root / 'code/Depth-Anything-3')
    from depth_anything_3.bench.registries import MV_REGISTRY
    dataset_names = [Path(archive).stem for archive in cfg['archives']]
    if len(dataset_names) != len(set(dataset_names)):
        raise RuntimeError('Duplicate benchmark dataset derived from the pinned archive list')
    tasks = []
    dataset_manifests = {}
    for name in dataset_names:
        scenes = list(MV_REGISTRY.get(name)().SCENES)
        if not scenes:
            raise RuntimeError(f'{name}: official scene manifest is empty')
        if len(scenes) != len(set(scenes)):
            raise RuntimeError(f'{name}: official scene manifest contains duplicates')
        dataset_manifests[name] = scenes
        tasks.extend({'dataset': name, 'scene': scene} for scene in scenes)
    return tasks, dataset_manifests


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--verified-run', type=Path, required=True)
    p.add_argument('--batch-id', required=True)
    args = p.parse_args()
    script_path = Path(__file__).resolve()
    args.config = args.config.resolve()
    cfg = json.loads(args.config.read_text())
    validation = json.loads((args.verified_run / 'validation.json').read_text())
    smoke = json.loads((args.verified_run / 'run_config.json').read_text())
    review = json.loads((args.verified_run / 'review.json').read_text())
    if not validation['passed'] or not review['artifact_review_passed'] or smoke['config'] != cfg:
        raise RuntimeError('Single-scene verification gate not satisfied')
    smoke_hashes = json.loads((args.verified_run / 'runner_sources.json').read_text())
    validated_names = set(smoke_hashes) - {script_path.name}
    source_hashes = {name: hashlib.sha256((script_path.parent / name).read_bytes()).hexdigest()
                     for name in validated_names}
    if source_hashes != {name: smoke_hashes[name] for name in validated_names}:
        raise RuntimeError('Per-scene runner changed after the verified single-scene run')
    tasks, dataset_manifests = build_tasks(args.root, cfg)
    folder = args.root / 'runs' / args.batch_id
    folder.mkdir(exist_ok=False)
    (folder / 'manifest.json').write_text(json.dumps(tasks, indent=2))
    (folder / 'dataset_manifests.json').write_text(json.dumps(dataset_manifests, indent=2))
    (folder / 'config.json').write_text(json.dumps(cfg, indent=2))
    (folder / 'runner.json').write_text(json.dumps({
        'batch_runner_sha256': hashlib.sha256(script_path.read_bytes()).hexdigest(),
        'per_scene_sources_verified_against': str(args.verified_run),
        'per_scene_source_hashes': source_hashes,
    }, indent=2))
    failures = 0
    for index, task in enumerate(tasks):
        if (task['dataset'], task['scene']) == (smoke['dataset'], smoke['scene']):
            record = {**task, 'run': str(args.verified_run), 'state': 'reused_verified_run'}
        else:
            run_id = f'{args.batch_id}-{index:03d}-{task["dataset"]}'
            command = [sys.executable, str(script_path.with_name('run_scene.py')),
                       '--root', str(args.root), '--config', str(args.config.resolve()),
                       '--dataset', task['dataset'], '--scene', task['scene'], '--run-id', run_id]
            print(f'Start {index + 1}/{len(tasks)}: {task}', flush=True)
            with (folder / f'{index:03d}.log').open('w') as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
            record = {**task, 'run': str(args.root / 'runs' / run_id), 'exit_code': result.returncode,
                      'state': 'completed' if result.returncode == 0 else 'failed'}
            failures += result.returncode != 0
        with (folder / 'results.jsonl').open('a') as handle:
            handle.write(json.dumps(record) + '\n')
        print(json.dumps(record), flush=True)
    (folder / 'summary.json').write_text(json.dumps({'total': len(tasks), 'failed': failures,
                                                   'finished': True}, indent=2))
    return int(failures > 0)


if __name__ == '__main__':
    raise SystemExit(main())
