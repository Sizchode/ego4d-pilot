"""Prepare immutable GGPT assets; stage RGB separately from evaluator-only GT.

This entry point never imports a reconstruction model or calculates a score.
The RGB tree is an input allowlist, not an OS sandbox: inference launchers must
restrict their reads to that tree and must not call GGPT's GT-bearing loader.
"""
import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def expected_sequence_files(prefix, sequence, views):
    base = f'{prefix}/{sequence}'
    return ({f'{base}/images/{i:06d}.jpg' for i in range(views)} |
            {f'{base}/depths/{i:06d}.npy' for i in range(views)} |
            {f'{base}/extrinsics.npy', f'{base}/intrinsics.npy'})


def validate_file_manifest(dataset):
    actual = [item['path'] for item in dataset['files']]
    for name in actual:
        path = PurePosixPath(name)
        if path.is_absolute() or '..' in path.parts or path.as_posix() != name:
            raise ValueError(f'Unsafe dataset path: {name}')
    if len(actual) != len(set(actual)):
        raise ValueError('Duplicate dataset files')
    expected = set()
    for subset in dataset['subsets']:
        ids = subset['sequence_ids']
        if len(ids) != len(set(ids)) or len(ids) != subset['sequence_count'] or not ids:
            raise ValueError('Invalid sequence inventory')
        for sequence in ids:
            expected.update(expected_sequence_files(subset['prefix'], sequence, subset['views']))
    if set(actual) != expected:
        raise ValueError(f'Published sequence schema mismatch: missing={len(expected-set(actual))}, '
                         f'unexpected={len(set(actual)-expected)}')


def link_verified(source, target):
    """Use hardlinks, so the policy tree cannot traverse a link into the GT tree."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink():
        raise ValueError(f'Symlink not permitted in staged inputs: {target}')
    if target.exists():
        if not os.path.samefile(source, target):
            raise FileExistsError(f'Refusing to replace previously staged input: {target}')
        return
    os.link(source, target)


def stage_sequence(raw, policy, evaluator, prefix, sequence, views):
    base = Path(prefix) / sequence
    source = raw / base
    shapes = []
    for i in range(views):
        name = f'{i:06d}'
        rgb_path = source / 'images' / f'{name}.jpg'
        with Image.open(rgb_path) as image:
            shapes.append((image.height, image.width))
            image.verify()
        depth_path = source / 'depths' / f'{name}.npy'
        depth = np.load(depth_path, mmap_mode='r', allow_pickle=False)
        if depth.shape != shapes[-1] or not np.issubdtype(depth.dtype, np.floating):
            raise ValueError(f'Depth/image schema mismatch: {base}/{name}')
    extrinsics = np.load(source / 'extrinsics.npy', allow_pickle=False)
    intrinsics = np.load(source / 'intrinsics.npy', allow_pickle=False)
    if extrinsics.shape != (views, 4, 4) or intrinsics.shape != (views, 3, 3):
        raise ValueError(f'Camera array shape mismatch: {base}')
    if not np.isfinite(extrinsics).all() or not np.isfinite(intrinsics).all():
        raise ValueError(f'Nonfinite camera arrays: {base}')
    policy_scene = policy / base
    expected = {f'images/{i:06d}.jpg' for i in range(views)}
    if policy_scene.exists():
        # Exact bounded sequence directory, on a Slurm compute node.
        entries = list(policy_scene.rglob('*'))
        if any(p.is_symlink() for p in entries):
            raise ValueError(f'Symlink not permitted in policy tree: {base}')
        existing = {str(p.relative_to(policy_scene)) for p in entries if p.is_file()}
        if existing - expected:
            raise ValueError(f'Non-RGB file in policy tree: {base}')
    for name in sorted(expected):
        link_verified(source / name, policy_scene / name)
    for name in [f'depths/{i:06d}.npy' for i in range(views)] + ['extrinsics.npy', 'intrinsics.npy']:
        link_verified(source / name, evaluator / base / name)
    return {'sequence_key': base.as_posix(), 'views': views, 'image_shapes': shapes,
            'policy_images': [f'{base}/images/{i:06d}.jpg' for i in range(views)],
            'source_scene_mapping': 'unresolved'}


def prepare_code(root, cfg):
    destination = root / 'code/GGPT'
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        subprocess.run(['git', 'init', str(destination)], check=True)
        subprocess.run(['git', '-C', str(destination), 'remote', 'add', 'origin', cfg['repository']], check=True)
    remote = subprocess.check_output(['git', '-C', str(destination), 'remote', 'get-url', 'origin'], text=True).strip()
    if remote != cfg['repository']:
        raise ValueError('Existing GGPT checkout has a different origin')
    dirty = subprocess.check_output(['git', '-C', str(destination), 'status', '--porcelain'], text=True)
    if dirty:
        raise ValueError('Existing GGPT checkout has local changes; will not overwrite')
    subprocess.run(['git', '-C', str(destination), 'fetch', '--depth', '1', 'origin', cfg['revision']], check=True)
    subprocess.run(['git', '-C', str(destination), 'checkout', '--detach', cfg['revision']], check=True)
    subprocess.run(['git', '-c', 'url.https://github.com/.insteadOf=git@github.com:', '-C', str(destination),
                    'submodule', 'update', '--init', '--recursive', '--jobs', '1'], check=True)
    revision = subprocess.check_output(['git', '-C', str(destination), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != cfg['revision']:
        raise ValueError('Source revision mismatch')
    submodules = subprocess.check_output(['git', '-C', str(destination), 'submodule', 'status', '--recursive'], text=True)
    if any(line[0] != ' ' for line in submodules.splitlines()):
        raise ValueError('Submodule does not match its upstream gitlink')
    return {'revision': revision, 'submodules': submodules.splitlines()}


def main():
    from huggingface_hub import hf_hub_download

    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    validate_file_manifest(cfg['dataset'])
    provenance = args.root / 'provenance/ggpt'
    provenance.mkdir(parents=True, exist_ok=True)
    record = {'stage': 'assets_only_no_inference_or_scoring', 'config': cfg, 'model_files': {},
              'dataset_files': {}, 'sequences': [], 'completed': False}
    output = provenance / 'assets.json'

    def save():
        temporary = output.with_suffix('.tmp')
        temporary.write_text(json.dumps(record, indent=2) + '\n')
        temporary.replace(output)

    save()
    record['code'] = prepare_code(args.root, cfg['code'])
    save()
    for name in cfg['model']['files']:
        path = Path(hf_hub_download(cfg['model']['repository'], name, revision=cfg['model']['revision'],
                                   local_dir=args.root / 'models/GGPT'))
        record['model_files'][name] = {'bytes': path.stat().st_size, 'sha256': digest(path)}
        save()
        print(f'Prepared model asset: {name}', flush=True)
    raw = args.root / 'datasets/archives/GGPT_eval'
    file_records = {item['path']: item for item in cfg['dataset']['files']}
    total = sum(subset['sequence_count'] for subset in cfg['dataset']['subsets'])
    for subset in cfg['dataset']['subsets']:
        for sequence in subset['sequence_ids']:
            for name in sorted(expected_sequence_files(subset['prefix'], sequence, subset['views'])):
                item = file_records[name]
                path = raw / name
                # Resume by verifying against the immutable publisher's digest,
                # without spending another HTTP request on completed inputs.
                if not path.is_file():
                    path = Path(hf_hub_download(cfg['dataset']['repository'], name, repo_type='dataset',
                                               revision=cfg['dataset']['revision'], local_dir=raw))
                checksum = digest(path)
                if path.stat().st_size != item['bytes'] or (item['sha256'] and checksum != item['sha256']):
                    raise ValueError(f'Pinned file integrity failure: {name}')
                record['dataset_files'][name] = {'bytes': path.stat().st_size, 'sha256': checksum}
            staged = stage_sequence(raw, args.root / 'datasets/GGPT_eval_16/rgb_only',
                                    args.root / 'datasets/GGPT_eval_16/evaluator_gt',
                                    subset['prefix'], sequence, subset['views'])
            record['sequences'].append(staged)
            save()
            print(f"Prepared sequence {len(record['sequences'])}/{total}: {staged['sequence_key']}", flush=True)
    record['completed'] = True
    record['inference_started'] = False
    record['scores_opened'] = False
    save()
    print(f'Completed {total} sequences. No inference or scoring executed.', flush=True)


if __name__ == '__main__':
    main()
