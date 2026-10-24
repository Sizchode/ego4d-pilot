"""Materialize official matcher weights/code in a dedicated GT-free cache."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--code', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    cache = args.cache.resolve()
    # A separate cache allows the inference sandbox to mount only model assets.
    os.environ['TORCH_HOME'] = str(cache / 'torch')
    os.environ['HF_HOME'] = str(cache / 'huggingface')
    sys.path.insert(0, str(args.code.resolve()))
    import pycolmap
    import torch
    from matching import init_match_models
    from lightglue import SuperPoint
    torch.set_num_threads(1)
    matchers = init_match_models('romav2-base', device='cpu')
    superpoint = SuperPoint(nms_radius=0, max_num_keypoints=None,
                            detection_threshold=0, remove_borders=None).eval()
    inventory = []
    for path in sorted(cache.rglob('*')):
        if path.is_symlink():
            raise ValueError('Model cache must be self-contained, without symlinks')
        if not path.is_file() or '__pycache__' in path.parts:
            continue
        digest = hashlib.sha256()
        with path.open('rb') as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
                digest.update(block)
        inventory.append({'path': str(path.relative_to(cache)), 'bytes': path.stat().st_size,
                          'sha256': digest.hexdigest()})
    report = {'state': 'matcher_assets_ready', 'matching_models': list(matchers),
              'superpoint_loaded': isinstance(superpoint, torch.nn.Module),
              'cache': str(cache), 'files': inventory, 'inference_executed': False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(f'Prepared RoMaV2-base and SuperPoint: {len(inventory)} cache files', flush=True)


if __name__ == '__main__':
    main()
