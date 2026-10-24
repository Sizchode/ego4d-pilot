"""Native DA3 inference from a sealed RGB input mount; no evaluator imports.

Run inside the launcher filesystem sandbox. Raw outputs retain model-native
coordinates; public-canvas conversion and scoring are separate later stages.
"""
import argparse
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np
from PIL import Image


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def verify_input(directory):
    directory = Path(directory).resolve()
    manifest = json.loads((directory / 'input_manifest.json').read_text())
    expected = {'input_manifest.json'}
    paths = []
    for index, row in enumerate(manifest):
        relative = Path(row['path'])
        if row['index'] != index or relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Input manifest order or path is invalid')
        path = directory / relative
        if path.is_symlink() or path.suffix.lower() not in {'.jpg', '.jpeg', '.png'}:
            raise ValueError('Only regular RGB image files are accepted')
        if hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
            raise ValueError('Input bytes do not match the sealed manifest')
        with Image.open(path) as image:
            if [image.height, image.width] != row['original_hw']:
                raise ValueError('Input shape changed')
        expected.add(str(relative))
        paths.append(str(path))
    actual = {str(p.relative_to(directory)) for p in directory.rglob('*') if p.is_file()}
    if actual != expected or len(paths) < 2:
        raise ValueError('Unexpected files or insufficient views in RGB mount')
    return manifest, paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--forbidden-path', action='append', required=True)
    args = parser.parse_args()
    # Fail closed before importing the model if any GT/source archive is visible.
    for forbidden in args.forbidden_path:
        if Path(forbidden).exists():
            raise RuntimeError(f'Forbidden host path visible inside inference: {forbidden}')
    manifest, images = verify_input(args.input)
    config = json.loads(args.config.read_text())
    if config['model_repository'] != 'depth-anything/DA3-GIANT':
        raise ValueError('This runner implements the DA3-GIANT baseline identity')
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'input_manifest.json', manifest)
    write_json(output / 'configuration.json', config)
    write_json(output / 'isolation.json', {'forbidden_paths_absent': args.forbidden_path,
               'input_files': [row['path'] for row in manifest], 'network_offline': True})

    import torch
    from depth_anything_3.api import DepthAnything3
    random.seed(config['sampling_seed'])
    np.random.seed(config['sampling_seed'])
    torch.manual_seed(config['sampling_seed'])
    torch.cuda.manual_seed_all(config['sampling_seed'])
    torch.set_num_threads(1)
    started = time.monotonic()
    model = DepthAnything3.from_pretrained(str(args.model)).to('cuda').eval()
    model.input_processor = partial(model.input_processor, num_workers=1, sequential=True)
    raw = output / 'raw'
    raw.mkdir()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    inference_started = time.monotonic()
    with torch.inference_mode():
        pred = model.inference(images, export_dir=None,
                               ref_view_strategy=config['ref_view_strategy'],
                               process_res=config['process_res'],
                               process_res_method=config['process_res_method'])
    torch.cuda.synchronize()
    inference_seconds = time.monotonic() - inference_started
    for name in ['depth', 'conf', 'extrinsics', 'intrinsics', 'sky', 'processed_images']:
        value = getattr(pred, name)
        if value is not None:
            np.save(raw / f'{name}.npy', value)
    depth = np.asarray(pred.depth)
    if depth.ndim != 3 or depth.shape[0] != len(images):
        raise ValueError('Unexpected output view layout')
    if not np.isfinite(pred.extrinsics).all() or not np.isfinite(pred.intrinsics).all():
        raise ValueError('Nonfinite camera predictions')
    np.save(raw / 'valid_depth.npy', np.isfinite(depth) & (depth > 0))
    # Record upstream calibration transform without supplying it to the network.
    identity = np.repeat(np.eye(3, dtype=np.float32)[None], len(images), axis=0)
    _, _, native_transform = model.input_processor(
        images, intrinsics=identity, process_res=config['process_res'],
        process_res_method=config['process_res_method'], num_workers=1, sequential=True)
    np.save(raw / 'upstream_intrinsics_transform.npy', native_transform.numpy())
    write_json(output / 'result.json', {
        'state': 'native_prediction_complete', 'views': len(images),
        'depth_shape': list(depth.shape), 'valid_depth_fraction': float((np.isfinite(depth) & (depth > 0)).mean()),
        'model': config['model_repository'], 'input_camera_conditioning': False,
        'gt_access': False, 'public_canvas_validated': False, 'scores_computed': False,
        'inference_seconds': inference_seconds, 'total_seconds': time.monotonic() - started,
        'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(),
        'peak_cuda_reserved_bytes': torch.cuda.max_memory_reserved(),
        'torch_version': torch.__version__, 'cuda_version': torch.version.cuda,
        'gpu': torch.cuda.get_device_name(), 'slurm_job_id': os.environ.get('SLURM_JOB_ID')})
    print(json.dumps(json.loads((output / 'result.json').read_text())), flush=True)


if __name__ == '__main__':
    main()
