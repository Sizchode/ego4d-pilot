"""Immutable per-pixel point-map archive for baseline and recovery tools."""
import argparse
import json
from pathlib import Path

import numpy as np

try:
    from .geometry_state import da3_image_transforms, unproject_depth
except ImportError:
    from geometry_state import da3_image_transforms, unproject_depth


def validate_state(arrays):
    required = {'points', 'valid', 'intrinsics', 'extrinsics', 'rgb', 'original_to_native'}
    if not required <= arrays.keys():
        raise ValueError(f'Missing state fields: {required - arrays.keys()}')
    points = arrays['points']
    if points.ndim != 4 or points.shape[-1] != 3:
        raise ValueError('Expected N x H x W x 3 world point map')
    n, h, w, _ = points.shape
    if arrays['valid'].shape != (n, h, w) or arrays['valid'].dtype != bool:
        raise ValueError('State validity must be an N x H x W boolean array')
    if arrays['rgb'].shape != points.shape or arrays['rgb'].dtype != np.uint8:
        raise ValueError('State RGB must be uint8 on the same pixel canvas')
    if arrays['extrinsics'].shape != (n, 3, 4):
        raise ValueError('State uses N x 3 x 4 OpenCV world-to-camera poses')
    for name in ['intrinsics', 'original_to_native']:
        if arrays[name].shape != (n, 3, 3):
            raise ValueError(f'Invalid {name} shape')
        if not np.isfinite(arrays[name]).all() or (np.linalg.det(arrays[name]) == 0).any():
            raise ValueError(f'Invalid {name} transform')
    if not np.isfinite(arrays['extrinsics']).all() or not np.isfinite(points[arrays['valid']]).all():
        raise ValueError('Nonfinite cameras or points marked valid')


def save_state(directory, arrays, manifest, metadata):
    validate_state(arrays)
    if len(manifest) != len(arrays['points']):
        raise ValueError('Input manifest differs from state view count')
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(directory / 'state.npz', **arrays)
    (directory / 'input_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (directory / 'state.json').write_text(json.dumps({
        'schema': 'world-point-map-v1', 'point_authority': 'state.npz:points',
        'camera_convention': 'OpenCV world-to-camera; integer pixel centers',
        **metadata}, indent=2, allow_nan=False) + '\n')


def load_state(directory):
    directory = Path(directory)
    with np.load(directory / 'state.npz', allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    validate_state(arrays)
    manifest = json.loads((directory / 'input_manifest.json').read_text())
    metadata = json.loads((directory / 'state.json').read_text())
    if metadata['schema'] != 'world-point-map-v1' or len(manifest) != len(arrays['points']):
        raise ValueError('Inconsistent state metadata')
    return arrays, manifest, metadata


def convert_da3(run, destination):
    run = Path(run)
    config = json.loads((run / 'configuration.json').read_text())
    result = json.loads((run / 'result.json').read_text())
    manifest = json.loads((run / 'input_manifest.json').read_text())
    if result['state'] != 'native_prediction_complete' or result['input_camera_conditioning']:
        raise ValueError('Expected completed RGB-only native prediction')
    if config['process_res_method'] != 'upper_bound_resize':
        raise ValueError('DA3 preprocessing mode has not been traced by this adapter')
    raw = run / 'raw'
    depth = np.load(raw / 'depth.npy')
    k, t = np.load(raw / 'intrinsics.npy'), np.load(raw / 'extrinsics.npy')[:, :3, :4]
    image_transform = da3_image_transforms(
        [row['original_hw'] for row in manifest], config['process_res'], depth.shape[1:],
        np.load(raw / 'upstream_intrinsics_transform.npy'))
    points = unproject_depth(depth, k, t).astype(np.float32)
    arrays = {'points': points, 'valid': np.isfinite(depth) & (depth > 0) & np.isfinite(points).all(-1),
              'intrinsics': k, 'extrinsics': t, 'original_to_native': image_transform,
              'rgb': np.load(raw / 'processed_images.npy'), 'confidence': np.load(raw / 'conf.npy')}
    save_state(destination, arrays, manifest, {'method': 'DA3-GIANT', 'parent': str(run),
               'gt_conditioning': False, 'preprocessing': config['process_res_method'],
               'image_transform': 'actual resize pixel centers and native batch crop; distinct from K scaling'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--da3-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    convert_da3(args.da3_run, args.output)
