"""Map authoritative world points onto the frozen, full-image scoring canvas.

This prediction-only module cannot access ground truth. Cropped or unsupported
pixels stay invalid; it never derives replacement geometry from camera depth.
"""
import argparse
import json
from pathlib import Path

import numpy as np

try:
    from .geometry_state import pixel_resize, sample_pointmap
    from .native_state import load_state
except ImportError:
    from geometry_state import pixel_resize, sample_pointmap
    from native_state import load_state


def public_shape(original_hw, long_edge=504, patch=14):
    original = np.asarray(original_hw)
    if original.shape != (2,) or (original <= 0).any() or long_edge <= 0 or patch <= 0:
        raise ValueError('Positive image dimensions and canvas parameters required')
    scaled = original.astype(np.float64) * long_edge / original.max()
    # Nearest patch multiple, with exact half-patch ties rounded upward.
    return tuple((np.maximum(1, np.floor(scaled / patch + 0.5)) * patch).astype(int))


def pixel_grid(hw):
    y, x = np.indices(hw, dtype=np.float64)
    return np.stack([x, y, np.ones_like(x)], -1)


def transform_pixels(homography, grid):
    mapped = np.einsum('ij,hwj->hwi', homography, grid)
    return mapped[..., :2] / mapped[..., 2:]


def map_prediction(points, valid, intrinsics, original_hw, original_to_native):
    shape = public_shape(original_hw)
    original_to_public = pixel_resize(original_hw, shape)
    public_to_native = original_to_native @ np.linalg.inv(original_to_public)
    uv = transform_pixels(public_to_native, pixel_grid(shape))
    public_points, public_valid = sample_pointmap(points, valid, uv)
    native_to_public = np.linalg.inv(public_to_native)
    return {'points': public_points.astype(np.float32), 'valid': public_valid,
            'intrinsics': native_to_public @ intrinsics,
            'original_to_public': original_to_public,
            'native_to_public': native_to_public}


def export_prediction(state_directory, output):
    arrays, manifest, metadata = load_state(state_directory)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    records = []
    for index, row in enumerate(manifest):
        public = map_prediction(arrays['points'][index], arrays['valid'][index],
                                arrays['intrinsics'][index], row['original_hw'],
                                arrays['original_to_native'][index])
        public['extrinsics'] = arrays['extrinsics'][index]
        name = f'{index:06d}.npz'
        np.savez_compressed(output / name, **public)
        records.append({'index': index, 'file': name, 'shape': list(public['valid'].shape),
                        'valid_pixels': int(public['valid'].sum())})
    (output / 'input_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    (output / 'canvas.json').write_text(json.dumps({
        'schema': 'public-world-point-map-v1', 'parent_state': str(state_directory),
        'parent_metadata': metadata, 'long_edge': 504, 'patch': 14,
        'rounding': 'nearest_multiple_ties_up', 'pixel_centers': 'integer_half_pixel_resize',
        'sampling': 'bilinear_world_points_all_four_neighbors_valid',
        'extrapolation': False, 'gt_access': False, 'views': records}, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--state', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    export_prediction(args.state, args.output)
