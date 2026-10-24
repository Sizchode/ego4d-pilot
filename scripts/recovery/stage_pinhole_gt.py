"""Evaluator-only adapter for published OpenCV pinhole camera-Z arrays.

Uses the existing full-image public canvas; no predictions, confidence, ROI or
model-selected views are inputs. Source depth and camera units are preserved.
"""
import argparse
import json
from pathlib import Path

import numpy as np

try:
    from .geometry_state import pixel_resize
    from .gt_canvas import nearest_exact
    from .public_canvas import public_shape, pixel_grid, transform_pixels
except ImportError:
    from geometry_state import pixel_resize
    from gt_canvas import nearest_exact
    from public_canvas import public_shape, pixel_grid, transform_pixels


def pinhole_to_public(depth, intrinsics, extrinsics):
    depth = np.asarray(depth)
    k, t = np.asarray(intrinsics, dtype=np.float64), np.asarray(extrinsics, dtype=np.float64)
    if depth.ndim != 2 or not np.issubdtype(depth.dtype, np.floating):
        raise ValueError('Floating camera-Z raster required')
    if (k.shape != (3, 3) or not np.isfinite(k).all() or not np.allclose(k[2], [0, 0, 1])
            or k[0, 0] <= 0 or k[1, 1] <= 0):
        raise ValueError('Finite OpenCV pinhole intrinsics required')
    if (t.shape != (4, 4) or not np.isfinite(t).all() or not np.allclose(t[3], [0, 0, 0, 1])
            or not np.allclose(t[:3, :3]@t[:3, :3].T, np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(t[:3, :3]), 1, atol=1e-5)):
        raise ValueError('Finite rigid world-to-camera transform required')
    transform = pixel_resize(depth.shape, public_shape(depth.shape))
    original_uv = transform_pixels(np.linalg.inv(transform), pixel_grid(public_shape(depth.shape)))
    sampled, valid = nearest_exact(depth, np.isfinite(depth)&(depth>0), original_uv)
    rays = np.concatenate([original_uv, np.ones(original_uv.shape[:-1]+(1,))], -1)@np.linalg.inv(k).T
    camera_points = rays*sampled[..., None]
    points = (camera_points-t[:3, 3])@t[:3, :3]
    valid &= np.isfinite(points).all(-1)
    points[~valid] = np.nan
    return {'points': points.astype(np.float32), 'valid': valid, 'depth': sampled.astype(np.float32),
            'original_to_public': transform, 'extrinsics': t, 'intrinsics': transform@k}


def stage(source, manifest_path, output):
    source, output = Path(source), Path(output)
    manifest = json.loads(Path(manifest_path).read_text())
    if len(manifest) < 2 or [row['index'] for row in manifest] != list(range(len(manifest))):
        raise ValueError('Ordered multi-view RGB identity required')
    if any(row['path'] != f'images/{index:06d}.jpg' for index, row in enumerate(manifest)):
        raise ValueError('RGB order differs from the extracted dataset naming convention')
    intrinsics = np.load(source/'intrinsics.npy', allow_pickle=False)
    extrinsics = np.load(source/'extrinsics.npy', allow_pickle=False)
    if intrinsics.shape != (len(manifest), 3, 3) or extrinsics.shape != (len(manifest), 4, 4):
        raise ValueError('GT camera count differs from fixed RGB inputs')
    output.mkdir(parents=True, exist_ok=False)
    records = []
    for index, row in enumerate(manifest):
        depth = np.load(source/'depths'/f'{index:06d}.npy', allow_pickle=False)
        if list(depth.shape) != row['original_hw']:
            raise ValueError('GT raster does not match the original RGB grid')
        result = pinhole_to_public(depth, intrinsics[index], extrinsics[index])
        name = f'{index:06d}.npz'
        np.savez_compressed(output/name, **result)
        records.append({'index': index, 'file': name, 'shape': list(result['valid'].shape),
                        'valid_pixels': int(result['valid'].sum())})
    (output/'input_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    (output/'canvas.json').write_text(json.dumps({
        'schema': 'public-ground-truth-v1', 'long_edge': 504, 'patch': 14,
        'pixel_centers': 'published OpenCV integer centers; no additional principal-point shift',
        'sampling': 'nearest-exact camera-Z at full-image half-pixel resize coordinates',
        'valid_domain': 'finite positive source camera-Z; independent of predictions',
        'camera': 'pinhole, zero lens distortion; world-to-camera extrinsics',
        'units': 'metres, unchanged from published source', 'views': records}, indent=2)+'\n')
    return {'staged_views': len(records), 'gt_valid_pixels': sum(r['valid_pixels'] for r in records),
            'prediction_access': False, 'scores_computed': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ['source', 'manifest', 'output']:
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(stage(args.source, args.manifest, args.output)))
