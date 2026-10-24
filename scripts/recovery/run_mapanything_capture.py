"""Archive MapAnything before its official mask, preserving the normal output.

One model forward yields both filtered and unfiltered point/depth states. The
upstream postprocessor receives and returns exactly its original arguments.
"""
import argparse
import importlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np

from geometry_state import unproject_depth
from native_state import load_state, save_state
import run_mapanything_rgb


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--code', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args, _ = parser.parse_known_args()
    sys.path.insert(0, str(args.code))
    module = importlib.import_module('mapanything.models.mapanything.model')
    original = module.postprocess_model_outputs_for_inference
    captured = []

    def capture(*positional, **keywords):
        if captured:
            raise RuntimeError('Expected exactly one original model postprocessing call')
        raw_outputs = keywords['raw_outputs']
        directory = args.output/'unmasked-raw'; directory.mkdir()
        for index, output in enumerate(raw_outputs):
            arrays = {key: output[key][0].detach().float().cpu().numpy()
                      for key in ['pts3d', 'pts3d_cam']}
            path = directory/f'{index:06d}.npz'
            np.savez_compressed(path, **arrays)
            captured.append(path)
        return original(*positional, **keywords)

    with patch.object(module, 'postprocess_model_outputs_for_inference', capture):
        run_mapanything_rgb.main()
    arrays, manifest, metadata = load_state(args.output/'point-state')
    assert len(captured) == len(manifest) and not metadata['gt_conditioning']
    world, local = [], []
    for path in captured:
        with np.load(path, allow_pickle=False) as archive:
            world.append(archive['pts3d']); local.append(archive['pts3d_cam'])
    world, local = np.stack(world), np.stack(local)
    assert world.shape == local.shape == arrays['points'].shape
    # Original filtered positions must be exact subsets of the captured raw data.
    np.testing.assert_array_equal(arrays['points'][arrays['valid']], world[arrays['valid']])
    point_valid = np.isfinite(world).all(-1)
    save_state(args.output/'unmasked-point-state', dict(arrays, points=world, valid=point_valid), manifest,
               {'method': 'MapAnything-point-unmasked', 'gt_conditioning': False,
                'parent': str(args.output/'point-state'), 'point_branch': 'exact model pts3d before default mask',
                'additional_model_forward': False, 'mask_policy': 'finite points only'})
    depth_arrays, depth_manifest, depth_metadata = load_state(args.output/'depth-camera-state')
    assert depth_manifest == manifest and not depth_metadata['gt_conditioning']
    depth = local[..., 2]
    depth_points = unproject_depth(depth, depth_arrays['intrinsics'], depth_arrays['extrinsics']).astype(np.float32)
    np.testing.assert_array_equal(depth_arrays['points'][depth_arrays['valid']], depth_points[depth_arrays['valid']])
    depth_valid = np.isfinite(depth_points).all(-1) & np.isfinite(depth) & (depth > 0)
    save_state(args.output/'unmasked-depth-camera-state',
               dict(depth_arrays, points=depth_points, valid=depth_valid), manifest,
               {'method': 'MapAnything-depth+camera-unmasked', 'gt_conditioning': False,
                'parent': str(args.output/'depth-camera-state'),
                'point_branch': 'captured raw camera-Z unprojected with unchanged predicted pinhole K and cameras',
                'additional_model_forward': False, 'mask_policy': 'finite points and positive predicted depth'})
    report = {'state': 'unmasked_capture_complete', 'views': len(manifest), 'gt_access': False,
              'model_forward_calls': 1, 'filtered_xyz_exact_subset_of_raw': True,
              'filtered_depth_xyz_exact_subset_of_raw': True,
              'filtered_point_valid_pixels': int(arrays['valid'].sum()),
              'unmasked_point_valid_pixels': int(point_valid.sum()),
              'unmasked_depth_valid_pixels': int(depth_valid.sum()), 'scores_computed': False}
    (args.output/'capture-result.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
