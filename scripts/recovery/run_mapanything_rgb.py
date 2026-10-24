"""Run the pinned image-only MapAnything baseline on sealed, ordered RGBs."""
import argparse
import json
from pathlib import Path
import random
import sys
import time

import numpy as np

from geometry_state import pixel_resize, unproject_depth
from native_state import save_state
from run_da3_rgb import verify_input, write_json


def image_transforms(records):
    """Map pixel centers through the actual PIL raster resize and integer crop."""
    transforms = []
    for row in records:
        width, height = row['original_wh']
        rw, rh = row['resized_wh']
        crop = np.array([[1., 0, -row['crop_left']], [0, 1., -row['crop_top']], [0, 0, 1.]])
        transforms.append(crop @ pixel_resize((height, width), (rh, rw)))
    return np.stack(transforms)


def main():
    parser = argparse.ArgumentParser()
    for name in ['input', 'model', 'code', 'config', 'output']:
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--forbidden-path', action='append', required=True)
    args = parser.parse_args()
    if any(Path(path).exists() for path in args.forbidden_path):
        raise RuntimeError('A forbidden GT/source directory is accessible')
    manifest, paths = verify_input(args.input)
    config = json.loads(args.config.read_text())
    assert config['model_repository'] == 'facebook/map-anything'
    assert config['primary_geometry_branch'] == 'official_masked_depth_z_plus_predicted_camera'
    sys.path.insert(0, str(args.code))
    import torch
    from mapanything.models import MapAnything
    from mapanything_preprocess import load_images_with_metadata
    torch.set_num_threads(1)
    random.seed(config['sampling_seed']); np.random.seed(config['sampling_seed'])
    torch.manual_seed(config['sampling_seed']); torch.cuda.manual_seed_all(config['sampling_seed'])
    args.output.mkdir(parents=True, exist_ok=False)
    raw = args.output/'raw'; raw.mkdir()
    write_json(args.output/'input_manifest.json', manifest)
    write_json(args.output/'configuration.json', config)
    views, _, records, _, _ = load_images_with_metadata(paths, resolution_set=config['resolution_set'])
    allowed = {'img', 'data_norm_type', 'true_shape', 'idx', 'instance'}
    assert all(set(view) <= allowed for view in views), 'Only RGB and image metadata may enter model'
    transforms = image_transforms(records)
    write_json(args.output/'preprocessing.json', records)
    model = MapAnything.from_pretrained(str(args.model)).to('cuda').eval()
    parameters = ['memory_efficient_inference', 'minibatch_size', 'use_amp', 'amp_dtype',
                  'apply_mask', 'mask_edges', 'edge_normal_threshold', 'edge_depth_threshold',
                  'apply_confidence_mask', 'confidence_percentile', 'use_multiview_confidence']
    torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); started = time.monotonic()
    with torch.inference_mode():
        outputs = model.infer(views, **{key: config[key] for key in parameters})
    torch.cuda.synchronize(); seconds = time.monotonic()-started
    assert len(outputs) == len(manifest)
    keys = ['pts3d', 'pts3d_cam', 'depth_z', 'depth_along_ray', 'ray_directions', 'intrinsics',
            'camera_poses', 'cam_trans', 'cam_quats', 'conf', 'mask', 'non_ambiguous_mask',
            'non_ambiguous_mask_logits', 'metric_scaling_factor', 'img_no_norm']
    arrays = {}
    for key in keys:
        values = []
        for output in outputs:
            value = output[key][0].detach().cpu()
            if value.dtype == torch.bfloat16:
                value = value.float()
            values.append(value.numpy())
        arrays[key] = np.stack(values)
        np.save(raw/(key+'.npy'), arrays[key])
    depth = arrays['depth_z'][..., 0]
    k = arrays['intrinsics']
    t = np.linalg.inv(arrays['camera_poses'].astype(np.float64))[:, :3, :].astype(np.float32)
    mask = arrays['mask'][..., 0].astype(bool)
    rgb = np.rint(arrays['img_no_norm']*255).astype(np.uint8)
    shared = {'intrinsics': k, 'extrinsics': t, 'rgb': rgb, 'confidence': arrays['conf'],
              'original_to_native': transforms}
    points = unproject_depth(depth, k, t).astype(np.float32)
    valid = mask & np.isfinite(depth) & (depth > 0) & np.isfinite(points).all(-1)
    save_state(args.output/'depth-camera-state', dict(shared, points=points, valid=valid), manifest,
               {'method': 'MapAnything-depth+camera', 'gt_conditioning': False,
                'point_branch': 'official masked depth_z unprojected with predicted pinhole K and cameras',
                'preprocessing': 'official fixed_mapping; actual raster pixel-center resize and integer crop',
                'parent': None})
    points = arrays['pts3d'].astype(np.float32)
    save_state(args.output/'point-state', dict(shared, points=points,
               valid=mask & np.isfinite(points).all(-1)), manifest,
               {'method': 'MapAnything-point', 'gt_conditioning': False,
                'point_branch': 'official masked pts3d world points; retained separately', 'parent': None})
    write_json(args.output/'result.json', {'state': 'native_prediction_complete', 'views': len(manifest),
               'gt_access': False, 'forbidden_paths_absent': args.forbidden_path,
               'inference_seconds': seconds, 'native_hw': list(depth.shape[1:]),
               'primary_branch': config['primary_geometry_branch'], 'primary_valid_pixels': int(valid.sum()),
               'torch_version': torch.__version__, 'gpu': torch.cuda.get_device_name(),
               'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(), 'scores_computed': False})
    print((args.output/'result.json').read_text(), flush=True)


if __name__ == '__main__':
    main()
