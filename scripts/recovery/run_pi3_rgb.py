"""Original Pi3 checkpoint on fixed RGB inputs, preserving its native points."""
import argparse
import json
from pathlib import Path
import random
import sys
import time

import numpy as np

from geometry_state import pixel_resize
from native_state import save_state
from run_da3_rgb import verify_input, write_json


def main():
    parser = argparse.ArgumentParser()
    for name in ['input', 'model', 'code', 'config', 'output']:
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--forbidden-path', action='append', required=True)
    args = parser.parse_args()
    if any(Path(path).exists() for path in args.forbidden_path):
        raise RuntimeError('A forbidden GT/source path is visible')
    manifest, paths = verify_input(args.input)
    config = json.loads(args.config.read_text())
    assert config['method'] == 'Pi3' and config['model_repository'] == 'yyfz233/Pi3'
    assert config['inference_dtype'] == 'bfloat16'
    sys.path.insert(0, str(args.code))
    import torch
    from safetensors.torch import load_file
    from pi3.models.pi3 import Pi3
    from pi3.utils.geometry import depth_normal_edge, recover_intrinsic_from_rays_d
    from pi3_preprocess import stage_and_load_images
    torch.set_num_threads(1)
    random.seed(config['sampling_seed']); np.random.seed(config['sampling_seed'])
    torch.manual_seed(config['sampling_seed']); torch.cuda.manual_seed_all(config['sampling_seed'])
    args.output.mkdir(parents=True, exist_ok=False)
    raw = args.output/'raw'; raw.mkdir()
    write_json(args.output/'configuration.json', config)
    write_json(args.output/'input_manifest.json', manifest)
    images, _, records, original_wh, target_wh = stage_and_load_images(
        paths, args.output/'rgb-stage', pixel_limit=config['pixel_limit'])
    # The shared loader records the legacy camera transform as well. Sampling
    # back to original RGB uses actual pixel centers, not that zero-offset K scale.
    transforms = np.stack([pixel_resize((h, w), (target_wh[1], target_wh[0])) for w, h in original_wh])
    write_json(args.output/'preprocessing.json', records)
    model = Pi3(**config['model_config'])
    model.load_state_dict(load_file(str(args.model/config['model_filename']), device='cpu'), strict=True)
    model = model.to('cuda').eval()
    torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); started = time.monotonic()
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        output = model(images[None].to('cuda'))
    torch.cuda.synchronize(); seconds = time.monotonic()-started
    with torch.inference_mode():
        probability = torch.sigmoid(output['conf'][..., 0])
        confident = probability > config['confidence_probability_threshold']
        mask = confident & ~depth_normal_edge(output['local_points'], rtol=config['edge_depth_rtol'], mask=confident)
        rays = torch.nn.functional.normalize(output['local_points'], dim=-1)
        intrinsics = recover_intrinsic_from_rays_d(rays, force_center_principal_point=True)
    def numpy(value):
        return value.detach().float().cpu().numpy()
    for name in ['points', 'local_points', 'conf', 'camera_poses']:
        np.save(raw/(name+'.npy'), numpy(output[name]))
    np.save(raw/'recovered_intrinsics.npy', numpy(intrinsics))
    np.save(raw/'confidence_probability.npy', numpy(probability))
    np.save(raw/'official_mask.npy', mask.cpu().numpy())
    points = numpy(output['points'][0])
    t = np.linalg.inv(numpy(output['camera_poses'][0]).astype(np.float64))[:, :3, :].astype(np.float32)
    valid = mask[0].cpu().numpy() & np.isfinite(points).all(-1)
    arrays = {'points': points, 'valid': valid, 'intrinsics': numpy(intrinsics[0]), 'extrinsics': t,
              'confidence': numpy(probability[0]), 'original_to_native': transforms,
              'rgb': np.rint(images.permute(0, 2, 3, 1).numpy()*255).astype(np.uint8)}
    save_state(args.output/'point-state', arrays, manifest,
               {'method': 'Pi3-original-point', 'gt_conditioning': False, 'parent': None,
                'point_branch': 'original Pi3 points; author example.py confidence and depth-normal edge mask',
                'intrinsics': 'author recover_intrinsic_from_rays_d on predicted local points, centered principal point',
                'preprocessing': 'author image loader; actual half-pixel resize for RGB correspondence'})
    write_json(args.output/'result.json', {'state': 'native_prediction_complete', 'views': len(manifest),
               'model_class': type(model).__name__, 'gt_access': False,
               'forbidden_paths_absent': args.forbidden_path, 'inference_seconds': seconds,
               'native_hw': list(images.shape[-2:]), 'valid_pixels': int(valid.sum()),
               'torch_version': torch.__version__, 'gpu': torch.cuda.get_device_name(),
               'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(), 'scores_computed': False})
    print((args.output/'result.json').read_text(), flush=True)


if __name__ == '__main__':
    main()
