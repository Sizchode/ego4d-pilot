"""Full AMB3R benchmark frontend+backend from a sealed, fixed RGB manifest."""
import argparse
import json
from pathlib import Path
import random
import sys
import time
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'recovery'))
from native_state import save_state
from run_da3_rgb import verify_input
from preprocess import prepare_rgb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--code', type=Path, required=True)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--forbidden-path', action='append', required=True)
    args = parser.parse_args()
    for forbidden in args.forbidden_path:
        if Path(forbidden).exists():
            raise RuntimeError(f'Forbidden host path visible inside inference: {forbidden}')
    manifest, image_paths = verify_input(args.input)
    sys.path.insert(0, str(args.code))
    sys.path.insert(0, str(args.code/'thirdparty'))
    import torch
    from amb3r.model import AMB3R
    random.seed(47)
    np.random.seed(47)
    torch.manual_seed(47)
    torch.cuda.manual_seed_all(47)
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    images, rgb, transforms, preprocessing = prepare_rgb(image_paths, seed=47)
    (args.output/'preprocessing.json').write_text(json.dumps(preprocessing, indent=2)+'\n')
    (args.output/'isolation.json').write_text(json.dumps({
        'forbidden_paths_absent': args.forbidden_path, 'model_input_keys': ['images'],
        'network_offline': True, 'input_files': [row['path'] for row in manifest]}, indent=2)+'\n')
    model = AMB3R().eval()
    model.load_weights(str(args.checkpoint), data_type='bf16', strict=True)
    model.cuda()
    frames = {'images': images.cuda()}
    decode = model.front_end.decode_patch_tokens_and_heads
    snapshots = []
    retained = ['world_points', 'world_points_conf', 'pts3d_by_unprojection', 'depth',
                'depth_conf', 'depth_metric', 'extrinsic', 'intrinsic', 'pose', 'median_metric_z']

    def capture_decode(*positional, **keywords):
        result = decode(*positional, **keywords)
        snapshots.append({key: result[key].detach().float().cpu().numpy()[0] for key in retained})
        return result

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.monotonic()
    with torch.inference_mode(), torch.autocast(device_type='cuda', dtype=torch.bfloat16), \
            patch.object(model.front_end, 'decode_patch_tokens_and_heads', capture_decode):
        result = model.run_amb3r_benchmark(frames)
    torch.cuda.synchronize()
    seconds = time.monotonic()-started
    if len(snapshots) != 2:
        raise RuntimeError('Expected one frontend and one backend decode from official benchmark entry')
    np.testing.assert_array_equal(result['pts3d_by_unprojection'].float().cpu().numpy()[0],
                                  snapshots[1]['pts3d_by_unprojection'])
    for name, prediction in zip(['frontend', 'backend'], snapshots):
        raw = args.output/name/'raw'
        raw.mkdir(parents=True)
        for key, value in prediction.items():
            np.save(raw/f'{key}.npy', value)
        points = prediction['pts3d_by_unprojection']
        depth = prediction['depth'][..., 0]
        confidence = prediction['depth_conf']
        if confidence.shape == depth.shape+(1,):
            confidence = confidence[..., 0]
        arrays = {'points': points, 'valid': np.isfinite(points).all(-1)&np.isfinite(depth)&(depth>0),
                  'intrinsics': prediction['intrinsic'], 'extrinsics': prediction['extrinsic'],
                  'rgb': rgb, 'original_to_native': transforms, 'confidence': confidence}
        save_state(args.output/name/'state', arrays, manifest, {
            'method': f'AMB3R-{name}-depth+camera', 'gt_conditioning': False,
            'entry': 'AMB3R.run_amb3r_benchmark', 'backend_iterations': int(name=='backend'),
            'point_branch': 'pts3d_by_unprojection; official eval_mvrecon default',
            'metric_scale': 'native coordinates; metric-depth prediction retained separately',
            'preprocessing': 'official 518x392 crop/resize and portrait rotation, RGB only'})
    (args.output/'result.json').write_text(json.dumps({
        'state': 'native_prediction_complete', 'method': 'AMB3R-backend-depth+camera',
        'views': len(manifest), 'backend_iterations': 1, 'decoder_calls': len(snapshots),
        'gt_access': False, 'scores_computed': False, 'seconds_including_raw_capture': seconds,
        'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(),
        'gpu': torch.cuda.get_device_name(), 'torch_version': torch.__version__}, indent=2)+'\n')
    print(f'Completed full AMB3R benchmark forward on {len(manifest)} fixed RGB inputs.', flush=True)


if __name__ == '__main__':
    main()
