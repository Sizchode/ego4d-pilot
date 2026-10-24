"""Run GGPT on an existing point-map state, without any feed-forward rerun.

The launcher supplies a filesystem with RGB, prediction, code and model assets
only. Full SfM outputs and the point transformer's native output are retained.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--code', type=Path, required=True)
    parser.add_argument('--state', type=Path, required=True)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--forbidden-path', action='append', required=True)
    args = parser.parse_args()
    for path in args.forbidden_path:
        if Path(path).exists():
            raise RuntimeError(f'Forbidden host path visible: {path}')
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from recovery.native_state import load_state, save_state
    from recovery.geometry_state import pixel_resize
    from recovery.run_da3_rgb import verify_input
    arrays, manifest, metadata = load_state(args.state)
    rgb_manifest, image_paths = verify_input(args.input)
    if manifest != rgb_manifest:
        raise ValueError('Prediction state and RGB are not the identical ordered inputs')
    if not arrays['valid'].all():
        raise ValueError('Upstream DemoDataset requires a complete dense initialization; invalid input recorded as failure')
    if 'confidence' not in arrays or not np.isfinite(arrays['confidence']).all():
        raise ValueError('GGPT requires finite per-pixel feed-forward confidence')
    if os.environ.get('PYTHONHASHSEED') is None:
        raise RuntimeError('Freeze PYTHONHASHSEED for upstream chunk sampling')
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.code.resolve()))
    import pycolmap  # Must precede spconv according to upstream.
    import torch
    import cv2
    from omegaconf import OmegaConf
    from hydra.utils import instantiate
    from matching import init_match_models
    import sfm.sfm_func as sfm
    from utils.basic import set_seed
    from ggpt.dataloader.demo_dataset import DemoDataset
    from utils.points import aggregate_chunks

    torch.set_num_threads(1)
    torch.set_float32_matmul_precision('highest')
    cv2.setNumThreads(1)
    # Preserve the released benchmark's matching/BA/DLT settings. Demo supplies
    # the checkpoint architecture only; no benchmark or train/val loader runs.
    benchmark = OmegaConf.load(args.code / 'configs/benchmark_sfm.yaml')
    demo = OmegaConf.load(args.code / 'configs/demo.yaml')
    cfg = OmegaConf.create({key: OmegaConf.to_container(benchmark[key], resolve=True)
                           for key in ['match_config', 'ba_config', 'dlt_config', 'common_config']})
    cfg.common_config.reduce_memory = True
    cfg.common_config.save_vis = False
    cfg.match_config.save_vis = False
    # Ceres owns its thread pool; OMP_NUM_THREADS does not constrain it.
    # A controlled replay of identical BA inputs diverged with automatic
    # threads and was bit-exact with one thread. Use the allocated CPU count.
    cfg.ba_config.solver_num_threads = 1
    if cfg.ba_config.calibrated:
        raise ValueError('Ground-truth calibration is forbidden')
    set_seed(cfg.common_config.seed)
    OmegaConf.save(cfg, args.output / 'sfm_config.yaml')
    OmegaConf.save(demo.ggptmodel_config, args.output / 'ggpt_model_config.yaml')
    n, h, w, _ = arrays['points'].shape
    match_hw = tuple(max(14, round(side * 1280 / max(h, w) / 14) * 14) for side in (h, w))
    native_to_match = pixel_resize((h, w), match_hw)
    matching_images = []
    for i, path in enumerate(image_paths):
        rgb = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        transform = native_to_match @ arrays['original_to_native'][i]
        matching_images.append(cv2.warpPerspective(rgb, transform, (match_hw[1], match_hw[0]),
                                                   flags=cv2.INTER_LINEAR))
    ff = {'points': torch.from_numpy(arrays['points']).float().cuda(),
          'points_conf': torch.from_numpy(arrays['confidence']).float().cuda(),
          'intrinsics': torch.from_numpy(arrays['intrinsics']).float().cuda(),
          'extrinsics': torch.from_numpy(arrays['extrinsics']).float().cuda(),
          'images_ff': torch.from_numpy(arrays['rgb']).float().cuda() / 255}
    images = torch.from_numpy(np.stack(matching_images)).float().cuda() / 255
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    matchers = init_match_models(cfg.match_config.models, device='cuda')
    audit = args.output / 'sfm'
    audit.mkdir()
    original_matching = sfm.match_images
    original_eigh = torch.linalg.eigh
    original_bundle_adjuster = pycolmap.create_default_bundle_adjuster

    def create_bundle_adjuster(options, configuration, reconstruction):
        options.solver_options.num_threads = cfg.ba_config.solver_num_threads
        (audit / 'ba_options.json').write_text(
            json.dumps(options.todict(), indent=2, default=str) + '\n')
        return original_bundle_adjuster(options, configuration, reconstruction)

    def capture_matching(*positional, **keywords):
        result = original_matching(*positional, **keywords)
        for key, value in result.items():
            np.save(audit / f'matching_{key}.npy', value.detach().cpu().numpy())
        return result

    def capture_eigh(matrix, *positional, **keywords):
        # Preserve the failing numerical input without changing solver choice,
        # track filtering, chunk size, precision, or the returned result.
        try:
            return original_eigh(matrix, *positional, **keywords)
        except RuntimeError as error:
            values = matrix.detach().cpu().numpy()
            np.save(audit / 'failed_eigh_input.npy', values)
            finite = np.isfinite(values)
            libraries = [line.strip() for line in Path('/proc/self/maps').read_text().splitlines()
                         if 'libcusolver' in line or 'libcublas' in line]
            (audit / 'failed_eigh.json').write_text(json.dumps({
                'shape': list(values.shape), 'dtype': str(values.dtype),
                'finite_values': int(finite.sum()), 'total_values': int(values.size),
                'maximum_absolute_finite': float(np.abs(values[finite]).max()) if finite.any() else None,
                'torch_version': torch.__version__, 'cuda_version': torch.version.cuda,
                'loaded_cuda_libraries': libraries, 'error': str(error)}, indent=2)+'\n')
            raise

    # SfM increments its input principal points in-place for COLMAP. Give it
    # owned copies so neither the parent state nor GGPT initialization changes.
    with torch.inference_mode(), patch.object(sfm, 'match_images', capture_matching), \
            patch.object(torch.linalg, 'eigh', capture_eigh), \
            patch.object(pycolmap, 'create_default_bundle_adjuster', create_bundle_adjuster):
        geo = sfm.run_sfm(images, {k: v.clone() for k, v in ff.items()}, matchers, cfg,
                          gt=None, output_dir=str(audit))
    torch.cuda.synchronize()
    sfm_seconds = time.monotonic() - started
    flags = {}
    for key, value in geo.items():
        if isinstance(value, torch.Tensor):
            np.save(audit / f'{key}.npy', value.detach().cpu().numpy())
        else:
            flags[key] = value
    (audit / 'status.json').write_text(json.dumps(flags, indent=2) + '\n')
    if not geo.get('points_success', False):
        raise RuntimeError('GGPT SfM did not produce valid geometric guidance; no replacement scene')
    del matchers, images
    torch.cuda.empty_cache()

    model = instantiate(demo.ggptmodel_config).eval().cuda()
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    model.load_state_dict({k.removeprefix('module.'): v for k, v in checkpoint.items()}, strict=True)
    dataset = DemoDataset(name='scene', ff_data=ff, geo_data=geo)
    if dataset.mode != 'demo':
        raise RuntimeError('Only the prediction-only DemoDataset is permitted')
    chunks, scene = dataset[0]
    np.save(args.output / 'aligned_initial_points.npy', scene['ff_pts'].detach().cpu().numpy())
    chunks_dir = args.output / 'chunks'
    chunks_dir.mkdir()
    all_points, all_confidence = [], []
    refine_started = time.monotonic()
    for index, chunk in enumerate(chunks):
        chunk = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in chunk.items()}
        with torch.inference_mode():
            prediction = model([chunk])
        points = dataset.unnormalize_pts(chunk, prediction['ff_pts_out'])
        all_points.append(points)
        all_confidence.append(prediction['ff_pts_conf_out'])
        np.savez_compressed(chunks_dir / f'{index:04d}.npz',
            mask=chunk['msks_in_scene'].cpu().numpy(),
            center=chunk['chunk_center'].detach().cpu().numpy(), radius=chunk['chunk_radius'],
            points=points.detach().cpu().numpy(), confidence=prediction['ff_pts_conf_out'].detach().cpu().numpy())
    if not chunks:
        raise RuntimeError('GGPT produced no inference chunks')
    points, confidence, refined = aggregate_chunks(
        torch.cat(all_points), torch.cat(all_confidence),
        torch.stack([chunk['msks_in_scene'] for chunk in chunks]).cuda(), scene)
    torch.cuda.synchronize()
    output_points = points.detach().cpu().numpy()
    # Upstream aggregation retains aligned initial points outside refined chunks.
    # Those are genuine output, while refined_mask separately describes support.
    child = dict(arrays, points=output_points,
                 valid=arrays['valid'] & np.isfinite(output_points).all(-1),
                 intrinsics=geo['intrinsics'].cpu().numpy(), extrinsics=geo['extrinsics'].cpu().numpy(),
                 confidence=confidence.detach().cpu().numpy(), refined_mask=refined.cpu().numpy())
    save_state(args.output / 'state', child, manifest, {
        'method': metadata['method'] + ' + GGPT (RoMaV2 benchmark settings, no GT)',
        'parent': str(args.state), 'gt_conditioning': False,
        'point_frame': 'SfM world coordinates after DemoDataset initialization alignment',
        'unrefined_pixels': 'upstream aligned initial point retained'})
    report = {'state': 'reconstruction_complete', 'views': n, 'chunks': len(chunks),
              'sfm_seconds_including_matching_capture': sfm_seconds,
              'refinement_seconds_including_capture': time.monotonic() - refine_started,
              'total_seconds': time.monotonic() - started, 'gt_access': False,
              'forbidden_paths_absent': args.forbidden_path, 'scoring_executed': False,
              'refined_fraction': float(refined.float().mean()),
              'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(),
              'gpu': torch.cuda.get_device_name(), 'python_hash_seed': os.environ['PYTHONHASHSEED'],
              'ba_solver_num_threads': cfg.ba_config.solver_num_threads,
              'torch_deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
              'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
              'matching_hw': list(match_hw), 'pixel_canvas_hw': [h, w]}
    (args.output / 'result.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
