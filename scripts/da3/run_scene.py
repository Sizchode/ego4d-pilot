"""Run one frozen DA3 scene with observational diagnostics and strict failure states.

No reconstruction repair is performed here. The pinned upstream evaluator owns
sampling, fusion and scores. Instrumentation retains its intermediate quantities.
The upstream random-point fallback for invalid meshes is replaced by an exception.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import threading
import time
import traceback
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

try:
    from .geometry import pair_diagnostics, unproject
except ImportError:  # Preserve direct execution as ``python run_scene.py``.
    from geometry import pair_diagnostics, unproject


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def npz(path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


class AuditedAPI:
    def __init__(self, api, root, cfg):
        self.api, self.root, self.cfg = api, root, cfg
        self.prediction = None

    def inference(self, image_files, *, export_dir, export_format, ref_view_strategy):
        raw = self.root / 'raw'
        raw.mkdir()
        manifest = []
        for i, filename in enumerate(image_files):
            with Image.open(filename) as image:
                width, height = image.size
            manifest.append({'index': i, 'path': str(Path(filename).resolve()),
                             'sha256': sha256(filename), 'original_hw': [height, width]})
        dump(self.root / 'input_manifest.json', manifest)

        forward_start = torch.cuda.Event(enable_timing=True)
        forward_end = torch.cuda.Event(enable_timing=True)

        def start_network(module, inputs):
            forward_start.record()

        def save_network_output(module, inputs, output):
            forward_end.record()
            network = raw / 'network'
            network.mkdir()
            for key, tensor in output.items():
                if isinstance(tensor, torch.Tensor):
                    np.save(network / (key + '.npy'), tensor.detach().float().cpu().numpy())

        pre_hook = self.api.model.register_forward_pre_hook(start_network)
        hook = self.api.model.register_forward_hook(save_network_output)
        started = time.monotonic()
        try:
            pred = self.api.inference(
                image_files, export_dir=None, ref_view_strategy=ref_view_strategy,
                process_res=self.cfg['process_res'],
                process_res_method=self.cfg['process_res_method'])
        finally:
            pre_hook.remove()
            hook.remove()
        torch.cuda.synchronize()
        dump(self.root / 'inference_timing.json', {
            'seconds_including_raw_capture': time.monotonic() - started,
            'network_cuda_seconds_excluding_capture': forward_start.elapsed_time(forward_end) / 1000})
        self.prediction = pred
        for name in ['depth', 'conf', 'extrinsics', 'intrinsics', 'sky', 'processed_images']:
            value = getattr(pred, name)
            if value is not None:
                np.save(raw / (name + '.npy'), value)
        np.save(raw / 'valid_depth.npy', np.isfinite(pred.depth) & (pred.depth > 0))

        # Same numerical contents as upstream mini_npz, but synchronous to avoid
        # racing an asynchronous export against the evaluator's file readiness test.
        npz(Path(export_dir) / 'exports/mini_npz/results.npz',
            depth=np.round(pred.depth, 8), conf=np.round(pred.conf, 2),
            extrinsics=pred.extrinsics, intrinsics=pred.intrinsics)

        # Identity intrinsics track the exact image resize/crop homography. They
        # are passed only through the image processor, never into the model.
        identity = np.repeat(np.eye(3, dtype=np.float32)[None], len(image_files), axis=0)
        processed, _, transform = self.api.input_processor(
            image_files, intrinsics=identity, process_res=self.cfg['process_res'],
            process_res_method=self.cfg['process_res_method'], num_workers=1, sequential=True)
        np.save(raw / 'original_to_processed_pixel.npy', transform.numpy())
        dump(raw / 'conventions.json', {
            'extrinsics': 'OpenCV world-to-camera; X right, Y down, Z forward',
            'depth': 'camera Z; model-native scale, not assumed metric',
            'points': 'world coordinates in model-native scale; view index + pixel row/column retained',
            'image_hw': list(pred.depth.shape[-2:]),
            'preprocessing_tensor_shape': list(processed.shape),
            'confidence': 'unrounded model-native score; not a calibrated probability',
            'matching_or_tracks': 'not produced by this baseline',
            'gt_conditioning': False})
        point_dir = raw / 'world_points'
        point_dir.mkdir()
        for i, depth in enumerate(pred.depth):
            npz(point_dir / f'{i:06d}.npz',
                xyz=unproject(depth, pred.intrinsics[i], pred.extrinsics[i]).astype(np.float32),
                rgb=pred.processed_images[i], valid=np.isfinite(depth) & (depth > 0))
        return pred


class EvaluationCapture:
    """Observe exact evaluation calls, without changing valid outputs or formulas."""
    def __init__(self, directory, code):
        self.directory, self.code = directory, str(code.resolve())
        self.counts = {}
        self.handlers = {'_prep_unposed': self.prepared, 'align_poses_umeyama': self.alignment,
                         'nn_correspondance': self.neighbors,
                         '_evaluate_reconstruction': self.dtu_distances,
                         'fuse_depth_to_tsdf': self.mesh}

    def path(self, name, suffix='.npz'):
        index = self.counts.get(name, 0)
        self.counts[name] = index + 1
        return self.directory / f'{name}_{index:03d}{suffix}'

    def profile(self, frame, event, result):
        if event != 'return' or result is None:
            return
        callback = self.handlers.get(frame.f_code.co_name)
        if callback is not None and frame.f_code.co_filename.startswith(self.code + os.sep):
            callback(frame.f_locals, result)

    def prepared(self, values, result):
        depth, intrinsics, extrinsics = result
        npz(self.path('fusion_inputs'), depth=depth, intrinsics=intrinsics, extrinsics=extrinsics,
            retained_mask=np.isfinite(depth) & (depth > 0))

    def alignment(self, values, result):
        if len(result) == 4:
            rot, trans, scale, aligned = result
            npz(self.path('alignment'), rotation=rot, translation=trans, scale=scale,
                aligned_world_to_camera=aligned,
                estimated_world_to_camera=values['ext_est'], reference_world_to_camera=values['ext_ref'])

    def neighbors(self, values, result):
        npz(self.path('surface_distance'), reference=values['verts1'], query=values['verts2'], distance=result)

    def dtu_distances(self, values, result):
        npz(self.path('dtu_surface_distance'), accuracy_points=values['data_in_obs'],
            accuracy_distance=values['dist_d2s'], completeness_points=values['stl_above'],
            completeness_distance=values['dist_s2d'], max_distance=values['max_dist'],
            accuracy_included=values['dist_d2s'] < values['max_dist'],
            completeness_included=values['dist_s2d'] < values['max_dist'])

    def mesh(self, values, result):
        import open3d as o3d
        self.directory.mkdir(exist_ok=True, parents=True)
        if not o3d.io.write_triangle_mesh(str(self.path('fused_mesh', '.ply')), result):
            raise IOError('Failed to retain fused mesh')

    @contextmanager
    def active(self, dataset):
        import open3d as o3d
        module = sys.modules[type(dataset).__module__]

        def strict_sample(mesh, num_points=1000000):
            # Same success path as upstream, but never replace invalid meshes
            # with fabricated random points.
            pcd = mesh.sample_points_uniformly(number_of_points=num_points)
            if pcd.has_colors():
                pcd.colors = o3d.utility.Vector3dVector(np.clip(np.asarray(pcd.colors), 0., 1.))
            return pcd

        from contextlib import ExitStack
        with ExitStack() as stack:
            if hasattr(module, 'sample_points_from_mesh'):
                stack.enter_context(patch.object(module, 'sample_points_from_mesh', strict_sample))
            self.directory.mkdir(parents=True, exist_ok=True)
            previous = sys.getprofile()
            previous_threads = threading.getprofile()
            threading.setprofile(self.profile)
            sys.setprofile(self.profile)
            try:
                yield
            finally:
                sys.setprofile(previous)
                threading.setprofile(previous_threads)
                dump(self.directory / 'capture_counts.json', self.counts)


def save_pose_errors(prediction, gt, directory):
    from depth_anything_3.bench.utils import align_to_first_camera, build_pair_index, se3_to_relative_pose_error
    from depth_anything_3.utils.geometry import as_homogeneous
    predicted = align_to_first_camera(torch.from_numpy(as_homogeneous(prediction.extrinsics)))
    reference = align_to_first_camera(torch.from_numpy(as_homogeneous(gt.extrinsics)))
    rotation, translation = se3_to_relative_pose_error(predicted, reference, len(predicted))
    a, b = build_pair_index(len(predicted))
    npz(directory / 'camera_pairs.npz', index_a=a.numpy(), index_b=b.numpy(),
        rotation_degrees=rotation.numpy(), translation_degrees=translation.numpy())


def save_diagnostics(pred, directory):
    directory.mkdir(parents=True)
    pairs = [pair for i in range(len(pred.depth) - 1) for pair in [(i, i + 1), (i + 1, i)]]
    dump(directory / 'protocol.json', {
        'pairs': pairs, 'selection': 'adjacent input views in both directions, fixed before inference',
        'visibility': 'projectable mask only; signed depth ordering retained, not a visibility ground truth',
        'threshold': None, 'uses_ground_truth': False})
    for a, b in pairs:
        result = pair_diagnostics(pred.depth[a], pred.depth[b], pred.intrinsics[a], pred.intrinsics[b],
                                  pred.extrinsics[a], pred.extrinsics[b])
        npz(directory / f'{a:06d}_{b:06d}.npz', **result)


def save_previews(pred, root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    indices = np.unique(np.linspace(0, len(pred.depth) - 1, min(6, len(pred.depth)), dtype=int))
    fig, axes = plt.subplots(len(indices), 3, figsize=(12, 3 * len(indices)), squeeze=False)
    for row, i in enumerate(indices):
        axes[row, 0].imshow(pred.processed_images[i])
        axes[row, 0].set_title(f'Input {i}')
        for col, name in [(1, 'depth'), (2, 'conf')]:
            values = getattr(pred, name)[i]
            finite = values[np.isfinite(values)]
            norm = Normalize(vmin=float(finite.min()), vmax=float(finite.max()))
            im = axes[row, col].imshow(values, norm=norm, cmap='viridis')
            axes[row, col].set_title(name + ' (native scale)')
            fig.colorbar(im, ax=axes[row, col], fraction=.04)
        for ax in axes[row]:
            ax.axis('off')
    fig.tight_layout()
    fig.savefig(root / 'preview.png', dpi=100)
    plt.close(fig)
    pairs = np.load(root / 'evaluation_capture/camera_pairs.npz')
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, key in zip(axes, ['rotation_degrees', 'translation_degrees']):
        matrix = np.full((len(pred.depth), len(pred.depth)), np.nan)
        matrix[pairs['index_a'], pairs['index_b']] = pairs[key]
        matrix[pairs['index_b'], pairs['index_a']] = pairs[key]
        im = ax.imshow(matrix, cmap='magma', vmin=0)
        ax.set_title(key)
        ax.set_xlabel('Input index')
        ax.set_ylabel('Input index')
        fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(root / 'camera_errors.png', dpi=120)
    plt.close(fig)
    # Each archived surface-distance call stores the exact points used by the
    # official metric, after its observation mask/crop/downsampling.
    import open3d as o3d
    for path in sorted((root / 'evaluation_capture').glob('surface_distance_*.npz')):
        with np.load(path) as values:
            distance = values['distance']
            cloud = o3d.geometry.PointCloud()
            cloud.points = o3d.utility.Vector3dVector(values['query'])
            norm = Normalize(vmin=0., vmax=float(distance.max()))
            cloud.colors = o3d.utility.Vector3dVector(plt.get_cmap('magma')(norm(distance))[:, :3])
            if not o3d.io.write_point_cloud(str(path.with_suffix('.ply')), cloud):
                raise IOError(f'Failed to save surface-error colors: {path}')
            dump(path.with_suffix('.colors.json'), {
                'quantity': 'nearest-reference distance at each query point',
                'color_map': 'magma', 'vmin': 0., 'vmax': float(distance.max()),
                'call_order': 'accuracy (prediction to GT), then completeness (GT to prediction)'})


def validate(root, dataset, scene, modes):
    raw = root / 'raw'
    depth = np.load(raw / 'depth.npy', mmap_mode='r')
    ext = np.load(raw / 'extrinsics.npy')
    intrinsics = np.load(raw / 'intrinsics.npy')
    manifest = json.loads((root / 'input_manifest.json').read_text())
    if len(depth) != len(manifest) or len(ext) != len(depth):
        raise ValueError('Image/prediction counts differ')
    if not np.isfinite(ext).all() or not np.isfinite(intrinsics).all():
        raise ValueError('Non-finite camera prediction')
    np.testing.assert_allclose(np.linalg.det(ext[:, :3, :3]), 1., atol=1e-3)
    expected = root / 'evaluation/model_results' / dataset / scene / 'unposed/exports/mini_npz/results.npz'
    with np.load(expected) as exported:
        np.testing.assert_array_equal(exported['depth'], np.round(depth, 8))
        np.testing.assert_array_equal(exported['conf'], np.round(np.load(raw / 'conf.npy'), 2))
    if len(list((raw / 'world_points').glob('*.npz'))) != len(depth):
        raise ValueError('Incomplete per-view point archive')
    if len(list((root / 'diagnostics').glob('*.npz'))) != 2 * (len(depth) - 1):
        raise ValueError('Incomplete cross-view archive')
    metrics = {}
    for mode in modes:
        values = json.loads((root / 'evaluation/metric_results' / f'{dataset}_{mode}.json').read_text())
        metrics[mode] = values[scene]
        if not all(np.isfinite(x) for x in metrics[mode].values()):
            raise ValueError(f'Invalid {mode} score')
    if 'recon_unposed' in modes:
        counts = json.loads((root / 'evaluation_capture/capture_counts.json').read_text())
        if counts.get('fusion_inputs', 0) < 1:
            raise ValueError('Missing fusion input capture')
        if counts.get('surface_distance', 0) < 2 and counts.get('dtu_surface_distance', 0) < 1:
            raise ValueError('Missing two-sided surface error evidence')
        if counts.get('alignment', 0) < 1:
            raise ValueError('Missing evaluation alignment')
    required = [root / 'evaluation_capture/camera_pairs.npz', root / 'preview.png', root / 'camera_errors.png',
                raw / 'original_to_processed_pixel.npy', root / 'runtime.json']
    for path in required:
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f'Missing artifact: {path}')
    dump(root / 'validation.json', {'passed': True, 'views': len(depth), 'shape': list(depth.shape),
                                   'metrics': metrics, 'export_values_equal_upstream': True})
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--dataset', required=True, choices=['hiroom', 'eth3d', 'scannetpp', '7scenes', 'dtu', 'dtu64'])
    parser.add_argument('--scene', required=True, help='Exact scene ID, or @first for the first official scene')
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    root = args.root / 'runs' / args.run_id
    root.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    stage = 'initialization'
    try:
        cfg = json.loads(args.config.read_text())
        code = args.root / 'code/Depth-Anything-3'
        revision = subprocess.check_output(['git', '-C', str(code), 'rev-parse', 'HEAD'], text=True).strip()
        if revision != cfg['code_revision']:
            raise ValueError('Code revision differs from pinned configuration')
        subprocess.run(['git', '-C', str(code), 'diff', '--exit-code', 'HEAD'], check=True)
        os.chdir(code)
        from depth_anything_3.api import DepthAnything3
        from depth_anything_3.bench.evaluator import Evaluator
        from depth_anything_3.bench.registries import MV_REGISTRY
        import open3d as o3d
        random.seed(cfg['sampling_seed'])
        np.random.seed(cfg['sampling_seed'])
        torch.manual_seed(cfg['sampling_seed'])
        o3d.utility.random.seed(cfg['sampling_seed'])
        torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', '1')))
        dataset = MV_REGISTRY.get(args.dataset)()
        scene = dataset.SCENES[0] if args.scene == '@first' else args.scene
        if scene not in dataset.SCENES:
            raise ValueError(f'Scene not in official manifest: {scene}')
        protocol_modes = {'dtu': ['recon_unposed'], 'dtu64': ['pose']}
        modes = protocol_modes.get(args.dataset, ['pose', 'recon_unposed'])
        evaluator = Evaluator(work_dir=str(root / 'evaluation'), datas=[args.dataset], modes=modes,
                              scenes=[scene], max_frames=cfg['max_frames'],
                              ref_view_strategy=cfg['ref_view_strategy'], num_fusion_workers=1)
        dump(root / 'run_config.json', {'dataset': args.dataset, 'scene': scene, 'modes': modes,
                                      'config': cfg, 'gpu': torch.cuda.get_device_name(),
                                      'dataset_parameters': {
                                          key: value for key, value in vars(type(dataset)).items()
                                          if not key.startswith('_') and isinstance(value, (str, int, float, bool))},
                                      'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
                                      'scratch_only': True, 'has_agent': False})
        for name in ['assets.json', 'environment.txt']:
            shutil.copy2(args.root / 'provenance' / name, root / name)
        dump(root / 'runner_sources.json', {p.name: sha256(p) for p in Path(__file__).parent.glob('*.py')})
        shutil.copytree(Path(__file__).parent, root / 'runner_source', ignore=shutil.ignore_patterns('__pycache__'))
        dump(root / 'instrumentation.json', {
            'model_changes': None, 'synchronous_export': 'upstream mini_npz rounding retained',
            'failure_change': 'invalid mesh raises; never generates replacement random points',
            'evaluation_random_seed': cfg['sampling_seed'],
            'gt_metadata': 'evaluator-owned files under evaluation; never passed to model',
            'additional_diagnostics': 'observational only; not used to alter predictions'})
        stage = 'model_load'
        torch.cuda.reset_peak_memory_stats()
        model = DepthAnything3.from_pretrained(str(args.root / 'models/DA3-GIANT')).to('cuda').eval()
        adapter = AuditedAPI(model, root, cfg)
        stage = 'inference'
        evaluator.infer(adapter)
        torch.cuda.synchronize()
        inference_end = time.monotonic()
        stage = 'evaluation'
        capture = EvaluationCapture(root / 'evaluation_capture', code)
        with capture.active(dataset):
            evaluator.eval()
        # Retain camera-pair errors for all inputs, even when only reconstruction
        # is scored by the dataset's official protocol (DTU-49).
        gt = evaluator._sample_frames(dataset.get_data(scene), scene)
        save_pose_errors(adapter.prediction, gt, root / 'evaluation_capture')
        evaluation_end = time.monotonic()
        stage = 'diagnostics'
        save_diagnostics(adapter.prediction, root / 'diagnostics')
        save_previews(adapter.prediction, root)
        dump(root / 'runtime.json', {'seconds_until_inference_complete': inference_end - started,
                                    'evaluation_seconds': evaluation_end - inference_end,
                                    'diagnostics_seconds': time.monotonic() - evaluation_end,
                                    'total_seconds': time.monotonic() - started,
                                    'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(),
                                    'peak_cuda_reserved_bytes': torch.cuda.max_memory_reserved(),
                                    'gpu': torch.cuda.get_device_name()})
        stage = 'artifact_validation'
        metrics = validate(root, args.dataset, scene, modes)
        dump(root / 'status.json', {'state': 'completed', 'dataset': args.dataset, 'scene': scene,
                                   'elapsed_seconds': time.monotonic() - started})
        print(json.dumps({'run': str(root), 'validated': True, 'metrics': metrics}), flush=True)
    except Exception as exc:
        dump(root / 'status.json', {'state': 'failed', 'stage': stage, 'type': type(exc).__name__,
                                   'message': str(exc), 'elapsed_seconds': time.monotonic() - started})
        (root / 'traceback.txt').write_text(traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
