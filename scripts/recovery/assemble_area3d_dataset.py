"""Join official AREA3D NeRF exports without resampling or changing cameras.

This is dataset plumbing, not a reconstruction or view-selection method.
PGSR's NeRF loader needs transforms_train/test.json and extensionless image paths.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image


CAMERA_KEYS = ('fl_x', 'fl_y', 'cx', 'cy', 'w', 'h', 'camera_angle_x')


def read_export(directory):
    directory = Path(directory).resolve()
    meta = json.loads((directory / 'transforms.json').read_text())
    camera = {k: float(meta[k]) for k in CAMERA_KEYS}
    if not all(math.isfinite(v) for v in camera.values()):
        raise ValueError('Nonfinite camera intrinsics')
    w, h = camera['w'], camera['h']
    if w <= 0 or h <= 0 or w != int(w) or h != int(h) or camera['fl_x'] <= 0:
        raise ValueError('Invalid image size or focal length')
    # PGSR maps NDC to integer pixel centers via ((v+1)*size-1)/2.
    # Thus its centered rasterizer has principal point (size-1)/2, not size/2.
    expected = [camera['fl_x'], (w - 1) / 2, (h - 1) / 2,
                2 * math.atan(w / (2 * camera['fl_x']))]
    actual = [camera['fl_y'], camera['cx'], camera['cy'], camera['camera_angle_x']]
    if not np.allclose(actual, expected, rtol=1e-6, atol=1e-6):
        raise ValueError('Intrinsics are not representable by the PGSR NeRF loader')
    records = []
    for frame in meta['frames']:
        image = (directory / frame['file_path']).resolve(strict=True)
        if image.suffix.lower() != '.png':
            raise ValueError('PGSR NeRF image extension must be PNG')
        with Image.open(image) as im:
            if im.size != (int(w), int(h)):
                raise ValueError('Image size does not match camera metadata')
        pose = np.asarray(frame['transform_matrix'], dtype=float)
        if pose.shape != (4, 4) or not np.isfinite(pose).all():
            raise ValueError('Invalid camera pose')
        if not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-6):
            raise ValueError('Invalid homogeneous camera pose')
        rot = pose[:3, :3]
        if not np.allclose(rot.T @ rot, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(rot), 1, atol=1e-5):
            raise ValueError('Camera pose must be a proper rigid transform')
        records.append({'image': image, 'pose': pose, 'source_frame': frame['file_path']})
    if not records:
        raise ValueError('Empty exported split')
    return directory, camera, records


def assemble(initial, acquired, test, output, initial_count, acquired_count):
    sources = {role: read_export(p) for role, p in
               [('initial', initial), ('acquired', acquired), ('test', test)]}
    if len(sources['initial'][2]) != initial_count or len(sources['acquired'][2]) != acquired_count:
        raise ValueError('Actual view counts do not match the requested acquisition budget')
    reference = sources['initial'][1]
    for _, camera, _ in sources.values():
        if not np.allclose(list(camera.values()), list(reference.values()), rtol=1e-6, atol=1e-6):
            raise ValueError('Camera intrinsics differ across exports')
    train = sources['initial'][2] + sources['acquired'][2]
    heldout = sources['test'][2]
    train_paths = {r['image'] for r in train}
    if len(train_paths) != len(train):
        raise ValueError('Repeated training image source')
    for row in heldout:
        if row['image'] in train_paths or any(np.allclose(row['pose'], t['pose'], rtol=0, atol=1e-6) for t in train):
            raise ValueError('Training/test image or camera overlap')
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('Output already exists; do not overwrite an earlier dataset')
    # All inputs and splits validated before any output is created.
    output.mkdir(parents=True)
    frames = {'train': [], 'test': []}
    provenance = []
    for role, (source, _, rows) in sources.items():
        (output / role).mkdir()
        split = 'test' if role == 'test' else 'train'
        for index, row in enumerate(rows):
            stem = f'{role}/{role}_{index:06d}'
            (output / (stem + '.png')).symlink_to(row['image'])
            frames[split].append({'file_path': stem, 'transform_matrix': row['pose'].tolist()})
            provenance.append({'split': split, 'role': role, 'image': stem + '.png',
                               'source': str(source), 'source_frame': row['source_frame']})
    for split, rows in frames.items():
        (output / f'transforms_{split}.json').write_text(json.dumps({**reference, 'frames': rows}, indent=2)+'\n')
    report = {'source_exports': {r: str(v[0]) for r, v in sources.items()},
              'view_counts': {r: len(v[2]) for r, v in sources.items()},
              'camera_convention': 'Original AREA3D OpenGL camera-to-world, unchanged',
              'images': 'Symlinks to source RGB; no resampling or duplicate bytes',
              'pgsr_requirement': 'Use --eval; centered integer-pixel convention checked against upstream rasterizer.',
              'records': provenance}
    (output / 'assembly.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('initial', 'acquired', 'test', 'output'):
        parser.add_argument('--'+name, required=True, type=Path)
    parser.add_argument('--initial-count', required=True, type=int)
    parser.add_argument('--acquired-count', required=True, type=int)
    args = parser.parse_args()
    report = assemble(**vars(args))
    print(json.dumps({'output': str(args.output), 'view_counts': report['view_counts']}))


if __name__ == '__main__':
    main()
