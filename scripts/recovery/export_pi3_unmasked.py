"""Export Pi3's saved, unfiltered predictions; never invent missing geometry."""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np

from native_state import load_state, save_state
from public_canvas import export_prediction


def read(path):
    return json.loads(Path(path).read_text())


def export(parent, output):
    parent, output = Path(parent), Path(output)
    receipt = read(parent/'result.json')
    assert receipt['state'] == 'native_prediction_complete' and receipt['model_class'] == 'Pi3'
    assert not receipt['gt_access']
    arrays, manifest, metadata = load_state(parent/'point-state')
    assert metadata['method'] == 'Pi3-original-point' and not metadata['gt_conditioning']
    raw = np.load(parent/'raw/points.npy', mmap_mode='r', allow_pickle=False)
    assert raw.shape == (1, *arrays['points'].shape)
    # Pi3's original state retained raw XYZ everywhere; only its validity mask
    # excluded example-filtered pixels. This equality rejects zero-filled data.
    np.testing.assert_array_equal(arrays['points'], raw[0])
    valid = np.isfinite(raw[0]).all(-1)
    assert np.all(valid[arrays['valid']])
    output.mkdir(parents=True, exist_ok=False)
    save_state(output/'state', dict(arrays, valid=valid), manifest,
               {'method': 'Pi3-original-point-unmasked', 'gt_conditioning': False,
                'parent': str(parent/'point-state'), 'point_branch': 'exact archived Pi3 points, finite only',
                'geometry_changed': False, 'mask_policy': 'no confidence or depth-edge removal'})
    export_prediction(output/'state', output/'public')
    report = {'state': 'export_complete', 'gt_access': False, 'model_inference_executed': False,
              'raw_xyz_exact_equal': True, 'parent_valid_pixels': int(arrays['valid'].sum()),
              'unmasked_valid_pixels': int(valid.sum()), 'views': len(manifest)}
    (output/'result.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    batch = read(args.batch)
    assert batch['completed'] and batch['method'] == 'Pi3-original-point'
    assert len(batch['cases']) == batch['expected_scenes']
    assert len({row['scene'] for row in batch['cases']}) == batch['expected_scenes']
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(args.batch, args.output/'parent-batch.json')
    report = {'method': 'Pi3-original-point-unmasked', 'expected_scenes': batch['expected_scenes'],
              'completed': False, 'scores_computed': False, 'cases': []}
    for index, row in enumerate(batch['cases']):
        record = {key: row[key] for key in ['scene', 'partition', 'input', 'views']}
        record['parent'] = row
        if row['state'] == 'execution_failed':
            record['state'] = 'upstream_prediction_failed'
        else:
            assert row['state'] in {'completed', 'reused_verified_prediction'}
            case = args.output/'cases'/f'{index:06d}'
            exported = export(row['native_prediction'], case)
            record.update(state='completed', export=exported, native_state=str(case/'state'),
                          public_prediction=str(case/'public'))
        report['cases'].append(record)
        (args.output/'batch.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(record), flush=True)
    report['completed'] = True
    report['failed_scenes'] = sum(row['state'] == 'upstream_prediction_failed' for row in report['cases'])
    (args.output/'batch.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
