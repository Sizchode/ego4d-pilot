"""Remove write-only matching debug arrays after a GGPT case finishes.

Keep authoritative predictions, camera/SfM results, inputs, configuration and cost.
The four arrays are saved by capture_matching solely for debugging; they are not
read by the reconstruction, scoring or state-replay paths.
"""
import argparse
import json
from pathlib import Path


MATCHING_AUDITS = (
    'matching_pred_matches_lr.npy',
    'matching_pred_cycle_error.npy',
    'matching_pred_scores.npy',
    'matching_sp_scores.npy',
)


def read(path):
    return json.loads(Path(path).read_text())


def candidates(root, batch, input_captures=False):
    root, batch = Path(root).resolve(), Path(batch).resolve()
    if root / 'runs' not in batch.parents:
        raise ValueError('Batch must be within this project runs directory')
    report = read(batch)
    if 'GGPT' not in report['method'].upper():
        raise ValueError('Only GGPT matching audits are covered by this policy')
    selected = []
    for row in report['cases']:
        if row['state'] != 'completed':
            continue
        case = Path(row['run']).resolve()
        if case.parent != batch.parent / 'cases':
            raise ValueError('Case must belong to this population; references are protected')
        native, public = case / 'output/ggpt', case / 'output/ggpt-public'
        result = read(native / 'result.json')
        canvas = read(public / 'canvas.json')
        if result['state'] != 'reconstruction_complete' or result['gt_access'] or canvas['gt_access']:
            raise ValueError('Completed reconstruction and public output required')
        required = [native / 'state' / name for name in
                    ['state.npz', 'state.json', 'input_manifest.json']]
        required += [public / view['file'] for view in canvas['views']]
        if not canvas['views'] or any(not p.is_file() or not p.stat().st_size for p in required):
            raise ValueError('Authoritative output is incomplete; retain debug data')
        for name in MATCHING_AUDITS:
            path = native / 'sfm' / name
            if path.resolve() != path:
                raise ValueError('Never follow symlinks while pruning')
            if path.is_file():
                selected.append(path)
        if input_captures:
            # Replay references are outside population/cases and cannot enter here.
            # These copies capture NN inputs; final state and SfM outputs stay intact.
            captures = [native / 'aligned_initial_points.npy']
            captures += sorted((native / 'chunks').glob('[0-9][0-9][0-9][0-9].npz'))
            for path in captures:
                if path.resolve() != path:
                    raise ValueError('Never follow symlinks while pruning')
                if path.is_file():
                    selected.append(path)
    return selected


def prune(root, batches, apply=False, input_captures=False):
    # Validate every population before deleting anything.
    paths = sorted({p for batch in batches for p in candidates(root, batch, input_captures)})
    summary = {'apply': apply, 'files': len(paths),
               'bytes': sum(p.stat().st_size for p in paths),
               'batches': list(map(str, batches)), 'allowed_filenames': MATCHING_AUDITS,
               'input_captures': input_captures}
    if apply:
        for path in paths:
            path.unlink()
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--batch', type=Path, action='append', required=True)
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--input-captures', action='store_true')
    args = parser.parse_args()
    print(json.dumps(prune(args.root, args.batch, args.apply, args.input_captures)))


if __name__ == '__main__':
    main()
