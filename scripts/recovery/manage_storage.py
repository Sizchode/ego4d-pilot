"""执行主实验磁盘保留策略；默认仅预览，--apply 才删除。"""
import argparse
from datetime import datetime, timezone
import getpass
import json
from pathlib import Path
import shutil
import subprocess


MATCHING_AUDITS = ('matching_pred_matches_lr.npy', 'matching_pred_cycle_error.npy', 'matching_pred_scores.npy', 'matching_sp_scores.npy')


def read(path):
    return json.loads(Path(path).read_text())


def matching_candidates(root, batch, input_captures=False):
    (root, batch) = (Path(root).resolve(), Path(batch).resolve())
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
        (native, public) = (case / 'output/ggpt', case / 'output/ggpt-public')
        result = read(native / 'result.json')
        canvas = read(public / 'canvas.json')
        if result['state'] != 'reconstruction_complete' or result['gt_access'] or canvas['gt_access']:
            raise ValueError('Completed reconstruction and public output required')
        required = [native / 'state' / name for name in ['state.npz', 'state.json', 'input_manifest.json']]
        required += [public / view['file'] for view in canvas['views']]
        if not canvas['views'] or any((not p.is_file() or not p.stat().st_size for p in required)):
            raise ValueError('Authoritative output is incomplete; retain debug data')
        for name in MATCHING_AUDITS:
            path = native / 'sfm' / name
            if path.resolve() != path:
                raise ValueError('Never follow symlinks while pruning')
            if path.is_file():
                selected.append(path)
        if input_captures:
            captures = [native / 'aligned_initial_points.npy']
            captures += sorted((native / 'chunks').glob('[0-9][0-9][0-9][0-9].npz'))
            for path in captures:
                if path.resolve() != path:
                    raise ValueError('Never follow symlinks while pruning')
                if path.is_file():
                    selected.append(path)
    return selected


def prune_matching(root, batches, apply=False, input_captures=False):
    paths = sorted({p for batch in batches for p in matching_candidates(root, batch, input_captures)})
    summary = {'apply': apply, 'files': len(paths), 'bytes': sum((p.stat().st_size for p in paths)), 'batches': list(map(str, batches)), 'allowed_filenames': MATCHING_AUDITS, 'input_captures': input_captures}
    if apply:
        for path in paths:
            path.unlink()
    return summary


def export_candidates(root, batch):
    (root, batch) = (Path(root).resolve(), Path(batch).resolve())
    if root / 'runs' not in batch.parents:
        raise ValueError('Batch outside project')
    paths = set()
    for row in read(batch)['cases']:
        if row['state'] != 'completed' or not row.get('public_branches'):
            continue
        case = Path(row['run']).resolve()
        if case.parent != batch.parent / 'cases':
            raise ValueError('Reference runs cannot be pruned')
        output = case / 'output'
        primary = Path(row['public_prediction']).resolve()
        state = Path(row['native_state']).resolve()
        raw = output / 'prediction/raw'
        if output not in primary.parents or output not in state.parents:
            raise ValueError('Primary outputs must belong to the case')
        if not (state / 'state.npz').is_file() or not raw.is_dir() or (not any(raw.iterdir())):
            raise ValueError('Raw model prediction and primary state must exist')
        canvas = read(primary / 'canvas.json')
        if not canvas['views'] or any((not (primary / v['file']).is_file() for v in canvas['views'])):
            raise ValueError('Primary scoring export is incomplete')
        for directory in row['public_branches'].values():
            public = Path(directory)
            if public.resolve() != public or public.parent != output:
                raise ValueError('Unexpected export location or symlink')
            if public == primary or not public.exists():
                continue
            parent = Path(read(public / 'canvas.json')['parent_state'])
            native = output / parent.relative_to('/output')
            if native.resolve() != native or native.parent != output / 'prediction':
                raise ValueError('Unexpected derived state location')
            paths.add(public)
            if native != state and native.exists():
                paths.add(native)
    return paths


def prune_exports(root, batches, apply=False):
    paths = sorted({p for batch in batches for p in export_candidates(root, batch)})
    size = sum((f.stat().st_size for p in paths for f in p.rglob('*') if f.is_file()))
    summary = {'apply': apply, 'directories': len(paths), 'bytes': size, 'batches': list(map(str, batches)), 'policy': 'Retain raw model prediction, primary native state and primary public export'}
    if apply:
        for p in paths:
            shutil.rmtree(p)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy', type=Path, required=True)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    policy = json.loads(args.policy.read_text())
    root = Path(policy['root']).resolve()
    matching = [Path(p) for p in policy['matching_batches'] if Path(p).is_file()]
    exports = [Path(p) for p in policy['derived_export_batches'] if Path(p).is_file()]
    # Validate both policies before either mutates files.
    prune_matching(root, matching, input_captures=True)
    prune_exports(root, exports)
    results = [prune_matching(root, matching, args.apply, input_captures=True),
               prune_exports(root, exports, args.apply)]
    text = subprocess.check_output(['checkquota'], text=True, timeout=60)
    quota_rows = [line.split() for line in text.splitlines()
                  if line.split()[:2] == [getpass.getuser(), '/oscar/scratch']]
    if len(quota_rows) != 1:
        raise ValueError('Cannot identify the actual user scratch quota')
    row = quota_rows[0]
    quota = {'used_GB': float(row[2]), 'soft_limit_GB': float(row[4]),
             'hard_limit_GB': float(row[5]), 'state': row[9],
             'grace_remaining': ' '.join(row[10:])}
    quota['headroom_to_soft_limit_GB'] = quota['soft_limit_GB'] - quota['used_GB']
    quota['maximum_usage_GB'] = policy.get('maximum_usage_GB', quota['soft_limit_GB'])
    quota['headroom_to_usage_budget_GB'] = quota['maximum_usage_GB'] - quota['used_GB']
    quota['below_requested_headroom'] = quota['headroom_to_usage_budget_GB'] < policy['reserve_GB']
    record = {'checked_at': datetime.now(timezone.utc).isoformat(), 'apply': args.apply,
              'quota': quota, 'cleanup': results,
              'removed_logical_bytes_this_pass': sum(r['bytes'] for r in results),
              'note': 'Quota is the checkquota report; deleted logical bytes are not a substitute for actual released quota.'}
    destination = Path(policy['status_file'])
    if root not in destination.resolve().parents:
        raise ValueError('Status file must stay in the project')
    destination.write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps({'checked_at': record['checked_at'], 'quota': quota,
                      'removed_logical_GiB': record['removed_logical_bytes_this_pass'] / 1024**3}))


if __name__ == '__main__':
    main()
