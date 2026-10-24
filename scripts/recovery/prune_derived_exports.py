"""Prune unselected exports while retaining model raw outputs and primary geometry."""
import argparse
import json
from pathlib import Path
import shutil


def read(path):
    return json.loads(Path(path).read_text())


def candidates(root, batch):
    root, batch = Path(root).resolve(), Path(batch).resolve()
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
        if not (state / 'state.npz').is_file() or not raw.is_dir() or not any(raw.iterdir()):
            raise ValueError('Raw model prediction and primary state must exist')
        canvas = read(primary / 'canvas.json')
        if not canvas['views'] or any(not (primary / v['file']).is_file() for v in canvas['views']):
            raise ValueError('Primary scoring export is incomplete')
        for directory in row['public_branches'].values():
            public = Path(directory)
            if public.resolve() != public or public.parent != output:
                raise ValueError('Unexpected export location or symlink')
            if public == primary or not public.exists():
                continue
            parent = Path(read(public / 'canvas.json')['parent_state'])
            # public_canvas records the isolated worker's /output mount.
            native = output / parent.relative_to('/output')
            if native.resolve() != native or native.parent != output / 'prediction':
                raise ValueError('Unexpected derived state location')
            paths.add(public)
            if native != state and native.exists():
                paths.add(native)
    return paths


def prune(root, batches, apply=False):
    paths = sorted({p for batch in batches for p in candidates(root, batch)})
    size = sum(f.stat().st_size for p in paths for f in p.rglob('*') if f.is_file())
    summary = {'apply': apply, 'directories': len(paths), 'bytes': size,
               'batches': list(map(str, batches)),
               'policy': 'Retain raw model prediction, primary native state and primary public export'}
    if apply:
        for p in paths:
            shutil.rmtree(p)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--batch', type=Path, action='append', required=True)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    print(json.dumps(prune(args.root, args.batch, args.apply)))


if __name__ == '__main__':
    main()
