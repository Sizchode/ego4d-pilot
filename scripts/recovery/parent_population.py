"""Bind recovery execution to an exact RGB population and authoritative parents."""
import hashlib
import json
from pathlib import Path
import shutil

try:
    from .run_da3_population import validate_plan, write
except ImportError:
    from run_da3_population import validate_plan, write


FAILED = {'execution_failed', 'upstream_prediction_failed', 'output_validation_failed'}
SUCCESS = {'completed', 'reused_verified_prediction'}


def read(path):
    return json.loads(Path(path).read_text())


def population_arguments(parser):
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument('--prepared', type=Path)
    selection.add_argument('--population-plan', type=Path)
    parser.add_argument('--release', type=Path)
    parser.add_argument('--rerun-accepted', action='store_true')


def load_population(args, suffix):
    batch = read(args.parent_batch)
    if args.population_plan:
        plan = read(args.population_plan)
        release = read(args.release) if args.release else None
        cases = validate_plan(plan, release, expected_method=batch['method']+suffix)
        rows = [{**r, 'scene': r['case_id'], 'partition': plan['stage']} for r in cases]
    else:
        prepared = read(args.prepared/'preparation.json')
        if not prepared['completed']:
            raise ValueError('Incomplete prepared population')
        rows = prepared['scenes']
        if any(r['exposure'] != 'development_already_analyzed' for r in rows):
            raise ValueError('Unexposed input in development execution')
    # Validate release before reading any RGB manifest or native parent state.
    count = batch.get('expected_cases', batch.get('expected_scenes'))
    if not batch['completed'] or not rows or len(rows) != count or len(batch['cases']) != count:
        raise ValueError('Completed parent batch must cover the exact population')
    parents = {r.get('case_id', r.get('scene')): r for r in batch['cases']}
    if len(parents) != len(rows) or set(parents) != {r['scene'] for r in rows}:
        raise ValueError('Parent identifiers differ from population')
    for row in rows:
        parent = parents[row['scene']]
        if parent['state'] not in SUCCESS | FAILED:
            raise ValueError('Unresolved parent execution')
        if Path(parent['input']).resolve() != Path(row['input']).resolve() or parent['views'] != row['views']:
            raise ValueError('Parent RGB identity differs from population')
        path = Path(row['input'])/'input_manifest.json'
        if (hashlib.sha256(path.read_bytes()).hexdigest() != row['input_manifest_sha256']
                or len(read(path)) != row['views']):
            raise ValueError('Input manifest or view count changed')
        if parent['state'] in SUCCESS:
            native = Path(parent['native_state'])
            if (read(native/'input_manifest.json') != read(path)
                    or read(native/'state.json')['gt_conditioning']
                    or not (native/'state.npz').is_file()):
                raise ValueError('Invalid authoritative parent state')
    return rows, batch, parents


def archive_population(args, output):
    if args.population_plan:
        shutil.copy2(args.population_plan, output/'plan.json')
        if args.release:
            shutil.copy2(args.release, output/'release.json')
    else:
        shutil.copy2(args.prepared/'inventory.json', output/'inventory.json')
    shutil.copy2(__file__, output/'parent_population.py')
    shutil.copy2(Path(__file__).with_name('run_da3_population.py'), output/'population_gate.py')


def verify_geometry(native, public, manifest):
    if (read(native/'input_manifest.json') != manifest or read(public/'input_manifest.json') != manifest
            or read(native/'state.json')['gt_conditioning'] or not (native/'state.npz').is_file()):
        raise ValueError('Output state identity or geometry interface changed')
    canvas = read(public/'canvas.json')
    if canvas['gt_access'] or len(canvas['views']) != len(manifest):
        raise ValueError('Invalid public geometry receipt')
    if any(not (public/r['file']).is_file() for r in canvas['views']):
        raise ValueError('Incomplete public point archive')
