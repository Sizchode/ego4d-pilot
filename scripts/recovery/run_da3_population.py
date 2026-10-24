"""Run an explicit RGB population through the accepted DA3 worker and sandbox.

The acceptance scene need not belong to the population. Execution always uses a
new output directory: interrupted work is retained, never silently relaunched.
A formal population requires a separate release bound to this exact plan.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

try:
    from .run_strong_development_set import command_for
except ImportError:
    from run_strong_development_set import command_for


WORKERS = ('run_da3_rgb.py', 'native_state.py', 'geometry_state.py', 'public_canvas.py')


def read(path):
    return json.loads(Path(path).read_text())


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def validate_plan(plan, release=None, expected_method="DA3-GIANT"):
    if plan['schema'] != 'rgb-population-plan-v1' or plan['method'] != expected_method:
        raise ValueError('Population method differs from the selected recipe')
    cases = plan['cases']
    if not cases or len({r['case_id'] for r in cases}) != len(cases):
        raise ValueError('Empty or duplicate population')
    inputs = [str(Path(r['input']).resolve()) for r in cases]
    if len(set(inputs)) != len(inputs):
        raise ValueError('Repeated input in population')
    for row in cases:
        if (not Path(row['input']).is_absolute() or row['views'] != 16
                or len(row['input_manifest_sha256']) != 64):
            raise ValueError('Exact 16-view RGB input identity required')
    if plan['stage'] == 'development':
        if any(r['exposure'] != 'development_already_analyzed' for r in cases):
            raise ValueError('Unexposed input in development execution')
    elif plan['stage'] == 'baseline_evaluation':
        if (release is None or release.get('baseline_evaluation_released') is not True
                or release.get('plan_sha256') != identity(plan)
                or release.get('scope') != 'baseline_only'
                or release.get('agent_execution_released') is not False):
            raise ValueError('Baseline evaluation needs an exact-plan baseline-only release')
    elif plan['stage'] == 'formal':
        if (release is None or release.get('formal_test_released') is not True
                or release.get('plan_sha256') != identity(plan)
                or release.get('protocol_stage') != 'formal_frozen'
                or release.get('live_controller_verified') is not True):
            raise ValueError('Formal execution needs an exact-plan frozen protocol release')
    else:
        raise ValueError('Unknown population stage')
    return cases


def recipe(root, configuration):
    code = root/'code/Depth-Anything-3'
    return {'code': code/'src', 'python': root/'envs/da3/bin/python',
            'mounts': [(root/'envs/da3', root/'envs/da3'),
                       (root/'envs/python', root/'envs/python'),
                       (code/'src', code/'src'), (root/'models/DA3-GIANT', '/model'),
                       (configuration, '/configuration.json')], 'environment': {}}


SCRIPT = '''set -euo pipefail
python=$1
root=$3
"$python" /runner/scripts/run_da3_rgb.py --input /input --output /output/initial --model /model --config /configuration.json --forbidden-path "$root/datasets" --forbidden-path "$root/runs" --forbidden-path "$root/provenance"
"$python" /runner/scripts/native_state.py --da3-run /output/initial --output /output/state
"$python" /runner/scripts/public_canvas.py --state /output/state --output /output/public
'''


def verify_output(output, manifest, config):
    result = read(output/'initial/result.json')
    if (result['state'] != 'native_prediction_complete' or result['gt_access']
            or result['input_camera_conditioning'] or result['views'] != len(manifest)
            or read(output/'initial/configuration.json') != config):
        raise ValueError('Worker receipt disagrees with RGB-only recipe')
    for part in ['initial', 'state', 'public']:
        if read(output/part/'input_manifest.json') != manifest:
            raise ValueError('Output input identity changed')
    state = read(output/'state/state.json')
    canvas = read(output/'public/canvas.json')
    if state['gt_conditioning'] or canvas['gt_access'] or len(canvas['views']) != len(manifest):
        raise ValueError('State/public output violates geometry interface')
    for row in canvas['views']:
        if not (output/'public'/row['file']).is_file():
            raise ValueError('Incomplete public point archive')
    if not (output/'state/state.npz').is_file():
        raise ValueError('Missing authoritative native state')
    return result


def execute(plan, root, accepted_batch, output, release=None):
    cases = validate_plan(plan, release)  # before reading inputs or making outputs
    root, accepted_batch, output = map(lambda p: Path(p).resolve(), (root, accepted_batch, output))
    runner = root/'code/runner'
    config = read(runner/'configs/da3_pinned.json')
    accepted = read(accepted_batch/'batch.json')
    if (not accepted['completed'] or accepted['failed_scenes'] != 0
            or accepted['method'] != 'DA3-GIANT'
            or read(accepted_batch/'configuration.json') != config):
        raise ValueError('Completed same-configuration acceptance batch required')
    code = root/'code/Depth-Anything-3'
    revision = subprocess.check_output(['git', '-C', str(code), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != config['code_revision']:
        raise ValueError('Model code revision changed')
    subprocess.run(['git', '-C', str(code), 'diff', '--exit-code', 'HEAD'], check=True)
    for name in WORKERS:
        if (runner/'scripts/recovery'/name).read_bytes() != (accepted_batch/'runner-source'/name).read_bytes():
            raise ValueError('Inference worker differs from accepted batch: '+name)
    manifests = []
    for row in cases:
        path = Path(row['input'])/'input_manifest.json'
        if hashlib.sha256(path.read_bytes()).hexdigest() != row['input_manifest_sha256']:
            raise ValueError('Input manifest changed')
        manifest = read(path)
        if len(manifest) != row['views']:
            raise ValueError('View count changed')
        manifests.append(manifest)
    output.mkdir(parents=True, exist_ok=False)
    source = output/'runner-source'; source.mkdir()
    for name in WORKERS:
        shutil.copy2(runner/'scripts/recovery'/name, source/name)
    shutil.copy2(__file__, output/'orchestrator.py')
    shutil.copy2(Path(__file__).with_name('run_strong_development_set.py'), output/'command_builder.py')
    write(output/'configuration.json', config)
    write(output/'plan.json', plan)
    if release is not None:
        write(output/'release.json', release)
    report = {'schema': 'rgb-baseline-batch-v1', 'method': 'DA3-GIANT',
              'plan_sha256': identity(plan), 'stage': plan['stage'],
              'expected_cases': len(cases), 'completed': False, 'scores_computed': False, 'cases': []}
    write(output/'batch.json', report)
    for index, (row, manifest) in enumerate(zip(cases, manifests)):
        case = output/'cases'/f'{index:06d}'; destination = case/'output'
        destination.mkdir(parents=True)
        command = command_for(recipe(root, output/'configuration.json'), root, source,
                              row['input'], destination)
        write(case/'command.json', command)
        (case/'worker.sh').write_text(SCRIPT)
        record = {**row, 'state': 'launch_intent', 'run': str(case)}
        report['cases'].append(record)
        write(output/'batch.json', report)  # durable intent before spawning
        started = time.monotonic()
        with (case/'execution.log').open('w') as log:
            completed = subprocess.run(command, input=SCRIPT, text=True, stdout=log, stderr=subprocess.STDOUT)
        record.update(exit_code=completed.returncode, wall_seconds=time.monotonic()-started,
                      state='execution_failed', native_prediction=str(destination/'initial'),
                      native_state=str(destination/'state'), public_prediction=str(destination/'public'))
        if completed.returncode == 0:
            try:
                record['worker_receipt'] = verify_output(destination, manifest, config)
                record['state'] = 'completed'
            except (ValueError, KeyError, OSError) as error:
                record.update(state='output_validation_failed', error=str(error))
        write(output/'batch.json', report)
    report['completed'] = True
    report['failed_cases'] = sum(row['state'] != 'completed' for row in report['cases'])
    write(output/'batch.json', report)
    if report['failed_cases']:
        raise RuntimeError('Population includes failed cases; all outputs retained without retries')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ['plan', 'root', 'accepted-batch', 'output']:
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--release', type=Path)
    args = parser.parse_args()
    execute(read(args.plan), args.root, args.accepted_batch, args.output,
            read(args.release) if args.release else None)
