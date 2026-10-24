"""在冻结的 RGB 输入清单上运行已验收的 VGGT、原版 π³ 和 MapAnything。"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import time

try:
    from .run_backend_population import command_for
    from .run_da3_population import validate_plan, write, identity
except ImportError:
    from run_backend_population import command_for
    from run_da3_population import validate_plan, write, identity


def read(path):
    return json.loads(Path(path).read_text())


RECIPES = {
    'mapanything': {
        'config': 'mapanything', 'entrypoint': 'run_mapanything_rgb.py',
        'sources': ['run_mapanything_rgb.py'], 'preprocessor_alias': 'mapanything',
        'additional_receipts': {}, 'replay_branches': [],
        'method': 'MapAnything-depth+camera', 'code': 'MapAnything', 'environment': 'mapanything',
        'model': 'MapAnything', 'preprocessor': 'mapanything', 'primary': 'depth-camera',
        'branches': ['depth-camera', 'point'], 'code_paths': ['mapanything', 'configs'],
        'auxiliary_revisions': {'DINOv2': 'dinov2_revision'}, 'prediction_config_path': [],
        'sandbox_environment': {'TORCH_HOME': '/tmp/torch'},
        'setup': 'mkdir -p /tmp/torch/hub\nln -s "$root/code/DINOv2" /tmp/torch/hub/facebookresearch_dinov2_main\n',
    },
    'pi3': {
        'config': 'pi3', 'entrypoint': 'run_pi3_rgb.py',
        'sources': ['run_pi3_rgb.py'], 'preprocessor_alias': 'pi3',
        'additional_receipts': {}, 'replay_branches': [],
        'method': 'Pi3-original-point', 'code': 'Pi3', 'environment': 'pi3x',
        'model': 'Pi3', 'preprocessor': 'pi3x', 'primary': 'point',
        'branches': ['point'], 'code_paths': ['pi3'], 'auxiliary_revisions': {}, 'prediction_config_path': [],
        'sandbox_environment': {}, 'setup': '',
    },
}

RECIPES['vggt'] = {
    'config': 'vggt', 'entrypoint': 'run_vggt_rgb.py',
    'sources': ['run_vggt_rgb.py'], 'preprocessor_alias': 'vggt',
    'additional_receipts': {}, 'replay_branches': [],
    'method': 'VGGT-1B-point', 'code': 'VGGT', 'environment': 'vggt',
    'model': 'VGGT-1B', 'preprocessor': 'vggt', 'primary': 'point',
    'branches': ['point', 'depth-camera'], 'code_paths': ['vggt'],
    'auxiliary_revisions': {}, 'prediction_config_path': ['pinned_baseline_config'],
    'sandbox_environment': {}, 'setup': '',
}


RECIPES['mapanything-capture'] = {
    **RECIPES['mapanything'],
    'method': 'MapAnything-point-unmasked',
    'entrypoint': 'run_mapanything_capture.py',
    'sources': ['run_mapanything_rgb.py', 'run_mapanything_capture.py'],
    'primary': 'unmasked-point',
    'replay_branches': ['point', 'depth-camera'],
    'branches': ['unmasked-point', 'unmasked-depth-camera', 'point', 'depth-camera'],
    'additional_receipts': {'capture-result.json': {
        'state': 'unmasked_capture_complete', 'gt_access': False,
        'model_forward_calls': 1, 'filtered_xyz_exact_subset_of_raw': True,
        'filtered_depth_xyz_exact_subset_of_raw': True,
    }},
}


def prediction_configuration(document, spec):
    for key in spec['prediction_config_path']:
        document = document[key]
    return document


def verify_prediction(result_root, spec, manifest, config):
    prediction = result_root/'prediction'
    receipt = read(prediction/'result.json')
    if (receipt['state'] != 'native_prediction_complete' or receipt['gt_access']
            or receipt['views'] != len(manifest) or prediction_configuration(read(prediction/'configuration.json'), spec) != config
            or read(prediction/'input_manifest.json') != manifest):
        raise ValueError('Native worker result disagrees with fixed RGB input/configuration')
    for branch in spec['branches']:
        state = prediction/(branch+'-state')
        public = result_root/(branch+'-public')
        canvas = read(public/'canvas.json')
        if (read(state/'state.json')['gt_conditioning'] or canvas['gt_access']
                or read(state/'input_manifest.json') != manifest
                or read(public/'input_manifest.json') != manifest
                or len(canvas['views']) != len(manifest) or not (state/'state.npz').is_file()):
            raise ValueError('Invalid state or public input identity')
        for row in canvas['views']:
            if not (public/row['file']).is_file():
                raise ValueError('Missing public point map')
    for filename, required in spec['additional_receipts'].items():
        extra = read(prediction/filename)
        if any(extra[key] != value for key, value in required.items()):
            raise ValueError('Capture receipt differs from accepted inference recipe')
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--launcher', type=Path, required=True)
    parser.add_argument('--method', choices=RECIPES, required=True)
    for name in ['root', 'accepted-run', 'output']:
        parser.add_argument('--'+name, type=Path, required=True)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--prepared', type=Path)
    inputs.add_argument('--population-plan', type=Path)
    parser.add_argument('--release', type=Path)
    parser.add_argument('--rerun-accepted', action='store_true')
    parser.add_argument('--replay-reference', type=Path)
    args = parser.parse_args()
    spec = RECIPES[args.method]
    root, output = args.root.resolve(), args.output.resolve()
    runner = root/'code/runner'
    config_path = runner/'configs'/(spec['config']+'_pinned.json')
    config = read(config_path)
    plan = None
    release = read(args.release) if args.release else None
    if args.population_plan:
        plan = read(args.population_plan)
        planned = validate_plan(plan, release, expected_method=spec['method'])
        rows = []
        import hashlib
        for row in planned:
            manifest_path = Path(row['input'])/'input_manifest.json'
            if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != row['input_manifest_sha256']:
                raise ValueError('Population RGB manifest changed')
            if len(read(manifest_path)) != row['views']:
                raise ValueError('Population view count changed')
            rows.append({**row, 'scene': row['case_id'], 'partition': plan['stage'],
                         'provenance': str(args.population_plan)})
        inventory = plan
    else:
        prepared, inventory = read(args.prepared/'preparation.json'), read(args.prepared/'inventory.json')
        rows = prepared['scenes']
        assert prepared['completed'] and not prepared['inference_started']
        assert len(rows) == inventory['expected_scenes'] == len({row['scene'] for row in rows})
        assert all(row['exposure'] == 'development_already_analyzed' for row in rows)
    accepted = args.accepted_run/'prediction'
    receipt = read(accepted/'result.json')
    assert receipt['state'] == 'native_prediction_complete' and not receipt['gt_access']
    for filename, required in spec['additional_receipts'].items():
        extra = read(accepted/filename)
        assert all(extra[key] == value for key, value in required.items())
        assert extra['views'] == receipt['views']
    assert prediction_configuration(read(accepted/'configuration.json'), spec) == read(args.accepted_run/'configuration.json') == config
    accepted_manifest = read(accepted/'input_manifest.json')
    matches = [row['scene'] for row in rows
               if read(Path(row['input'])/'input_manifest.json') == accepted_manifest]
    if len(matches) > 1:
        raise ValueError("Acceptance RGB manifest occurs more than once in population")
    for branch in spec['branches']:
        canvas = read(args.accepted_run/(branch+'-public')/'canvas.json')
        assert not canvas['gt_access'] and len(canvas['views']) == receipt['views']
        assert read(accepted/(branch+'-state')/'input_manifest.json') == accepted_manifest
        assert not read(accepted/(branch+'-state')/'state.json')['gt_conditioning']
    replay = {}
    if spec['replay_branches']:
        if args.replay_reference is None:
            raise ValueError('Capture recipe requires the original filtered reference run')
        import numpy as np
        from native_state import load_state
        for branch in spec['replay_branches']:
            before, before_manifest, before_metadata = load_state(args.replay_reference/'prediction'/(branch+'-state'))
            after, after_manifest, after_metadata = load_state(accepted/(branch+'-state'))
            assert before_manifest == after_manifest == accepted_manifest
            assert not before_metadata['gt_conditioning'] and not after_metadata['gt_conditioning']
            assert before.keys() == after.keys()
            for name in before:
                np.testing.assert_array_equal(before[name], after[name])
            replay[branch] = {'all_fields_exact_equal': True, 'fields': sorted(before)}
        del before, after
    revisions = {spec['code']: 'code_revision', **spec['auxiliary_revisions']}
    for checkout, key in revisions.items():
        path = root/'code'/checkout
        assert subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip() == config[key]
        subprocess.run(['git', '-C', str(path), 'diff', '--exit-code', 'HEAD'], check=True)
    output.mkdir(parents=True, exist_ok=False)
    (output/'replay-verification.json').write_text(json.dumps({
        'reference_run': str(args.replay_reference), 'branches': replay}, indent=2)+'\n')
    source = output/'runner-source'; source.mkdir()
    filenames = [*spec['sources'], 'run_da3_rgb.py', 'native_state.py',
                 'geometry_state.py', 'public_canvas.py']
    for filename in filenames:
        path = runner/'scripts/recovery'/filename
        assert path.read_bytes() == (args.accepted_run/'runner-source'/filename).read_bytes()
        shutil.copy2(path, source/filename)
    preprocessor = runner/'scripts'/spec['preprocessor']/'preprocess.py'
    alias = spec['preprocessor_alias']+'_preprocess.py'
    assert preprocessor.read_bytes() == (args.accepted_run/'runner-source'/alias).read_bytes()
    shutil.copy2(preprocessor, source/alias)
    shutil.copy2(config_path, output/'configuration.json')
    write(output/'inventory.json', inventory)
    if release is not None:
        write(output/'release.json', release)
    shutil.copy2(Path(__file__).with_name('run_da3_population.py'), output/'population_contract.py')
    shutil.copy2(__file__, output/'orchestrator.py')
    shutil.copy2(Path(__file__).with_name('run_backend_population.py'), output/'command_builder.py')
    shutil.copy2(args.launcher, output/'launcher.slurm')
    code = root/'code'/spec['code']
    mounts = [(root/'envs'/spec['environment'], root/'envs'/spec['environment']),
              (root/'envs/python', root/'envs/python'), (root/'models'/spec['model'], '/model'),
              (output/'configuration.json', '/configuration.json')]
    mounts += [(code/path, code/path) for path in spec['code_paths']]
    mounts += [(root/'code'/path, root/'code'/path) for path in spec['auxiliary_revisions']]
    recipe = {'code': code, 'python': root/'envs'/spec['environment']/'bin/python', 'mounts': mounts,
              'environment': spec['sandbox_environment']}
    script = 'set -euo pipefail\npython=$1\ncode=$2\nroot=$3\n'+spec['setup']
    script += ('"$python" /runner/scripts/'+spec['entrypoint']+' --input /input '
               '--output /output/prediction --model /model --code "$code" --config /configuration.json '
               '--forbidden-path "$root/datasets" --forbidden-path "$root/runs" --forbidden-path "$root/provenance"\n')
    for branch in spec['branches']:
        script += ('"$python" /runner/scripts/public_canvas.py --state /output/prediction/'
                   +branch+'-state --output /output/'+branch+'-public\n')
    report = {'schema': 'development-baseline-batch-v1', 'method': spec['method'],
              'expected_scenes': len(rows), 'completed': False, 'scores_computed': False, 'cases': []}
    if plan is not None:
        report.update(schema='rgb-baseline-batch-v1', stage=plan['stage'], plan_sha256=identity(plan))
    write(output/'batch.json', report)
    for index, row in enumerate(rows):
        record = {'scene': row['scene'], 'partition': row['partition'], 'input': row['input'],
                  'input_provenance': row['provenance'], 'views': row['views']}
        report['cases'].append(record)
        if row['scene'] in matches and not args.rerun_accepted:
            result_root = args.accepted_run
            record.update(state='reused_verified_prediction', original_run=str(result_root))
        else:
            case = output/'cases'/f'{index:06d}'
            result_root = case/'output'; result_root.mkdir(parents=True)
            command = command_for(recipe, root, source, row['input'], result_root)
            (case/'command.json').write_text(json.dumps(command, indent=2)+'\n')
            record.update(state='launch_intent', run=str(case))
            write(output/'batch.json', report)
            (case/'worker.sh').write_text(script)
            started = time.monotonic()
            with (case/'execution.log').open('w') as log:
                result = subprocess.run(command, input=script, text=True, stdout=log, stderr=subprocess.STDOUT)
            record.update(exit_code=result.returncode, wall_seconds=time.monotonic()-started,
                          state='completed' if result.returncode == 0 else 'execution_failed', run=str(case))
        record.update(native_prediction=str(result_root/'prediction'),
                      native_state=str(result_root/'prediction'/(spec['primary']+'-state')),
                      public_prediction=str(result_root/(spec['primary']+'-public')),
                      public_branches={branch: str(result_root/(branch+'-public')) for branch in spec['branches']})
        if record['state'] in ['completed', 'reused_verified_prediction']:
            try:
                record['worker_receipt'] = verify_prediction(
                    result_root, spec, read(Path(row['input'])/'input_manifest.json'), config)
            except (ValueError, KeyError, OSError) as error:
                record.update(state='output_validation_failed', error=str(error))
        write(output/'batch.json', report)
        print(json.dumps(record), flush=True)
    report['completed'] = True
    report['failed_scenes'] = sum(row['state'] not in ['completed', 'reused_verified_prediction'] for row in report['cases'])
    (output/'batch.json').write_text(json.dumps(report, indent=2)+'\n')
    if report['failed_scenes']:
        raise RuntimeError(f"{report['failed_scenes']} failed scenes; complete population and logs retained")


if __name__ == '__main__':
    main()
