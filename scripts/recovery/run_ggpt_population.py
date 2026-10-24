"""Apply accepted GGPT to an explicit frozen population of native point states.

Every parent must name its authoritative state. No feed-forward inference,
depth conversion, sample replacement, or ground-truth access occurs here.
"""
import argparse
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import time

try:
    from .run_backend_population import command_for
    from .parent_population import population_arguments, load_population, archive_population, verify_geometry, FAILED, write
except ImportError:
    from run_backend_population import command_for
    from parent_population import population_arguments, load_population, archive_population, verify_geometry, FAILED, write


def read(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser()
    for name in ['root', 'parent-batch', 'accepted-run', 'runtime-check', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--launcher', type=Path, required=True)
    population_arguments(parser)
    args = parser.parse_args()
    root, output = args.root.resolve(), args.output.resolve()
    rows, batch, parents = load_population(args, '+GGPT')
    runtime = read(args.runtime_check)
    assert runtime['checkpoint_strict_load'] and runtime['cuda_tested']
    assert importlib.metadata.version('torch') == runtime['packages']['torch']
    accepted_manifest = read(args.accepted_run/'ggpt/state/input_manifest.json')
    accepted_result = read(args.accepted_run/'ggpt/result.json')
    assert accepted_result['state'] == 'reconstruction_complete' and not accepted_result['gt_access']
    canvas = read(args.accepted_run/'ggpt-public/canvas.json')
    assert not canvas['gt_access'] and len(canvas['views']) == len(accepted_manifest)
    accepted_parent = Path(read(args.accepted_run/'parent.json')['parent_state']).resolve()
    output.mkdir(parents=True, exist_ok=False)
    source = output/'runner-source'
    for folder, names in {
        'ggpt': ['run_prediction.py'],
        'recovery': ['native_state.py', 'geometry_state.py', 'public_canvas.py', 'run_da3_rgb.py'],
    }.items():
        (source/folder).mkdir(parents=True)
        for name in names:
            path = root/'code/runner/scripts'/folder/name
            assert path.read_bytes() == (args.accepted_run/'runner-source'/folder/name).read_bytes()
            shutil.copy2(path, source/folder/name)
    shutil.copy2(__file__, output/'orchestrator.py')
    shutil.copy2(Path(__file__).with_name('run_backend_population.py'), output/'command_builder.py')
    shutil.copy2(args.parent_batch, output/'parent-batch.json')
    shutil.copy2(args.runtime_check, output/'runtime-check.json')
    archive_population(args, output)
    shutil.copy2(args.launcher, output/'launcher.slurm')
    code = root/'code/GGPT'
    mounts = [(root/'envs/ggpt-torch271', root/'envs/ggpt-torch271'),
              (root/'envs/python', root/'envs/python'),
              (root/'models/GGPT', '/model'), (root/'models/GGPT-matchers', '/matcher-assets')]
    mounts += [(code/path, code/path) for path in
               ['ggpt', 'sfm', 'matching', 'utils', 'configs', 'Pointcept/pointcept', 'RoMaV2/src']]
    script = '''set -euo pipefail
python=$1
code=$2
root=$3
"$python" /runner/scripts/ggpt/run_prediction.py --code "$code" --state /previous --input /input --checkpoint /model/model.step228000.pth --output /output/ggpt --forbidden-path "$root/datasets" --forbidden-path "$root/runs" --forbidden-path "$root/provenance"
"$python" /runner/scripts/recovery/public_canvas.py --state /output/ggpt/state --output /output/ggpt-public
'''
    deterministic = accepted_result['torch_deterministic_algorithms']
    if deterministic:
        (source/'deterministic_entry.py').write_text(
            'import runpy,sys,torch\n'
            'torch.use_deterministic_algorithms(True)\n'
            'target=sys.argv.pop(1)\n'
            'runpy.run_path(target,run_name="__main__")\n')
        script = script.replace('"$python" /runner/scripts/ggpt/run_prediction.py',
                                '"$python" /runner/scripts/deterministic_entry.py /runner/scripts/ggpt/run_prediction.py')
    report = {'method': batch['method'] + '+GGPT', 'expected_scenes': len(rows),
              'completed': False, 'scores_computed': False, 'cases': []}
    write(output/'batch.json', report)
    for index, row in enumerate(rows):
        parent = parents[row['scene']]
        record = {'scene': row['scene'], 'partition': row['partition'], 'input': row['input'],
                  'views': row['views'], 'parent': parent}
        report['cases'].append(record)
        if parent['state'] in FAILED:
            record.update(state='upstream_prediction_failed')
        else:
            assert parent['state'] in {'completed', 'reused_verified_prediction'}
            native = Path(parent['native_state']).resolve()
            assert not read(native/'state.json')['gt_conditioning']
            assert read(native/'input_manifest.json') == read(Path(row['input'])/'input_manifest.json')
            if native == accepted_parent and not args.rerun_accepted:
                assert read(native/'input_manifest.json') == accepted_manifest
                record.update(state='reused_verified_prediction', run=str(args.accepted_run),
                              native_state=str(args.accepted_run/'ggpt/state'),
                              public_prediction=str(args.accepted_run/'ggpt-public'))
            else:
                case = output/'cases'/f'{index:06d}'
                (case/'output').mkdir(parents=True)
                recipe = {'code': code, 'python': root/'envs/ggpt-torch271/bin/python',
                          'mounts': mounts + [(native, '/previous')],
                          'environment': {'HF_HOME': '/matcher-assets/huggingface',
                                          'TORCH_HOME': '/matcher-assets/torch'}}
                if deterministic:
                    recipe['environment']['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
                command = command_for(recipe, root, source, row['input'], case/'output')
                (case/'command.json').write_text(json.dumps(command, indent=2)+'\n')
                (case/'worker.sh').write_text(script)
                record.update(state='launch_intent', run=str(case))
                write(output/'batch.json', report)
                started = time.monotonic()
                with (case/'execution.log').open('w') as log:
                    result = subprocess.run(command, input=script, text=True, stdout=log, stderr=subprocess.STDOUT)
                record.update(exit_code=result.returncode, wall_seconds=time.monotonic()-started,
                              state='completed' if result.returncode == 0 else 'execution_failed',
                              run=str(case), native_state=str(case/'output/ggpt/state'),
                              public_prediction=str(case/'output/ggpt-public'))
            record['native_prediction'] = str(Path(record['native_state']).parent)
            if record['state'] in {'completed', 'reused_verified_prediction'}:
                try:
                    result = read(Path(record['native_prediction'])/'result.json')
                    if result['state'] != 'reconstruction_complete' or result['gt_access']:
                        raise ValueError('Invalid G output receipt')
                    if result['torch_deterministic_algorithms'] != deterministic:
                        raise ValueError('GGPT execution settings differ from reference')
                    verify_geometry(Path(record['native_state']), Path(record['public_prediction']),
                                    read(Path(row['input'])/'input_manifest.json'))
                except (ValueError, KeyError, OSError) as error:
                    record.update(state='output_validation_failed', error=str(error))
        write(output/'batch.json', report)
        print(json.dumps(record), flush=True)
    report['completed'] = True
    report['failed_scenes'] = sum(row['state'] in FAILED
                                  for row in report['cases'])
    write(output/'batch.json', report)
    if report['failed_scenes']:
        raise RuntimeError(f"{report['failed_scenes']} cases failed; complete population and logs retained")


if __name__ == '__main__':
    main()
