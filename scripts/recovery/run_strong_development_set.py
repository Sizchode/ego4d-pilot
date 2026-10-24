"""Execute accepted GGPT/AMB3R recipes on the frozen exposed RGB population."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time


def read(path):
    return json.loads(Path(path).read_text())


def ggpt_recipe(root, parent):
    code = root/'code/GGPT'
    mounts = [(root/'envs/ggpt-torch271',root/'envs/ggpt-torch271'),
              (root/'envs/python',root/'envs/python'),
              (root/'models/GGPT','/model'),(root/'models/GGPT-matchers','/matcher-assets'),
              (Path(parent['native_prediction']).parent/'state','/previous')]
    mounts += [(code/path,code/path) for path in ['ggpt','sfm','matching','utils','configs',
                                                'Pointcept/pointcept','RoMaV2/src']]
    return {'code':code,'python':root/'envs/ggpt-torch271/bin/python','mounts':mounts,
            'environment':{'HF_HOME':'/matcher-assets/huggingface','TORCH_HOME':'/matcher-assets/torch'},
            'native':'ggpt','public':'ggpt-public','script':'''set -euo pipefail
python=$1; code=$2; root=$3
"$python" /runner/scripts/ggpt/run_prediction.py --code "$code" --state /previous --input /input --checkpoint /model/model.step228000.pth --output /output/ggpt --forbidden-path "$root/datasets" --forbidden-path "$root/runs" --forbidden-path "$root/provenance"
"$python" /runner/scripts/recovery/public_canvas.py --state /output/ggpt/state --output /output/ggpt-public
'''}


def amb3r_recipe(root, parent):
    code = root/'code/AMB3R'
    python = root/'envs/amb3r/bin/python'
    purelib = subprocess.check_output([str(python),'-c','import sysconfig; print(sysconfig.get_path("purelib"))'],text=True).strip()
    nvrtc = Path(purelib)/'nvidia/cuda_nvrtc/lib'
    if not (nvrtc/'libnvrtc.so.11.2').is_file():
        raise FileNotFoundError('Prepared CUDA 11.8 runtime lacks its NVRTC library')
    mounts = [(root/'envs/amb3r',root/'envs/amb3r'),(root/'models/AMB3R','/model')]
    mounts += [(code/path,code/path) for path in ['amb3r','thirdparty/vggt','thirdparty/ptv3',
                                                'thirdparty/moge','thirdparty/dust3r']]
    return {'code':code,'python':python,'mounts':mounts,'environment':{'LD_LIBRARY_PATH':str(nvrtc)},
            'native':'prediction/backend','public':'backend-public','script':'''set -euo pipefail
python=$1; code=$2; root=$3
"$python" -c 'import ctypes; ctypes.CDLL("libnvrtc.so.11.2"); print("Prepared NVRTC library resolves inside inference sandbox.")'
"$python" /runner/scripts/amb3r/run_prediction.py --code "$code" --input /input --checkpoint /model/amb3r.pt --output /output/prediction --forbidden-path "$root/datasets" --forbidden-path "$root/runs" --forbidden-path "$root/provenance"
"$python" /runner/scripts/recovery/public_canvas.py --state /output/prediction/backend/state --output /output/backend-public
"$python" /runner/scripts/recovery/public_canvas.py --state /output/prediction/frontend/state --output /output/frontend-public
'''}


def command_for(recipe, root, source, input_path, output):
    command = ['bwrap','--die-with-parent','--unshare-pid','--unshare-net','--unshare-ipc','--clearenv']
    for path in ['/usr','/lib','/lib64','/bin','/sbin','/etc','/sys']:
        command += ['--ro-bind',path,path]
    command += ['--proc','/proc','--dev-bind','/dev','/dev','--tmpfs','/tmp']
    for original,target in [*recipe['mounts'],(source,'/runner/scripts'),(input_path,'/input')]:
        command += ['--ro-bind',str(original),str(target)]
    command += ['--bind',str(output),'/output']
    environment = {'PATH':'/usr/bin:/bin','HOME':'/tmp','HF_HOME':'/tmp/huggingface',
        'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','XDG_CACHE_HOME':'/tmp/cache',
        'PYTHONHASHSEED':'47','PYTHONNOUSERSITE':'1','PYTHONUNBUFFERED':'1',
        'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1',
        'CUDA_VISIBLE_DEVICES':os.environ['CUDA_VISIBLE_DEVICES'],'SLURM_JOB_ID':os.environ['SLURM_JOB_ID'],
        **recipe['environment']}
    for key,value in environment.items():
        command += ['--setenv',key,value]
    return command+['--chdir',str(recipe['code']),'/bin/bash','-s','--',str(recipe['python']),str(recipe['code']),str(root)]


def verify_amb3r(output, manifest):
    result = read(output/'prediction/result.json')
    if (result['state'] != 'native_prediction_complete' or result['gt_access']
            or result['views'] != len(manifest) or result['backend_iterations'] != 1
            or result['decoder_calls'] != 2):
        raise ValueError('Expected the accepted full AMB3R frontend/backend execution')
    for branch in ['frontend','backend']:
        state = output/'prediction'/branch/'state'
        public = output/(branch+'-public')
        canvas = read(public/'canvas.json')
        if (read(state/'state.json')['gt_conditioning'] or canvas['gt_access']
                or read(state/'input_manifest.json') != manifest
                or read(public/'input_manifest.json') != manifest
                or len(canvas['views']) != len(manifest) or not (state/'state.npz').is_file()):
            raise ValueError('Missing state or wrong input identity')
        for row in canvas['views']:
            if not (public/row['file']).is_file():
                raise ValueError('Incomplete public output')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    population = parser.add_mutually_exclusive_group(required=True)
    population.add_argument('--prepared',type=Path)
    population.add_argument('--population-plan',type=Path)
    parser.add_argument('--release',type=Path)
    parser.add_argument('--method',choices=['ggpt','amb3r'],required=True)
    parser.add_argument('--runtime-check',type=Path,required=True)
    parser.add_argument('--accepted-run',type=Path,required=True)
    parser.add_argument('--da3-batch',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    root,output = args.root.resolve(),args.output.resolve()
    plan = None
    release = read(args.release) if args.release else None
    if args.population_plan:
        if args.method != 'amb3r':
            raise ValueError('Explicit GGPT populations use the authoritative-state runner')
        from run_da3_population import validate_plan
        import hashlib
        plan = read(args.population_plan)
        planned = validate_plan(plan, release, expected_method='AMB3R-backend-depth+camera')
        rows = []
        for row in planned:
            manifest = Path(row['input'])/'input_manifest.json'
            if (hashlib.sha256(manifest.read_bytes()).hexdigest() != row['input_manifest_sha256']
                    or len(read(manifest)) != row['views']):
                raise ValueError('Fixed RGB input identity changed')
            rows.append({**row, 'scene': row['case_id'], 'partition': plan['stage']})
        inventory = plan
    else:
        population = read(args.prepared/'preparation.json')
        inventory = read(args.prepared/'inventory.json')
        rows = population['scenes']
        assert population['completed'] and len(rows)==inventory['expected_scenes']
        assert len({r['scene'] for r in rows})==len(rows)
        assert all(r['exposure']=='development_already_analyzed' for r in rows)
    runtime = read(args.runtime_check)
    assert runtime['checkpoint_strict_load']
    recipes = {'ggpt':ggpt_recipe,'amb3r':amb3r_recipe}
    sources = {'ggpt':['run_prediction.py'],'amb3r':['run_prediction.py','preprocess.py']}
    accepted_states = {'ggpt':'ggpt/state','amb3r':'prediction/backend/state'}
    accepted_results = {'ggpt':'ggpt/result.json','amb3r':'prediction/result.json'}
    accepted_manifest = read(args.accepted_run/accepted_states[args.method]/'input_manifest.json')
    assert not read(args.accepted_run/accepted_results[args.method])['gt_access']
    matches = [r['scene'] for r in rows if read(Path(r['input'])/'input_manifest.json')==accepted_manifest]
    if len(matches)>1:
        raise ValueError('Duplicate acceptance input in target population')
    if args.method=='ggpt':
        assert len(matches)==1
    parents = {}
    if args.method=='ggpt':
        if args.da3_batch is None:
            parser.error('GGPT requires the completed DA3 population')
        batch=read(args.da3_batch/'batch.json')
        assert batch['completed'] and runtime['cuda_tested']
        parents={row['scene']:row for row in batch['cases']}
        assert set(parents)=={r['scene'] for r in rows}
        accepted_parent=read(args.accepted_run/'parent.json')
        assert parents[matches[0]]['original_run']==accepted_parent['parent_run']
    if args.method=='amb3r':
        assets=read(root/'provenance/amb3r/assets.json')
        code=root/'code/AMB3R'
        revision=subprocess.check_output(['git','-C',str(code),'rev-parse','HEAD'],text=True).strip()
        if not assets['completed'] or revision!=assets['source_revision']:
            raise ValueError('AMB3R code differs from verified assets')
        subprocess.run(['git','-C',str(code),'diff','--exit-code','HEAD'],check=True)
    output.mkdir(parents=True,exist_ok=False)
    source=output/'runner-source'
    (source/args.method).mkdir(parents=True)
    (source/'recovery').mkdir()
    for name in sources[args.method]:
        path=root/'code/runner/scripts'/args.method/name
        assert path.read_bytes()==(args.accepted_run/'runner-source'/args.method/name).read_bytes(), 'Worker differs from accepted single-scene execution'
        shutil.copy2(path,source/args.method/name)
    for name in ['native_state.py','geometry_state.py','public_canvas.py','run_da3_rgb.py']:
        path=root/'code/runner/scripts/recovery'/name
        assert path.read_bytes()==(args.accepted_run/'runner-source/recovery'/name).read_bytes(), 'Geometry adapter differs from accepted execution'
        shutil.copy2(path,source/'recovery'/name)
    shutil.copy2(__file__,output/'orchestrator.py')
    shutil.copy2(args.runtime_check,output/'runtime-check.json')
    (output/'inventory.json').write_text(json.dumps(inventory,indent=2)+'\n')
    if release is not None:
        (output/'release.json').write_text(json.dumps(release,indent=2)+'\n')
    shutil.copy2(Path(__file__).with_name('run_da3_population.py'), output/'population_gate.py')
    if args.method=='ggpt':
        shutil.copy2(args.da3_batch/'batch.json',output/'parent-batch.json')
    shutil.copy2(root/'code/runner/launchers'/f'recovery_{args.method}_development_set.slurm',output/'launcher.slurm')
    report={'method':args.method,'expected_scenes':len(rows),'completed':False,'scores_computed':False,
            'amb3r_first_scene_repeated_for_nvrtc_loader_verification':args.method=='amb3r','cases':[]}
    if plan is not None:
        from run_da3_population import identity
        report.update(method=plan['method'],stage=plan['stage'],plan_sha256=identity(plan))
    for index,row in enumerate(rows):
        record={'scene':row['scene'],'partition':row['partition'],'input':row['input'],'views':row['views']}
        report['cases'].append(record)
        parent=parents.get(row['scene'])
        if parent is not None:
            record['parent']=parent
        if args.method=='ggpt' and row['scene'] in matches:
            record.update(state='reused_verified_prediction',run=str(args.accepted_run),
                          public_prediction=str(args.accepted_run/'ggpt-public'))
        elif args.method=='ggpt' and parent['state']!='completed':
            record.update(state='upstream_prediction_failed',parent=parent)
        else:
            case=output/'cases'/f'{index:06d}'
            (case/'output').mkdir(parents=True)
            recipe=recipes[args.method](root,parent)
            version=subprocess.check_output([str(recipe['python']),'-c','import importlib.metadata; print(importlib.metadata.version("torch"))'],text=True).strip()
            assert version==runtime['packages']['torch']
            command=command_for(recipe,root,source,row['input'],case/'output')
            (case/'command.json').write_text(json.dumps(command,indent=2)+'\n')
            record.update(state='launch_intent',run=str(case))
            (output/'batch.json').write_text(json.dumps(report,indent=2)+'\n')
            (case/'worker.sh').write_text(recipe['script'])
            started=time.monotonic()
            with (case/'execution.log').open('w') as log:
                result=subprocess.run(command,input=recipe['script'],text=True,stdout=log,stderr=subprocess.STDOUT)
            record.update(state='completed' if result.returncode==0 else 'execution_failed',
                          run=str(case),exit_code=result.returncode,wall_seconds=time.monotonic()-started,
                          public_prediction=str(case/'output'/recipe['public']))
            if args.method=='amb3r':
                record.update(frontend_public_prediction=str(case/'output/frontend-public'),
                              native_prediction=str(case/'output/prediction'),
                              native_state=str(case/'output/prediction/backend/state'))
        if args.method=='amb3r' and record['state']=='completed':
            try:
                verify_amb3r(case/'output',read(Path(row['input'])/'input_manifest.json'))
            except (ValueError,KeyError,OSError) as error:
                record.update(state='output_validation_failed',error=str(error))
        (output/'batch.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(record),flush=True)
    report['completed']=True
    report['failed_scenes']=sum(r['state'] not in ['completed','reused_verified_prediction'] for r in report['cases'])
    (output/'batch.json').write_text(json.dumps(report,indent=2)+'\n')
    if report['failed_scenes']:
        raise RuntimeError('Cases failed; retained in batch population with their original logs')


if __name__=='__main__':
    main()
