#!/usr/bin/env bash
#SBATCH --job-name=recovery-main-baseline-native
#SBATCH --partition=gpu-he
#SBATCH --account=default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=96G
#SBATCH --gres=gpu:nvidia_b200:1
#SBATCH --time=02:00:00
#SBATCH --output=/oscar/scratch/zliu328/3dv-reconstruction/logs/%x-%j.out
set -euo pipefail
exec 9>"/tmp/$(id -un).agent-lock"
flock -n 9
task_root=/oscar/scratch/zliu328/3dv-reconstruction
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
"$task_root/envs/mapanything/bin/python" - "$task_root" <<'PY'
import os,subprocess,sys,shutil
from pathlib import Path
root=Path(sys.argv[1]);name=os.environ['BASELINE_NAME']
recipes={
 'vggt':('vggt','vggt','recovery-vggt-smoke-7715752',[]),
 'pi3':('pi3','pi3x','recovery-pi3-smoke-7718923',[]),
 'mapanything':('mapanything-capture','mapanything','recovery-mapanything-capture-7720095',
               ['--replay-reference',str(root/'runs/recovery-mapanything-smoke-7718336')]),
 'amb3r':('amb3r','amb3r','amb3r-dev-smoke-7709057',
           ['--runtime-check',str(root/'provenance/amb3r/runtime-7708309/checkpoint-load.json')]),
}
method,environment,accepted,extra=recipes[name]
entry='run_strong_development_set.py' if name=='amb3r' else 'run_native_development_set.py'
run=root/'runs'/('recovery-main-baseline-'+name+'-'+os.environ['SLURM_JOB_ID'])
run.mkdir(exist_ok=False)
shutil.copy2(root/'code/runner/launchers/recovery_main_baseline_native.slurm',run/'launcher.slurm')
command=[str(root/'envs'/environment/'bin/python'),str(root/'code/runner/scripts/recovery'/entry),
         '--root',str(root),'--method',method,'--accepted-run',str(root/'runs'/accepted),
         '--population-plan',str(root/'provenance/main-baseline-evaluation'/(name+'-plan.json')),
         '--release',str(root/'provenance/main-baseline-evaluation'/(name+'-release.json')),
         '--output',str(run/'population'),*extra]
subprocess.run(command,check=True)
PY
