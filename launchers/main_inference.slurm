#!/usr/bin/env bash
#SBATCH --job-name=main-inference
#SBATCH --partition=gpu-he
#SBATCH --account=default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=/oscar/scratch/zliu328/3dv-reconstruction/logs/%x-%j.out
set -euo pipefail
# 提交时明确指定 --gres；AMB3R/GGPT 用 H100，其余沿用 B200。
: "${BASELINE_NAME:?da3 / vggt / pi3 / mapanything / amb3r / ggpt}"
: "${ACCEPTED_RUN:?已验收运行目录；不自动选择历史作业}"
task_root=/oscar/scratch/zliu328/3dv-reconstruction
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
# 一种方法仅启动一个 population；独立方法可并行。
[[ "$BASELINE_NAME" =~ ^(da3|vggt|pi3|mapanything|amb3r|ggpt)$ ]]
mkdir -p "$task_root/provenance/locks"
exec 9>"$task_root/provenance/locks/main-baseline-$BASELINE_NAME.agent-lock"
flock -n 9
python3 - "$task_root" <<'PY'
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

root = Path(sys.argv[1])
name = os.environ['BASELINE_NAME']
# 这里只选择已有执行器及其参数；模型计算、预处理和验收检查保持不变。
recipes = {
    'da3': ('da3', 'run_da3_population.py', '.',
            '--plan {plan} --accepted-batch {ACCEPTED_RUN}'),
    'vggt': ('vggt', 'run_native_population.py', 'population',
             '--method vggt --launcher {launcher} --population-plan {plan} --accepted-run {ACCEPTED_RUN}'),
    'pi3': ('pi3x', 'run_native_population.py', 'population',
            '--method pi3 --launcher {launcher} --population-plan {plan} --accepted-run {ACCEPTED_RUN}'),
    'mapanything': ('mapanything', 'run_native_population.py', 'population',
                    '--method mapanything-capture --launcher {launcher} --population-plan {plan} --accepted-run {ACCEPTED_RUN} --replay-reference {REPLAY_REFERENCE}'),
    'amb3r': ('amb3r', 'run_backend_population.py', 'population',
              '--method amb3r --launcher {launcher} --population-plan {plan} --accepted-run {ACCEPTED_RUN} --runtime-check {RUNTIME_CHECK}'),
    'ggpt': ('ggpt-torch271', 'run_ggpt_population.py', '.',
             '--population-plan {plan} --accepted-run {ACCEPTED_RUN} --runtime-check {RUNTIME_CHECK} --parent-batch {PARENT_BATCH} --rerun-accepted --launcher {launcher}'),
}
environment, entry, suffix, arguments = recipes[name]
runner = root / 'code/runner'
launcher = runner / 'launchers/main_inference.slurm'
plan = root / f'provenance/main-baseline-evaluation/{name}-plan.json'
release = plan.with_name(f'{name}-release.json')
run = root / f'runs/recovery-main-baseline-{name}-{os.environ["SLURM_JOB_ID"]}'
context = dict(os.environ, plan=str(plan), launcher=str(launcher))
# Split the template before substitution, preserving spaces in supplied paths.
options = [token.format_map(context) for token in arguments.split()]
python = root / 'envs' / environment / 'bin/python'
command = [str(python), str(runner / 'scripts/recovery' / entry),
           '--root', str(root), '--release', str(release), '--output', str(run / suffix), *options]
receipt = root / f'provenance/main-inference-{os.environ["SLURM_JOB_ID"]}'
receipt.mkdir(parents=True, exist_ok=False)
shutil.copy2(launcher, receipt / 'launcher.slurm')
(receipt / 'command.json').write_text(json.dumps(command, indent=2) + '\n')
# 所有方法先执行真实 CUDA 运算，避免不兼容的环境产生整批假失败。
subprocess.run([str(python), '-c', '''import json, sys, torch
x = torch.ones(4, device='cuda')
y = (x + 1.0) / 2.0
torch.cuda.synchronize()
assert torch.equal(y, x)
with open(sys.argv[1], 'w') as stream:
    json.dump(dict(cuda_kernel_test_passed=True, torch_version=torch.__version__,
                   gpu=torch.cuda.get_device_name(), device_capability=torch.cuda.get_device_capability(),
                   compiled_architectures=torch.cuda.get_arch_list()), stream, indent=2)
''', str(receipt / 'runtime-preflight.json')], check=True)
subprocess.run(command, check=True)
shutil.copy2(launcher, run / 'launcher.slurm')
PY
