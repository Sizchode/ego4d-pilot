#!/usr/bin/env bash
#SBATCH --job-name=recovery-main-baseline-amb3r
#SBATCH --partition=gpu-he
#SBATCH --account=default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=96G
#SBATCH --gres=gpu:h100:1
#SBATCH --time=02:00:00
#SBATCH --output=/oscar/scratch/zliu328/3dv-reconstruction/logs/%x-%j.out
set -euo pipefail
task_root=/oscar/scratch/zliu328/3dv-reconstruction
mkdir -p "$task_root/provenance/locks"
# Serialize this population while allowing independent methods on other allocated GPUs.
exec 9>"$task_root/provenance/locks/main-baseline-amb3r.agent-lock"
flock -n 9
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
run="$task_root/runs/recovery-main-baseline-amb3r-$SLURM_JOB_ID"
mkdir "$run"
cp "$task_root/code/runner/launchers/recovery_main_baseline_amb3r.slurm" "$run/launcher.slurm"
# Exercise an actual CUDA kernel before launching the population. An incompatible
# allocation must fail once here, rather than generate a failed result per input.
"$task_root/envs/amb3r/bin/python" - "$run/runtime-preflight.json" <<'PY'
import json
import sys
from pathlib import Path
import torch
x = torch.ones(4, device='cuda')
y = (x + 1.0) / 2.0
torch.cuda.synchronize()
assert torch.equal(y, x)
Path(sys.argv[1]).write_text(json.dumps({
    'cuda_kernel_test_passed': True,
    'torch_version': torch.__version__,
    'gpu': torch.cuda.get_device_name(),
    'device_capability': torch.cuda.get_device_capability(),
    'compiled_architectures': torch.cuda.get_arch_list(),
}, indent=2)+'\n')
PY
"$task_root/envs/amb3r/bin/python" "$task_root/code/runner/scripts/recovery/run_strong_development_set.py" \
  --root "$task_root" --method amb3r \
  --accepted-run "$task_root/runs/amb3r-dev-smoke-7709057" \
  --runtime-check "$task_root/provenance/amb3r/runtime-7708309/checkpoint-load.json" \
  --population-plan "$task_root/provenance/main-baseline-evaluation/amb3r-plan.json" \
  --release "$task_root/provenance/main-baseline-evaluation/amb3r-release.json" \
  --output "$run/population"
