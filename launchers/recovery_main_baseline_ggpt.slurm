#!/usr/bin/env bash
#SBATCH --job-name=recovery-main-baseline-ggpt
#SBATCH --partition=gpu-he
#SBATCH --account=default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=96G
#SBATCH --gres=gpu:h100:1
#SBATCH --time=03:00:00
#SBATCH --output=/oscar/scratch/zliu328/3dv-reconstruction/logs/%x-%j.out
set -euo pipefail
exec 9>"/tmp/$(id -un).agent-lock"
flock 9
task_root=/oscar/scratch/zliu328/3dv-reconstruction
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
"$task_root/envs/ggpt-torch271/bin/python" "$task_root/code/runner/scripts/recovery/run_ggpt_state_development_set.py" \
  --root "$task_root" \
  --population-plan "$task_root/provenance/main-baseline-evaluation/ggpt-plan.json" \
  --release "$task_root/provenance/main-baseline-evaluation/ggpt-release.json" \
  --parent-batch "$task_root/runs/recovery-main-baseline-da3-7748832/batch.json" \
  --accepted-run "$task_root/runs/recovery-ggpt-single-thread-acceptance-7739203/repeat0" \
  --runtime-check "$task_root/provenance/ggpt/validate271-7709413/cuda-check.json" \
  --rerun-accepted \
  --launcher "$task_root/code/runner/launchers/recovery_main_baseline_ggpt.slurm" \
  --output "$task_root/runs/recovery-main-baseline-ggpt-$SLURM_JOB_ID"
