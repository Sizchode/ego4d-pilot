#!/usr/bin/env bash
#SBATCH --job-name=recovery-main-baseline-da3
#SBATCH --partition=gpu-he
#SBATCH --account=default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --gres=gpu:nvidia_b200:1
#SBATCH --time=02:00:00
#SBATCH --output=/oscar/scratch/zliu328/3dv-reconstruction/logs/%x-%j.out
set -euo pipefail
exec 9>"/tmp/$(id -un).agent-lock"
flock -n 9
task_root=/oscar/scratch/zliu328/3dv-reconstruction
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
"$task_root/envs/da3/bin/python" "$task_root/code/runner/scripts/recovery/run_da3_population.py" \
  --root "$task_root" --plan "$task_root/provenance/main-baseline-evaluation/da3-plan.json" \
  --release "$task_root/provenance/main-baseline-evaluation/da3-release.json" \
  --accepted-batch "$task_root/runs/recovery-da3-devset-7710713" \
  --output "$task_root/runs/recovery-main-baseline-da3-$SLURM_JOB_ID"
cp "$task_root/code/runner/launchers/recovery_main_baseline_da3.slurm" "$task_root/runs/recovery-main-baseline-da3-$SLURM_JOB_ID/launcher.slurm"
