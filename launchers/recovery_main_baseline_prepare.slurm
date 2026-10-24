#!/usr/bin/env bash
#SBATCH --job-name=recovery-main-baseline-prepare
#SBATCH --partition=batch
#SBATCH --account=default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/oscar/scratch/zliu328/3dv-reconstruction/logs/%x-%j.out
set -euo pipefail
exec 9>"/tmp/$(id -un).agent-lock"
flock -n 9
task_root=/oscar/scratch/zliu328/3dv-reconstruction
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONNOUSERSITE=1
"$task_root/envs/da3/bin/python" "$task_root/code/runner/scripts/recovery/prepare_main_baseline_evaluation.py" \
  --inventory "$task_root/provenance/main-exp-execution/formal-rgb-inventory-prepared.json" \
  --source-manifests "$task_root/provenance/main-exp-execution/formal-source-manifests" \
  --output "$task_root/provenance/main-baseline-evaluation"
cp "$task_root/code/runner/launchers/recovery_main_baseline_prepare.slurm" "$task_root/provenance/main-baseline-evaluation/launcher.slurm"
