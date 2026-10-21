#!/usr/bin/env bash
#SBATCH --job-name=pi3x-failure-mining
#SBATCH --partition=batch
#SBATCH --account=default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/oscar/scratch/zliu328/3dv-reconstruction/logs/%x-%j.out

set -euo pipefail
task_root=/oscar/scratch/zliu328/3dv-reconstruction
batch_dir="${1:?pass the completed Pi3X batch directory}"
analysis_dir="${2:?pass the Pi3X analysis output directory}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export TMPDIR="${SLURM_TMPDIR:-$task_root/tmp}"
cd "$task_root/code/runner"
"$task_root/envs/pi3x/bin/python" scripts/failure_mining/analyze_da3_batch.py \
  --batch-dir "$batch_dir" --output "$analysis_dir"
