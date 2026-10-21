#!/usr/bin/env bash
#SBATCH --job-name=baseline-visibility-mining
#SBATCH --partition=batch
#SBATCH --account=default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/oscar/scratch/zliu328/3dv-reconstruction/logs/%x-%j.out

set -euo pipefail
task_root=/oscar/scratch/zliu328/3dv-reconstruction
batch_dir="${1:?pass one completed baseline batch directory}"
analysis_dir="${2:?pass the visibility-analysis output directory}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export TMPDIR="${SLURM_TMPDIR:-$task_root/tmp}"
export PYTHONPATH="$task_root/code/runner:$task_root/code/Depth-Anything-3/src"
cd "$task_root/code/runner"
"$task_root/envs/da3/bin/python" -m unittest tests.test_da3_geometry -v
"$task_root/envs/da3/bin/python" -m scripts.failure_mining.analyze_visibility_batch \
  --batch-dir "$batch_dir" --output "$analysis_dir"
