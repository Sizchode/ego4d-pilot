#!/usr/bin/env bash
#SBATCH --job-name=render-residual-gaps
#SBATCH --partition=batch
#SBATCH --account=default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=/oscar/scratch/zliu328/3dv-reconstruction/logs/%x-%j.out

set -euo pipefail
task_root=/oscar/scratch/zliu328/3dv-reconstruction
gt_depth="${1:?pass GT-depth analysis directory}"
selection_csv="${2:?pass selection CSV}"
selection_manifest="${3:?pass selection manifest}"
output="${4:?pass a new output directory}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLBACKEND=Agg PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
export PYTHONPATH="$task_root/code/runner"
cd "$task_root/code/runner"
"$task_root/envs/da3/bin/python" -m scripts.failure_mining.render_da3_gt_depth_extremes \
  --gt-depth-analysis "$gt_depth" \
  --selection-csv "$selection_csv" \
  --selection-manifest "$selection_manifest" \
  --output "$output"
