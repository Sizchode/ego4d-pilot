#!/usr/bin/env bash
#SBATCH --job-name=view-depth-evidence
#SBATCH --partition=batch
#SBATCH --account=default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/oscar/scratch/zliu328/3dv-reconstruction/logs/%x-%j.out

set -euo pipefail
task_root=/oscar/scratch/zliu328/3dv-reconstruction
gt_depth="${1:?pass GT-depth analysis directory}"
image_analysis="${2:?pass image-evidence analysis directory}"
visibility_analysis="${3:?pass visibility analysis directory}"
output="${4:?pass a new output directory}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
export PYTHONPATH="$task_root/code/runner"
cd "$task_root/code/runner"
"$task_root/envs/da3/bin/python" -m scripts.failure_mining.compare_view_depth_error_evidence \
  --gt-depth-analysis "$gt_depth" \
  --image-analysis "$image_analysis" \
  --visibility-analysis "$visibility_analysis" \
  --output "$output"
