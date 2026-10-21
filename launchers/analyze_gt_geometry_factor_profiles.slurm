#!/usr/bin/env bash
#SBATCH --job-name=gt-geometry-factor-profiles
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
input="${1:?pass factor-summary directory}"
output="${2:?pass a new profile directory}"
export PYTHONPATH="$task_root/code/runner"
cd "$task_root/code/runner"
"$task_root/envs/da3/bin/python" \
  -m scripts.failure_mining.analyze_gt_geometry_factor_profiles \
  --input "$input" --output "$output"
