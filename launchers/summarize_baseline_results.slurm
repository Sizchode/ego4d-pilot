#!/usr/bin/env bash
#SBATCH --job-name=summarize-baselines
#SBATCH --partition=batch
#SBATCH --account=default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/oscar/scratch/zliu328/3dv-reconstruction/logs/%x-%j.out

set -euo pipefail
task_root=/oscar/scratch/zliu328/3dv-reconstruction
cross_dir="${1:?pass standard cross-model output directory}"
output="${2:?pass a new summary output directory}"
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export PYTHONPATH="$task_root/code/runner"
cd "$task_root/code/runner"
"$task_root/envs/da3/bin/python" -m scripts.failure_mining.summarize_baseline_results \
  --cross-dir "$cross_dir" \
  --output "$output"
