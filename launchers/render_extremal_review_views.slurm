#!/usr/bin/env bash
#SBATCH --job-name=render-failure-review-views
#SBATCH --partition=batch
#SBATCH --account=default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/oscar/scratch/zliu328/3dv-reconstruction/logs/%x-%j.out

set -euo pipefail
task_root=/oscar/scratch/zliu328/3dv-reconstruction
bundles="${1:?pass audited failure-review bundle directory}"
output="${2:?pass a new rendered-view directory}"
export PYTHONPATH="$task_root/code/runner"
cd "$task_root/code/runner"
"$task_root/envs/da3/bin/python" \
  -m scripts.failure_mining.render_extremal_review_views \
  --bundles "$bundles" --output "$output" \
  --max-side 1024 --jpeg-quality 90
