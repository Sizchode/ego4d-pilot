#!/usr/bin/env bash
#SBATCH --job-name=audit-failure-review-views
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
bundles="${1:?pass audited failure-review bundle directory}"
views="${2:?pass rendered-view directory}"
output="${3:?pass a new audit JSON path}"
export PYTHONPATH="$task_root/code/runner"
cd "$task_root/code/runner"
"$task_root/envs/da3/bin/python" \
  -m scripts.audit.extremal_review_views \
  --bundles "$bundles" --views "$views" --output "$output"
