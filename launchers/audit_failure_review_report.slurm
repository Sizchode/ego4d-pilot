#!/usr/bin/env bash
#SBATCH --job-name=audit-failure-review-report
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
bundles="${1:?pass failure-review bundle directory}"
views="${2:?pass rendered-view directory}"
views_audit="${3:?pass passing rendered-view audit JSON}"
report="${4:?pass failure-review report directory}"
output="${5:?pass a new audit JSON path}"
export PYTHONPATH="$task_root/code/runner"
cd "$task_root/code/runner"
"$task_root/envs/da3/bin/python" \
  -m scripts.audit.failure_review_report \
  --bundles "$bundles" --views "$views" \
  --views-audit "$views_audit" --report "$report" --output "$output"
