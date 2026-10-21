#!/usr/bin/env bash
#SBATCH --job-name=build-failure-review-bundles
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
selection="${1:?pass selection directory}"
summary="${2:?pass baseline-summary directory}"
summary_audit="${3:?pass passing summary audit JSON}"
output="${4:?pass a new bundle directory}"
export PYTHONPATH="$task_root/code/runner"
cd "$task_root/code/runner"
"$task_root/envs/da3/bin/python" \
  -m scripts.failure_mining.build_failure_review_bundles \
  --selection "$selection" --summary "$summary" \
  --summary-audit "$summary_audit" --output "$output"
