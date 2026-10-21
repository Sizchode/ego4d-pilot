#!/usr/bin/env bash
#SBATCH --job-name=select-failure-review
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
analysis="${1:?pass final failure-mining directory}"
audit="${2:?pass passing final audit JSON}"
output="${3:?pass a new selection directory}"
export PYTHONPATH="$task_root/code/runner"
cd "$task_root/code/runner"
"$task_root/envs/da3/bin/python" \
  -m scripts.failure_mining.select_failure_review_cases \
  --analysis "$analysis" --audit "$audit" --output "$output"
