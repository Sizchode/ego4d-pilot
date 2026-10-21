#!/usr/bin/env bash
#SBATCH --job-name=archive-snapshot-review
#SBATCH --partition=batch
#SBATCH --account=default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/oscar/scratch/zliu328/3dv-reconstruction/logs/%x-%j.out

set -euo pipefail
task_root=/oscar/scratch/zliu328/3dv-reconstruction
snapshot="${1:?pass immutable snapshot directory}"
dataset="${2:?pass dataset name}"
output="${3:?pass a new output tar path}"
shift 3
(( $# > 0 )) || { echo "pass at least one scene" >&2; exit 2; }
scene_args=()
for scene in "$@"; do
  scene_args+=(--scene "$scene")
done
export PYTHONPATH="$task_root/code/runner"
cd "$task_root/code/runner"
"$task_root/envs/da3/bin/python" \
  -m scripts.failure_mining.archive_case_bundles \
  --metrics-jsonl "$snapshot/completed_metrics.jsonl" \
  --dataset "$dataset" "${scene_args[@]}" \
  --model da3 --model vggt --model pi3x --model mapanything \
  --output-tar "$output"
