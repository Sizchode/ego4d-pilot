#!/usr/bin/env bash
#SBATCH --job-name=audit-baseline-batches
#SBATCH --partition=batch
#SBATCH --account=default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=/oscar/scratch/zliu328/3dv-reconstruction/logs/%x-%j.out

set -euo pipefail
task_root=/oscar/scratch/zliu328/3dv-reconstruction
audit_dir="${1:?pass a new audit output directory}"
shift
if (( $# == 0 )); then
  echo "pass at least one completed batch directory" >&2
  exit 2
fi
mkdir "$audit_dir"
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export PYTHONPATH="$task_root/code/runner"
cd "$task_root/code/runner"
for batch_dir in "$@"; do
  batch_name="$(basename "$batch_dir")"
  "$task_root/envs/da3/bin/python" -m scripts.audit.batch_outputs \
    --batch-dir "$batch_dir" \
    --output "$audit_dir/$batch_name.json"
done
