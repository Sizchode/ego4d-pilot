#!/usr/bin/env bash
#SBATCH --job-name=combine-gt-extrinsics
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
output="${1:?pass a new combined intervention directory}"
shift
(( $# >= 2 )) || { echo "pass at least two disjoint intervention directories" >&2; exit 2; }
input_args=()
for intervention in "$@"; do
  input_args+=(--intervention-dir "$intervention")
done
export PYTHONPATH="$task_root/code/runner"
cd "$task_root/code/runner"
"$task_root/envs/da3/bin/python" \
  -m scripts.failure_mining.combine_gt_extrinsics_interventions \
  "${input_args[@]}" --output "$output"
