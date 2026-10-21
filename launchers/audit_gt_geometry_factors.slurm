#!/usr/bin/env bash
#SBATCH --job-name=audit-gt-geometry-factors
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
input="${1:?pass factor-intervention directory}"
output="${2:?pass a new audit JSON path}"
shift 2
export PYTHONPATH="$task_root/code/runner"
cd "$task_root/code/runner"
"$task_root/envs/da3/bin/python" -m scripts.audit.gt_geometry_factor_outputs \
  --input "$input" --output "$output" "$@"
