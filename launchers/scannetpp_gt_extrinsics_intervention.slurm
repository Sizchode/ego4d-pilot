#!/usr/bin/env bash
#SBATCH --job-name=scannetpp-gt-extrinsics
#SBATCH --partition=batch
#SBATCH --account=default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/oscar/scratch/zliu328/3dv-reconstruction/logs/%x-%j.out

set -euo pipefail
task_root=/oscar/scratch/zliu328/3dv-reconstruction
snapshot="${1:?pass immutable snapshot directory}"
output="${2:?pass a new intervention output directory}"
shift 2
if (( $# == 0 )); then
  echo "pass at least one ScanNet++ scene" >&2
  exit 2
fi
scene_args=()
for scene in "$@"; do
  scene_args+=(--scene "$scene")
done
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export PYTHONPATH="$task_root/code/runner:$task_root/code/Depth-Anything-3/src"
export TMPDIR="${SLURM_TMPDIR:-$task_root/tmp}"
cd "$task_root/code/runner"
"$task_root/envs/da3/bin/python" -m scripts.failure_mining.analyze_gt_extrinsics_intervention \
  --metrics-jsonl "$snapshot/completed_metrics.jsonl" \
  --dataset scannetpp \
  "${scene_args[@]}" \
  --model da3 \
  --model vggt \
  --model pi3x \
  --model mapanything \
  --output "$output" \
  --seed 42
