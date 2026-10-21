#!/usr/bin/env bash
#SBATCH --job-name=scannetpp-gt-geometry-factors
#SBATCH --partition=batch
#SBATCH --account=default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/oscar/scratch/zliu328/3dv-reconstruction/logs/%x-%j.out

set -euo pipefail
task_root=/oscar/scratch/zliu328/3dv-reconstruction
snapshot="${1:?pass immutable snapshot directory}"
reference="${2:?pass GT-extrinsics reference directory}"
output="${3:?pass a new output directory}"
shift 3
(( $# > 0 )) || { echo "pass --scene and --model arguments" >&2; exit 2; }
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
export PYTHONPATH="$task_root/code/runner:$task_root/code/Depth-Anything-3/src"
export TMPDIR="${SLURM_TMPDIR:-$task_root/tmp}"
cd "$task_root/code/Depth-Anything-3"
"$task_root/envs/da3/bin/python" -m scripts.failure_mining.analyze_gt_geometry_factors \
  --metrics-jsonl "$snapshot/completed_metrics.jsonl" \
  --extrinsics-reference "$reference" \
  --dataset-root "$task_root/code/Depth-Anything-3/workspace/benchmark_dataset/scannetpp" \
  --output "$output" "$@"
