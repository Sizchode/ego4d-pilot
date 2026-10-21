#!/usr/bin/env bash
#SBATCH --job-name=da3-gt-depth-error
#SBATCH --partition=batch
#SBATCH --account=default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/oscar/scratch/zliu328/3dv-reconstruction/logs/%x-%j.out

set -euo pipefail
task_root=/oscar/scratch/zliu328/3dv-reconstruction
snapshot="${1:?pass immutable snapshot directory}"
output="${2:?pass a new output directory}"
shift 2
(( $# > 0 )) || { echo "pass --scene arguments" >&2; exit 2; }
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
export PYTHONPATH="$task_root/code/runner:$task_root/code/Depth-Anything-3/src"
export TMPDIR="${SLURM_TMPDIR:-$task_root/tmp}"
cd "$task_root/code/Depth-Anything-3"
"$task_root/envs/da3/bin/python" -m scripts.failure_mining.analyze_da3_gt_depth_error \
  --metrics-jsonl "$snapshot/completed_metrics.jsonl" \
  --dataset-root "$task_root/code/Depth-Anything-3/workspace/benchmark_dataset/scannetpp" \
  --model da3 \
  --output "$output" "$@"
