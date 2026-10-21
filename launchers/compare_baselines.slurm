#!/usr/bin/env bash
#SBATCH --job-name=compare-baselines
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
output_dir="${1:?pass the cross-model output directory}"
shift
if (( $# < 2 )); then
  printf '%s\n' 'pass at least two MODEL=ANALYSIS_DIRECTORY inputs' >&2
  exit 2
fi
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export PYTHONPATH="$task_root/code/runner"
cd "$task_root/code/runner"
command=("$task_root/envs/da3/bin/python" -m scripts.failure_mining.compare_baselines)
for analysis in "$@"; do
  command+=(--analysis "$analysis")
done
command+=(--output "$output_dir")
"${command[@]}"
