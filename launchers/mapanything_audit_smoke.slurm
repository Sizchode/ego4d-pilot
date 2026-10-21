#!/usr/bin/env bash
#SBATCH --job-name=mapanything-audit-smoke
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
run_dir="${1:?pass the completed MapAnything smoke directory}"
smoke_log="${2:?pass the MapAnything smoke Slurm log}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
cd "$task_root/code/runner"
"$task_root/envs/mapanything/bin/python" scripts/audit/mapanything_smoke.py \
  --run "$run_dir" \
  --runner-root "$task_root/code/runner" \
  --slurm-log "$smoke_log" \
  --output "$run_dir/automated_audit.json"
