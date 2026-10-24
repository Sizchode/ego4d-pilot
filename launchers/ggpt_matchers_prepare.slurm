#!/usr/bin/env bash
#SBATCH --job-name=ggpt-match-assets
#SBATCH --partition=batch
#SBATCH --account=default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/oscar/scratch/zliu328/3dv-reconstruction/logs/%x-%j.out

set -euo pipefail
exec 9>"/tmp/$(id -un).agent-lock"
flock -n 9
task_root=/oscar/scratch/zliu328/3dv-reconstruction
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export HF_HUB_DISABLE_XET=1 HF_HUB_DISABLE_TELEMETRY=1
export TMPDIR="${SLURM_TMPDIR:-$task_root/tmp}"
mkdir -p "$TMPDIR"
cd "$task_root/code/GGPT"
"$task_root/envs/ggpt/bin/python" "$task_root/code/runner/scripts/ggpt/prepare_matchers.py" \
  --code "$task_root/code/GGPT" --cache "$task_root/models/GGPT-matchers" \
  --output "$task_root/provenance/ggpt/matchers-$SLURM_JOB_ID.json"
