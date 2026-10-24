#!/usr/bin/env bash
#SBATCH --job-name=amb3r-assets
#SBATCH --partition=batch
#SBATCH --account=default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=/oscar/scratch/zliu328/3dv-reconstruction/logs/%x-%j.out

set -euo pipefail
exec 9>"/tmp/$(id -un).agent-lock"
flock -n 9
task_root=/oscar/scratch/zliu328/3dv-reconstruction
export UV_CACHE_DIR="$task_root/cache/uv"
export UV_CONCURRENT_DOWNLOADS=1 UV_CONCURRENT_BUILDS=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export TMPDIR="${SLURM_TMPDIR:-$task_root/tmp}"
export XDG_CACHE_HOME="$task_root/cache/amb3r-assets"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"
env="$task_root/envs/amb3r-assets"
uv venv --python "$task_root/envs/da3/bin/python" --allow-existing "$env"
uv pip install --python "$env/bin/python" 'gdown==5.2.0'
"$env/bin/python" "$task_root/code/runner/scripts/amb3r/prepare_assets.py" \
  --root "$task_root" --config "$task_root/code/runner/configs/amb3r_pinned.json"
