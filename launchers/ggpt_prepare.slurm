#!/usr/bin/env bash
#SBATCH --job-name=ggpt-assets
#SBATCH --partition=batch
#SBATCH --account=default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --output=/oscar/scratch/zliu328/3dv-reconstruction/logs/%x-%j.out

set -euo pipefail
exec 9>"/tmp/$(id -un).agent-lock"
flock -n 9
task_root=/oscar/scratch/zliu328/3dv-reconstruction
export HF_HOME="$task_root/cache/huggingface"
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_DISABLE_PROGRESS_BARS=1
export HF_HUB_DISABLE_XET=1
export UV_CACHE_DIR="$task_root/cache/uv"
export UV_CONCURRENT_DOWNLOADS=1
export UV_CONCURRENT_BUILDS=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TMPDIR="${SLURM_TMPDIR:-$task_root/tmp}"
mkdir -p "$TMPDIR"
# Keep the baseline inference environment unchanged. Hub >=1.2 observes the
# server's RateLimit reset header; use its resolver download transport here.
uv venv --python "$task_root/envs/da3/bin/python" --allow-existing "$task_root/envs/ggpt-assets"
uv pip install --python "$task_root/envs/ggpt-assets/bin/python" \
  'huggingface_hub==1.2.0' 'numpy==2.2.6' 'Pillow==11.3.0'
"$task_root/envs/ggpt-assets/bin/python" -u \
  "$task_root/code/runner/scripts/ggpt/prepare_assets.py" \
  --root "$task_root" \
  --config "$task_root/code/runner/configs/ggpt_pinned.json"
