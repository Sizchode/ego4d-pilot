#!/usr/bin/env bash
#SBATCH --job-name=da3-prepare
#SBATCH --partition=batch
#SBATCH --account=default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=/oscar/scratch/zliu328/3dv-reconstruction/logs/%x-%j.out

set -euo pipefail
task_root=/oscar/scratch/zliu328/3dv-reconstruction
export UV_CACHE_DIR="$task_root/cache/uv"
export UV_PYTHON_INSTALL_DIR="$task_root/envs/python"
export HF_HOME="$task_root/cache/huggingface"
export XDG_CACHE_HOME="$task_root/cache"
export TMPDIR="${SLURM_TMPDIR:-$task_root/tmp}"
export UV_CONCURRENT_DOWNLOADS=1
export UV_CONCURRENT_BUILDS=1
export HF_HUB_DISABLE_TELEMETRY=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
mkdir -p "$TMPDIR"
uv python install 3.11.11
uv venv --python 3.11.11 "$task_root/envs/da3"
uv pip install --python "$task_root/envs/da3/bin/python" \
  'torch==2.8.0' 'torchvision==0.23.0' 'xformers==0.0.32.post2' \
  --index-url https://download.pytorch.org/whl/cu128
uv pip install --python "$task_root/envs/da3/bin/python" \
  --constraint "$task_root/code/runner/configs/da3_constraints.txt" \
  -e "$task_root/code/Depth-Anything-3" \
  'addict==2.4.0' 'scipy==1.15.3' 'matplotlib==3.10.3'
uv pip check --python "$task_root/envs/da3/bin/python"
uv pip freeze --python "$task_root/envs/da3/bin/python" > "$task_root/provenance/environment.txt"
"$task_root/envs/da3/bin/python" "$task_root/code/runner/scripts/da3/prepare_assets.py" \
  --root "$task_root" --config "$task_root/code/runner/configs/da3_pinned.json"
"$task_root/envs/da3/bin/python" -c 'from depth_anything_3.api import DepthAnything3; from depth_anything_3.bench.evaluator import Evaluator; import torch; print("DA3 imports OK; torch", torch.__version__)'
