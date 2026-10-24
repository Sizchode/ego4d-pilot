#!/usr/bin/env bash
#SBATCH --job-name=vggt-prepare
#SBATCH --partition=batch
#SBATCH --account=default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --output=/oscar/scratch/zliu328/3dv-reconstruction/logs/%x-%j.out

set -euo pipefail
task_root=/oscar/scratch/zliu328/3dv-reconstruction
runner_root="$task_root/code/runner"
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

code_revision=$(python3 -c 'import json; print(json.load(open("'"$runner_root"'/configs/vggt_pinned.json"))["code_revision"])')
if [[ ! -d "$task_root/code/VGGT/.git" ]]; then
  git clone --filter=blob:none --no-checkout https://github.com/facebookresearch/vggt.git "$task_root/code/VGGT"
fi
git -C "$task_root/code/VGGT" fetch origin "$code_revision"
git -C "$task_root/code/VGGT" checkout --detach "$code_revision"
git -C "$task_root/code/VGGT" diff --exit-code HEAD

uv python install 3.11.11
uv venv --python 3.11.11 "$task_root/envs/vggt"
uv pip install --python "$task_root/envs/vggt/bin/python" \
  'torch==2.8.0' 'torchvision==0.23.0' \
  --index-url https://download.pytorch.org/whl/cu128
uv pip install --python "$task_root/envs/vggt/bin/python" \
  --constraint "$runner_root/configs/vggt_constraints.txt" \
  -e "$task_root/code/VGGT" -e "$task_root/code/Depth-Anything-3" \
  'addict==2.4.0' 'scipy==1.15.3' 'matplotlib==3.10.3'
uv pip check --python "$task_root/envs/vggt/bin/python"
uv pip freeze --python "$task_root/envs/vggt/bin/python" > "$task_root/provenance/vggt_environment.txt"
"$task_root/envs/vggt/bin/python" "$runner_root/scripts/vggt/prepare_assets.py" \
  --root "$task_root" --config "$runner_root/configs/vggt_pinned.json"
PYTHONPATH="$runner_root:$task_root/code/VGGT:$task_root/code/Depth-Anything-3/src" \
  "$task_root/envs/vggt/bin/python" -c 'from vggt.models.vggt import VGGT; from depth_anything_3.bench.evaluator import Evaluator; import torch; print("VGGT and benchmark imports OK; torch", torch.__version__)'
