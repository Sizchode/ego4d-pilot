#!/usr/bin/env bash
#SBATCH --job-name=mapanything-prepare
#SBATCH --partition=batch
#SBATCH --account=default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
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

code_revision=$(python3 -c 'import json; print(json.load(open("'"$runner_root"'/configs/mapanything_pinned.json"))["code_revision"])')
dinov2_repository=$(python3 -c 'import json; print(json.load(open("'"$runner_root"'/configs/mapanything_pinned.json"))["dinov2_repository"])')
dinov2_revision=$(python3 -c 'import json; print(json.load(open("'"$runner_root"'/configs/mapanything_pinned.json"))["dinov2_revision"])')
dinov2_cache_name=$(python3 -c 'import json; print(json.load(open("'"$runner_root"'/configs/mapanything_pinned.json"))["dinov2_torch_hub_cache_name"])')
if [[ ! -d "$task_root/code/MapAnything/.git" ]]; then
  git clone --filter=blob:none --no-checkout https://github.com/facebookresearch/map-anything.git "$task_root/code/MapAnything"
fi
git -C "$task_root/code/MapAnything" fetch origin "$code_revision"
git -C "$task_root/code/MapAnything" checkout --detach "$code_revision"
git -C "$task_root/code/MapAnything" diff --exit-code HEAD

if [[ ! -d "$task_root/code/DINOv2/.git" ]]; then
  git clone --filter=blob:none --no-checkout "$dinov2_repository" "$task_root/code/DINOv2"
fi
git -C "$task_root/code/DINOv2" fetch origin "$dinov2_revision"
git -C "$task_root/code/DINOv2" checkout --detach "$dinov2_revision"
git -C "$task_root/code/DINOv2" diff --exit-code HEAD
mkdir -p "$task_root/cache/torch/hub"
rm -rf "$task_root/cache/torch/hub/$dinov2_cache_name"
ln -s "$task_root/code/DINOv2" "$task_root/cache/torch/hub/$dinov2_cache_name"

uv python install 3.12.7
uv venv --clear --python 3.12.7 "$task_root/envs/mapanything"
uv pip install --python "$task_root/envs/mapanything/bin/python" \
  'torch==2.8.0' 'torchvision==0.23.0' \
  --index-url https://download.pytorch.org/whl/cu128
uv pip install --python "$task_root/envs/mapanything/bin/python" \
  --constraint "$runner_root/configs/mapanything_constraints.txt" \
  -e "$task_root/code/Depth-Anything-3" \
  'huggingface-hub==0.36.0' 'hydra-core' 'natsort' \
  'opencv-python-headless==4.10.0.84' 'orjson' 'pillow-heif' \
  'plyfile' 'python-box' 'requests' 'safetensors==0.6.2' \
  'tensorboard' 'tqdm' 'trimesh' 'uniception==0.1.7' \
  'open3d==0.19.0' 'scipy==1.15.3' 'matplotlib==3.10.3'
uv pip check --python "$task_root/envs/mapanything/bin/python"
uv pip freeze --python "$task_root/envs/mapanything/bin/python" > "$task_root/provenance/mapanything_environment.txt"
"$task_root/envs/mapanything/bin/python" "$runner_root/scripts/mapanything/prepare_assets.py" \
  --root "$task_root" --config "$runner_root/configs/mapanything_pinned.json"
PYTHONPATH="$runner_root:$task_root/code/MapAnything:$task_root/code/Depth-Anything-3/src" \
  "$task_root/envs/mapanything/bin/python" -c 'from mapanything.models import MapAnything; from depth_anything_3.bench.evaluator import Evaluator; import torch; print("MapAnything and benchmark imports OK; torch", torch.__version__)'
