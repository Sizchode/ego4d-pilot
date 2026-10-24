#!/usr/bin/env bash
#SBATCH --job-name=ggpt_prepare
#SBATCH --partition=batch
#SBATCH --account=default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=48G
#SBATCH --time=10:00:00
#SBATCH --output=/oscar/scratch/zliu328/3dv-reconstruction/logs/%x-%j.out
set -euo pipefail
exec 9>"/tmp/$(id -un).agent-lock"
flock -n 9

# ggpt_prepare.slurm
(
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

)

# ggpt_runtime271_prepare.slurm
(
task_root=/oscar/scratch/zliu328/3dv-reconstruction
env="$task_root/envs/ggpt-torch271"
export UV_CACHE_DIR="$task_root/cache/uv"
export UV_CONCURRENT_DOWNLOADS=1 UV_CONCURRENT_BUILDS=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MAX_JOBS=1
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export TMPDIR="${SLURM_TMPDIR:-$task_root/tmp}"
mkdir -p "$TMPDIR" "$task_root/provenance/ggpt/runtime271-$SLURM_JOB_ID"
audit="$task_root/provenance/ggpt/runtime271-$SLURM_JOB_ID"
uv venv --python "$task_root/envs/da3/bin/python" "$env"
uv pip install --python "$env/bin/python" \
  --index https://download.pytorch.org/whl/cu126 \
  'torch==2.7.1+cu126' 'torchvision==0.22.1+cu126'
uv pip install --python "$env/bin/python" \
  'numpy==2.2.6' 'Pillow==12.0.0' 'huggingface_hub==1.2.0' \
  'einops==0.8.1' 'pycolmap==3.12.0' 'hydra-core==1.3.7' \
  'opencv-python==4.12.0.88' 'spconv-cu126==2.3.8' \
  'torch-geometric==2.8.1' 'timm==1.0.30' 'plyfile==1.1.5' \
  'addict==2.4.0' 'gin-config==0.5.0' 'rich==14.2.0' \
  'h5py==3.15.1' 'scipy==1.16.3' 'matplotlib==3.10.7' \
  'safetensors==0.6.2' 'pyyaml==6.0.3' 'tqdm==4.67.1' 'yapf==0.43.0'
uv pip install --python "$env/bin/python" --no-deps \
  'https://data.pyg.org/whl/torch-2.7.0%2Bcu126/torch_scatter-2.1.2%2Bpt27cu126-cp311-cp311-linux_x86_64.whl' \
  'https://data.pyg.org/whl/torch-2.7.0%2Bcu126/torch_sparse-0.6.18%2Bpt27cu126-cp311-cp311-linux_x86_64.whl' \
  'https://data.pyg.org/whl/torch-2.7.0%2Bcu126/torch_cluster-1.6.3%2Bpt27cu126-cp311-cp311-linux_x86_64.whl' \
  'https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3.post2/flash_attn-2.8.3.post2%2Bcu12torch2.7cxx11abiTRUE-cp311-cp311-linux_x86_64.whl'
# Dependencies are installed explicitly above. RoMaV2 requests the Python 3.6
# dataclasses backport unconditionally; Python 3.11 already provides dataclasses.
# No matcher or reconstruction code is changed to work around that metadata.
uv pip install --python "$env/bin/python" --no-deps \
  -e "$task_root/code/GGPT/RoMaV2" -e "$task_root/code/GGPT/Pointcept"
lightglue_revision=$("$env/bin/python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["lightglue"]["revision"])' \
  "$task_root/code/runner/configs/ggpt_runtime_pinned.json")
uv pip install --python "$env/bin/python" \
  "git+https://github.com/cvg/LightGlue.git@$lightglue_revision"
"$env/bin/python" -c 'import torch; assert torch.__version__ == "2.7.1+cu126"'
uv pip freeze --python "$env/bin/python" > "$audit/packages.txt"
cd "$task_root/code/GGPT"
"$env/bin/python" "$task_root/code/runner/scripts/ggpt/validate_runtime.py" \
  --code "$task_root/code/GGPT" --checkpoint "$task_root/models/GGPT/model.step228000.pth" \
  --output "$audit/checkpoint-load.json"

)

# ggpt_matchers_prepare.slurm
(
task_root=/oscar/scratch/zliu328/3dv-reconstruction
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export HF_HUB_DISABLE_XET=1 HF_HUB_DISABLE_TELEMETRY=1
export TMPDIR="${SLURM_TMPDIR:-$task_root/tmp}"
mkdir -p "$TMPDIR"
cd "$task_root/code/GGPT"
"$task_root/envs/ggpt-torch271/bin/python" "$task_root/code/runner/scripts/ggpt/prepare_matchers.py" \
  --code "$task_root/code/GGPT" --cache "$task_root/models/GGPT-matchers" \
  --output "$task_root/provenance/ggpt/matchers-$SLURM_JOB_ID.json"

)
