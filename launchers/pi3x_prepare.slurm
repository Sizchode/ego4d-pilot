#!/usr/bin/env bash
#SBATCH --job-name=pi3x-prepare
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

code_revision=$(python3 -c 'import json; print(json.load(open("'"$runner_root"'/configs/pi3x_pinned.json"))["code_revision"])')
if [[ ! -d "$task_root/code/Pi3/.git" ]]; then
  git clone --filter=blob:none --no-checkout https://github.com/yyfz/Pi3.git "$task_root/code/Pi3"
fi
git -C "$task_root/code/Pi3" fetch origin "$code_revision"
git -C "$task_root/code/Pi3" checkout --detach "$code_revision"
git -C "$task_root/code/Pi3" diff --exit-code HEAD

uv python install 3.11.11
uv venv --python 3.11.11 "$task_root/envs/pi3x"
uv pip install --python "$task_root/envs/pi3x/bin/python" \
  'torch==2.8.0' 'torchvision==0.23.0' \
  --index-url https://download.pytorch.org/whl/cu128
uv pip install --python "$task_root/envs/pi3x/bin/python" \
  --constraint "$runner_root/configs/pi3x_constraints.txt" \
  -e "$task_root/code/Pi3" -e "$task_root/code/Depth-Anything-3" \
  'pillow==11.2.1' 'plyfile==1.1.2' 'opencv-python==4.11.0.86' \
  'huggingface-hub==0.36.0' 'open3d==0.19.0' 'scipy==1.15.3' 'matplotlib==3.10.3'
uv pip check --python "$task_root/envs/pi3x/bin/python"
uv pip freeze --python "$task_root/envs/pi3x/bin/python" > "$task_root/provenance/pi3x_environment.txt"
"$task_root/envs/pi3x/bin/python" "$runner_root/scripts/pi3x/prepare_assets.py" \
  --root "$task_root" --config "$runner_root/configs/pi3x_pinned.json"
PYTHONPATH="$runner_root:$task_root/code/Pi3:$task_root/code/Depth-Anything-3/src" \
  "$task_root/envs/pi3x/bin/python" -c 'from pi3.models.pi3x import Pi3X; from depth_anything_3.bench.evaluator import Evaluator; import torch; print("Pi3X and benchmark imports OK; torch", torch.__version__)'
