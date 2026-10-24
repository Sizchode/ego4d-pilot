#!/usr/bin/env bash
#SBATCH --job-name=amb3r-runtime
#SBATCH --partition=batch
#SBATCH --account=default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/oscar/scratch/zliu328/3dv-reconstruction/logs/%x-%j.out

set -euo pipefail
exec 9>"/tmp/$(id -un).agent-lock"
flock -n 9
module load gcc/11.5.0-wfyo cuda/11.8.0-kuhf
task_root=/oscar/scratch/zliu328/3dv-reconstruction
code="$task_root/code/AMB3R"
env="$task_root/envs/amb3r"
export UV_CACHE_DIR="$task_root/cache/uv"
export UV_PYTHON_INSTALL_DIR="$task_root/envs/python"
export UV_CONCURRENT_DOWNLOADS=1 UV_CONCURRENT_BUILDS=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MAX_JOBS=1
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export TMPDIR="${SLURM_TMPDIR:-$task_root/tmp}"
export FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST=9.0
export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
mkdir -p "$TMPDIR" "$task_root/provenance/amb3r/runtime-$SLURM_JOB_ID"
audit="$task_root/provenance/amb3r/runtime-$SLURM_JOB_ID"
uv venv --python 3.9.21 --allow-existing "$env"
uv pip install --python "$env/bin/python" --index https://download.pytorch.org/whl/cu118 \
  'torch==2.5.0+cu118' 'torchvision==0.20.0+cu118' 'torchaudio==2.5.0+cu118'
uv pip install --python "$env/bin/python" \
  'setuptools==75.8.0' 'wheel==0.45.1' 'ninja==1.11.1.3' 'packaging==24.2' \
  'numpy==1.26.4' 'iopath==0.1.10' 'fvcore==0.1.5.post20221221'
uv pip install --python "$env/bin/python" --no-deps \
  'https://data.pyg.org/whl/torch-2.5.0%2Bcu118/torch_scatter-2.1.2%2Bpt25cu118-cp39-cp39-linux_x86_64.whl'
runtime_config="$task_root/code/runner/configs/amb3r_runtime_pinned.json"
pytorch3d_revision=$("$env/bin/python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["pytorch3d_revision"])' "$runtime_config")
flash_wheel=$("$env/bin/python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["flash_attention_url"])' "$runtime_config")
uv pip install --python "$env/bin/python" --no-build-isolation \
  "git+https://github.com/facebookresearch/pytorch3d.git@$pytorch3d_revision"
uv pip install --python "$env/bin/python" --no-deps "$flash_wheel"
cat > "$audit/constraints.txt" <<'CONSTRAINTS'
torch==2.5.0+cu118
torchvision==0.20.0+cu118
torchaudio==2.5.0+cu118
CONSTRAINTS
uv pip install --python "$env/bin/python" --constraint "$audit/constraints.txt" -r "$code/requirements.txt"
uv pip freeze --python "$env/bin/python" > "$audit/packages.txt"
nvcc --version > "$audit/nvcc.txt"
cd "$code"
"$env/bin/python" "$task_root/code/runner/scripts/amb3r/validate_runtime.py" \
  --code "$code" --checkpoint "$task_root/models/AMB3R/amb3r.pt" \
  --output "$audit/checkpoint-load.json"
