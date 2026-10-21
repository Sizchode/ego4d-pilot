#!/usr/bin/env bash
#SBATCH --job-name=mapanything-smoke
#SBATCH --partition=gpu-he
#SBATCH --account=default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:nvidia_b200:1
#SBATCH --time=04:00:00
#SBATCH --output=/oscar/scratch/zliu328/3dv-reconstruction/logs/%x-%j.out

set -euo pipefail
task_root=/oscar/scratch/zliu328/3dv-reconstruction
export HF_HOME="$task_root/cache/huggingface"
export XDG_CACHE_HOME="$task_root/cache"
export TORCH_HOME="$task_root/cache/torch"
export HF_HUB_OFFLINE=1
export http_proxy=http://127.0.0.1:9
export https_proxy=http://127.0.0.1:9
export HTTP_PROXY="$http_proxy"
export HTTPS_PROXY="$https_proxy"
export MPLBACKEND=Agg
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TMPDIR="${SLURM_TMPDIR:-$task_root/tmp}"
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export PYTHONPATH="$task_root/code/runner:$task_root/code/MapAnything:$task_root/code/Depth-Anything-3/src"
cd "$task_root/code/runner"
test "$(readlink -f "$TORCH_HOME/hub/facebookresearch_dinov2_main")" = "$task_root/code/DINOv2"
nvidia-smi
"$task_root/envs/mapanything/bin/python" -m unittest tests.test_da3_geometry tests.test_mapanything_preprocess -v
"$task_root/envs/mapanything/bin/python" -m scripts.mapanything.run_scene \
  --root "$task_root" --config configs/mapanything_pinned.json \
  --dataset hiroom --scene @first --run-id "mapanything-smoke-hiroom-$SLURM_JOB_ID"
