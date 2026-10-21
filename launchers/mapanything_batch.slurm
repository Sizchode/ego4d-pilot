#!/usr/bin/env bash
#SBATCH --job-name=mapanything-benchmark
#SBATCH --partition=gpu-he
#SBATCH --account=default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:nvidia_b200:1
#SBATCH --time=24:00:00
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
"$task_root/envs/mapanything/bin/python" -m scripts.mapanything.run_batch \
  --root "$task_root" --config "$task_root/code/runner/configs/mapanything_pinned.json" \
  --verified-run "${1:?pass the reviewed MapAnything smoke-run directory}" \
  --batch-id "mapanything-batch-$SLURM_JOB_ID"
