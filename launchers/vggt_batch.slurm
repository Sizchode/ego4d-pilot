#!/usr/bin/env bash
#SBATCH --job-name=vggt-benchmark
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
export MPLBACKEND=Agg
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TMPDIR="${SLURM_TMPDIR:-$task_root/tmp}"
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export PYTHONPATH="$task_root/code/runner:$task_root/code/VGGT:$task_root/code/Depth-Anything-3/src"
cd "$task_root/code/runner"
"$task_root/envs/vggt/bin/python" -m scripts.vggt.run_batch \
  --root "$task_root" --config "$task_root/code/runner/configs/vggt_pinned.json" \
  --verified-run "${1:?pass the reviewed VGGT smoke-run directory}" \
  --batch-id "vggt-batch-$SLURM_JOB_ID"
