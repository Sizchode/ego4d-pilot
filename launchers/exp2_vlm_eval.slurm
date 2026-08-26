#!/usr/bin/env bash
#SBATCH --job-name=exp2_vlm_eval
#SBATCH --partition=gpu-he
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=196G
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

if [[ "$#" -ne 1 ]]; then
  echo "usage: sbatch $0 OUTPUT_DIR" >&2
  exit 2
fi

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cache_root="${MODEL_CACHE_ROOT:-/oscar/scratch/$USER/temporal-transfer/models}"
environment_root="${EXPERIMENT_ENV:-/oscar/scratch/$USER/temporal-transfer/env}"
export HF_HOME="$cache_root/huggingface"
export TORCH_HOME="$cache_root/torch"
cd "$project_root"

"$environment_root/bin/python" experiments/exp2_vlm_eval/evaluate.py --output-dir "$1"
