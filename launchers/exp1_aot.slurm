#!/usr/bin/env bash
#SBATCH --job-name=exp1_aot
#SBATCH --partition=gpu-he
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=84
#SBATCH --mem=1100G
#SBATCH --gres=gpu:l40s:6
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

if [[ "$#" -ne 5 ]]; then
  echo "usage: sbatch $0 MODEL TRAIN_MANIFEST EVAL_MANIFEST MODEL_ROOT OUTPUT_DIR" >&2
  exit 2
fi

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

torchrun --standalone --nproc-per-node=6 experiments/exp1_aot/train.py \
  --model "$1" \
  --train-manifest "$2" \
  --eval-manifest "$3" \
  --model-root "$4" \
  --output-dir "$5" \
  --workers 12
