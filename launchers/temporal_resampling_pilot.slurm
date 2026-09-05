#!/usr/bin/env bash
#SBATCH --job-name=temporal_resampling
#SBATCH --partition=gpu-he
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

if [[ "$#" -lt 5 || "$#" -gt 6 ]]; then
  echo "usage: sbatch $0 MODEL MANIFEST CANONICAL_SCHEDULES OUTPUT_DIR OFFSET_SECONDS [PROMPTS_FILE]" >&2
  exit 2
fi

project_root="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?submit from repository root or set PROJECT_ROOT}}"
environment_root="${EXPERIMENT_ENV:-/oscar/scratch/$USER/temporal-transfer/env-cu128}"
cache_root="${MODEL_CACHE_ROOT:-/oscar/scratch/$USER/temporal-transfer/models}"
export HF_HOME="$cache_root/huggingface"
export TORCH_HOME="$cache_root/torch"
cd "$project_root"

model="$1"
manifest="$2"
canonical_schedules="$3"
output_dir="$4"
offset_seconds="$5"
prompts_file="${6:-}"

prompt_args=()
if [[ -n "$prompts_file" ]]; then
  prompt_args=(--prompts-file "$prompts_file")
fi

"$environment_root/bin/python" -m scripts.extract_features \
  --config configs/temporal_resampling.yaml \
  --model "$model" \
  --manifest "$manifest" \
  --canonical-schedules "$canonical_schedules" \
  --output-dir "$output_dir/features" \
  --offset-seconds "$offset_seconds" \
  --conditions C0 C2 \
  "${prompt_args[@]}"
