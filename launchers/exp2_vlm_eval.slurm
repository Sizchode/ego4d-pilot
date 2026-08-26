#!/usr/bin/env bash
#SBATCH --job-name=exp2_vlm_eval
#SBATCH --partition=gpu-he
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=196G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

if [[ "$#" -lt 1 || "$#" -gt 2 ]]; then
  echo "usage: sbatch $0 OUTPUT_DIR [TASK]" >&2
  exit 2
fi

project_root="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?submit from the repository root or set PROJECT_ROOT}}"
if [[ ! -f "$project_root/experiments/exp2_vlm_eval/evaluate.py" ]]; then
  echo "Experiment 2 entry point not found under $project_root" >&2
  exit 2
fi
cache_root="${MODEL_CACHE_ROOT:-/oscar/scratch/$USER/temporal-transfer/models}"
environment_root="${EXPERIMENT_ENV:-/oscar/scratch/$USER/temporal-transfer/env-cu128}"
export HF_HOME="$cache_root/huggingface"
export TORCH_HOME="$cache_root/torch"
cd "$project_root"

command=("$environment_root/bin/python" experiments/exp2_vlm_eval/evaluate.py --output-dir "$1")
if [[ "$#" -eq 2 ]]; then
  command+=(--tasks "$2")
fi
"${command[@]}"
