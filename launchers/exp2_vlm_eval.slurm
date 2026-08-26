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

if [[ "$#" -ne 2 ]]; then
  echo "usage: sbatch $0 MODEL_ROOT OUTPUT_DIR" >&2
  exit 2
fi

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

python experiments/exp2_vlm_eval/evaluate.py --model-root "$1" --output-dir "$2"
