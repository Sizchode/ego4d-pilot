#!/usr/bin/env bash
#SBATCH --job-name=cross_sampling_probe
#SBATCH --partition=gpu-he
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

if [[ "$#" -ne 8 ]]; then
  echo "usage: sbatch $0 MODEL READOUT REPRESENTATION TRAIN_C0 VAL_C0 VAL_C1 VAL_C2 VAL_C3" >&2
  exit 2
fi

project_root="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?submit from repository root or set PROJECT_ROOT}}"
environment_root="${EXPERIMENT_ENV:-/oscar/scratch/$USER/temporal-transfer/env-cu128}"
cd "$project_root"

"$environment_root/bin/python" -m scripts.train_cross_sampling_probe \
  --model "$1" \
  --readout "$2" \
  --representation "$3" \
  --train-c0 "$4" \
  --val-c0 "$5" \
  --val-c1 "$6" \
  --val-c2 "$7" \
  --val-c3 "$8" \
  --output "${OUTPUT_FILE:?set OUTPUT_FILE to the result JSON path}"
