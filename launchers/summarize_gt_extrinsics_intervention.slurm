#!/bin/bash
#SBATCH --job-name=scannetpp-gt-extrinsics-summary
#SBATCH --partition=batch
#SBATCH --account=default
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/oscar/scratch/zliu328/3dv-reconstruction/logs/%x-%j.out
#SBATCH --error=/oscar/scratch/zliu328/3dv-reconstruction/logs/%x-%j.err

set -euo pipefail

CODE_ROOT=/oscar/scratch/zliu328/3dv-reconstruction/code/runner
INTERVENTION_DIR=${1:?intervention directory is required}
OUTPUT_DIR=${2:?output directory is required}

test -f "${INTERVENTION_DIR}/manifest.json"
test -f "${INTERVENTION_DIR}/results.jsonl"

cd "${CODE_ROOT}"
export PYTHONPATH="${CODE_ROOT}:${PYTHONPATH:-}"
python scripts/failure_mining/summarize_gt_extrinsics_intervention.py \
  --intervention-dir "${INTERVENTION_DIR}" \
  --output "${OUTPUT_DIR}"
