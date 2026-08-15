#!/usr/bin/env bash
# Reproduce EgoPack's single-task Temporal Graph baseline with Omnivore features.
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: $0 <oscc|pnr> <65|72|97> [omnivore_video_swinl|vjepa2_vitl16]" >&2
  exit 2
fi

readonly TASK="$1"
readonly SEED="$2"
readonly FEATURES="${3:-omnivore_video_swinl}"
case "${TASK}" in
  oscc|pnr) ;;
  *) echo "Unsupported task: ${TASK}" >&2; exit 2 ;;
esac
case "${FEATURES}" in
  omnivore_video_swinl|vjepa2_vitl16) ;;
  *) echo "Unsupported feature backbone: ${FEATURES}" >&2; exit 2 ;;
esac
case "${SEED}" in
  65|72|97) ;;
  *) echo "Unsupported paper seed: ${SEED}" >&2; exit 2 ;;
esac

readonly PROJECT_ROOT="${EGO4D_PILOT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
readonly PROJECT_SCRATCH="${EGO4D_PILOT_SCRATCH:-/oscar/scratch/${USER}/ego4d-state-change}"
readonly EGOPACK_ROOT="${EGOPACK_ROOT:-${PROJECT_ROOT}/third_party/EgoPack}"
readonly PYTHON="${PROJECT_SCRATCH}/envs/egopack/bin/python"
readonly RUN_ID="${SLURM_JOB_ID:-manual}"
readonly RUN_DIR="${PROJECT_SCRATCH}/runs/temporal_graph/${FEATURES}/${TASK}/seed_${SEED}/job_${RUN_ID}"

mkdir -p "${RUN_DIR}/wandb" "${RUN_DIR}/wandb-cache" "${RUN_DIR}/hydra"
export WANDB_MODE=offline
export WANDB_DIR="${RUN_DIR}/wandb"
export WANDB_CACHE_DIR="${RUN_DIR}/wandb-cache"
export WANDB__EXECUTABLE="${PYTHON}"
export HYDRA_FULL_ERROR=1

task_overrides=()
if [[ "${TASK}" == "pnr" ]]; then
  # EgoPack supplementary, Appendix A: PNR uses 16 temporal sub-segments.
  # The released dataset_pnr YAML says 3 and the sweep does not override it.
  task_overrides+=(dataset_pnr.num_segments=16)
fi

cd "${EGOPACK_ROOT}"
exec "${PYTHON}" main_temporal.py \
  seed="${SEED}" \
  wandb_name_pattern="TemporalGraph-${TASK}-seed-${SEED}" \
  artifact_prefix=TemporalGraph \
  k=1 \
  model=graph \
  model/temporal_pooling=trn \
  model.temporal_pooling.hidden_size=1024 \
  model.temporal_pooling.dropout=0.5 \
  model.hidden_size=1024 \
  model.depth=3 \
  num_workers=1 \
  batch_size=16 \
  save_model=False \
  enable_graphone=False \
  enabled_tasks="[${TASK}]" \
  dataset_recognition.features="${FEATURES}" \
  dataset_oscc.features="${FEATURES}" \
  dataset_lta.features="${FEATURES}" \
  dataset_pnr.features="${FEATURES}" \
  num_epochs=10 \
  optimizer._target_=torch.optim.Adam \
  optimizer.lr=1e-6 \
  "${task_overrides[@]}" \
  hydra.run.dir="${RUN_DIR}/hydra"
