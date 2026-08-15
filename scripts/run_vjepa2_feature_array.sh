#!/usr/bin/env bash
# Resolve one official OSCC/PNR video UID from a Slurm array index.
set -euo pipefail

readonly PROJECT_ROOT="${EGO4D_PILOT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
readonly SCRATCH_ROOT="${EGO4D_PILOT_SCRATCH:-/oscar/scratch/${USER}/ego4d-state-change}"
readonly UID_MANIFEST="${SCRATCH_ROOT}/data/manifests/oscc_pnr_train_val_video_uids.txt"
readonly ARRAY_INDEX="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
VIDEO_UID="$(sed -n "${ARRAY_INDEX}p" "${UID_MANIFEST}")"
readonly VIDEO_UID
if [[ -z "${VIDEO_UID}" ]]; then
  echo "No video UID at manifest line ${ARRAY_INDEX}." >&2
  exit 2
fi

exec "${PROJECT_ROOT}/scripts/extract_vjepa2_for_video.sh" "${VIDEO_UID}"
