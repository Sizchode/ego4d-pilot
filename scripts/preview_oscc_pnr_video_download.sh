#!/usr/bin/env bash
set -euo pipefail

readonly PROJECT_SCRATCH="${EGO4D_PILOT_SCRATCH:?Set EGO4D_PILOT_SCRATCH to the experiment scratch directory}"
readonly EGO4D_CLI="${PROJECT_SCRATCH}/envs/ego4d-cli-pip/bin/ego4d"
readonly OUTPUT_DIR="${PROJECT_SCRATCH}/data/ego4d"
readonly UID_FILE="${EGO4D_UID_FILE:-${PROJECT_SCRATCH}/data/manifests/oscc_pnr_train_val_video_uids.txt}"
readonly AWS_PROFILE_NAME="${AWS_PROFILE_NAME:-ego4d}"
readonly VIDEO_DATASET="${EGO4D_VIDEO_DATASET:-full_scale}"
readonly LOG_FILE="${PROJECT_SCRATCH}/logs/oscc_pnr_${VIDEO_DATASET}_size_preview.log"

case "${VIDEO_DATASET}" in
  full_scale|video_540ss|omnivore_video_swinl) ;;
  *)
    echo "Unsupported Ego4D dataset: ${VIDEO_DATASET}" >&2
    exit 1
    ;;
esac

if [[ ! -x "${EGO4D_CLI}" ]]; then
  echo "Ego4D CLI is missing: ${EGO4D_CLI}" >&2
  exit 1
fi
if [[ ! -s "${UID_FILE}" ]]; then
  echo "UID manifest is missing or empty: ${UID_FILE}" >&2
  exit 1
fi
if [[ ! -r "${HOME}/.aws/credentials" ]]; then
  echo "AWS credentials are not configured for profile '${AWS_PROFILE_NAME}'." >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}" "$(dirname "${LOG_FILE}")"

# Ego4D CLI 1.7.3 rejects UID filters for feature datasets. Canonical-video
# previews use the annotation-derived UID manifest; the published Omnivore
# feature release must therefore be previewed in full.
dataset_filter=()
if [[ "${VIDEO_DATASET}" != "omnivore_video_swinl" ]]; then
  dataset_filter=(--video_uid_file "${UID_FILE}")
fi

# Sending "n" makes this a size-only run.
printf 'n\n' | "${EGO4D_CLI}" \
  --output_directory "${OUTPUT_DIR}" \
  --datasets "${VIDEO_DATASET}" \
  --version v1 \
  "${dataset_filter[@]}" \
  --aws_profile_name "${AWS_PROFILE_NAME}" \
  --no-metadata | tee "${LOG_FILE}"

if ! grep -q "Aborting the download operation" "${LOG_FILE}"; then
  echo "The CLI did not reach its normal size-preview abort; inspect ${LOG_FILE}." >&2
  exit 1
fi
