#!/usr/bin/env bash
# Download one official Ego4D video, extract both frozen backbones, then remove the MP4.
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <video_uid>" >&2
  exit 2
fi

readonly VIDEO_UID="$1"
readonly PROJECT_ROOT="${EGO4D_PILOT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
readonly SCRATCH_ROOT="${EGO4D_PILOT_SCRATCH:-/oscar/scratch/${USER}/ego4d-state-change}"
export EGO4D_PILOT_ROOT="${PROJECT_ROOT}"
export EGO4D_PILOT_SCRATCH="${SCRATCH_ROOT}"
readonly UID_MANIFEST="${SCRATCH_ROOT}/data/manifests/oscc_pnr_train_val_video_uids.txt"
readonly DOWNLOADER="${SCRATCH_ROOT}/envs/ego4d-cli/bin/ego4d"
readonly PYTHON="${SCRATCH_ROOT}/envs/egopack/bin/python"
readonly EXTRACTOR="${PROJECT_ROOT}/third_party/ego4d/ego4d/features/extract_features.py"
readonly DINO_OUTPUT="${SCRATCH_ROOT}/features/dinov2_vitl14/${VIDEO_UID}.pt"
readonly VJEPA_OUTPUT="${SCRATCH_ROOT}/features/vjepa2_vitl16/${VIDEO_UID}.pt"
readonly STAGING_ROOT="${SCRATCH_ROOT}/data/feature-extraction-staging"
readonly CANONICAL_MANIFEST="${SCRATCH_ROOT}/data/raw-v1/v1/full_scale/manifest.csv"

if ! [[ "${VIDEO_UID}" =~ ^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$ ]]; then
  echo "Invalid Ego4D video UID: ${VIDEO_UID}" >&2
  exit 2
fi
if ! rg -Fxq -- "${VIDEO_UID}" "${UID_MANIFEST}"; then
  echo "UID is not in the released OSCC/PNR train+val manifest: ${VIDEO_UID}" >&2
  exit 2
fi

export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/third_party/ego4d"
mkdir -p "${STAGING_ROOT}" "$(dirname "${DINO_OUTPUT}")" "$(dirname "${VJEPA_OUTPUT}")"

validate_feature() {
  "${PYTHON}" "${PROJECT_ROOT}/scripts/validate_extracted_feature.py" "$1" --dimension 1024
}

if [[ -f "${DINO_OUTPUT}" && -f "${VJEPA_OUTPUT}" ]]; then
  validate_feature "${DINO_OUTPUT}"
  validate_feature "${VJEPA_OUTPUT}"
  echo "Both feature files already exist; no video download is required."
  exit 0
fi

shopt -s nullglob
existing_stages=("${STAGING_ROOT}/${VIDEO_UID}."*)
shopt -u nullglob
if [[ ${#existing_stages[@]} -gt 1 ]]; then
  echo "Multiple staging directories found for ${VIDEO_UID}; refusing an ambiguous resume." >&2
  exit 1
elif [[ ${#existing_stages[@]} -eq 1 ]]; then
  STAGE_DIR="${existing_stages[0]}"
  echo "Resuming ${VIDEO_UID} from ${STAGE_DIR}"
else
  STAGE_DIR="$(mktemp -d "${STAGING_ROOT}/${VIDEO_UID}.XXXXXX")"
fi
readonly STAGE_DIR
echo "Staging ${VIDEO_UID} in ${STAGE_DIR}"

VIDEO_DIR="${STAGE_DIR}/v1/full_scale"
if [[ ! -f "${VIDEO_DIR}/${VIDEO_UID}.mp4" ]]; then
  "${DOWNLOADER}" \
    --output_directory "${STAGE_DIR}" \
    --datasets full_scale \
    --video_uids "${VIDEO_UID}" \
    --version v1 \
    --aws_profile_name ego4d \
    --yes
fi
readonly VIDEO_DIR
if [[ ! -f "${VIDEO_DIR}/${VIDEO_UID}.mp4" ]]; then
  echo "Official video object was not downloaded for ${VIDEO_UID}." >&2
  exit 1
fi

# The full_scale manifest provides the canonical frame fields required by the
# official feature extractor. Retain the global released manifest as fallback.
if ! head -n 1 "${VIDEO_DIR}/manifest.csv" | rg -q 'canonical_num_frames'; then
  mv "${VIDEO_DIR}/manifest.csv" "${VIDEO_DIR}/download_manifest.csv"
  ln -s "${CANONICAL_MANIFEST}" "${VIDEO_DIR}/manifest.csv"
fi

run_extractor() {
  local config_name="$1"
  local output_dir="$2"
  local hydra_dir="$3"
  "${PYTHON}" "${EXTRACTOR}" \
    --config-path "${PROJECT_ROOT}/configs/features" \
    --config-name "${config_name}" \
    "io.video_dir_path=${VIDEO_DIR}" \
    "io.ego4d_download_dir=${STAGE_DIR}" \
    "io.uid_list=[${VIDEO_UID}]" \
    "io.out_path=${output_dir}" \
    "hydra.run.dir=${hydra_dir}"
}

if [[ ! -f "${DINO_OUTPUT}" ]]; then
  run_extractor dinov2_vitl14 "$(dirname "${DINO_OUTPUT}")" "${STAGE_DIR}/hydra-dinov2"
fi
validate_feature "${DINO_OUTPUT}"

if [[ ! -f "${VJEPA_OUTPUT}" ]]; then
  run_extractor vjepa2_vitl16 "$(dirname "${VJEPA_OUTPUT}")" "${STAGE_DIR}/hydra-vjepa2"
fi
validate_feature "${VJEPA_OUTPUT}"

case "${STAGE_DIR}" in
  "${STAGING_ROOT}/${VIDEO_UID}."*) rm -rf -- "${STAGE_DIR}" ;;
  *) echo "Refusing to remove unexpected staging path: ${STAGE_DIR}" >&2; exit 1 ;;
esac
echo "Removed verified staging video for ${VIDEO_UID}."
