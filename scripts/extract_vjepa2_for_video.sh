#!/usr/bin/env bash
# Stream one official Ego4D full-scale video through V-JEPA 2, then remove it.
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
readonly OUTPUT_DIR="${SCRATCH_ROOT}/features/vjepa2_vitl16"
readonly OUTPUT_FILE="${OUTPUT_DIR}/${VIDEO_UID}.pt"
readonly STAGING_ROOT="${SCRATCH_ROOT}/data/feature-extraction-staging"
readonly CANONICAL_MANIFEST="${SCRATCH_ROOT}/data/raw-v1/v1/full_scale/manifest.csv"
readonly LOCAL_JOB_ID="${SLURM_JOB_ID:-manual}-${SLURM_ARRAY_TASK_ID:-0}"
readonly LOCAL_STAGE="/tmp/ego4d-vjepa-${LOCAL_JOB_ID}-${VIDEO_UID}"

if ! [[ "${VIDEO_UID}" =~ ^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$ ]]; then
  echo "Invalid Ego4D video UID: ${VIDEO_UID}" >&2
  exit 2
fi
if ! rg -Fxq -- "${VIDEO_UID}" "${UID_MANIFEST}"; then
  echo "UID is not in the released OSCC/PNR train+val manifest: ${VIDEO_UID}" >&2
  exit 2
fi

export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/third_party/ego4d"
mkdir -p "${STAGING_ROOT}" "${OUTPUT_DIR}"

validate_feature() {
  "${PYTHON}" "${PROJECT_ROOT}/scripts/validate_extracted_feature.py" "$1" --dimension 1024
}

if [[ -f "${OUTPUT_FILE}" ]]; then
  validate_feature "${OUTPUT_FILE}"
  echo "Validated existing V-JEPA 2 feature file for ${VIDEO_UID}."
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
readonly VIDEO_DIR="${STAGE_DIR}/v1/full_scale"

if [[ ! -f "${VIDEO_DIR}/${VIDEO_UID}.mp4" ]]; then
  "${DOWNLOADER}" \
    --output_directory "${STAGE_DIR}" \
    --datasets full_scale \
    --video_uids "${VIDEO_UID}" \
    --version v1 \
    --aws_profile_name ego4d \
    --yes
fi
if [[ ! -f "${VIDEO_DIR}/${VIDEO_UID}.mp4" ]]; then
  echo "Official video object was not downloaded for ${VIDEO_UID}." >&2
  exit 1
fi

if ! head -n 1 "${VIDEO_DIR}/manifest.csv" | rg -q 'canonical_num_frames'; then
  mv "${VIDEO_DIR}/manifest.csv" "${VIDEO_DIR}/download_manifest.csv"
  ln -s "${CANONICAL_MANIFEST}" "${VIDEO_DIR}/manifest.csv"
fi

# Each DataLoader worker reads the source independently. Keep the immutable
# video on node-local storage so repeated reads do not traverse Oscar NFS.
rm -rf -- "${LOCAL_STAGE}"
mkdir -p "${LOCAL_STAGE}/v1/full_scale"
cp "${VIDEO_DIR}/${VIDEO_UID}.mp4" "${LOCAL_STAGE}/v1/full_scale/"
cp "${STAGE_DIR}/ego4d.json" "${LOCAL_STAGE}/"
ln -s "${CANONICAL_MANIFEST}" "${LOCAL_STAGE}/v1/full_scale/manifest.csv"
readonly EXTRACT_VIDEO_DIR="${LOCAL_STAGE}/v1/full_scale"

"${PYTHON}" "${EXTRACTOR}" \
  --config-path "${PROJECT_ROOT}/configs/features" \
  --config-name vjepa2_vitl16 \
  "io.video_dir_path=${EXTRACT_VIDEO_DIR}" \
  "io.ego4d_download_dir=${LOCAL_STAGE}" \
  "io.uid_list=[${VIDEO_UID}]" \
  "io.out_path=${OUTPUT_DIR}" \
  "hydra.run.dir=${LOCAL_STAGE}/hydra-vjepa2"

validate_feature "${OUTPUT_FILE}"
rm -rf -- "${LOCAL_STAGE}"
case "${STAGE_DIR}" in
  "${STAGING_ROOT}/${VIDEO_UID}."*) rm -rf -- "${STAGE_DIR}" ;;
  *) echo "Refusing to remove unexpected staging path: ${STAGE_DIR}" >&2; exit 1 ;;
esac
echo "Removed verified staging video for ${VIDEO_UID}."
