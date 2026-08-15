#!/usr/bin/env bash
# Download and extract a contiguous group of official OSCC/PNR videos.
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <first_manifest_line> <last_manifest_line>" >&2
  exit 2
fi

readonly FIRST_LINE="$1"
readonly LAST_LINE="$2"
readonly PROJECT_ROOT="${EGO4D_PILOT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
readonly SCRATCH_ROOT="${EGO4D_PILOT_SCRATCH:-/oscar/scratch/${USER}/ego4d-state-change}"
export EGO4D_PILOT_ROOT="${PROJECT_ROOT}"
export EGO4D_PILOT_SCRATCH="${SCRATCH_ROOT}"
readonly UID_MANIFEST="${SCRATCH_ROOT}/data/manifests/oscc_pnr_train_val_video_uids.txt"
readonly DOWNLOADER="${SCRATCH_ROOT}/envs/ego4d-cli/bin/ego4d"
readonly PYTHON="${SCRATCH_ROOT}/envs/egopack/bin/python"
readonly EXTRACTOR="${PROJECT_ROOT}/third_party/ego4d/ego4d/features/extract_features.py"
readonly OUTPUT_DIR="${SCRATCH_ROOT}/features/vjepa2_vitl16"
readonly CANONICAL_MANIFEST="${SCRATCH_ROOT}/data/raw-v1/v1/full_scale/manifest.csv"
readonly LOCAL_JOB_ID="${SLURM_JOB_ID:-manual}-${SLURM_ARRAY_TASK_ID:-0}"
readonly LOCAL_STAGE="/tmp/ego4d-vjepa-group-${LOCAL_JOB_ID}"

if ! [[ "${FIRST_LINE}" =~ ^[0-9]+$ && "${LAST_LINE}" =~ ^[0-9]+$ ]]; then
  echo "Manifest line bounds must be positive integers." >&2
  exit 2
fi
if (( FIRST_LINE < 1 || LAST_LINE < FIRST_LINE )); then
  echo "Invalid manifest line interval: ${FIRST_LINE}-${LAST_LINE}" >&2
  exit 2
fi

mapfile -t requested_uids < <(sed -n "${FIRST_LINE},${LAST_LINE}p" "${UID_MANIFEST}")
missing_uids=()
for uid in "${requested_uids[@]}"; do
  if [[ ! -f "${OUTPUT_DIR}/${uid}.pt" ]]; then
    missing_uids+=("${uid}")
  fi
done
if [[ ${#missing_uids[@]} -eq 0 ]]; then
  echo "All features already exist for manifest lines ${FIRST_LINE}-${LAST_LINE}."
  exit 0
fi

cleanup_local_stage() {
  case "${LOCAL_STAGE}" in
    /tmp/ego4d-vjepa-group-*) rm -rf -- "${LOCAL_STAGE}" ;;
    *) echo "Refusing to remove unexpected local path: ${LOCAL_STAGE}" >&2 ;;
  esac
}
trap cleanup_local_stage EXIT

mkdir -p "${LOCAL_STAGE}" "${OUTPUT_DIR}"
"${DOWNLOADER}" \
  --output_directory "${LOCAL_STAGE}" \
  --datasets full_scale \
  --video_uids "${missing_uids[@]}" \
  --version v1 \
  --aws_profile_name ego4d \
  --yes

readonly VIDEO_DIR="${LOCAL_STAGE}/v1/full_scale"
for uid in "${missing_uids[@]}"; do
  if [[ ! -f "${VIDEO_DIR}/${uid}.mp4" ]]; then
    echo "Official video object was not downloaded for ${uid}." >&2
    exit 1
  fi
done

if [[ -e "${VIDEO_DIR}/manifest.csv" || -L "${VIDEO_DIR}/manifest.csv" ]]; then
  mv "${VIDEO_DIR}/manifest.csv" "${VIDEO_DIR}/download_manifest.csv"
fi
ln -s "${CANONICAL_MANIFEST}" "${VIDEO_DIR}/manifest.csv"

uid_list="$(IFS=,; echo "${missing_uids[*]}")"
readonly uid_list
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/third_party/ego4d"
"${PYTHON}" "${EXTRACTOR}" \
  --config-path "${PROJECT_ROOT}/configs/features" \
  --config-name vjepa2_vitl16 \
  "io.video_dir_path=${VIDEO_DIR}" \
  "io.ego4d_download_dir=${LOCAL_STAGE}" \
  "io.uid_list=[${uid_list}]" \
  "io.out_path=${OUTPUT_DIR}" \
  "hydra.run.dir=${LOCAL_STAGE}/hydra-vjepa2"

for uid in "${missing_uids[@]}"; do
  "${PYTHON}" "${PROJECT_ROOT}/scripts/validate_extracted_feature.py" \
    "${OUTPUT_DIR}/${uid}.pt" --dimension 1024
done
echo "Validated ${#missing_uids[@]} grouped V-JEPA 2 feature files."
