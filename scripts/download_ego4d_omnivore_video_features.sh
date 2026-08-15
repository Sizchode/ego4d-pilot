#!/usr/bin/env bash
# Download the official Ego4D v1 Omnivore Video Swin-L features used by EgoPack.
set -euo pipefail

readonly PROJECT_SCRATCH="${EGO4D_PILOT_SCRATCH:?Set EGO4D_PILOT_SCRATCH to the experiment scratch directory}"
readonly EGO4D_CLI="${PROJECT_SCRATCH}/envs/ego4d-cli-pip/bin/ego4d"
readonly OUTPUT_DIR="${PROJECT_SCRATCH}/data/ego4d"
readonly AWS_PROFILE_NAME="${AWS_PROFILE_NAME:-ego4d}"

if [[ ! -x "${EGO4D_CLI}" ]]; then
  echo "Ego4D CLI is missing: ${EGO4D_CLI}" >&2
  exit 1
fi
if [[ ! -r "${HOME}/.aws/credentials" ]]; then
  echo "AWS credentials are not configured for profile '${AWS_PROFILE_NAME}'." >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

# Ego4D CLI 1.7.3 does not support --video_uid_file for feature datasets.
# EgoPack uses this official feature release directly, so download the complete
# v1 feature set (official preview: 149.6726 GB) and select task videos through
# EgoPack's unmodified annotation-driven dataset class.
"${EGO4D_CLI}" \
  --output_directory "${OUTPUT_DIR}" \
  --datasets omnivore_video_swinl \
  --version v1 \
  --aws_profile_name "${AWS_PROFILE_NAME}" \
  --no-metadata \
  --yes
