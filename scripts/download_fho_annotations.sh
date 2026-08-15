#!/usr/bin/env bash
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
  echo "AWS credentials are not configured. Create profile '${AWS_PROFILE_NAME}' locally first." >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

# The official CLI applies the FHO benchmark filter. Do not use --yes: the
# reported download size must be reviewed at the confirmation prompt.
exec "${EGO4D_CLI}" \
  --output_directory "${OUTPUT_DIR}" \
  --datasets annotations \
  --benchmarks FHO \
  --version v1 \
  --aws_profile_name "${AWS_PROFILE_NAME}" \
  --no-metadata
