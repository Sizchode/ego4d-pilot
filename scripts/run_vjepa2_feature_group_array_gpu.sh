#!/usr/bin/env bash
# Run grouped extraction under the standard-GPU QOS (two GPUs, 12 CPUs total).
set -euo pipefail

export EGO4D_NUM_WORKERS="${EGO4D_NUM_WORKERS:-6}"

readonly PROJECT_ROOT="${EGO4D_PILOT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
exec "${PROJECT_ROOT}/scripts/run_vjepa2_feature_group_array.sh"
