#!/usr/bin/env bash
# Map a zero-based Slurm array index to a contiguous manifest-line group.
set -euo pipefail

readonly ARRAY_INDEX="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
readonly FIRST_UID_LINE="${FIRST_UID_LINE:?FIRST_UID_LINE is required}"
readonly LAST_UID_LINE="${LAST_UID_LINE:-717}"
readonly GROUP_SIZE="${GROUP_SIZE:-4}"
readonly PROJECT_ROOT="${EGO4D_PILOT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

first_line=$((FIRST_UID_LINE + ARRAY_INDEX * GROUP_SIZE))
last_line=$((first_line + GROUP_SIZE - 1))
if (( last_line > LAST_UID_LINE )); then
  last_line="${LAST_UID_LINE}"
fi
if (( first_line > LAST_UID_LINE )); then
  echo "Array index ${ARRAY_INDEX} begins beyond manifest line ${LAST_UID_LINE}." >&2
  exit 2
fi

exec "${PROJECT_ROOT}/scripts/extract_vjepa2_video_group.sh" \
  "${first_line}" "${last_line}"
