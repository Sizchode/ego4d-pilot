#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 3 ]]; then
  echo "usage: $0 PROTOCOL QWEN_MODEL_DIRECTORY OUTPUT_DIRECTORY" >&2
  exit 2
fi

protocol="$1"
model_directory="$2"
output_directory="$3"
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

python "$project_root/scripts/validate_protocol.py" "$protocol"
mkdir -p "$output_directory"

cd "$project_root/third_party/lmms-eval"
accelerate launch --num_processes 1 -m lmms_eval \
  --model qwen3_vl \
  --model_args "pretrained=$model_directory,max_num_frames=32,min_pixels=50176,max_pixels=1605632,attn_implementation=sdpa" \
  --tasks vitatecs_direction,tvbench,mvbench \
  --batch_size 1 \
  --log_samples \
  --output_path "$output_directory"
