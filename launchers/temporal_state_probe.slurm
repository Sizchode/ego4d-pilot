#!/usr/bin/env bash
#SBATCH --job-name=temporal_state_probe
#SBATCH --partition=gpu-he
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

if [[ "$#" -lt 5 || "$#" -gt 6 ]]; then
  echo "usage: sbatch $0 MODEL TRAIN_MANIFEST VAL_MANIFEST DATASET OUTPUT_ROOT [PROMPTS_FILE]" >&2
  exit 2
fi

project_root="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?submit from repository root or set PROJECT_ROOT}}"
environment_root="${EXPERIMENT_ENV:-/oscar/scratch/$USER/temporal-transfer/env-cu128}"
cache_root="${MODEL_CACHE_ROOT:-/oscar/scratch/$USER/temporal-transfer/models}"
export HF_HOME="$cache_root/huggingface"
export TORCH_HOME="$cache_root/torch"
cd "$project_root"

model="$1"
train_manifest="$2"
val_manifest="$3"
dataset="$4"
output_root="$5"
prompts_file="${6:-}"
config="experiments/temporal_state/config.yaml"

readouts=(compact distributed)
if [[ "$model" == "vjepa2_vitl" ]]; then
  readouts=(distributed)
fi

train_features="$output_root/features/${dataset}/${model}/train.pt"
val_features="$output_root/features/${dataset}/${model}/val.pt"
if [[ ! -f "$train_features" ]]; then
  "$environment_root/bin/python" -m experiments.temporal_state.extract_features \
      --config "$config" --model "$model" --manifest "$train_manifest" \
      --output "$train_features" --prompts-file "$prompts_file" --readouts "${readouts[@]}"
fi
if [[ ! -f "$val_features" ]]; then
  "$environment_root/bin/python" -m experiments.temporal_state.extract_features \
      --config "$config" --model "$model" --manifest "$val_manifest" \
      --output "$val_features" --prompts-file "$prompts_file" --readouts "${readouts[@]}"
fi
for readout in "${readouts[@]}"; do
  "$environment_root/bin/python" -m experiments.temporal_state.train_probe \
    --config "$config" --train-features "$train_features" --val-features "$val_features" \
    --readout "$readout" --device cuda --output "$output_root/probes/${dataset}/${model}/${readout}.json"
done
