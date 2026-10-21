#!/usr/bin/env bash
#SBATCH --job-name=final-failure-mining
#SBATCH --partition=batch
#SBATCH --account=default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/oscar/scratch/zliu328/3dv-reconstruction/logs/%x-%j.out

set -euo pipefail
task_root=/oscar/scratch/zliu328/3dv-reconstruction
cross_dir="${1:?pass standard cross-model output directory}"
output="${2:?pass a new failure-mining output directory}"
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export PYTHONPATH="$task_root/code/runner"
cd "$task_root/code/runner"
"$task_root/envs/da3/bin/python" -m scripts.failure_mining.analyze_final_cross_model \
  --cross-dir "$cross_dir" \
  --metric-left metric.pose.auc30 \
  --metric-right metric.recon_unposed.fscore \
  --evidence-field fusion.retained_ratio \
  --evidence-field diagnostic.projectable_ratio \
  --evidence-field diagnostic.absolute_relative_z.p95 \
  --evidence-field diagnostic.roundtrip_pixels.p95 \
  --output "$output"
