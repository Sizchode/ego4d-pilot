#!/usr/bin/env bash
#SBATCH --job-name=snapshot-baseline-batches
#SBATCH --partition=batch
#SBATCH --account=default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/oscar/scratch/zliu328/3dv-reconstruction/logs/%x-%j.out

set -euo pipefail
task_root=/oscar/scratch/zliu328/3dv-reconstruction
output="${1:?pass a new immutable snapshot directory}"
export PYTHONPATH="$task_root/code/runner"
cd "$task_root/code/runner"
"$task_root/envs/da3/bin/python" \
  -m scripts.failure_mining.snapshot_baseline_batches \
  --batch "da3=$task_root/runs/batch-7649571" \
  --batch "vggt=$task_root/runs/vggt-batch-7669620" \
  --batch "pi3x=$task_root/runs/pi3x-batch-7670606" \
  --batch "mapanything=$task_root/runs/mapanything-batch-7672161" \
  --output "$output"
