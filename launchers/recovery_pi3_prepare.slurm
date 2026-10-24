#!/usr/bin/env bash
#SBATCH --job-name=recovery-pi3-prepare
#SBATCH --partition=batch
#SBATCH --account=default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/oscar/scratch/zliu328/3dv-reconstruction/logs/%x-%j.out

set -euo pipefail
exec 9>"/tmp/$(id -un).agent-lock"
flock -n 9
task_root=/oscar/scratch/zliu328/3dv-reconstruction
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export HF_HOME="$task_root/cache/huggingface"
export HF_HUB_DISABLE_TELEMETRY=1
run="$task_root/provenance/recovery-pi3-prepare-$SLURM_JOB_ID"
mkdir -p "$run/source"
cp "$task_root/code/runner/scripts/recovery/prepare_pi3.py" "$run/source/"
cp "$task_root/code/runner/configs/pi3_pinned.json" "$run/configuration.json"
cp "$task_root/code/runner/launchers/recovery_pi3_prepare.slurm" "$run/launcher.slurm"
"$task_root/envs/pi3x/bin/python" "$run/source/prepare_pi3.py" \
  --root "$task_root" --config "$run/configuration.json" --output "$run/validation"
