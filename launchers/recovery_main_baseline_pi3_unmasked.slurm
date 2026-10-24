#!/usr/bin/env bash
#SBATCH --job-name=recovery-main-pi3-unmasked
#SBATCH --partition=batch
#SBATCH --account=default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/oscar/scratch/zliu328/3dv-reconstruction/logs/%x-%j.out
set -euo pipefail
exec 9>"/tmp/$(id -un).agent-lock"
flock 9
task_root=/oscar/scratch/zliu328/3dv-reconstruction
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
run="$task_root/runs/recovery-main-baseline-pi3-unmasked-$SLURM_JOB_ID"
mkdir -p "$run/source"
for name in export_pi3_unmasked native_state public_canvas geometry_state; do
  cp "$task_root/code/runner/scripts/recovery/$name.py" "$run/source/"
done
cp "$task_root/code/runner/launchers/recovery_main_baseline_pi3_unmasked.slurm" "$run/launcher.slurm"
"$task_root/envs/pi3x/bin/python" "$run/source/export_pi3_unmasked.py" \
  --batch "$task_root/runs/recovery-main-baseline-pi3-7748834/population/batch.json" \
  --output "$run/population"
