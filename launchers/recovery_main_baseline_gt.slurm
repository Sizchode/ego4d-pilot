#!/usr/bin/env bash
#SBATCH --job-name=recovery-main-baseline-gt
#SBATCH --partition=batch
#SBATCH --account=default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/oscar/scratch/zliu328/3dv-reconstruction/logs/%x-%j.out
set -euo pipefail
exec 9>"/tmp/$(id -un).agent-lock"
flock 9
task_root=/oscar/scratch/zliu328/3dv-reconstruction
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
run="$task_root/provenance/recovery-main-baseline-gt-$SLURM_JOB_ID"
mkdir -p "$run/source"
for name in prepare_main_evaluator_gt stage_pinhole_gt gt_canvas public_canvas geometry_state native_state; do
  cp "$task_root/code/runner/scripts/recovery/$name.py" "$run/source/"
done
cp "$task_root/code/runner/launchers/recovery_main_baseline_gt.slurm" "$run/launcher.slurm"
"$task_root/envs/da3/bin/python" "$run/source/prepare_main_evaluator_gt.py" \
  --plan "$task_root/provenance/main-baseline-evaluation/da3-plan.json" \
  --source-root "$task_root/datasets/GGPT_eval_16/evaluator_gt" \
  --output "$run/gt"
