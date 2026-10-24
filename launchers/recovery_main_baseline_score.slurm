#!/usr/bin/env bash
#SBATCH --job-name=recovery-main-baseline-score
#SBATCH --partition=batch
#SBATCH --account=default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH --time=02:00:00
#SBATCH --output=/oscar/scratch/zliu328/3dv-reconstruction/logs/%x-%j.out
set -euo pipefail
exec 9>"/tmp/$(id -un).agent-lock"
flock 9
task_root=/oscar/scratch/zliu328/3dv-reconstruction
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
run="$task_root/provenance/recovery-main-baseline-score-$SLURM_JOB_ID"
mkdir -p "$run/source"
for name in score_main_baseline score_archives pointmap_score public_canvas geometry_state native_state; do
  cp "$task_root/code/runner/scripts/recovery/$name.py" "$run/source/"
done
cp "$task_root/code/runner/launchers/recovery_main_baseline_score.slurm" "$run/launcher.slurm"
cp "${BASELINE_BATCH:?Explicit completed baseline batch required}" "$run/input-batch.json"
cp "$task_root/provenance/main-baseline-evaluation/${PLAN_NAME:?Frozen plan name required}-plan.json" "$run/input-plan.json"
cp "$task_root/provenance/recovery-main-baseline-gt-7763533/gt/preparation.json" "$run/gt-preparation.json"
"$task_root/envs/da3/bin/python" "$run/source/score_main_baseline.py" \
  --batch "$run/input-batch.json" --plan "$run/input-plan.json" \
  --gt-preparation "$run/gt-preparation.json" \
  --expected-method "${EXPECTED_METHOD:?Exact baseline method required}" \
  --output "$run/scores"
