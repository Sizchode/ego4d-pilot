#!/usr/bin/env bash
#SBATCH --job-name=benchmark-protocol-audit
#SBATCH --partition=batch
#SBATCH --account=default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/oscar/scratch/zliu328/3dv-reconstruction/logs/%x-%j.out
set -euo pipefail
exec 9>"/tmp/$(id -un).agent-lock"
flock 9
task_root=/oscar/scratch/zliu328/3dv-reconstruction
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
run="$task_root/provenance/benchmark-protocol-audit-$SLURM_JOB_ID"
mkdir -p "$run/source"
cp "$task_root/code/runner/scripts/evaluation/"*.py "$run/source/"
cp "$task_root/code/GGPT/evaluation.py" "$run/source/official-evaluation.py"
cp "$task_root/code/runner/launchers/audit_remaining_benchmarks.slurm" "$run/launcher.slurm"
args=(--inventory "$task_root/provenance/main-baseline-evaluation/protocol-audit-inventory.json" --runner "$task_root/code/runner" --ggpt "$task_root/code/GGPT" --root "$task_root" --seed 0)
for dataset in eth3d scannetpp; do
 for method in DA3 VGGT 'π³' MapAnything AMB3R 'DA3→GGPT'; do
  for canvas in public504 official518; do
   "$task_root/envs/da3/bin/python" "$run/source/audit_tandt_protocol.py" "${args[@]}" --dataset "$dataset" --method "$method" --canvas "$canvas" --output "$run/$dataset-$method-$canvas"
  done
 done
done
