#!/usr/bin/env bash
#SBATCH --job-name=main-benchmark-score
#SBATCH --partition=batch
#SBATCH --account=default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/oscar/scratch/zliu328/3dv-reconstruction/logs/%x-%j.out
set -euo pipefail
: "${DATASET:?eth3d / scannetpp / tandt；一次一个数据集}"
[[ "$DATASET" =~ ^(eth3d|scannetpp|tandt)$ ]]
task_root=/oscar/scratch/zliu328/3dv-reconstruction
mkdir -p "$task_root/provenance/locks"
exec 9>"$task_root/provenance/locks/score-$DATASET.agent-lock"
flock -n 9
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
run="$task_root/provenance/main-benchmark-score-$DATASET-$SLURM_JOB_ID"
mkdir -p "$run/source"
cp "$task_root/code/runner/scripts/evaluation/"*.py "$run/source/"
cp "$task_root/code/GGPT/evaluation.py" "$run/source/official-evaluation.py"
cp "$task_root/code/runner/launchers/score_benchmarks.slurm" "$run/launcher.slurm"
args=(--inventory "$task_root/provenance/main-baseline-evaluation/protocol-audit-inventory.json" --runner "$task_root/code/runner" --ggpt "$task_root/code/GGPT" --root "$task_root" --seed 0 --dataset "$DATASET" --canvas official518)
# 每次评分仍校验旧档案可复现；不再额外重复运行 public504 整套对照。
methods=(DA3 VGGT 'π³' MapAnything AMB3R 'DA3→GGPT')
for method in "${methods[@]}"; do
 "$task_root/envs/da3/bin/python" "$run/source/audit_tandt_protocol.py" "${args[@]}" --method "$method" --output "$run/$method"
done
"$task_root/envs/da3/bin/python" - "$run" "$DATASET" "$task_root/provenance/main-baseline-evaluation/protocol-audit-inventory.json" "${methods[@]}" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, str(Path(sys.argv[1]) / 'source'))
from audit_tandt_protocol import summarize_benchmark
summarize_benchmark(sys.argv[1], sys.argv[3], sys.argv[2], sys.argv[4:])
PY
