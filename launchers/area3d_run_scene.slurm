#!/usr/bin/env bash
#SBATCH --job-name=area3d-scene
#SBATCH --partition=gpu-he
#SBATCH --account=default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=96G
#SBATCH --gres=gpu:nvidia_b200:1
#SBATCH --time=04:00:00
#SBATCH --output=/oscar/scratch/zliu328/3dv-reconstruction/logs/%x-%j.out
set -euo pipefail
: "${SCENE:?Select scene from the approved AREA3D benchmark}"
R=/oscar/scratch/zliu328/3dv-reconstruction
exec 9>"$R/provenance/area3d-original/scene-evaluation.agent-lock"
flock -n 9 || exit 75
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
export HF_HOME="$R/cache/huggingface" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TORCH_HOME="$R/cache/torch" MPLCONFIGDIR="$R/cache/matplotlib"
export CUDA_CACHE_PATH="$R/cache/cuda" XDG_CACHE_HOME="$R/cache/xdg"
export VGGT_MODEL_PATH="$R/models/VGGT-1B"
unset VGGT_OMEGA_CHECKPOINT
P="$R/envs/area3d/bin/python"
G="$R/envs/pgsr/bin/python"
TOOLS="$R/code/runner/scripts/recovery"
SOURCE="$R/code/AREA3D/src"
OUT="$R/runs/area3d-$SCENE-$SLURM_JOB_ID"
"$P" - "$R" "$SCENE" <<'PY'
import json,sys,getpass,subprocess
from pathlib import Path
r=Path(sys.argv[1]);scene=sys.argv[2]
assets=json.loads((r/'provenance/area3d-original/replica-assets.json').read_text())
assert scene in assets['scenes'],'Scene is outside staged benchmark'
runtime=json.loads((r/'provenance/area3d-original/pgsr-runtime/imports.json').read_text())
assert runtime['pgsr_train_render_import']
rows=[x.split() for x in subprocess.check_output(['checkquota'],text=True,timeout=60).splitlines() if x.split()[:2]==[getpass.getuser(),'/oscar/scratch']]
assert len(rows)==1 and float(rows[0][2])+7+8<=540,'Scene peak7GB+reserve8GB exceeds budget'
PY
mkdir "$OUT"
cp "$R/code/runner/launchers/area3d_run_scene.slurm" "$OUT/launcher.slurm"
printf '%s\n' "$SCENE" > "$OUT/scene.txt"
"$P" "$TOOLS/prepare_area3d_scene_config.py" --source "$SOURCE" \
  --scene "$SOURCE/configs/scenario/scene/$SCENE.yaml" --output "$OUT/scene-camera.yaml"
cd "$SOURCE"
/usr/bin/time -v -o "$OUT/replay-train-resources.txt" "$P" -m envs.replay \
  --config "$OUT/scene-camera.yaml" \
  --pose-file "$R/datasets/AREA3D/initial_15/${SCENE}_seq1/train/traj_w_c.txt" --out-dir "$OUT/initial"
/usr/bin/time -v -o "$OUT/replay-test-resources.txt" "$P" -m envs.replay \
  --config "$OUT/scene-camera.yaml" \
  --pose-file "$R/datasets/AREA3D/initial_15/${SCENE}_seq1/test/traj_w_c.txt" --out-dir "$OUT/test"
"$P" "$TOOLS/verify_area3d_replay.py" --source "$SOURCE" \
  --poses "$R/datasets/AREA3D/initial_15/${SCENE}_seq1/train/traj_w_c.txt" --export "$OUT/initial" --expected-views 15
"$P" "$TOOLS/verify_area3d_replay.py" --source "$SOURCE" \
  --poses "$R/datasets/AREA3D/initial_15/${SCENE}_seq1/test/traj_w_c.txt" --export "$OUT/test" --expected-views 5
cp "$SOURCE/configs/runs/$SCENE.yaml" "$OUT/source-run.yaml"
cp "$SOURCE/configs/policy/"*.yaml "$OUT/"
cd "$OUT"
/usr/bin/time -v -o "$OUT/acquire-resources.txt" "$P" "$TOOLS/run_area3d_audited.py" \
  --source "$SOURCE" --trace "$OUT/vlm-calls.jsonl" --config "configs/runs/$SCENE.yaml" \
  "scene_cfg=$OUT/scene-camera.yaml" "initial_dataset=$OUT/initial" "out_dir=$OUT/acquired" \
  "vlm_model_name=$R/models/InternVL2_5-8B" top_k_views=25 save_depth=true
"$P" "$TOOLS/assemble_area3d_dataset.py" --initial "$OUT/initial" --test "$OUT/test" \
  --acquired "$OUT/acquired" --output "$OUT/pgsr-input" --initial-count 15 --acquired-count 25
cp "$R/code/PGSR/arguments/__init__.py" "$OUT/author-default-arguments.py"
cd "$R/code/PGSR"
/usr/bin/time -v -o "$OUT/train-resources.txt" "$G" train.py --source_path "$OUT/pgsr-input" --model_path "$OUT/model" --eval
/usr/bin/time -v -o "$OUT/render-resources.txt" "$G" render.py --model_path "$OUT/model" --iteration 30000 --skip_train
/usr/bin/time -v -o "$OUT/metrics-resources.txt" "$G" metrics.py --model_paths "$OUT/model"
"$G" "$TOOLS/verify_pgsr_scores.py" --dataset "$OUT/pgsr-input" --model "$OUT/model"
