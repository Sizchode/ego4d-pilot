#!/usr/bin/env bash
#SBATCH --job-name=mapanything-pin-dinov2
#SBATCH --partition=batch
#SBATCH --account=default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/oscar/scratch/zliu328/3dv-reconstruction/logs/%x-%j.out

set -euo pipefail
task_root=/oscar/scratch/zliu328/3dv-reconstruction
runner_root="$task_root/code/runner"
export HF_HOME="$task_root/cache/huggingface"
export XDG_CACHE_HOME="$task_root/cache"
export TORCH_HOME="$task_root/cache/torch"
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

dinov2_repository=$(python3 -c 'import json; print(json.load(open("'"$runner_root"'/configs/mapanything_pinned.json"))["dinov2_repository"])')
dinov2_revision=$(python3 -c 'import json; print(json.load(open("'"$runner_root"'/configs/mapanything_pinned.json"))["dinov2_revision"])')
dinov2_cache_name=$(python3 -c 'import json; print(json.load(open("'"$runner_root"'/configs/mapanything_pinned.json"))["dinov2_torch_hub_cache_name"])')

if [[ ! -d "$task_root/code/DINOv2/.git" ]]; then
  git clone --filter=blob:none --no-checkout "$dinov2_repository" "$task_root/code/DINOv2"
fi
git -C "$task_root/code/DINOv2" fetch origin "$dinov2_revision"
git -C "$task_root/code/DINOv2" checkout --detach "$dinov2_revision"
git -C "$task_root/code/DINOv2" diff --exit-code HEAD
test "$(git -C "$task_root/code/DINOv2" rev-parse HEAD)" = "$dinov2_revision"

mkdir -p "$TORCH_HOME/hub"
rm -rf "$TORCH_HOME/hub/$dinov2_cache_name"
ln -s "$task_root/code/DINOv2" "$TORCH_HOME/hub/$dinov2_cache_name"
test "$(readlink -f "$TORCH_HOME/hub/$dinov2_cache_name")" = "$task_root/code/DINOv2"

"$task_root/envs/mapanything/bin/python" "$runner_root/scripts/mapanything/prepare_assets.py" \
  --root "$task_root" --config "$runner_root/configs/mapanything_pinned.json"
