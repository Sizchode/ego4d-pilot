#!/usr/bin/env bash
#SBATCH --job-name=area3d-vlm-assets
#SBATCH --partition=batch
#SBATCH --account=default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --output=/oscar/scratch/zliu328/3dv-reconstruction/logs/%x-%j.out
set -euo pipefail
R=/oscar/scratch/zliu328/3dv-reconstruction
exec 9>"$R/provenance/area3d-original/vlm-assets.agent-lock"
flock -n 9 || exit 75
export HF_HUB_DISABLE_XET=1 HF_HUB_DISABLE_PROGRESS_BARS=1 HF_HUB_DOWNLOAD_TIMEOUT=120
export HF_HOME="$R/cache/huggingface" TMPDIR="$R/tmp" PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
"$R/envs/vggt/bin/python" "$R/code/runner/scripts/recovery/prepare_area3d_vlm.py" \
  --manifest "$R/provenance/area3d-original/internvl-download.json" \
  --destination "$R/models/InternVL2_5-8B" --maximum-usage-gb 540 \
  --reserved-other-growth-gb 24 \
  --receipt "$R/provenance/area3d-original/internvl-assets.json"
