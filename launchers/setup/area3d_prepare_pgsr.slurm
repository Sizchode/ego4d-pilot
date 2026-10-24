#!/usr/bin/env bash
#SBATCH --job-name=area3d-pgsr-runtime
#SBATCH --partition=batch
#SBATCH --account=default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/oscar/scratch/zliu328/3dv-reconstruction/logs/%x-%j.out
set -euo pipefail
R=/oscar/scratch/zliu328/3dv-reconstruction
exec 9>"$R/provenance/area3d-original/pgsr-runtime.agent-lock"
flock -n 9 || exit 75
module load gcc/11.5.0-wfyo cuda/12.9.0-cinr
export PYTHONNOUSERSITE=1 UV_NO_CACHE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MAX_JOBS=4 FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST=10.0
export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
export TMPDIR="$R/tmp/area3d-pgsr-$SLURM_JOB_ID"
BASE="$R/envs/area3d/bin/python"
P="$R/envs/pgsr/bin/python"
export PATH="$R/envs/pgsr/bin:$PATH"
OUT="$R/provenance/area3d-original/pgsr-runtime"
mkdir -p "$TMPDIR" "$OUT"
cleanup() { rm -rf -- "$TMPDIR"; }
trap cleanup EXIT
"$BASE" - <<'PY'
import subprocess,getpass
rows=[x.split() for x in subprocess.check_output(['checkquota'],text=True,timeout=60).splitlines() if x.split()[:2]==[getpass.getuser(),'/oscar/scratch']]
assert len(rows)==1 and float(rows[0][2])+6+8<=540,'PGSR install peak6GB + reserve8GB exceeds budget'
PY
test ! -e "$R/envs/pgsr"
"$BASE" -m venv --system-site-packages "$R/envs/pgsr"
"$P" -m pip list --format=freeze > "$OUT/version-constraints.txt"
"$P" -m pip install --no-cache-dir --constraint "$OUT/version-constraints.txt" \
  'open3d==0.19.0' 'plyfile==1.1.2' 'lpips==0.1.4' 'iopath==0.1.10' 'ninja==1.11.1.4' 'wheel==0.45.1'
git clone --depth 1 --branch v0.7.9 https://github.com/facebookresearch/pytorch3d.git "$TMPDIR/pytorch3d"
"$P" -m pip install --no-cache-dir --no-deps --no-build-isolation "$TMPDIR/pytorch3d"
for name in diff-plane-rasterization simple-knn; do
  cp -a "$R/code/PGSR/submodules/$name" "$TMPDIR/$name"
  "$P" -m pip install --no-cache-dir --no-deps --no-build-isolation "$TMPDIR/$name"
done
cd "$R/code/PGSR"
"$P" - "$OUT" <<'PY'
import json,sys
from pathlib import Path
import torch,pytorch3d,open3d
from simple_knn._C import distCUDA2
from diff_plane_rasterization import _C
import train,render
record={'python_prefix':sys.prefix,'base_prefix':sys.base_prefix,'torch':torch.__version__,'pytorch3d':pytorch3d.__version__,'open3d':open3d.__version__,'pgsr_train_render_import':True,'gpu_execution_validated':False}
(Path(sys.argv[1])/'imports.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record))
PY
"$P" -m pip freeze > "$OUT/after-packages.txt"
