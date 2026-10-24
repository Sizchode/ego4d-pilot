#!/usr/bin/env bash
#SBATCH --job-name=area3d-runtime
#SBATCH --partition=batch
#SBATCH --account=default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/oscar/scratch/zliu328/3dv-reconstruction/logs/%x-%j.out
set -euo pipefail
R=/oscar/scratch/zliu328/3dv-reconstruction
exec 9>"$R/provenance/area3d-original/runtime.agent-lock"
flock -n 9 || exit 75
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONNOUSERSITE=1 UV_NO_CACHE=1 UV_CONCURRENT_DOWNLOADS=1 UV_CONCURRENT_BUILDS=1
export TMPDIR="$R/tmp/area3d-runtime-$SLURM_JOB_ID"
export MAMBA_ROOT_PREFIX="$R/cache/area3d-mamba"
mkdir -p "$TMPDIR" "$R/provenance/area3d-original/runtime" "$R/code/area3d-runtime-bin"
cleanup() { rm -rf -- "$TMPDIR"; }
trap cleanup EXIT
python3 - <<'PY'
import json,getpass,subprocess
from pathlib import Path
r=Path('/oscar/scratch/zliu328/3dv-reconstruction')
rows=[x.split() for x in subprocess.check_output(['checkquota'],text=True,timeout=60).splitlines() if x.split()[:2]==[getpass.getuser(),'/oscar/scratch']]
assert len(rows)==1
spec=json.loads((r/'provenance/area3d-original/internvl-download.json').read_text())
remaining=sum(x['bytes'] for x in spec['files'] if not (r/'models/InternVL2_5-8B'/x['name']).exists() or (r/'models/InternVL2_5-8B'/x['name']).stat().st_size!=x['bytes'])
assert float(rows[0][2])+remaining/1e9+16+8<=540,'Insufficient budget: runtime peak16GB plus scenes8GB and remaining VLM'
PY
curl -L --fail --retry 2 --max-filesize 20000000 \
  https://micro.mamba.pm/api/micromamba/linux-64/latest -o "$TMPDIR/micromamba.tar.bz2"
tar -xjf "$TMPDIR/micromamba.tar.bz2" -C "$R/code/area3d-runtime-bin" --strip-components=1 bin/micromamba
MAMBA="$R/code/area3d-runtime-bin/micromamba"
"$MAMBA" --version > "$R/provenance/area3d-original/runtime/micromamba-version.txt"
"$MAMBA" create -y --no-rc --prefix "$R/envs/area3d" \
  --override-channels -c conda-forge -c aihabitat --strict-channel-priority \
  python=3.9 habitat-sim=0.3.3 headless numpy=1.26.4 pip
"$MAMBA" clean --all --yes
uv pip install --python "$R/envs/area3d/bin/python" --no-cache \
  --index https://download.pytorch.org/whl/cu128 'torch==2.8.0' 'torchvision==0.23.0'
uv pip install --python "$R/envs/area3d/bin/python" --no-cache \
  'numpy==1.26.4' 'transformers==4.37.2' 'huggingface-hub==0.36.0' \
  'accelerate==1.10.1' 'timm==0.9.12' 'einops==0.8.1' 'sentencepiece==0.2.1' \
  'opencv-python-headless==4.11.0.86' 'omegaconf==2.3.0' 'trimesh==4.7.4' \
  'scipy==1.13.1' 'Pillow==11.3.0' 'joblib==1.5.2' 'matplotlib==3.9.4' \
  'psutil==7.1.0' 'numpy-quaternion==2023.0.4'
"$MAMBA" list -p "$R/envs/area3d" --json > "$R/provenance/area3d-original/runtime/conda-packages.json"
uv pip freeze --python "$R/envs/area3d/bin/python" > "$R/provenance/area3d-original/runtime/python-packages.txt"
cd "$R/code/AREA3D/src"
"$R/envs/area3d/bin/python" - <<'PY'
import json,sys
from pathlib import Path
import torch,transformers,habitat_sim,quaternion,magnum
import active_recon
r=Path('/oscar/scratch/zliu328/3dv-reconstruction')
sys.path.insert(0,str(r/'code/AREA3D/src/vggt'))
from vggt_runner import VGGT_runner
record={'python':sys.version,'torch':torch.__version__,'transformers':transformers.__version__,'habitat_import':True,'area3d_import':True,'vggt_runner_import':True,'rendering_validated':False,'gpu_inference_validated':False}
(r/'provenance/area3d-original/runtime/import-check.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record))
PY
