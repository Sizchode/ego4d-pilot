#!/usr/bin/env bash
#SBATCH --job-name=area3d-replica-assets
#SBATCH --partition=batch
#SBATCH --account=default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=06:00:00
#SBATCH --output=/oscar/scratch/zliu328/3dv-reconstruction/logs/%x-%j.out
set -euo pipefail
R=/oscar/scratch/zliu328/3dv-reconstruction
exec 9>"$R/provenance/area3d-original/replica-assets.agent-lock"
flock -n 9 || exit 75
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONNOUSERSITE=1
python3 - <<'PY'
import json,getpass,subprocess
from pathlib import Path
r=Path('/oscar/scratch/zliu328/3dv-reconstruction')
rows=[x.split() for x in subprocess.check_output(['checkquota'],text=True,timeout=60).splitlines() if x.split()[:2]==[getpass.getuser(),'/oscar/scratch']]
assert len(rows)==1
spec=json.loads((r/'provenance/area3d-original/internvl-download.json').read_text())
remaining=sum(x['bytes'] for x in spec['files'] if not (r/'models/InternVL2_5-8B'/x['name']).exists() or (r/'models/InternVL2_5-8B'/x['name']).stat().st_size!=x['bytes'])
runtime_growth=0 if (r/'provenance/area3d-original/runtime/import-check.json').exists() else 16
assert float(rows[0][2])+remaining/1e9+runtime_growth+8<=540,'Insufficient total stage budget'
PY
python3 "$R/code/runner/scripts/recovery/prepare_replica_habitat.py" \
  --manifest "$R/provenance/area3d-original/replica-stream.json" \
  --destination "$R/datasets/AREA3D/replica_v1" --max-asset-bytes 8000000000 \
  --receipt "$R/provenance/area3d-original/replica-assets.json"
