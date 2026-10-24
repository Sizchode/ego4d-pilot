"""Download pinned official AREA3D VLM assets under an explicit quota budget."""
import argparse
from datetime import datetime, timezone
import getpass
import json
from pathlib import Path
import subprocess

from huggingface_hub import hf_hub_download


def quota():
    report = subprocess.check_output(['checkquota'], text=True, timeout=60)
    rows = [line.split() for line in report.splitlines()
            if line.split()[:2] == [getpass.getuser(), '/oscar/scratch']]
    if len(rows) != 1:
        raise ValueError('Cannot identify scratch quota')
    return float(rows[0][2])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--destination', required=True, type=Path)
    parser.add_argument('--maximum-usage-gb', required=True, type=float)
    parser.add_argument('--reserved-other-growth-gb', required=True, type=float)
    parser.add_argument('--receipt', required=True, type=Path)
    args = parser.parse_args()
    spec = json.loads(args.manifest.read_text())
    files = spec['files']
    assert sum(row['bytes'] for row in files) == spec['total_bytes']
    args.destination.mkdir(parents=True, exist_ok=True)
    checked = []
    for item in files:
        name = item['name']
        if Path(name).name != name:
            raise ValueError('Expected flat official model assets')
        missing = sum(row['bytes'] for row in files
                      if not (args.destination/row['name']).is_file()
                      or (args.destination/row['name']).stat().st_size != row['bytes'])
        # Decimal-byte conversion is conservative relative to binary quota G.
        used = quota()
        if used + missing/1e9 + args.reserved_other_growth_gb > args.maximum_usage_gb:
            raise RuntimeError('Insufficient quota budget for remaining files and reserved stages')
        path = Path(hf_hub_download(repo_id=spec['repo_id'], revision=spec['revision'],
                                   filename=name, local_dir=str(args.destination)))
        if path.stat().st_size != item['bytes']:
            raise ValueError(f'Unexpected downloaded size: {name}')
        checked.append({'name': name, 'bytes': path.stat().st_size})
        print(json.dumps({'verified_file': name, 'bytes': path.stat().st_size}), flush=True)
    with (args.destination/'model.safetensors.index.json').open() as f:
        index = json.load(f)
    assert set(index['weight_map'].values()).issubset({row['name'] for row in checked})
    receipt = {'checked_at': datetime.now(timezone.utc).isoformat(), 'model': spec['repo_id'],
               'files': checked, 'total_bytes': sum(row['bytes'] for row in checked),
               'scratch_used_GB': quota(), 'model_inference_executed': False}
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps({'assets_complete': True, 'total_bytes': receipt['total_bytes']}))


if __name__ == '__main__':
    main()
