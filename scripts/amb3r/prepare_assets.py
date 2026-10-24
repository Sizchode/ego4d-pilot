"""Fetch the pinned official AMB3R release without executing model inference."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile


def main():
    import gdown
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    code = args.root / 'code/AMB3R'
    code.parent.mkdir(parents=True, exist_ok=True)
    if not code.exists():
        subprocess.run(['git', 'init', str(code)], check=True)
        subprocess.run(['git', '-C', str(code), 'remote', 'add', 'origin', config['repository']], check=True)
    origin = subprocess.check_output(['git', '-C', str(code), 'remote', 'get-url', 'origin'], text=True).strip()
    if origin != config['repository']:
        raise ValueError('Existing checkout has a different origin')
    if subprocess.check_output(['git', '-C', str(code), 'status', '--porcelain'], text=True):
        raise ValueError('Existing checkout has local changes; refusing to overwrite')
    subprocess.run(['git', '-C', str(code), 'fetch', '--depth', '1', 'origin', config['revision']], check=True)
    subprocess.run(['git', '-C', str(code), 'checkout', '--detach', config['revision']], check=True)
    subprocess.run(['git', '-C', str(code), 'submodule', 'update', '--init', '--recursive', '--jobs', '1'], check=True)
    revision = subprocess.check_output(['git', '-C', str(code), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != config['revision']:
        raise ValueError('Source revision mismatch')
    weights = args.root / 'models/AMB3R' / config['checkpoint_filename']
    weights.parent.mkdir(parents=True, exist_ok=True)
    if not weights.exists():
        partial = weights.with_suffix(weights.suffix + '.download')
        result = gdown.download(config['checkpoint_url'], str(partial), fuzzy=True, resume=True)
        if result is None:
            raise RuntimeError('Official checkpoint download failed')
        # Reject HTML/login pages and interrupted Torch ZIP files before commit.
        with zipfile.ZipFile(partial) as archive:
            damaged = archive.testzip()
            if damaged is not None:
                raise ValueError(f'Checkpoint member failed CRC: {damaged}')
        partial.replace(weights)
    checksum = hashlib.sha256()
    with weights.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            checksum.update(block)
    publisher_hash = config['publisher_checkpoint_sha256']
    if publisher_hash is not None and checksum.hexdigest() != publisher_hash:
        raise ValueError('Checkpoint differs from publisher digest')
    provenance = args.root / 'provenance/amb3r'
    provenance.mkdir(parents=True, exist_ok=True)
    (provenance / 'assets.json').write_text(json.dumps({
        'completed': True, 'inference_started': False, 'config': config,
        'checkpoint': {'bytes': weights.stat().st_size, 'sha256': checksum.hexdigest(),
                       'publisher_digest_available': publisher_hash is not None},
        'source_revision': revision}, indent=2)+'\n')
    print('Official AMB3R code and checkpoint prepared; model loading not yet verified.', flush=True)


if __name__ == '__main__':
    main()
