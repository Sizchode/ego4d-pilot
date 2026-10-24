"""Stage only RGB from an archived baseline's exact input manifest.

Selection is fixed by input order, without reconstruction results or GT access.
Provenance stays outside the inference mount; filenames inside it are anonymous.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
from PIL import Image


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def stage(manifest_path, destination, provenance_path, count):
    rows = json.loads(Path(manifest_path).read_text())
    if count < 2 or count > len(rows):
        raise ValueError('Requested view count must be between 2 and manifest length')
    if [row['index'] for row in rows] != list(range(len(rows))):
        raise ValueError('Archived input order must be contiguous and explicit')
    indices = np.rint(np.linspace(0, len(rows) - 1, count)).astype(int).tolist()
    destination = Path(destination)
    provenance_path = Path(provenance_path)
    if destination.exists() or provenance_path.exists():
        raise FileExistsError('Input and provenance destinations must be new')
    destination.parent.mkdir(parents=True, exist_ok=True)
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent, prefix='.rgb-stage-') as temporary:
        temporary = Path(temporary)
        (temporary / 'images').mkdir()
        public = []
        for output_index, source_index in enumerate(indices):
            row = rows[source_index]
            source = Path(row['path'])
            if source.suffix.lower() not in {'.jpg', '.jpeg', '.png'}:
                raise ValueError('Manifest contains a non-image entry')
            target = temporary / 'images' / f'{output_index:06d}{source.suffix.lower()}'
            shutil.copyfile(source, target)
            if digest(target) != row['sha256']:
                raise ValueError(f'Archived input changed at index {source_index}')
            with Image.open(target) as image:
                image.load()
                hw = [image.height, image.width]
            if hw != row['original_hw']:
                raise ValueError('Archived dimensions do not match image')
            public.append({'index': output_index, 'path': str(target.relative_to(temporary)),
                           'sha256': row['sha256'], 'original_hw': hw})
        (temporary / 'input_manifest.json').write_text(json.dumps(public, indent=2) + '\n')
        provenance = {'source_manifest': str(Path(manifest_path).resolve()),
                      'source_manifest_sha256': digest(manifest_path),
                      'selection': 'rint(linspace(0, N-1, count)); ties-to-even',
                      'original_indices': indices, 'source_entries': [rows[i] for i in indices]}
        temporary.rename(destination)
        provenance_path.write_text(json.dumps(provenance, indent=2) + '\n')
    return public


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--provenance', type=Path, required=True)
    parser.add_argument('--count', type=int, required=True)
    args = parser.parse_args()
    stage(args.manifest, args.destination, args.provenance, args.count)
