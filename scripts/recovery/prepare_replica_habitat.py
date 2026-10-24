"""Stream official Replica archive parts; retain only selected Habitat assets."""
import argparse
from datetime import datetime, timezone
import gzip
import io
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import urllib.request
import zipfile


class ArchiveParts(io.RawIOBase):
    def __init__(self, parts):
        self.parts = iter(parts)
        self.response = None
        self.part = None
        self.part_bytes = 0
        self.transferred = 0

    def readable(self):
        return True

    def read(self, size=-1):
        if size < 0:
            raise ValueError('Archive reads must be bounded')
        result = bytearray()
        while len(result) < size:
            if self.response is None:
                self.part = next(self.parts, None)
                if self.part is None:
                    break
                self.response = urllib.request.urlopen(self.part['url'], timeout=120)
                self.part_bytes = 0
                print(json.dumps({'reading_part': self.part['name']}), flush=True)
            block = self.response.read(size-len(result))
            if not block:
                self.response.close()
                self.response = None
                if self.part_bytes != self.part['bytes']:
                    raise ValueError('Archive part was truncated')
                continue
            self.part_bytes += len(block)
            self.transferred += len(block)
            if self.part_bytes > self.part['bytes']:
                raise ValueError('Archive part exceeds declared length')
            result.extend(block)
        return bytes(result)

    def close(self):
        if self.response is not None:
            self.response.close()
        super().close()


def selected_path(name, scenes):
    p = PurePosixPath(name)
    if p.is_absolute() or '..' in p.parts:
        raise ValueError('Unsafe archive path')
    parts = p.parts
    if len(parts) >= 3 and parts[0] in scenes and parts[1] == 'habitat':
        return Path(*parts)
    return None


def save_member(stream, destination, size, remaining):
    if size > remaining:
        raise ValueError('Selected asset bytes exceed stage budget')
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name+'.partial')
    try:
        with temporary.open('wb') as output:
            copied = 0
            while copied < size:
                block = stream.read(min(1024*1024, size-copied))
                if not block:
                    raise ValueError('Truncated member')
                output.write(block)
                copied += len(block)
        if destination.exists():
            # Never replace a different pre-existing asset silently.
            with destination.open('rb') as existing, temporary.open('rb') as new:
                while True:
                    a, b = existing.read(1024*1024), new.read(1024*1024)
                    if a != b:
                        raise ValueError('Existing asset differs from official archive')
                    if not a:
                        break
            temporary.unlink()
        else:
            temporary.rename(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def extract_tar(archive, root, scenes, max_bytes):
    records = []
    used = 0
    for member in archive:
        rel = selected_path(member.name, scenes)
        if rel is None or member.isdir():
            continue
        if not member.isfile():
            raise ValueError('Selected asset is not a regular file')
        with archive.extractfile(member) as source:
            save_member(source, root/rel, member.size, max_bytes-used)
        used += member.size
        records.append({'path': str(rel), 'bytes': member.size})
        print(json.dumps({'extracted': str(rel), 'bytes': member.size}), flush=True)
    return records


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--destination', type=Path, required=True)
    p.add_argument('--max-asset-bytes', type=int, required=True)
    p.add_argument('--receipt', type=Path, required=True)
    args = p.parse_args()
    spec = json.loads(args.manifest.read_text())
    records = []
    with ArchiveParts(spec['parts']) as source:
        with gzip.GzipFile(fileobj=source, mode='rb') as decoded:
            with tarfile.open(fileobj=decoded, mode='r|') as archive:
                records = extract_tar(archive, args.destination, set(spec['scenes']), args.max_asset_bytes)
            while decoded.read(1024*1024):
                pass  # Read footer to verify gzip CRC and complete source lengths.
        transferred = source.transferred
    if transferred != sum(part['bytes'] for part in spec['parts']):
        raise ValueError('Source byte count does not match release metadata')
    with urllib.request.urlopen(spec['additional_configs_url'], timeout=120) as response:
        data = response.read(spec['additional_configs_max_bytes']+1)
    if len(data) > spec['additional_configs_max_bytes']:
        raise ValueError('Additional-config archive exceeds budget')
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        for entry in archive.infolist():
            rel = selected_path(entry.filename, set(spec['scenes']))
            if rel is None or entry.is_dir():
                continue
            with archive.open(entry) as source:
                save_member(source, args.destination/rel, entry.file_size,
                            args.max_asset_bytes-sum(row['bytes'] for row in records))
            records.append({'path': str(rel), 'bytes': entry.file_size})
    for scene in spec['scenes']:
        for name in ('mesh_semantic.ply','mesh_semantic.navmesh','info_semantic.json'):
            path = args.destination/scene/'habitat'/name
            if not path.is_file() or path.stat().st_size == 0:
                raise ValueError(f'Missing required Habitat asset: {path}')
    report = {'checked_at': datetime.now(timezone.utc).isoformat(), 'scenes': spec['scenes'],
              'files': records, 'stored_bytes': sum(row['bytes'] for row in records),
              'transferred_bytes': transferred+len(data), 'gzip_crc_verified': True,
              'rendering_validated': False}
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'assets_complete': True,'stored_bytes':report['stored_bytes']}))


if __name__ == '__main__':
    main()
