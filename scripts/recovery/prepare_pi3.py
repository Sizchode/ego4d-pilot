"""Pin and verify the original published Pi3 checkpoint, separately from Pi3X."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    for name in ['root', 'config', 'output']:
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    assert config['method'] == 'Pi3' and config['model_repository'] == 'yyfz233/Pi3'
    code = args.root/'code/Pi3'
    assert subprocess.check_output(['git', '-C', str(code), 'rev-parse', 'HEAD'], text=True).strip() == config['code_revision']
    subprocess.run(['git', '-C', str(code), 'diff', '--exit-code', 'HEAD'], check=True)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'configuration.json').write_text(json.dumps(config, indent=2)+'\n')
    from huggingface_hub import snapshot_download
    model_dir = args.root/'models/Pi3'
    snapshot_download(config['model_repository'], revision=config['model_revision'],
                      local_dir=model_dir, allow_patterns=['config.json', config['model_filename']],
                      max_workers=1)
    path = model_dir/config['model_filename']
    assert path.stat().st_size == config['model_bytes']
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(chunk)
    assert digest.hexdigest() == config['model_sha256'], 'Checkpoint differs from publisher digest'
    model_config = json.loads((model_dir/'config.json').read_text())
    assert model_config == config['model_config']
    sys.path.insert(0, str(code))
    import torch
    from safetensors.torch import load_file
    from pi3.models.pi3 import Pi3
    torch.set_num_threads(1)
    model = Pi3(**model_config).eval()
    model.load_state_dict(load_file(str(path), device='cpu'), strict=True)
    packages = {distribution.metadata['Name']: distribution.version
                for distribution in importlib.metadata.distributions()}
    (args.output/'packages.json').write_text(json.dumps(packages, indent=2, sort_keys=True)+'\n')
    report = {'method': 'Pi3', 'model_class': type(model).__name__, 'checkpoint_strict_load': True,
              'publisher_digest_verified': True, 'checkpoint_bytes': path.stat().st_size,
              'parameter_count': sum(parameter.numel() for parameter in model.parameters()),
              'torch_version': torch.__version__, 'cuda_tested': False,
              'inference_executed': False, 'runtime_environment': str(args.root/'envs/pi3x'),
              'model_directory': str(model_dir)}
    (args.output/'result.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
