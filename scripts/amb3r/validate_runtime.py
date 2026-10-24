"""Strictly load the official full AMB3R checkpoint on CPU."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--code', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.code.resolve()))
    import torch
    import flash_attn
    import pytorch3d._C
    import spconv.pytorch
    from amb3r.model import AMB3R
    torch.set_num_threads(1)
    model = AMB3R(device='cpu').eval()
    # Official loading handles the upstream interp_v2 checkpoint version field.
    model.load_weights(str(args.checkpoint), data_type='bf16', strict=True)
    report = {'checkpoint_strict_load': True, 'cuda_tested': False,
              'parameter_count': sum(p.numel() for p in model.parameters()),
              'backend_interp_v2': bool(model.backend.interp_v2.item()),
              'packages': {name: importlib.metadata.version(name) for name in
                           ['torch', 'torchvision', 'pytorch3d', 'spconv-cu118',
                            'flash-attn', 'torch-scatter', 'xformers']}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
