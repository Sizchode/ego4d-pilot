"""Load the actual GGPT checkpoint and optionally exercise CUDA kernels.

No benchmark inputs or GT are loaded. This is dependency verification, not a
reconstruction result and not a replacement for the development-scene run.
"""
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
    parser.add_argument('--cuda', action='store_true')
    args = parser.parse_args()
    sys.path.insert(0, str(args.code.resolve()))
    import pycolmap  # Upstream documents that this must precede spconv.
    import torch
    import torch_scatter
    import spconv.pytorch as spconv
    import flash_attn
    import romav2
    from omegaconf import OmegaConf
    from hydra.utils import instantiate
    config = OmegaConf.load(args.code / 'configs/demo.yaml')
    model = instantiate(config.ggptmodel_config).eval()
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    model.load_state_dict({k.removeprefix('module.'): v for k, v in state.items()}, strict=True)
    report = {'checkpoint_strict_load': True, 'cuda_tested': False,
              'parameter_count': sum(p.numel() for p in model.parameters()),
              'packages': {name: importlib.metadata.version(name) for name in
                           ['torch', 'torchvision', 'spconv-cu126', 'flash-attn',
                            'torch-scatter', 'pycolmap', 'romav2', 'pointcept']}}
    if args.cuda:
        device = torch.device('cuda')
        # Analytic CPU result checks the actual scatter extension used by PTv3.
        values = torch.arange(8, dtype=torch.float32, device=device)
        groups = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3], device=device)
        actual = torch_scatter.scatter_add(values, groups)
        torch.testing.assert_close(actual.cpu(), torch.tensor([1., 5., 9., 13.]))
        # Verify FlashAttention numerically against PyTorch's dense definition.
        torch.manual_seed(47)
        qkv = torch.randn(1, 32, 3, 2, 32, device=device, dtype=torch.float16)
        result = flash_attn.flash_attn_qkvpacked_func(qkv)
        q, k, v = qkv.float().unbind(2)
        reference = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2)
        torch.testing.assert_close(result.float(), reference, atol=2e-3, rtol=2e-3)
        # Exercise the actual sparse CUDA kernel. spconv special-cases 1x1
        # convolutions as torch.mm, so that path cannot validate its extension.
        # MaskImplicitGemm stores weights as [out, z, y, x, in] (KRSC).
        from spconv.core import ConvAlgo
        features = torch.randn(4, 16, device=device)
        indices = torch.tensor([[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]],
                               device=device, dtype=torch.int32)
        sparse = spconv.SparseConvTensor(features, indices, [2, 2, 2], 1)
        layer = spconv.SubMConv3d(16, 24, kernel_size=3, padding=1, bias=False,
                                  algo=ConvAlgo.MaskImplicitGemm, fp32_accum=True).to(device)
        output = layer(sparse)
        dense = torch.zeros(1, 16, 2, 2, 2)
        b, z, y, x = indices.cpu().long().unbind(1)
        dense[b, :, z, y, x] = features.cpu()
        dense_result = torch.nn.functional.conv3d(
            dense, layer.weight.detach().cpu().permute(0, 4, 1, 2, 3).contiguous(), padding=1)
        ob, oz, oy, ox = output.indices.cpu().long().unbind(1)
        reference = dense_result[ob, :, oz, oy, ox]
        torch.testing.assert_close(output.features.cpu(), reference)
        report.update(cuda_tested=True, gpu=torch.cuda.get_device_name(), cuda=torch.version.cuda,
                      sparse_kernel='SubMConv3d 3x3x3 MaskImplicitGemm',
                      sparse_reference='CPU dense conv3d sampled at active output voxels',
                      sparse_max_absolute_error=float((output.features.cpu()-reference).abs().max()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
