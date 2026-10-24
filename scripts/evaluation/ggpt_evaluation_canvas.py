"""Offline adapter of unchanged native predictions to GGPT's GT pixel canvas.

Uses the author's actual crop/resize and depth-unprojection functions. GT is
accessible here only for evaluation; no model or reconstruction code is called.
"""
import ast
import math
from pathlib import Path
import json

import cv2
import numpy as np
import torch


def load_definitions(path, namespace, selected=None):
    tree = ast.parse(Path(path).read_text(), filename=str(path))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and (selected is None or n.name in selected)]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)


def official_preprocessors(repository):
    repository = Path(repository)
    ns = {'np':np, 'torch':torch, 'cv2':cv2, 'math':math}
    load_definitions(repository/'sfm/dataloader/transform2D_utils.py', ns)
    path = repository/'sfm/dataloader/base_dataset.py'
    cls = next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='BaseDataset')
    names = {'preprocess_2D_simple','preprocess_2D_vggt'}
    nodes = [n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in names]
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),ns)
    load_definitions(repository/'utils/geometry.py',ns)
    return ns


def native_directory(public):
    public = Path(public)
    parent = Path(json.loads((public/'canvas.json').read_text())['parent_state'])
    # Resolve the recorded container /output mount, not a guessed model layout.
    if parent.is_relative_to('/output'):
        if public.parent.name != 'output':
            raise ValueError('Cannot resolve recorded /output mount from public archive')
        parent = public.parent / parent.relative_to('/output')
    if not (parent/'state.npz').is_file():
        raise FileNotFoundError(parent/'state.npz')
    return parent


def iter_official_views(row, root, repository, view_indices=None):
    from native_state import load_state
    from public_canvas import pixel_grid, transform_pixels
    from geometry_state import sample_pointmap
    native, manifest, metadata = load_state(native_directory(row['result']['prediction_archive']))
    public_manifest = json.loads((Path(row['result']['prediction_archive'])/'input_manifest.json').read_text())
    if manifest != public_manifest or metadata.get('gt_conditioning') is not False:
        raise ValueError('Native provenance/order mismatch or GT-conditioned prediction')
    selected = set(range(len(manifest))) if view_indices is None else set(view_indices)
    if not selected <= set(range(len(manifest))):
        raise ValueError('Requested view lies outside the fixed input manifest')
    source = Path(root)/'datasets/GGPT_eval_16/evaluator_gt'/row['sequence_key']
    ks, es = np.load(source/'intrinsics.npy'), np.load(source/'extrinsics.npy')
    official = official_preprocessors(repository)
    for i, view in enumerate(manifest):
        if i not in selected:
            continue
        h,w = view['original_hw']
        depth = np.load(source/'depths'/f'{i:06d}.npy')
        if depth.shape != (h,w):
            raise ValueError('Original GT and RGB dimensions differ')
        target = np.array([round(h*518/w/14)*14,518])
        # The identical branch selection used by extracted BaseDataset.__getitem__.
        mode = 'preprocess_2D_simple' if (h,w)==tuple(target) else 'preprocess_2D_vggt'
        _, d, k = official[mode](None,target_shape=target,image=np.zeros((h,w,3),np.uint8),depth=depth,K=ks[i],intr_convention='opencv')
        original_to_target = k.astype(np.float64) @ np.linalg.inv(ks[i].astype(np.float64))
        uv = transform_pixels(native['original_to_native'][i] @ np.linalg.inv(original_to_target),pixel_grid(d.shape))
        p,pv = sample_pointmap(native['points'][i],native['valid'][i],uv)
        refinement_regions = {}
        if 'refined_mask' in native:
            if native['refined_mask'].shape != native['valid'].shape or native['refined_mask'].dtype != bool:
                raise ValueError('Recorded refinement support must match the native point map')
            _, updated = sample_pointmap(native['points'][i],native['valid'][i]&native['refined_mask'][i],uv)
            _, retained = sample_pointmap(native['points'][i],native['valid'][i]&~native['refined_mask'][i],uv)
            refinement_regions = {'all_neighbors_refined':updated,'all_neighbors_retained':retained,
                                  'mixed_refinement_boundary':pv&~updated&~retained}
        nh,nw = native['valid'][i].shape
        covered = (np.isfinite(uv).all(-1) & (uv[...,0]>=0) & (uv[...,0]<=nw-1)
                   & (uv[...,1]>=0) & (uv[...,1]<=nh-1))
        g = official['unproject_depth_map_to_point_map_torch'](torch.from_numpy(d[None]),torch.from_numpy(es[i:i+1].astype(np.float32)),torch.from_numpy(k[None].astype(np.float32)))[0].numpy()
        gv = d > 1e-5
        audit = {'view':i,'shape':list(d.shape),'gt_pixels':int(gv.sum()),'gt_depth_0_to_1e5':int(((d>0)&(d<=1e-5)).sum()),
                      'gt_mask_but_unprojection_zero':int(((d>1e-5)&(d<=1e-4)).sum()),
                      'missing_prediction_gt_pixels':int((gv&~pv).sum()),'original_to_target':original_to_target.tolist()}
        yield {'prediction':p.astype(np.float32),'prediction_valid':pv,'truth':g,'gt_valid':gv,
               'covered':covered,'predicted_extrinsics':native['extrinsics'][i],
               'predicted_intrinsics':native['intrinsics'][i],
               'gt_extrinsics':es[i],'gt_intrinsics':k,'native_rgb':native['rgb'][i],
               'native_uv':uv,'refinement_regions':refinement_regions,'audit':audit}


def read_official_canvas(row, root, repository):
    predictions, masks, truths, gt_masks, audit = [],[],[],[],[]
    for view in iter_official_views(row,root,repository):
        predictions.append(view['prediction']); masks.append(view['prediction_valid'])
        truths.append(view['truth']); gt_masks.append(view['gt_valid']); audit.append(view['audit'])
    flatten = lambda xs,xyz:np.concatenate([x.reshape(-1,3) if xyz else x.reshape(-1) for x in xs])
    return flatten(predictions,True),flatten(masks,False),flatten(truths,True),flatten(gt_masks,False),audit
