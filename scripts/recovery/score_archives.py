"""Isolated offline scoring of matched public prediction/GT archives."""
import argparse
import json
from pathlib import Path

import numpy as np

try:
    from .pointmap_score import score_pointmaps
    from .public_canvas import public_shape
except ImportError:
    from pointmap_score import score_pointmaps
    from public_canvas import public_shape


def rotation_error_degrees(rotation):
    cosine = (np.trace(rotation, axis1=-2, axis2=-1)-1)/2
    return np.degrees(np.arccos(np.clip(cosine, -1, 1)))


def camera_metrics(predicted, reference, alignment):
    predicted, reference = np.asarray(predicted), np.asarray(reference)
    if predicted.shape[0]!=reference.shape[0] or predicted.shape[-2:] not in {(3,4),(4,4)} \
            or reference.shape[-2:] not in {(3,4),(4,4)}:
        raise ValueError('Camera arrays must contain the same ordered views')
    if not np.isfinite(predicted).all() or not np.isfinite(reference).all():
        raise ValueError('Camera archive has nonfinite poses')
    pr, gr = predicted[:,:3,:3], reference[:,:3,:3]
    pc = -np.einsum('nji,nj->ni',pr,predicted[:,:3,3])
    gc = -np.einsum('nji,nj->ni',gr,reference[:,:3,3])
    pairs = np.array([(i,j) for i in range(len(pr)) for j in range(i+1,len(pr))],dtype=int)
    if len(pairs)==0:
        raise ValueError('Relative camera evaluation requires at least two views')
    i,j=pairs.T
    prot=pr[i]@pr[j].transpose(0,2,1)
    grot=gr[i]@gr[j].transpose(0,2,1)
    rotation=rotation_error_degrees(prot@grot.transpose(0,2,1))
    pt=np.einsum('nij,nj->ni',pr[i],pc[j]-pc[i])
    gt=np.einsum('nij,nj->ni',gr[i],gc[j]-gc[i])
    pn,gn=np.linalg.norm(pt,axis=-1),np.linalg.norm(gt,axis=-1)
    gt_domain=gn>0
    pred_defined=pn>0
    translation=np.full(len(pairs),180.)
    usable=gt_domain&pred_defined
    translation[usable]=np.degrees(np.arccos(np.clip(
        np.einsum('ni,ni->n',pt[usable],gt[usable])/(pn[usable]*gn[usable]),-1,1)))
    report={'camera_pairs':len(pairs),'translation_gt_defined_pairs':int(gt_domain.sum()),
            'translation_prediction_missing_pairs':int((gt_domain&~pred_defined).sum()),
            'relative_rotation_mean_deg':float(rotation.mean()),
            'relative_translation_direction_mean_deg':float(translation[gt_domain].mean()) if gt_domain.any() else None,
            'camera_ate_rmse_m_using_point_alignment':None}
    if alignment is not None:
        aligned=alignment['scale']*(pc@alignment['rotation'].T)+alignment['translation']
        report['camera_ate_rmse_m_using_point_alignment']=float(np.sqrt(np.square(aligned-gc).sum(-1).mean()))
    return report, {'pairs':pairs,'relative_rotation_deg':rotation,
                    'relative_translation_direction_deg':translation,'translation_gt_domain':gt_domain}


def read_views(directory, schema):
    directory=Path(directory)
    metadata=json.loads((directory/'canvas.json').read_text())
    if metadata['schema']!=schema or metadata['long_edge']!=504 or metadata['patch']!=14:
        raise ValueError('Unknown or incompatible evaluation canvas')
    rows=metadata['views']
    manifest=json.loads((directory/'input_manifest.json').read_text())
    if len(manifest)!=len(rows) or [row['index'] for row in manifest]!=list(range(len(rows))):
        raise ValueError('Canvas and input manifest populations differ')
    if [row['index'] for row in rows]!=list(range(len(rows))):
        raise ValueError('View ordering is not explicit and contiguous')
    views=[]
    for row in rows:
        if tuple(row['shape'])!=public_shape(manifest[row['index']]['original_hw']):
            raise ValueError('Stored canvas is not the frozen full-image public grid')
        path=Path(row['file'])
        if path.name!=str(path) or path.suffix!='.npz':
            raise ValueError('View archive must be a local NPZ basename')
        with np.load(directory/path,allow_pickle=False) as data:
            view={key:data[key] for key in ['points','valid','extrinsics']}
        if view['points'].shape!=tuple(row['shape'])+(3,) or view['valid'].shape!=tuple(row['shape']):
            raise ValueError('Canvas inventory disagrees with stored view')
        views.append(view)
    return views,manifest


def score_archives(prediction, ground_truth, output):
    predicted,pm=read_views(prediction,'public-world-point-map-v1')
    truth,gm=read_views(ground_truth,'public-ground-truth-v1')
    if pm!=gm:
        raise ValueError('Prediction and evaluator do not use identical ordered RGB inputs')
    for p,g in zip(predicted,truth):
        if p['points'].shape!=g['points'].shape:
            raise ValueError('Per-view pixel correspondence is inconsistent')
    flatten=lambda views,key:np.concatenate([v[key].reshape(-1,3) if key=='points' else v[key].reshape(-1) for v in views])
    report,errors,alignment=score_pointmaps(flatten(predicted,'points'),flatten(predicted,'valid'),
                                           flatten(truth,'points'),flatten(truth,'valid'))
    poses,pose_arrays=camera_metrics(np.stack([v['extrinsics'] for v in predicted]),
                                     np.stack([v['extrinsics'] for v in truth]),alignment)
    output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    cursor=0
    for index,(p,g) in enumerate(zip(predicted,truth)):
        size=g['valid'].size
        np.savez_compressed(output/f'{index:06d}.npz',error_m=errors[cursor:cursor+size].reshape(g['valid'].shape),
                            gt_valid=g['valid'],prediction_valid=p['valid'])
        cursor+=size
    if alignment is not None:
        np.savez(output/'similarity.npz',**alignment)
    np.savez_compressed(output/'camera_pairs.npz',**pose_arrays)
    report.update(cameras=poses,prediction_archive=str(prediction),gt_archive=str(ground_truth),views=len(truth))
    (output/'result.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--prediction',type=Path,required=True)
    parser.add_argument('--ground-truth',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(score_archives(args.prediction,args.ground_truth,args.output)))
