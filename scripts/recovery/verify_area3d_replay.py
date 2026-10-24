"""Verify real AREA3D replay outputs against requested poses and sensor metadata."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
from PIL import Image


def verify(source, poses, export, expected_views):
    sys.path.insert(0,str(Path(source).resolve()))
    from envs.transformation import Twc_to_Thc
    export=Path(export)
    meta=json.loads((export/'transforms.json').read_text())
    requested=np.loadtxt(poses).reshape(-1,4,4)
    if len(meta['frames']) != expected_views or len(requested) != expected_views:
        raise ValueError('Rendered/requested view counts differ from protocol')
    rows=[]
    shape=(int(meta['h']),int(meta['w']))
    for pose,frame in zip(requested,meta['frames']):
        with Image.open(export/frame['file_path']) as im:
            rgb=np.asarray(im.convert('RGB'))
        depth=np.load(export/frame['depth_file_path'])
        actual=np.asarray(frame['transform_matrix'])
        expected=Twc_to_Thc(pose)
        if rgb.shape != (*shape,3) or np.ptp(rgb)==0:
            raise ValueError('Invalid or constant rendered RGB')
        if depth.shape != shape or not np.isfinite(depth).all() or not np.any(depth>0):
            raise ValueError('Invalid rendered depth')
        if not np.allclose(actual,expected,rtol=0,atol=1e-5):
            raise ValueError('Actual sensor pose differs from requested pose')
        rows.append({'image':frame['file_path'],'depth_valid_fraction':float((depth>0).mean()),
                     'pose_max_abs_error':float(np.max(np.abs(actual-expected)))})
    return {'views':rows,'count':len(rows),'validation':'Actual RGB/depth/sensor pose structure; visual review separate'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['source','poses','export']:
        p.add_argument('--'+name,required=True,type=Path)
    p.add_argument('--expected-views',required=True,type=int)
    args=p.parse_args();report=verify(**vars(args))
    (args.export/'replay-validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'count':report['count'],'export':str(args.export)}))


if __name__=='__main__':
    main()
