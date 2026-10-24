"""RGB-only VGGT with separate authoritative point and depth-camera states."""
import argparse
import json
from pathlib import Path
import random
import sys
import time

import numpy as np

from geometry_state import pixel_resize, unproject_depth
from native_state import save_state
from run_da3_rgb import verify_input, write_json


def image_transforms(records):
    transforms = []
    for row in records:
        width, height = row['original_wh']
        rw, rh = row['resized_wh']
        dx = row['prepad_left']+row['batch_pad_left']-row['crop_left']
        dy = row['prepad_top']+row['batch_pad_top']-row['crop_top']
        translation = np.array([[1.,0,dx],[0,1.,dy],[0,0,1.]])
        transforms.append(translation@pixel_resize((height,width),(rh,rw)))
    return np.stack(transforms)


def main():
    parser = argparse.ArgumentParser()
    for name in ['input','model','code','config','output']:
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--forbidden-path',action='append',required=True)
    args = parser.parse_args()
    if any(Path(path).exists() for path in args.forbidden_path):
        raise RuntimeError('A forbidden GT/source path is visible inside the sandbox')
    manifest, paths = verify_input(args.input)
    config = json.loads(args.config.read_text())
    assert config['model_repository']=='facebook/VGGT-1B'
    assert config['preprocess_mode']=='crop' and config['inference_dtype']=='bfloat16'
    sys.path.insert(0,str(args.code))
    import torch
    from vggt.models.vggt import VGGT
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    from vggt_preprocess import load_and_preprocess_with_metadata
    torch.set_num_threads(1)
    random.seed(config['sampling_seed']);np.random.seed(config['sampling_seed'])
    torch.manual_seed(config['sampling_seed']);torch.cuda.manual_seed_all(config['sampling_seed'])
    args.output.mkdir(parents=True,exist_ok=False)
    raw=args.output/'raw';raw.mkdir()
    write_json(args.output/'input_manifest.json',manifest)
    write_json(args.output/'configuration.json',{'pinned_baseline_config':config,
               'this_execution':'native point and depth-camera states; no old square-canvas evaluator',
               'branch_selection':'Both retained; point branch is the declared GGPT parent'})
    cpu_images, _, records = load_and_preprocess_with_metadata(paths,config['preprocess_mode'])
    transforms=image_transforms(records)
    write_json(args.output/'preprocessing.json',records)
    model=VGGT.from_pretrained(str(args.model)).to('cuda').eval()
    torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();started=time.monotonic()
    with torch.inference_mode(),torch.autocast(device_type='cuda',dtype=torch.bfloat16):
        output=model(cpu_images.to('cuda'))
    torch.cuda.synchronize();seconds=time.monotonic()-started
    extrinsics,intrinsics=pose_encoding_to_extri_intri(output['pose_enc'],cpu_images.shape[-2:])
    numpy=lambda tensor:tensor.detach().float().cpu().numpy()
    for key in ['pose_enc','depth','depth_conf','world_points','world_points_conf']:
        np.save(raw/(key+'.npy'),numpy(output[key]))
    k,t=numpy(intrinsics[0]),numpy(extrinsics[0])
    np.save(raw/'intrinsics.npy',k);np.save(raw/'extrinsics.npy',t)
    rgb=np.rint(cpu_images.permute(0,2,3,1).numpy()*255).astype(np.uint8)
    shared={'intrinsics':k,'extrinsics':t,'rgb':rgb,'original_to_native':transforms}
    points=numpy(output['world_points'][0]);confidence=numpy(output['world_points_conf'][0])
    save_state(args.output/'point-state',dict(shared,points=points,
               valid=np.isfinite(points).all(-1),confidence=confidence),manifest,
               {'method':'VGGT-1B-point','gt_conditioning':False,'point_branch':'world_points',
                'preprocessing':'official crop; half-pixel resize plus recorded crop/pad',
                'parent':None})
    depth=numpy(output['depth'][0,...,0]);conf=numpy(output['depth_conf'][0])
    points=unproject_depth(depth,k,t).astype(np.float32)
    save_state(args.output/'depth-camera-state',dict(shared,points=points,
               valid=np.isfinite(points).all(-1)&np.isfinite(depth)&(depth>0),confidence=conf),manifest,
               {'method':'VGGT-1B-depth+camera','gt_conditioning':False,
                'point_branch':'depth unprojection with predicted cameras','parent':None})
    write_json(args.output/'result.json',{'state':'native_prediction_complete','views':len(manifest),
               'gt_access':False,'forbidden_paths_absent':args.forbidden_path,'inference_seconds':seconds,
               'native_hw':list(cpu_images.shape[-2:]),'torch_version':torch.__version__,
               'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),'gpu':torch.cuda.get_device_name(),
               'scores_computed':False,'point_branch_selected_before_scoring':True})
    print((args.output/'result.json').read_text(),flush=True)


if __name__=='__main__':
    main()
