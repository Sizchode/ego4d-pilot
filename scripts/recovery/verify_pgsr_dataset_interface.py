"""CPU integration check of the actual PGSR camera reader and assembled data.

Only the pure camera declarations/functions are loaded from upstream source so
this check does not need the CUDA rasterizer. It is not a GPU/backend acceptance.
"""
import argparse
import ast
import json
import math
import os
import re
import subprocess
from pathlib import Path
import tempfile
from typing import NamedTuple

import numpy as np
from PIL import Image

from assemble_area3d_dataset import assemble


def definitions(path, names, namespace):
    tree = ast.parse(path.read_text(), filename=str(path))
    selected = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    if {n.name for n in selected} != set(names):
        raise ValueError(f'Upstream definitions changed: {names}')
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), namespace)


def verify(pgsr):
    namespace = dict(np=np, os=os, json=json, math=math, Path=Path, Image=Image, NamedTuple=NamedTuple)
    definitions(pgsr/'utils/graphics_utils.py', ['fov2focal', 'focal2fov'], namespace)
    definitions(pgsr/'scene/dataset_readers.py', ['CameraInfo', 'readCamerasFromTransforms'], namespace)
    reader = namespace['readCamerasFromTransforms']
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        sources = {}
        expected = {}
        counts = {'initial': 2, 'acquired': 1, 'test': 2}
        ordinal = 0
        for role, count in counts.items():
            source = root/role
            source.mkdir()
            frames = []
            for i in range(count):
                ordinal += 1
                angle = ordinal*0.17
                rotation = np.array([[math.cos(angle),0,math.sin(angle)], [0,1,0],[-math.sin(angle),0,math.cos(angle)]])
                pose = np.eye(4)
                pose[:3,:3] = rotation
                pose[:3,3] = [ordinal,ordinal/3,-ordinal/2]
                Image.new('RGB',(16,12),(ordinal,20,30)).save(source/f'{i}.png')
                frames.append({'file_path':f'{i}.png','transform_matrix':pose.tolist()})
                expected[(role,i)] = pose
            (source/'transforms.json').write_text(json.dumps(dict(fl_x=8.,fl_y=8.,cx=7.5,cy=5.5,w=16,h=12,camera_angle_x=math.pi/2,frames=frames)))
            sources[role] = source
        out = root/'assembled'
        receipt = assemble(**sources,output=out,initial_count=2,acquired_count=1)
        cameras = {split:reader(str(out),f'transforms_{split}.json',False) for split in ('train','test')}
        assert len(cameras['train'])==3 and len(cameras['test'])==2
        names = [camera.image_name for group in cameras.values() for camera in group]
        assert len(names)==len(set(names)), 'PGSR output filenames collide'
        raster_inputs = []
        expected_pixels = []
        for split, group in cameras.items():
            rows = [r for r in receipt['records'] if r['split']==split]
            for camera, row in zip(group,rows):
                assert Path(camera.image_path).is_file()
                np.testing.assert_allclose([camera.fx,camera.fy],[8.,8.],atol=1e-12)
                original = expected[(row['role'],int(Path(row['source_frame']).stem))]
                # Independently recover camera center and known world-point projection.
                np.testing.assert_allclose(-camera.R@camera.T,original[:3,3],atol=1e-12)
                camera_gl_point = np.array([0.25,-0.5,-2.])
                world_point = original[:3,:3]@camera_gl_point + original[:3,3]
                camera_cv_point = camera.R.T@world_point + camera.T
                np.testing.assert_allclose(camera_cv_point,[0.25,0.5,2.],atol=1e-12)
                pixel = camera_cv_point[:2]/camera_cv_point[2]*[camera.fx,camera.fy] + [7.5,5.5]
                np.testing.assert_allclose(pixel,[8.5,7.5],atol=1e-12)
                ndc = camera_cv_point[:2]/camera_cv_point[2]/np.tan([camera.FovX/2,camera.FovY/2])
                raster_inputs.extend(zip(ndc,[camera.width,camera.height]))
                expected_pixels.extend(pixel)
        # Execute the actual CUDA helper's arithmetic as C++, stripping only
        # its device annotation; this checks pixel convention without a GPU.
        header=(pgsr/'submodules/diff-plane-rasterization/cuda_rasterizer/auxiliary.h').read_text()
        match=re.search(r'float ndc2Pix\(float v, int S\)\s*\{([^}]+)\}',header)
        assert match, 'Upstream ndc2Pix signature changed'
        cpp=root/'ndc.cpp'
        cpp.write_text('#include <iostream>\n#include <iomanip>\nfloat ndc2Pix(float v,int S){'+match.group(1)+'}\nint main(){float v;int s;std::cout<<std::setprecision(10);while(std::cin>>v>>s)std::cout<<ndc2Pix(v,s)<<"\\n";}')
        subprocess.run(['c++',str(cpp),'-o',str(root/'ndc')],check=True,capture_output=True)
        result=subprocess.run([str(root/'ndc')],input=''.join(f'{v} {size}\n' for v,size in raster_inputs),text=True,capture_output=True,check=True)
        np.testing.assert_allclose([float(x) for x in result.stdout.split()],expected_pixels,atol=1e-6)
        return dict(rasterizer_pixel_helper='actual upstream C++ arithmetic',camera_reader='actual upstream function, isolated from CUDA imports',training_views=3,test_views=2,unique_output_names=True,pose_and_intrinsics_preserved=True,gpu_backend_tested=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pgsr',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(verify(args.pgsr)))
