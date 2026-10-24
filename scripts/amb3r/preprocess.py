"""Official AMB3R demo RGB preprocessing, with actual raster transforms traced.

No dataset __getitem__ or pseudo-depth/camera samples are constructed. The
image-center matrix is used only by the official crop helper, never the model.
"""
from pathlib import Path
import sys
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'recovery'))
from geometry_state import pixel_resize


def prepare_rgb(image_paths, seed=47):
    import torch
    from dust3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
    import dust3r.datasets.utils.cropping as cropping
    dataset = BaseStereoViewDataset(resolution=(518, 392), aug_crop=False, seed=seed)
    rng = np.random.default_rng(seed)
    images, arrays, matrices, audits = [], [], [], []
    for path in image_paths:
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f'Cannot decode input image: {path}')
        image = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        w, h = image.size
        image_center = np.array([[1., 0, w//2], [0, 1., h//2], [0, 0, 1.]], dtype=np.float32)
        matrix = np.eye(3)
        operations = []
        original_crop, original_resize = cropping.crop_image_depthmap, cropping.rescale_image_depthmap

        def crop(rgb, depth, k, box):
            nonlocal matrix
            result = original_crop(rgb, depth, k, box)
            left, top, right, bottom = box
            matrix = np.array([[1., 0, -left], [0, 1., -top], [0, 0, 1.]])@matrix
            operations.append({'operation': 'crop', 'bbox': [int(x) for x in box]})
            return result

        def resize(rgb, depth, k, resolution):
            nonlocal matrix
            before = rgb.size[::-1]
            result = original_resize(rgb, depth, k, resolution)
            after = result[0].size[::-1]
            matrix = pixel_resize(before, after)@matrix
            operations.append({'operation': 'resize', 'input_hw': list(before), 'output_hw': list(after)})
            return result

        with patch.object(cropping, 'crop_image_depthmap', crop), \
                patch.object(cropping, 'rescale_image_depthmap', resize):
            image, _, _ = dataset._crop_resize_if_necessary(
                image, None, image_center, (518, 392), rng=rng, info=str(path), disable_crop=False)
        rgb = np.asarray(image).copy()
        # Official transpose_to_landscape uses a clockwise rotation, not xy swap.
        if image.width < image.height:
            matrix = np.array([[0., -1, image.height-1], [1., 0, 0], [0, 0, 1.]])@matrix
            rgb = np.rot90(rgb, k=-1).copy()
            operations.append({'operation': 'rotate_clockwise_90'})
        images.append(dataset.transform(Image.fromarray(rgb)))
        arrays.append(rgb)
        matrices.append(matrix)
        audits.append(operations)
    return torch.stack(images)[None], np.stack(arrays), np.stack(matrices), audits
