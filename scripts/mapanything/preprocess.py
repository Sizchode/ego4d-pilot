"""Metadata for MapAnything's official fixed-mapping image preprocessing."""

from pathlib import Path

import numpy as np
from PIL import Image
from PIL.ImageOps import exif_transpose


def derive_preprocess_metadata(original_wh, target_wh):
    """Mirror the pinned upstream rescale-then-center-crop transform.

    The upstream implementation chooses one target resolution for the entire
    sequence, rescales each image by ``max(target / input) + 1e-8``, floors the
    resulting integer dimensions, and applies a centered crop.  The homography
    follows upstream's OpenCV↔COLMAP half-pixel camera conversion during resize,
    followed by the integer crop actually used by the image-only loader.
    """
    if not original_wh:
        raise ValueError("At least one image is required")
    target_width, target_height = target_wh
    if target_width <= 0 or target_height <= 0:
        raise ValueError(f"Invalid target image size: {target_wh}")

    transforms = []
    records = []
    for width, height in original_wh:
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid original image size: {(width, height)}")
        scale = max(target_width / width, target_height / height) + 1e-8
        resized_width, resized_height = np.floor(np.array([width, height]) * scale).astype(int)
        crop_left = (resized_width - target_width) // 2
        crop_top = (resized_height - target_height) // 2
        if crop_left < 0 or crop_top < 0:
            raise RuntimeError("Upstream rescale did not contain the requested crop")
        # camera_matrix_of_crop converts OpenCV pixel centers to COLMAP (+0.5),
        # applies the continuous uniform scale, centers the floored raster, and
        # converts back (-0.5).  load_images then takes the no-intrinsics crop
        # branch, whose offset is integer floor division.
        resize_margin_x = width * scale - resized_width
        resize_margin_y = height * scale - resized_height
        translate_x = 0.5 * scale - 0.5 * resize_margin_x - 0.5 - crop_left
        translate_y = 0.5 * scale - 0.5 * resize_margin_y - 0.5 - crop_top
        transforms.append(
            np.array(
                [
                    [scale, 0.0, translate_x],
                    [0.0, scale, translate_y],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            )
        )
        records.append(
            {
                "original_wh": [width, height],
                "target_wh": [target_width, target_height],
                "continuous_scale": float(scale),
                "resized_wh": [int(resized_width), int(resized_height)],
                "resize_center_margin_xy": [
                    float(resize_margin_x), float(resize_margin_y)
                ],
                "crop_left": int(crop_left),
                "crop_top": int(crop_top),
                "opencv_pixel_translation_xy": [
                    float(translate_x), float(translate_y)
                ],
            }
        )
    return np.stack(transforms), records


def load_images_with_metadata(image_paths, *, resolution_set):
    """Run MapAnything's official loader and independently retain its transform."""
    from mapanything.utils.image import find_closest_aspect_ratio, load_images

    paths = [str(Path(path)) for path in image_paths]
    original_wh = []
    for path in paths:
        with Image.open(path) as image:
            original_wh.append(exif_transpose(image).size)
    average_aspect_ratio = sum(width / height for width, height in original_wh) / len(original_wh)
    target_wh = find_closest_aspect_ratio(average_aspect_ratio, resolution_set)
    transforms, records = derive_preprocess_metadata(original_wh, target_wh)
    views = load_images(
        paths,
        resize_mode="fixed_mapping",
        norm_type="dinov2",
        patch_size=14,
        resolution_set=resolution_set,
        stride=1,
    )
    if len(views) != len(paths):
        raise RuntimeError(f"MapAnything loaded {len(views)} of {len(paths)} benchmark images")
    expected_hw = (target_wh[1], target_wh[0])
    for index, view in enumerate(views):
        if tuple(view["img"].shape[-2:]) != expected_hw:
            raise RuntimeError(
                f"MapAnything loader shape {tuple(view['img'].shape[-2:])} "
                f"differs from derived shape {expected_hw} for view {index}"
            )
    return views, transforms, records, original_wh, target_wh
