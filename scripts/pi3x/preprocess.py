"""Exact metadata for Pi3X's official image loader."""

import math
from pathlib import Path

import numpy as np
from PIL import Image


def target_size_from_first_image(first_wh, pixel_limit):
    """Mirror ``pi3.utils.basic.load_images_as_tensor`` target selection."""
    width, height = first_wh
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid first image size: {first_wh}")
    if pixel_limit <= 0:
        raise ValueError(f"Invalid pixel limit: {pixel_limit}")
    scale = math.sqrt(pixel_limit / (width * height))
    target_width = width * scale
    target_height = height * scale
    k = round(target_width / 14)
    m = round(target_height / 14)
    while (k * 14) * (m * 14) > pixel_limit:
        if k / m > target_width / target_height:
            k -= 1
        else:
            m -= 1
    return max(1, k) * 14, max(1, m) * 14


def derive_preprocess_metadata(original_wh, target_wh):
    """Return per-view transforms for the official direct resize."""
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
        scale_x = target_width / width
        scale_y = target_height / height
        transforms.append(
            np.array([[scale_x, 0.0, 0.0], [0.0, scale_y, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        )
        records.append(
            {
                "original_wh": [width, height],
                "target_wh": [target_width, target_height],
                "scale_xy": [scale_x, scale_y],
                "resize_filter": "PIL.Image.Resampling.LANCZOS",
            }
        )
    return np.stack(transforms), records


def stage_and_load_images(image_paths, stage_dir, *, pixel_limit):
    """Call the official directory loader on an order-preserving symlink stage."""
    from pi3.utils.basic import load_images_as_tensor

    paths = [Path(path).resolve() for path in image_paths]
    if not paths:
        raise ValueError("At least one benchmark image is required")
    stage_dir = Path(stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=False)
    original_wh = []
    for index, path in enumerate(paths):
        with Image.open(path) as image:
            original_wh.append(image.size)
        suffix = path.suffix.lower()
        if suffix not in {".png", ".jpg", ".jpeg"}:
            raise ValueError(f"Unsupported Pi3X input extension: {path}")
        (stage_dir / f"{index:06d}{suffix}").symlink_to(path)

    target_wh = target_size_from_first_image(original_wh[0], pixel_limit)
    transforms, records = derive_preprocess_metadata(original_wh, target_wh)
    images = load_images_as_tensor(
        str(stage_dir), interval=1, PIXEL_LIMIT=pixel_limit, verbose=False
    )
    if images.shape[0] != len(paths):
        raise RuntimeError(f"Pi3X loaded {images.shape[0]} of {len(paths)} benchmark images")
    if tuple(images.shape[-2:]) != (target_wh[1], target_wh[0]):
        raise RuntimeError(
            f"Pi3X loader shape {tuple(images.shape[-2:])} differs from derived shape "
            f"{(target_wh[1], target_wh[0])}"
        )
    return images, transforms, records, original_wh, target_wh
