"""Metadata for VGGT's official image preprocessing.

The model input itself is always produced by VGGT's
``load_and_preprocess_images``.  This module derives the corresponding pixel
homography and records every resize, crop, and padding operation so predictions
can be traced back to the original image without reimplementing the loader.
"""
from pathlib import Path

import numpy as np
from PIL import Image


def derive_preprocess_metadata(original_wh, mode, target_size=518):
    """Return final shape, homographies, and operation records for VGGT.

    ``original_wh`` is an ordered sequence of ``(width, height)`` pairs.  The
    formulas mirror the pinned upstream loader.  They contain no data-dependent
    quality threshold or fallback.
    """
    if not original_wh:
        raise ValueError("At least one image is required")
    if mode not in {"crop", "pad"}:
        raise ValueError("VGGT preprocessing mode must be 'crop' or 'pad'")

    records = []
    for width, height in original_wh:
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid image size: {(width, height)}")
        if mode == "pad":
            if width >= height:
                resized_width = target_size
                resized_height = round(height * (resized_width / width) / 14) * 14
            else:
                resized_height = target_size
                resized_width = round(width * (resized_height / height) / 14) * 14
        else:
            resized_width = target_size
            resized_height = round(height * (resized_width / width) / 14) * 14

        crop_top = (resized_height - target_size) // 2 if mode == "crop" and resized_height > target_size else 0
        cropped_height = min(resized_height, target_size) if mode == "crop" else resized_height
        cropped_width = resized_width

        if mode == "pad":
            prepad_top = (target_size - cropped_height) // 2
            prepad_left = (target_size - cropped_width) // 2
            stage_height = target_size
            stage_width = target_size
        else:
            prepad_top = 0
            prepad_left = 0
            stage_height = cropped_height
            stage_width = cropped_width

        records.append({
            "original_wh": [width, height],
            "resized_wh": [resized_width, resized_height],
            "crop_top": crop_top,
            "crop_left": 0,
            "prepad_top": prepad_top,
            "prepad_left": prepad_left,
            "stage_hw": [stage_height, stage_width],
        })

    final_height = max(record["stage_hw"][0] for record in records)
    final_width = max(record["stage_hw"][1] for record in records)
    transforms = []
    for record in records:
        stage_height, stage_width = record["stage_hw"]
        batch_pad_top = (final_height - stage_height) // 2
        batch_pad_left = (final_width - stage_width) // 2
        width, height = record["original_wh"]
        resized_width, resized_height = record["resized_wh"]
        translate_x = record["prepad_left"] + batch_pad_left - record["crop_left"]
        translate_y = record["prepad_top"] + batch_pad_top - record["crop_top"]
        transform = np.array([
            [resized_width / width, 0.0, translate_x],
            [0.0, resized_height / height, translate_y],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)
        record["batch_pad_top"] = batch_pad_top
        record["batch_pad_left"] = batch_pad_left
        record["final_hw"] = [final_height, final_width]
        transforms.append(transform)

    return (final_height, final_width), np.stack(transforms), records


def load_and_preprocess_with_metadata(image_paths, mode):
    """Run the pinned official loader and attach independently derived metadata."""
    from vggt.utils.load_fn import load_and_preprocess_images

    paths = [str(Path(path)) for path in image_paths]
    original_wh = []
    for path in paths:
        with Image.open(path) as image:
            original_wh.append(image.size)
    expected_hw, transforms, records = derive_preprocess_metadata(original_wh, mode)
    images = load_and_preprocess_images(paths, mode=mode)
    actual_hw = tuple(images.shape[-2:])
    if actual_hw != expected_hw:
        raise RuntimeError(f"VGGT loader shape {actual_hw} differs from derived shape {expected_hw}")
    return images, transforms, records


def derive_evaluation_homographies(original_wh, original_to_processed, output_hw):
    """Return processed→evaluation and original→evaluation homographies."""
    original_to_processed = np.asarray(original_to_processed)
    output_height, output_width = output_hw
    if output_height <= 0 or output_width <= 0:
        raise ValueError(f"Invalid evaluation canvas: {output_hw}")
    if len(original_wh) != original_to_processed.shape[0]:
        raise ValueError("Image sizes and preprocessing transforms differ in length")
    processed_to_evaluation = []
    original_to_evaluation = []
    for index, (original_width, original_height) in enumerate(original_wh):
        original_to_output = np.array([
            [output_width / original_width, 0.0, 0.0],
            [0.0, output_height / original_height, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        processed_to_output = original_to_output @ np.linalg.inv(original_to_processed[index])
        processed_to_evaluation.append(processed_to_output)
        original_to_evaluation.append(original_to_output)
    return np.stack(processed_to_evaluation), np.stack(original_to_evaluation)


def normalize_to_evaluation_canvas(
    depth,
    confidence,
    processed_images,
    intrinsics,
    original_wh,
    original_to_processed,
    output_hw,
):
    """Express VGGT predictions on a full-image evaluation canvas.

    The official loader may crop or pad its tensor.  The DA3 benchmark reader,
    however, rescales every exported raster directly to the corresponding
    original image.  This deterministic change of pixel coordinates removes
    loader padding and records cropped regions as invalid zero depth.  Camera
    rays are preserved by applying the same homography to the intrinsics.
    """
    import cv2

    depth = np.asarray(depth)
    confidence = np.asarray(confidence)
    processed_images = np.asarray(processed_images)
    intrinsics = np.asarray(intrinsics)
    original_to_processed = np.asarray(original_to_processed)
    count = len(original_wh)
    if not (
        depth.shape[0] == confidence.shape[0] == processed_images.shape[0]
        == intrinsics.shape[0] == original_to_processed.shape[0] == count
    ):
        raise ValueError("VGGT prediction and preprocessing batch sizes differ")
    output_height, output_width = output_hw
    processed_to_evaluation, original_to_evaluation = derive_evaluation_homographies(
        original_wh, original_to_processed, output_hw
    )

    normalized_depth = []
    normalized_confidence = []
    normalized_images = []
    normalized_intrinsics = []
    for index in range(count):
        processed_to_output = processed_to_evaluation[index]
        size = (output_width, output_height)
        normalized_depth.append(cv2.warpPerspective(
            depth[index], processed_to_output, size,
            flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0,
        ))
        normalized_confidence.append(cv2.warpPerspective(
            confidence[index], processed_to_output, size,
            flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0,
        ))
        normalized_images.append(cv2.warpPerspective(
            processed_images[index], processed_to_output, size,
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(1.0, 1.0, 1.0),
        ))
        normalized_intrinsics.append(processed_to_output @ intrinsics[index])

    return {
        "depth": np.stack(normalized_depth).astype(np.float32),
        "confidence": np.stack(normalized_confidence).astype(np.float32),
        "images": np.stack(normalized_images).astype(np.float32),
        "intrinsics": np.stack(normalized_intrinsics).astype(np.float32),
        "processed_to_evaluation": processed_to_evaluation.astype(np.float32),
        "original_to_evaluation": original_to_evaluation.astype(np.float32),
    }
