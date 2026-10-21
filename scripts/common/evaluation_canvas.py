"""Deterministic pixel-coordinate conversion for the DA3 benchmark adapter.

Feed-forward reconstruction models preprocess images in different ways, while
the DA3 benchmark interprets every exported raster as covering the full source
image.  These helpers explicitly map model rasters onto a full-image evaluation
canvas and transform intrinsics by the same homography.  Pixels discarded by a
model's official crop remain invalid zero depth.
"""

import numpy as np


def derive_evaluation_homographies(original_wh, original_to_processed, output_hw):
    """Return processed-to-evaluation and original-to-evaluation homographies."""
    original_to_processed = np.asarray(original_to_processed)
    output_height, output_width = output_hw
    if output_height <= 0 or output_width <= 0:
        raise ValueError(f"Invalid evaluation canvas: {output_hw}")
    if len(original_wh) != original_to_processed.shape[0]:
        raise ValueError("Image sizes and preprocessing transforms differ in length")

    processed_to_evaluation = []
    original_to_evaluation = []
    for index, (original_width, original_height) in enumerate(original_wh):
        if original_width <= 0 or original_height <= 0:
            raise ValueError(f"Invalid original image size: {(original_width, original_height)}")
        original_to_output = np.array(
            [
                [output_width / original_width, 0.0, 0.0],
                [0.0, output_height / original_height, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
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
    """Warp camera rasters to a full-source-image canvas while preserving rays."""
    import cv2

    depth = np.asarray(depth)
    confidence = np.asarray(confidence)
    processed_images = np.asarray(processed_images)
    intrinsics = np.asarray(intrinsics)
    original_to_processed = np.asarray(original_to_processed)
    count = len(original_wh)
    if not (
        depth.shape[0]
        == confidence.shape[0]
        == processed_images.shape[0]
        == intrinsics.shape[0]
        == original_to_processed.shape[0]
        == count
    ):
        raise ValueError("Prediction and preprocessing batch sizes differ")

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
        normalized_depth.append(
            cv2.warpPerspective(
                depth[index],
                processed_to_output,
                size,
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0.0,
            )
        )
        normalized_confidence.append(
            cv2.warpPerspective(
                confidence[index],
                processed_to_output,
                size,
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0.0,
            )
        )
        normalized_images.append(
            cv2.warpPerspective(
                processed_images[index],
                processed_to_output,
                size,
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(1.0, 1.0, 1.0),
            )
        )
        normalized_intrinsics.append(processed_to_output @ intrinsics[index])

    return {
        "depth": np.stack(normalized_depth).astype(np.float32),
        "confidence": np.stack(normalized_confidence).astype(np.float32),
        "images": np.stack(normalized_images).astype(np.float32),
        "intrinsics": np.stack(normalized_intrinsics).astype(np.float32),
        "processed_to_evaluation": processed_to_evaluation.astype(np.float32),
        "original_to_evaluation": original_to_evaluation.astype(np.float32),
    }
