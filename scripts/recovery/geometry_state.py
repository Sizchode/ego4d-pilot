"""Prediction-only geometry primitives, independent of any evaluator or GT.

Point maps are authoritative. Depth is only an input conversion or a display
quantity; a refined point map must never be replaced by depth unprojection.
"""
import numpy as np


def pixel_resize(source_hw, target_hw):
    h, w = source_hw
    th, tw = target_hw
    if min(h, w, th, tw) <= 0:
        raise ValueError('Image dimensions must be positive')
    sx, sy = tw / w, th / h
    return np.array([[sx, 0, (sx - 1) / 2], [0, sy, (sy - 1) / 2], [0, 0, 1.]])


def da3_image_transforms(original_hw, process_res, observed_hw, calibration_transforms):
    """Trace upstream upper_bound_resize, then batch center crop.

    DA3 scales K without pixel-center translation. Its OpenCV image resizes do
    use pixel centers. These two transforms are recorded/checked separately.
    This implementation covers the preprocessing explicitly selected in our
    baseline configuration; other preprocessing modes require their own trace.
    """
    original_hw = np.asarray(original_hw, dtype=int)
    if original_hw.ndim != 2 or original_hw.shape[1] != 2 or (original_hw <= 0).any():
        raise ValueError('Expected positive N x 2 original dimensions')
    sized = []
    for h, w in original_hw:
        first = [max(1, round(int(side) * process_res / max(h, w))) for side in (h, w)]
        # Upstream chooses the upper multiple on exact half-patch ties.
        final = [max(1, 14 * ((side + 7) // 14)) for side in first]
        sized.append(final)
    sized = np.asarray(sized)
    common = sized.min(0)
    if tuple(common) != tuple(observed_hw):
        raise ValueError('Observed DA3 canvas disagrees with native preprocessing')
    image_transforms, expected_calibration = [], []
    for source, resized in zip(original_hw, sized):
        crop_y, crop_x = (resized - common) // 2
        crop = np.array([[1., 0, -crop_x], [0, 1., -crop_y], [0, 0, 1.]])
        image_transforms.append(crop @ pixel_resize(source, resized))
        scale = np.diag([resized[1] / source[1], resized[0] / source[0], 1.])
        expected_calibration.append(crop @ scale)
    # Calibration was produced in float32 by upstream; tolerate its rounding.
    np.testing.assert_allclose(calibration_transforms, expected_calibration,
                               rtol=4 * np.finfo(np.float32).eps, atol=0)
    return np.stack(image_transforms)


def unproject_depth(depth, intrinsics, world_to_camera):
    depth, intrinsics, world_to_camera = map(np.asarray, (depth, intrinsics, world_to_camera))
    n, h, w = depth.shape
    if intrinsics.shape != (n, 3, 3) or world_to_camera.shape not in {(n, 3, 4), (n, 4, 4)}:
        raise ValueError('Depth and camera arrays have incompatible shapes')
    y, x = np.indices((h, w), dtype=np.float64)
    pixels = np.stack([x, y, np.ones_like(x)], -1)
    rays = np.einsum('hwj,nij->nhwi', pixels, np.linalg.inv(intrinsics))
    camera = rays * depth[..., None]
    rotation, translation = world_to_camera[:, :3, :3], world_to_camera[:, :3, 3]
    return np.einsum('nhwj,nji->nhwi', camera - translation[:, None, None, :], rotation)


def camera_z(points, world_to_camera):
    """Derived display depth; never sufficient to recover a refined point map."""
    return np.einsum('nhwj,nj->nhw', points, world_to_camera[:, 2, :3]) + world_to_camera[:, None, None, 2, 3]


def sample_pointmap(points, valid, uv):
    """Bilinear 3D sampling with strict validity of all contributing neighbors."""
    points, valid, uv = np.asarray(points), np.asarray(valid), np.asarray(uv)
    h, w, xyz = points.shape
    if xyz != 3 or valid.shape != (h, w) or valid.dtype != bool or uv.shape[-1] != 2:
        raise ValueError('Malformed point-map sampling arguments')
    inside = (np.isfinite(uv).all(-1) & (uv[..., 0] >= 0) & (uv[..., 0] <= w - 1)
              & (uv[..., 1] >= 0) & (uv[..., 1] <= h - 1))
    safe = np.where(inside[..., None], uv, 0)
    x0 = np.floor(safe[..., 0]).astype(int)
    y0 = np.floor(safe[..., 1]).astype(int)
    # Replicated endpoint is the bilinear definition at the last row/column.
    x1, y1 = np.minimum(x0 + 1, w - 1), np.minimum(y0 + 1, h - 1)
    dx, dy = safe[..., 0] - x0, safe[..., 1] - y0
    result = np.zeros(uv.shape[:-1] + (3,), dtype=np.result_type(points, np.float64))
    supported = inside.copy()
    for yy, xx, weight in [(y0, x0, (1-dx)*(1-dy)), (y0, x1, dx*(1-dy)),
                           (y1, x0, (1-dx)*dy), (y1, x1, dx*dy)]:
        neighbor = points[yy, xx]
        supported &= valid[yy, xx] & np.isfinite(neighbor).all(-1)
        result += np.where(np.isfinite(neighbor), neighbor, 0) * weight[..., None]
    result[~supported] = np.nan
    return result, supported
