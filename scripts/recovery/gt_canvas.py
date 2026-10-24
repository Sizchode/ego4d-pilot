"""Evaluator-only depth/ray conversion onto the fixed RGB public canvas.

The source depth camera and RGB camera calibration are explicit. No predicted
camera, confidence, error, or model output may influence the GT valid domain.
"""
import numpy as np

try:
    from .geometry_state import pixel_resize
    from .public_canvas import public_shape, pixel_grid, transform_pixels
except ImportError:
    from geometry_state import pixel_resize
    from public_canvas import public_shape, pixel_grid, transform_pixels


def nearest_exact(values, valid, uv):
    """Nearest pixel-center sampling, including the half-pixel edge support."""
    values, valid, uv = np.asarray(values), np.asarray(valid), np.asarray(uv)
    if values.ndim != 2 or valid.shape != values.shape or valid.dtype != bool or uv.shape[-1] != 2:
        raise ValueError('Expected HxW values/mask and matching pixel-coordinate queries')
    h, w = values.shape
    inside = np.isfinite(uv).all(-1)&(uv[..., 0]>=-.5)&(uv[..., 0]<w-.5)&(uv[..., 1]>=-.5)&(uv[..., 1]<h-.5)
    safe = np.where(inside[..., None], uv, 0)
    indices = np.floor(safe+.5).astype(int)
    sampled = values[indices[..., 1], indices[..., 0]]
    supported = inside&valid[indices[..., 1], indices[..., 0]]&np.isfinite(sampled)
    return np.where(supported, sampled, np.nan), supported


def opencv_rays(pixels, intrinsics, distortion):
    import cv2
    pixels = np.asarray(pixels, dtype=np.float64)
    xy = cv2.undistortPointsIter(
        pixels.reshape(-1, 1, 2), np.asarray(intrinsics, dtype=np.float64),
        np.asarray(distortion, dtype=np.float64), None, None,
        (cv2.TERM_CRITERIA_COUNT|cv2.TERM_CRITERIA_EPS, 100, 1e-12))
    xy = xy.reshape(pixels.shape)
    return np.concatenate([xy, np.ones(xy.shape[:-1]+(1,))], -1)


def rendered_depth_to_public(depth, depth_valid, original_hw, rgb_intrinsics,
                             rgb_distortion, depth_intrinsics, world_to_camera):
    """A pinhole depth rendering and distorted RGB share the same camera pose.

    First recover each original RGB pixel's physical ray. Project that ray into
    the renderer's pinhole raster to sample camera-Z, then place it in the world.
    This also handles already undistorted RGB using explicit zero distortion.
    """
    shape = public_shape(original_hw)
    original_to_public = pixel_resize(original_hw, shape)
    original_uv = transform_pixels(np.linalg.inv(original_to_public), pixel_grid(shape))
    rays = opencv_rays(original_uv, rgb_intrinsics, rgb_distortion)
    depth_homogeneous = np.einsum('ij,hwj->hwi', depth_intrinsics, rays)
    depth_uv = depth_homogeneous[..., :2]/depth_homogeneous[..., 2:]
    sampled_depth, valid = nearest_exact(depth, depth_valid, depth_uv)
    valid &= sampled_depth>0
    camera_points = rays*sampled_depth[..., None]
    rotation, translation = world_to_camera[:3, :3], world_to_camera[:3, 3]
    points = (camera_points-translation)@rotation
    valid &= np.isfinite(points).all(-1)
    points[~valid] = np.nan
    return {'points': points.astype(np.float32), 'valid': valid,
            'depth': sampled_depth.astype(np.float32), 'depth_uv': depth_uv.astype(np.float32),
            'original_to_public': original_to_public, 'extrinsics': world_to_camera}
