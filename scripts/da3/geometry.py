"""Numerical diagnostics only; these functions never modify a prediction."""
import numpy as np
from scipy.ndimage import map_coordinates


def pixel_grid(height, width):
    y, x = np.indices((height, width), dtype=np.float64)
    return np.stack((x, y, np.ones_like(x)), axis=-1)


def unproject(depth, intrinsics, world_to_camera):
    rays = pixel_grid(*depth.shape) @ np.linalg.inv(intrinsics).T
    camera = rays * depth[..., None]
    return (camera - world_to_camera[:3, 3]) @ world_to_camera[:3, :3]


def project(points, intrinsics, world_to_camera):
    camera = points @ world_to_camera[:3, :3].T + world_to_camera[:3, 3]
    image_h = camera @ intrinsics.T
    uv = np.full(image_h.shape[:-1] + (2,), np.nan)
    np.divide(image_h[..., :2], image_h[..., 2:3], out=uv,
              where=image_h[..., 2:3] != 0)
    return uv, camera[..., 2]


def pair_diagnostics(depth_a, depth_b, ka, kb, ea, eb):
    """Return full-resolution directed residuals, without a heuristic inlier threshold.

    projectable means in bounds, positive depth and finite samples; it does not
    assert true visibility. Positive signed_z_residual means the projected source
    is behind the target's predicted surface (possible occlusion or model error).
    """
    points = unproject(depth_a, ka, ea)
    uv, projected_z = project(points, kb, eb)
    h, w = depth_b.shape
    in_bounds = (np.isfinite(uv).all(-1) & (uv[..., 0] >= 0) &
                 (uv[..., 0] <= w - 1) & (uv[..., 1] >= 0) & (uv[..., 1] <= h - 1))
    samples = map_coordinates(depth_b, [uv[..., 1], uv[..., 0]], order=1,
                              mode="constant", cval=np.nan, prefilter=False)
    valid = (in_bounds & np.isfinite(depth_a) & (depth_a > 0) &
             np.isfinite(samples) & (samples > 0) & (projected_z > 0))
    target_rays = np.concatenate((uv, np.ones(uv.shape[:-1] + (1,))), axis=-1) @ np.linalg.inv(kb).T
    target_camera = target_rays * samples[..., None]
    target_world = (target_camera - eb[:3, 3]) @ eb[:3, :3]
    roundtrip_uv, roundtrip_z = project(target_world, ka, ea)
    valid &= np.isfinite(roundtrip_uv).all(-1) & (roundtrip_z > 0)
    signed_z = np.where(valid, projected_z - samples, np.nan)
    relative_z = np.full_like(signed_z, np.nan)
    np.divide(signed_z, samples, out=relative_z, where=valid)
    reprojection = np.linalg.norm(roundtrip_uv - pixel_grid(*depth_a.shape)[..., :2], axis=-1)
    return {"target_uv": uv.astype(np.float32), "projectable": valid,
            "signed_z_residual": signed_z.astype(np.float32),
            "relative_z_residual": relative_z.astype(np.float32),
            "roundtrip_pixels": np.where(valid, reprojection, np.nan).astype(np.float32)}
