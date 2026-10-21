"""Threshold-free geometric layers for cross-view depth diagnostics.

The masks in this module are descriptive layers, not visibility ground truth.
They separate three facts that the original ``projectable`` mask conflates:
whether a source point lands on the target raster, whether it is the nearest
projected source point at that target pixel, and whether the target prediction
contains an even closer surface.  No tolerance or learned decision is used.
"""

from __future__ import annotations

import numpy as np

from scripts.da3.geometry import project, unproject


def pair_visibility_layers(depth_a, depth_b, ka, kb, ea, eb) -> dict[str, np.ndarray]:
    """Return deterministic z-buffer and target-depth ordering layers.

    Pixel assignment uses nearest-neighbor rasterization with half-up rounding.
    If several source points land on the same target pixel, the one with the
    smallest positive target-camera Z wins; an exact depth tie is resolved by
    the source pixel's row-major index.  ``target_depth_closer`` means the
    target prediction is strictly in front of that winner.  These masks do not
    assert physical visibility because either predicted depth may be wrong.
    """
    depth_a = np.asarray(depth_a)
    depth_b = np.asarray(depth_b)
    if depth_a.ndim != 2 or depth_b.ndim != 2:
        raise ValueError("depth inputs must be two-dimensional")

    points = unproject(depth_a, ka, ea)
    target_uv, projected_z = project(points, kb, eb)
    height, width = depth_b.shape
    finite_uv = np.isfinite(target_uv).all(axis=-1)
    in_bounds = (
        finite_uv
        & (target_uv[..., 0] >= 0)
        & (target_uv[..., 0] <= width - 1)
        & (target_uv[..., 1] >= 0)
        & (target_uv[..., 1] <= height - 1)
    )
    rasterizable = (
        in_bounds
        & np.isfinite(depth_a)
        & (depth_a > 0)
        & np.isfinite(projected_z)
        & (projected_z > 0)
    )

    target_x = np.full(depth_a.shape, -1, dtype=np.int64)
    target_y = np.full(depth_a.shape, -1, dtype=np.int64)
    target_x[rasterizable] = np.floor(target_uv[..., 0][rasterizable] + 0.5).astype(np.int64)
    target_y[rasterizable] = np.floor(target_uv[..., 1][rasterizable] + 0.5).astype(np.int64)

    source_linear = np.arange(depth_a.size, dtype=np.int64).reshape(depth_a.shape)
    candidate_source = source_linear[rasterizable]
    candidate_target = target_y[rasterizable] * width + target_x[rasterizable]
    candidate_z = projected_z[rasterizable]
    order = np.lexsort((candidate_source, candidate_z, candidate_target))
    ordered_targets = candidate_target[order]
    first_for_target = np.ones(len(order), dtype=bool)
    if len(order) > 1:
        first_for_target[1:] = ordered_targets[1:] != ordered_targets[:-1]
    winner_source = candidate_source[order[first_for_target]]
    source_zbuffer_winner = np.zeros(depth_a.size, dtype=bool)
    source_zbuffer_winner[winner_source] = True
    source_zbuffer_winner = source_zbuffer_winner.reshape(depth_a.shape)

    target_depth_nearest = np.full(depth_a.shape, np.nan, dtype=np.float64)
    target_depth_nearest[source_zbuffer_winner] = depth_b[
        target_y[source_zbuffer_winner], target_x[source_zbuffer_winner]
    ]
    target_depth_available = (
        source_zbuffer_winner
        & np.isfinite(target_depth_nearest)
        & (target_depth_nearest > 0)
    )
    target_depth_closer = target_depth_available & (target_depth_nearest < projected_z)
    source_depth_not_behind = target_depth_available & (projected_z <= target_depth_nearest)

    signed_z_nearest = np.full(depth_a.shape, np.nan, dtype=np.float64)
    signed_z_nearest[target_depth_available] = (
        projected_z[target_depth_available] - target_depth_nearest[target_depth_available]
    )
    relative_z_nearest = np.full(depth_a.shape, np.nan, dtype=np.float64)
    np.divide(
        signed_z_nearest,
        target_depth_nearest,
        out=relative_z_nearest,
        where=target_depth_available,
    )
    return {
        "target_uv": target_uv.astype(np.float32),
        "projected_z": projected_z.astype(np.float32),
        "target_x": target_x,
        "target_y": target_y,
        "rasterizable": rasterizable,
        "source_zbuffer_winner": source_zbuffer_winner,
        "target_depth_available": target_depth_available,
        "target_depth_closer": target_depth_closer,
        "source_depth_not_behind": source_depth_not_behind,
        "target_depth_nearest": target_depth_nearest.astype(np.float32),
        "signed_z_nearest": signed_z_nearest.astype(np.float32),
        "relative_z_nearest": relative_z_nearest.astype(np.float32),
    }
