# Audited MapAnything baseline

This adapter runs the pinned official MapAnything image-only inference path on
the six DA3-BENCH configurations used by the baseline study. It supplies no
camera, depth, pose, scale, or ground-truth metadata to the model.

The primary result follows the official image-only example: memory-efficient
inference with BF16, the learned non-ambiguous mask, and the normal/depth edge
mask. The evaluator receives `depth_z`, recovered pinhole intrinsics, and the
inverse of the predicted OpenCV camera-to-world pose. The official `pts3d`,
`pts3d_cam`, masks, logits, confidence, ray and scale outputs are archived for
failure analysis but are not selected using benchmark scores.

MapAnything resizes every sequence to one fixed-mapping aspect ratio and
center-crops individual frames. The adapter records the exact upstream
rescale/crop transform, maps the prediction to a full-image 518×518 evaluation
canvas, applies the same homography to intrinsics, and keeps unseen crop regions
invalid. This is a coordinate conversion only; it does not fill depth or alter
predicted geometry.

Execution is gated in four stages:

1. `mapanything_prepare.slurm` pins code, weights, environment, and benchmark assets.
2. `mapanything_smoke.slurm` runs tests and one fixed HiRoom scene.
3. A manual `review.json` records numerical and visual artifact acceptance.
4. `mapanything_batch.slurm` verifies source/config hashes and runs all 103 scenes serially.

All data, weights, environments, logs, runs, and analysis outputs live below
`/oscar/scratch/zliu328/3dv-reconstruction`.

## Reproducibility status

The inspected MapAnything revision constructs its VGGT patch embed through
`torch.hub.load("facebookresearch/dinov2", ...)`. The implicit DINOv2 dependency
is frozen separately; both exact revisions are recorded in the local run
manifest, which is intentionally excluded from Git. The Torch Hub cache entry is a
verified symlink to that detached checkout, and smoke/main launchers block HTTP
access so a missing dependency fails instead of downloading mutable code.

Smoke job `7671294` completed end to end but downloaded DINOv2 `main`; it is
retained only as interface evidence and is not eligible for the batch gate.
Pinning job `7671822` completed with exit code 0 and recorded all three code
revisions plus the MapAnything weight digest. Offline replacement smoke job
`7671877` completed inference and artifact validation without a network download.
Numerical audit job `7672097` reproduced all official metrics and provenance
checks; the saved preview and camera-error matrix also passed visual review.
Main job `7672161` is running the frozen 103-entry manifest. Standard and
visibility-aware failure-mining jobs `7672164` and `7672165` are queued with
`afterok:7672161` dependencies.
