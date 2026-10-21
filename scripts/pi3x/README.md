# Audited Pi3X baseline

This adapter runs the pinned Pi3X image-only model on the six DA3-BENCH
configurations. The multimodal branch is explicitly disabled, and the model is
given no depth, rays, intrinsics, camera poses, metric scale, or ground truth.

The evaluated branch follows the official `example_mm.py`: intrinsics are
recovered from normalized local points with
`recover_intrinsic_from_rays_d(..., force_center_principal_point=True)`, and
valid pixels satisfy both `sigmoid(conf) > 0.1` and the official
`depth_normal_edge(..., rtol=0.03)` exclusion. Camera-to-world predictions are
inverted into the benchmark's OpenCV world-to-camera convention. Global points,
local points, rays, confidence logits, poses, metric scale, recovered
intrinsics, and masks are all archived before evaluation.

Pi3X's official loader chooses one patch-aligned resolution from the first
image under a 255,000-pixel limit, then directly resizes every selected frame.
The runner stages order-preserving symlinks and calls that loader unchanged. It
records the corresponding per-frame pixel homography and maps outputs to a
full-image 518×518 evaluation canvas while applying the same transform to the
recovered intrinsics.

Preparation, smoke validation, manual artifact review, serial 103-scene batch,
and failure mining are separate hash-gated Slurm stages. All artifacts remain
under `/oscar/scratch/zliu328/3dv-reconstruction`.

The accepted smoke is `7670162`. Main job `7670606` is running the frozen
103-entry manifest serially. Standard and visibility-aware failure-mining jobs
`7671930` and `7671932` are queued with `afterok:7670606` dependencies.
