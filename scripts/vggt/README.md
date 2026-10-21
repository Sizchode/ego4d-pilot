# VGGT baseline execution

VGGT uses the same frozen scene manifests, frame order, GT isolation, evaluator,
fusion code, metrics and artifact validation as the completed DA3 baseline.
Everything remote lives under `/oscar/scratch/zliu328/3dv-reconstruction`; no
`sbach` storage and no language-model API are used.

## Frozen implementation

- Code and model: exact upstream revisions and the verified weight digest are
  recorded in the local run manifest, which is intentionally excluded from Git.
- Primary reconstruction: VGGT depth head plus predicted cameras, as recommended
  by the official repository. The point-map head is retained losslessly as raw
  evidence but is not selected or evaluated using benchmark scores.
- Inference: official crop preprocessor and bfloat16 on B200. No frame reduction,
  retry policy, repair, BA, or test-time parameter search is permitted.

The official loader's exact resize, crop, and padding transform is recorded for
every image. Predictions are deterministically re-expressed on a 518×518
full-image evaluation canvas before entering the common evaluator. Intrinsics
receive the identical homography, so camera rays are unchanged. Areas removed by
the official crop remain zero-depth invalid regions; they are never completed.
The untouched network rasters and point maps remain under `raw/network/`.

## Execution gate

Preparation job `7669160` was submitted on 2026-10-20. It builds a separate
environment in scratch, verifies both code revisions and the existing DA3-BENCH
assets, downloads only the pinned safetensors/config files, checks the model
digest, and records the resolved environment.

After preparation succeeds, `vggt_smoke.slurm` runs the same first official
HiRoom scene used by DA3. A human-readable artifact review must accept its input
manifest, raw branches, coordinate transforms, camera/depth previews, official
metrics and runtime before `vggt_batch.slurm` can run the 103 scene/config entries
serially. The batch runner refuses changed per-scene source hashes and does not
use Slurm arrays.

The accepted smoke is `7669429`. Main job `7669620` is running the frozen
103-entry manifest serially. Standard and visibility-aware failure-mining jobs
`7671926` and `7671927` are queued with `afterok:7669620` dependencies.
