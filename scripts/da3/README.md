# DA3 baseline execution

All remote inputs, environments, caches, logs and results live under
`/oscar/scratch/zliu328/3dv-reconstruction`. Do not use `sbach` storage.
No language model or API key is used by this baseline.

## Current execution state — 2026-10-20

- Preparation job `7644988` completed with exit code 0. DA3-GIANT and all six
  benchmark archives have pinned revisions, SHA-256 hashes, sizes and validated
  extraction records.
- Single-scene job `7646382` completed with exit code 0 on a B200. The first
  official HiRoom scene contains 23 views. All required artifacts passed the
  numerical and visual audit recorded in `runs/smoke-hiroom-7646382/review.json`.
- Initial main job `7649490` failed before inference because its launcher
  incorrectly asserted that HiRoom had 24 scenes. The pinned registry contains
  30. No scene result was produced by this failed job.
- Replacement main DA3 job `7649571` completed all 103 scene/config entries
  serially with exit code 0. It writes `runs/batch-7649571` plus one run
  directory per scene. Every entry is completed or the previously accepted
  smoke scene reused; there are no execution failures.
- Failure-mining smoke-test job `7649530` completed successfully on the accepted
  scene (44 pair rows and two surface-distance rows). Job `7649531` failed only
  because the original main job never created a manifest. Replacement formal
  analysis job `7649573` completed with exit code 0 and writes
  `analysis/da3-batch-7649571`: 103 scene rows, 10,358 directed pair rows, 180
  surface-distance rows and zero execution-failure records.
- Failure mining preserves continuous per-scene and per-pair evidence and emits
  separate rankings for interpretable metrics. It does not assign taxonomy
  labels or combine metrics with heuristic weights.
- No scheduler polling loop is running and no related job remains queued. All
  paths above are relative to the scratch root; job logs are under `logs/`.
- The first evidence review is documented in
  `research/2026-10-19-3dv-agent/da3-main-results-failure-mining.md`. It finds
  both visibility-induced false alarms and self-consistent pose failures, so raw
  cross-view residuals must not be treated as failure labels.

## Frozen inputs

`configs/da3_pinned.json` records immutable Git and Hugging Face revisions.
`da3_prepare.slurm` creates an isolated Python 3.11 environment, downloads
DA3-GIANT and all six benchmark archives, checks ZIP integrity, and saves
SHA-256 checksums and the resolved Python environment. It uses the CPU batch
partition; model execution is submitted separately to Slurm.

## Single-scene acceptance

`da3_smoke.slurm` requests one B200, 8 CPU cores and 96 GB RAM. It runs the first
scene in the official HiRoom list, with the official 100-frame cap, sampling seed
42, first reference view and 504-pixel preprocessing. It does not choose a scene
based on reconstruction quality. A successful result must include:

- `input_manifest.json`, `run_config.json`, asset checksums, environment and runner source;
- `raw/`: unrounded network and API predictions, processed RGB, pixel transforms,
  camera conventions, valid depth masks and per-view world point maps;
- `evaluation/`: official rounded export, evaluator-only GT metadata, fused cloud,
  and official per-scene metrics;
- `evaluation_capture/`: actual fusion input arrays, alignment transform, pairwise
  camera errors and the exact query points/distances used in surface evaluation;
- `diagnostics/`: full-resolution depth and round-trip pixel residuals for each
  pair of adjacent input views, in both directions;
- preview images, runtime/peak VRAM, typed status and `validation.json`.

Cross-view masks indicate projectability, not proven visibility. Signed depth
ordering is retained without applying an inlier threshold or repairing geometry.
Other view pairs can be regenerated from the retained raw predictions.

## Upstream behavior and instrumentation

The model and valid evaluation formulas are unchanged. The adapter writes the
same rounded `mini_npz` values synchronously and separately retains unrounded
arrays. An observational Python profiler records exact intermediate evaluator
outputs. The upstream mesh-sampling fallback that generates random points is
disabled: invalid meshes raise and are recorded as failures. Open3D sampling is
seeded. Instrumentation time is included in runtime records.

The pinned DTU evaluator names its two directional errors in reverse (`comp`
contains prediction-to-GT and `acc` contains GT-to-prediction). The official JSON
is retained without relabeling; `dtu_surface_distance_*.npz` explicitly records
the two directions. Overall is unaffected. Do not interpret the original labels
without this provenance note.

## Batch gate

Only after numerical and visual review of the single-scene artifacts should
`review.json` record `artifact_review_passed: true`. `da3_batch.slurm` requires
this reviewed run and its successful validation before running 103 scene/config
entries serially, with no Slurm arrays. The verified scene is reused. Every other
entry runs in a fresh process and writes either a validated result or a failure
record; there is no automatic frame reduction, model substitution or repair.

The batch is a baseline archive. It does not use evaluation scenes to design
tools. The separate ScanNet++ training development set has not been prepared by
these scripts and is not silently substituted with validation data.

## Oscar access verified on 2026-10-20

- User: `zliu328`; SSH alias: `jump-box`.
- `gpu-he` QOS: up to 6 GPUs, 84 CPUs and 1176 GB RAM per user.
- `gpu` QOS: up to 2 GPUs, 12 CPUs and 192 GB RAM per user.
- `gpu-debug`: one-hour wall limit; GH200 and scavenger associations also exist.
- Availability and allocation are scheduler-dependent. Existing jobs are untouched.
- Home was over its soft quota; all new task storage is in scratch.
- Follow `/oscar/AGENTS.md`: lock remote agent work, use Slurm for heavy tasks,
  show scripts before submission, and do not poll the scheduler.
