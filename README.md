# Audited 3D Reconstruction Baselines and Failure Mining

This repository contains the reproducible code and research record for a
training-free 3D reconstruction diagnosis project. The current completed stage
does **not** use an LLM or agent: it establishes the reconstruction baselines,
failure evidence, and deterministic controls that any later VLM policy must
beat.

## Completed baseline study

- Backbones: DA3-Giant, VGGT, π³-X, and MapAnything.
- Population: 103 scenes across DTU-49, DTU-64, ETH3D, ScanNet++, 7Scenes,
  and HiRoom.
- Execution: 412/412 model-scene runs completed with no execution failure.
- Failure mining: 10,358 directed view pairs per model, with standard and
  visibility-controlled diagnostics.
- Reproducibility: inputs, batch outputs, cross-model tables, and final mining
  outputs passed independent audits and local reproduction checks.

Large datasets, model weights, predictions, logs, and generated artifacts stay
on scratch storage and are excluded from Git.

## Repository map

- `scripts/{da3,vggt,pi3x,mapanything}`: model adapters and scene runners.
- `scripts/failure_mining`: cross-view diagnostics, controlled interventions,
  ranking analysis, and report generation.
- `scripts/audit`: population, provenance, hash, and output audits.
- `launchers`: Oscar Slurm launchers for preparation, inference, analysis,
  and audit jobs.
- `configs`: dependency constraints and public run configuration. Exact
  external revisions and file digests remain in local ignored manifests.
- `tests`: local unit and integration tests for adapters and analyses.
- `research/2026-10-19-3dv-agent`: survey, experiment plan, final baseline
  report, and teaching material.

The concise result is in
[`final-baseline-failure-mining.md`](research/2026-10-19-3dv-agent/final-baseline-failure-mining.md).
The current paper scope is in
[`related-work-survey.md`](research/2026-10-19-3dv-agent/related-work-survey.md).

## Local verification

The tests use synthetic fixtures and do not require checkpoints or benchmark
data:

```bash
PYTHONPATH=. pytest -q --ignore=tests/test_temporal_resampling.py
python -m compileall -q scripts tests
git diff --check
```

The legacy video temporal-resampling pilot remains under `src/` and its
original top-level scripts. It is independent of the 3D reconstruction study.
