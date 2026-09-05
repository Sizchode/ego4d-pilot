# Video Temporal Resampling / Latent Stability

This repository implements the first-stage pilot in the current plan. The
minimal closed loop is:

```text
SSv2 + DINOv3       image-foundation control
SSv2 + V-JEPA 2     native predictive-video reference
NExT-QA + Qwen3-VL  target Video-MLLM behavior
C0 canonical vs C2 temporal offset first
```

The old AoT, VideoMAE, InternVL/MiniCPM, and broad benchmark experiments were
removed. Model loaders now live under [`src/models`](src/models); data,
timestamp schedules, probes, and experiment scripts live under
[`src/temporal_resampling`](src/temporal_resampling) and [`scripts`](scripts).

Local development does not require checkpoints, videos, GPU, or PyAV. It checks
the protocol, manifest schema, and schedule math only:

```bash
python -m unittest discover -s tests -v
python -m compileall -q src scripts tests
```

## HPC preparation

All model and dataset revisions must be immutable. The scripts reject missing
revisions and do not invent canonical schedules. First export the official
SSv2 label map and canonical timestamp JSON, then build a manifest:

```bash
python -m scripts.prepare_data manifest ssv2 \
  --annotations /path/to/official/ssv2_train.json \
  --video-root /path/to/ssv2/videos \
  --split train \
  --label-map /path/to/official/ssv2-label-map.json \
  --annotation-revision official-source-revision \
  --output /scratch/temporal-resampling/ssv2-train.jsonl
```

The canonical schedule file is a JSON/YAML mapping with immutable provenance:
`source`, `revision`, `dataset`, `model`, and `schedules` (`sample_id` to the
official C0 timestamp list). C1 additionally requires an explicit
`--lower-rate-ratio`; the default audit/render commands run C0 and C2 first.

Audit C0/C1/C2/C3 with the exact official C0 timestamps:

```bash
python -m scripts.audit_resampling \
  --manifest /scratch/temporal-resampling/ssv2-val.jsonl \
  --canonical-schedules /scratch/temporal-resampling/ssv2-c0-timestamps.json \
  --offset-seconds 0.10 \
  --output /scratch/temporal-resampling/ssv2-schedule-audit.json
```

Render the first 20 samples and inspect the contact sheets before scaling up:

```bash
python -m scripts.render_schedule_samples \
  --manifest /scratch/temporal-resampling/ssv2-val.jsonl \
  --canonical-schedules /scratch/temporal-resampling/ssv2-c0-timestamps.json \
  --offset-seconds 0.10 \
  --output-dir /scratch/temporal-resampling/sample-sheets
```

The C0-trained cross-sampling probe is run by
[`scripts/train_cross_sampling_probe.py`](scripts/train_cross_sampling_probe.py).
It reports only `Top1(C0..C3)` and `Delta(C1..C3)`. Oracle probes are not the
primary result.

See [`ENVIRONMENT.md`](ENVIRONMENT.md) and [`OPEN_QUESTIONS.md`](OPEN_QUESTIONS.md)
for HPC gates. No optional benchmark is selected by this scaffold.
