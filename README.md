# Temporal representation transfer for modern Video-VLMs

This repository diagnoses whether dedicated video foundation models expose
arrow-of-time information that `Qwen/Qwen3-VL-8B-Instruct` does not preserve
or use. It implements the two studies fixed in the experiment plan:

1. train the published attentive Arrow-of-Time probe on frozen Qwen3-VL,
   V-JEPA 2, and VideoMAE V2 representations using SSv2 train, then evaluate
   on AoT-PsyPhyBENCH; and
2. evaluate unmodified Qwen3-VL on VITATECS-Direction, TVBench, and MVBench
   through a pinned `lmms-eval` checkout.

The code intentionally contains no temporal adapter, projector, distillation
loss, new dataset split, or custom benchmark metric. An intervention is chosen
only after the diagnostic results are available.

The previous Ego4D OSCC/PNR pilot is preserved by the Git tag
`ego4d-pilot-v1`.

The staged hypotheses, diagnostic interpretations, and stopping rule are in
[`docs/experiment_plan.md`](docs/experiment_plan.md).

## Reproducibility contract

[`configs/protocols/pilot.yaml`](configs/protocols/pilot.yaml) is the
authoritative protocol. It pins model and dataset revisions, upstream code,
sampling, preprocessing, extraction points, probe training, and reported
metrics. Every command validates that manifest before touching a model or
dataset. Runtime paths are supplied separately so the scientific protocol
does not depend on a cluster filesystem layout.

The AoT probe is a **paper-spec implementation**, not an official TAT probe
code reproduction: the public TAT-VQA repository does not release the probe
implementation. Its disclosed architecture and hyperparameters are followed;
the manifest explicitly identifies implementation choices sourced from the
official V-JEPA 2 frozen-probe code.

## Installation

Python 3.10 or newer and a CUDA PyTorch build are required for experiments.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[train,eval,test]'
bash scripts/bootstrap_upstreams.sh
python scripts/validate_protocol.py configs/protocols/pilot.yaml
```

Model snapshots are placed under scratch at their pinned Hugging Face
revisions:

```bash
python scripts/download_model_snapshots.py \
  --protocol configs/protocols/pilot.yaml \
  --output-root /oscar/scratch/$USER/temporal-transfer/models
```

## Data preparation

No videos are redistributed.

- Obtain SSv2 from its official provider, preserving the official train split.
  Extract the videos and labels, then build a deterministic AoT manifest:

  ```bash
  python scripts/build_ssv2_aot_manifest.py \
    --protocol configs/protocols/pilot.yaml \
    --annotations /path/to/labels/train.json \
    --video-root /path/to/20bn-something-something-v2 \
    --output /oscar/scratch/$USER/temporal-transfer/manifests/ssv2_train_aot.jsonl
  ```

- Obtain the 212 AoT-PsyPhyBENCH source clips using the official metadata and
  Multi-Moments in Time instructions. The evaluator represents each source
  clip once forward and once virtually reversed, producing the published 424
  examples without re-encoding video:

  ```bash
  python scripts/build_aot_psyphybench_manifest.py \
    --protocol configs/protocols/pilot.yaml \
    --metadata third_party/AoT-PsyPhyBENCH/neuro_paper_data/neuro_paper_data/AoT-PsyPhyBENCH_dataset_metadata.csv \
    --video-root /path/to/aot/source/videos \
    --output /oscar/scratch/$USER/temporal-transfer/manifests/aot_psyphybench.jsonl
  ```

Both builders fail on missing or duplicate samples and never reassign the
official split.

## Experiment 1: frozen attentive probes

Run one provider per distributed job. Qwen3 trains all architecture-fixed
representation probes in one frozen-backbone pass; no full-dataset features
are materialized.

```bash
torchrun --standalone --nproc-per-node=6 scripts/train_aot_probe.py \
  --protocol configs/protocols/pilot.yaml \
  --provider qwen3 \
  --train-manifest /path/to/ssv2_train_aot.jsonl \
  --eval-manifest /path/to/aot_psyphybench.jsonl \
  --model-root /oscar/scratch/$USER/temporal-transfer/models \
  --output-dir /oscar/scratch/$USER/temporal-transfer/runs/qwen3
```

Use `--provider vjepa2` and `--provider videomaev2` for the frozen VFM
comparisons. The output contains the resolved protocol, per-class counts,
accuracy, Forward/Backward F1, and small probe checkpoints only.

## Experiment 2: official language-level evaluation

The wrapper invokes the pinned `lmms-eval` checkout and its published task
definitions. The Qwen model argument is a local snapshot resolved at the
pinned Hugging Face revision.

```bash
bash scripts/run_language_evaluation.sh \
  configs/protocols/pilot.yaml \
  /oscar/scratch/$USER/temporal-transfer/models/qwen3 \
  /oscar/scratch/$USER/temporal-transfer/lmms-results
```

The first pilot tasks are exactly `vitatecs_direction`, `tvbench`, and
`mvbench`. Full VITATECS and Video-MME are outside the first pilot.

## Validation

CPU-only tests validate the protocol, manifest construction, virtual temporal
reversal, representation names, and metrics without downloading weights:

```bash
python -m unittest discover -s tests -v
```
