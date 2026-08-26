# Temporal representation transfer

Research code for diagnosing temporal representation failures in Qwen3-VL.
The current repository implements the two diagnostic experiments in the
Google Doc. Experiments 3 and 4 are intentionally not scaffolded: the plan
requires choosing Experiment 3 from the Experiment 1–2 diagnosis, and starting
Experiment 4 only if a stable residual failure remains.

## Repository layout

```text
experiments/
  exp1_aot/              # config, data loader, backbones, probe, and training
  exp2_vlm_eval/         # config and official lmms-eval entry point
launchers/               # Slurm launchers only
requirements.txt
```

Each experiment owns its configuration. There is no package CLI, global
protocol layer, report directory, vendored upstream repository, or test suite.

## Installation

Create an environment with a CUDA build of PyTorch, then install the pinned
research dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Transformers and lmms-eval are installed directly from the commits used by the
experiments. Qwen3-VL and VideoMAE V2 are loaded from pinned Hugging Face
revisions. V-JEPA2.1 is loaded through Meta's official PyTorch Hub entry at a
pinned commit because Meta has not released V-JEPA2.1 through its Hugging Face
collection; its checkpoint comes from Meta's official download URL.

There is no separate model-download step. Each experiment resolves the exact
release on first use and reuses the shared Hugging Face/PyTorch cache under
`/oscar/scratch/$USER/temporal-transfer/models`. Set `MODEL_CACHE_ROOT` before
`sbatch` to use another cache location.

## Experiment 1: Arrow-of-Time probing

The experiment-local settings and source revisions are in
[`experiments/exp1_aot/config.yaml`](experiments/exp1_aot/config.yaml).
It uses the released SSv2 train split and AoT-PsyPhyBENCH evaluation set; it
does not create a new split or metric.

Build the two manifests after obtaining the official videos and annotations:

```bash
python experiments/exp1_aot/prepare_data.py ssv2 \
  --annotations /path/to/something-something-v2-train.json \
  --video-root /path/to/20bn-something-something-v2 \
  --output /scratch/path/ssv2_train_aot.jsonl

python experiments/exp1_aot/prepare_data.py aot_psyphybench \
  --annotations /path/to/AoT-PsyPhyBENCH_dataset_metadata.csv \
  --video-root /path/to/aot/videos \
  --output /scratch/path/aot_psyphybench.jsonl
```

Submit one six-GPU job per backbone:

```bash
sbatch launchers/exp1_aot.slurm \
  vjepa2_1 \
  /scratch/path/ssv2_train_aot.jsonl \
  /scratch/path/aot_psyphybench.jsonl \
  /oscar/scratch/$USER/temporal-transfer/runs/vjepa2_1
```

The other model names are `qwen3` and `videomaev2`.

## Experiment 2: language-level evaluation

This calls the pinned official lmms-eval implementation for
`vitatecs_direction`, `tvbench`, and `mvbench` using the dataset splits and
metrics recorded in
[`experiments/exp2_vlm_eval/config.yaml`](experiments/exp2_vlm_eval/config.yaml).

```bash
sbatch launchers/exp2_vlm_eval.slurm \
  /oscar/scratch/$USER/temporal-transfer/runs/qwen3_vlm_eval
```
