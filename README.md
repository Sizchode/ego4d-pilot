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

The Oscar launchers use `/oscar/scratch/$USER/temporal-transfer/env-cu128` by
default. Create it with Python 3.11 and install the research dependencies:

```bash
uv venv --python 3.11 /oscar/scratch/$USER/temporal-transfer/env-cu128
UV_CACHE_DIR=/oscar/scratch/$USER/temporal-transfer/uv-cache uv pip install \
  --python /oscar/scratch/$USER/temporal-transfer/env-cu128/bin/python \
  -r requirements.txt
```

Transformers and lmms-eval are installed directly from the commits used by the
experiments. Qwen3-VL and VideoMAE V2 are loaded from pinned Hugging Face
revisions. V-JEPA2.1 uses the pinned
`Dev-Jahn/vjepa2.1-vitl-fpc64-384` Hugging Face conversion. Before adoption,
its encoder output was checked against Meta's official V-JEPA2.1 ViT-L/384
checkpoint at Meta source revision `204698b45b3712590f06245fbfba32d3be539812`:
the 16-frame maximum absolute error was zero and the one-frame error was
`5.62e-4`, both below the conversion author's published `1e-3` threshold.
Because the conversion has no processor, Experiment 1 implements Meta's exact
released frozen-probe spatial transform rather than introducing a new one. The
adapter runs in fp32 because this conversion's SDPA path produces mixed Q/K/V
dtypes when the full model is cast to bf16.

There is no separate model-download step. Each experiment resolves the exact
release on first use and reuses the shared Hugging Face/PyTorch cache under
`/oscar/scratch/$USER/temporal-transfer/models`. Set `MODEL_CACHE_ROOT` before
`sbatch` to use another cache location.
Set `EXPERIMENT_ENV` to override the environment path.

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
