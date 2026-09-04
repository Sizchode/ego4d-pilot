# Video temporal state experiments

This repository now contains a scaffold for the final plan in
`video_temporal_state_experiment_plan_cn_final.md`. The existing
`experiments/exp1_aot` and `experiments/exp2_vlm_eval` directories are kept as
historical experiments; the new protocol lives under
`experiments/temporal_state`.

The local development path is intentionally resource-free. It validates the
protocol and manifest files, while model weights, video media, feature shards,
and probe runs belong on HPC. Data and run products are ignored by git.

## Local checks

```bash
python -m unittest discover -s tests -v
python -m compileall -q experiments
```

## HPC data preparation

Pin a revision for every Hugging Face download when starting a real run:

```bash
python -m experiments.temporal_state.prepare_data download ssv2 \
  --output-dir /scratch/temporal-state/data/ssv2 --revision <exact-revision>

python -m experiments.temporal_state.prepare_data manifest ssv2 \
  --annotations /path/to/official/train.json \
  --video-root /path/to/ssv2/videos \
  --label-map /path/to/official/ssv2-label-map.json \
  --split train \
  --output /scratch/temporal-state/manifests/ssv2-train.jsonl
```

Feature extraction additionally requires a JSON file containing the exact
official prompt for every manifest `example_id`; the scaffold does not invent
an H1 or NExT-QA prompt template.

Use `download nextvideo` for NExT-QA media and `download vstar` for V-STaR
media. NExT-QA annotations come from the official NExT-QA repository and are
converted with the `nextqa` manifest command. EgoTempo is intentionally not
downloaded by this tool: obtain only its official clip IDs through the
Ego4D-authorized workflow.

## Feature extraction and probes

The public adapter seam is:

```text
extract_compact_state(video_paths, prompts)
extract_distributed_state(video_paths, prompts)
```

Each decoder layer is stored independently. There is no cross-layer
concatenation, common mean/max pooling, synthetic V-JEPA compact state, or
new metric. NExT-QA reports overall, temporal, causal, and descriptive
accuracy only.

The Qwen3-VL adapter and V-JEPA 2 protocol boundary are present. InternVL3.5
and MiniCPM-V 4.5 stop with explicit errors until their official local
processor/chat-template/hook paths are pinned. See
[`experiments/temporal_state/OPEN_QUESTIONS.md`](experiments/temporal_state/OPEN_QUESTIONS.md).

After those gates are resolved, submit:

```bash
sbatch launchers/temporal_state_probe.slurm \
  qwen3_vl \
  /scratch/temporal-state/manifests/ssv2-train.jsonl \
  /scratch/temporal-state/manifests/ssv2-validation.jsonl \
  ssv2 \
  /scratch/temporal-state/runs/qwen3_vl \
  /scratch/temporal-state/prompts/ssv2-official.json
```

EgoTempo and V-STaR must be run through their official evaluators. The
`run_native_eval.py` wrapper archives the exact command and raw stdout/stderr;
it does not reimplement their metrics.
