# Experiment plan: temporal representation transfer for modern Video-VLMs

## Research question and hypotheses

The diagnostic question is whether a dedicated video foundation model (VFM)
contains arrow-of-time information that Qwen3-VL fails to preserve or fails to
use. The layer-wise outcomes distinguish three hypotheses rather than defining
an arbitrary pass/fail threshold:

1. **Representation gap:** V-JEPA 2 or VideoMAE V2 probe accuracy is stronger
   than every Qwen3-VL visual representation.
2. **Interface loss:** Qwen3-VL vision-block probes retain the signal, but it
   falls at the visual merger or after DeepStack injection into the language
   model.
3. **Utilization gap:** Qwen3-VL's internal probes retain the signal, but the
   unmodified model performs poorly on temporal language tasks.

These outcomes are not mutually exclusive. The first study is diagnostic and
does not claim that one pre-declared score makes the project successful.

## Experiment 1: frozen Arrow-of-Time probes

Use the released Something-Something V2 train split to train independent
one-query attentive probes on frozen representations from:

- Qwen3-VL vision blocks 8, 16, 24, and 26;
- the Qwen3-VL visual merger;
- Qwen3-VL visual-token states after DeepStack stages 0, 1, and 2;
- final Qwen3-VL language-model visual-token states;
- V-JEPA 2 last hidden-state tokens; and
- VideoMAE V2 final encoder tokens.

The SSv2 direction assignment, 16-frame sampling, spatial transforms, one-epoch
probe training, and optimizer are fixed in
`configs/protocols/pilot.yaml`. Every frozen representation receives its own
probe; no probe weights are reused between models or layers. The backbone pass
and probe training happen online, so full-dataset features are not stored.

Evaluate every probe on the released 212 AoT-PsyPhyBENCH source clips paired
forward and backward (424 examples). Report accuracy as the primary metric and
Forward F1, Backward F1, and the 2-by-2 confusion matrix as diagnostics.

## Experiment 2: unmodified temporal language evaluation

Evaluate the same pinned Qwen3-VL checkpoint without an adapter or fine-tuning
using the pinned lmms-eval implementations of VITATECS-Direction, TVBench, and
MVBench. Task-defined generation and metrics remain unchanged. MVBench is a
general control; full VITATECS and Video-MME are deferred until after the pilot.

## Experiment 3: intervention selected from the diagnosis

Only after Experiments 1 and 2 are complete, select one published intervention
whose mechanism matches the observed bottleneck:

- a representation-transfer intervention is relevant to a representation gap;
- a published visual-language connector or fusion intervention is relevant to
  interface loss; or
- a published temporal supervision/utilization intervention is relevant to a
  utilization gap.

The intervention's official data split, evaluator, optimization, and released
code must be recorded in a new protocol revision before it is run. No new loss,
dataset curation rule, metric, or decision threshold is introduced during the
diagnostic pilot.

## Stopping rule

If the frozen probes and unmodified language evaluation show no meaningful
localizable deficit, do not force a transfer method. Report the diagnostic
result and reconsider the research question. A new method is attempted only if
a residual limitation remains after the closest published intervention is
reproduced.
