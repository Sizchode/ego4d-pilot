# Hypothesis and experiment framework

## Research question

Does latent-predictive video pretraining make fine-grained object-state-change
and state-change timing information more accessible to a frozen downstream
readout than supervised video pretraining or image-only pretraining?

This pilot evaluates representation transfer. It does not currently train a
student model, perform knowledge distillation, or claim to improve V-JEPA or a
video-language model.

## Hypothesis

Because V-JEPA is pretrained to predict masked spatiotemporal representations
from visible video context, while Omnivore is trained with supervised category
objectives and DINOv2 is pretrained on images, we hypothesize that frozen
V-JEPA features will expose temporally localized object-state-change
information more effectively. The most diagnostic endpoint is PNR temporal
localization; OSCC determines whether the same representation also exposes
the semantic presence of a state change.

## Dataset and released split

The experiment uses Ego4D v1 Forecasting Hands and Objects annotations:

- `fho_oscc-pnr_train.json`: 41,085 labeled clips from 456 videos.
- `fho_oscc-pnr_val.json`: 28,348 labeled clips from 350 videos.
- Train and validation together reference 717 unique source videos.
- The released train and validation assignments are used unchanged.
- The unannotated test split is not used for local model selection.

This is an existing Ego4D benchmark, not a newly curated dataset. EgoPack is a
published processing and model codebase used on Ego4D; it is not the source of
the data or labels.

## Tasks and primary endpoints

### Object State Change Classification (OSCC)

Given a short video clip, classify whether an object state change occurred.
The released evaluation metric is classification accuracy; higher is better.

### Point-of-No-Return localization (PNR)

Given a positive state-change clip, localize the keyframe at which the state
change begins or becomes irreversible. The released primary metric is mean
absolute temporal localization error in seconds; lower is better.

No replacement metric or custom split is introduced.

## Backbone comparison

| Backbone | Pretraining role | Experimental role |
|---|---|---|
| Omnivore Video Swin-L | Supervised image, video, and RGB-D training | Historical in-domain Ego4D reference |
| DINOv2 ViT-L/14 | Image-only self-supervised training | Static/appearance control |
| V-JEPA 2.1 ViT-L/16 | Latent-predictive video training | Target temporal representation |
| VideoMAE V2 or InternVideo2 | Modern video-foundation-model comparison | Required follow-up before a full paper claim |

Omnivore alone is insufficient to establish superiority over modern video
foundation models. It remains necessary because the official Ego4D feature
release and the published EgoPack pipeline use it.

## Controlled inputs

- Frozen backbone weights for every representation.
- Official Ego4D 32-frame feature windows with stride 16.
- Released Ego4D v1 train and validation splits.
- Published dataset conversion and task metrics.
- Independently initialized downstream readouts for each backbone and seed.
- The same downstream readout family, optimization search space, selection
  rule, and training budget across compared backbones.

Feature dimensionality may differ across backbones. Each readout therefore has
its own input projection, but no readout weights are transferred between
backbones.

## Evaluation tracks

### Track A: published task reference

Reproduce EgoPack's single-task Temporal Graph model with the released
Omnivore features on OSCC and PNR, using the published seeds 65, 72, and 97.
This track answers how the historical feature/task-model combination performs.
It is not described as a linear probe.

### Track B: controlled frozen readout

Train independent identical readouts over frozen Omnivore, DINOv2, and V-JEPA
features. The precise readout and optimizer must come from an applicable
published frozen-evaluation protocol and must be fixed before inspecting
comparative validation results. This track isolates linearly or attentively
accessible representation quality rather than maximum downstream capacity.

### Track C: small motion-centric pilot

If a lower-storage pilot is needed, use the official Diving48 v2 split and the
published V-JEPA attentive-probe evaluation pipeline. Diving48 contains about
18,000 trimmed clips and tests fine-grained long-range motion, but it is not
egocentric and does not replace Ego4D state-change evaluation.

## Reporting

- Report every seed, plus mean and standard deviation across the three fixed
  seeds.
- Report OSCC accuracy and PNR localization error without selecting a custom
  composite score.
- Preserve failed runs and rerun reasons in the experiment log.
- Distinguish final-epoch, best-validation, and test-server results; do not mix
  selection conventions across backbones.
- Report model size, feature dimensionality, input frames, temporal stride,
  trainable parameter count, and extraction/training compute.

## Interpretation

- Improvement over DINOv2 supports a benefit from video rather than image-only
  pretraining.
- Improvement over Omnivore supports transfer beyond the historical supervised
  video representation used by Ego4D/EgoPack.
- Improvement over a scale-matched VideoMAE V2 model would more directly
  support latent prediction over masked pixel reconstruction.
- A larger improvement on PNR than OSCC is consistent with stronger temporal
  localization, but it does not by itself demonstrate causal or physical
  understanding.
- A gain only against Omnivore is insufficient for a modern foundation-model
  claim.

## Route beyond the pilot

The pilot is a prerequisite test for a possible V-JEPA-to-video-language-model
transfer project. A full method paper would require:

1. Evidence that the V-JEPA teacher contains temporal information absent from
   the selected student and competing teachers.
2. A method grounded in the closest published temporal knowledge-transfer
   literature rather than an ad hoc feature-matching loss.
3. Comparisons against the unmodified student and alternative teachers under
   the same training data and budget.
4. Consistent evaluation across motion recognition, egocentric state-change
   localization, temporal video QA, and general semantic benchmarks.
5. Evidence that temporal improvements do not arise from additional labels or
   unacceptable degradation of the student's original capabilities.

Until those stages are completed, the contribution is a controlled pilot and
not "distilling V-JEPA to improve motion semantics."
