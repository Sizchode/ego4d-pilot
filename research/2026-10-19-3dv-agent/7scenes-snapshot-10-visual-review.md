# Snapshot 10: 7Scenes visual and metric review

**Status.** This is a non-final review of the complete seven-scene 7Scenes population available in snapshot 10. It compares four frozen baseline outputs—DA3, VGGT, π³-X, and MapAnything—but does not assign failure labels or define an agent action. The purpose is to turn the first complete benchmark population into testable intervention hypotheses while the remaining main runs continue.

## Evidence and audit

The review archive contains 28 cases (7 scenes × 4 models), 224 files, and 100 evaluated views per case. Every archived file was checked against its recorded byte count and SHA-256. The downloaded tar matches the remote SHA-256 recorded in the local run manifest. Seven model-aligned contact sheets were regenerated from the verified archives and then inspected manually.

The descriptive tables contain 28 case rows and 2,800 per-view rows. All cases are completed, validation-passed runs with 100 views and 4,950 unordered camera pairs. Their 280 official pose and reconstruction metric values are exactly equal to the immutable snapshot-10 cross-model table (maximum absolute difference `0.0`). The machine-readable audit is in `artifacts/partial-snapshot-10-review/local-audit.json`; it passed without errors.

The contact sheets show six sampled input/depth/confidence panels and the full pairwise camera-error matrices. The camera errors are computed against benchmark ground truth and are **offline failure-analysis evidence, not test-time policy inputs**. Depth and confidence panels use each view's native range, and each camera-error subplot chooses its own upper color limit. Color intensity therefore cannot be compared across models; the numerical tables are used for magnitude comparisons.

## Main result: the hard views recur, but the best pose does not determine the best geometry

The GT-based view-level error ordering is often similar across backbones. For each scene and camera quantity, the table below reports the median Spearman correlation across the six model pairs, using each view's incident-pair P95 error. This statistic measures whether the models tend to fail on the same views; it does not show that their errors have the same magnitude or cause, and it is unavailable to a deployable controller.

| Scene | Rotation median ρ | Translation median ρ | Interpretation limited to this evidence |
|---|---:|---:|---|
| `chess` | 0.800 | 0.820 | Strong shared ordering of difficult views |
| `fire` | 0.461 | 0.612 | Moderate shared ordering |
| `heads` | not interpretable¹ | 0.407 | Rotation errors are constant zero for several models |
| `office` | 0.927 | 0.916 | Very strong shared ordering |
| `pumpkin` | 0.968 | 0.929 | Very strong shared ordering |
| `redkitchen` | 0.624 | 0.659 | Moderate shared ordering |
| `stairs` | 0.773 | 0.341 | Shared rotation pattern, weaker translation pattern |

¹ Only one of six rotation correlations is finite because several model series are constant; that single coefficient is not summarized.

The official endpoints nevertheless disagree about which backbone is best:

| Scene | Best pose AUC@3° | Best geometry F-score | Best F-score | What the discrepancy rules out |
|---|---|---|---:|---|
| `chess` | DA3 | π³-X | 0.4790 | Choosing a backbone from pose alone |
| `fire` | VGGT | DA3 | 0.8049 | Treating a 0.0011 pose-AUC lead as meaningful geometry evidence |
| `heads` | DA3 | π³-X | 0.8761 | Choosing a backbone from pose alone |
| `office` | MapAnything | MapAnything | 0.6451 | Nothing yet; this is a valid routing candidate |
| `pumpkin` | VGGT | VGGT | 0.1833 | Assuming routing is sufficient when every output is poor |
| `redkitchen` | π³-X | DA3 | 0.6238 | Choosing a backbone from pose alone |
| `stairs` | π³-X | DA3 | 0.1952 | Choosing a backbone from pose alone |

The implication is narrower than “an agent is needed.” A camera-quality gate is insufficient because the final geometry also depends on depth, confidence, visibility, and fusion. A useful controller would have to predict downstream metric benefit among actions such as changing camera state, changing depth/fusion state, routing to another backbone, or doing nothing. This review does not yet show that a VLM can make that decision better than numerical diagnostics.

## Scene-by-scene observations and experiments they motivate

### `chess` and `heads`: pose winner versus geometry winner

DA3 has the best pose AUC@3° in both scenes, while π³-X has the best F-score: `0.4790` versus DA3's `0.3960` on `chess`, and `0.8761` versus `0.8352` on `heads`. The sampled π³-X depth panels show crisp object boundaries, but native depth visualizations are not metrically comparable and cannot establish the cause of the F-score gain.

The appropriate next experiment is a controlled component intervention: keep cameras fixed while exchanging or recalibrating depth/confidence and rerunning the identical fusion/evaluator, followed by the complementary camera-only intervention. If the geometry gain follows depth/fusion rather than pose, these cases provide direct evidence for factor-specific recovery rather than backbone routing.

### `redkitchen` and `stairs`: better pose, worse geometry

π³-X has the best pose AUC@3° on both scenes, but DA3 has the best F-score. On `redkitchen`, the gap is large: DA3 `0.6238` versus π³-X `0.3257`. On `stairs`, all F-scores are low and DA3's best value is only `0.1952`. The contact sheets expose repeated thin rails, step edges, self-occlusion, and narrow visible surfaces in `stairs`; these are plausible depth/visibility/fusion stressors, but the images do not isolate which factor failed.

These scenes are negative controls for any `repair the worst camera` rule. They require camera-only and depth/fusion-only interventions before a repair target can be assigned.

### `office`: routing headroom is real, cause remains unresolved

MapAnything is best on both pose AUC@3° (`0.1477`) and F-score (`0.6451`), improving F-score by `0.1705` over DA3. All four models show large block-structured camera errors, and the per-view error ordering is highly correlated across models. This supports a scene-level routing opportunity but does not show whether MapAnything wins through camera, depth, confidence, or fusion. It should be retained as a positive routing case and decomposed with component interventions.

### `pumpkin`: shared failure, limited routing value

VGGT is best on both endpoints, yet its F-score is only `0.1833`; the other models range from `0.0969` to `0.1323`. The sampled views include weakly textured or reflective cabinetry, repeated revisits, and transitions across parts of the room. The very high cross-model view-error correlations indicate common difficult portions of the sequence. These observations make `pumpkin` a useful shared-hard-case candidate, not evidence that a backbone-selection action solves it.

### `fire`: endpoint difference without useful pose signal

VGGT's pose AUC@3° exceeds DA3 by only `0.0011`, while DA3's F-score is `0.8049` versus VGGT's `0.6551`. The pose difference is too small to support a routing explanation. This is another control showing that pose ranking alone is not an adequate action target.

## What is established and what remains open

This population establishes three facts under the frozen evaluator:

1. pose and final geometry select different backbones in four of seven scenes;
2. several backbones identify similar difficult portions of a sequence even when their endpoint quality differs; and
3. `pumpkin` and `stairs` remain poor under every backbone, so backbone routing alone cannot define the recovery task.

It does not establish a failure cause, a deployable diagnostic, or a VLM advantage. The next causal evidence must come from controlled camera and depth/fusion interventions. The later controller comparison must hold the toolkit and execution budget fixed and compare numerical rules, search/oracle, one-shot VLM, and sequential VLM decisions on the same actions.

## Artifacts

- verified archive: `artifacts/partial-snapshot-10-review/7scenes-all/`
- model-aligned contact sheets: `artifacts/partial-snapshot-10-review/7scenes-contact-sheets/`
- case metrics: `artifacts/partial-snapshot-10-review/7scenes-case-metrics.csv`
- per-view metrics: `artifacts/partial-snapshot-10-review/7scenes-view-metrics.csv`
- reproducible cross-model view-profile analysis: `artifacts/partial-snapshot-10-review/7scenes-view-profile-agreement/`
- local audit: `artifacts/partial-snapshot-10-review/local-audit.json`
