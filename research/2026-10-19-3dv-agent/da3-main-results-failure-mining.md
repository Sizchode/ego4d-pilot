# DA3 Main Experiment and Failure Mining

2026-10-20｜103/103 configurations completed｜visibility-controlled findings before cross-model comparison

## What was run

DA3-GIANT was evaluated on the six pinned configurations in the experiment plan: HiRoom, ETH3D, ScanNet++, 7Scenes, DTU-49 and DTU-64. The Slurm main job `7649571` completed all 103 entries with exit code 0. The dependent failure-mining job `7649573` also completed with exit code 0.

The authoritative outputs remain in scratch:

- baseline archive: `/oscar/scratch/zliu328/3dv-reconstruction/runs/batch-7649571`;
- failure-mining tables: `/oscar/scratch/zliu328/3dv-reconstruction/analysis/da3-batch-7649571`;
- manifest: 103 records, 0 execution failures, 10,358 directed view-pair rows and 180 surface-distance rows;
- batch manifest SHA-256: recorded in the local run manifest;
- batch results SHA-256: recorded in the local run manifest.

A separate, threshold-free visibility pass completed as Slurm job `7670744`:

- output: `/oscar/scratch/zliu328/3dv-reconstruction/analysis/da3-visibility-batch-7649571`;
- 103 scene rows, 10,358 directed view-pair rows and 0 failures;
- each projected layer records rasterizability, the nearest source sample under a source-view z-buffer, whether the target depth lies in front, and whether the source depth is not behind;
- no tolerance, failure threshold or automatic failure label is used, and these layers are diagnostics rather than visibility ground truth.

An independent local audit in `artifacts/da3-analysis-audit/analysis-audit.json` re-read both analyses rather than trusting their Slurm exit codes. It verified the identical ordered population of 103 scenes, all 10,358 directed camera pairs, empty failure lists, complete within-dataset ranking groups, matching source batch/result hashes, and exact SHA-256 for every archived table. The audit assigns no quality threshold or failure label.

The table below reports medians within each benchmark. Reconstruction `overall` is lower-is-better; pose AUC and F-score are higher-is-better. Values across datasets are not directly comparable because the official evaluators use different scene scales and protocols.

| Dataset | Scenes | Pose AUC30 | Recon overall | F-score | Fusion retained ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| HiRoom | 30 | 0.9899 | 0.0190 | 0.9639 | 0.9707 |
| ETH3D | 11 | 0.9106 | 0.1800 | 0.7678 | 0.1497 |
| ScanNet++ | 20 | 0.9863 | 0.0600 | 0.7869 | 0.9156 |
| 7Scenes | 7 | 0.8824 | 0.0658 | 0.4746 | 0.8574 |
| DTU-49 | 22 | — | 1.5882 | — | 0.6517 |
| DTU-64 | 13 | 0.9960 | — | — | — |

## The first important result: the current residual has both false positives and false negatives

Consider two HiRoom scenes.

**False positive: `828812/cam_sampled_10`.** DA3 is strong under the official evaluation: pose AUC30 is 0.9970, reconstruction overall is 0.0376 and F-score is 0.9343. Nevertheless, its aggregated cross-view absolute-relative-depth p95 is 1.523 and round-trip p95 is 241.5 pixels, the worst values in HiRoom.

The stored pixel maps explain the contradiction. A source point is currently called valid when it has positive depth and projects inside the target image. The mask does not test whether that point is visible in the target view. Ceiling, wall and foreground-object points therefore get compared with the depth of an occluding target surface. The error is strongly directional: pair 10→9 has absolute-relative-depth p95 6.01 and round-trip p95 897.5 pixels, while 9→10 has 0.56 and 67.6 pixels. This is a geometric visibility problem in the diagnostic, not evidence that reconstruction needs repair.

The completed visibility pass confirms this explanation. Across all projectable, source-z-buffer-selected samples, the scene-level p95 is 1.989. Restricting the descriptive layer to samples whose source depth is not behind the target prediction reduces p95 to 0.0215. This layer retains 21.7% of all source pixels, compared with 43.0% selected as source z-buffer winners. The original large residual therefore comes from depth ordering and occlusion rather than a globally inaccurate reconstruction. This test does not assert ground-truth visibility; it shows that the false alarm disappears under an explicit, deterministic occlusion check.

![Directional false positive](../../artifacts/da3-batch-7649571-cases/hiroom-828812-directional-diagnostics.png)

**False negative: `828824/cam_sampled_08`.** This is the worst HiRoom pose case: pose AUC30 is 0.7064 and reconstruction overall is 0.2176. Yet its absolute-relative-depth p95 is only 0.0369 and round-trip p95 is 5.01 pixels. The model's prediction is internally consistent enough to pass the current self-consistency check while being wrong relative to ground truth.

Most of its 13 frames observe a low-texture ceiling, white wall, curtains and repeated linear lights. The camera-error matrix identifies particular frames, but the self-consistency residual does not. This is a credible upstream diagnosis problem: the system must recognize that the available views do not constrain pose well, then decide whether to exclude frames, change the reference set or rerun pose estimation.

Visibility filtering does not expose this failure. The source-depth-not-behind p95 remains only 0.0312, close to the original projectable-only p95 of 0.0369, even though pose AUC30 is 0.7064. The failure is therefore not hidden by the same occlusion mechanism as `828812`; the prediction remains internally consistent after the deterministic visibility check.

![Self-consistent pose failure inputs](../../artifacts/da3-batch-7649571-cases/hiroom-828824-all-inputs.jpg)

Across all 103 scenes, pair-level residuals are not a universal pose-failure detector. The median within-scene Spearman correlation between rotation error and absolute-relative-depth residual is only +0.067; for translation-direction error it is -0.005. Low projectability tracks rotation error more consistently (median correlation -0.364), but its relation to translation error reverses on several datasets. These values establish the limitation of the current state representation; they do not establish that a VLM solves it.

## Concrete failure families observed so far

| Family | Cases and measured evidence | Present interpretation | VLM status |
| --- | --- | --- | --- |
| Visibility-induced diagnostic false alarm | HiRoom `828812`: excellent official metrics but residual p95 1.523 / 241.5 px; large direction asymmetry and errors on occluded surfaces | The diagnostic confuses projectability with visibility | Do not assign to VLM; add a visibility-aware geometric check before agent work |
| Self-consistent camera failure under weak visual evidence | HiRoom `828824`: pose AUC30 0.706 with residual p95 0.0369 / 5.01 px; inputs dominated by ceiling/wall/repeated lights | Absolute pose can be wrong while predicted cameras and depths agree with one another | Best current VLM candidate, but must beat blur, feature-coverage and parallax baselines |
| Pose-correct but incomplete geometry | ETH3D `electro`: pose AUC30 0.968, prediction→GT mean 0.144 m, but GT→prediction mean 4.664 m and p90 18.43 m; fusion retention 0.201 | A large region of reference geometry is missing even though cameras are good | Start with depth confidence, coverage and fusion analysis; no VLM claim yet |
| Sparse/thin-structure reconstruction | ETH3D `playground`: pose AUC30 0.861, fusion retention 0.074, F-score 0.724; inputs contain thin wooden structures, occlusion and strong glare | Depth/fusion retains very little geometry and produces a small number of distant outliers | Cross-model comparison is required before choosing a semantic masking or fusion tool |
| Trajectory-segment failure with blur or repetitive structure | 7Scenes `pumpkin`: pose AUC30 0.782, F-score 0.132; `stairs`: F-score 0.195; camera errors have block structure and inspected frames show blur/repetitive rails or cabinets | Different trajectory portions are mutually inconsistent; raw adjacent-pair residuals can remain small | VLM is plausible only if it improves over image-quality, retrieval and geometric connectivity baselines |
| Geometry error with strong pose and high fusion retention | ScanNet++ `578511c8a9`: pose AUC30 0.983, fusion retention 0.965, but F-score 0.681 and overall 0.130 | The error is unlikely to be fixed by dropping low-confidence fusion points alone | Requires per-region depth analysis and another reconstruction baseline before assigning a cause |

The qualitative surface plots retain the evaluator's prediction→reference and reference→prediction color values. Each scene has its own visualization range, so colors may be compared within a scene but not across scenes.

![Bidirectional surface errors](../../artifacts/da3-batch-7649571-cases/bidirectional-surface-errors.png)

## Existing scalar state evidence is not a portable failure detector

Before introducing image or language reasoning, four directly available state fields were compared with every official within-dataset failure ranking. No fields were combined and no threshold or failure label was fitted. The table gives the range of Spearman correlations across benchmark configurations where both quantities exist; failure rank 1 is worst, so a sign reversal means the same evidence value changes its relationship with reconstruction quality.

| Observable state field | Pose AUC@30 rank correlation range | F-score rank correlation range |
|---|---:|---:|
| Fusion retained ratio | -0.071 to +0.248 | -0.290 to +0.509 |
| Projectable pixel ratio | -0.341 to +0.544 | -0.643 to +0.102 |
| Cross-view absolute-relative-depth p95 | -0.700 to +0.643 | -0.327 to +0.286 |
| Round-trip pixel residual p95 | -0.555 to +0.607 | -0.441 to +0.893 |

The sign changes remain when the seven-scene configuration is excluded. For example, absolute-relative-depth p95 versus pose failure rank is -0.700 on ETH3D, +0.046 on HiRoom, +0.159 on DTU-64 and +0.302 on ScanNet++; fusion retention versus F-score rank is +0.509 on ETH3D, +0.144 on HiRoom and -0.290 on ScanNet++. A single scalar threshold is therefore not a credible cross-benchmark diagnostic.

This result narrows the claim but does not establish a VLM advantage. A multivariate deterministic selector, dataset-specific calibration and cross-backbone consistency are still required baselines. The machine-readable table contains 276 within-dataset correlations over all available official metrics: `artifacts/da3-state-evidence-batch-7649571/state_failure_rank_correlations.csv`, with its population and input hashes in the adjacent `manifest.json`.

## Global RGB statistics are also insufficient as a portable diagnostic

The independent image-evidence pass archived 5,282 input images from the same ordered 103-scene population, with zero execution failures and the same batch manifest/results hashes as the official analysis. It records luminance distributions, histogram entropy, gradient distributions and floor/ceiling saturation ratios; it uses neither reconstruction predictions nor ground truth.

On the 30 HiRoom scenes, the strongest absolute correlation among 95 finite image fields is only 0.353 for pose AUC@30 failure rank and 0.363 for F-score failure rank. Predefined interpretable summaries are weaker: mean gradient is +0.152 / +0.049 for pose/F-score rank, mean entropy is +0.066 / +0.290, and mean full-channel ceiling saturation is +0.076 / -0.130. Failure rank 1 remains the worst, so these signs are descriptive rather than a universal “more is worse” direction.

The same fields do not transfer consistently across benchmarks. Mean gradient versus F-score rank changes from +0.500 on 7Scenes to -0.436 on ETH3D, +0.049 on HiRoom and -0.074 on ScanNet++; mean entropy changes from +0.179 to -0.545, +0.290 and -0.005 on those datasets. Small populations such as 7Scenes (7 scenes) and ETH3D (11 scenes), together with the exhaustive field sweep, make isolated large correlations unsuitable for selecting a rule.

The current 30-scene four-backbone oracle gives the same caution. Across 73 finite non-count image fields, median absolute correlation with oracle gain is 0.103 for pose AUC@30 and 0.159 for F-score. The respective maxima are 0.367 and 0.565, but both come from sparse floor-saturation summaries after inspecting all fields; they have not been validated on a held-out population. Mean gradient correlates -0.053 with pose oracle gain and +0.326 with F-score oracle gain, while mean entropy gives -0.122 and -0.149.

A Benjamini–Hochberg correction was applied separately within each HiRoom endpoint over the 73 non-count image fields. No field reaches `q < 0.05`. For direct DA3 failure rank, the minimum q-values are 0.990 for pose AUC@30 and 0.984 for F-score. For oracle gain, the minimum is 0.971 for pose; the strongest uncorrected geometry association is the any-channel floor-saturation p50 field (`rho = 0.565`, raw `p = 0.00115`) but becomes `q = 0.0836`. The fields are correlated and the sample contains only 30 scenes, so this correction is a guard against field selection rather than a definitive independence model.

The audited cases show why a single threshold cannot select the right action. In ascending within-HiRoom percentiles, the self-consistent pose failure `828824` is low but not extreme in mean gradient (0.172) and mean entropy (0.207). Scene `828790` is lower in both (0.069 / 0.069) while all four backbones have pose AUC@30 at or above 0.905 and instead share a dense-geometry deficit. DA3's model-specific collapse on `828815` has high gradient and entropy percentiles (0.759 / 0.793). The image statistics can describe capture conditions, but they do not distinguish “repair pose,” “repair geometry,” “select another backbone,” and “do nothing.”

Therefore a fixed global blur, texture, brightness or saturation scalar is not yet an adequate intervention gate. This still does not prove a VLM advantage: local image regions, camera motion, overlap, feature tracks and multivariate non-language models remain required comparisons. Machine-readable evidence is in `artifacts/da3-image-evidence-batch-7649571`, `artifacts/da3-image-vs-failure-ranks`, `artifacts/da3-image-vs-oracle-hiroom`, `artifacts/da3-image-fdr-failure-hiroom` and `artifacts/da3-image-fdr-oracle-hiroom`.

## What this means for the paper

The first DA3 run supports a narrower problem than “a VLM sees an error and calls a tool.” The useful task is to distinguish three states that simple reconstruction residuals conflate:

1. a large residual caused by visibility or evaluation geometry, where no reconstruction repair is needed;
2. an internally consistent but externally wrong camera/depth solution, where the present numeric state misses the failure;
3. a correctly posed reconstruction with local or global geometry loss, where pose repair is the wrong action.

This is upstream reconstruction diagnosis, which separates the project from GRASE's editable-scene refinement. It is not yet evidence for a VLM contribution. A VLM becomes justified only if image-and-geometry reasoning distinguishes these states better than visibility-aware residuals, blur scores, feature coverage, parallax, confidence and fixed decision rules.

## Next evidence gate

Do not design the toolkit from these evaluation examples. First:

1. run VGGT, π³-X and MapAnything on the same frozen scene manifests;
2. determine which failures recur across models and which are DA3-specific;
3. reproduce the case analysis on a disjoint ScanNet++ training development split before tuning any repair action.

Only recurring failures that survive these checks should enter the toolkit. The current strongest candidate is self-consistent pose failure under visually weak or ambiguous observations; visibility errors and simple blur should remain geometric/image-processing baselines rather than being claimed as VLM reasoning.
