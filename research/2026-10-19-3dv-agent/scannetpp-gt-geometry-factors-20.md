# ScanNet++ 20-scene reconstruction-factor audit

**Question.** When a feed-forward reconstruction has poor final geometry, is the recoverable error mainly in cameras, intrinsics, or depth? This offline experiment replaces those quantities with ground truth in a fixed nested order for DA3, VGGT, π³-X, and MapAnything on the same 20 ScanNet++ scenes. It measures where oracle headroom exists; it is not a deployable recovery method and does not give ground truth to an agent.

## Intervention and audit

For every model–scene pair, the official native result is followed by three stages:

`native → GT extrinsics → +GT intrinsics → +GT depth`.

The resulting mesh is evaluated with the same ScanNet++ reconstruction evaluator at every stage. There are 80 model–scene cases and 480 case–metric records. The remote artifact audit verifies all 80 cases, stage order, input provenance, metric values, 80 result records, and 160 generated intervention meshes. A second local audit independently regenerates both summary tables, and a third independently regenerates all factor profiles; every regenerated CSV is byte-identical to the stored output.

The differences below are **ordered conditional gains**. For example, “depth” means the additional gain from GT depth after extrinsics and intrinsics are already GT. Because the interventions are nested, these numbers are not independent causal effects and would change under a different order.

![Conditional F-score gains](../../artifacts/scannetpp-gt-geometry-factors-20-final/fscore-conditional-gains.png)

## F-score result

| Backbone | GT extrinsics: mean / positive / unique best | +GT intrinsics: mean / positive / unique best | +GT depth: mean / positive / unique best | Full-GT mean gain |
|---|---:|---:|---:|---:|
| DA3 | +0.0146 / 17 / 0 | +0.0007 / 11 / 0 | **+0.0649 / 20 / 20** | +0.0801 |
| VGGT | +0.0286 / 13 / 2 | −0.0102 / 7 / 0 | **+0.1692 / 20 / 18** | +0.1876 |
| π³-X | **+0.0825 / 18 / 11** | +0.0044 / 9 / 0 | +0.0535 / 20 / 9 | +0.1405 |
| MapAnything | +0.1994 / 19 / 8 | +0.1442 / 18 / 4 | **+0.2252 / 20 / 8** | +0.5689 |

“Positive” counts scenes with gain above zero; “unique best” counts scenes where that factor has the largest positive conditional gain among the three. No quality or success threshold is used.

There is no universal camera-repair story. DA3 and VGGT are depth-dominant on this population. π³-X splits between extrinsics and depth. MapAnything has substantial headroom in every factor and is the only backbone for which intrinsics is uniquely best in any scene. The negative mean intrinsics gain for VGGT also shows that inserting a more accurate component into a coupled prediction can reduce final geometry quality.

## The pose-winner/geometry-winner discrepancy is a depth case

In all 20 ScanNet++ scenes, DA3 has the best pose AUC@30 among the four native baselines. Another backbone nevertheless has higher native F-score in eight scenes. For those eight scenes, DA3's largest positive conditional gain is **depth in every case**:

| Scene | Native geometry winner | Switch gain over DA3 | DA3 GT-extrinsics gain | DA3 +GT-intrinsics gain | DA3 +GT-depth gain |
|---|---|---:|---:|---:|---:|
| 7831862f02 | π³-X | +0.0587 | +0.0189 | +0.0018 | **+0.0831** |
| fb5a96b1a2 | π³-X | +0.0361 | +0.0023 | +0.0080 | **+0.1383** |
| acd95847c5 | π³-X | +0.0346 | +0.0134 | +0.0023 | **+0.0307** |
| 40aec5fffa | π³-X | +0.0228 | +0.0108 | −0.0069 | **+0.1122** |
| c5439f4607 | π³-X | +0.0107 | +0.0041 | −0.0007 | **+0.0969** |
| c4c04e6d6c | π³-X | +0.0029 | +0.0353 | −0.0098 | **+0.0708** |
| 286b55a2bf | π³-X | +0.0013 | −0.0017 | −0.0006 | **+0.0240** |
| 38d58a7a31 | VGGT | +0.0008 | +0.0253 | +0.0046 | **+0.0819** |

This resolves one ambiguity in the earlier cross-model review. A controller that reacts to these cases by repairing camera pose targets the wrong variable. The first executable intervention to test for this DA3/ScanNet++ family should modify local depth evidence while holding cameras fixed. Candidate actions must still be derived from observed failures—for example, region-level rerouting to another frozen depth predictor, local re-estimation at higher resolution, or confidence/visibility-aware selection before the unchanged fusion backend. This experiment does not isolate a fusion-algorithm error, because the fusion procedure is fixed while its depth input changes.

## Existing observables separate camera headroom, but not depth headroom

We next correlated DA3's pre-existing test-time evidence with each of the three continuous conditional-gain targets across all 20 scenes. The four predeclared reconstruction-state scalars were tested separately for each factor, with Benjamini–Hochberg correction over the four tests in that factor family. The image analysis tested 75 global RGB summaries per factor and corrected over those 75 tests.

| Target factor | Strongest state evidence | Spearman ρ | BH q | Global RGB summaries passing q≤0.05 |
|---|---|---:|---:|---:|
| Extrinsics | round-trip pixel residual p95 | +0.665 | **0.0041** | 0 / 75 |
| Extrinsics | absolute-relative depth residual p95 | +0.647 | **0.0041** | — |
| Intrinsics after extrinsics | fusion retained ratio | −0.502 | 0.0961 | 0 / 75 |
| Depth after cameras | fusion retained ratio | +0.341 | 0.3125 | 0 / 75 |

For depth headroom, the strongest global image statistic was the minimum per-view mean gradient (ρ=0.589, raw p=0.0062), but its corrected q-value was 0.468. This is a useful positive control: the same analysis recovers a strong, factor-specific association for camera replacement, yet finds no corrected scalar or global-image predictor for depth replacement. It does not prove that a VLM is necessary. It shows that the next predictor must use richer evidence than these global summaries—most plausibly which views and regions disagree, what surfaces they depict, and whether disagreement follows occlusion boundaries, reflective/transparent content, or loss of fine structure.

## What this does and does not establish

The experiment establishes three facts:

1. all four native reconstructions have measurable upstream headroom on the same 20-scene population;
2. the responsible component differs by backbone, so a universal fixed camera action is unsuitable;
3. DA3's eight pose/geometry routing discrepancies are depth-dominant under the declared intervention order.

It does **not** show that a training-free agent can recognize the factor without GT, that a VLM is needed, or that an available depth tool realizes the oracle gain. Simple global RGB statistics and scalar reconstruction diagnostics already failed to predict cross-backbone routing gain after multiple-testing correction. The next evidence must therefore be localized: per-view and per-region disagreement, visibility support, confidence, image content, and the realized gain from one executable depth action. The controller comparison must keep the same tools and compute budget for a fixed policy, deterministic numerical policy, one-shot VLM, iterative VLM, and action oracle.

## Per-view localization: residuals work, scene averages do not

We then measured DA3 predicted-versus-GT depth error for the same 20 scenes and all 2,000 sampled views. This remains an offline target construction step: GT depth never enters a test-time policy. The output passed an independent audit that recomputed view/scene populations, valid-pixel counts, coverage ratios, and pooled means and extrema. Four predeclared scene-level error summaries were compared with the conditional GT-depth replacement gain. None survived BH correction. Even the strongest, GT-uncovered ratio, reached only ρ=0.461 and q=0.163. A scene with larger average depth error therefore does not necessarily have more recoverable F-score under the declared intervention.

The view-level question has a different answer. Within each scene, we ranked views by GT depth error and by nine non-GT signals, then treated the 20 scene-level Spearman coefficients as the statistical sample.

| Offline target | Observable evidence | Median scene ρ | Positive scenes | BH q |
|---|---|---:|---:|---:|
| mean absolute-relative depth error | outgoing residual p95 | +0.331 | 19/20 | 4.29e-5 |
| mean absolute-relative depth error | incoming residual p95 | +0.330 | 18/20 | 4.29e-5 |
| p95 absolute-relative depth error | outgoing residual p95 | +0.404 | 20/20 | 8.58e-6 |
| p95 absolute-relative depth error | incoming residual p95 | +0.396 | 20/20 | 8.58e-6 |
| mean absolute-log depth error | outgoing residual p95 | +0.381 | 20/20 | 8.58e-6 |
| mean absolute-log depth error | incoming residual p95 | +0.403 | 20/20 | 8.58e-6 |

Simple image evidence is weaker but nonzero. Mean gradient, gradient p95, and luminance entropy are positively associated with several depth-error targets after correction. This rules out the simplistic story that failures are merely low-texture views. The pattern is also consistent with errors concentrated around edges, thin structures, and occlusion boundaries, although visual case review is still required before assigning those causes. Visibility support ratio by itself is mostly uninformative; the magnitude of the geometric disagreement matters.

This narrows the prospective agent contribution. A deterministic residual can already localize suspect views, so asking a VLM to rediscover them would add little. The open problem is whether image content inside those localized views or regions identifies *why* depth is inconsistent and therefore which fixed action to execute—for example, boundary-aware refusion, local depth rerouting, dynamic-region masking, or leaving the estimate unchanged. That claim still requires qualitative error verification followed by executable tool gains and same-budget controller baselines.

## Residual misranking is strongly coupled to visibility support

We rendered the exact residual-ranking disagreement extrema in every scene: the view for which residual most underestimates GT depth-error rank and the view for which it most overestimates it. Across these 20 within-scene pairs, the underestimated-error view has higher outgoing visibility support in 19/20 scenes (median paired difference +0.248, BH q=4.01e-5) and higher incoming support in 18/20 (median +0.194, q=3.67e-4). It also has higher mean gradient in 18/20 and higher luminance entropy in 17/20 after correction. The residual's worst false positives are therefore often sparse-support comparisons, while its worst false negatives contain well-supported but visually rich structure.

The images repeatedly show thin structures and occlusion boundaries, reflective or transparent regions, and repeated high-frequency geometry among false negatives. These are qualitative hypotheses rather than causal labels. The immediate method implication is firm: raw residual magnitude is an incomplete baseline, and the next deterministic diagnostic must model residual together with valid support and uncertainty. Only its remaining disagreements can support a claim that VLM image understanding adds useful cause or action information. The complete case-selection rule, paired statistics, panels, and examples are documented in `scannetpp-da3-residual-disagreement-review.md`.

An exploratory target-free calibration supports this conclusion. Within each scene, subtracting the OLS projection of residual rank on support rank raises median scene-level correlation with offline GT depth error from 0.363 to 0.417; it improves 17/20 scenes (paired p=7.08e-4) and yields a positive correlation in all 20. Because this score was motivated and evaluated on the same scenes, it is not a final baseline and must be frozen on an independent development split.

## Reproducible artifacts

- Raw 80-case intervention: `artifacts/scannetpp-gt-geometry-factors-20-final/raw/`
- Stored summary and byte-reproduction audit: `summary/`, `audit.json`, `local-summary-audit.json`
- Factor profiles and byte-reproduction audit: `profiles/`, `local-profile-audit.json`
- DA3 continuous factor targets: `artifacts/scannetpp-da3-factor-targets-20/`
- State evidence versus factor gain: `artifacts/scannetpp-da3-state-vs-factor-gain-fdr-20/`
- Global RGB evidence versus factor gain: `artifacts/scannetpp-da3-image-vs-factor-gain-fdr-20/`
- Audited per-view GT depth errors: `artifacts/scannetpp-da3-gt-depth-error-20/`
- Scene depth error versus conditional depth gain: `artifacts/scannetpp-da3-depth-error-vs-gain-20/`
- Per-view observable evidence versus GT error: `artifacts/scannetpp-da3-view-depth-evidence-20/`
- Independent per-view reproduction: `artifacts/scannetpp-da3-view-depth-evidence-20-local-reproduction/`
- Exact residual-disagreement profiles: `artifacts/scannetpp-da3-view-residual-disagreement-extremes-20/`
- Verified residual-disagreement panels: `artifacts/scannetpp-da3-residual-disagreement-panels-20/`
- Paired support/image comparison: `artifacts/scannetpp-da3-residual-disagreement-observables-20/`
- Exploratory support-adjusted residual: `artifacts/scannetpp-da3-support-adjusted-residual-20/`
- Figure generator: `scripts/failure_mining/plot_gt_geometry_factor_profiles.py`
- Profile auditor: `scripts/audit/gt_geometry_factor_profile_outputs.py`
- Raw result SHA-256: recorded in the local run manifest
