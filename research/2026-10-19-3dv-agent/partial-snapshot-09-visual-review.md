# Partial Snapshot 09: Visual Failure Review

**Scope.** This is a provisional review of ten metric-selected cases from the completed common population of snapshot 09. It does not define the final taxonomy. Each case was selected because it attains an exact metric extremum—maximum backbone-oracle headroom or maximum cross-model rank disagreement—and is reviewed across all four baselines. The original evidence archives and hashes are retained under `artifacts/partial-snapshot-09-review/`.

## What the evidence currently supports

### 1. Catastrophic global pose failure with plausible per-frame depth

The clearest recoverable failure is not a small local geometry defect. DA3 produces visually plausible single-frame depth but a badly inconsistent camera solution on two HiRoom sequences.

| Scene | DA3 | Best observed alternative | Evidence |
|---|---:|---:|---|
| `20241230/828815/cam_sampled_04` | Pose AUC@3° 0.1061; F-score 0.0674 | π³-X: 0.9848; 0.7739 | DA3 pairwise rotation/translation errors are broadly high; π³-X is uniformly low. |
| `20241230/828785/cam_sampled_12` | Pose AUC@3° 0.0519; F-score 0.0907 | VGGT: 0.8593; 0.9396 | DA3 shows large camera errors across many pairs despite smooth-looking depth maps. |

The source frames contain large bright or weakly textured surfaces, oblique views, window/outdoor regions, and thin foreground structures. This is an observed visual correlation, not yet a causal label.

**Tool implied by the failure:** detect global camera inconsistency, then re-estimate or replace the camera trajectory before changing local depth. A same-budget alternative-backbone rerun is already a strong baseline.

**Possible VLM increment:** recognizing that a sequence is dominated by blank walls, overexposed openings, or thin foreground structure could help choose a camera fallback. The current evidence does not prove that this requires semantics: pairwise camera consistency or confidence statistics may already identify the failure. A VLM claim survives only if it selects the correct recovery when deterministic diagnostics are ambiguous.

### 2. Geometry-detail failure while pose remains usable

Other scenes separate geometry quality from pose quality.

| Scene | Pose evidence | Geometry evidence | Visual observation |
|---|---|---|---|
| HiRoom `20241230/828763/cam_sampled_12` | DA3 0.8296 AUC@3°; π³-X 0.8074 | DA3 0.8526 F-score; π³-X 0.9753 | Thin lamp, plant boundaries, furniture edges, and large planar surfaces. |
| ScanNet++ `7831862f02` | DA3 0.9542; π³-X 0.9043 | DA3 0.8225; π³-X 0.8812 | Dense chairs, cubbies, cabinet edges, and close depth discontinuities. |

Changing the global camera solution is not the obvious action in these cases. π³-X preserves sharper object boundaries and improves geometry despite equal or worse pose.

**Tool implied by the failure:** region- or edge-aware depth replacement, confidence recalibration, and detail-preserving fusion under fixed cameras. The 20-scene GT extrinsics/intrinsics/depth factor intervention will determine whether this interpretation is consistent across ScanNet++.

**Possible VLM increment:** a VLM could distinguish a real thin object boundary from texture and select the region whose depth should be re-estimated or fused differently. This remains unproven until compared with image gradients, depth discontinuities, confidence, and segmentation-only deterministic policies.

#### Audited factor intervention supports the depth hypothesis

An independently audited 14-scene ScanNet++ intervention replaces, in order, extrinsics, then intrinsics, then depth while keeping the sampled views and evaluator fixed. These are **ordered conditional gains**, not independent causal effects. Ground truth is used only for offline diagnosis.

| Model | Native F-score | +GT extrinsics | +GT intrinsics | +GT depth | Mean extrinsics gain | Mean conditional intrinsics gain | Mean conditional depth gain |
|---|---:|---:|---:|---:|---:|---:|---:|
| DA3 | 0.7774 | 0.7916 | 0.7906 | 0.8586 | +0.0143 | −0.0010 | +0.0680 |
| VGGT | 0.6709 | 0.6955 | 0.6836 | 0.8586 | +0.0246 | −0.0120 | +0.1751 |
| π³-X | 0.7388 | 0.8011 | 0.7982 | 0.8586 | +0.0624 | −0.0029 | +0.0604 |
| MapAnything | 0.2773 | 0.4999 | 0.6429 | 0.8586 | +0.2226 | +0.1430 | +0.2157 |

For DA3, VGGT, and π³-X, GT depth improves F-score on all 14 scenes; it is the unique largest positive conditional factor on 14/14 DA3, 13/14 VGGT, and 9/14 π³-X scenes. Intrinsics are not a general repair target for these three models. MapAnything has substantial conditional headroom at all three stages, so a single “depth-only” explanation would be wrong for that backbone.

The selected scenes agree with this aggregate pattern. On ScanNet++ `7831862f02`, DA3 gains +0.0189 from extrinsics, +0.0018 from intrinsics, and +0.0831 from depth; π³-X gains +0.0260, −0.0036, and +0.0227. On `f3d64c30f8`, DA3 gains only +0.0043 from extrinsics but +0.0693 from depth, while VGGT's large deficit is mostly depth (+0.3999 after the earlier interventions).

This establishes **where oracle headroom exists**, but not yet an executable repair. The next tool experiment should therefore measure whether image-observable regions can select a local depth re-estimation or fusion action without GT, rather than starting with a generic camera agent.

#### The 20-scene pose intervention rules out a fixed camera-replacement policy

The completed 20-scene ScanNet++ population gives a sharper test of camera repair. For each backbone, the intervention keeps its native depth and intrinsics but substitutes GT extrinsics before running the same fusion and evaluator. Gains are measured against a native-resample control produced by the same intervention code path. This control is close to, but not exactly equal to, the official native result: the maximum absolute per-scene F-score difference is 0.000029 for DA3, 0.000400 for VGGT, 0.001140 for π³-X, and zero for MapAnything. This bounded control discrepancy is kept separate from the intervention gain.

| Model | Native-resample control F-score | +GT extrinsics F-score | Mean gain vs. control | Positive / negative scenes | Largest negative scene |
|---|---:|---:|---:|---:|---|
| DA3 | 0.7833 | 0.7978 | +0.0146 | 17 / 3 | `7bc286c1b6`: −0.0021 |
| VGGT | 0.6758 | 0.7044 | +0.0286 | 13 / 7 | `cc5237fd77`: −0.0961 |
| π³-X | 0.7229 | 0.8055 | +0.0826 | 18 / 2 | `3e8bba0176`: −0.0016 |
| MapAnything | 0.2945 | 0.4940 | +0.1994 | 19 / 1 | `bde1e479ad`: −0.0112 |

The intervention has substantial average headroom for π³-X and MapAnything, but it is not uniformly safe. VGGT is the strongest counterexample: a correct camera trajectory can be less compatible with its native depth and fusion state than the camera it predicted jointly. This is consistent with camera–depth–fusion coupling; the intervention alone does not identify which component caused each loss.

Even the GT-evaluated native pose score is a poor selector for this action. Across the 20 scenes, Spearman correlation between pose AUC@3° and the F-score gain from GT extrinsics is +0.071 for VGGT, +0.009 for DA3, −0.274 for π³-X, and −0.209 for MapAnything. These are descriptive correlations on a small diagnostic population, not significance claims. They show that “low pose AUC → replace cameras” is not supported even with privileged GT diagnostics. By contrast, lower native F-score is more associated with intervention headroom (correlations from −0.352 to −0.774), but native F-score also uses GT and cannot serve as a test-time policy input.

This narrows the recovery problem: the system must predict **whether a camera action improves the downstream reconstruction**, not merely detect camera error. Any later agent must be compared with a deterministic policy using the same observable camera, depth, consistency, and fusion diagnostics; GT pose and GT intervention gains remain offline labels only.

#### DA3 camera-action headroom is predictable from geometric diagnostics

For DA3, all 20 ScanNet++ scenes already have native, test-time-computable geometry, visibility, fusion, and image evidence. We tested twelve predeclared fields against the offline F-score gain from the GT-extrinsics intervention and applied Benjamini–Hochberg correction across the twelve correlations.

| Test-time evidence | Spearman ρ with camera-intervention gain | Two-sided p | BH q |
|---|---:|---:|---:|
| Cross-view round-trip pixel error P95 | +0.665 | 0.0014 | 0.0124 |
| Cross-view relative-depth inconsistency P95 | +0.647 | 0.0021 | 0.0124 |
| Mean image gradient | +0.474 | 0.0349 | 0.1395 |
| Fusion retained ratio | +0.373 | 0.1053 | 0.1877 |

Only the two geometric consistency diagnostics remain associated with intervention gain after correction. The four simple image statistics—gradient, luminance entropy, and saturated-dark/bright pixel ratios—do not. This is a small single-dataset analysis and does not establish a deployable action threshold, but it is direct evidence against requiring a VLM for this DA3 failure family. A deterministic camera-recovery trigger based on geometric inconsistency is therefore a required baseline and may be sufficient. The VLM question should move to residual cases where these diagnostics cannot distinguish camera, depth, and visibility failures, and it must be tested on held-out scenes and other backbones.

### 3. Repetition and sequence transitions expose backbone-specific camera errors

ETH3D `terrains` and `relief` contain structured transitions visible as cross-shaped bands in pairwise camera-error matrices. `terrains` moves between corridor views and near-frontal wall/display views; `relief` contains repeated doors, arches, and corridor appearances.

- `terrains`: DA3 reaches 0.5869 pose AUC@3° and 0.8934 F-score; VGGT falls to 0.1018 and 0.6669.
- `relief`: DA3 reaches 0.7232 and 0.8371; MapAnything falls to 0.1189 and 0.2526.

These are useful negative controls for an agent claim: DA3 already handles them best. They show the kind of evidence a diagnostic must read, but they do not show that DA3 itself needs repair.

**Tool implied by the failure:** view-graph segmentation, overlap verification, and loop-edge rejection around the transition frame.

**Possible VLM increment:** distinguishing repeated architectural instances or recognizing a true change of place could help when appearance retrieval is ambiguous. Cycle consistency, overlap prediction, and retrieval scores must be tested first; otherwise VLM reasoning is unnecessary.

### 4. Fine repeated outdoor structures are a shared hard case

ETH3D `playground` contains many thin, repeated poles and rails. All four methods have low-to-moderate reconstruction quality: F-score ranges only from 0.7004 to 0.7403, while accuracy/completeness show large and different tradeoffs. There is no decisive backbone rescue.

This may reflect information or representation limits rather than a recoverable pipeline error. A recovery method should not count such a case as agent headroom unless an allowed action demonstrably improves the official evaluator.

### 5. A pose improvement need not improve the reconstruction endpoint

On ETH3D `kicker`, VGGT improves pose AUC@3° from DA3's 0.4638 to 0.5799, but every model's F-score is already above 0.996. This case prevents an overly broad claim that every camera-metric gain produces a meaningful geometry gain. Pose and reconstruction metrics must remain separate endpoints.

## Current task definition implied by the cases

The evidence does not yet support one universal “reconstruction repair” action. It supports testing three distinct recovery decisions:

1. **Global camera fallback or re-estimation** when pairwise camera inconsistency is widespread.
2. **Local depth/confidence/fusion repair under fixed cameras** when pose is good but geometry detail is worse.
3. **View-graph segmentation or edge rejection** when errors concentrate around repeated appearances or sequence transitions.

The final taxonomy should retain an error family only if two conditions hold: an executable action improves the official metric, and the observable state available at test time predicts when to apply that action. VLM input is justified only for the residual cases where image semantics add predictive information beyond deterministic geometric diagnostics.

## Evidence artifacts

- Exact selected metrics: `artifacts/partial-snapshot-09-review/selected_case_metrics.csv`
- HiRoom contact sheets: `artifacts/partial-snapshot-09-review/contact-sheets/hiroom-*.jpg`
- ETH3D contact sheets: `artifacts/partial-snapshot-09-review/contact-sheets/eth3d-*.jpg`
- ScanNet++ contact sheets: `artifacts/partial-snapshot-09-review/contact-sheets/scannetpp-*.jpg`
- Original immutable archives:
  - HiRoom SHA-256 recorded in the local run manifest
  - ETH3D SHA-256 recorded in the local run manifest
  - ScanNet++ SHA-256 recorded in the local run manifest
- Audited 14-scene factor artifacts: `artifacts/scannetpp-gt-geometry-factors-14-final/`
- Combined 20-scene GT-extrinsics intervention: `artifacts/scannetpp-gt-extrinsics-snapshot-09-combined-20/`
- 20-scene pose/gain diagnostic with hashes and explicit GT-use flags: `artifacts/scannetpp-gt-extrinsics-snapshot-09-predictors/`
- DA3 test-time evidence versus camera-intervention gain, with BH correction: `artifacts/scannetpp-da3-gt-extrinsics-testtime-evidence-20/`
- Exact positive/negative camera-action extrema and visual review: `research/2026-10-19-3dv-agent/scannetpp-gt-extrinsics-extremal-review.md`
