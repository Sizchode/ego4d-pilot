# Baseline failure evidence: cameras and dense geometry

Verified 2026-10-19. Scope: RGB-only test-time improvement of frozen VGGT, VGGT-Ω, Pi3/Pi3X, MapAnything and DA3 on the DA3 benchmark. **Literature provides several genuine failure mechanisms, but does not provide a mechanism-labelled per-scene failure inventory for these baselines across all five DA3 datasets. No examined source establishes that a VLM selects geometric refinements better than a geometry-only controller.** Suggested interventions below are experimental hypotheses unless explicitly identified as existing fixes.

## 1. Input-order sensitivity: a demonstrated mechanism, substantially addressed already

**Observation / exact evidence.** Pi3 v3 §4.4, Table 6 measures reconstruction changes when every frame is made the first frame. On **DTU**, mean accuracy standard deviation is VGGT 0.033 versus Pi3 0.003; on **ETH3D**, 0.049 versus 0.000 (rounded). Figure 2 shows variation even with DINO-based reference selection. These are Pi3's protocols, not automatically the DA3 release's frame subsets. Pi3 eliminates designated-reference dependence architecturally. [Pi3 v3 §4.4, Table 6, Fig. 2](https://arxiv.org/html/2507.13347v3#S4.SS4)

**Existing remedy versus proposed tool.** Pi3 is already a strong non-agent remedy. For VGGT, repeated reference selection plus geometric output validation is a possible training-free intervention, but the cited experiment demonstrates sensitivity, not a successful selection algorithm. Do not market reference selection as an unsolved problem across all baselines. A VLM might describe an uninformative anchor, but no evidence here shows an advantage over match connectivity, coverage, or repeated-inference agreement. The necessary comparison is identical candidate runs and compute, differing only in selection policy.

## 2. Weak camera evidence: repeated appearance, large roll, blur

**Observation / exact evidence.** VGGT-Ω §4.2, **Figure 6** explicitly reports DA3 estimating nearly no motion in a snow-lift sequence with repetitive terrain, and duplicating tower geometry in a rolling drone sequence. Ω improves both examples. **These are additional qualitative sequences, not identified as DTU, ETH3D, ScanNet++, 7Scenes or HiRoom.** Appendix C reports residual Ω failures with strong blur, abrupt FOV changes and distortion. [VGGT-Ω §4.2 Fig. 6 and Appendix C](https://arxiv.org/html/2605.15195v1#S4.SS2)

VGGT's own §5 reports degradation under extreme rotations, unsupported fisheye/panoramic imagery, and failure under substantial non-rigid deformation. It proposes targeted fine-tuning; this is outside the training-free scope. [VGGT §5](https://arxiv.org/html/2503.11651v1#S5)

**Dataset association.** DA3's benchmark documentation already excludes unusually rotated ETH3D images and recalibrates ScanNet++ after removing blurry frames. Therefore those raw-dataset defects cannot simply be asserted as untouched failure modes of the released benchmark. 7Scenes is explicitly described as low-resolution with substantial motion blur in DA3 §6.3, but the paper does not causally attribute its score to blur. [DA3 benchmark documentation, Dataset Details](https://github.com/ByteDance-Seed/Depth-Anything-3/blob/main/docs/BENCHMARK.md)

**Existing remedy versus proposed tool.** Ω already fixes the two illustrated DA3 examples through its model; this is not a training-free agent result. Candidate tools are correspondence verification, track/pose optimization, overlap-aware view selection, and additional informative frames when available. Geometry must still establish observable motion. A VLM could flag camera-roll or repetitive-scene semantics, but cannot manufacture parallax or disambiguate identical surfaces from absent evidence. No reported experiment establishes the incremental benefit of VLM routing.

## 3. Transparent surfaces and uncertain fine geometry: distinguish unresolved ambiguity from already-fixed artifacts

**Observation / exact evidence.** Pi3 v3 **Appendix A.8** explicitly says transparent objects are unsupported because complex light transport is not modelled; it also reports missing fine detail and grid artifacts from MLP/pixel-shuffle upsampling, particularly under high uncertainty. **No DA3 benchmark scene or dataset is assigned to these failures.** [Pi3 v3 Appendix A.8](https://arxiv.org/html/2507.13347v3#A1.SS8)

**Existing remedy.** Pi3X's official engineering update replaces the output head with a convolutional head to reduce grid artifacts and changes confidence learning from binary-mask approximation to continuous quality scoring for noise filtering. It also supports pose/intrinsics/depth conditioning. This is an existing checkpoint capability, not a proposed agent contribution; the README supplies no quantitative DA3-benchmark ablation of those individual changes. [Pi3/Pi3X official README, Introducing Pi3X](https://github.com/yyfz/Pi3#introducing-pi3x-engineering-update)

**Proposed tool and limits.** Local reruns at higher resolution, visibility-aware multi-view checks, and confidence/coverage audits can test whether a local artifact is recoverable. Rejecting uncertain geometry trades completeness for accuracy; it does not reconstruct a transparent surface. A VLM might identify a mirror/window/transparent vessel to propose a region for inspection. It cannot be treated as a source of metric depth or a license to replace geometry by a plausible plane. Compare against confidence-only and geometric-consistency masks, and measure retained coverage and false exclusions. No source above validates this VLM increment.

## 4. Noisy conditioning and coupled pose/depth error: real opportunity, but posed/unposed is not causal isolation

**Observation / exact evidence.** MapAnything **§5 Limitations** states that uncertainty/noise in supplied geometric inputs is not explicitly modelled and effective iterative inference remains unexplored. Its Fig. 4 already demonstrates improvements over VGGT on textureless surfaces, water, large disparity changes and large scenes using common edge/sky masking. These examples are not mapped to the five DA3 test sets. [MapAnything §5 and Fig. 4](https://arxiv.org/html/2509.13414v1#S5)

**Proposed tool.** RGB-estimated cameras could be refined and re-injected into a conditioning-capable model, then verified using held-out observations and geometry. The mechanism is not “use GT pose”; GT remains evaluator-only in the RGB-only treatment. Refinement can make wrong inputs more self-consistent, so reprojection loss alone is insufficient evidence of correctness. VLM participation would need an actual decision advantage over observable residuals, view coverage, and consistency, not a narrative explanation of the error.

### What the DA3 reported paired results actually show

DA3 v1 Table 3 reports the following **without / with GT poses**. These are aggregate protocol results, not scene-level diagnoses. All entries are F1↑ except DTU CD↓ in mm. [DA3 v1 Table 3](https://arxiv.org/html/2511.10647v1#S7.SS1)

| Method | HiRoom | ETH3D | DTU | 7Scenes | ScanNet++ |
|---|---:|---:|---:|---:|---:|
| MapAnything | 32.4 / 69.2 | 54.8 / 71.9 | 7.91 / 3.97 | 44.8 / 55.2 | 39.4 / 71.3 |
| Pi3 | 75.8 / 85.0 | 72.7 / 80.6 | 3.28 / 1.72 | 44.2 / 57.5 | 63.1 / 73.3 |
| VGGT | 56.7 / 70.2 | 57.2 / 66.7 | 2.05 / 1.44 | 47.9 / 51.4 | 66.4 / 70.7 |
| DA3-Giant | 85.1 / 95.6 | 79.0 / 87.1 | 1.85 / 1.85 | 53.5 / 56.5 | 77.0 / 79.3 |

Inference from this table: camera-conditioned reconstruction has considerable headroom for some model/dataset pairs, especially MapAnything on HiRoom/ScanNet++. DA3-Giant has residual errors even with GT cameras. Neither observation identifies transparency, drift, nor bad correspondences as the cause. Pi3X and Ω are absent from this table and require fresh reproduction. Do not combine these paper numbers with current README expected numbers as though identical releases/protocols.

### Verified implementation of the current DA3 modes

The exact inspected revision is recorded in the local run manifest.

- `bench/evaluator.py` lines 182–201 invokes **separate inference calls**: unposed receives images only; posed receives images, `scene_data.extrinsics` and `scene_data.intrinsics`. [Pinned evaluator](https://github.com/ByteDance-Seed/Depth-Anything-3/blob/main/src/depth_anything_3/bench/evaluator.py#L182)
- `api.py` lines 196–218 preprocesses those cameras, normalizes extrinsics, and sends both camera inputs through `_run_model_forward`. Thus GT affects depth prediction. [Pinned inference](https://github.com/ByteDance-Seed/Depth-Anything-3/blob/main/src/depth_anything_3/api.py#L196)
- `api.py` lines 341–365 then replaces output intrinsics with input intrinsics. With default `align_to_input_ext_scale=True`, it replaces output poses with input poses and divides depth by fitted pose scale. [Pinned alignment](https://github.com/ByteDance-Seed/Depth-Anything-3/blob/main/src/depth_anything_3/api.py#L341)

Consequently current DA3 `recon_posed` versus `recon_unposed` changes **conditioning, cameras used downstream, and scale handling**. It is not simply fusing identical predicted depths with GT cameras. This verification does not establish how every historical baseline adapter produced the paper's posed columns.

## Ground-truth defects are not test-time repair claims

Ω **Appendix B, Fig. 10** shows ScanNet++ training-annotation defects around chairs and lamps; Appendix C associates office-monitor instability with noisy training data. These are **not named DA3 evaluation failures**. Appendix B also reports bent pseudo-ground-truth geometry despite low reprojection error. [VGGT-Ω Appendix B/C](https://arxiv.org/html/2605.15195v1#A2)

The defensible use is to audit supervision/evaluation ambiguity and design independent verification. Do not count modifying benchmark GT as a reconstruction improvement. Do not claim ScanNet++ lamp examples prove all five selected baselines fail on transparent test objects.

## Exactly what is known for the five datasets

| DA3 dataset | Direct mechanism evidence in examined sources | Missing evidence |
|---|---|---|
| DTU | VGGT reference-order sensitivity measured by Pi3; paired reconstruction table | Which DA3 scan has which remaining mechanism, especially Pi3X/Ω |
| ETH3D | Reference-order sensitivity measured by Pi3; release removes unusual camera rotations; paired reconstruction table | Residual low-overlap/repetition/drift causal labels for exact release scenes |
| ScanNet++ | Camera recalibration/filtering documented; paired reconstruction gap; training-annotation defects separately documented | Model-output failure on a named DA3 scene with verified reflection/transparent cause |
| 7Scenes | Low-resolution/blur dataset description; paired reconstruction table | Per-scene causal attribution of residual errors to blur, texture or overlap |
| HiRoom | Paired reconstruction gap | Published baseline failure labelled by material/occlusion/room geometry on a named scene |

Ω's own §4.2 uses 10 randomly sampled frames per sequence on its evaluation datasets; this is not the DA3 benchmark protocol. Its DA3-suite row must be labelled **our reproduction** until run with frozen DA3 data, preprocessing, frames and evaluator. [VGGT-Ω evaluation protocol](https://arxiv.org/html/2605.15195v1#S4.SS2)

A defensible first experiment therefore preserves the exact RGB-only task, runs these baselines on identical inputs, and saves actual per-scene cameras/depth/fused geometry plus residual/coverage diagnostics. Only then select the failing cases whose proposed intervention can be measured. For each accepted repair compare fixed pipeline, geometry-only selection, and VLM selection with the same tools, candidate budget and evaluator; keep a no-change action. No scene-name exceptions, assumed semantic depth, or hardcoded corner-case repair rules are justified by this evidence.
