# Evidence for reconstruction repair tools, 2025–2026

Research date: 2026-10-19. Scope: a fixed collection of RGB observations; improve estimated cameras and dense geometry without updating any network parameters. Existing pretrained tools are allowed. This is a primary-source evidence note, not a demonstrated agent result. Dataset membership alone does not establish the prevalence of a failure mode.

## What the evidence supports

The strongest concrete portfolio is (1) MP-SfM or GLUEMAP for camera estimation, (2) Doppelgangers++ for candidate pair disambiguation, and (3) GGPT for dense geometry refinement from RGB-derived SfM guidance. These provide substantial non-agent baselines. A VLM controller must beat their default pipelines and an equally budgeted numerical controller; simply invoking them is not a new reconstruction capability.

### 1. Low overlap / low parallax / ambiguous registration → MP-SfM

**Evidence.** [MP-SfM, 2025](https://arxiv.org/html/2504.20040v1), §3.3–3.4, Fig. 4, Tables 1, 3–6. It combines monocular depth/normal priors with multiview constraints and propagates uncertainty. Table 1's deliberately reduced-overlap ETH3D setting gives AUC@1/5/20° of **27.3/55.9/71.8** versus **7.9/12.7/14.6** for SP+LG COLMAP. This is a controlled sampling result, not a failure-rate estimate for ordinary ETH3D inputs. Table 3 tests low parallax on RealEstate10k. Fig. 4 and Table 6 support depth/free-space consistency checks for symmetry-induced bad registration; the large symmetry gains are on SMERF. Table 5 explicitly says depth regularization can reduce accuracy for under-constrained two-view tracks. Table 4 establishes the importance of prior uncertainty.

**Eligible tool:** inference with frozen priors plus optimization of depths, poses, and points. **Gap:** reliable priors and observable geometry remain necessary; more prior weight is not universally better. **VLM distinction (proposal):** identifying candidate repeated structures can focus inspection, but uncertainty and free-space inconsistency are already numerically measurable. MP-SfM does not establish that a VLM improves them.

### 2. Accurate local predictions but poor global alignment / scaling → GLUEMAP

**Evidence.** [GLUEMAP, CVPR 2026](https://arxiv.org/html/2605.26103v1), §3, Table 1, Table 5, §4.4. Local feedforward reconstructions are combined by global averaging and augmented bundle adjustment. ETH3D Table 1 reports AUC@1/3/5°: **Pi3 13.2/36.1/48.9**, **Pi3+BA 30.6/55.1/65.1**, **GLUEMAP 53.0/76.9/83.6**. Ground-truth-calibrated results are separately marked; do not merge protocols. Table 5 shows large-scene LaMAR results where whole-scene feedforward baselines run out of memory. §4.4 identifies dependence on local reconstruction quality and pinhole-model training. Its pure-rotation sentence appears inconsistent in wording; do not quote it as a clear limitation without checking the final PDF.

**Eligible tool:** frozen backbones plus global geometric optimization. **Gap:** this is primarily camera/sparse reconstruction evidence, not proof that final dense surfaces are repaired. **VLM distinction (proposal):** local-to-global residuals and resource limits need no semantics; a controller must add something beyond invoking the established solver.

[Official code](https://github.com/colmap/gluemap) documents SALAD retrieval, optional Doppelgangers++, local Pi3/Pi3X/VGGT/MapAnything inference, global mapping, and SIFT track snapping. It outputs a COLMAP sparse reconstruction.

### 3. Visually similar but different surfaces → Doppelgangers++

**Evidence.** [Doppelgangers++, CVPR 2025](https://arxiv.org/html/2412.05826v2), §3.2, §4.3–4.5, Figs. 3, 6–8. Fig. 7 illustrates that MASt3R can conflate visually similar but distinct surface pairs. The learned classifier uses multilevel MASt3R features; the backbone is frozen while classification heads were trained beforehand. Fig. 6 demonstrates correction on VisymScenes everyday street scenes. Its evaluation includes pair classification and reconstructed-model disambiguation; it is not a five-benchmark failure audit. MP-SfM's comparison to the older Doppelgangers must not be misreported as a comparison to Doppelgangers++.

**Eligible tool:** pretrained pair classifier at inference. **Gap:** pair rejection can disconnect the observation graph; rejected pairs do not themselves recover geometry. **VLM distinction (proposal):** object identity, signs, layout, or materially different surrounding context may contribute evidence beyond local appearance, but the classifier already has strong 3D-aware features. No primary result checked here proves that a general VLM surpasses it. Evaluate semantic access against the classifier plus graph/geometry diagnostics, including false rejection and component fragmentation.

### 4. Multilayer artifacts / inconsistent dense points → GGPT (important baseline)

**Evidence.** [GGPT, CVPR 2026](https://arxiv.org/html/2603.11174v1), §2–3, Fig. 1–2, Table 1, Appendix H. Guidance is derived **solely from the input RGB views**: dense matching, sparse BA, then multiview DLT triangulation. A pretrained 3D point transformer propagates sparse geometric guidance into dense predictions. Table 1, ETH3D 8-image AUC@5/10cm: **VGGT 23/36 → 47/61**, **Pi3 25/41 → 36/53** with the full proposed pipeline. These numbers combine improved SfM and GGPT; they are not an isolated refiner ablation. Training used ScanNet++ with VGGT predictions; application to other backbones needs no fine-tuning. Appendix H discusses unrecoverable details, weak guidance, and patch artifacts.

**Eligible tool:** yes, a frozen pretrained refiner plus RGB-only SfM; no extra depth sensor or test-set GT required at inference. **Gap:** unreliable or missing guide points limit repair. **VLM distinction (proposal):** semantics might identify suspicious regions requiring alternative guidance; a VLM is unnecessary for merely detecting point-cloud multilayers.

[Official repository](https://github.com/ChenYutongTHU/GGPT) supplies an RGB-directory demo and links [released model weights](https://huggingface.co/YutongGoose/GGPT). It lists extreme low overlap/parallax, lighting changes, misleading symmetry, and dynamic scenes as SfM failures, plus monocular extrapolation and large-scene patch artifacts. It currently uses exhaustive matching. Its evaluation loader requests GT for evaluation; that must not be confused with required inference input. [CVPR supplement](https://openaccess.thecvf.com/content/CVPR2026/supplemental/Chen_GGPT_Geometry-Grounded_Point_CVPR_2026_supplemental.pdf) provides additional ablations and error maps; match figure numbering to that version before embedding figures.

### 5. Blur/degradation → geometry-aware restoration (evidence, availability caveat)

[GARD, May 2026 preprint](https://arxiv.org/html/2605.26230v1), Tables 1–3 and Appendix B.3, evaluates pose/reconstruction/restoration on the DA3 benchmark spanning **HiRoom, ETH3D, DTU, 7Scenes, ScanNet++**. It trains a feature denoiser on Hypersim/TartanAir with synthesized motion blur and geometry-aware supervision. This establishes a relevant remedy and evaluation design, not that these five datasets naturally have a particular blur prevalence. It is not an inherently training-free method. An independently released frozen checkpoint could be used by a training-free agent; checkpoint availability and exact inputs require a separate deployment check. Ordinary single-image restoration should not be assumed to preserve inter-view correspondence merely because the image looks sharper.

## Material effects: evidence is strong, universal masking repair is not

[3DReflecNet, 2026](https://arxiv.org/html/2605.10204v1), Tables 3–5, isolates reflective, transparent, and low-texture materials. Table 5 explicitly measures surface reconstruction by Chamfer distance across material categories. This supports material-dependent geometric failures, but it is a different benchmark; do not transfer its prevalence or exact degradation to DTU/ETH3D/ScanNet++/7Scenes/HiRoom.

[HiNeuS, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/papers/Wang_HiNeuS_High-fidelity_Neural_Surface_Mitigating_Low-texture_and_Reflective_Ambiguity_ICCV_2025_paper.pdf), Fig. 1 and method description, handles reflection ambiguity with SDF visibility, textureless regions with planar regularization, and detail loss with relaxed Eikonal constraints. It optimizes neural surfaces and appearance during training; exclude it from a strict no-network-training tool budget. It demonstrates that explicit physical/geometric modeling, rather than deletion, is a meaningful remedy direction.

**Proposed semantics-dependent intervention, not established by the above papers:** a VLM/segmenter could label a suspected mirror, glass pane, monitor, or specular object and localize its image evidence. Such a label should select a correspondence/visibility model or a focused verification experiment, not directly erase the object. Suppressing reflected matches might improve camera estimation yet reduce surface completeness. Keep separate (a) unreliable observation masks, (b) reconstructed physical surface support, and (c) unknown geometry. Compare pose, completeness, and accuracy jointly. Textureless walls likewise contain real geometry. No 2025+ result checked in this bounded search establishes that generic VLM masks improve all five benchmark datasets.

## Resolution, crops, consistency, and fusion: what can actually be claimed

| Candidate action | Evidential status | Appropriate claim / constraint |
|---|---|---|
| BA after feedforward inference | Directly supported | GLUEMAP Table 1 above isolates Pi3+BA as a useful baseline; it does not establish universal gains. |
| Sparse-guided dense refinement | Directly supported | GGPT is a concrete released solution; retain its correspondence map and common coordinate system. |
| Depth/normal and free-space checks | Directly supported | MP-SfM supports uncertainty-aware integration and occlusion-aware registration rejection. |
| Increase raw inference resolution | Plausible intervention, not generally established here | Requires a controlled native-resolution ablation; out-of-distribution resolution and memory can hurt. |
| Crop salient/failing regions then rerun | Research hypothesis for this agent | Preserve exact image-coordinate/intrinsic transforms and align predictions through actual shared evidence; do not claim crop-only inference restores missing geometry. |
| TSDF or other depth fusion | Standard reconstruction operation, not new agent evidence | Requires calibrated views and scale-consistent depths; fusion can consolidate systematic errors. No specific 2025+ universal repair result verified in this note. |
| Remove points by confidence | Filtering, not recovery | Accuracy-only improvement may be deletion; report completeness and retained coverage. |

[HD-VGGT, March 2026](https://arxiv.org/abs/2603.27222) addresses high-resolution reconstruction with a learned dual branch and feature upsampler; its claims do not validate naive resolution increases on frozen VGGT. [URGT, CVPR 2026](https://arxiv.org/abs/2603.03026) uses a trained multipatch model with depth/normal priors and GridMix training; it is monocular high-resolution geometry refinement, not evidence that an arbitrary frozen multiview crop-and-fuse wrapper works.

## Nearby test-time methods excluded by the network-freeze constraint

- [Free Geometry, April 2026](https://arxiv.org/html/2604.14048v1), §3.2–3.3, Algorithm 1 / implementation appendix, trains LoRA parameters and camera tokens by full-view/partial-view feature distillation. The implementation says **dataset-wise** adaptation, not simply an independent frozen inference call. Appendix LoRA-rank table reports ETH3D DA3 AUC3 **0.514 → 0.581**, F1 **0.730 → 0.755** for rank 32. Relevant competitor, excluded action under strict no network updates.
- [Self-Geometry, August 2026](https://arxiv.org/html/2608.10708v1), §III-B–D, uses explicit correspondence-based multiview/epipolar objectives and LoRA updates. Its evaluation covers 7Scenes, ETH3D, ScanNet++, HiRoom. Ground-truth-free does not mean training-free. It is useful evidence that numerical correspondences can drive adaptation without semantics, but not an eligible frozen-tool remedy.

## Experimental implication (our synthesis, not a published result)

Start from actual baseline failures, then allocate repairs by measurable evidence. Build a failure ledger containing input view pairs/regions, competing camera hypotheses, match and depth consistency, missing-surface coverage, candidate action, and observed before/after change. Do not hardcode dataset names into failure labels or action routing.

The decisive VLM test is an information ablation: equal tool access, equal compute, and the same numeric diagnostics, with versus without RGB/semantic access. Include (1) base reconstructor, (2) each strong fixed pipeline above, (3) bounded numerical search/controller, (4) VLM controller, and (5) an oracle action selector to measure the portfolio's upper bound. Report camera accuracy, dense accuracy and completeness, per-failure wins/regressions, compute, and unrepaired cases. If semantics only renames an already-detected geometric inconsistency without changing the successful action, it has not established value.

This evidence set does **not** yet justify claiming that these tools repair every failure on all five target datasets. It does justify an evidence-driven agent study with strong geometric baselines and narrowly testable semantic interventions.
