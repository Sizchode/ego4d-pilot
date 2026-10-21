# Task-first baseline evidence

Verified against primary sources on 2026-10-19. This is a research recommendation, not a record of downloaded weights or completed benchmark runs.

## 1. Define the evaluated task before the toolkit

Recommended primary task: **recover cameras and dense scene geometry from a fixed collection of RGB images**, without supplied ground-truth intrinsics, poses, depth, or additional captured views. An agent can inspect intermediate results, select computations and revise a reconstruction under a declared budget. “Training-free” should mean no new training of the agent or reconstruction networks; decide explicitly whether test-time geometric optimization is allowed. Frozen pretrained networks themselves are trained models.

Measure both cameras and dense geometry. Pose improvement alone does not establish more accurate or complete surfaces; a plausible rendering alone does not establish geometric recovery. If the intended output becomes a mesh, long video map, dynamic scene, or simulation asset, revise the task, datasets and baselines accordingly before selecting tools.

## 2. A suitable existing benchmark anchor

DA3's released benchmark covers ETH3D, ScanNet++, 7Scenes, HiRoom and DTU, with separate `pose`, `recon_unposed` and `recon_posed` modes. It reports camera AUC@3°/30° and geometry F-score / Overall, with TSDF fusion and pose alignment. DTU-49 evaluates geometry on 22 scenes; DTU-64 evaluates poses on 13 scenes. DTU Overall uses millimeters while other datasets use meters; its reported cross-dataset geometry average excludes DTU. Its ETH3D subset filters images and its ScanNet++ poses are recalibrated. Therefore pin this processed data and evaluator, rather than silently mixing original dataset protocols. [Official benchmark instructions](https://github.com/ByteDance-Seed/Depth-Anything-3/blob/main/docs/BENCHMARK.md)

For the primary task use `recon_unposed` plus pose evaluation. A separately labeled posed reconstruction experiment diagnoses depth/fusion quality, but cannot support a claim of recovering unknown cameras. Include per-scene accuracy and completeness, not only their aggregate. Follow official per-dataset masks and thresholds; report failure and excluded-scene counts rather than scoring only successful outputs. These are proposed experiment requirements.

MapAnything's Dense Up-to-N-View benchmark supplies another useful protocol: ETH3D, ScanNet++V2 and TartanAirV2-WB, with varying view counts and connected covisibility graphs. It is useful for testing sparse-to-dense input scaling; disconnected-input recovery is outside that default sampling contract. [Official protocol](https://github.com/facebookresearch/map-anything/blob/main/benchmarking/dense_n_view/README.md)

## 3. Small, current baseline set

| Complete reconstruction system | First release / publication | Why include it; executable evidence |
|---|---|---|
| **VGGT** | March 2025 preprint; CVPR 2025 | Recognizable 2025 anchor jointly predicting cameras, depth, point maps and tracks. Official repository provides inference, checkpoints, Co3D pose evaluation and COLMAP export with optional BA. Treat feed-forward and BA variants as different configurations. [Paper](https://arxiv.org/abs/2503.11651), [official code](https://github.com/facebookresearch/vggt) |
| **VGGT-Ω** | May 2026 preprint; CVPR 2026 | Strong verified 2026 comparator rather than relying only on 2025 backbones. Official inference and evaluation code exist; weight access is gated and has **not** been obtained in this research. [Paper](https://arxiv.org/abs/2605.15195), [official code](https://github.com/facebookresearch/vggt-omega) |
| **Pi3 / Pi3X** | Pi3: July 2025 preprint, ICLR 2026; Pi3X: December 28, 2025 engineering update | Pi3 supplies a permutation-equivariant alternative without a fixed reference view. Pi3X adds conditioning, smoother output and approximate metric scale; it is explicitly an engineering update, not a separate verified paper. Official `example_mm.py` runs Pi3X on images/video; original Pi3 has a released evaluation branch. Use Pi3X as the current implementation, retaining Pi3 when reproducing published tables. [Paper](https://arxiv.org/abs/2507.13347), [official code](https://github.com/yyfz/Pi3) |
| **MapAnything** | September 2025 preprint; 3DV 2026 | A universal metric geometry model covering uncalibrated reconstruction and calibrated MVS. Includes inference, weights, evaluation and adapters for other models. Pin the exact checkpoint: current weights and September 2025 `-v1` weights coexist. [Paper and venue](https://arxiv.org/abs/2509.13414), [official code](https://github.com/facebookresearch/map-anything) |
| **Depth Anything 3** | November 13, 2025 preprint; ICLR 2026 | Joint depth/ray/camera geometry and a released pose-plus-reconstruction evaluator make it a strong complete-system comparator. Use a named main-series model (e.g. DA3-Giant), not its monocular-only or metric-only series. [2025 preprint](https://arxiv.org/abs/2511.10647), [ICLR proceedings](https://proceedings.iclr.cc/paper_files/paper/2026/hash/e4cd50120b6d7e8daff1749d6bbaa889-Abstract-Conference.html), [official code](https://github.com/ByteDance-Seed/Depth-Anything-3) |

**Important VGGT-Ω checkpoint qualification:** the authors report a potential training-data contamination risk whose impact on the original weights could not be conclusively established. They explicitly supersede the original benchmark reference with `VGGT-Omega-1B-416-Reproduction` and require `image_resolution=416`. Their reproduced ETH3D AUC@3° is 51.5 versus DA3's 46.1 under their primary protocol. These author-reported results justify including Ω, not declaring it universally best across protocols. Use the reproduction weights for comparisons; record inability to obtain access rather than substituting the original checkpoint. [Official reproduction note](https://github.com/facebookresearch/vggt-omega/blob/main/reproduction.md)

Recommendation: start with DA3, Pi3X, MapAnything and VGGT-Ω-reproduction; keep VGGT as a lightweight historical anchor if resources permit. This covers strong direct RGB reconstruction families without predetermining the agent's eventual toolkit. There is no verified universal winner across all tasks, datasets and compute budgets.

## 4. Input fairness and output contract

MapAnything accepts optional calibration, poses and depth; its COLMAP-input demo defaults to calibrated posed MVS. Pi3X also accepts poses/intrinsics/depth. DA3 accepts optional known poses. These capabilities are useful for a second calibrated track but are not admissible GT inputs in the RGB-only track. [MapAnything input API](https://github.com/facebookresearch/map-anything), [Pi3X conditioning](https://github.com/yyfz/Pi3), [DA3 API](https://github.com/ByteDance-Seed/Depth-Anything-3/blob/main/docs/API.md)

Proposed comparison contract: same image identities and candidate-view availability; declared resolution and preprocessing; explicit camera convention; fixed geometric fusion/evaluation settings or a clearly labeled native complete-pipeline track; budgets including failed runs, model calls, GPU time and agent cost. Do not equate equal call counts across differently sized models. Estimate scale/alignment only in the evaluator according to the benchmark; keep GT and evaluation feedback unavailable to the agent on held-out scenes. For known-intrinsics and known-pose tracks, list supplied inputs separately and do not combine those scores with RGB-only scores.

Do not confuse replacing predicted poses with GT **after** inference with conditioning a network on GT poses **during** inference: these are distinct diagnostic interventions.

## 5. Is an optimization/SfM category missing?

Yes, if claiming improvement over modern reconstruction systems broadly. **MP-SfM** (CVPR 2025) integrates monocular depth/normal priors into SfM and releases code/models. **GLUEMAP** (May 2026 preprint) combines global SfM with feed-forward reconstruction; its official implementation supports Pi3, Pi3X, VGGT and MapAnything backbones. They are relevant hybrid camera/sparse-structure baselines. [MP-SfM official code](https://github.com/cvg/mpsfm), [paper](https://arxiv.org/abs/2504.20040), [GLUEMAP paper](https://arxiv.org/abs/2605.26103), [official code](https://github.com/colmap/gluemap)

They should not be presented as complete dense-geometry competitors merely because they output a sparse reconstruction. To include them in the main dense task, specify and evaluate an actual dense backend and its cost, e.g. predicted SfM cameras followed by a supported posed reconstruction backend. Both the agent and comparator must receive the same sensor information. A hybrid baseline is particularly important if observed development-set errors implicate pose estimation; it need not dictate the entire research problem. For a surface-quality claim, also evaluate an appropriate established MVS complete system under the same unknown/known camera track. Select its exact implementation after benchmark and output requirements are settled.

## 6. Closest real agentic work, with scope boundaries

| Work | Verified relation to this task |
|---|---|
| **AREA3D**, 2025 preprint / CVPR 2026 | Active reconstruction agent using feed-forward geometry and VLM view guidance, evaluated on Replica and OmniObject3D. Official code exists, with current instructions for VGGT-Ω and InternVL2.5. It acquires/selects new viewpoints, so it is not automatically a fixed-input reconstruction-recovery baseline. [CVF paper](https://openaccess.thecvf.com/content/CVPR2026/html/Xu_AREA3D_Active_Reconstruction_Agent_with_Unified_Feed-Forward_3D_Perception_and_CVPR_2026_paper.html), [official code](https://github.com/TianlingXu/AREA3D) |
| **GRASE**, October 15, 2026 preprint | Closely related observation-driven repair: revises structural shell, asset placement and geometry by checking rendered multi-view agreement. Its output is editable real-to-sim scenes rather than raw camera-plus-dense mapping. Paper verified; runnable official code not verified here. [Primary paper](https://arxiv.org/abs/2610.21498) |
| **CoDimRecon**, September 28, 2026 preprint | Multi-view RGB agentic reconstruction with geometric priors, generated object meshes and behavioral repair of simulation assets. Relevant to tool development and feedback, but its simulation-ready compositional task differs from generic dense mapping. Runnable code not verified here. [Primary paper](https://arxiv.org/abs/2609.36024) |
| **AstraLOD3**, September 23, 2026 preprint | Dynamically executes Python/Blender procedures for building reconstruction, but receives calibrated cameras and a filtered sparse SfM cloud. It cannot be a same-input RGB-only comparator without changing the experiment. [Primary paper](https://arxiv.org/abs/2609.28061) |

This evidence does **not** establish that fixed-input agentic camera/dense-geometry recovery has no prior art. It establishes nearby tasks that must be distinguished in any novelty claim.

## 7. Develop tools only after measuring failures

Run complete baselines on a development split first. Inspect errors and select/develop tools against repeated, measurable failure mechanisms; do not freeze a guessed menu or encode scene-specific `if/else` repair rules. Keep the task and evaluation contract stable while the toolkit evolves.

Once a viable toolkit exists, isolate the agent's contribution with the **same tools and budget**: a fixed workflow, one-shot agent plan, and iterative agent with diagnostic feedback. Compare these to the complete reconstruction baselines above. Then evaluate on held-out scenes with frozen agent/tool versions and no GT feedback. Tool development and test protocol freezing are different stages; this design does not require freezing tools before the failure study.
