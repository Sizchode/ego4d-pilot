# Concrete VLM diagnosis cases for reconstruction

Research date: 2026-10-19. This synthesis separates demonstrated capabilities from proposed extensions. Detailed evidence is in identity-vlm-error-cases.md, reflection-vlm-error-cases.md, and dynamic-vlm-error-cases.md in this directory.

## Main conclusion

The useful distinction is not geometry versus semantics in general. A reconstruction can fit an incorrect correspondence, scene-rigidity assumption, or image-formation model. Visual context may help identify which assumption failed. However, specialized visual methods already use much of this evidence, and several recent works explicitly use VLMs. Neither a vacant research niche nor the necessity of an agent is established.

## 1. Duplicate object: association failure or camera failure?

Concrete candidate: two similar boxes are incorrectly linked through printed texture; alternatively, one real chair appears twice because the camera trajectory drifted or because object tracks fragmented. These require different corrections. An object-node merge does not repair a wrong camera trajectory; deleting all same-class objects destroys useful observations.

[Think While You Map / ThinkGraphs](https://arxiv.org/html/2606.31471v1) already uses images and geometry to merge fragmented chair/appliance tracks and preserve distinct stacked boxes. It uses known-pose RGB-D and repairs the object graph. It also reports an incorrect merge caused by misleading bounding-box containment.

Unverified extension: infer an instance identity conflict despite an erroneous camera state, localize the implicated feature correspondences, and repair the cameras with a fixed backend. Same-instance semantics alone must never force all pixels into one 3D point.

Necessary comparators: ThinkGraphs-style object critic, instance re-identification plus geometric constraints, [IRIS-SLAM](https://arxiv.org/html/2602.18709v1), [Doppelgangers++](https://arxiv.org/html/2412.05826v2), and [LFOE](https://openaccess.thecvf.com/content/CVPR2025/papers/Damblon_Learning_to_Filter_Outlier_Edges_in_Global_SfM_CVPR_2025_paper.pdf). These are related baselines with differing task interfaces, not interchangeable published numbers.

## 2. Mirror: virtual geometry or corrupted camera constraints?

A fixed planar mirror observing a static scene can produce consistent virtual geometry. Low residuals do not establish that this is a physical room behind the wall; the real camera poses may still be correct. A different failure is mixing direct and reflected observations into the same track. The latter can require camera re-estimation, whereas virtual geometry alone may only require a different surface interpretation.

[3D Visual Illusion Depth Estimation, NeurIPS 2025](https://wu-yuwei-bit.github.io/paper/neurips2025_yao.pdf) already gives a VLM RGB, monocular depth, and binocular disparity to assess regional reliability and guide fusion. [Detect Any Mirrors, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Xing_Detect_Any_Mirrors_Boosting_Learning_Reliability_on_Large-Scale_Unlabeled_Data_CVPR_2025_paper.html) already uses multimodal agents in mirror-label quality assessment. [Reflect3r](https://arxiv.org/abs/2509.20607) illustrates that reflections can be useful reconstruction evidence.

Unverified extension: link a material/image-formation explanation to the exact affected constraints and choose between surface correction and pose recovery. Compare against a specialist mirror detector plus an appropriate fixed solver, not just unmodified COLMAP. Static posters/screens are a negative control: classical multiview geometry can correctly recover their plane when a monocular prior hallucinates depth.

## 3. Door changes state: local motion or failed registration?

Concrete candidate: a cupboard door is closed in one segment and open in another. A static reconstruction binds its observations incorrectly and compromises either the door or the camera solution. The useful visual evidence includes the door/frame boundary, hinge relationship, temporal state, and whether neighboring static surfaces support the same camera correction.

This is not untouched: [FunREC, CVPR 2026](https://arxiv.org/html/2604.05621v2) uses a VLM to divide static/interaction segments and predict revolute/prismatic types, then estimates cameras and articulation from RGB-D with known intrinsics. [The Dynamic Prior / Dynapo](https://arxiv.org/html/2512.05398v1) already uses VLM video reasoning and SAM2 masks in pose/depth/4D optimization. [SCaR-3D](https://arxiv.org/html/2512.22830v1) covers change-aware reconstruction updates.

Unverified extension: distinguish true local state change from wrong global registration when the initial camera solution itself is unreliable, and revise the initial explanation using additional evidence. Strong baselines include articulated reconstruction, motion segmentation and fixed recovery, and one-shot VLM diagnosis with identical tools. This is a narrow hypothesis, not an established gap.

## What would establish an agent contribution?

A one-shot semantic label followed by a fixed tool call does not establish sequential reasoning. A concrete test would withhold decisive evidence initially: e.g., two similar boxes are ambiguous until another original frame shows their simultaneous presence and distinct surroundings. Compare a controller that requests relevant existing observations or localized geometric verification against one-shot VLM, learned non-language selection, and equal-cost generic inspection. All receive the same possible observations, tools, and budget. Evidence removal should increase uncertainty rather than leave an unsupported explanation unchanged.

The proposed measurable target is correct localization of harmful correspondences and better camera/geometry outcomes, not more plausible explanations. Source datasets need separate scrutiny: known-pose object-map results, depth-only illusion datasets, and place-recognition datasets do not independently demonstrate unknown-camera reconstruction recovery.

## Current prioritization

The most concrete remaining question is whether instance identity evidence can identify harmful correspondences when current camera geometry is wrong. It has a direct positive precursor in ThinkGraphs and a clear additional burden in camera recovery, but no demonstrated VLM advantage over strong instance/graph models yet. Mirror and articulated-motion examples explain the intuition well; direct prior coverage makes them weaker standalone novelty claims.
