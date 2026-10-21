# Benchmark extension evidence — 2026-10-19

**Recommendation: add exactly SMERF and Tanks & Temples (T&T)** to PhotoTourism/IMC and ETH3D, prioritizing SMERF first. This is a practical recommendation for a short training-free SfM-recovery study, not evidence that a VLM will improve either dataset. SMERF has a directly reusable low-overlap protocol; T&T needs an explicitly published custom subset unless the original MP-SfM image lists are obtained.

## Recent-method evidence

- **MP-SfM, CVPR 2025:** evaluates ETH3D, RealEstate10k, seven T&T scenes and four SMERF scenes. For the full-scene T&T/SMERF setting, reference camera poses come from COLMAP on 10–100 times more images; the paper says five image collections per scene and overlap bucket. These are reconstruction-derived camera references, not independently measured camera ground truth. [Paper, §4](https://openaccess.thecvf.com/content/CVPR2025/papers/Pataki_MP-SfM_Monocular_Surface_Priors_for_Robust_Structure-from-Motion_CVPR_2025_paper.pdf)
- **GLUEMAP, CVPR 2026:** evaluates ETH3D, IMC2021, CO3Dv2, SMERF and LaMAR; T&T is not among its five stated datasets. It explicitly follows MP-SfM's SMERF setup. It reports multi-room symmetry/collapse, low overlap, and scale drift on SMERF. This makes SMERF particularly relevant to recovery beyond landmark photo bags. [Paper, §§4.3, 11](https://arxiv.org/html/2605.26103v2)
- **Doppelgangers++, CVPR 2025:** its disambiguation tests use DG, VisymScenes and Mapillary; reconstruction demonstrations include MegaScenes. Its geotag-based SfM validation aligns registered Mapillary cameras with GPS using RANSAC and measures inlier ratio. This is useful ambiguity evidence, but not a drop-in precise camera-pose benchmark for T&T/ScanNet/DTU/CO3D. [Paper, §§3–4](https://openaccess.thecvf.com/content/CVPR2025/papers/Xiangli_Doppelgangers_Improved_Visual_Disambiguation_with_Geometric_3D_Features_CVPR_2025_paper.pdf), [official project](https://doppelgangers25.github.io/doppelgangers_plusplus/)

## What is actually reusable from MP-SfM

Inspected official repository revision recorded in the local run manifest via GitHub's recursive tree and raw files. The benchmark CLI accepts **only `eth3d` and `smerf`**. Preparation scripts, dataset loaders, aggregation scripts and checked-in testsets exist for those two. No T&T loader/preparation/testset directory was present in this inspected tree. Do not claim the released code exposes an exact T&T reproduction. [Benchmark entry point](https://github.com/cvg/mpsfm/blob/main/scripts/benchmark.py), [testset directory](https://github.com/cvg/mpsfm/tree/main/local/testsets)

SMERF scenes are `alameda`, `berlin`, `london`, `nyc`. The preparation script downloads original undistorted images and sparse references, uses `images_2`, rescales calibration, and writes a COLMAP reconstruction. [Preparation script](https://github.com/cvg/mpsfm/blob/main/mpsfm/data_proc/prepare/smerf.py)

The following counts were computed directly from all 16 checked-in YAML files, not inferred from the paper:

| Mode | Bags per scene | Total bags | Images per bag, across four scenes |
|---|---:|---:|---:|
| minimal | 10 | 40 | 29–44 |
| low | 5 | 20 | 55–106 |
| medium | 5 | 20 | 75–142 |
| high | 5 | 20 | 112–265 |

[Pinned SMERF YAMLs](https://github.com/cvg/mpsfm/tree/main/local/testsets/smerf). **The current minimal release has ten bags per scene, whereas the paper says five.** For a bounded main experiment, use the released **low + high** settings: 4 scenes × 5 bags × 2 settings = **40 episodes**, spanning 55–106 and 112–265 images per bag. This contrasts difficult overlap with a denser control while avoiding the minimal-release count discrepancy. These are not 25-image bags, so retain their original sizes rather than silently resampling. Minimal can remain a later stress test.

Official command templates: `python scripts/benchmark.py -d smerf -m low` and the same command with `-m high`; aggregate with `python scripts/aggregate_experiments/smerf.py -m low high`. The CLI also permits a scene and testset ID. These commands were inspected, not executed or performance-validated. [README](https://github.com/cvg/mpsfm/blob/main/README.md)

## T&T protocol and limits

Use the seven public training scenes as **our T&T sparse-camera protocol**. Proposed bounded choice: at most 100 frames per scene, sampled uniformly and deterministically over the ordered reference-covered frame list, saving exact image manifests. This produces seven scene episodes; it does not replicate MP-SfM’s overlap-stratified bags. The official download page supplies training image sets, COLMAP camera `.log` files, COLMAP reconstructions, scan alignment, and laser geometry. The camera files are explicitly results of COLMAP, distinct from the laser-scanned geometry. Image sets are extracted at 1 fps from 29.97 fps video; exact frame-to-pose mapping must be checked. [Official download documentation](https://www.tanksandtemples.org/download/)

Suggested implementation policy: freeze frame lists before comparing methods; use only reference-covered images; preserve identical intrinsics and image preprocessing across methods; report relative-pose AUC plus registration/completeness and recovered/incorrectly changed bags. Label pose results **agreement with dense COLMAP reference**. Verify pose convention, scale/alignment and image names before measuring. Do not rerun COLMAP on each evaluated sparse bag to define its reference. T&T adds realistic video-derived scenes and object/building-centric trajectories; this is a complementary evaluation distribution, not a guaranteed VLM advantage. These are proposed experimental choices, not claims of an existing official camera-recovery benchmark.

## Why not the other candidates for this quick extension

| Candidate | Added value and reference caveat | Decision |
|---|---|---|
| ScanNet | Indoor RGB-D sequences with released calibration/poses; reconstruction uses BundleFusion, so poses are estimated RGB-D trajectories. Requires explicit frame extraction/subsampling and sequence protocol. [Official repository](https://github.com/ScanNet/ScanNet) | Useful later for sequential indoor failures; less immediately reusable than MP-SfM SMERF. |
| ScanNet++ | High-quality indoor scans, DSLR and iPhone captures. DSLR COLMAP cameras are aligned to the scans; iPhone supplies raw/aligned ARKit poses and a COLMAP model. These are different pose products, not interchangeable independent GT. [Official documentation](https://scannetpp.mlsg.cit.tum.de/scannetpp/documentation) | More preprocessing/protocol decisions; defer for speed. |
| DTU | Controlled robot acquisition and calibration targets provide a different calibration regime. [Official dataset](https://roboimagedata.compute.dtu.dk/) | Good controlled geometry sanity test; weaker fit to multi-room/large-scene recovery and no verified direct protocol in the three methods checked. |
| CO3D/CO3Dv2 | Object-centric diversity; GLUEMAP uses CO3Dv2. CO3D cameras are COLMAP estimates, with poor reconstructions filtered; backgrounds often anchor tracking. [Dataset paper, §3](https://arxiv.org/html/2109.00512) | Valid alternative for object-centric scope, but less aligned with multi-room recovery than SMERF; not an independent camera-GT fix. |

**Reporting constraint:** adding two datasets increases coverage, not statistical independence or proof of semantic reasoning. Freeze selection, retain negative results, report scene-level variation and the denominator of failed/recovered cases, and describe all pretraining overlap information that can actually be verified.
