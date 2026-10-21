# Partial Snapshot 09: Evidence Before Final Failure Mining

**Status:** non-final. This snapshot freezes only completed, validated prefixes of the four active baseline batches. It is evidence for directing the remaining analysis, not a final result table.

## Population and provenance

- Snapshot metric rows: **306**; SHA-256 recorded in the local run manifest.
- DA3: 103/103 scenes; VGGT: 71/103; π³-X: 64/103; MapAnything: 68/103.
- All four models have completed the same **30 HiRoom, 11 ETH3D, and 20 ScanNet++** scenes. The common 7Scenes prefix contains 3/7 scenes. DTU and DTU-64 are excluded from this partial comparison because all four models do not yet share a completed population.
- Every captured result row in this snapshot is completed and validated; failure rows are zero.
- Metrics remain separate. No composite score, success threshold, or failure label is introduced.

## Backbone-switching oracle

The oracle chooses the best of the four already-computed backbones for each scene using ground truth. It is an offline upper bound and cannot be used by a deployable policy.

| Dataset | Common scenes | Metric | Best fixed backbone | Fixed mean | Oracle mean | Mean gain | Scenes with gain | Maximum gain |
|---|---:|---|---|---:|---:|---:|---:|---:|
| 7Scenes* | 3 | Pose AUC@3° | DA3 | 0.4481 | 0.4485 | 0.0004 | 1/3 | 0.0011 |
| 7Scenes* | 3 | Geometry F-score | DA3 | 0.6787 | 0.7200 | 0.0413 | 2/3 | 0.0829 |
| ETH3D | 11 | Pose AUC@3° | DA3 | 0.4852 | 0.4979 | 0.0126 | 2/11 | 0.1161 |
| ETH3D | 11 | Geometry F-score | DA3 | 0.7848 | 0.7884 | 0.0036 | 2/11 | 0.0226 |
| HiRoom | 30 | Pose AUC@3° | DA3 | 0.8015 | 0.8693 | 0.0679 | 5/30 | 0.8788 |
| HiRoom | 30 | Geometry F-score | DA3 | 0.8580 | 0.9192 | 0.0612 | 8/30 | 0.8490 |
| ScanNet++ | 20 | Pose AUC@3° | DA3 | 0.8470 | 0.8470 | 0.0000 | 0/20 | 0.0000 |
| ScanNet++ | 20 | Geometry F-score | DA3 | 0.7833 | 0.7917 | 0.0084 | 8/20 | 0.0587 |

\* 7Scenes is incomplete and should not support a final claim.

## Exact extremal cases to review

| Dataset | Signal | Scene | Fixed backbone → oracle | Absolute gain |
|---|---|---|---|---:|
| HiRoom | Largest pose headroom | `20241230/828815/cam_sampled_04` | DA3 → π³-X | 0.8788 AUC@3° |
| HiRoom | Largest geometry headroom | `20241230/828785/cam_sampled_12` | DA3 → VGGT | 0.8490 F-score |
| ETH3D | Largest pose headroom | `kicker` | DA3 → VGGT | 0.1161 AUC@3° |
| ETH3D | Largest geometry headroom | `pipes` | DA3 → π³-X | 0.0226 F-score |
| ScanNet++ | Largest geometry headroom | `7831862f02` | DA3 → π³-X | 0.0587 F-score |
| 7Scenes* | Largest geometry headroom | `chess` | DA3 → π³-X | 0.0829 F-score |

The largest cross-model failure-rank disagreements are also localized rather than uniform: HiRoom `20241230/828815/cam_sampled_04` for pose; HiRoom `20241230/828763/cam_sampled_12` and `20241230/828785/cam_sampled_12` for geometry; ETH3D `terrains` for pose; and ScanNet++ `f3d64c30f8` for geometry.

## What this changes

1. **Backbone choice is a necessary baseline, especially on HiRoom.** A small number of scenes account for large oracle gains. Any recovery method must beat or complement same-budget backbone selection; otherwise the contribution is model routing.
2. **ScanNet++ separates pose from geometry.** DA3 is the best pose backbone on every one of the 20 scenes, while another backbone improves F-score on 8 scenes. On this benchmark, a pose-repair story is not supported by the current evidence. The useful error is downstream of pose or coupled to depth, intrinsics, confidence, visibility, or fusion.
3. **The 20-scene GT factor intervention is now mandatory.** Replacing extrinsics, then intrinsics, then depth under the same evaluator will test where ScanNet++ geometry headroom comes from. The earlier 14-scene intervention remains a pipeline check, not the final population.
4. **No agent action space follows yet.** The visual review must determine whether the extremal cases share image-understandable causes. If residuals already identify the correct numerical correction, the appropriate method is deterministic recovery rather than a VLM agent.

## Artifacts

- Immutable snapshot: `artifacts/partial-results-snapshots/snapshot-09/`
- Common-population analysis: `artifacts/partial-results-snapshots/snapshot-09-cross-model/`
