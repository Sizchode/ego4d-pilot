# Partial Snapshot 10 Findings

**Status.** Snapshot 10 is a non-final immutable copy taken while VGGT, π³-X, and MapAnything were still running. It contains 329 completed, validated rows: DA3 103, VGGT 80, π³-X 70, and MapAnything 76. All copied-result, combined-metric, and six comparison-output hashes were verified locally against their manifests.

## Common populations

| Benchmark | Four-model common scenes | Change from snapshot 09 |
|---|---:|---:|
| 7Scenes | 7 | +4 |
| DTU | 2 | +2, first common population |
| ETH3D | 11 | unchanged |
| HiRoom | 30 | unchanged |
| ScanNet++ | 20 | unchanged |

ETH3D, HiRoom, and ScanNet++ comparison tables are byte-for-byte unchanged from snapshot 09. All raw model/scene/metric values for the three overlapping 7Scenes scenes are also unchanged. The new evidence therefore comes from four additional 7Scenes sequences and DTU scans 1 and 4 rather than a changed evaluator or rerun.

## 7Scenes: pose selection and geometry selection disagree

Across all seven scenes, DA3 is the best fixed model for both pose AUC@3° and F-score. A per-scene backbone oracle nevertheless improves mean pose AUC@3° by 0.0201 and mean F-score by 0.0494; maximum gains are 0.0750 and 0.1705, respectively.

| Scene | Best pose model | Best geometry model | Best F-score | Geometry gain over fixed DA3 |
|---|---|---|---:|---:|
| `chess` | DA3 | π³-X | 0.4790 | +0.0829 |
| `fire` | VGGT | DA3 | 0.8049 | 0 |
| `heads` | DA3 | π³-X | 0.8761 | +0.0410 |
| `office` | MapAnything | MapAnything | 0.6451 | +0.1705 |
| `pumpkin` | VGGT | VGGT | 0.1833 | +0.0510 |
| `redkitchen` | π³-X | DA3 | 0.6238 | 0 |
| `stairs` | π³-X | DA3 | 0.1952 | 0 |

Only `office` and `pumpkin` select the same non-DA3 model for pose and geometry. `redkitchen` and `stairs` are direct counterexamples to choosing a recovery backbone from pose alone: π³-X has the best pose score, while DA3 gives the best final geometry. `chess` and `heads` show the reverse pattern, where DA3 has the best pose but π³-X gives better geometry. The recovery target must therefore be downstream reconstruction benefit, not camera accuracy in isolation.

The low F-scores on `pumpkin` and `stairs` across every model may represent shared hard cases rather than recoverable routing errors; an allowed intervention must demonstrate official-metric improvement before either is counted as agent headroom.

The complete archived 7Scenes outputs were subsequently audited and inspected. Across most scenes, different backbones rank the same input views similarly by incident camera error, especially `office` and `pumpkin`, but the endpoint disagreement remains: shared difficult views do not determine whether pose, depth, visibility, or fusion is responsible. The detailed evidence, limitations, and component-intervention hypotheses are in [the 7Scenes visual review](7scenes-snapshot-10-visual-review.md).

## DTU: the first two scans show an accuracy–completeness tradeoff

DTU exposes only reconstruction accuracy, completeness, and their overall mean in this protocol. On scans 1 and 4:

- π³-X is the best fixed model for accuracy on both scans, with mean 0.881 cm.
- VGGT is the best fixed model for completeness on both scans, with mean 1.927 cm.
- Overall error is best with π³-X on scan 1 (1.710 cm) and VGGT on scan 4 (1.380 cm).
- Per-scene routing improves mean overall error by only 0.0118 cm over fixed π³-X on this two-scan population.
- MapAnything is substantially worse on both scans; its mean overall error is 7.330 cm.

Two scans are insufficient for a task claim. The immediate value of this population is protocol validation: the four backbones expose different accuracy/completeness tradeoffs under the same official evaluator, and the full 22-scan population will determine whether routing or repair headroom is material.

## Evidence

- Immutable snapshot: `artifacts/partial-results-snapshots/snapshot-10/`
- Cross-model outputs: `artifacts/partial-results-snapshots/snapshot-10-cross-model/`
- Independent local hash/population/stability audit: `artifacts/partial-results-snapshots/snapshot-10-cross-model/local-audit.json`
- Complete 7Scenes visual/metric review: `research/2026-10-19-3dv-agent/7scenes-snapshot-10-visual-review.md`
- 7Scenes archive/contact-sheet/metric audit: `artifacts/partial-snapshot-10-review/local-audit.json`
- Snapshot manifest SHA-256: recorded in the local run manifest
- Combined metrics SHA-256: recorded in the local run manifest

No composite score or failure label is introduced. All metrics remain separate, and this snapshot will be superseded by the audited 103/103 final population.
