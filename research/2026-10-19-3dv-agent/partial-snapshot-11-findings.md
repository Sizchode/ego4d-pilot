# Partial Snapshot 11 Findings

**Status.** Snapshot 11 is a non-final immutable copy taken while VGGT, π³-X, and MapAnything were still running. It freezes 340 completed, validated rows: DA3 103, VGGT 84, π³-X 73, and MapAnything 80. The only common benchmark population that changed from snapshot 10 is DTU, which grew from two to five scans.

## Independent audit

A reusable local auditor reloaded all four copied `results.jsonl` streams, verified their hashes, row counts, state counts, dataset counts, identities, and ordering, and matched their completed cases to `completed_metrics.jsonl`. It then independently recomputed all six cross-model CSV outputs and required byte-for-byte equality with the downloaded results.

The audit passed. The snapshot contains no execution failures. Every model's snapshot-10 result stream is an exact prefix of its snapshot-11 stream. All comparison rows for 7Scenes, ETH3D, HiRoom, and ScanNet++ are byte-for-byte unchanged. The audit is recorded in `artifacts/partial-results-snapshots/snapshot-11-cross-model/local-audit.json`.

## Common populations

| Benchmark | Four-model common scenes | Change from snapshot 10 |
|---|---:|---:|
| 7Scenes | 7 | unchanged |
| DTU | 5 | +3 |
| ETH3D | 11 | unchanged |
| HiRoom | 30 | unchanged |
| ScanNet++ | 20 | unchanged |

The five DTU scans are `scan1`, `scan4`, `scan9`, `scan10`, and `scan11`. DTU64 has not entered the four-model common population because the incomplete baselines are still processing DTU.

## DTU: the first five scans reverse the two-scan fixed-model conclusion

The two-scan snapshot favored π³-X for mean overall error by a small margin. Adding scans 9–11 changes the best fixed model to DA3 because DA3 wins all three new scans. This is direct evidence that a two-scan partial population was too unstable for method selection.

| Scan | Best accuracy | Best completeness | Best overall | Best overall error (cm) |
|---|---|---|---|---:|
| `scan1` | π³-X | VGGT | π³-X | 1.7101 |
| `scan4` | π³-X | VGGT | VGGT | 1.3800 |
| `scan9` | DA3 | DA3 | DA3 | 0.9887 |
| `scan10` | DA3 | DA3 | DA3 | 0.6429 |
| `scan11` | DA3 | DA3 | DA3 | 1.0322 |

Across these five scans:

| Endpoint | Best fixed model | Best fixed mean (cm) | Per-scan oracle mean (cm) | Mean oracle gain (cm) | Positive-gain scans |
|---|---|---:|---:|---:|---:|
| Accuracy | DA3 | 0.6880 | 0.5964 | 0.0916 | 2/5 |
| Completeness | VGGT | 1.9559 | 1.5924 | 0.3635 | 3/5 |
| Overall | DA3 | 1.4779 | 1.1508 | 0.3271 | 2/5 |

For overall error, routing away from fixed DA3 helps on `scan1` by 0.1174 cm and on `scan4` by 1.5182 cm; it gives no oracle gain on the other three scans because DA3 already wins. MapAnything does not win any endpoint on this population and has a mean overall error of 7.9454 cm.

The accuracy/completeness tradeoff remains visible: DA3 is the best fixed accuracy model, while VGGT is the best fixed completeness model. The best overall model also changes by scan. This supports evaluating action selection against the actual downstream endpoint rather than a single proxy, but it does not yet establish that a deployable diagnostic can predict the winning action.

## Current interpretation

Snapshot 11 supplies two useful controls for later failure mining:

1. **Partial-population conclusions can reverse.** The fixed overall winner changed from π³-X on two scans to DA3 on five scans. No method ranking from this subset should be treated as final before all 22 DTU scans are common.
2. **Oracle headroom is concentrated.** Most of the overall gain comes from `scan4`; three scans need no routing relative to fixed DA3. A controller must identify the exceptional cases without degrading the majority, under the same action and compute budget.

This is still an oracle analysis based on official ground truth. It provides candidate cases for later evidence and intervention review, not test-time policy inputs or failure labels.

## Pose winner versus geometry winner across benchmarks

For the four benchmarks that report both camera pose and geometry, a separate analysis compares the best backbone under pose AUC with the best backbone under reconstruction F-score on the same scene. It never combines metric values across datasets. Pose AUC@30 and geometry F-score select the same model at the following rates:

| Benchmark | Shared scenes | Same winner | Different winner | Different + positive routing headroom | Max F-score headroom | Median four-model rank ρ |
|---|---:|---:|---:|---:|---:|---:|
| 7Scenes | 7 | 3 | 4 | 3 | 0.0829 | 0.2 |
| ETH3D | 11 | 6 | 5 | 2 | 0.0226 | 0.8 |
| HiRoom | 30 | 24 | 6 | 4 | 0.1226 | 0.8 |
| ScanNet++ | 20 | 12 | 8 | 8 | 0.0587 | 0.8 |

The distinction between rank agreement and action selection matters. A median ρ of 0.8 means grossly poor models often rank poorly on both endpoints, but it does not identify the best reconstruction: 23 of the 68 scene decisions have different top models. Six of those disagreements have zero routing headroom because the best fixed geometry model already wins; they are pose-gate counterexamples, not useful recovery cases. The other 17 have positive per-scene oracle headroom without introducing a gain threshold. On ScanNet++, DA3 has the best pose AUC@30 in all 20 scenes, while π³-X has the best geometry in seven and VGGT in one. AUC@3 produces the same qualitative result: winner overlap is 2/7, 7/11, 25/30, and 12/20 respectively.

This is evidence against a pose-only recovery gate. It does not identify whether depth, confidence calibration, visibility, or fusion caused the geometry difference, and it does not show that a VLM can predict the correct action from test-time evidence. Reproducible outputs are stored under `artifacts/partial-snapshot-11-review/pose30-geometry-winner-agreement/`, `pose03-geometry-winner-agreement/`, and `pose30-geometry-actionable-disagreements/`.

The first test-time-evidence audit also found no adequate scalar routing signal. Four DA3 state variables and 72–78 valid global RGB summaries were correlated with F-score oracle gain separately by benchmark and corrected within each endpoint family. No association reached BH q ≤ 0.05. Fusion retention was negatively correlated with gain on all four datasets, but its q-values ranged from 0.242 to 0.797; the strongest image correlations were dataset-specific and also did not survive correction. This supports moving from global summaries to localized view/region/factor evidence, while leaving open whether the needed representation should be numerical, learned visual, or language-based. See [the routing-evidence audit](snapshot-11-routing-evidence.md).

## DTU visual and per-view review

The verified 20-case DTU archive was also reviewed with model-aligned depth/confidence previews and evaluator-produced camera-error matrices. A reproducible analysis compares each view's incident-pair P95 error across backbones. Unlike several 7Scenes sequences, the five DTU scans do not show a stable shared hard-view ordering: median cross-model Spearman correlations range from 0.051 to 0.427 for rotation and 0.044 to 0.172 for translation. At case level, VGGT has the lowest GT camera P95 on all five scans but wins overall geometry only on `scan4`; even privileged camera error therefore selects the geometry winner on only 1/5 scans.

The review supplies concrete controls for later interventions. `scan4` is the main positive routing case; `scan9` shows that a sharp depth preview can accompany poor official geometry; `scan10` shows that model-native confidence appearance is not calibrated correctness; and `scan11` is a low-headroom do-nothing case. These observations separate pose, depth/confidence, and fusion as candidate factors but do not assign a cause or demonstrate a VLM advantage. See [the audited DTU review](dtu-snapshot-11-visual-review.md).

## Evidence

- immutable snapshot: `artifacts/partial-results-snapshots/snapshot-11/`
- cross-model comparison: `artifacts/partial-results-snapshots/snapshot-11-cross-model/`
- independent local audit: `artifacts/partial-results-snapshots/snapshot-11-cross-model/local-audit.json`
- DTU five-scan visual review: `dtu-snapshot-11-visual-review.md`
- DTU review audit: `artifacts/partial-snapshot-11-review/review-output-audit.json`
- snapshot manifest SHA-256: recorded in the local run manifest
- snapshot metrics SHA-256: recorded in the local run manifest

No composite score or heuristic failure threshold is introduced. This snapshot will be superseded by the audited 103/103 final population.
