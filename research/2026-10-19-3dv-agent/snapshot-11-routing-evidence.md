# Snapshot 11: can simple test-time evidence predict routing gain?

**Question.** DA3 is the best fixed F-score model on the complete 7Scenes, ETH3D, HiRoom, and ScanNet++ populations. Per-scene oracle routing to one of the other three backbones improves some cases. Can simple DA3 state variables or global RGB statistics predict the size of that improvement without inspecting ground truth?

**Answer under the current evidence.** No tested scalar survives within-benchmark multiple-testing correction. This rules out several simple one-variable gates; it does not establish that a VLM is necessary or that no multivariate numerical method can work.

## Protocol

The target is the official per-scene F-score gain of the four-backbone oracle over the best fixed backbone, computed only for offline analysis. The evidence contains no GT:

- four DA3 reconstruction-state scalars: fusion retention, projectable ratio, cross-view absolute-relative-depth P95, and round-trip P95;
- global input-image summaries of gradients, luminance, entropy, and saturation across views.

Spearman correlations are computed separately for each benchmark. Benjamini–Hochberg correction is applied within each benchmark/endpoint family: four tests for reconstruction state and 72–78 valid tests for image summaries. No correlation is turned into a threshold or failure label.

## Reconstruction-state evidence

| Benchmark | Largest observed |ρ| | Field | Two-sided p | BH q |
|---|---:|---|---:|---:|
| 7Scenes | 0.704 | `fusion.retained_ratio` | 0.077 | 0.309 |
| ETH3D | 0.580 | `fusion.retained_ratio` | 0.062 | 0.246 |
| HiRoom | 0.309 | `diagnostic.roundtrip_pixels.p95` | 0.097 | 0.242 |
| ScanNet++ | 0.280 | `diagnostic.projectable_ratio` | 0.232 | 0.797 |

Fusion retention has a negative correlation with routing gain on all four datasets: −0.704, −0.580, −0.289, and −0.191. Lower retention is therefore a plausible warning signal, but none of the four estimates survives correction. The other diagnostic signs vary by benchmark. In particular, large cross-view residual is not a stable routing signal, consistent with the earlier visibility-induced false positives.

## Global RGB evidence

| Benchmark | Smallest raw p field | ρ | Raw p | BH q | Tests |
|---|---|---:|---:|---:|---:|
| 7Scenes | maximum luminance entropy | 0.889 | 0.0073 | 0.528 | 72 |
| ETH3D | minimum all-channel ceiling saturation | 0.593 | 0.0543 | 0.948 | 78 |
| HiRoom | median any-channel floor saturation | 0.565 | 0.00115 | 0.0836 | 73 |
| ScanNet++ | P95 of per-view luminance P99 | 0.349 | 0.132 | 0.985 | 75 |

The attractive raw correlations on 7Scenes and HiRoom do not survive correction over the predeclared image family. They also select different visual quantities in different datasets. Global blur, brightness, entropy, and saturation summaries therefore do not yet supply a reproducible routing rule.

## Consequence for the recovery task

The current evidence narrows the missing signal. A useful controller likely needs **localized, structured evidence**—which views, regions, surfaces, or visibility relations disagree, and whether the downstream loss follows cameras, depth, confidence, or fusion. That evidence could be handled by a learned visual model, a VLM, or a stronger numerical representation; these tests do not choose among them.

The next causal step remains the 20-scene ScanNet++ factor decomposition. If camera/depth/fusion interventions explain the gains and numerical evidence predicts the responsible factor, an agent is unnecessary. A VLM becomes defensible only if image-conditioned reasoning improves action selection over those numerical baselines under the same tools and budget.

## Artifacts

- DA3 state versus all oracle endpoints: `artifacts/partial-snapshot-11-review/da3-state-vs-backbone-oracle/`
- four state fields versus F-score gain with BH correction: `artifacts/partial-snapshot-11-review/da3-state-vs-fscore-oracle-fdr/`
- global RGB summaries versus F-score gain with BH correction: `artifacts/partial-snapshot-11-review/da3-image-vs-fscore-oracle-fdr/`
- oracle source: `artifacts/partial-results-snapshots/snapshot-11-cross-model/oracle_scene_headroom.csv`

The four benchmark populations used here are already complete across all backbones; later DTU/DTU64 completion does not change these rows.
