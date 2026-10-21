# Snapshot 11: DTU five-scan visual and metric review

**Status.** This is a non-final review of the five DTU scans shared by DA3, VGGT, π³-X, and MapAnything in immutable snapshot 11. It uses official metrics and evaluator-produced camera errors to select and understand cases while the complete 22-scan population is still running. It does not assign failure labels, define an agent action, or support a final model ranking.

## Evidence and audit

The review archive contains 20 cases (5 scans × 4 models), 160 files, and 49 evaluated views per case. The downloaded tar matches its remote SHA-256 `04a2af8f777f55437d3ba4217d59f44ea3e62fdef7bf0cb36d8061c8a48d6f1fa`; every archived file was checked against its recorded size and SHA-256. Five model-aligned contact sheets were regenerated from these verified files and inspected manually.

The derived tables contain 20 case rows and 980 per-view rows. All cases are completed, validation-passed runs. The 60 official DTU values in the case table—accuracy, completeness, and overall error—are exactly equal to the immutable snapshot-11 metric stream. A second audit rechecked the complete per-view population, contact-sheet hashes and dimensions, and byte-for-byte reproduction of the view-profile analysis. Both audits passed:

- archive transfer audit: `artifacts/partial-snapshot-11-review/local-audit.json`;
- derived-output audit: `artifacts/partial-snapshot-11-review/review-output-audit.json`.

Camera errors are computed against DTU ground truth and are therefore **offline failure-analysis evidence, never test-time controller inputs**. Preview depth and confidence panels use each model/view's native visualization range; camera-error plots also auto-scale independently. Apparent sharpness, confidence color, and heatmap intensity are not numerically comparable across models.

## Main result: DTU does not show a stable shared hard-view ordering

Each per-view score is the P95 camera error among that view's 48 incident camera pairs. For each scan, the table reports the median Spearman correlation across the six backbone pairs. A high value would mean different backbones tend to identify the same views as difficult. The observed medians are mostly weak, and every scan/quantity includes at least one much lower or negative model-pair correlation.

| Scan | Rotation median ρ | Rotation range | Translation median ρ | Translation range |
|---|---:|---:|---:|---:|
| `scan1` | 0.055 | −0.248 to 0.358 | 0.138 | −0.433 to 0.520 |
| `scan4` | 0.374 | −0.042 to 0.511 | 0.172 | −0.050 to 0.344 |
| `scan9` | 0.216 | 0.003 to 0.479 | 0.044 | −0.564 to 0.186 |
| `scan10` | 0.427 | 0.189 to 0.542 | 0.165 | −0.108 to 0.324 |
| `scan11` | 0.051 | −0.267 to 0.476 | 0.098 | −0.322 to 0.463 |

The contact sheets contain block structure in several camera-error matrices, but the numerical comparison shows that these blocks do not imply a common per-view ranking across all models. This differs from the stronger shared-view ordering observed on several 7Scenes sequences. A controller should not assume a backbone-independent list of bad DTU views.

## Final geometry cannot be selected from preview sharpness, native confidence, or camera error alone

Even the GT-based case-level camera P95 is insufficient for selecting the best reconstruction. VGGT has the lowest rotation and translation P95 on all five scans, but it has the best overall geometry only on `scan4`. Thus the winner overlaps on **1/5 scans** for either camera quantity. Across all four models, camera-error and geometry rankings still have a median Spearman ρ of 0.8 because the grossly inaccurate models tend to be poor on both. That coarse association does not resolve the top-model decision needed by a recovery controller.

| Scan | Best overall | Overall error (cm) | Relevant control |
|---|---|---:|---|
| `scan1` | π³-X | 1.7101 | VGGT has best completeness; the endpoint changes with the objective |
| `scan4` | VGGT | 1.3800 | Largest routing gain over fixed DA3: 1.5182 cm |
| `scan9` | DA3 | 0.9887 | π³-X looks visually sharp but reaches 4.8660 cm overall |
| `scan10` | DA3 | 0.6429 | MapAnything confidence appears high/uniform in sampled views but reaches 8.9272 cm overall |
| `scan11` | DA3 | 1.0322 | π³-X is nearly tied at 1.0672 cm; an unnecessary action has little upside |

The purpose of these examples is not to rank visual quality. They rule out three tempting shortcuts:

1. **Sharp depth boundaries are not sufficient evidence of metric reconstruction quality.** π³-X preserves a crisp silhouette and architectural detail on `scan9`, yet its official overall error is almost five times DA3's. Scale, calibration, camera state, and fusion can dominate a visually plausible preview.
2. **Model-native confidence is not calibrated correctness.** On `scan10`, MapAnything's sampled confidence panels do not visually announce the very large official geometry error. Confidence values are model-specific and cannot be used as a shared action score without calibration.
3. **Low camera error does not uniquely determine the geometry winner.** VGGT has the lowest rotation/translation P95 on several scans, including `scan9`, but DA3 produces the best final geometry there. The pose estimate and depth/fusion state must be evaluated as separate intervention factors.

## Scan-level evidence and the experiment each case supports

### `scan4`: positive routing case

`scan4` is the clearest available routing opportunity. DA3's overall error is 2.8982 cm, while VGGT reaches 1.3800 cm and π³-X 1.4037 cm. π³-X has the best accuracy; VGGT has the best completeness and overall result. The object includes thin protrusions and large view-dependent silhouette changes, but the images alone do not identify the winning factor.

This case should enter controlled camera-only and depth/fusion-only interventions. It is evidence that another backbone contains useful information, not evidence that a VLM can recognize or recover it.

### `scan9`: sharp-preview counterexample

DA3 wins accuracy, completeness, and overall error. π³-X's sampled depth remains visually detailed around the roof and façade, yet its overall error is 4.8660 cm versus DA3's 0.9887 cm. The failure cannot be diagnosed by choosing the visually sharpest depth map. A factor intervention is needed to determine whether the loss follows scale/calibration, camera state, depth, or fusion.

### `scan10`: confidence counterexample

DA3 again wins every official endpoint. MapAnything reaches 8.9272 cm overall despite sampled confidence panels that appear comparatively uniform. Since confidence colormaps and ranges are native to each model, this case directly warns against treating confidence display intensity as a cross-model correctness signal. It motivates measured confidence calibration, not a semantic interpretation claim.

### `scan1`: endpoint-dependent action

π³-X has the best accuracy and overall error, while VGGT has the best completeness. The difference between π³-X and VGGT overall is only 0.100 cm, and both improve over DA3 modestly. A recovery policy must state which official endpoint it optimizes; “better reconstruction” is not a unique action target.

### `scan11`: do-nothing control

DA3 and π³-X are close—1.0322 versus 1.0672 cm overall—while the other two models are worse. This is a useful no-action or low-headroom control. A controller that frequently intervenes on such cases can erase the concentrated gains available on `scan4`.

## What this review establishes

For these five scans, the evidence establishes that:

1. per-view GT camera-error rankings are not stable across the four backbones;
2. preview sharpness and model-native confidence do not select the best official geometry;
3. the largest routing opportunity is concentrated in `scan4`, while `scan11` provides a low-headroom control; and
4. pose, depth/confidence, and fusion must remain separate candidate repair factors.

It does **not** establish the cause of any failure, a GT-free diagnosis, or a VLM advantage. The next evidence must come from controlled component interventions and then from policy comparisons with identical tools and compute: fixed numerical rules, search/oracle, one-shot VLM, and sequential VLM decisions. Full-DTU conclusions remain deferred until all 22 scans are present in the four-model common population.

## Artifacts

- verified archive: `artifacts/partial-snapshot-11-review/dtu-five/`
- model-aligned contact sheets: `artifacts/partial-snapshot-11-review/dtu-contact-sheets/`
- case metrics: `artifacts/partial-snapshot-11-review/dtu-case-metrics.csv`
- per-view metrics: `artifacts/partial-snapshot-11-review/dtu-view-metrics.csv`
- reproducible view-profile analysis: `artifacts/partial-snapshot-11-review/dtu-view-profile-agreement/`
- rotation-P95 versus geometry ranking: `artifacts/partial-snapshot-11-review/dtu-camera-geometry-rank-rotation/`
- translation-P95 versus geometry ranking: `artifacts/partial-snapshot-11-review/dtu-camera-geometry-rank-translation/`
- archive transfer audit: `artifacts/partial-snapshot-11-review/local-audit.json`
- derived-output audit: `artifacts/partial-snapshot-11-review/review-output-audit.json`
