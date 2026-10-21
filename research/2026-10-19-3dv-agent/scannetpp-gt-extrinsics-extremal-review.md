# ScanNet++ GT-Extrinsics Extremal Review

**Scope.** This review covers the exact per-backbone maximum and minimum F-score gains from replacing native extrinsics with GT extrinsics on the completed 20-scene ScanNet++ population. All exact ties would be retained; this population has eight selected model-scene rows and seven unique scenes. Selection uses the official evaluator and remains offline analysis, never a test-time policy input.

## Metric evidence

| Backbone | Extremum | Scene | Native-control F-score | GT-extrinsics F-score | Gain | Native pose AUC@3° | Rotation P95 | Translation P95 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| DA3 | maximum | `5f99900f09` | 0.7792 | 0.8303 | +0.0511 | 0.8767 | 1.44° | 1.95° |
| DA3 | minimum | `7bc286c1b6` | 0.7758 | 0.7736 | −0.0021 | 0.7314 | 1.65° | 3.66° |
| VGGT | maximum | `bcd2436daf` | 0.5720 | 0.7731 | +0.2011 | 0.4414 | 3.04° | 6.10° |
| VGGT | minimum | `cc5237fd77` | 0.8915 | 0.7954 | −0.0961 | 0.5590 | 2.00° | 6.62° |
| π³-X | maximum | `5f99900f09` | 0.4803 | 0.7764 | +0.2961 | 0.5877 | 2.34° | 2.88° |
| π³-X | minimum | `3e8bba0176` | 0.7960 | 0.7944 | −0.0016 | 0.7392 | 0.99° | 3.01° |
| MapAnything | maximum | `38d58a7a31` | 0.1651 | 0.5479 | +0.3828 | 0.1706 | 4.95° | 4.94° |
| MapAnything | minimum | `bde1e479ad` | 0.4407 | 0.4295 | −0.0112 | 0.3127 | 3.66° | 6.23° |

The native-control values come from the intervention code path. Archived official-native outputs independently match the intervention table's `official_native` values. The two paths differ slightly on a few cases; the largest 20-scene F-score discrepancy is 0.001140 for π³-X and is not folded into the reported intervention gain.

## What the extrema show

### DA3: native geometric consistency is more informative than GT pose AUC

The DA3 maximum-gain scene has a *higher* native pose AUC than its minimum-gain scene. A simple rule based on pose accuracy would therefore order this pair incorrectly. The native test-time diagnostics order it correctly:

| Scene | GT-camera F-score gain | Round-trip pixels P95 | Relative-depth inconsistency P95 |
|---|---:|---:|---:|
| `5f99900f09` | +0.0511 | 7.74 px | 0.1000 |
| `7bc286c1b6` | −0.0021 | 1.44 px | 0.0309 |

The preview of `5f99900f09` contains repeated chairs, broad walls, an opening/window, and several large view changes. The camera-error matrix contains structured bands across many frames. `7bc286c1b6` is a smaller bathroom/kitchen sequence; its DA3 errors are more localized. These visual descriptions do not establish cause, but they agree with the quantitative consistency diagnostics.

### Camera error alone does not identify a safe camera-only repair

MapAnything has poor camera metrics in both extrema, yet GT extrinsics produce opposite geometry outcomes: +0.3828 on `38d58a7a31` and −0.0112 on `bde1e479ad`. The latter is particularly important because it rejects the rule “bad camera metric implies replace cameras.” Its native depth, intrinsics, confidence, and fusion state were estimated jointly with the native cameras; replacing one component can reduce their compatibility.

VGGT gives a stronger version of the same counterexample. Both `bcd2436daf` and `cc5237fd77` have structured row/column bands in their camera-error matrices and similar translation-error P95, but the intervention changes F-score by +0.2011 and −0.0961. The former is a low-texture bedroom sequence with repeated bedding and large planar regions; the latter is a cluttered office with a bookcase, desk, and repeated revisits. This pair is a candidate for testing whether depth/fusion diagnostics explain the opposite outcomes. Image semantics should not be invoked until those component results are available.

π³-X is less ambiguous in this sample: its maximum-gain scene has worse pose accuracy and widespread error, while its minimum is only a 0.0016 loss. This pair is consistent with a deterministic diagnostic, but seven scenes are insufficient to define one.

## Implication for the recovery task

The actionable decision is not “is the camera estimate inaccurate?” It is:

> Will changing the camera state improve the final geometry while keeping the native depth, intrinsics, visibility, and fusion state compatible?

DA3 currently has test-time geometric diagnostics that correlate with this action's benefit. VGGT and MapAnything supply counterexamples where camera error alone is insufficient. The pending full component intervention must determine whether depth, intrinsics, or their interaction explains those residuals before any VLM or agent is justified.

## Evidence and limitations

- Selection manifest: `artifacts/scannetpp-gt-extrinsics-snapshot-09-extremal-review/manifest.json`
- Audited archive copies and hashes: `artifacts/scannetpp-gt-extrinsics-extremal-review/`
- Joined metric table: `artifacts/scannetpp-gt-extrinsics-extremal-review/selected-extrema-review-metrics.csv`
- Contact sheets: `artifacts/scannetpp-gt-extrinsics-extremal-review/contact-sheets/`

Each camera-error panel uses its own color scale; color intensity cannot be compared across models or scenes. Numeric AUC and error quantiles in the table are the cross-case evidence. Preview frames are uniformly sampled visual summaries rather than metric-selected inputs. The GT-extrinsics intervention is an offline diagnostic oracle, not an executable test-time recovery tool.
