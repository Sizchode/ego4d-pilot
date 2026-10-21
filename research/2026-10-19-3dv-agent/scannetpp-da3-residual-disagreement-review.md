# ScanNet++ DA3 residual-disagreement review

## One pair that exposes the problem

In scene `fb5a96b1a2`, the numerical residual ranks view 6 as relatively safe: its bidirectional residual is 0.021, yet its GT mean depth error is 0.091. The same residual ranks view 9 as dangerous: residual 0.260, but GT error only 0.030. Thus, following the residual alone would inspect or modify the better view and leave the worse view untouched.

The paired observables explain part of this reversal. View 6 has outgoing visibility support 0.503; view 9 has only 0.026. A large residual computed from very sparse cross-view support is not comparable to a small residual backed by broad support. The RGB images also differ, but this pair does not by itself establish a semantic cause.

![Residual disagreement examples](../../artifacts/scannetpp-da3-residual-disagreement-panels-20/panels/fb5a96b1a2.png)

## How cases were selected

For each of the 20 ScanNet++ scenes, all views with GT depth error and at least one directed visibility residual were ranked twice: once by GT mean absolute-relative depth error and once by the mean available incoming/outgoing residual. The review contains the exact maximum and minimum of

`GT-error worstness rank − residual worstness rank`.

This gives one residual false-negative extreme and one residual false-positive extreme per scene, or 40 views total. There is no error threshold, quality label, or tuned top-k. GT is used only to build this offline diagnostic set and cannot enter a test-time policy. All 20 rendered panels match their manifest hashes.

## The strongest repeatable finding is geometric support

The two selected views form a within-scene pair. Seven predeclared test-time observables were compared across the 20 pairs with two-sided Wilcoxon tests and Benjamini–Hochberg correction.

| Observable: false negative minus false positive | Median paired difference | Positive scenes | BH q |
|---|---:|---:|---:|
| Outgoing visibility support | +0.248 | 19/20 | 4.01e-5 |
| Incoming visibility support | +0.194 | 18/20 | 3.67e-4 |
| Mean image gradient | +0.00408 | 18/20 | 6.68e-5 |
| Gradient p95 | +0.0239 | 16/20 | 1.91e-4 |
| Luminance entropy | +0.134 bits | 17/20 | 0.00680 |
| Ceiling-pixel ratio | +0.00215 | 12/20 | 0.167 |
| Floor-pixel ratio | −0.00061 | 10/20 | 0.701 |

The residual's largest false positives are usually **low-support comparisons**, while its largest false negatives usually have broad support and richer image structure. This result is conditioned on exact disagreement extrema, so it describes why the current ranking fails; it is not a population-wide causal estimate. It nevertheless rules out using raw residual magnitude as the only diagnosis score.

As an exploratory numerical check, we formed a score without GT: within each scene, convert residual and support to ranks, then subtract the ordinary-least-squares projection of residual rank on support rank. The median within-scene correlation with offline GT depth error rises from 0.363 for raw residual to 0.417 for the support-adjusted score. The adjusted score is positive in 20/20 scenes and improves over raw residual in 17/20; the paired difference has p=7.08e-4. This is evidence that support is a real calibration variable, but the score was motivated and evaluated on the same 20 scenes. It must be frozen on an independent development split before serving as a policy baseline.

## What the images add

The 20 false-negative panels repeatedly contain geometry that a single smooth depth field can represent poorly even when adjacent views agree enough to produce a small residual:

- thin structures and dense occlusion boundaries: chair and table legs in `21d970d8de`, `bde1e479ad`, and `c4c04e6d6c`;
- reflective or transparent surfaces: the stainless sink in `40aec5fffa`, the glass or mirror region in `3e8bba0176`, and windows or blinds in `cc5237fd77`;
- repeated high-frequency structure: cubbies in `9071e139d9`, radiator slats in `bcd2436daf`, and shelving or equipment in `c5439f4607`;
- mixed foreground clutter and depth discontinuities: `286b55a2bf`, `5f99900f09`, `7831862f02`, `acd95847c5`, and `fb5a96b1a2`.

Several false positives are plain walls, doors, or oblique room boundaries with little GT error, consistent with low overlap rather than difficult object geometry. Other false positives also contain clutter. The panels therefore do not support a rule such as “chairs imply failure” or “reflection implies failure.” These visual categories are hypotheses from a fixed qualitative sample, not categorical annotations or causal labels.

## Consequence for the proposed method

The first diagnostic baseline must be support-aware and numerical. It should report residual magnitude together with valid overlap, direction agreement, and uncertainty. A VLM cannot claim an increment merely by finding the bad view, because geometry already localizes error above chance and support explains a major residual failure mode.

The remaining plausible VLM contribution is narrower: after a support-aware diagnostic localizes an ambiguous region, use the RGB content to distinguish causes that imply different actions. A reflective window may call for material masking or another-view routing; a thin chair boundary may call for higher-resolution local depth or boundary-aware fusion; a low-overlap but geometrically correct view should be left unchanged. Those actions remain hypotheses until the executable tools produce measured geometry gains. The agent experiment should begin only after that tool-level oracle is established.

## Reproducible artifacts

- Exact rank profiles and selected extrema: `artifacts/scannetpp-da3-view-residual-disagreement-extremes-20/`
- Verified 20-scene panels and selection table: `artifacts/scannetpp-da3-residual-disagreement-panels-20/`
- Paired observable analysis: `artifacts/scannetpp-da3-residual-disagreement-observables-20/`
- Exploratory support-adjusted residual analysis: `artifacts/scannetpp-da3-support-adjusted-residual-20/`
- Analysis program: `scripts/failure_mining/analyze_residual_disagreement_extremes.py`
- Support-adjustment program: `scripts/failure_mining/compare_support_adjusted_residual.py`
- Unit tests: `tests/test_analyze_residual_disagreement_extremes.py`
