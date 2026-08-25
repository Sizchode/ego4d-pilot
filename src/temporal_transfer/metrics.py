"""Published AoT accuracy with official class-bias diagnostics."""

from __future__ import annotations

from collections.abc import Sequence


def _class_f1(targets: Sequence[int], predictions: Sequence[int], positive: int) -> float:
    true_positive = sum(t == positive and p == positive for t, p in zip(targets, predictions))
    false_positive = sum(t != positive and p == positive for t, p in zip(targets, predictions))
    false_negative = sum(t == positive and p != positive for t, p in zip(targets, predictions))
    denominator = 2 * true_positive + false_positive + false_negative
    return 0.0 if denominator == 0 else (2.0 * true_positive) / denominator


def arrow_of_time_metrics(targets: Sequence[int], predictions: Sequence[int]) -> dict[str, object]:
    """Return accuracy, per-direction F1, and a fixed [target][prediction] matrix."""

    if len(targets) != len(predictions) or not targets:
        raise ValueError("targets and predictions must be non-empty and have equal length")
    if any(value not in (0, 1) for value in list(targets) + list(predictions)):
        raise ValueError("AoT labels must be 0 (forward) or 1 (backward)")
    confusion = [[0, 0], [0, 0]]
    for target, prediction in zip(targets, predictions):
        confusion[target][prediction] += 1
    return {
        "num_examples": len(targets),
        "accuracy": sum(t == p for t, p in zip(targets, predictions)) / len(targets),
        "forward_f1": _class_f1(targets, predictions, positive=0),
        "backward_f1": _class_f1(targets, predictions, positive=1),
        "confusion_matrix": confusion,
    }
