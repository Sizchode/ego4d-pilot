import numpy as np
import pytest

from scripts.failure_mining.analyze_gt_extrinsics_intervention import (
    metrics_from_distances,
    metrics_from_points,
)


def test_metrics_from_distances_matches_official_formulas() -> None:
    metrics = metrics_from_distances(
        np.array([0.01, 0.07]), np.array([0.02, 0.03, 0.08]), threshold=0.05
    )
    assert metrics["acc"] == pytest.approx(0.04)
    assert metrics["comp"] == pytest.approx(0.13 / 3)
    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["recall"] == pytest.approx(2 / 3)
    assert metrics["fscore"] == pytest.approx(4 / 7)


def test_metrics_from_points_uses_bidirectional_nearest_neighbors() -> None:
    gt = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    predicted = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    metrics = metrics_from_points(gt, predicted, threshold=0.5)
    assert metrics["acc"] == pytest.approx(0.5)
    assert metrics["comp"] == pytest.approx(0.5)
    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["recall"] == pytest.approx(0.5)
    assert metrics["fscore"] == pytest.approx(0.5)


def test_metrics_rejects_empty_or_nonfinite_populations() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        metrics_from_distances(np.array([]), np.array([1.0]), threshold=0.05)
    with pytest.raises(ValueError, match="finite"):
        metrics_from_points(
            np.array([[np.nan, 0.0, 0.0]]),
            np.array([[0.0, 0.0, 0.0]]),
            threshold=0.05,
        )
