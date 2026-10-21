import numpy as np
import pytest

from scripts.failure_mining.analyze_da3_gt_depth_error import depth_error_summary


def test_depth_error_summary_uses_benchmark_validity_domain() -> None:
    predicted = np.array([[1.0, 2.0, 0.0], [6.0, np.nan, 4.0]])
    ground_truth = np.array([[1.0, 1.0, 2.0], [4.0, 3.0, 0.0]])
    summary, arrays = depth_error_summary(predicted, ground_truth, max_depth=5.0)
    assert summary["pixels"] == 6
    assert summary["predicted_valid_count"] == 3
    assert summary["ground_truth_valid_count"] == 5
    assert summary["joint_valid_count"] == 2
    np.testing.assert_allclose(arrays["signed_relative"], [0.0, 1.0])
    assert summary["depth.absolute_relative.mean"] == pytest.approx(0.5)
    assert summary["joint_over_ground_truth_valid_ratio"] == pytest.approx(0.4)


def test_depth_error_summary_rejects_invalid_contract() -> None:
    with pytest.raises(ValueError, match="same-shape"):
        depth_error_summary(np.ones((2, 2)), np.ones((3, 2)), max_depth=5.0)
    with pytest.raises(ValueError, match="max_depth"):
        depth_error_summary(np.ones((2, 2)), np.ones((2, 2)), max_depth=0.0)
