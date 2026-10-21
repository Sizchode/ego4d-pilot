import pytest

from scripts.failure_mining.compare_scene_depth_error_to_gain import (
    DEPTH_FACTOR,
    correlations,
    merge_scene_targets,
)


def depth_row(scene: str, value: float) -> dict[str, str]:
    return {
        "dataset": "scannetpp",
        "scene": scene,
        "depth.absolute_relative.mean": str(value),
        "depth.absolute_relative.p95": str(value * 2),
        "depth.absolute_log_ratio.mean": str(value * 3),
        "joint_over_ground_truth_valid_ratio": str(1 - value / 10),
    }


def target_row(scene: str, value: float, factor: str = DEPTH_FACTOR) -> dict[str, str]:
    return {
        "dataset": "scannetpp",
        "scene": scene,
        "factor": factor,
        "conditional_gain": str(value),
    }


def test_merge_and_correlate_predeclared_depth_fields() -> None:
    depth = [depth_row(scene, value) for scene, value in (("a", 0.1), ("b", 0.2), ("c", 0.3))]
    targets = [target_row(scene, value) for scene, value in (("a", 1), ("b", 2), ("c", 3))]
    targets.append(target_row("ignored", 7, "extrinsics"))
    merged = merge_scene_targets(depth, targets)
    assert merged[0]["depth.gt_uncovered_ratio"] == pytest.approx(0.01)
    result = correlations(merged)
    assert len(result) == 4
    assert all(row["spearman_rho"] == pytest.approx(1.0) for row in result)
    assert {row["tests_in_family"] for row in result} == {4}


def test_population_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError, match="populations differ"):
        merge_scene_targets(
            [depth_row("a", 0.1), depth_row("b", 0.2)],
            [target_row("a", 1.0)],
        )
