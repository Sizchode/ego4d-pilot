import pytest

from scripts.failure_mining.compare_support_adjusted_residual import (
    RESIDUAL_FIELDS,
    SUPPORT_FIELDS,
    TARGET,
    aggregate,
    scene_scores,
)


def row(index: int, target: float, residual: float, support: float) -> dict[str, str]:
    value = {
        "dataset": "d",
        "scene": "s",
        "view_index": str(index),
        TARGET: str(target),
    }
    value.update({field: str(residual) for field in RESIDUAL_FIELDS})
    value.update({field: str(support) for field in SUPPORT_FIELDS})
    return value


def test_score_fit_uses_residual_and_support_without_target() -> None:
    rows = [
        row(0, target=1, residual=1, support=1),
        row(1, target=3, residual=3, support=1),
        row(2, target=2, residual=2, support=2),
        row(3, target=4, residual=4, support=2),
    ]
    views, scenes = scene_scores(rows)
    assert len(views) == 4
    assert len(scenes) == 1
    assert all("support_adjusted_residual_score" in item for item in views)
    assert scenes[0]["raw_residual_spearman_rho"] == pytest.approx(1.0)


def test_uses_available_direction_without_imputation() -> None:
    rows = [
        row(0, target=1, residual=1, support=2),
        row(1, target=2, residual=2, support=1),
        row(2, target=3, residual=3, support=3),
    ]
    rows[0][RESIDUAL_FIELDS[1]] = ""
    rows[0][SUPPORT_FIELDS[1]] = ""
    views, _ = scene_scores(rows)
    assert views[0]["residual_directions"] == 1
    assert views[0]["support_directions"] == 1


def test_aggregate_compares_scenes_as_paired_units() -> None:
    rows = [
        {
            "raw_residual_spearman_rho": 0.1,
            "support_adjusted_residual_spearman_rho": 0.3,
        },
        {
            "raw_residual_spearman_rho": 0.2,
            "support_adjusted_residual_spearman_rho": 0.4,
        },
    ]
    result = aggregate(rows)
    assert result["scenes"] == 2
    assert result["adjusted_better_scenes"] == 2
    assert result["median_adjusted_minus_raw_rho"] == pytest.approx(0.2)
