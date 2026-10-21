import pytest

from scripts.failure_mining.select_view_residual_disagreements import (
    RESIDUAL_FIELDS,
    TARGET,
    build_profiles,
    select_extremes,
)


def row(index: int, error: float, residual: float) -> dict[str, str]:
    return {
        "dataset": "d",
        "scene": "s",
        "view_index": str(index),
        TARGET: str(error),
        RESIDUAL_FIELDS[0]: str(residual),
        RESIDUAL_FIELDS[1]: str(residual),
    }


def test_selects_exact_rank_disagreement_extrema() -> None:
    profiles = build_profiles(
        [
            row(0, error=3, residual=1),
            row(1, error=2, residual=2),
            row(2, error=1, residual=3),
        ]
    )
    selected = select_extremes(profiles)
    assert [(item["role"], item["view_index"]) for item in selected] == [
        ("residual_underestimates_gt_error", 0),
        ("residual_overestimates_gt_error", 2),
    ]
    assert selected[0]["gt_error_minus_residual_worstness"] == pytest.approx(1.0)
    assert selected[1]["gt_error_minus_residual_worstness"] == pytest.approx(-1.0)


def test_uses_available_direction_without_imputation() -> None:
    rows = [row(0, 3, 1), row(1, 2, 2), row(2, 1, 3)]
    rows[0][RESIDUAL_FIELDS[1]] = ""
    profiles = build_profiles(rows)
    assert profiles[0]["residual_directions_available"] == 1
    assert profiles[0]["bidirectional_residual_mean"] == pytest.approx(1.0)
