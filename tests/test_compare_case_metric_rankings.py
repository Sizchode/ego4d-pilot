import pytest

from scripts.failure_mining.compare_case_metric_rankings import compare


def rows(proxy_values, target_values):
    return [
        {
            "dataset": "set",
            "scene": "scene",
            "model": model,
            "proxy": str(proxy),
            "target": str(target),
        }
        for model, proxy, target in zip(("a", "b", "c", "d"), proxy_values, target_values)
    ]


def test_reports_rank_and_winner_agreement() -> None:
    scene_rows, summary_rows = compare(
        rows([1, 2, 3, 4], [2, 1, 3, 4]),
        proxy_field="proxy",
        target_field="target",
        proxy_direction="lower_is_better",
        target_direction="lower_is_better",
    )
    assert scene_rows[0]["spearman_rho"] == pytest.approx(0.8)
    assert scene_rows[0]["proxy_best_models"] == "a"
    assert scene_rows[0]["target_best_models"] == "b"
    assert scene_rows[0]["winner_overlap"] is False
    assert summary_rows[0]["winner_overlap_scenes"] == 0


def test_direction_is_applied_before_comparing_rankings() -> None:
    scene_rows, _ = compare(
        rows([1, 2, 3, 4], [4, 3, 2, 1]),
        proxy_field="proxy",
        target_field="target",
        proxy_direction="lower_is_better",
        target_direction="higher_is_better",
    )
    assert scene_rows[0]["spearman_rho"] == pytest.approx(1.0)
    assert scene_rows[0]["winner_overlap"] is True


def test_duplicate_model_is_rejected() -> None:
    duplicated = rows([1, 2, 3, 4], [1, 2, 3, 4])
    duplicated.append(dict(duplicated[0]))
    with pytest.raises(ValueError, match="Duplicate"):
        compare(
            duplicated,
            proxy_field="proxy",
            target_field="target",
            proxy_direction="lower_is_better",
            target_direction="lower_is_better",
        )
