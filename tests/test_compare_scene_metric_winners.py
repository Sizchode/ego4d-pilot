import pytest

from scripts.failure_mining.compare_scene_metric_winners import compare


def metric_rows(left, right):
    rows = []
    for model, left_value, right_value in zip(("a", "b", "c", "d"), left, right):
        rows.extend(
            [
                {
                    "dataset": "set",
                    "scene": "scene",
                    "model": model,
                    "metric": "left",
                    "value": str(left_value),
                    "direction": "lower_is_worse",
                },
                {
                    "dataset": "set",
                    "scene": "scene",
                    "model": model,
                    "metric": "right",
                    "value": str(right_value),
                    "direction": "higher_is_worse",
                },
            ]
        )
    return rows


def test_compares_scene_winners_with_metric_directions() -> None:
    scene_rows, summary_rows = compare(
        metric_rows([4, 3, 2, 1], [1, 2, 3, 4]), "left", "right"
    )
    assert scene_rows[0]["spearman_rho"] == pytest.approx(1.0)
    assert scene_rows[0]["left_best_models"] == "a"
    assert scene_rows[0]["right_best_models"] == "a"
    assert scene_rows[0]["winner_overlap"] is True
    assert summary_rows[0]["winner_overlap_fraction"] == 1.0


def test_reports_different_top_models_despite_positive_rank_agreement() -> None:
    scene_rows, _ = compare(
        metric_rows([4, 3, 2, 1], [2, 1, 3, 4]), "left", "right"
    )
    assert scene_rows[0]["spearman_rho"] == pytest.approx(0.8)
    assert scene_rows[0]["left_best_models"] == "a"
    assert scene_rows[0]["right_best_models"] == "b"
    assert scene_rows[0]["winner_overlap"] is False


def test_rejects_partial_model_population() -> None:
    rows = metric_rows([4, 3, 2, 1], [1, 2, 3, 4])
    rows = [row for row in rows if not (row["model"] == "d" and row["metric"] == "right")]
    with pytest.raises(ValueError, match="populations differ"):
        compare(rows, "left", "right")
