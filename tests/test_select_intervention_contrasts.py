import pytest

from scripts.failure_mining.select_intervention_contrasts import select_extrema


def row(model: str, scene: str, gain: float, metric: str = "fscore") -> dict[str, str]:
    return {
        "model": model,
        "scene": scene,
        "metric": metric,
        "improvement_vs_control": str(gain),
        "native_resample_control": "0.4",
        "gt_extrinsics_intervention": str(0.4 + gain),
    }


def test_selects_per_model_extrema_and_preserves_ties() -> None:
    selected = select_extrema(
        [
            row("a", "a-min-1", -0.1),
            row("a", "a-min-2", -0.1),
            row("a", "a-mid", 0.0),
            row("a", "a-max", 0.2),
            row("b", "b-min", 0.1),
            row("b", "b-max", 0.3),
            row("b", "ignored", 1.0, metric="acc"),
        ],
        "fscore",
    )

    assert [(item["model"], item["extremum"], item["scene"]) for item in selected] == [
        ("a", "maximum_gain", "a-max"),
        ("a", "minimum_gain", "a-min-1"),
        ("a", "minimum_gain", "a-min-2"),
        ("b", "maximum_gain", "b-max"),
        ("b", "minimum_gain", "b-min"),
    ]


def test_rejects_duplicate_model_scene_metric_rows() -> None:
    with pytest.raises(ValueError, match="Duplicate intervention row"):
        select_extrema([row("a", "same", 0.1), row("a", "same", 0.2)], "fscore")


def test_rejects_missing_metric() -> None:
    with pytest.raises(ValueError, match="No rows found"):
        select_extrema([row("a", "scene", 0.1, metric="acc")], "fscore")
