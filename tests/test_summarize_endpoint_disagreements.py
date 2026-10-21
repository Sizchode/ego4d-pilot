import pytest

from scripts.failure_mining.summarize_endpoint_disagreements import summarize


def winner(scene, overlap):
    return {
        "dataset": "set",
        "scene": scene,
        "metric_left": "pose",
        "metric_right": "geometry",
        "left_best_models": "a",
        "right_best_models": "b",
        "spearman_rho": "0.8",
        "winner_overlap": str(overlap),
    }


def oracle(scene, gain):
    return {
        "dataset": "set",
        "scene": scene,
        "metric": "geometry",
        "best_fixed_model": "a",
        "best_fixed_value": "0.5",
        "oracle_models": "b",
        "oracle_value": str(0.5 + gain),
        "oracle_gain": str(gain),
    }


def test_separates_disagreement_from_positive_headroom() -> None:
    rows, summary = summarize(
        [winner("positive", False), winner("zero", False), winner("same", True)],
        [oracle("positive", 0.2), oracle("zero", 0.0), oracle("same", 0.1)],
        "geometry",
    )
    assert [row["scene"] for row in rows] == ["positive", "zero"]
    assert rows[0]["positive_oracle_headroom"] is True
    assert rows[1]["positive_oracle_headroom"] is False
    assert summary[0]["winner_disagreement_scenes"] == 2
    assert summary[0]["positive_oracle_headroom_scenes"] == 1


def test_rejects_missing_oracle_scene() -> None:
    with pytest.raises(ValueError, match="missing"):
        summarize([winner("scene", False)], [oracle("other", 0.1)], "geometry")
