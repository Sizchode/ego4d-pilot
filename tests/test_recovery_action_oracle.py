import pytest

from scripts.failure_mining.analyze_recovery_action_oracle import (
    aggregate_action_rows,
    build_action_rows,
)


MODEL_MAP = {"full-a": "a", "full-b": "b"}


def rank_rows():
    values = {
        ("a", "s1"): 0.5,
        ("b", "s1"): 0.8,
        ("a", "s2"): 0.9,
        ("b", "s2"): 0.7,
    }
    return [
        {
            "dataset": "bench",
            "metric": "geometry",
            "model": model,
            "scene": scene,
            "value": str(value),
        }
        for (model, scene), value in values.items()
    ]


def intervention_rows():
    native = {
        ("full-a", "s1"): (0.5, 0.9),
        ("full-b", "s1"): (0.8, 0.75),
        ("full-a", "s2"): (0.9, 0.85),
        ("full-b", "s2"): (0.7, 0.95),
    }
    return [
        {
            "model": model,
            "scene": scene,
            "metric": "fscore",
            "official_native": str(values[0]),
            "gt_extrinsics_intervention": str(values[1]),
        }
        for (model, scene), values in native.items()
    ]


def test_action_oracle_preserves_no_action_pose_and_switch_candidates():
    rows = build_action_rows(
        rank_rows(),
        intervention_rows(),
        MODEL_MAP,
        "bench",
        "geometry",
        "fscore",
        "higher_is_better",
    )
    by_case = {(row["source_model"], row["scene"]): row for row in rows}
    assert by_case[("a", "s1")]["oracle_actions"] == "gt_extrinsics"
    assert by_case[("b", "s1")]["oracle_actions"] == "no_action"
    assert by_case[("a", "s2")]["oracle_actions"] == "no_action"
    assert by_case[("b", "s2")]["oracle_actions"] == "gt_extrinsics"
    summaries = {row["source_model"]: row for row in aggregate_action_rows(rows)}
    assert summaries["a"]["gt_extrinsics_positive_scenes"] == 1
    assert summaries["a"]["switch_backbone_positive_scenes"] == 1
    assert summaries["b"]["combined_oracle_positive_scenes"] == 1


def test_native_metric_mismatch_is_rejected():
    interventions = intervention_rows()
    interventions[0]["official_native"] = "0.6"
    with pytest.raises(ValueError, match="Native metric disagrees"):
        build_action_rows(
            rank_rows(),
            interventions,
            MODEL_MAP,
            "bench",
            "geometry",
            "fscore",
            "higher_is_better",
        )


def test_missing_model_population_is_rejected():
    truncated = rank_rows()[:-1]
    with pytest.raises(ValueError, match="Baseline model population differs"):
        build_action_rows(
            truncated,
            intervention_rows(),
            MODEL_MAP,
            "bench",
            "geometry",
            "fscore",
            "higher_is_better",
        )
