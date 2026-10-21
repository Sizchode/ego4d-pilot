import pytest

from scripts.failure_mining.analyze_intervention_predictors import (
    build_correlations,
    build_joined_rows,
)


def native_rows():
    return [
        {
            "dataset": "bench",
            "metric": "pose.auc30",
            "model": "native",
            "scene": scene,
            "value": str(value),
        }
        for scene, value in zip(("a", "b", "c"), (0.9, 0.8, 0.7), strict=True)
    ]


def intervention_rows():
    rows = []
    for scene, native_fscore, gain in zip(
        ("a", "b", "c"), (0.8, 0.6, 0.4), (0.1, 0.2, 0.3), strict=True
    ):
        rows.append(
            {
                "model": "intervention",
                "scene": scene,
                "metric": "fscore",
                "official_native": str(native_fscore),
                "native_resample_control": str(native_fscore),
                "gt_extrinsics_intervention": str(native_fscore + gain),
                "improvement_vs_control": str(gain),
                "control_delta_vs_official": "0",
            }
        )
    return rows


def test_join_and_correlations_preserve_separate_predictors():
    joined = build_joined_rows(
        native_rows(),
        intervention_rows(),
        {"intervention": "native"},
        "bench",
        "pose.auc30",
    )
    assert len(joined) == 3
    correlations = {row["predictor"]: row for row in build_correlations(joined)}
    assert set(correlations) == {"native_pose_value", "official_native_fscore"}
    assert correlations["native_pose_value"][
        "spearman_predictor_vs_fscore_improvement"
    ] == pytest.approx(-1.0)
    assert correlations["official_native_fscore"][
        "spearman_predictor_vs_fscore_improvement"
    ] == pytest.approx(-1.0)


def test_missing_native_pose_row_is_rejected():
    with pytest.raises(ValueError, match="Missing native pose row"):
        build_joined_rows(
            native_rows()[:-1],
            intervention_rows(),
            {"intervention": "native"},
            "bench",
            "pose.auc30",
        )


def test_duplicate_intervention_case_is_rejected():
    duplicated = intervention_rows()
    duplicated.append(dict(duplicated[0]))
    with pytest.raises(ValueError, match="Duplicate intervention F-score row"):
        build_joined_rows(
            native_rows(),
            duplicated,
            {"intervention": "native"},
            "bench",
            "pose.auc30",
        )
