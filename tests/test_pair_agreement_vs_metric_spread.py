import pytest

from scripts.failure_mining.analyze_pair_agreement_vs_metric_spread import (
    build_correlations,
    build_scene_rows,
)


def pair_rows():
    rows = []
    for scene, rotation, translation in (
        ("a", 0.1, 0.9),
        ("b", 0.2, 0.8),
        ("c", 0.3, 0.7),
    ):
        for metric, value in (
            ("rotation_degrees", rotation),
            ("translation_degrees", translation),
        ):
            rows.append(
                {
                    "dataset": "bench",
                    "scene": scene,
                    "metric": metric,
                    "model_left": "m1",
                    "model_right": "m2",
                    "spearman_pair_error": str(value),
                }
            )
    return rows


def rank_rows():
    rows = []
    for scene, spread in (("a", 0.1), ("b", 0.2), ("c", 0.3)):
        for metric in ("pose", "geometry"):
            rows.extend(
                [
                    {
                        "dataset": "bench",
                        "scene": scene,
                        "metric": metric,
                        "model": "m1",
                        "value": "0.0",
                    },
                    {
                        "dataset": "bench",
                        "scene": scene,
                        "metric": metric,
                        "model": "m2",
                        "value": str(spread),
                    },
                ]
            )
    return rows


def test_separate_camera_predictors_and_endpoints_are_preserved():
    scenes = build_scene_rows(pair_rows(), rank_rows(), "bench", "pose", "geometry")
    correlations = {
        (row["predictor"], row["endpoint"]): row[
            "spearman_predictor_vs_endpoint"
        ]
        for row in build_correlations(scenes)
    }
    assert correlations[("median_rotation_pair_error_agreement", "pose_model_range")] == pytest.approx(1.0)
    assert correlations[("median_rotation_pair_error_agreement", "geometry_model_range")] == pytest.approx(1.0)
    assert correlations[("median_translation_pair_error_agreement", "pose_model_range")] == pytest.approx(-1.0)
    assert correlations[("median_translation_pair_error_agreement", "geometry_model_range")] == pytest.approx(-1.0)


def test_missing_camera_metric_is_rejected():
    truncated = [row for row in pair_rows() if not (row["scene"] == "c" and row["metric"] == "translation_degrees")]
    with pytest.raises(ValueError, match="Missing camera metric"):
        build_scene_rows(truncated, rank_rows(), "bench", "pose", "geometry")


def test_duplicate_pair_agreement_is_rejected():
    duplicated = pair_rows()
    duplicated.append(dict(duplicated[0]))
    with pytest.raises(ValueError, match="Duplicate pair-agreement row"):
        build_scene_rows(duplicated, rank_rows(), "bench", "pose", "geometry")
