import pytest

from scripts.failure_mining.analyze_view_image_camera_correlations import analyze


def image_rows() -> list[dict[str, str]]:
    rows = []
    for index in range(5):
        rows.append(
            {
                "dataset": "set",
                "scene": "one",
                "index": str(index),
                "gradient.mean": str(index),
                "luminance.mean": str(4 - index),
                "rgb.any_channel_at_ceiling_ratio": "0.0",
                "gradient.count": "10",
            }
        )
    return rows


def camera_rows(indices: tuple[int, ...] = (0, 1, 2, 3, 4)) -> list[dict[str, str]]:
    return [
        {
            "dataset": "set",
            "scene": "one",
            "model": "model",
            "view_index": str(index),
            "camera.rotation_degrees.p95": str(index),
            "camera.translation_degrees.p95": str(4 - index),
        }
        for index in indices
    ]


def test_correlations_are_corrected_within_each_camera_endpoint() -> None:
    rows = analyze(
        image_rows(),
        camera_rows(),
        datasets={"set"},
        scenes={"one"},
        camera_fields=(
            "camera.rotation_degrees.p95",
            "camera.translation_degrees.p95",
        ),
    )
    assert len(rows) == 4
    by_key = {(row["camera_field"], row["image_field"]): row for row in rows}
    assert by_key[("camera.rotation_degrees.p95", "gradient.mean")][
        "spearman_rho"
    ] == pytest.approx(1.0)
    assert by_key[("camera.rotation_degrees.p95", "luminance.mean")][
        "spearman_rho"
    ] == pytest.approx(-1.0)
    assert all(row["tests_in_endpoint_family"] == 2 for row in rows)


def test_rejects_population_mismatch() -> None:
    try:
        analyze(
            image_rows(),
            camera_rows((0, 1, 2, 3)),
            datasets={"set"},
            scenes={"one"},
            camera_fields=("camera.rotation_degrees.p95",),
        )
    except ValueError as error:
        assert "populations differ" in str(error)
    else:
        raise AssertionError("Expected population mismatch to be rejected")
