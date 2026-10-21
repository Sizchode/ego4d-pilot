import pytest

from scripts.failure_mining.compare_view_error_profiles import compare


FIELDS = ("camera.rotation_degrees.p95",)


def rows(indices=(0, 1, 2, 3, 4)):
    output = []
    for model, reverse in (("a", False), ("b", False), ("c", True)):
        for index in indices:
            value = 4 - index if reverse else index
            output.append(
                {
                    "dataset": "set",
                    "scene": "one",
                    "model": model,
                    "view_index": str(index),
                    FIELDS[0]: str(value),
                }
            )
    return output


def test_reports_pairwise_and_scene_summary() -> None:
    pairwise, summary = compare(rows(), fields=FIELDS)
    assert len(pairwise) == 3
    by_pair = {(row["model_a"], row["model_b"]): row for row in pairwise}
    assert by_pair[("a", "b")]["spearman_rho"] == pytest.approx(1.0)
    assert by_pair[("a", "c")]["spearman_rho"] == pytest.approx(-1.0)
    assert summary == [
        {
            "dataset": "set",
            "scene": "one",
            "field": FIELDS[0],
            "models": 3,
            "views": 5,
            "possible_model_pairs": 3,
            "finite_model_pairs": 3,
            "spearman_rho_min": pytest.approx(-1.0),
            "spearman_rho_median": pytest.approx(-1.0),
            "spearman_rho_max": pytest.approx(1.0),
        }
    ]


def test_records_constant_input_without_dropping_pair() -> None:
    source = rows()
    for row in source:
        if row["model"] == "b":
            row[FIELDS[0]] = "1"
    pairwise, summary = compare(source, fields=FIELDS)
    assert len(pairwise) == 3
    assert sum(row["status"] == "constant_input" for row in pairwise) == 2
    assert summary[0]["possible_model_pairs"] == 3
    assert summary[0]["finite_model_pairs"] == 1


def test_rejects_view_population_mismatch() -> None:
    source = [
        row for row in rows() if not (row["model"] == "c" and row["view_index"] == "4")
    ]
    with pytest.raises(ValueError, match="View populations differ"):
        compare(source, fields=FIELDS)
