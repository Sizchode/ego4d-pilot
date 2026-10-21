import pytest

from scripts.failure_mining.analyze_intervention_evidence import (
    correlate,
    join_intervention,
    merge_evidence,
)


def evidence_rows(field: str, values: list[float]) -> list[dict[str, str]]:
    return [
        {"dataset": "set", "scene": f"s{index}", field: str(value)}
        for index, value in enumerate(values)
    ]


def intervention_rows(values: list[float]) -> list[dict[str, str]]:
    return [
        {
            "model": "model",
            "scene": f"s{index}",
            "metric": "fscore",
            "improvement_vs_control": str(value),
        }
        for index, value in enumerate(values)
    ]


def test_merges_sources_joins_target_and_corrects_correlations() -> None:
    merged = merge_evidence(
        [
            ("geometry", evidence_rows("geometry.x", [1, 2, 3, 4])),
            ("image", evidence_rows("image.y", [4, 3, 2, 1])),
        ],
        "set",
        ["geometry.x", "image.y"],
    )
    joined = join_intervention(
        intervention_rows([0.1, 0.2, 0.3, 0.4]),
        merged,
        "set",
        "model",
        "fscore",
        ["geometry.x", "image.y"],
    )
    correlations = correlate(joined, ["geometry.x", "image.y"])

    assert len(joined) == 4
    assert correlations[0]["spearman_evidence_vs_intervention_gain"] == pytest.approx(1.0)
    assert correlations[1]["spearman_evidence_vs_intervention_gain"] == pytest.approx(-1.0)
    assert all(row["tests_in_family"] == 2 for row in correlations)
    assert all(0 <= row["bh_q_value_within_analysis"] <= 1 for row in correlations)


def test_rejects_evidence_field_with_multiple_sources() -> None:
    with pytest.raises(ValueError, match="multiple sources"):
        merge_evidence(
            [
                ("one", evidence_rows("e.x", [1, 2, 3])),
                ("two", evidence_rows("e.x", [4, 5, 6])),
            ],
            "set",
            ["e.x"],
        )


def test_rejects_intervention_without_evidence() -> None:
    merged = merge_evidence(
        [("one", evidence_rows("e.x", [1, 2, 3]))], "set", ["e.x"]
    )
    rows = intervention_rows([0.1, 0.2, 0.3, 0.4])
    with pytest.raises(ValueError, match="has no evidence"):
        join_intervention(rows, merged, "set", "model", "fscore", ["e.x"])
