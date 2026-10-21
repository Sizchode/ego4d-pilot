import pytest

from scripts.failure_mining.analyze_residual_disagreement_extremes import (
    OBSERVABLE_FIELDS,
    OVER,
    UNDER,
    pair_observables,
    summarize_pairs,
)


def evidence(scene: str, view: int, value: float) -> dict[str, str]:
    row = {"dataset": "d", "scene": scene, "view_index": str(view)}
    row.update({field: str(value) for field in OBSERVABLE_FIELDS})
    return row


def selection(scene: str, view: int, role: str) -> dict[str, str]:
    return {"dataset": "d", "scene": scene, "view_index": str(view), "role": role}


def test_pairs_within_scene_and_preserves_direction() -> None:
    paired = pair_observables(
        [selection("a", 1, UNDER), selection("a", 2, OVER)],
        [evidence("a", 1, 3.0), evidence("a", 2, 1.0)],
    )
    assert len(paired) == len(OBSERVABLE_FIELDS)
    assert {row["underestimate_minus_overestimate"] for row in paired} == {2.0}


def test_rejects_incomplete_scene_pair() -> None:
    with pytest.raises(ValueError, match="complete pair"):
        pair_observables([selection("a", 1, UNDER)], [evidence("a", 1, 3.0)])


def test_summarizes_scene_pairs_and_corrects_full_family() -> None:
    selections = []
    evidences = []
    for index in range(1, 7):
        scene = f"s{index}"
        selections.extend([selection(scene, 1, UNDER), selection(scene, 2, OVER)])
        evidences.extend([evidence(scene, 1, float(index + 1)), evidence(scene, 2, 1.0)])
    summary = summarize_pairs(pair_observables(selections, evidences))
    assert len(summary) == len(OBSERVABLE_FIELDS)
    assert all(row["scene_pairs"] == 6 for row in summary)
    assert all(row["underestimate_greater_scenes"] == 6 for row in summary)
    assert all(row["tests_in_family"] == len(OBSERVABLE_FIELDS) for row in summary)
    assert all(0 <= row["bh_q_value"] <= 1 for row in summary)
