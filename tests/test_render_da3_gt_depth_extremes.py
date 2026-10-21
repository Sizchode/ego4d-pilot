from scripts.failure_mining.render_da3_gt_depth_extremes import METRIC, select_extremes


def row(scene: str, index: int, value: float) -> dict[str, str]:
    return {
        "dataset": "scannetpp",
        "scene": scene,
        "view_index": str(index),
        METRIC: str(value),
        "depth.absolute_relative.p95": str(value * 2),
        "joint_over_ground_truth_valid_ratio": "0.9",
    }


def test_selects_exact_extrema_and_resolves_ties_by_lowest_index() -> None:
    selected = select_extremes(
        [
            row("b", 2, 0.5),
            row("a", 2, 0.1),
            row("a", 1, 0.1),
            row("a", 3, 0.8),
            row("a", 4, 0.8),
            row("b", 0, 0.2),
        ]
    )
    assert [(item["scene"], item["role"], item["view_index"]) for item in selected] == [
        ("a", "minimum", 1),
        ("a", "maximum", 3),
        ("b", "minimum", 0),
        ("b", "maximum", 2),
    ]
