import pytest

from scripts.failure_mining.compare_view_depth_error_evidence import (
    IMAGE_FIELDS,
    TARGET_FIELDS,
    VISIBILITY_FIELDS,
    aggregate_correlations,
    merge_view_evidence,
    scene_correlations,
)


def fixtures() -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    targets = []
    images = []
    pairs = []
    for scene in ("a", "b"):
        for index in range(4):
            target = {
                "dataset": "d",
                "scene": scene,
                "view_index": str(index),
                "ground_truth_valid_count": "10",
                "joint_valid_count": str(10 - index),
                "joint_over_ground_truth_valid_ratio": str(1 - index / 10),
            }
            for field in TARGET_FIELDS[:-1]:
                target[field] = str(index + 1)
            targets.append(target)
            image = {"dataset": "d", "scene": scene, "index": str(index)}
            for field in IMAGE_FIELDS:
                image[field] = str(index + 1)
            images.append(image)
        for source, target_index in ((0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2)):
            pair = {
                "dataset": "d",
                "scene": scene,
                "source_index": str(source),
                "target_index": str(target_index),
            }
            for field in VISIBILITY_FIELDS:
                pair[field] = str(source + 1)
            pairs.append(pair)
    return targets, images, pairs


def test_scene_is_statistical_unit_for_view_rank_evidence() -> None:
    targets, images, pairs = fixtures()
    merged = merge_view_evidence(targets, images, pairs)
    assert len(merged) == 8
    correlations = scene_correlations(merged)
    aggregate = aggregate_correlations(correlations)
    image_gradient = next(
        row
        for row in aggregate
        if row["target_field"] == "depth.absolute_relative.mean"
        and row["evidence_field"] == "image.gradient.mean"
    )
    assert image_gradient["scenes"] == 2
    assert image_gradient["median_scene_spearman_rho"] == pytest.approx(1.0)
    assert image_gradient["positive_scenes"] == 2


def test_rejects_missing_image_view() -> None:
    targets, images, pairs = fixtures()
    with pytest.raises(ValueError, match="population differs"):
        merge_view_evidence(targets, images[:-1], pairs)


def test_view_without_valid_gt_stays_missing_for_coverage_target() -> None:
    targets, images, pairs = fixtures()
    targets[0]["ground_truth_valid_count"] = "0"
    targets[0]["joint_valid_count"] = "0"
    targets[0]["joint_over_ground_truth_valid_ratio"] = ""
    for field in TARGET_FIELDS[:-1]:
        targets[0][field] = ""
    merged = merge_view_evidence(targets, images, pairs)
    assert merged[0]["depth.gt_uncovered_ratio"] is None
    assert merged[0]["depth.absolute_relative.mean"] is None
