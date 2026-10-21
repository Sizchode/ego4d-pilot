import csv
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pytest

from scripts.failure_mining.analyze_gt_geometry_factors import (
    image_key,
    load_extrinsics_reference,
    load_gt_depth,
    prepare_gt_intrinsics,
)
from scripts.failure_mining.compare_baselines import sha256
from scripts.failure_mining.summarize_gt_geometry_factors import summarize


def test_image_key_is_independent_of_absolute_prefix():
    relative = "workspace/benchmark_dataset/scannetpp/scene/merge_dslr_iphone/images/iphone/a.jpg"
    absolute = "/scratch/data/scannetpp/scene/merge_dslr_iphone/images/iphone/a.jpg"
    assert image_key(relative, "scene") == image_key(absolute, "scene")


def test_prepare_gt_intrinsics_matches_official_resize_rule():
    matrix = np.array([[[100.0, 0.0, 20.0], [0.0, 200.0, 30.0], [0.0, 0.0, 1.0]]])
    result = prepare_gt_intrinsics(matrix, [(100, 200)], input_h=50, input_w=100)
    expected = matrix.copy()
    expected[:, :2, 2] += 0.5
    expected[:, 0, :] *= 0.5
    expected[:, 1, :] *= 0.5
    np.testing.assert_allclose(result, expected)


def test_load_gt_depth_converts_millimeters_and_invalid_pixels(tmp_path: Path):
    path = tmp_path / "depth.png"
    imageio.imwrite(path, np.array([[0, 1000], [2000, 3000]], dtype=np.uint16))
    result = load_gt_depth(path, input_h=2, input_w=2)
    np.testing.assert_allclose(result, [[0.0, 1.0], [2.0, 3.0]])


def write_factor_input(root: Path) -> None:
    metrics = {
        "acc": 0.4,
        "comp": 0.5,
        "overall": 0.45,
        "precision": 0.5,
        "recall": 0.4,
        "fscore": 0.44,
    }
    rows = []
    for model in ("m1", "m2"):
        rows.append(
            {
                "model": model,
                "scene": "s1",
                "stages": [
                    "official_native",
                    "gt_extrinsics",
                    "gt_extrinsics_intrinsics",
                    "gt_extrinsics_intrinsics_depth",
                ],
                "official_native_metrics": metrics,
                "gt_extrinsics_metrics": {key: value - 0.1 if key in {"acc", "comp", "overall"} else value + 0.1 for key, value in metrics.items()},
                "gt_extrinsics_intrinsics_metrics": {key: value - 0.2 if key in {"acc", "comp", "overall"} else value + 0.2 for key, value in metrics.items()},
                "gt_extrinsics_intrinsics_depth_metrics": {key: value - 0.3 if key in {"acc", "comp", "overall"} else value + 0.3 for key, value in metrics.items()},
            }
        )
    results = root / "results.jsonl"
    results.write_text("".join(json.dumps(row) + "\n" for row in rows))
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "cases": 2,
                "models": ["m1", "m2"],
                "scenes": ["s1"],
                "results_sha256": sha256(results),
                "stages": rows[0]["stages"],
                "uses_ground_truth_intervention": True,
                "test_time_policy_input": False,
            }
        )
    )


def test_factor_summary_computes_separate_stage_gains(tmp_path: Path):
    write_factor_input(tmp_path)
    case_rows, summary_rows, provenance = summarize(tmp_path)
    assert len(case_rows) == 12
    assert len(summary_rows) == 12
    fscore = next(row for row in summary_rows if row["model"] == "m1" and row["metric"] == "fscore")
    assert fscore["extrinsics_gain_vs_native_mean"] == pytest.approx(0.1)
    assert fscore["intrinsics_gain_after_extrinsics_mean"] == pytest.approx(0.1)
    assert fscore["depth_gain_after_extrinsics_intrinsics_mean"] == pytest.approx(0.1)
    assert provenance["cases"] == 2


def test_reference_hash_mismatch_is_rejected(tmp_path: Path):
    rows = [
        {
            "model": "m",
            "scene": "s",
            "metric": metric,
            "official_native": "0.1",
            "gt_extrinsics_intervention": "0.2",
        }
        for metric in ("acc", "comp", "overall", "precision", "recall", "fscore")
    ]
    with (tmp_path / "case_metric_deltas.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "case_metric_output_sha256": "wrong",
                "uses_ground_truth_intervention": True,
                "test_time_policy_input": False,
            }
        )
    )
    with pytest.raises(ValueError, match="hash"):
        load_extrinsics_reference(tmp_path)


def test_factor_summary_rejects_identity_population_mismatch(tmp_path: Path):
    write_factor_input(tmp_path)
    rows = [json.loads(line) for line in (tmp_path / "results.jsonl").read_text().splitlines()]
    rows.reverse()
    results = tmp_path / "results.jsonl"
    results.write_text("".join(json.dumps(row) + "\n" for row in rows))
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    manifest["results_sha256"] = sha256(results)
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="population/order"):
        summarize(tmp_path)
