import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from scripts.failure_mining.render_selected_input_views import render_selected_views


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_snapshot(root: Path, *, mismatch: bool = False) -> Path:
    images = []
    for index, color in enumerate(((255, 0, 0), (0, 255, 0), (0, 0, 255))):
        path = root / "inputs" / f"{index}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (32 + index, 24), color).save(path)
        images.append(
            {
                "index": index,
                "path": str(path),
                "sha256": file_sha256(path),
                "original_hw": [24, 32 + index],
            }
        )

    rows = []
    for model_index, model in enumerate(("a", "b")):
        run = root / f"run-{model}"
        (run / "evaluation_capture").mkdir(parents=True)
        manifest = [dict(row) for row in images]
        if mismatch and model == "b":
            manifest[0]["sha256"] = "0" * 64
        (run / "input_manifest.json").write_text(json.dumps(manifest))
        (run / "status.json").write_text(json.dumps({"state": "completed"}))
        (run / "validation.json").write_text(json.dumps({"passed": True}))
        (run / "run_config.json").write_text(
            json.dumps({"dataset": "set", "scene": "one"})
        )
        # Complete undirected population for three views.  Model a selects view 2
        # for rotation and view 0 for translation; model b reverses those peaks.
        np.savez(
            run / "evaluation_capture" / "camera_pairs.npz",
            index_a=np.array([0, 0, 1]),
            index_b=np.array([1, 2, 2]),
            rotation_degrees=np.array(
                [1.0, 2.0, 9.0] if model_index == 0 else [9.0, 2.0, 1.0]
            ),
            translation_degrees=np.array(
                [8.0, 7.0, 1.0] if model_index == 0 else [1.0, 7.0, 8.0]
            ),
        )
        rows.append(
            {"model": model, "dataset": "set", "scene": "one", "run": str(run)}
        )
    metrics = root / "completed_metrics.jsonl"
    metrics.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return metrics


def test_renders_rank_selected_views_with_auditable_manifest(tmp_path: Path) -> None:
    metrics = make_snapshot(tmp_path)
    output = tmp_path / "rendered"
    manifest = render_selected_views(
        metrics,
        "set",
        ["one"],
        ["a", "b"],
        output,
        top_k=1,
        quantile=0.95,
        max_side=20,
        jpeg_quality=90,
    )
    assert manifest["uses_ground_truth_selection"] is True
    assert manifest["test_time_policy_input"] is False
    scene = manifest["scene_outputs"][0]
    assert scene["input_count"] == 3
    assert scene["selected_view_count"] >= 2
    for view in scene["views"]:
        rendered = output / view["output_path"]
        assert rendered.is_file()
        assert file_sha256(rendered) == view["output_sha256"]
        assert view["selection_reasons"]


def test_rejects_cross_model_input_population_mismatch(tmp_path: Path) -> None:
    metrics = make_snapshot(tmp_path, mismatch=True)
    with pytest.raises(ValueError, match="Input population differs across models"):
        render_selected_views(
            metrics,
            "set",
            ["one"],
            ["a", "b"],
            tmp_path / "rendered",
            top_k=1,
            quantile=0.95,
            max_side=20,
            jpeg_quality=90,
        )
