import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from scripts.audit.extremal_review_views import audit_views
from scripts.failure_mining.render_extremal_review_views import render_extrema


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_fixture(root: Path, mismatch: bool = False) -> Path:
    images = []
    for index, color in enumerate(((255, 0, 0), (0, 255, 0), (0, 0, 255))):
        path = root / "inputs" / f"{index}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (20 + index, 16), color).save(path)
        images.append({"index": index, "path": str(path), "sha256": digest(path), "original_hw": [16, 20 + index]})
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
        (run / "run_config.json").write_text(json.dumps({"dataset": "set", "scene": "one"}))
        np.savez(
            run / "evaluation_capture" / "camera_pairs.npz",
            index_a=np.array([0, 0, 1]),
            index_b=np.array([1, 2, 2]),
            rotation_degrees=np.array([1.0, 2.0, 9.0]) + model_index,
            translation_degrees=np.array([8.0, 7.0, 1.0]) + model_index,
        )
        rows.append({"model": model, "dataset": "set", "scene": "one", "run": str(run)})
    bundles = root / "bundles"
    bundles.mkdir()
    selected = bundles / "selected_runs.jsonl"
    selected.write_text("".join(json.dumps(row) + "\n" for row in rows))
    (bundles / "manifest.json").write_text(
        json.dumps({"models": ["a", "b"], "selected_runs_sha256": digest(selected), "test_time_policy_input": False}) + "\n"
    )
    return bundles


def test_renders_all_exact_extrema_without_threshold_or_top_k(tmp_path: Path) -> None:
    bundles = make_fixture(tmp_path)
    output = tmp_path / "rendered"
    manifest = render_extrema(bundles, output, max_side=16, jpeg_quality=90)
    assert manifest["selection_threshold"] is None
    assert manifest["top_k"] is None
    assert manifest["test_time_policy_input"] is False
    assert manifest["selected_scenes"] == 1
    scene = manifest["scene_outputs"][0]
    assert scene["selected_view_count"] >= 2
    for view in scene["views"]:
        assert digest(output / view["output_path"]) == view["output_sha256"]
        assert {reason["extremum"] for reason in view["selection_reasons"]} <= {"maximum", "minimum_control"}
    audit = audit_views(bundles, output)
    assert audit["passed"] is True
    assert audit["selected_views"] == manifest["selected_views"]


def test_audit_rejects_mutated_render(tmp_path: Path) -> None:
    bundles = make_fixture(tmp_path)
    output = tmp_path / "rendered"
    manifest = render_extrema(bundles, output, max_side=16, jpeg_quality=90)
    rendered = output / manifest["scene_outputs"][0]["views"][0]["output_path"]
    rendered.write_bytes(rendered.read_bytes() + b"mutated")
    with pytest.raises(ValueError, match="Rendered view hash differs"):
        audit_views(bundles, output)


def test_rejects_cross_model_input_mismatch(tmp_path: Path) -> None:
    bundles = make_fixture(tmp_path, mismatch=True)
    with pytest.raises(ValueError, match="Input population differs"):
        render_extrema(bundles, tmp_path / "rendered", max_side=16, jpeg_quality=90)
