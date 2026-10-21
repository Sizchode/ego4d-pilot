import json
from pathlib import Path

from PIL import Image

from scripts.failure_mining.compare_baselines import sha256
from scripts.failure_mining.render_archived_case_sheets import render_sheets


def build_case(root: Path, model: str, scene: str, color: str) -> dict:
    prefix = f"{model}-{scene}"
    case_root = root / prefix
    case_root.mkdir(parents=True)
    files = []
    for name in ("preview.png", "camera_errors.png"):
        path = case_root / name
        Image.new("RGB", (32, 24), color).save(path)
        files.append({"path": name, "bytes": path.stat().st_size, "sha256": sha256(path)})
    return {
        "model": model,
        "dataset": "set",
        "scene": scene,
        "archive_prefix": prefix,
        "files": files,
    }


def test_renders_model_aligned_archived_sheets(tmp_path: Path) -> None:
    archive = tmp_path / "archive"
    archive.mkdir()
    cases = [
        build_case(archive, "a", "scene", "red"),
        build_case(archive, "b", "scene", "blue"),
    ]
    (archive / "archive_manifest.json").write_text(
        json.dumps(
            {
                "dataset": "set",
                "case_count": 2,
                "file_count": 4,
                "cases": cases,
                "failure_labels": None,
            }
        )
    )
    output = tmp_path / "sheets"

    manifest = render_sheets(
        [archive], ["a", "b"], output, max_panel_width=64, max_panel_height=64
    )

    assert manifest["scene_count"] == 1
    assert manifest["scenes"] == ["scene"]
    assert manifest["models"] == ["a", "b"]
    assert manifest["test_time_policy_input"] is False
    assert (output / "scene.jpg").is_file()
    assert sha256(output / "scene.jpg") == manifest["outputs"][0]["sha256"]
