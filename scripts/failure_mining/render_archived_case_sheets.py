#!/usr/bin/env python3
"""Render model-aligned preview and camera-error sheets from evidence archives."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageOps, __version__ as pillow_version

from scripts.failure_mining.compare_baselines import sha256
from scripts.failure_mining.render_selected_input_views import slug


PANELS = ("preview.png", "camera_errors.png")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_manifest(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"Manifest is not an object: {path}")
    return value


def verified_cases(
    archive_dirs: list[Path], models: list[str]
) -> tuple[str, dict[str, dict[str, dict[str, Any]]], list[dict[str, Any]]]:
    require(archive_dirs and len(archive_dirs) == len(set(archive_dirs)), "Archive dirs must be unique")
    require(models and len(models) == len(set(models)), "Models must be unique")
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    archive_records = []
    datasets = set()
    for archive_dir in archive_dirs:
        manifest_path = archive_dir / "archive_manifest.json"
        manifest = read_manifest(manifest_path)
        datasets.add(manifest["dataset"])
        require(manifest.get("failure_labels") is None, "Archive contains failure labels")
        for case in manifest["cases"]:
            model, scene = case["model"], case["scene"]
            require(model in models, f"Unknown model in archive: {model}")
            require(model not in grouped[scene], f"Duplicate archived case: {(model, scene)}")
            case_root = archive_dir / case["archive_prefix"]
            files = {item["path"]: item for item in case["files"]}
            for panel in PANELS:
                require(panel in files, f"Missing panel record: {(model, scene, panel)}")
                path = case_root / panel
                require(path.is_file(), f"Missing panel file: {path}")
                require(path.stat().st_size == files[panel]["bytes"], f"Panel size mismatch: {path}")
                require(sha256(path) == files[panel]["sha256"], f"Panel hash mismatch: {path}")
            grouped[scene][model] = {"root": case_root, "files": files}
        archive_records.append(
            {
                "path": str(archive_dir.resolve()),
                "manifest_sha256": sha256(manifest_path),
                "cases": manifest["case_count"],
                "files": manifest["file_count"],
            }
        )
    require(len(datasets) == 1, f"Archives contain multiple datasets: {sorted(datasets)}")
    for scene, by_model in grouped.items():
        require(list(by_model) == models, f"Model order/population differs for {scene}")
    require(grouped, "No archived cases")
    return next(iter(datasets)), grouped, archive_records


def fit_image(path: Path, max_width: int, max_height: int) -> Image.Image:
    with Image.open(path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
    image.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
    return image


def render_sheets(
    archive_dirs: list[Path],
    models: list[str],
    output_dir: Path,
    *,
    max_panel_width: int,
    max_panel_height: int,
) -> dict[str, Any]:
    require(not output_dir.exists(), f"Refusing to overwrite output: {output_dir}")
    require(max_panel_width > 0 and max_panel_height > 0, "Panel bounds must be positive")
    dataset, grouped, archive_records = verified_cases(archive_dirs, models)
    font = ImageFont.load_default()
    margin, gap, label_height = 16, 12, 24

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent))
    try:
        outputs = []
        for scene, by_model in sorted(grouped.items()):
            loaded: dict[tuple[str, str], Image.Image] = {}
            column_widths = [0] * len(PANELS)
            row_heights = []
            for model in models:
                heights = []
                for column, panel in enumerate(PANELS):
                    image = fit_image(by_model[model]["root"] / panel, max_panel_width, max_panel_height)
                    loaded[(model, panel)] = image
                    column_widths[column] = max(column_widths[column], image.width)
                    heights.append(image.height)
                row_heights.append(max(heights) + label_height)
            label_width = max(96, max(int(font.getlength(model)) for model in models) + margin)
            width = margin * 2 + label_width + sum(column_widths) + gap * len(PANELS)
            title_height = label_height * 2
            height = margin * 2 + title_height + sum(row_heights) + gap * (len(models) - 1)
            sheet = Image.new("RGB", (width, height), "white")
            draw = ImageDraw.Draw(sheet)
            draw.text((margin, margin), f"{dataset} / {scene}", fill="black", font=font)
            x = margin + label_width
            for column, panel in enumerate(PANELS):
                draw.text((x, margin + label_height), panel, fill="black", font=font)
                x += column_widths[column] + gap
            y = margin + title_height
            for row_index, model in enumerate(models):
                draw.text((margin, y + label_height), model, fill="black", font=font)
                x = margin + label_width
                for column, panel in enumerate(PANELS):
                    image = loaded[(model, panel)]
                    sheet.paste(image, (x, y + label_height))
                    x += column_widths[column] + gap
                y += row_heights[row_index] + gap
            relative = f"{slug(scene)}.jpg"
            sheet.save(temporary / relative, "JPEG", quality=92, optimize=True)
            outputs.append(
                {
                    "scene": scene,
                    "output": relative,
                    "sha256": sha256(temporary / relative),
                    "width": sheet.width,
                    "height": sheet.height,
                }
            )
        manifest = {
            "schema_version": 1,
            "dataset": dataset,
            "models": models,
            "panels": list(PANELS),
            "scenes": sorted(grouped),
            "scene_count": len(grouped),
            "archives": archive_records,
            "outputs": outputs,
            "render": {
                "pillow_version": pillow_version,
                "max_panel_width": max_panel_width,
                "max_panel_height": max_panel_height,
                "jpeg_quality": 92,
            },
            "uses_official_evaluator_selection": True,
            "test_time_policy_input": False,
            "failure_labels": None,
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, output_dir)
        return manifest
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-dir", action="append", type=Path, required=True)
    parser.add_argument("--model", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-panel-width", type=int, default=720)
    parser.add_argument("--max-panel-height", type=int, default=520)
    args = parser.parse_args()
    manifest = render_sheets(
        args.archive_dir,
        args.model,
        args.output,
        max_panel_width=args.max_panel_width,
        max_panel_height=args.max_panel_height,
    )
    print(json.dumps({"scene_count": manifest["scene_count"], "scenes": manifest["scenes"]}, indent=2))


if __name__ == "__main__":
    main()
