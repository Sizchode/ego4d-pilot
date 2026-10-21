"""Materialize an offline, cross-model visual report for selected failure cases."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import io
import json
import os
import shutil
import tarfile
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image

from scripts.audit.analysis_outputs import load_json, require, sha256
from scripts.failure_mining.archive_case_bundles import REQUIRED_FILES
from scripts.failure_mining.render_selected_input_views import slug


DISPLAY_JSON = (
    "run_config.json",
    "status.json",
    "validation.json",
    "diagnostics/protocol.json",
)
DISPLAY_IMAGES = ("preview.png", "camera_errors.png")


def read_csv(path: Path) -> list[dict[str, str]]:
    require(path.is_file(), f"Required file is missing: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _tar_payloads(path: Path, expected_hash: str) -> tuple[dict[str, bytes], dict[str, Any]]:
    require(sha256(path) == expected_hash, f"Review tar hash mismatch: {path}")
    payloads: dict[str, bytes] = {}
    with tarfile.open(path, "r") as archive:
        for member in archive.getmembers():
            require(member.isfile(), f"Unexpected non-file tar member: {member.name}")
            require(member.name not in payloads, f"Duplicate tar member: {member.name}")
            extracted = archive.extractfile(member)
            require(extracted is not None, f"Cannot read tar member: {member.name}")
            payloads[member.name] = extracted.read()
    require("archive_manifest.json" in payloads, f"Archive manifest is missing: {path}")
    manifest = json.loads(payloads["archive_manifest.json"])
    require(set(manifest.get("required_files", [])) == set(REQUIRED_FILES), f"Required evidence set differs: {path}")
    require(manifest.get("file_count") == len(manifest.get("cases", [])) * len(REQUIRED_FILES), f"Archive file count differs: {path}")
    expected_members = {"archive_manifest.json"}
    for case in manifest.get("cases", []):
        prefix = case["archive_prefix"]
        files = {row["path"]: row for row in case.get("files", [])}
        require(set(files) == set(REQUIRED_FILES), f"Case evidence set differs: {(path, prefix)}")
        for relative, metadata in files.items():
            member = f"{prefix}/{relative}"
            expected_members.add(member)
            require(member in payloads, f"Archived evidence is missing: {member}")
            payload = payloads[member]
            require(len(payload) == metadata["bytes"], f"Archived byte count differs: {member}")
            require(hashlib.sha256(payload).hexdigest() == metadata["sha256"], f"Archived hash differs: {member}")
    require(set(payloads) == expected_members, f"Unexpected archive member population: {path}")
    return payloads, manifest


def _write_asset(root: Path, relative: Path, payload: bytes, outputs: dict[str, str]) -> None:
    destination = root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(payload)
    outputs[str(relative)] = sha256(destination)


def _image_dimensions(payload: bytes, identity: str) -> list[int]:
    with Image.open(io.BytesIO(payload)) as image:
        image.load()
        require(image.width > 0 and image.height > 0, f"Invalid image dimensions: {identity}")
        return [image.width, image.height]


def build_report(
    bundles_dir: Path,
    views_dir: Path,
    views_audit_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    require(not output_dir.exists(), f"Refusing to overwrite output: {output_dir}")
    bundle_manifest_path = bundles_dir / "manifest.json"
    selection_path = bundles_dir / "selected_review_cases.csv"
    views_manifest_path = views_dir / "manifest.json"
    bundle_manifest = load_json(bundle_manifest_path)
    views_manifest = load_json(views_manifest_path)
    views_audit = load_json(views_audit_path)
    require(views_audit.get("passed") is True, "Extremal-view audit did not pass")
    require(views_audit.get("bundle_manifest_sha256") == sha256(bundle_manifest_path), "View audit covers different bundles")
    require(views_audit.get("views_manifest_sha256") == sha256(views_manifest_path), "View audit covers different rendered views")
    require(views_manifest.get("test_time_policy_input") is False, "Rendered views are policy input")
    require(bundle_manifest.get("test_time_policy_input") is False, "Review bundles are policy input")
    models = bundle_manifest.get("models")
    require(models == views_manifest.get("models"), "Bundle and view model order differ")

    selection_by_scene: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in read_csv(selection_path):
        selection_by_scene[(row["dataset"], row["scene"])].append(row)
    require(selection_by_scene, "No selected review cases")
    view_scenes = {
        (row["dataset"], row["scene"]): row for row in views_manifest.get("scene_outputs", [])
    }
    require(set(view_scenes) == set(selection_by_scene), "Rendered and selected scene populations differ")

    case_payloads: dict[tuple[str, str, str], tuple[dict[str, bytes], dict[str, Any], str]] = {}
    tar_provenance = []
    for bundle in bundle_manifest.get("bundles", []):
        tar_path = bundles_dir / bundle["tar"]
        payloads, archive_manifest = _tar_payloads(tar_path, bundle["tar_sha256"])
        require(archive_manifest.get("dataset") == bundle["dataset"], f"Archive dataset differs: {tar_path}")
        require(archive_manifest.get("models") == models, f"Archive model order differs: {tar_path}")
        require(archive_manifest.get("scenes") == bundle["scenes"], f"Archive scene order differs: {tar_path}")
        for case in archive_manifest["cases"]:
            key = (case["model"], case["dataset"], case["scene"])
            require(key not in case_payloads, f"Duplicate archived case: {key}")
            case_payloads[key] = (payloads, case, bundle["tar"])
        tar_provenance.append({"path": bundle["tar"], "sha256": bundle["tar_sha256"]})
    expected_cases = {
        (model, dataset, scene)
        for dataset, scene in selection_by_scene
        for model in models
    }
    require(set(case_payloads) == expected_cases, "Archived cross-model case population differs")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent))
    outputs: dict[str, str] = {}
    try:
        scene_records = []
        sections = []
        for dataset, scene in sorted(selection_by_scene):
            scene_root = Path("assets") / slug(dataset) / slug(scene)
            view_row = view_scenes[(dataset, scene)]
            view_records = []
            view_cards = []
            for view in view_row["views"]:
                source = views_dir / view["output_path"]
                require(source.is_file() and sha256(source) == view["output_sha256"], f"Rendered input view differs: {source}")
                relative = scene_root / "inputs" / f"view-{int(view['index']):03d}.jpg"
                _write_asset(temporary, relative, source.read_bytes(), outputs)
                view_records.append({"index": view["index"], "path": str(relative), "sha256": outputs[str(relative)], "selection_reasons": view["selection_reasons"]})
                view_cards.append(f'<figure><img src="{html.escape(str(relative))}"><figcaption>Input view {int(view["index"])}</figcaption></figure>')

            model_records = []
            model_cards = []
            for model in models:
                payloads, case, tar_name = case_payloads[(model, dataset, scene)]
                prefix = case["archive_prefix"]
                assets: dict[str, Any] = {}
                image_cards = []
                for relative_name in DISPLAY_IMAGES:
                    member = f"{prefix}/{relative_name}"
                    payload = payloads[member]
                    dimensions = _image_dimensions(payload, member)
                    relative = scene_root / slug(model) / relative_name
                    _write_asset(temporary, relative, payload, outputs)
                    assets[relative_name] = {"path": str(relative), "sha256": outputs[str(relative)], "dimensions": dimensions}
                    image_cards.append(f'<figure><img src="{html.escape(str(relative))}"><figcaption>{html.escape(relative_name)}</figcaption></figure>')
                json_links = []
                for relative_name in DISPLAY_JSON:
                    member = f"{prefix}/{relative_name}"
                    payload = payloads[member]
                    json.loads(payload)
                    relative = scene_root / slug(model) / relative_name
                    _write_asset(temporary, relative, payload, outputs)
                    assets[relative_name] = {"path": str(relative), "sha256": outputs[str(relative)]}
                    json_links.append(f'<a href="{html.escape(str(relative))}">{html.escape(relative_name)}</a>')
                model_records.append({"model": model, "source_tar": tar_name, "archive_prefix": prefix, "assets": assets})
                model_cards.append(
                    '<article class="model"><h3>' + html.escape(model) + '</h3><div class="images">' + "".join(image_cards) + '</div><p>' + " · ".join(json_links) + '</p></article>'
                )

            reasons = selection_by_scene[(dataset, scene)]
            reason_rows = "".join(
                "<tr>" + "".join(f"<td>{html.escape(row.get(field, ''))}</td>" for field in ("model", "metric", "selection_reasons")) + "</tr>"
                for row in reasons
            )
            sections.append(
                f'<section><h2>{html.escape(dataset)} / {html.escape(scene)}</h2>'
                '<table><thead><tr><th>Model</th><th>Metric</th><th>Selection reason</th></tr></thead><tbody>' + reason_rows + '</tbody></table>'
                '<h3>Selected source views</h3><div class="images">' + "".join(view_cards) + '</div>' + "".join(model_cards) + '</section>'
            )
            scene_records.append({"dataset": dataset, "scene": scene, "selection_rows": reasons, "views": view_records, "models": model_records})

        stylesheet = """
body{font-family:system-ui,sans-serif;margin:2rem;color:#202124;background:#f7f7f7}h1,h2{margin:.4rem 0 1rem}section,.model{background:white;border:1px solid #ddd;border-radius:10px;padding:1rem;margin:1rem 0}.images{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:.8rem}figure{margin:0}img{display:block;max-width:100%;height:auto;border:1px solid #ccc}figcaption{font-size:.85rem;color:#555;margin-top:.25rem}table{border-collapse:collapse;width:100%;font-size:.9rem}th,td{border:1px solid #ddd;padding:.4rem;text-align:left;vertical-align:top}a{margin-right:.6rem}
""".strip()
        document = (
            '<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>Baseline failure review</title><style>' + stylesheet + '</style></head><body>'
            '<h1>Cross-model baseline failure review</h1><p>Offline evidence selected by official evaluator extrema. These cases and labels must not be used as test-time policy inputs.</p>'
            + "".join(sections) + '</body></html>\n'
        )
        index_path = temporary / "index.html"
        index_path.write_text(document, encoding="utf-8")
        outputs["index.html"] = sha256(index_path)
        manifest = {
            "schema_version": 1,
            "bundle_manifest_sha256": sha256(bundle_manifest_path),
            "selection_cases_sha256": sha256(selection_path),
            "views_manifest_sha256": sha256(views_manifest_path),
            "views_audit_sha256": sha256(views_audit_path),
            "models": models,
            "selected_scenes": len(scene_records),
            "expanded_cross_model_cases": len(case_payloads),
            "scene_records": scene_records,
            "source_tars": tar_provenance,
            "outputs": outputs,
            "failure_labels": None,
            "selection_threshold": None,
            "top_k": None,
            "uses_ground_truth_selection": True,
            "intended_use": "offline human failure analysis",
            "test_time_policy_input": False,
        }
        (temporary / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, output_dir)
        return manifest
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundles", type=Path, required=True)
    parser.add_argument("--views", type=Path, required=True)
    parser.add_argument("--views-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = build_report(args.bundles, args.views, args.views_audit, args.output)
    print(json.dumps({key: manifest[key] for key in ("selected_scenes", "expanded_cross_model_cases", "test_time_policy_input")}, indent=2))


if __name__ == "__main__":
    main()
