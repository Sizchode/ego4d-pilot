import csv
import json
from pathlib import Path

import pytest
from PIL import Image

from scripts.audit.snapshot_review_outputs import audit
from scripts.failure_mining.compare_baselines import sha256, write_csv
from scripts.failure_mining.compare_view_error_profiles import compare


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_review(tmp_path: Path) -> dict[str, Path]:
    archive = tmp_path / "archive"
    archive.mkdir()
    cases = []
    case_rows = []
    view_rows = []
    snapshot_rows = []
    for model_index, model in enumerate(("a", "b")):
        prefix = f"{model}-scene"
        case_dir = archive / prefix
        case_dir.mkdir()
        cases.append(
            {
                "archive_prefix": prefix,
                "dataset": "set",
                "model": model,
                "scene": "scene",
            }
        )
        metric = float(model_index + 1)
        snapshot_rows.append(
            {
                "dataset": "set",
                "scene": "scene",
                "model": model,
                "state": "completed",
                "views": 3,
                "metrics": {"geometry": {"score": metric}},
            }
        )
        case_rows.append(
            {
                "case_dir": str(case_dir.resolve()),
                "dataset": "set",
                "scene": "scene",
                "model": f"full-{model}",
                "state": "completed",
                "validation_passed": "True",
                "views": "3",
                "metric.geometry.score": str(metric),
            }
        )
        for view_index in range(3):
            view_rows.append(
                {
                    "case_dir": str(case_dir.resolve()),
                    "dataset": "set",
                    "scene": "scene",
                    "model": f"full-{model}",
                    "view_index": str(view_index),
                    "incident_pair_count": "2",
                    "camera.rotation_degrees.p95": str(view_index + model_index),
                    "camera.translation_degrees.p95": str(2 * view_index + model_index),
                }
            )

    archive_manifest = {
        "dataset": "set",
        "models": ["a", "b"],
        "scenes": ["scene"],
        "case_count": 2,
        "cases": cases,
        "failure_labels": None,
    }
    archive_manifest_path = archive / "archive_manifest.json"
    archive_manifest_path.write_text(json.dumps(archive_manifest))
    case_path = tmp_path / "case.csv"
    view_path = tmp_path / "view.csv"
    snapshot_path = tmp_path / "snapshot.jsonl"
    write_rows(case_path, case_rows)
    write_rows(view_path, view_rows)
    snapshot_path.write_text("".join(json.dumps(row) + "\n" for row in snapshot_rows))

    sheets = tmp_path / "sheets"
    sheets.mkdir()
    sheet_path = sheets / "scene.jpg"
    Image.new("RGB", (24, 16), "white").save(sheet_path)
    sheets_manifest = {
        "dataset": "set",
        "models": ["a", "b"],
        "scenes": ["scene"],
        "test_time_policy_input": False,
        "failure_labels": None,
        "archives": [{"manifest_sha256": sha256(archive_manifest_path)}],
        "outputs": [
            {
                "scene": "scene",
                "output": "scene.jpg",
                "sha256": sha256(sheet_path),
                "width": 24,
                "height": 16,
            }
        ],
    }
    (sheets / "manifest.json").write_text(json.dumps(sheets_manifest))

    profiles = tmp_path / "profiles"
    profiles.mkdir()
    fields = ("camera.rotation_degrees.p95", "camera.translation_degrees.p95")
    pair_rows, summary_rows = compare(view_rows, fields=fields, datasets={"set"})
    pair_path = profiles / "pairwise_view_error_rank_agreement.csv"
    summary_path = profiles / "view_error_rank_agreement_summary.csv"
    write_csv(pair_path, pair_rows)
    write_csv(summary_path, summary_rows)
    (profiles / "manifest.json").write_text(
        json.dumps(
            {
                "input_sha256": sha256(view_path),
                "fields": list(fields),
                "uses_ground_truth_target": True,
                "test_time_policy_input": False,
                "pairwise_sha256": sha256(pair_path),
                "summary_sha256": sha256(summary_path),
            }
        )
    )
    return {
        "archive": archive,
        "case": case_path,
        "view": view_path,
        "snapshot": snapshot_path,
        "sheets": sheets,
        "profiles": profiles,
    }


def run_audit(paths: dict[str, Path]):
    return audit(
        archive_dir=paths["archive"],
        case_metrics_path=paths["case"],
        view_metrics_path=paths["view"],
        snapshot_metrics_path=paths["snapshot"],
        contact_sheets_dir=paths["sheets"],
        view_profiles_dir=paths["profiles"],
        dataset="set",
    )


def test_audits_snapshot_review_outputs(tmp_path: Path) -> None:
    result = run_audit(build_review(tmp_path))
    assert result["passed"] is True
    assert result["official_metric_comparisons"] == 2
    assert result["per_view_rows"] == 6
    assert result["profile_pairwise_rows"] == 2


def test_rejects_case_metric_that_differs_from_snapshot(tmp_path: Path) -> None:
    paths = build_review(tmp_path)
    rows = list(csv.DictReader(paths["case"].open(newline="", encoding="utf-8")))
    rows[0]["metric.geometry.score"] = "99.0"
    write_rows(paths["case"], rows)
    with pytest.raises(ValueError, match="Official metric differs"):
        run_audit(paths)
