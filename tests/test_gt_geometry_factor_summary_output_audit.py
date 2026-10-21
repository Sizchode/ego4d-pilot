import json
from pathlib import Path

import pytest

from scripts.audit.gt_geometry_factor_summary_outputs import audit_summary
from scripts.failure_mining.compare_baselines import sha256, write_csv
from scripts.failure_mining.summarize_gt_extrinsics_intervention import METRIC_DIRECTIONS
from scripts.failure_mining.summarize_gt_geometry_factors import STAGES, summarize


def build_case(model: str, scene: str, offset: float) -> dict:
    blocks = {}
    for stage_index, stage in enumerate(STAGES):
        values = {}
        for metric, direction in METRIC_DIRECTIONS.items():
            base = offset + list(METRIC_DIRECTIONS).index(metric) / 100
            values[metric] = base + stage_index / 100 if direction == "higher_is_better" else base - stage_index / 100
        blocks[f"{stage}_metrics"] = values
    return {
        "model": model,
        "scene": scene,
        "stages": list(STAGES),
        **blocks,
    }


def build_outputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    raw = tmp_path / "raw"
    raw.mkdir()
    models, scenes = ["a", "b"], ["scene"]
    rows = [build_case(model, "scene", float(index + 1)) for index, model in enumerate(models)]
    results = raw / "results.jsonl"
    results.write_text("".join(json.dumps(row) + "\n" for row in rows))
    manifest = {
        "models": models,
        "scenes": scenes,
        "cases": 2,
        "stages": list(STAGES),
        "results_sha256": sha256(results),
        "uses_ground_truth_intervention": True,
        "test_time_policy_input": False,
        "success_threshold": None,
    }
    manifest_path = raw / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    raw_audit = tmp_path / "raw-audit.json"
    raw_audit.write_text(
        json.dumps(
            {
                "passed": True,
                "cases": 2,
                "models": models,
                "scenes": scenes,
                "factor_manifest_sha256": sha256(manifest_path),
                "factor_results_sha256": sha256(results),
            }
        )
    )

    summary = tmp_path / "summary"
    summary.mkdir()
    case_rows, aggregate_rows, provenance = summarize(raw)
    case_path = summary / "case_metric_stages.csv"
    aggregate_path = summary / "model_metric_stage_summary.csv"
    write_csv(case_path, case_rows)
    write_csv(aggregate_path, aggregate_rows)
    (summary / "manifest.json").write_text(
        json.dumps(
            {
                **provenance,
                "case_output_sha256": sha256(case_path),
                "summary_output_sha256": sha256(aggregate_path),
            }
        )
    )
    return raw, summary, raw_audit


def test_reproduces_factor_summary_outputs(tmp_path: Path) -> None:
    raw, summary, raw_audit = build_outputs(tmp_path)
    result = audit_summary(raw, summary, raw_audit)
    assert result["passed"] is True
    assert result["cases"] == 2
    assert result["case_metric_rows"] == 12


def test_rejects_tampered_summary_table(tmp_path: Path) -> None:
    raw, summary, raw_audit = build_outputs(tmp_path)
    path = summary / "case_metric_stages.csv"
    path.write_text(path.read_text().replace("0.03", "9.99", 1))
    with pytest.raises(ValueError, match="hash differs"):
        audit_summary(raw, summary, raw_audit)
