import csv
import json
from pathlib import Path

import pytest

from scripts.audit.final_failure_mining_outputs import (
    OUTPUTS,
    audit_final_failure_mining,
)
from scripts.failure_mining.analyze_backbone_oracle import analyze as analyze_oracle
from scripts.failure_mining.analyze_final_cross_model import (
    build_cross_model_consensus,
    build_pose_geometry_case_profiles,
)
from scripts.failure_mining.compare_baselines import sha256, write_csv
from scripts.failure_mining.compare_metric_rankings import compare as compare_metrics


def make_fixture(root: Path) -> tuple[Path, Path]:
    cross = root / "cross"
    analysis = root / "final"
    cross.mkdir()
    analysis.mkdir()
    models = ["a", "b"]
    rows = []
    for model, pose, geometry in (
        ("a", [0.2, 0.8], [0.9, 0.1]),
        ("b", [0.5, 0.8], [0.8, 0.4]),
    ):
        for index, scene in enumerate(("one", "two")):
            for metric, value in (("pose", pose[index]), ("geometry", geometry[index])):
                rank = 1 if value == min(pose if metric == "pose" else geometry) else 2
                rows.append(
                    {
                        "model": model,
                        "dataset": "set",
                        "scene": scene,
                        "metric": metric,
                        "direction": "lower_is_worse",
                        "value": value,
                        "failure_rank": rank,
                        "population": 2,
                        "worstness_fraction": 1.0 if rank == 1 else 0.0,
                        "run": f"/{model}/{scene}",
                    }
                )
    rankings = cross / "model_metric_rank_rows.csv"
    write_csv(rankings, rows)
    cross_manifest = {
        "models": models,
        "scene_population": 2,
        "inputs": {},
    }
    (cross / "manifest.json").write_text(json.dumps(cross_manifest) + "\n")

    oracle_scene, oracle_summary = analyze_oracle([{key: str(value) for key, value in row.items()} for row in rows])
    outputs = {
        "oracle_scene_headroom": oracle_scene,
        "oracle_summary": oracle_summary,
        "metric_rank_agreement": compare_metrics([{key: str(value) for key, value in row.items()} for row in rows], "pose", "geometry"),
        "cross_model_consensus_failure_ranks": build_cross_model_consensus([{key: str(value) for key, value in row.items()} for row in rows]),
        "pose_geometry_case_profiles": build_pose_geometry_case_profiles([{key: str(value) for key, value in row.items()} for row in rows], "pose", "geometry"),
        "state_failure_rank_correlations": [],
    }
    output_meta = {}
    for name, filename in OUTPUTS.items():
        path = analysis / filename
        write_csv(path, outputs[name])
        output_meta[name] = {"path": filename, "rows": len(outputs[name]), "sha256": sha256(path)}
    manifest = {
        "cross_manifest_sha256": sha256(cross / "manifest.json"),
        "cross_rankings_sha256": sha256(rankings),
        "models": models,
        "scene_population": 2,
        "metric_left": "pose",
        "metric_right": "geometry",
        "outputs": output_meta,
        "failure_labels": None,
        "quality_thresholds": None,
        "composite_score": None,
    }
    (analysis / "manifest.json").write_text(json.dumps(manifest) + "\n")
    return analysis, cross


def test_audit_accepts_complete_final_failure_mining(tmp_path: Path) -> None:
    analysis, cross = make_fixture(tmp_path)
    report = audit_final_failure_mining(analysis, cross, expected_scene_population=2)
    assert report["passed"] is True
    assert report["scene_population"] == 2


def test_audit_rejects_semantically_changed_consensus(tmp_path: Path) -> None:
    analysis, cross = make_fixture(tmp_path)
    path = analysis / "cross_model_consensus_failure_ranks.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows[0]["worstness_mean"] = "0.123"
    write_csv(path, rows)
    manifest = json.loads((analysis / "manifest.json").read_text())
    manifest["outputs"]["cross_model_consensus_failure_ranks"]["sha256"] = sha256(path)
    (analysis / "manifest.json").write_text(json.dumps(manifest) + "\n")
    with pytest.raises(ValueError, match="Consensus worstness_mean mismatch"):
        audit_final_failure_mining(analysis, cross)
