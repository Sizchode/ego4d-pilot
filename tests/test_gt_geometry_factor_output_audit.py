import json
from pathlib import Path

import pytest

from scripts.audit.gt_geometry_factor_outputs import (
    METRICS,
    STAGES,
    audit_factor_outputs,
)
from scripts.audit.analysis_outputs import sha256


def write_fixture(root: Path) -> None:
    models = ["m1", "m2"]
    scenes = ["s1", "s2"]
    rows = []
    for scene in scenes:
        for model in models:
            case = root / f"{model}-{scene}"
            case.mkdir(parents=True)
            meshes = {
                "gt_extrinsics_intrinsics_mesh.ply": f"ek-{model}-{scene}".encode(),
                "gt_extrinsics_intrinsics_depth_mesh.ply": f"ekd-{model}-{scene}".encode(),
            }
            for name, value in meshes.items():
                (case / name).write_bytes(value)
            metrics = {name: float(index + 1) / 10 for index, name in enumerate(METRICS)}
            row = {
                "model": model,
                "model_repository": f"owner/{model}",
                "dataset": "scannetpp",
                "scene": scene,
                "views": 100,
                "seed": 42,
                "run": f"/runs/{model}-{scene}",
                "stages": list(STAGES),
                "official_native_metrics": metrics,
                "gt_extrinsics_metrics": metrics,
                "gt_extrinsics_intrinsics_metrics": metrics,
                "gt_extrinsics_intrinsics_depth_metrics": metrics,
                "image_population_sha256": "1" * 64 if scene == "s1" else "2" * 64,
                "gt_meta_sha256": "3" * 64 if scene == "s1" else "4" * 64,
                "gt_depth_population_sha256": "5" * 64 if scene == "s1" else "6" * 64,
                "gt_extrinsics_intrinsics_mesh_sha256": sha256(case / "gt_extrinsics_intrinsics_mesh.ply"),
                "gt_extrinsics_intrinsics_depth_mesh_sha256": sha256(case / "gt_extrinsics_intrinsics_depth_mesh.ply"),
                "gt_point_count": 100,
                "gt_extrinsics_intrinsics_point_count": 90,
                "gt_extrinsics_intrinsics_depth_point_count": 95,
                "capture_sha256": {
                    "alignment_000.npz": "7" * 64,
                    "fused_mesh_000.ply": "8" * 64,
                    "fusion_inputs_000.npz": "9" * 64,
                    "surface_distance_000.npz": "a" * 64,
                    "surface_distance_001.npz": "b" * 64,
                },
                "uses_ground_truth_intervention": True,
                "test_time_policy_input": False,
            }
            (case / "result.json").write_text(json.dumps(row, sort_keys=True) + "\n")
            rows.append(row)
    results = root / "results.jsonl"
    results.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    manifest = {
        "dataset": "scannetpp",
        "models": models,
        "scenes": scenes,
        "cases": len(rows),
        "seed": 42,
        "stages": list(STAGES),
        "results_sha256": sha256(results),
        "uses_ground_truth_intervention": True,
        "test_time_policy_input": False,
        "success_threshold": None,
    }
    (root / "manifest.json").write_text(json.dumps(manifest) + "\n")


def test_audit_accepts_complete_cartesian_population(tmp_path: Path) -> None:
    write_fixture(tmp_path)
    report = audit_factor_outputs(
        tmp_path, expected_models=["m1", "m2"], expected_scenes=["s1", "s2"]
    )
    assert report["passed"] is True
    assert report["cases"] == 4
    assert len(report["case_artifact_sha256"]) == 4


def test_audit_rejects_cross_model_population_change(tmp_path: Path) -> None:
    write_fixture(tmp_path)
    results = tmp_path / "results.jsonl"
    rows = [json.loads(line) for line in results.read_text().splitlines()]
    rows[1]["gt_depth_population_sha256"] = "f" * 64
    results.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    manifest["results_sha256"] = sha256(results)
    (tmp_path / "manifest.json").write_text(json.dumps(manifest) + "\n")
    (tmp_path / "m2-s1" / "result.json").write_text(
        json.dumps(rows[1], sort_keys=True) + "\n"
    )
    with pytest.raises(ValueError, match="Cross-model GT population differs"):
        audit_factor_outputs(tmp_path)


def test_audit_rejects_mesh_hash_mismatch(tmp_path: Path) -> None:
    write_fixture(tmp_path)
    (tmp_path / "m1-s1" / "gt_extrinsics_intrinsics_mesh.ply").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Mesh hash mismatch"):
        audit_factor_outputs(tmp_path)


def test_audit_rejects_incomplete_cartesian_population(tmp_path: Path) -> None:
    write_fixture(tmp_path)
    results = tmp_path / "results.jsonl"
    rows = [json.loads(line) for line in results.read_text().splitlines()][:-1]
    results.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    manifest["cases"] = len(rows)
    manifest["results_sha256"] = sha256(results)
    (tmp_path / "manifest.json").write_text(json.dumps(manifest) + "\n")
    with pytest.raises(ValueError, match="model × scene"):
        audit_factor_outputs(tmp_path)
