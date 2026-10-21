import csv
import json
from pathlib import Path

from scripts.failure_mining.analyze_final_cross_model import (
    analyze_final,
    build_cross_model_consensus,
    build_pose_geometry_case_profiles,
)
from scripts.failure_mining.compare_baselines import sha256
from tests.test_cross_model_output_audit import make_cross_outputs, write_json


EVIDENCE_FIELDS = [
    "fusion.retained_ratio",
    "diagnostic.projectable_ratio",
    "diagnostic.absolute_relative_z.p95",
    "diagnostic.roundtrip_pixels.p95",
]


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_final_mining_consumes_manifest_driven_model_inputs(tmp_path: Path) -> None:
    cross_dir, _visibility, _batches = make_cross_outputs(tmp_path)
    manifest_path = cross_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    cross_rows = []
    values = {
        "model-a": {
            "metric.pose.auc30": [0.1, 0.9],
            "metric.recon_unposed.fscore": [0.6, 0.7],
        },
        "model-b": {
            "metric.pose.auc30": [0.3, 0.9],
            "metric.recon_unposed.fscore": [0.5, 0.6],
        },
    }
    for model_index, model in enumerate(manifest["models"]):
        analysis_dir = Path(manifest["inputs"][model]["analysis_dir"])
        with (analysis_dir / "scenes.csv").open(newline="", encoding="utf-8") as handle:
            scenes = list(csv.DictReader(handle))
        for scene_index, row in enumerate(scenes):
            for field_index, field in enumerate(EVIDENCE_FIELDS):
                row[field] = str((model_index + 1) * 10 + scene_index + field_index / 10)
        write_csv(analysis_dir / "scenes.csv", scenes)

        ranking_rows = []
        for metric in ("metric.pose.auc30", "metric.recon_unposed.fscore"):
            for scene_index, scene in enumerate(("one", "two")):
                value = values[model][metric][scene_index]
                ranking_rows.append(
                    {
                        "dataset": "set",
                        "scene": scene,
                        "metric": metric,
                        "direction": "lower_is_worse",
                        "failure_rank": scene_index + 1,
                        "population": 2,
                        "worstness_fraction": 1.0 - scene_index,
                        "run": f"/{model}/{scene}",
                        "value": value,
                    }
                )
                cross_rows.append({"model": model, **ranking_rows[-1]})
        write_csv(analysis_dir / "rankings.csv", ranking_rows)
        manifest["inputs"][model]["scenes_sha256"] = sha256(analysis_dir / "scenes.csv")
        manifest["inputs"][model]["rankings_sha256"] = sha256(analysis_dir / "rankings.csv")

    write_csv(cross_dir / "model_metric_rank_rows.csv", cross_rows)
    manifest["outputs"]["model_metric_rank_rows"] = len(cross_rows)
    write_json(manifest_path, manifest)

    outputs, provenance = analyze_final(
        cross_dir,
        "metric.pose.auc30",
        "metric.recon_unposed.fscore",
        EVIDENCE_FIELDS,
    )
    assert provenance["models"] == ["model-a", "model-b"]
    assert len(outputs["oracle_summary"]) == 2
    assert len(outputs["oracle_scene_headroom"]) == 4
    assert len(outputs["metric_rank_agreement"]) == 2
    assert len(outputs["cross_model_consensus_failure_ranks"]) == 4
    assert len(outputs["pose_geometry_case_profiles"]) == 4
    assert len(outputs["state_failure_rank_correlations"]) == 16


def test_case_profiles_rank_consensus_and_pose_geometry_disagreement() -> None:
    rows = []
    for model, pose, geometry in (
        ("a", [1.0, 0.0], [0.0, 1.0]),
        ("b", [1.0, 0.0], [0.5, 0.5]),
    ):
        for index, scene in enumerate(("hard", "easy")):
            for metric, worstness in (("pose", pose[index]), ("geometry", geometry[index])):
                rows.append(
                    {
                        "model": model,
                        "dataset": "set",
                        "scene": scene,
                        "metric": metric,
                        "direction": "lower_is_worse",
                        "value": str(worstness),
                        "failure_rank": str(index + 1),
                        "population": "2",
                        "worstness_fraction": str(worstness),
                        "run": f"/{model}/{scene}",
                    }
                )
    consensus = build_cross_model_consensus(rows)
    pose_consensus = {row["scene"]: row for row in consensus if row["metric"] == "pose"}
    assert pose_consensus["hard"]["consensus_failure_rank"] == 1.0
    profiles = build_pose_geometry_case_profiles(rows, "pose", "geometry")
    a_hard = next(row for row in profiles if row["model"] == "a" and row["scene"] == "hard")
    assert a_hard["rank_profile"] == "pose_ranked_worse_than_geometry"
    assert a_hard["pose_geometry_disagreement_rank"] == 1.5
