import json
from pathlib import Path

import pytest

from scripts.failure_mining.compare_baselines import sha256
from scripts.failure_mining.combine_gt_extrinsics_interventions import combine
from scripts.failure_mining.summarize_gt_extrinsics_intervention import summarize


def metric_block(offset: float) -> dict[str, float]:
    return {
        "acc": 0.1 + offset,
        "comp": 0.2 + offset,
        "overall": 0.15 + offset,
        "precision": 0.7 + offset,
        "recall": 0.6 + offset,
        "fscore": 0.646 + offset,
    }


def write_intervention(
    tmp_path: Path,
    *,
    name: str = "intervention",
    scenes: tuple[str, ...] = ("scene-1", "scene-2"),
) -> Path:
    root = tmp_path / name
    root.mkdir()
    rows = []
    for model in ("model-a", "model-b"):
        for scene in scenes:
            official = metric_block(0.0)
            control = metric_block(0.01)
            intervention = {
                key: value + (-0.03 if key in {"acc", "comp", "overall"} else 0.03)
                for key, value in control.items()
            }
            rows.append(
                {
                    "model": model,
                    "dataset": "scannetpp",
                    "scene": scene,
                    "official_native_metrics": official,
                    "native_resample_metrics": control,
                    "gt_extrinsics_metrics": intervention,
                    "native_resample_delta_vs_official": {
                        key: control[key] - official[key] for key in official
                    },
                    "gt_extrinsics_delta_vs_native_resample": {
                        key: intervention[key] - control[key] for key in official
                    },
                    "uses_ground_truth_intervention": True,
                    "test_time_policy_input": False,
                }
            )
    results = root / "results.jsonl"
    results.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    manifest = {
        "cases": 2 * len(scenes),
        "scenes": list(scenes),
        "models": ["a", "b"],
        "results_sha256": sha256(results),
        "uses_ground_truth_intervention": True,
        "test_time_policy_input": False,
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


def test_summary_uses_paired_control_and_metric_direction(tmp_path: Path) -> None:
    root = write_intervention(tmp_path)
    cases, summaries, provenance = summarize(root)
    assert len(cases) == 24
    assert len(summaries) == 12
    assert provenance["cases"] == 4
    assert provenance["success_threshold"] is None
    assert all(row["improvement_vs_control"] == pytest.approx(0.03) for row in cases)
    assert all(row["abs_control_delta_vs_official"] == pytest.approx(0.01) for row in cases)
    assert all(row["positive_improvement_cases"] == 2 for row in summaries)
    assert all(row["negative_improvement_cases"] == 0 for row in summaries)


def test_summary_rejects_result_hash_mismatch(tmp_path: Path) -> None:
    root = write_intervention(tmp_path)
    with (root / "results.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("{}\n")
    with pytest.raises(ValueError, match="hash"):
        summarize(root)


def test_summary_rejects_inconsistent_stored_delta(tmp_path: Path) -> None:
    root = write_intervention(tmp_path)
    rows = [json.loads(line) for line in (root / "results.jsonl").read_text().splitlines()]
    rows[0]["native_resample_delta_vs_official"]["acc"] = 999.0
    results = root / "results.jsonl"
    results.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["results_sha256"] = sha256(results)
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="Control delta mismatch"):
        summarize(root)


def test_combiner_reaudits_disjoint_case_populations(tmp_path: Path) -> None:
    first = write_intervention(tmp_path, name="first", scenes=("scene-1",))
    second = write_intervention(tmp_path, name="second", scenes=("scene-2", "scene-3"))
    cases, summaries, provenance = combine([first, second])
    assert len(cases) == 36
    assert len(summaries) == 12
    assert provenance["source_count"] == 2
    assert provenance["cases"] == 6
    assert provenance["scenes"] == ["scene-1", "scene-2", "scene-3"]
    assert all(row["cases"] == 3 for row in summaries)


def test_combiner_rejects_overlapping_cases(tmp_path: Path) -> None:
    first = write_intervention(tmp_path, name="first", scenes=("scene-1",))
    second = write_intervention(tmp_path, name="second", scenes=("scene-1",))
    with pytest.raises(ValueError, match="Duplicate combined intervention row"):
        combine([first, second])
