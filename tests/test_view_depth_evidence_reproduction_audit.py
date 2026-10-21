from pathlib import Path

import pytest

from scripts.audit.view_depth_evidence_reproduction import audit_reproduction
from scripts.failure_mining.compare_baselines import write_csv


def build(root: Path, *, p_value: float = 0.1, rho: float = 0.5) -> None:
    root.mkdir()
    write_csv(root / "merged_view_evidence.csv", [{"dataset": "d", "scene": "s", "view_index": 0}])
    write_csv(
        root / "scene_view_rank_correlations.csv",
        [{
            "dataset": "d",
            "scene": "s",
            "target_field": "target",
            "evidence_field": "evidence",
            "views": 10,
            "spearman_rho": rho,
            "p_value_two_sided_descriptive": p_value,
        }],
    )
    write_csv(root / "aggregate_view_rank_correlations.csv", [{"target_field": "target", "evidence_field": "evidence"}])


def test_accepts_machine_precision_descriptive_p_difference(tmp_path: Path) -> None:
    stored, reproduced = tmp_path / "stored", tmp_path / "reproduced"
    build(stored, p_value=0.1)
    build(reproduced, p_value=0.1 + 1e-16)
    report = audit_reproduction(stored, reproduced)
    assert report["passed"] is True
    assert report["scene_descriptive_p_value_maximum_absolute_difference"] > 0


def test_rejects_changed_spearman_rho(tmp_path: Path) -> None:
    stored, reproduced = tmp_path / "stored", tmp_path / "reproduced"
    build(stored, rho=0.5)
    build(reproduced, rho=0.4)
    with pytest.raises(ValueError, match="Spearman rho differs"):
        audit_reproduction(stored, reproduced)
