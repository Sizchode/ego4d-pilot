import json
from pathlib import Path

import pytest

from scripts.audit.gt_geometry_factor_profile_outputs import audit_profiles
from scripts.failure_mining.analyze_gt_geometry_factor_profiles import analyze
from scripts.failure_mining.compare_baselines import sha256, write_csv
from tests.test_gt_geometry_factor_summary_output_audit import build_outputs


def build_profiles(tmp_path: Path) -> tuple[Path, Path]:
    _raw, summary, _raw_audit = build_outputs(tmp_path)
    profile_dir = tmp_path / "profiles"
    profile_dir.mkdir()
    profiles, aggregate, provenance = analyze(summary)
    case_path = profile_dir / "case_metric_factor_profiles.csv"
    aggregate_path = profile_dir / "model_metric_factor_summary.csv"
    write_csv(case_path, profiles)
    write_csv(aggregate_path, aggregate)
    (profile_dir / "manifest.json").write_text(
        json.dumps(
            {
                **provenance,
                "profile_output_sha256": sha256(case_path),
                "aggregate_output_sha256": sha256(aggregate_path),
            }
        )
    )
    return summary, profile_dir


def test_reproduces_factor_profile_outputs(tmp_path: Path) -> None:
    summary, profiles = build_profiles(tmp_path)
    result = audit_profiles(summary, profiles)
    assert result["passed"] is True
    assert result["case_metric_profiles"] == 12
    assert result["model_metric_rows"] == 12


def test_rejects_tampered_factor_profile_table(tmp_path: Path) -> None:
    summary, profiles = build_profiles(tmp_path)
    path = profiles / "case_metric_factor_profiles.csv"
    path.write_text(path.read_text().replace("0.01", "9.99", 1))
    with pytest.raises(ValueError, match="hash differs"):
        audit_profiles(summary, profiles)
