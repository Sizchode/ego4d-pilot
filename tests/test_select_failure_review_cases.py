import csv
import json
from pathlib import Path

import pytest

from scripts.failure_mining.compare_baselines import sha256, write_csv
from scripts.failure_mining.select_failure_review_cases import select


def make_fixture(root: Path) -> tuple[Path, Path]:
    analysis = root / "analysis"
    analysis.mkdir()
    consensus = [
        {"dataset": "set", "scene": "hard", "metric": "geometry", "consensus_failure_rank": 1, "cross_model_disagreement_rank": 2},
        {"dataset": "set", "scene": "middle", "metric": "geometry", "consensus_failure_rank": 2, "cross_model_disagreement_rank": 1},
        {"dataset": "set", "scene": "easy", "metric": "geometry", "consensus_failure_rank": 3, "cross_model_disagreement_rank": 3},
    ]
    profiles = [
        {"model": "a", "dataset": "set", "scene": "hard", "rank_profile": "geometry_ranked_worse_than_pose", "absolute_pose_geometry_rank_gap": 0.8, "pose_geometry_disagreement_rank": 1},
        {"model": "a", "dataset": "set", "scene": "easy", "rank_profile": "pose_ranked_worse_than_geometry", "absolute_pose_geometry_rank_gap": 0.7, "pose_geometry_disagreement_rank": 2},
        {"model": "b", "dataset": "set", "scene": "middle", "rank_profile": "equal_failure_rank", "absolute_pose_geometry_rank_gap": 0.0, "pose_geometry_disagreement_rank": 3},
    ]
    consensus_path = analysis / "cross_model_consensus_failure_ranks.csv"
    profile_path = analysis / "pose_geometry_case_profiles.csv"
    write_csv(consensus_path, consensus)
    write_csv(profile_path, profiles)
    manifest = {
        "models": ["a", "b"],
        "outputs": {
            "cross_model_consensus_failure_ranks": {"sha256": sha256(consensus_path)},
            "pose_geometry_case_profiles": {"sha256": sha256(profile_path)},
        },
    }
    manifest_path = analysis / "manifest.json"
    manifest_path.write_text(json.dumps(manifest) + "\n")
    audit_path = root / "audit.json"
    audit_path.write_text(json.dumps({"passed": True, "analysis_manifest_sha256": sha256(manifest_path)}) + "\n")
    return analysis, audit_path


def test_selection_uses_extrema_and_keeps_controls(tmp_path: Path) -> None:
    analysis, audit = make_fixture(tmp_path)
    rows, provenance = select(analysis, audit)
    reasons = {reason for row in rows for reason in row["selection_reasons"].split(";")}
    assert "consensus_hardest" in reasons
    assert "consensus_easiest_control" in reasons
    assert "cross_model_most_disputed" in reasons
    assert "largest_gap:geometry_ranked_worse_than_pose" in reasons
    assert "largest_gap:pose_ranked_worse_than_geometry" in reasons
    assert provenance["selection_threshold"] is None
    assert provenance["top_k"] is None


def test_selection_rejects_audit_for_different_manifest(tmp_path: Path) -> None:
    analysis, audit = make_fixture(tmp_path)
    audit.write_text(json.dumps({"passed": True, "analysis_manifest_sha256": "0" * 64}) + "\n")
    with pytest.raises(ValueError, match="does not cover"):
        select(analysis, audit)
