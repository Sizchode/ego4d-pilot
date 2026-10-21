import unittest

from scripts.failure_mining.compare_image_evidence import parse_failure_rankings
from scripts.failure_mining.compare_state_evidence import build_correlations


class StateEvidenceCompareTest(unittest.TestCase):
    def setUp(self):
        self.scenes = [
            {"dataset": "bench", "scene": scene, "state.confidence": str(value)}
            for scene, value in zip(("a", "b", "c"), (1, 2, 3), strict=True)
        ]
        ranking_rows = [
            {
                "dataset": "bench",
                "scene": scene,
                "metric": "pose.auc30",
                "failure_rank": str(rank),
                "population": "3",
                "direction": "lower_is_worse",
            }
            for scene, rank in zip(("a", "b", "c"), (1, 2, 3), strict=True)
        ]
        self.rankings = parse_failure_rankings(
            ranking_rows, {("bench", scene) for scene in ("a", "b", "c")}
        )

    def test_monotonic_state_field_has_unit_rank_correlation(self):
        rows = build_correlations(self.scenes, self.rankings, ["state.confidence"])
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(
            rows[0]["spearman_evidence_value_vs_failure_rank"], 1.0
        )

    def test_unknown_state_field_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "absent"):
            build_correlations(self.scenes, self.rankings, ["state.unknown"])

    def test_duplicate_state_field_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unique"):
            build_correlations(
                self.scenes,
                self.rankings,
                ["state.confidence", "state.confidence"],
            )


if __name__ == "__main__":
    unittest.main()
