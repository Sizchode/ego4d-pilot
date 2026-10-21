import unittest

from scripts.failure_mining.compare_evidence_to_oracle import build_correlations


class EvidenceOracleCompareTest(unittest.TestCase):
    def setUp(self):
        self.evidence = [
            {"dataset": "bench", "scene": scene, "state.score": str(value)}
            for scene, value in zip(("a", "b", "c"), (1, 2, 3), strict=True)
        ]
        self.oracle = [
            {
                "dataset": "bench",
                "scene": scene,
                "metric": "pose.auc30",
                "direction": "higher_is_better",
                "oracle_gain": str(value),
            }
            for scene, value in zip(("a", "b", "c"), (0.0, 0.5, 1.0), strict=True)
        ]

    def test_monotonic_evidence_has_unit_rank_correlation(self):
        rows = build_correlations(self.evidence, self.oracle, ["state.score"])
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(
            rows[0]["spearman_evidence_value_vs_oracle_gain"], 1.0
        )

    def test_missing_oracle_scene_is_rejected(self):
        truncated = self.evidence[:-1]
        with self.assertRaisesRegex(ValueError, "no evidence"):
            build_correlations(truncated, self.oracle, ["state.score"])

    def test_negative_oracle_gain_is_rejected(self):
        self.oracle[0]["oracle_gain"] = "-0.1"
        with self.assertRaisesRegex(ValueError, "Invalid oracle gain"):
            build_correlations(self.evidence, self.oracle, ["state.score"])


if __name__ == "__main__":
    unittest.main()
