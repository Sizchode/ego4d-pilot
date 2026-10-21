import unittest

from scripts.failure_mining.compare_baselines import AnalysisInput
from scripts.failure_mining.compare_visibility import (
    average_ranks,
    pairwise_agreement,
    ranked_rows,
    validate_population,
)


class CompareVisibilityTest(unittest.TestCase):
    def test_average_ranks_preserve_ties(self):
        self.assertEqual(average_ranks([3.0, 1.0, 1.0]), [3.0, 1.5, 1.5])

    def test_population_order_mismatch_is_rejected(self):
        analyses = [AnalysisInput("a", None), AnalysisInput("b", None)]
        rows = {
            "a": [{"dataset": "d", "scene": "s1"}, {"dataset": "d", "scene": "s2"}],
            "b": [{"dataset": "d", "scene": "s2"}, {"dataset": "d", "scene": "s1"}],
        }
        with self.assertRaisesRegex(ValueError, "population or order differs"):
            validate_population(analyses, rows)

    def test_pairwise_rank_agreement_uses_within_dataset_order(self):
        analyses = [AnalysisInput("a", None), AnalysisInput("b", None)]
        rows = {
            "a": [
                {"dataset": "d", "scene": "s1", "visibility.x": "1"},
                {"dataset": "d", "scene": "s2", "visibility.x": "2"},
                {"dataset": "d", "scene": "s3", "visibility.x": "3"},
            ],
            "b": [
                {"dataset": "d", "scene": "s1", "visibility.x": "3"},
                {"dataset": "d", "scene": "s2", "visibility.x": "2"},
                {"dataset": "d", "scene": "s3", "visibility.x": "1"},
            ],
        }
        ranks = ranked_rows(analyses, rows, ["visibility.x"])
        agreement = pairwise_agreement(analyses, ranks)
        self.assertEqual(len(agreement), 1)
        self.assertAlmostEqual(agreement[0]["spearman_rank"], -1.0)


if __name__ == "__main__":
    unittest.main()
