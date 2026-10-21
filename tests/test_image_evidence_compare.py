import unittest

from scripts.failure_mining.compare_image_evidence import (
    average_ranks,
    build_correlations,
    parse_failure_rankings,
    validate_population,
)


class TestImageEvidenceCompare(unittest.TestCase):
    def test_average_ranks_preserves_ties(self):
        self.assertEqual(average_ranks([4.0, 1.0, 1.0, 3.0]), [4.0, 1.5, 1.5, 3.0])

    def test_population_requires_same_order(self):
        scenes = [
            {"dataset": "d", "scene": "a"},
            {"dataset": "d", "scene": "b"},
        ]
        with self.assertRaises(ValueError):
            validate_population(scenes, list(reversed(scenes)))

    def test_monotonic_image_metric_has_unit_rank_correlation(self):
        image_rows = [
            {"dataset": "d", "scene": scene, "image.gradient.p90.mean": str(value)}
            for scene, value in zip(("a", "b", "c", "d"), (1, 2, 3, 4), strict=True)
        ]
        ranking_rows = [
            {
                "dataset": "d",
                "scene": scene,
                "metric": "metric.pose.auc30",
                "failure_rank": str(rank),
                "population": "4",
                "direction": "lower_is_worse",
            }
            for scene, rank in zip(("a", "b", "c", "d"), (1, 2, 3, 4), strict=True)
        ]
        rankings = parse_failure_rankings(ranking_rows, {("d", x) for x in "abcd"})
        result = build_correlations(image_rows, rankings)
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(
            result[0]["spearman_image_value_vs_failure_rank"], 1.0
        )


if __name__ == "__main__":
    unittest.main()
