import unittest

from scripts.failure_mining.compare_metric_rankings import compare


def ranking(metric, scene, rank):
    return {
        "model": "model",
        "dataset": "bench",
        "metric": metric,
        "scene": scene,
        "failure_rank": str(rank),
        "direction": "higher_is_better",
    }


class MetricRankCompareTest(unittest.TestCase):
    def test_inverse_rankings(self):
        rows = [
            ranking("pose", "a", 1),
            ranking("pose", "b", 2),
            ranking("pose", "c", 3),
            ranking("geometry", "a", 3),
            ranking("geometry", "b", 2),
            ranking("geometry", "c", 1),
        ]
        result = compare(rows, "pose", "geometry")
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0]["spearman_failure_rank"], -1.0)

    def test_population_mismatch_is_rejected(self):
        rows = [
            ranking("pose", "a", 1),
            ranking("pose", "b", 2),
            ranking("geometry", "a", 1),
        ]
        with self.assertRaisesRegex(ValueError, "population differs"):
            compare(rows, "pose", "geometry")

    def test_group_missing_one_metric_is_skipped(self):
        result = compare([ranking("pose", "a", 1)], "pose", "geometry")
        self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()
