import unittest
from pathlib import Path

from scripts.failure_mining.compare_baselines import (
    AnalysisInput,
    build_pairwise_agreement,
    build_scene_metric_spread,
    parse_rankings,
    validate_scene_population,
)


class CompareBaselinesTest(unittest.TestCase):
    def setUp(self):
        self.analyses = [
            AnalysisInput("alpha", Path("/alpha")),
            AnalysisInput("beta", Path("/beta")),
        ]
        self.scenes = {
            model: [
                {"dataset": "bench", "scene": "one", "state": "completed", "run": "/one"},
                {"dataset": "bench", "scene": "two", "state": "completed", "run": "/two"},
                {"dataset": "bench", "scene": "three", "state": "completed", "run": "/three"},
            ]
            for model in ("alpha", "beta")
        }

    @staticmethod
    def ranking(scene, rank, value, direction="lower_is_worse"):
        return {
            "dataset": "bench",
            "scene": scene,
            "metric": "metric.pose.auc30",
            "direction": direction,
            "failure_rank": str(rank),
            "population": "3",
            "value": str(value),
            "run": f"/{scene}",
        }

    def test_identical_population_and_rank_agreement(self):
        population = validate_scene_population(self.analyses, self.scenes)
        self.assertEqual(population, [("bench", "one"), ("bench", "two"), ("bench", "three")])
        alpha = parse_rankings(
            self.analyses[0],
            [self.ranking("one", 1, 0.1), self.ranking("two", 2, 0.2), self.ranking("three", 3, 0.3)],
            set(population),
        )
        beta = parse_rankings(
            self.analyses[1],
            [self.ranking("one", 3, 0.3), self.ranking("two", 2, 0.2), self.ranking("three", 1, 0.1)],
            set(population),
        )
        rows = build_pairwise_agreement(self.analyses, {"alpha": alpha, "beta": beta})
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0]["spearman_failure_rank"], -1.0)

        rank_rows = [*alpha.values(), *beta.values()]
        spread = build_scene_metric_spread(rank_rows)
        first = next(row for row in spread if row["scene"] == "one")
        self.assertEqual(first["models_with_metric"], 2)
        self.assertAlmostEqual(first["worstness_range"], 1.0)

    def test_population_order_mismatch_is_rejected(self):
        swapped = {key: list(value) for key, value in self.scenes.items()}
        swapped["beta"] = [swapped["beta"][1], swapped["beta"][0], swapped["beta"][2]]
        with self.assertRaisesRegex(ValueError, "population or order differs"):
            validate_scene_population(self.analyses, swapped)

    def test_incomplete_rank_permutation_is_rejected(self):
        population = set(validate_scene_population(self.analyses, self.scenes))
        rows = [self.ranking("one", 1, 0.1), self.ranking("two", 1, 0.2), self.ranking("three", 3, 0.3)]
        with self.assertRaisesRegex(ValueError, "Rankings are incomplete"):
            parse_rankings(self.analyses[0], rows, population)

    def test_cross_model_direction_mismatch_is_rejected(self):
        population = set(validate_scene_population(self.analyses, self.scenes))
        alpha_rows = [self.ranking("one", 1, 0.1), self.ranking("two", 2, 0.2), self.ranking("three", 3, 0.3)]
        beta_rows = [
            self.ranking("one", 1, 0.1, "higher_is_worse"),
            self.ranking("two", 2, 0.2, "higher_is_worse"),
            self.ranking("three", 3, 0.3, "higher_is_worse"),
        ]
        rankings = {
            "alpha": parse_rankings(self.analyses[0], alpha_rows, population),
            "beta": parse_rankings(self.analyses[1], beta_rows, population),
        }
        with self.assertRaisesRegex(ValueError, "direction differs"):
            build_pairwise_agreement(self.analyses, rankings)


if __name__ == "__main__":
    unittest.main()
