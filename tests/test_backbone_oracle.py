import unittest

from scripts.failure_mining.analyze_backbone_oracle import analyze


def ranking(model, scene, value, direction="higher_is_better"):
    return {
        "model": model,
        "dataset": "bench",
        "scene": scene,
        "metric": "score",
        "direction": direction,
        "value": str(value),
    }


class BackboneOracleTest(unittest.TestCase):
    def test_higher_is_better_headroom(self):
        rows = [
            ranking("alpha", "one", 0.9),
            ranking("beta", "one", 0.8),
            ranking("alpha", "two", 0.1),
            ranking("beta", "two", 0.7),
        ]
        scene_rows, summaries = analyze(rows)
        self.assertEqual(summaries[0]["best_fixed_model"], "beta")
        self.assertAlmostEqual(summaries[0]["best_fixed_mean"], 0.75)
        self.assertAlmostEqual(summaries[0]["oracle_mean"], 0.8)
        self.assertAlmostEqual(summaries[0]["mean_oracle_gain"], 0.05)
        by_scene = {row["scene"]: row for row in scene_rows}
        self.assertAlmostEqual(by_scene["one"]["oracle_gain"], 0.1)
        self.assertAlmostEqual(by_scene["two"]["oracle_gain"], 0.0)

    def test_lower_is_better_headroom(self):
        rows = [
            ranking("alpha", "one", 1.0, "lower_is_better"),
            ranking("beta", "one", 2.0, "lower_is_better"),
            ranking("alpha", "two", 5.0, "lower_is_better"),
            ranking("beta", "two", 2.0, "lower_is_better"),
        ]
        scene_rows, summaries = analyze(rows)
        self.assertEqual(summaries[0]["best_fixed_model"], "beta")
        self.assertAlmostEqual(summaries[0]["mean_oracle_gain"], 0.5)
        self.assertEqual(sum(row["oracle_gain"] > 0 for row in scene_rows), 1)

    def test_failure_direction_alias_is_supported(self):
        rows = [
            ranking("alpha", "one", 0.9, "lower_is_worse"),
            ranking("beta", "one", 0.8, "lower_is_worse"),
            ranking("alpha", "two", 0.1, "lower_is_worse"),
            ranking("beta", "two", 0.7, "lower_is_worse"),
        ]
        _scene_rows, summaries = analyze(rows)
        self.assertEqual(summaries[0]["best_fixed_model"], "beta")

    def test_model_population_mismatch_is_rejected(self):
        rows = [
            ranking("alpha", "one", 0.9),
            ranking("beta", "one", 0.8),
            ranking("alpha", "two", 0.7),
        ]
        with self.assertRaisesRegex(ValueError, "population differs"):
            analyze(rows)


if __name__ == "__main__":
    unittest.main()
