import unittest
from pathlib import Path

import numpy as np

from scripts.failure_mining.compare_case_camera_errors import Case, compare_cases


def make_case(model, pairs):
    return Case(
        directory=Path(f"/{model}"),
        model=model,
        dataset="bench",
        scene="scene",
        pairs=pairs,
        camera_pairs_sha256=f"hash-{model}",
    )


class CompareCaseCameraErrorsTest(unittest.TestCase):
    def test_identical_and_reversed_pair_rankings(self):
        alpha = make_case(
            "alpha",
            {
                (0, 1): {"rotation_degrees": 1.0, "translation_degrees": 3.0},
                (0, 2): {"rotation_degrees": 2.0, "translation_degrees": 2.0},
                (1, 2): {"rotation_degrees": 3.0, "translation_degrees": 1.0},
            },
        )
        beta = make_case(
            "beta",
            {
                (0, 1): {"rotation_degrees": 1.0, "translation_degrees": 1.0},
                (0, 2): {"rotation_degrees": 2.0, "translation_degrees": 2.0},
                (1, 2): {"rotation_degrees": 3.0, "translation_degrees": 3.0},
            },
        )
        rows = compare_cases([alpha, beta])
        by_metric = {row["metric"]: row for row in rows}
        self.assertAlmostEqual(by_metric["rotation_degrees"]["spearman_pair_error"], 1.0)
        self.assertAlmostEqual(by_metric["translation_degrees"]["spearman_pair_error"], -1.0)

    def test_mismatched_pair_population_is_rejected(self):
        alpha = make_case(
            "alpha", {(0, 1): {"rotation_degrees": 1.0, "translation_degrees": 1.0}}
        )
        beta = make_case(
            "beta", {(0, 2): {"rotation_degrees": 1.0, "translation_degrees": 1.0}}
        )
        with self.assertRaisesRegex(ValueError, "population differs"):
            compare_cases([alpha, beta])

    def test_nonfinite_values_are_excluded_pairwise(self):
        alpha = make_case(
            "alpha",
            {
                (0, 1): {"rotation_degrees": 1.0, "translation_degrees": 1.0},
                (0, 2): {"rotation_degrees": np.nan, "translation_degrees": 2.0},
                (1, 2): {"rotation_degrees": 3.0, "translation_degrees": 3.0},
            },
        )
        beta = make_case(
            "beta",
            {
                (0, 1): {"rotation_degrees": 2.0, "translation_degrees": 1.0},
                (0, 2): {"rotation_degrees": 4.0, "translation_degrees": 2.0},
                (1, 2): {"rotation_degrees": 6.0, "translation_degrees": 3.0},
            },
        )
        rows = compare_cases([alpha, beta])
        by_metric = {row["metric"]: row for row in rows}
        self.assertEqual(by_metric["rotation_degrees"]["common_finite_pairs"], 2)
        self.assertEqual(by_metric["translation_degrees"]["common_finite_pairs"], 3)


if __name__ == "__main__":
    unittest.main()
