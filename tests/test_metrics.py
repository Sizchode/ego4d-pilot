from __future__ import annotations

import unittest

from temporal_transfer.metrics import arrow_of_time_metrics


class MetricTests(unittest.TestCase):
    def test_accuracy_f1_and_confusion(self) -> None:
        metrics = arrow_of_time_metrics([0, 0, 1, 1], [0, 1, 1, 1])
        self.assertEqual(metrics["accuracy"], 0.75)
        self.assertAlmostEqual(metrics["forward_f1"], 2 / 3)
        self.assertAlmostEqual(metrics["backward_f1"], 0.8)
        self.assertEqual(metrics["confusion_matrix"], [[1, 1], [0, 2]])


if __name__ == "__main__":
    unittest.main()
