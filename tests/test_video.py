from __future__ import annotations

import unittest

from temporal_transfer.video import virtual_reversed_indices


class VideoProtocolTests(unittest.TestCase):
    def test_forward_indices_are_unchanged(self) -> None:
        self.assertEqual(virtual_reversed_indices([0, 3, 9], 10, backward=False), [0, 3, 9])

    def test_reverse_is_exact_virtual_video_mapping(self) -> None:
        self.assertEqual(virtual_reversed_indices([0, 3, 9], 10, backward=True), [9, 6, 0])

    def test_invalid_index_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            virtual_reversed_indices([10], 10, backward=True)


if __name__ == "__main__":
    unittest.main()
