from __future__ import annotations

import unittest
from pathlib import Path

from temporal_transfer.protocol import load_protocol
from temporal_transfer.providers.qwen3 import representation_names

ROOT = Path(__file__).resolve().parents[1]


class QwenRepresentationTests(unittest.TestCase):
    def test_all_protocol_extraction_points_have_stable_names(self) -> None:
        config = load_protocol(ROOT / "configs/protocols/pilot.yaml").model("qwen3")
        self.assertEqual(
            representation_names(config),
            [
                "vision_block_8",
                "vision_block_16",
                "vision_block_24",
                "vision_block_26",
                "visual_merger",
                "post_deepstack_language_layer_0",
                "post_deepstack_language_layer_1",
                "post_deepstack_language_layer_2",
                "final_language_visual_tokens",
            ],
        )


if __name__ == "__main__":
    unittest.main()
