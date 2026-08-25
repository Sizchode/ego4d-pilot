from __future__ import annotations

import copy
import unittest
from pathlib import Path

from temporal_transfer.protocol import ProtocolError, load_protocol, validate_protocol_data

ROOT = Path(__file__).resolve().parents[1]


class ProtocolTests(unittest.TestCase):
    def test_pilot_protocol_is_valid(self) -> None:
        protocol = load_protocol(ROOT / "configs/protocols/pilot.yaml")
        self.assertEqual(protocol.seed, 42)

    def test_unpinned_revision_is_rejected(self) -> None:
        protocol = load_protocol(ROOT / "configs/protocols/pilot.yaml")
        altered = copy.deepcopy(protocol.data)
        altered["models"]["qwen3"]["revision"] = "main"
        with self.assertRaises(ProtocolError):
            validate_protocol_data(altered)

    def test_changed_split_is_rejected(self) -> None:
        protocol = load_protocol(ROOT / "configs/protocols/pilot.yaml")
        altered = copy.deepcopy(protocol.data)
        altered["datasets"]["ssv2"]["split"] = "validation"
        with self.assertRaises(ProtocolError):
            validate_protocol_data(altered)


if __name__ == "__main__":
    unittest.main()
