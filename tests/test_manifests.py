from __future__ import annotations

import csv
import importlib.util
import json
import random
import tempfile
import unittest
from pathlib import Path

from temporal_transfer.manifests import (
    assign_balanced_directions,
    build_aot_examples,
    build_ssv2_examples,
    read_jsonl,
    write_jsonl,
)

ROOT = Path(__file__).resolve().parents[1]


class ManifestTests(unittest.TestCase):
    def test_direction_assignment_is_seeded_balanced_and_order_preserving(self) -> None:
        ids = [str(index) for index in range(9)]
        first = assign_balanced_directions(ids, 42)
        second = assign_balanced_directions(ids, 42)
        self.assertEqual(first, second)
        self.assertEqual(list(first), ids)
        self.assertLessEqual(abs(sum(first.values()) - (len(ids) - sum(first.values()))), 1)

    def test_direction_assignment_matches_published_tat_function(self) -> None:
        source = ROOT / "third_party/TAT-VQA/aot.py"
        spec = importlib.util.spec_from_file_location("tat_aot", source)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        ids = ["10", "2", "1", "20", "3"]
        state = random.getstate()
        try:
            random.seed(42)
            published = module.assign_balanced_aot([Path(item) for item in sorted(ids)])
        finally:
            random.setstate(state)
        encoded = [0 if label == "forward" else 1 for label in published]
        expected = dict(zip(sorted(ids), encoded))
        self.assertEqual(assign_balanced_directions(ids, 42), {item: expected[item] for item in ids})

    def test_ssv2_uses_every_official_annotation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            annotations = root / "train.json"
            annotations.write_text(json.dumps([{"id": "1"}, {"id": "2"}]), encoding="utf-8")
            (root / "1.webm").touch()
            (root / "2.webm").touch()
            examples = build_ssv2_examples(annotations, root, seed=42, expected_examples=2)
            self.assertEqual({example.source_id for example in examples}, {"1", "2"})
            self.assertEqual({example.split for example in examples}, {"train"})

    def test_aot_pairs_each_released_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metadata = root / "metadata.csv"
            with metadata.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["id", "motion_category"])
                writer.writeheader()
                writer.writerow({"id": "00001.mp4", "motion_category": "['Fall', 'Explode']"})
            (root / "00001.mp4").touch()
            examples = build_aot_examples(metadata, root, expected_source_clips=1)
            self.assertEqual([example.label for example in examples], [0, 1])
            self.assertEqual(examples[0].video_path, examples[1].video_path)
            output = root / "manifest.jsonl"
            write_jsonl(examples, output)
            self.assertEqual(read_jsonl(output), examples)


if __name__ == "__main__":
    unittest.main()
