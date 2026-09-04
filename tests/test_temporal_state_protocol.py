import json
import tempfile
import unittest
from pathlib import Path

from experiments.temporal_state.manifest import build_nextqa_manifest, build_ssv2_manifest, read_jsonl, write_jsonl
from experiments.temporal_state.protocol import ProtocolError, load_protocol_config, nextqa_metrics, normalized_depth


class TemporalStateProtocolTests(unittest.TestCase):
    def test_final_config_has_exact_model_dataset_and_metric_contract(self):
        path = Path(__file__).parents[1] / "experiments" / "temporal_state" / "config.yaml"
        _, config = load_protocol_config(path)
        self.assertEqual(set(config["models"]), {"qwen3_vl", "internvl3_5", "minicpm_v45", "vjepa2_vitl"})
        self.assertEqual(set(config["datasets"]), {"ssv2", "nextqa", "egotempo", "vstar"})
        self.assertEqual(config["evaluations"]["nextqa_metrics"], [
            "overall_accuracy", "temporal_accuracy", "causal_accuracy", "descriptive_accuracy"
        ])

    def test_normalized_depth_is_plotting_only_and_bounds_checked(self):
        self.assertEqual(normalized_depth(8, 36), 8 / 36)
        with self.assertRaises(ProtocolError):
            normalized_depth(36, 36)

    def test_nextqa_reports_the_four_official_accuracy_views(self):
        metrics = nextqa_metrics(
            [0, 1, 0, 1, 0, 1], [0, 0, 0, 1, 1, 1],
            ["temporal", "temporal", "causal", "causal", "descriptive", "descriptive"],
        )
        self.assertEqual(metrics, {
            "overall_accuracy": 4 / 6,
            "temporal_accuracy": 1 / 2,
            "causal_accuracy": 1.0,
            "descriptive_accuracy": 1 / 2,
        })

    def test_ssv2_manifest_preserves_official_action_label(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "1.mp4"
            video.touch()
            annotations = root / "train.json"
            annotations.write_text(json.dumps([{"id": "1", "template": "Putting [something] on [something]"}]))
            label_map = root / "labels.json"
            label_map.write_text(json.dumps({"Putting [something] on [something]": 7}))
            records = build_ssv2_manifest(annotations, root, "train", label_map)
            self.assertEqual(records[0].label, "Putting [something] on [something]")
            self.assertEqual(records[0].label_id, 7)

    def test_nextqa_manifest_requires_five_choices_and_preserves_category(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "video.mp4").touch()
            annotation = root / "val.csv"
            annotation.write_text(
                "video,qid,question,answer,type,a0,a1,a2,a3,a4\n"
                "video.mp4,q1,What happens?,2,T,no,yes,first,last,none\n"
            )
            records = build_nextqa_manifest(annotation, root, "val")
            self.assertEqual(records[0].category, "temporal")
            self.assertEqual(records[0].answer_index, 2)
            manifest = root / "manifest.jsonl"
            write_jsonl(records, manifest)
            self.assertEqual(read_jsonl(manifest)[0].example_id, "q1")

    def test_protocol_rejects_a_new_composite_metric(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.yaml"
            path.write_text(
                "protocol_version: x\nmodels: {}\ndatasets: {}\nprobes: {}\nevaluations: {state_sufficiency_score: 1}\n"
            )
            with self.assertRaises(ProtocolError):
                load_protocol_config(path)


if __name__ == "__main__":
    unittest.main()
