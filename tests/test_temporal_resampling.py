import json
import tempfile
import unittest
from pathlib import Path

from src.temporal_resampling.manifest import build_nextqa, build_ssv2, read_jsonl, write_jsonl
from src.temporal_resampling.protocol import ProtocolError, cross_sampling_result, load_canonical_schedules, load_config
from src.temporal_resampling.probes import train_cross_sampling_probe
from src.temporal_resampling.sampling import build_schedule, map_to_actual_frames


class TemporalResamplingTests(unittest.TestCase):
    def test_config_has_only_three_families_and_two_pilot_datasets(self):
        _, config = load_config(Path("configs/temporal_resampling.yaml"))
        self.assertEqual(set(config["models"]), {"dinov3", "vjepa2", "qwen3_vl"})
        self.assertEqual(set(config["datasets"]), {"ssv2", "nextqa"})
        self.assertEqual(config["sampling"]["conditions"], ["C0", "C1", "C2", "C3"])

    def test_schedules_change_time_not_event_label(self):
        canonical = [0.0, 0.25, 0.5, 0.75]
        c0 = build_schedule("C0", canonical, 1.0)
        c2 = build_schedule("C2", canonical, 1.0, offset_seconds=0.1)
        c3 = build_schedule("C3", canonical, 1.0)
        self.assertEqual(c0.timestamps, tuple(canonical))
        self.assertNotEqual(c0.timestamps, c2.timestamps)
        self.assertEqual(c3.timestamps[0], c0.timestamps[0])
        self.assertEqual(c3.timestamps[-1], c0.timestamps[-1])

    def test_c2_boundary_is_rejected_instead_of_clipped(self):
        with self.assertRaises(ProtocolError):
            build_schedule("C2", [0.0, 0.5], 1.0, offset_seconds=0.6)

    def test_c1_requires_an_explicit_rate_ratio(self):
        with self.assertRaises(ProtocolError):
            build_schedule("C1", [0.0, 0.5, 1.0], 1.5)

    def test_c3_weights_each_canonical_interval_before_normalizing_coverage(self):
        schedule = build_schedule("C3", [0.0, 1.0, 3.0], 4.0)
        self.assertAlmostEqual(schedule.timestamps[1] - schedule.timestamps[0], 3 / 7)
        self.assertAlmostEqual(schedule.timestamps[2] - schedule.timestamps[1], 18 / 7)

    def test_canonical_schedule_file_requires_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "canonical.json"
            path.write_text(json.dumps({"schedules": {"sample": [0.0, 0.5]}}))
            with self.assertRaises(ProtocolError):
                load_canonical_schedules(path)

    def test_mapping_retains_requested_timestamps_and_actual_indices(self):
        schedule = build_schedule("C0", [0.0, 0.25, 0.5], 1.0)
        mapped = map_to_actual_frames(schedule, [0.0, 0.2, 0.55])
        self.assertEqual(mapped.timestamps, schedule.timestamps)
        self.assertEqual(mapped.frame_indices, (0, 1, 2))
        self.assertEqual(mapped.actual_timestamps, (0.0, 0.2, 0.55))

    def test_mapping_rejects_out_of_range_requests_instead_of_clipping(self):
        schedule = build_schedule("C0", [0.0, 0.5], 1.0)
        with self.assertRaises(ProtocolError):
            map_to_actual_frames(schedule, [0.1, 0.5, 0.9])

    def test_cross_sampling_reports_only_top1_and_deltas(self):
        result = cross_sampling_result(
            {condition: [0, 1] for condition in ("C0", "C1", "C2", "C3")},
            {"C0": [0, 1], "C1": [0, 0], "C2": [1, 1], "C3": [0, 1]},
        )
        self.assertEqual(result["top1_C0"], 1.0)
        self.assertEqual(result["delta_C1"], -0.5)
        self.assertEqual(set(result), {"top1_C0", "top1_C1", "top1_C2", "top1_C3", "delta_C1", "delta_C2", "delta_C3"})

    def test_attentive_probe_buckets_variable_token_counts_without_padding(self):
        import torch

        labels = torch.tensor([0, 1, 0, 1])
        train = [torch.randn(2, 4), torch.randn(3, 4), torch.randn(2, 4), torch.randn(3, 4)]
        validation = {condition: list(train) for condition in ("C0", "C1", "C2", "C3")}
        metrics = train_cross_sampling_probe(
            train, labels, validation, {condition: labels for condition in validation},
            classes=2, readout="attentive", heads=2, epochs=1, batch_size=2,
            learning_rate=1e-3, weight_decay=0.0,
        )
        self.assertEqual(set(metrics), {"top1_C0", "top1_C1", "top1_C2", "top1_C3", "delta_C1", "delta_C2", "delta_C3"})

    def test_ssv2_and_nextqa_builders_preserve_official_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "1.mp4").touch()
            (root / "clip.mp4").touch()
            label_map = {f"label-{index}": index for index in range(174) if index != 4}
            label_map["Put [something] on [something]"] = 4
            (root / "labels.json").write_text(json.dumps(label_map))
            (root / "ssv2.json").write_text(json.dumps([{"id": "1", "template": "Put [something] on [something]"}]))
            records = build_ssv2(root / "ssv2.json", root, "train", root / "labels.json", "ssv2-revision")
            self.assertEqual(records[0].label_id, 4)
            (root / "nextqa.csv").write_text(
                "video,qid,question,answer,type,a0,a1,a2,a3,a4\n"
                "clip.mp4,q1,What happens?,2,T,no,yes,first,last,none\n"
            )
            nextqa = build_nextqa(root / "nextqa.csv", root, "val", "nextqa-revision")
            self.assertEqual(nextqa[0].category, "temporal")
            output = root / "manifest.jsonl"
            write_jsonl(nextqa, output)
            self.assertEqual(read_jsonl(output)[0].answer_index, 2)


if __name__ == "__main__":
    unittest.main()
