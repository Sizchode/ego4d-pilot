"""Train a C0 probe once and evaluate C0/C1/C2/C3 validation shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.temporal_resampling.features import load_shard
from src.temporal_resampling.probes import train_cross_sampling_probe
from src.temporal_resampling.protocol import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/temporal_resampling.yaml")
    parser.add_argument("--model", choices=("dinov3", "vjepa2", "qwen3_vl"), required=True)
    parser.add_argument("--readout", choices=("mean", "attentive"), required=True)
    parser.add_argument("--train-c0", required=True)
    parser.add_argument("--val-c0", required=True)
    parser.add_argument("--val-c1", required=True)
    parser.add_argument("--val-c2", required=True)
    parser.add_argument("--val-c3", required=True)
    parser.add_argument("--representation", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    _, config = load_config(args.config)
    train = load_shard(args.train_c0)
    validation = {condition: load_shard(path) for condition, path in {"C0": args.val_c0, "C1": args.val_c1, "C2": args.val_c2, "C3": args.val_c3}.items()}
    if any(shard["condition"] != condition for condition, shard in validation.items()) or train["condition"] != "C0":
        raise ValueError("cross-sampling probe requires shards with their declared C0/C1/C2/C3 conditions")
    if train["split"] != "train":
        raise ValueError("cross-sampling training shard must use the train split")
    if any(shard["dataset"] != train["dataset"] or shard["model"] != train["model"] or shard["split"] != "validation" for shard in validation.values()):
        raise ValueError("cross-sampling shards must use one model/dataset with train and validation splits")
    sample_ids = next(iter(validation.values()))["sample_ids"]
    if any(shard["sample_ids"] != sample_ids for shard in validation.values()):
        raise ValueError("validation cross-sampling shards must preserve the exact sample order")
    if args.model != train["model"]:
        raise ValueError(f"requested model {args.model!r} does not match the training shard {train['model']!r}")
    if any(shard["canonical_provenance"] != train["canonical_provenance"] for shard in validation.values()):
        raise ValueError("cross-sampling shards must use the same canonical schedule provenance")
    device = torch.device(args.device)
    train_features = train["representations"][args.representation]
    validation_features = {condition: shard["representations"][args.representation] for condition, shard in validation.items()}
    train_features = [feature.to(device) for feature in train_features] if isinstance(train_features, list) else train_features.to(device)
    validation_features = {
        condition: [feature.to(device) for feature in features] if isinstance(features, list) else features.to(device)
        for condition, features in validation_features.items()
    }
    labels = train["labels"].to(device)
    validation_labels = {condition: shard["labels"].to(device) for condition, shard in validation.items()}
    optimization = config["probes"]["optimization"]
    metrics = train_cross_sampling_probe(train_features, labels, validation_features, validation_labels, classes=int(config["datasets"]["ssv2"]["classes"]), readout=args.readout, heads=int(optimization["attentive_heads"]), epochs=int(optimization["epochs"]), batch_size=int(optimization["batch_size"]), learning_rate=float(optimization["learning_rate"]), weight_decay=float(optimization["weight_decay"]))
    result = {"model": args.model, "representation": args.representation, "readout": args.readout, "train_condition": "C0", "metrics": metrics}
    destination = Path(args.output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
